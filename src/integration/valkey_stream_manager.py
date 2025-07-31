# -*- coding: utf-8 -*-
"""
Valkey Stream Manager - Gerencia streams para armazenamento e time travel
"""

import valkey
import orjson
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

class ValkeyStreamManager:
    """
    Gerencia streams Valkey para armazenamento e time travel
    """
    
    def __init__(self):
        from src.config.zmq_valkey_config import ZMQValkeyConfig
        
        self.config = ZMQValkeyConfig
        self.client = None
        self.logger = logging.getLogger('ValkeyManager')
        self.active_streams = set()
        self._connect()
        
    def _connect(self):
        """Conecta ao Valkey"""
        try:
            self.client = valkey.Valkey(
                host=self.config.VALKEY_HOST,
                port=self.config.VALKEY_PORT,
                password=self.config.VALKEY_PASSWORD,
                db=self.config.VALKEY_DB,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_keepalive=True,
                socket_keepalive_options={1: 1, 3: 3}
            )
            
            # Testar conexão
            self.client.ping()
            self.logger.info(f"Conectado ao Valkey em {self.config.VALKEY_HOST}:{self.config.VALKEY_PORT}")
            
        except Exception as e:
            self.logger.error(f"Erro ao conectar Valkey: {e}")
            raise
    
    def create_symbol_streams(self, symbol: str):
        """Cria streams necessários para um símbolo"""
        streams = {
            'ticks': f"stream:ticks:{symbol}",
            'candles_1m': f"stream:candles:1m:{symbol}",
            'candles_5m': f"stream:candles:5m:{symbol}",
            'features': f"stream:features:{symbol}",
            'signals': f"stream:signals:{symbol}",
            'predictions': f"stream:predictions:{symbol}",
            'trades': f"stream:trades:{symbol}"
        }
        
        for stream_type, stream_key in streams.items():
            try:
                # Verificar se stream existe
                exists = self.client.exists(stream_key)
                
                if not exists:
                    # Criar stream com entry inicial
                    self.client.xadd(
                        stream_key,
                        {b"init": b"true", b"timestamp": str(datetime.now()).encode()},
                        maxlen=self.config.STREAM_MAX_LEN,
                        approximate=True
                    )
                    self.logger.info(f"Stream {stream_key} criado")
                
                self.active_streams.add(stream_key)
                
            except Exception as e:
                self.logger.error(f"Erro ao criar stream {stream_key}: {e}")
        
        return streams
    
    def add_tick(self, symbol: str, tick_data: Dict):
        """Adiciona tick ao stream"""
        stream_key = f"stream:ticks:{symbol}"
        
        try:
            # Usar timestamp como ID para ordenação temporal
            timestamp_ms = tick_data.get('timestamp_ms', int(datetime.now().timestamp() * 1000))
            
            # Converter para bytes
            tick_bytes = {}
            for k, v in tick_data.items():
                if isinstance(v, (int, float)):
                    tick_bytes[k.encode()] = str(v).encode()
                elif isinstance(v, str):
                    tick_bytes[k.encode()] = v.encode()
                else:
                    tick_bytes[k.encode()] = orjson.dumps(v)
            
            # Adicionar ao stream
            entry_id = self.client.xadd(
                stream_key,
                tick_bytes,
                id=f"{timestamp_ms}-*",
                maxlen=self.config.STREAM_MAX_LEN,
                approximate=True
            )
            
            return entry_id
            
        except Exception as e:
            self.logger.error(f"Erro ao adicionar tick: {e}")
            return None
    
    def add_candle(self, symbol: str, timeframe: str, candle_data: Dict):
        """Adiciona candle ao stream"""
        stream_key = f"stream:candles:{timeframe}:{symbol}"
        
        try:
            timestamp_ms = candle_data.get('timestamp_ms', int(datetime.now().timestamp() * 1000))
            
            # Converter para bytes
            candle_bytes = {}
            for k, v in candle_data.items():
                if isinstance(v, (int, float)):
                    candle_bytes[k.encode()] = str(v).encode()
                else:
                    candle_bytes[k.encode()] = str(v).encode()
            
            entry_id = self.client.xadd(
                stream_key,
                candle_bytes,
                id=f"{timestamp_ms}-*",
                maxlen=self.config.STREAM_MAX_LEN,
                approximate=True
            )
            
            return entry_id
            
        except Exception as e:
            self.logger.error(f"Erro ao adicionar candle: {e}")
            return None
    
    def add_features(self, symbol: str, features: Dict):
        """Adiciona features calculadas ao stream"""
        stream_key = f"stream:features:{symbol}"
        
        try:
            timestamp_ms = int(datetime.now().timestamp() * 1000)
            
            # Serializar features como JSON
            features_json = orjson.dumps(features)
            
            feature_data = {
                b'timestamp_ms': str(timestamp_ms).encode(),
                b'features': features_json,
                b'feature_count': str(len(features)).encode()
            }
            
            entry_id = self.client.xadd(
                stream_key,
                feature_data,
                id=f"{timestamp_ms}-*",
                maxlen=10000  # Menos features que ticks
            )
            
            return entry_id
            
        except Exception as e:
            self.logger.error(f"Erro ao adicionar features: {e}")
            return None
    
    def time_travel_query(self, symbol: str, start_time: datetime, end_time: datetime, 
                         data_type: str = 'ticks', count: Optional[int] = None) -> List[Dict]:
        """
        Realiza time travel query entre timestamps
        """
        # Construir chave do stream
        if data_type == 'ticks':
            stream_key = f"stream:ticks:{symbol}"
        elif data_type.startswith('candles'):
            timeframe = data_type.split('_')[1] if '_' in data_type else '1m'
            stream_key = f"stream:candles:{timeframe}:{symbol}"
        else:
            stream_key = f"stream:{data_type}:{symbol}"
        
        # Converter para IDs de stream
        start_id = f"{int(start_time.timestamp() * 1000)}-0"
        end_id = f"{int(end_time.timestamp() * 1000)}-0"
        
        try:
            # Query com ou sem limite
            if count:
                entries = self.client.xrange(stream_key, start_id, end_id, count=count)
            else:
                entries = self.client.xrange(stream_key, start_id, end_id)
            
            # Converter de volta para dicts
            results = []
            for entry_id, fields in entries:
                data = {}
                
                # Decodificar campos
                for k, v in fields.items():
                    key = k.decode() if isinstance(k, bytes) else k
                    
                    # Tentar decodificar valor
                    if isinstance(v, bytes):
                        try:
                            # Tentar como JSON primeiro
                            if key == 'features':
                                data[key] = orjson.loads(v)
                            else:
                                # Decodificar como string
                                value_str = v.decode()
                                
                                # Tentar converter para número
                                if key in ['price', 'volume', 'high', 'low', 'open', 'close']:
                                    try:
                                        data[key] = float(value_str)
                                    except:
                                        data[key] = value_str
                                elif key in ['quantity', 'trade_id', 'timestamp_ms']:
                                    try:
                                        data[key] = int(value_str)
                                    except:
                                        data[key] = value_str
                                else:
                                    data[key] = value_str
                        except:
                            data[key] = v
                    else:
                        data[key] = v
                
                # Adicionar ID do stream
                data['stream_id'] = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                
                results.append(data)
            
            self.logger.debug(f"Time travel query {stream_key}: {len(results)} entries")
            return results
            
        except Exception as e:
            self.logger.error(f"Erro no time travel query: {e}")
            return []
    
    def get_latest_features(self, symbol: str, count: int = 1) -> Optional[Dict]:
        """Obtém últimas features calculadas"""
        stream_key = f"stream:features:{symbol}"
        
        try:
            entries = self.client.xrevrange(stream_key, count=count)
            
            if entries:
                _, fields = entries[0]
                
                features = {}
                for k, v in fields.items():
                    key = k.decode() if isinstance(k, bytes) else k
                    
                    if key == 'features' and isinstance(v, bytes):
                        features['features'] = orjson.loads(v)
                    elif isinstance(v, bytes):
                        features[key] = v.decode()
                    else:
                        features[key] = v
                
                return features
            
        except Exception as e:
            self.logger.error(f"Erro ao obter features: {e}")
        
        return None
    
    def get_stream_info(self, symbol: str, data_type: str = 'ticks') -> Dict:
        """Obtém informações sobre um stream"""
        if data_type == 'ticks':
            stream_key = f"stream:ticks:{symbol}"
        else:
            stream_key = f"stream:{data_type}:{symbol}"
        
        try:
            info = self.client.xinfo_stream(stream_key)
            
            # Converter bytes para strings
            info_dict = {}
            for k, v in info.items():
                key = k.decode() if isinstance(k, bytes) else k
                if isinstance(v, bytes):
                    info_dict[key] = v.decode()
                else:
                    info_dict[key] = v
            
            return info_dict
            
        except Exception as e:
            self.logger.error(f"Erro ao obter info do stream: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: Optional[int] = None):
        """Remove dados antigos dos streams"""
        if days_to_keep is None:
            days_to_keep = self.config.STREAM_RETENTION_DAYS
        
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        cutoff_id = f"{int(cutoff_time.timestamp() * 1000)}-0"
        
        cleaned = 0
        
        # Buscar todos os streams
        for stream_key in self.active_streams:
            try:
                # Remover entries antigas
                removed = self.client.xtrim(stream_key, minid=cutoff_id, approximate=True)
                if removed:
                    cleaned += removed
                    self.logger.info(f"Removidas {removed} entries antigas de {stream_key}")
                    
            except Exception as e:
                self.logger.error(f"Erro ao limpar {stream_key}: {e}")
        
        self.logger.info(f"Limpeza concluída: {cleaned} entries removidas")
        return cleaned
    
    def time_travel_to_dataframe(self, symbol: str, start_time: datetime, 
                                end_time: datetime, data_type: str = 'ticks') -> pd.DataFrame:
        """
        Converte resultado de time travel para DataFrame
        Útil para integração com sistema atual
        """
        data = self.time_travel_query(symbol, start_time, end_time, data_type)
        
        if not data:
            return pd.DataFrame()
        
        # Criar DataFrame
        df = pd.DataFrame(data)
        
        # Converter timestamp_ms para datetime index
        if 'timestamp_ms' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'].astype(float), unit='ms')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
        
        return df
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas dos streams"""
        stats = {
            'connected': self.client.ping() if self.client else False,
            'active_streams': len(self.active_streams),
            'streams': {}
        }
        
        for stream_key in self.active_streams:
            try:
                info = self.get_stream_info(stream_key.split(':')[-1], stream_key.split(':')[1])
                stats['streams'][stream_key] = {
                    'length': info.get('length', 0),
                    'first_entry': info.get('first-entry', ''),
                    'last_entry': info.get('last-entry', '')
                }
            except:
                pass
        
        return stats
    
    def close(self):
        """Fecha conexão com Valkey"""
        if self.client:
            self.client.close()
            self.logger.info("Conexão Valkey fechada")