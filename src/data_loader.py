import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging
import os

class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_candle = None
        self.last_candle_minute = None
        
        # Inicializar DataFrames e buffers para armazenar candles
        self.candles_df = pd.DataFrame()
        self.candles_buffer = []
        
    def create_or_update_candle(self, trade_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Cria ou atualiza candle baseado nos dados de trade
        Retorna candle completo quando o minuto muda
        """
        try:
            # Extrair timestamp e preço do trade
            if 'timestamp' in trade_data:
                timestamp = pd.to_datetime(trade_data['timestamp'])
            else:
                timestamp = pd.Timestamp.now()
                
            price = float(trade_data.get('price', 0))
            volume = int(trade_data.get('volume', 0))
            
            # Definir o minuto do candle (arredondar para baixo)
            candle_minute = timestamp.floor('1min')
            
            # Se é um novo candle
            if self.last_candle_minute is None or candle_minute > self.last_candle_minute:
                # Salvar candle anterior se existir
                completed_candle = None
                if self.current_candle is not None:
                    completed_candle = self.current_candle.copy()
                
                # Iniciar novo candle
                self.current_candle = pd.Series({
                    'timestamp': candle_minute,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume,
                    'trades': 1,
                    'buy_volume': volume if trade_data.get('side', 'buy') == 'buy' else 0,
                    'sell_volume': volume if trade_data.get('side', 'buy') == 'sell' else 0
                })
                
                self.last_candle_minute = candle_minute
                
                # Retornar e armazenar candle completo se existir
                if completed_candle is not None:
                    # Criar DataFrame com o candle completo
                    candle_df = pd.DataFrame([completed_candle])
                    candle_df.set_index('timestamp', inplace=True)
                    
                    # Adicionar ao DataFrame principal
                    if self.candles_df.empty:
                        self.candles_df = candle_df.copy()
                    else:
                        self.candles_df = pd.concat([self.candles_df, candle_df])
                    
                    # Adicionar ao buffer também
                    self.candles_buffer.append(completed_candle.to_dict())
                    
                    # Manter apenas os últimos 1000 candles para não consumir muita memória
                    if len(self.candles_df) > 1000:
                        self.candles_df = self.candles_df.tail(1000)
                    if len(self.candles_buffer) > 1000:
                        self.candles_buffer = self.candles_buffer[-1000:]
                    
                    self.logger.info(f"📊 Candle armazenado. Total de candles: {len(self.candles_df)}")
                    
                    return candle_df
                    
            else:
                # Atualizar candle atual
                if self.current_candle is not None:
                    self.current_candle['high'] = max(self.current_candle['high'], price)
                    self.current_candle['low'] = min(self.current_candle['low'], price)
                    self.current_candle['close'] = price
                    self.current_candle['volume'] += volume
                    self.current_candle['trades'] += 1
                    
                    if trade_data.get('side', 'buy') == 'buy':
                        self.current_candle['buy_volume'] += volume
                    else:
                        self.current_candle['sell_volume'] += volume
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro criando/atualizando candle: {e}")
            return None
    
    def finalize_pending_candle(self):
        """Finaliza candle pendente quando dados históricos terminam"""
        try:
            if self.current_candle is not None:
                # Criar DataFrame com o candle pendente
                candle_df = pd.DataFrame([self.current_candle])
                candle_df.set_index('timestamp', inplace=True)
                
                # Adicionar ao DataFrame principal
                if self.candles_df.empty:
                    self.candles_df = candle_df.copy()
                else:
                    self.candles_df = pd.concat([self.candles_df, candle_df])
                
                self.logger.info(f"📊 Candle pendente finalizado. Total: {len(self.candles_df)}")
                
                # Limpar candle atual
                self.current_candle = None
                
                return candle_df
            return None
            
        except Exception as e:
            self.logger.error(f"Erro finalizando candle pendente: {e}")
            return None
    def create_sample_data(self, count: int = 100) -> pd.DataFrame:
        '''Cria dados de exemplo para testes'''
        
        # Gerar timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=count)
        timestamps = pd.date_range(start=start_time, periods=count, freq='1min')
        
        # Preço base WDO
        base_price = 5600
        
        # Gerar preços com walk random
        price_changes = np.random.randn(count) * 0.2
        prices = base_price + np.cumsum(price_changes)
        
        # Criar OHLC realístico
        spreads = np.random.uniform(0.5, 2.0, count)
        
        data = {
            'timestamp': timestamps,
            'open': prices + np.random.uniform(-spreads/2, spreads/2, count),
            'high': prices + np.abs(np.random.uniform(0, spreads, count)),
            'low': prices - np.abs(np.random.uniform(0, spreads, count)),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, count),
            'trades': np.random.randint(100, 2000, count),
            'buy_volume': np.random.randint(400000, 6000000, count),
            'sell_volume': np.random.randint(400000, 6000000, count)
        }
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Garantir que high >= open,close >= low
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        self.logger.info(f"📊 Dados de exemplo criados: {len(df)} candles")
        
        return df
        
    def load_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        '''Carrega dados históricos'''
        try:
            return self.create_sample_data(days * 24 * 60)  # Minutely data
        except Exception as e:
            self.logger.error(f"Erro carregando dados históricos: {e}")
            return pd.DataFrame()
