# -*- coding: utf-8 -*-
"""
ZMQ Valkey Bridge - Ponte entre ZMQ e Valkey
"""

import zmq
import orjson
import threading
import logging
import time
from typing import Dict, Optional
from datetime import datetime

class ZMQValkeyBridge:
    """
    Ponte entre ZMQ e Valkey
    Consome dados ZMQ e armazena em Valkey para time travel
    """
    
    def __init__(self, valkey_manager):
        self.valkey_manager = valkey_manager
        self.context = zmq.Context()
        self.running = False
        self.threads = []
        self.logger = logging.getLogger('ZMQValkeyBridge')
        self.stats = {
            'ticks_bridged': 0,
            'features_bridged': 0,
            'signals_bridged': 0,
            'errors': 0,
            'last_tick_time': None
        }
        
    def start(self):
        """Inicia ponte ZMQ → Valkey"""
        if self.running:
            self.logger.warning("Bridge já está rodando")
            return
            
        from src.config.zmq_valkey_config import ZMQValkeyConfig
        
        self.running = True
        urls = ZMQValkeyConfig.get_zmq_urls()
        
        # Criar thread para cada tipo de dado
        for data_type, url in urls.items():
            thread = threading.Thread(
                target=self._bridge_loop,
                args=(data_type, url),
                daemon=True,
                name=f"Bridge-{data_type}"
            )
            thread.start()
            self.threads.append(thread)
            self.logger.info(f"Bridge thread para {data_type} iniciada")
        
        self.logger.info("ZMQ-Valkey Bridge iniciada com sucesso")
    
    def _bridge_loop(self, data_type: str, url: str):
        """Loop principal da ponte para um tipo de dado"""
        socket = self.context.socket(zmq.SUB)
        socket.connect(url)
        socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscrever a tudo
        socket.setsockopt(zmq.RCVTIMEO, 1000)  # Timeout 1s
        socket.setsockopt(zmq.RCVHWM, 10000)  # High water mark
        
        self.logger.info(f"Bridge {data_type} conectada em {url}")
        
        while self.running:
            try:
                # Receber mensagem
                topic, data = socket.recv_multipart()
                
                # Parse dados
                parsed_data = orjson.loads(data)
                
                # Processar baseado no tipo
                if data_type == 'tick':
                    self._process_tick_to_valkey(parsed_data)
                elif data_type == 'signal':
                    self._process_signal_to_valkey(parsed_data)
                elif data_type == 'history':
                    self._process_history_to_valkey(parsed_data)
                elif data_type == 'book':
                    self._process_book_to_valkey(parsed_data)
                    
            except zmq.Again:
                # Timeout normal, continuar
                continue
            except Exception as e:
                self.stats['errors'] += 1
                if self.stats['errors'] % 100 == 0:
                    self.logger.error(f"Erro na ponte {data_type} ({self.stats['errors']} erros): {e}")
                time.sleep(0.01)  # Pequena pausa em caso de erro
        
        socket.close()
        self.logger.info(f"Bridge {data_type} finalizada")
    
    def _process_tick_to_valkey(self, tick_data: Dict):
        """Processa tick e armazena no Valkey"""
        try:
            symbol = tick_data.get('symbol')
            if not symbol:
                return
            
            # Adicionar ao stream de ticks
            entry_id = self.valkey_manager.add_tick(symbol, tick_data)
            
            if entry_id:
                self.stats['ticks_bridged'] += 1
                self.stats['last_tick_time'] = datetime.now()
                
                # Agregar em candles (opcional - implementar se necessário)
                # self._aggregate_to_candles(symbol, tick_data)
                
                # Log a cada 1000 ticks
                if self.stats['ticks_bridged'] % 1000 == 0:
                    self.logger.info(f"Bridge: {self.stats['ticks_bridged']} ticks processados")
                
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Erro ao processar tick para Valkey: {e}")
    
    def _process_signal_to_valkey(self, signal_data: Dict):
        """Processa sinais/features e armazena no Valkey"""
        try:
            symbol = signal_data.get('symbol')
            data_type = signal_data.get('type', 'signal')
            
            if not symbol:
                return
            
            if data_type == 'features':
                # Armazenar features
                features = signal_data.get('features', {})
                entry_id = self.valkey_manager.add_features(symbol, features)
                
                if entry_id:
                    self.stats['features_bridged'] += 1
                    
            elif data_type == 'signal':
                # Armazenar sinal de trading
                stream_key = f"stream:signals:{symbol}"
                timestamp_ms = signal_data.get('timestamp_ms', int(time.time() * 1000))
                
                signal_bytes = {
                    b'timestamp_ms': str(timestamp_ms).encode(),
                    b'signal': orjson.dumps(signal_data.get('signal', {}))
                }
                
                self.valkey_manager.client.xadd(
                    stream_key,
                    signal_bytes,
                    id=f"{timestamp_ms}-*",
                    maxlen=10000,
                    approximate=True
                )
                
                self.stats['signals_bridged'] += 1
                
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Erro ao processar signal para Valkey: {e}")
    
    def _process_history_to_valkey(self, history_data: Dict):
        """Processa dados históricos e armazena no Valkey"""
        try:
            symbol = history_data.get('symbol')
            if not symbol:
                return
            
            # Armazenar como tick histórico
            stream_key = f"stream:ticks:{symbol}:historical"
            
            # Usar timestamp do dado histórico
            timestamp_ms = history_data.get('timestamp_ms', int(time.time() * 1000))
            
            history_bytes = {}
            for k, v in history_data.items():
                if isinstance(v, (int, float)):
                    history_bytes[k.encode()] = str(v).encode()
                else:
                    history_bytes[k.encode()] = str(v).encode()
            
            self.valkey_manager.client.xadd(
                stream_key,
                history_bytes,
                id=f"{timestamp_ms}-*",
                maxlen=1000000,  # Maior limite para histórico
                approximate=True
            )
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Erro ao processar history: {e}")
    
    def _process_book_to_valkey(self, book_data: Dict):
        """Processa book e armazena no Valkey"""
        try:
            symbol = book_data.get('symbol')
            if not symbol:
                return
            
            # Armazenar snapshot do book
            stream_key = f"stream:book:{symbol}"
            timestamp_ms = book_data.get('timestamp_ms', int(time.time() * 1000))
            
            book_bytes = {}
            for k, v in book_data.items():
                if isinstance(v, (int, float)):
                    book_bytes[k.encode()] = str(v).encode()
                else:
                    book_bytes[k.encode()] = str(v).encode()
            
            self.valkey_manager.client.xadd(
                stream_key,
                book_bytes,
                id=f"{timestamp_ms}-*",
                maxlen=50000,  # Book muda muito
                approximate=True
            )
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Erro ao processar book: {e}")
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas da ponte"""
        return self.stats.copy()
    
    def stop(self):
        """Para a ponte"""
        if not self.running:
            return
            
        self.logger.info("Parando ZMQ-Valkey Bridge...")
        self.running = False
        
        # Aguardar threads terminarem
        for thread in self.threads:
            thread.join(timeout=2)
        
        # Fechar contexto ZMQ
        self.context.term()
        
        self.logger.info(f"Bridge parada - Stats finais: {self.stats}")