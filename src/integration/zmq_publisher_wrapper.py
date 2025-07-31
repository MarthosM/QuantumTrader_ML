# -*- coding: utf-8 -*-
"""
ZMQ Publisher Wrapper - Adiciona publicação ZMQ aos callbacks existentes
"""

import zmq
import orjson
import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime
import time

class ZMQPublisherWrapper:
    """
    Wrapper que adiciona publicação ZMQ aos callbacks existentes
    sem modificar o código original
    """
    
    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
        self.context = zmq.Context()
        self.publishers = {}
        self.logger = logging.getLogger('ZMQPublisher')
        self.stats = {
            'ticks_published': 0,
            'trades_published': 0,
            'books_published': 0,
            'errors': 0
        }
        
        # Criar publishers
        self._setup_publishers()
        
        # Interceptar callbacks
        self._setup_callback_interceptors()
        
    def _setup_publishers(self):
        """Cria publishers ZMQ para diferentes tipos de dados"""
        try:
            from src.config.zmq_valkey_config import ZMQValkeyConfig
            
            urls = ZMQValkeyConfig.get_zmq_urls()
            
            for data_type, url in urls.items():
                socket = self.context.socket(zmq.PUB)
                socket.bind(url)
                socket.setsockopt(zmq.LINGER, 0)  # Não bloquear ao fechar
                socket.setsockopt(zmq.SNDHWM, 10000)  # High water mark
                self.publishers[data_type] = socket
                self.logger.info(f"ZMQ Publisher {data_type} iniciado em {url}")
                
        except Exception as e:
            self.logger.error(f"Erro ao criar publishers ZMQ: {e}")
            raise
    
    def _setup_callback_interceptors(self):
        """Intercepta callbacks do ConnectionManager"""
        
        # Verificar se tem callback configurado
        if not hasattr(self.connection_manager, 'callback'):
            self.logger.warning("ConnectionManager sem callback configurado")
            return
            
        callback = self.connection_manager.callback
        
        # Interceptar newTradeCallback
        if hasattr(callback, 'newTradeCallback'):
            original_trade = callback.newTradeCallback
            
            def enhanced_trade_callback(assetId, date, tradeNumber, price, vol, qty, buyAgent, sellAgent, tradeType, bIsEdit):
                # Executar callback original primeiro
                try:
                    original_trade(assetId, date, tradeNumber, price, vol, qty, buyAgent, sellAgent, tradeType, bIsEdit)
                except Exception as e:
                    self.logger.error(f"Erro no callback original: {e}")
                
                # Publicar via ZMQ (não bloquear sistema principal)
                try:
                    # Extrair ticker do assetId
                    ticker = assetId.ticker if hasattr(assetId, 'ticker') else str(assetId)
                    
                    tick_data = {
                        'type': 'tick',
                        'symbol': ticker,
                        'timestamp': str(date),
                        'timestamp_ms': int(datetime.now().timestamp() * 1000),
                        'trade_id': tradeNumber,
                        'price': float(price),
                        'volume': float(vol),
                        'quantity': int(qty),
                        'buyer': buyAgent,
                        'seller': sellAgent,
                        'trade_type': tradeType,
                        'is_edit': bIsEdit
                    }
                    
                    topic = f"tick_{ticker}".encode()
                    data = orjson.dumps(tick_data)
                    
                    # Enviar sem bloquear
                    self.publishers['tick'].send_multipart([topic, data], zmq.NOBLOCK)
                    self.stats['ticks_published'] += 1
                    
                    # Log apenas a cada 100 mensagens para não poluir
                    if self.stats['ticks_published'] % 100 == 0:
                        self.logger.debug(f"ZMQ: {self.stats['ticks_published']} ticks publicados")
                    
                except zmq.Again:
                    # Buffer cheio, não bloquear
                    pass
                except Exception as e:
                    self.stats['errors'] += 1
                    if self.stats['errors'] % 10 == 0:
                        self.logger.error(f"Erro ao publicar tick ZMQ ({self.stats['errors']} erros): {e}")
            
            # Substituir callback
            callback.newTradeCallback = enhanced_trade_callback
            self.logger.info("newTradeCallback interceptado com sucesso")
        
        # Interceptar newHistoryCallback
        if hasattr(callback, 'newHistoryCallback'):
            original_history = callback.newHistoryCallback
            
            def enhanced_history_callback(assetId, date, tradeNumber, price, vol, qty, buyAgent, sellAgent, tradeType):
                # Executar original
                try:
                    original_history(assetId, date, tradeNumber, price, vol, qty, buyAgent, sellAgent, tradeType)
                except Exception as e:
                    self.logger.error(f"Erro no callback history original: {e}")
                
                # Publicar via ZMQ
                try:
                    ticker = assetId.ticker if hasattr(assetId, 'ticker') else str(assetId)
                    
                    history_data = {
                        'type': 'history',
                        'symbol': ticker,
                        'timestamp': str(date),
                        'timestamp_ms': int(time.time() * 1000),
                        'trade_id': tradeNumber,
                        'price': float(price),
                        'volume': float(vol),
                        'quantity': int(qty),
                        'buyer': buyAgent,
                        'seller': sellAgent,
                        'trade_type': tradeType
                    }
                    
                    topic = f"history_{ticker}".encode()
                    data = orjson.dumps(history_data)
                    
                    self.publishers['history'].send_multipart([topic, data], zmq.NOBLOCK)
                    self.stats['trades_published'] += 1
                    
                except zmq.Again:
                    pass
                except Exception as e:
                    self.stats['errors'] += 1
                    if self.stats['errors'] % 10 == 0:
                        self.logger.error(f"Erro ao publicar history: {e}")
            
            callback.newHistoryCallback = enhanced_history_callback
            self.logger.info("newHistoryCallback interceptado com sucesso")
        
        # Interceptar newBookCallback se existir
        if hasattr(callback, 'newBookCallback'):
            original_book = callback.newBookCallback
            
            def enhanced_book_callback(assetId, side, position, qtd, count, price, stopQtd, avgPx):
                # Executar original
                try:
                    original_book(assetId, side, position, qtd, count, price, stopQtd, avgPx)
                except Exception as e:
                    self.logger.error(f"Erro no callback book original: {e}")
                
                # Publicar via ZMQ
                try:
                    ticker = assetId.ticker if hasattr(assetId, 'ticker') else str(assetId)
                    
                    book_data = {
                        'type': 'book',
                        'symbol': ticker,
                        'timestamp_ms': int(datetime.now().timestamp() * 1000),
                        'side': side,
                        'position': position,
                        'quantity': qtd,
                        'count': count,
                        'price': float(price),
                        'stop_qty': stopQtd,
                        'avg_price': float(avgPx) if avgPx else 0
                    }
                    
                    topic = f"book_{ticker}".encode()
                    data = orjson.dumps(book_data)
                    
                    self.publishers['book'].send_multipart([topic, data], zmq.NOBLOCK)
                    self.stats['books_published'] += 1
                    
                except zmq.Again:
                    pass
                except Exception as e:
                    self.stats['errors'] += 1
            
            callback.newBookCallback = enhanced_book_callback
            self.logger.info("newBookCallback interceptado com sucesso")
    
    def publish_feature_update(self, symbol: str, features: Dict[str, Any]):
        """Publica atualização de features calculadas"""
        try:
            feature_data = {
                'type': 'features',
                'symbol': symbol,
                'timestamp_ms': int(datetime.now().timestamp() * 1000),
                'features': features,
                'feature_count': len(features)
            }
            
            topic = f"features_{symbol}".encode()
            data = orjson.dumps(feature_data)
            
            self.publishers['signal'].send_multipart([topic, data], zmq.NOBLOCK)
            
        except Exception as e:
            self.logger.error(f"Erro ao publicar features: {e}")
    
    def publish_signal(self, symbol: str, signal: Dict[str, Any]):
        """Publica sinal de trading"""
        try:
            signal_data = {
                'type': 'signal',
                'symbol': symbol,
                'timestamp_ms': int(datetime.now().timestamp() * 1000),
                'signal': signal
            }
            
            topic = f"signal_{symbol}".encode()
            data = orjson.dumps(signal_data)
            
            self.publishers['signal'].send_multipart([topic, data], zmq.NOBLOCK)
            
        except Exception as e:
            self.logger.error(f"Erro ao publicar signal: {e}")
    
    def get_stats(self):
        """Retorna estatísticas de publicação"""
        return self.stats.copy()
    
    def close(self):
        """Fecha publishers ZMQ"""
        self.logger.info(f"Fechando ZMQ Publisher - Stats: {self.stats}")
        
        for socket in self.publishers.values():
            socket.close()
        
        self.context.term()
        self.logger.info("ZMQ Publisher fechado")