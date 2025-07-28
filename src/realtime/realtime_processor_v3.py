"""
RealTimeProcessorV3 - Processamento em tempo real com dados V3
Baseado no DEVELOPER_GUIDE_V3_REFACTORING.md
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import threading
import queue
from collections import deque
import time

# Imports internos
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from features.ml_features_v3 import MLFeaturesV3
from data.trading_data_structure_v3 import TradingDataStructureV3


class RealTimeProcessorV3:
    """
    Processa dados em tempo real e mantém features atualizadas
    
    Features:
    - Processa callbacks de trades e book em tempo real
    - Calcula features ML continuamente
    - Mantém buffer otimizado de dados recentes
    - Thread-safe para múltiplos acessos
    - Monitora latência e performance
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o processador em tempo real
        
        Args:
            config: Configurações opcionais
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configurações
        self.buffer_size = self.config.get('buffer_size', 1000)
        self.feature_update_interval = self.config.get('feature_update_interval', 1.0)  # segundos
        self.max_latency_ms = self.config.get('max_latency_ms', 100)
        
        # Estrutura de dados principal
        self.data_structure = TradingDataStructureV3(max_history=self.buffer_size)
        
        # Calculador de features
        self.feature_calculator = MLFeaturesV3()
        
        # Filas para processamento assíncrono
        self.trade_queue = queue.Queue(maxsize=10000)
        self.book_queue = queue.Queue(maxsize=10000)
        
        # Cache de features
        self.features_cache = pd.DataFrame()
        self.features_lock = threading.RLock()
        
        # Métricas de performance
        self.metrics = {
            'trades_processed': 0,
            'books_processed': 0,
            'features_calculated': 0,
            'avg_latency_ms': 0,
            'max_latency_ms': 0,
            'errors': 0
        }
        self.latency_buffer = deque(maxlen=1000)
        
        # Threads de processamento
        self.processing_active = False
        self.trade_thread = None
        self.book_thread = None
        self.feature_thread = None
        
        # Último timestamp de atualização
        self.last_feature_update = None
        self.last_data_timestamp = None
        
    def start(self):
        """Inicia processamento em tempo real"""
        self.logger.info("Iniciando RealTimeProcessorV3...")
        
        self.processing_active = True
        
        # Iniciar threads
        self.trade_thread = threading.Thread(target=self._process_trades, daemon=True)
        self.book_thread = threading.Thread(target=self._process_books, daemon=True)
        self.feature_thread = threading.Thread(target=self._update_features_loop, daemon=True)
        
        self.trade_thread.start()
        self.book_thread.start()
        self.feature_thread.start()
        
        self.logger.info("RealTimeProcessorV3 iniciado com 3 threads")
        
    def stop(self):
        """Para processamento em tempo real"""
        self.logger.info("Parando RealTimeProcessorV3...")
        
        self.processing_active = False
        
        # Aguardar threads terminarem
        if self.trade_thread:
            self.trade_thread.join(timeout=2)
        if self.book_thread:
            self.book_thread.join(timeout=2)
        if self.feature_thread:
            self.feature_thread.join(timeout=2)
        
        self.logger.info("RealTimeProcessorV3 parado")
        
    def add_trade(self, trade_data: Dict):
        """
        Adiciona trade para processamento
        
        Args:
            trade_data: Dados do trade incluindo side real
        """
        try:
            # Adicionar timestamp de recebimento
            trade_data['received_at'] = time.time()
            
            # Adicionar à fila (non-blocking)
            self.trade_queue.put_nowait(trade_data)
            
        except queue.Full:
            self.logger.warning("Fila de trades cheia, descartando trade")
            self.metrics['errors'] += 1
            
    def add_book_update(self, book_data: Dict):
        """
        Adiciona atualização de book
        
        Args:
            book_data: Dados do book
        """
        try:
            # Adicionar timestamp de recebimento
            book_data['received_at'] = time.time()
            
            # Adicionar à fila (non-blocking)
            self.book_queue.put_nowait(book_data)
            
        except queue.Full:
            self.logger.warning("Fila de book cheia, descartando update")
            self.metrics['errors'] += 1
            
    def _process_trades(self):
        """Thread para processar trades"""
        self.logger.info("Thread de processamento de trades iniciada")
        
        while self.processing_active:
            try:
                # Pegar trade da fila com timeout
                trade = self.trade_queue.get(timeout=0.1)
                
                # Calcular latência
                latency_ms = (time.time() - trade['received_at']) * 1000
                self.latency_buffer.append(latency_ms)
                
                # Processar trade
                self._process_single_trade(trade)
                
                # Atualizar métricas
                self.metrics['trades_processed'] += 1
                
                if latency_ms > self.max_latency_ms:
                    self.logger.warning(f"Latência alta no trade: {latency_ms:.1f}ms")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Erro processando trade: {e}")
                self.metrics['errors'] += 1
                
    def _process_single_trade(self, trade: Dict):
        """Processa um único trade"""
        
        # Extrair dados necessários
        timestamp = trade.get('datetime')
        price = float(trade.get('price', 0))
        volume = float(trade.get('volume', 0))
        side = trade.get('side', 'UNKNOWN')
        
        if not timestamp or price <= 0 or volume <= 0:
            return
        
        # Adicionar à estrutura de dados
        self.data_structure.add_trade({
            'timestamp': timestamp,
            'price': price,
            'volume': volume,
            'side': side,
            'quantity': trade.get('quantity', 1),
            'trade_id': trade.get('trade_id'),
            'sequence': trade.get('sequence')
        })
        
        # Atualizar último timestamp
        self.last_data_timestamp = timestamp
        
    def _process_books(self):
        """Thread para processar book updates"""
        self.logger.info("Thread de processamento de book iniciada")
        
        while self.processing_active:
            try:
                # Pegar book update da fila
                book = self.book_queue.get(timeout=0.1)
                
                # Calcular latência
                latency_ms = (time.time() - book['received_at']) * 1000
                
                # Processar book
                self._process_single_book(book)
                
                # Atualizar métricas
                self.metrics['books_processed'] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Erro processando book: {e}")
                self.metrics['errors'] += 1
                
    def _process_single_book(self, book: Dict):
        """Processa um único book update"""
        
        # Extrair dados
        timestamp = book.get('datetime')
        side = book.get('side')  # 'bid' ou 'ask'
        level = book.get('level', 0)
        price = float(book.get('price', 0))
        quantity = int(book.get('quantity', 0))
        
        if not timestamp or price <= 0:
            return
        
        # Adicionar à estrutura de dados
        self.data_structure.add_book_update({
            'timestamp': timestamp,
            'side': side,
            'level': level,
            'price': price,
            'quantity': quantity,
            'total_quantity': book.get('total_quantity', quantity)
        })
        
    def _update_features_loop(self):
        """Thread para atualizar features periodicamente"""
        self.logger.info("Thread de atualização de features iniciada")
        
        while self.processing_active:
            try:
                # Aguardar intervalo
                time.sleep(self.feature_update_interval)
                
                # Verificar se há dados novos
                if self.last_data_timestamp and (
                    not self.last_feature_update or 
                    self.last_data_timestamp > self.last_feature_update
                ):
                    # Atualizar features
                    self._update_features()
                    
            except Exception as e:
                self.logger.error(f"Erro atualizando features: {e}")
                self.metrics['errors'] += 1
                
    def _update_features(self):
        """Atualiza cálculo de features"""
        
        start_time = time.time()
        
        try:
            # Obter dados mais recentes
            candles = self.data_structure.get_candles(100)
            microstructure = self.data_structure.get_microstructure(100)
            
            if candles.empty or microstructure.empty:
                return
            
            # Calcular features
            features = self.feature_calculator.calculate_all(
                candles=candles,
                microstructure=microstructure
            )
            
            # Atualizar cache com lock
            with self.features_lock:
                self.features_cache = features
                self.last_feature_update = datetime.now()
            
            # Métricas
            calc_time_ms = (time.time() - start_time) * 1000
            self.metrics['features_calculated'] += 1
            
            if calc_time_ms > 100:
                self.logger.warning(f"Cálculo de features lento: {calc_time_ms:.1f}ms")
                
            self.logger.debug(f"Features atualizadas em {calc_time_ms:.1f}ms")
            
        except Exception as e:
            self.logger.error(f"Erro calculando features: {e}")
            self.metrics['errors'] += 1
            
    def get_latest_features(self, n_rows: int = 1) -> pd.DataFrame:
        """
        Retorna features mais recentes
        
        Args:
            n_rows: Número de linhas a retornar
            
        Returns:
            DataFrame com features mais recentes
        """
        with self.features_lock:
            if self.features_cache.empty:
                return pd.DataFrame()
            
            return self.features_cache.tail(n_rows).copy()
            
    def get_metrics(self) -> Dict:
        """Retorna métricas de performance"""
        
        # Calcular latência média
        if self.latency_buffer:
            avg_latency = np.mean(list(self.latency_buffer))
            max_latency = np.max(list(self.latency_buffer))
        else:
            avg_latency = 0
            max_latency = 0
            
        self.metrics['avg_latency_ms'] = avg_latency
        self.metrics['max_latency_ms'] = max_latency
        
        # Adicionar uptime
        if hasattr(self, 'start_time'):
            self.metrics['uptime_seconds'] = time.time() - self.start_time
            
        return self.metrics.copy()
        
    def health_check(self) -> Dict[str, bool]:
        """Verifica saúde do processador"""
        
        health = {
            'threads_alive': all([
                self.trade_thread and self.trade_thread.is_alive(),
                self.book_thread and self.book_thread.is_alive(),
                self.feature_thread and self.feature_thread.is_alive()
            ]),
            'queues_healthy': all([
                self.trade_queue.qsize() < self.trade_queue.maxsize * 0.8,
                self.book_queue.qsize() < self.book_queue.maxsize * 0.8
            ]),
            'features_recent': (
                self.last_feature_update and
                (datetime.now() - self.last_feature_update).seconds < 10
            ),
            'low_errors': self.metrics['errors'] < 100
        }
        
        health['overall'] = all(health.values())
        
        return health


def main():
    """Teste do RealTimeProcessorV3"""
    
    print("="*60)
    print("TESTE DO REALTIME PROCESSOR V3")
    print("="*60)
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Criar processador
    processor = RealTimeProcessorV3({
        'buffer_size': 500,
        'feature_update_interval': 1.0,
        'max_latency_ms': 50
    })
    
    # Iniciar processamento
    processor.start()
    
    # Simular dados chegando
    print("\nSimulando dados em tempo real...")
    
    base_price = 5900
    for i in range(100):
        # Simular trade
        trade = {
            'datetime': datetime.now(),
            'price': base_price + np.random.randn() * 5,
            'volume': np.random.randint(1000, 5000),
            'side': np.random.choice(['BUY', 'SELL']),
            'quantity': np.random.randint(1, 10),
            'trade_id': f'T{i}',
            'sequence': i
        }
        processor.add_trade(trade)
        
        # Simular book update
        if i % 5 == 0:
            book = {
                'datetime': datetime.now(),
                'side': np.random.choice(['bid', 'ask']),
                'level': 0,
                'price': base_price + np.random.randn() * 2,
                'quantity': np.random.randint(10, 100)
            }
            processor.add_book_update(book)
        
        time.sleep(0.01)  # 10ms entre eventos
    
    # Aguardar processamento
    print("\nAguardando processamento...")
    time.sleep(3)
    
    # Verificar métricas
    metrics = processor.get_metrics()
    print(f"\nMétricas:")
    print(f"  Trades processados: {metrics['trades_processed']}")
    print(f"  Books processados: {metrics['books_processed']}")
    print(f"  Features calculadas: {metrics['features_calculated']}")
    print(f"  Latência média: {metrics['avg_latency_ms']:.1f}ms")
    print(f"  Erros: {metrics['errors']}")
    
    # Verificar features
    features = processor.get_latest_features(5)
    print(f"\nFeatures shape: {features.shape}")
    if not features.empty:
        print(f"Últimas features calculadas")
    
    # Health check
    health = processor.health_check()
    print(f"\nHealth check:")
    for key, value in health.items():
        status = "[OK]" if value else "[FAIL]"
        print(f"  {key}: {status}")
    
    # Parar processador
    processor.stop()
    
    print("\n[OK] Teste concluído!")


if __name__ == "__main__":
    main()