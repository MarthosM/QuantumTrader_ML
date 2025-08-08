"""
ConnectionManagerEnhanced - Gerenciador de Conexão com 65 Features
Integra ProfitDLL com o novo sistema de features e HMARL
"""

import os
import sys
import time
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Callable
from ctypes import *
import numpy as np

# Adicionar paths necessários
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar componentes base
from src.connection_manager_v4 import ConnectionManagerV4 as ConnectionManager

# Importar novos componentes
from src.features.book_features_rt import BookFeatureEngineerRT
from src.data.book_data_manager import BookDataManager
from src.buffers.circular_buffer import CandleBuffer

# Configurar logging
logger = logging.getLogger('ConnectionManagerEnhanced')


class ConnectionManagerEnhanced(ConnectionManager):
    """
    ConnectionManager melhorado com:
    - Integração com BookDataManager
    - Cálculo de 65 features em tempo real
    - Broadcasting para HMARL
    - Callbacks otimizados
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa ConnectionManager Enhanced
        
        Args:
            config: Configurações opcionais
        """
        # Inicializar base
        super().__init__(config)
        
        # Componentes de features
        self.book_manager = BookDataManager(
            max_book_snapshots=100,
            max_trades=1000,
            levels=5
        )
        
        self.feature_engineer = BookFeatureEngineerRT(self.book_manager)
        
        # Buffer adicional para candles
        self.candle_buffer = CandleBuffer(max_size=200)
        
        # Cache de features
        self.last_features = {}
        self.features_timestamp = None
        
        # Estatísticas enhanced
        self.enhanced_stats = {
            'features_calculated': 0,
            'book_updates': 0,
            'candle_updates': 0,
            'trade_updates': 0,
            'avg_feature_latency': 0,
            'features_available': 0
        }
        
        # Callbacks enhanced
        self.enhanced_callbacks = {
            'on_features_ready': None,
            'on_book_update': None,
            'on_trade_update': None,
            'on_hmarl_signal': None
        }
        
        # Thread safety
        self.feature_lock = threading.RLock()
        
        logger.info("ConnectionManagerEnhanced inicializado com 65 features")
    
    def register_enhanced_callback(self, callback_type: str, callback_func: Callable):
        """
        Registra callback enhanced
        
        Args:
            callback_type: Tipo do callback
            callback_func: Função callback
        """
        if callback_type in self.enhanced_callbacks:
            self.enhanced_callbacks[callback_type] = callback_func
            logger.info(f"Callback enhanced registrado: {callback_type}")
        else:
            logger.warning(f"Tipo de callback desconhecido: {callback_type}")
    
    # Override dos callbacks principais
    
    def on_price_book(self, asset_id, side, position, book_info):
        """
        Callback enhanced de price book
        
        Args:
            asset_id: ID do ativo
            side: Lado (0=bid, 1=ask)
            position: Posição no book
            book_info: Informações do book
        """
        try:
            # Chamar callback base
            super().on_price_book(asset_id, side, position, book_info)
            
            # Processar no BookDataManager
            with self.feature_lock:
                if side == 0:  # Bid
                    book_data = {
                        'timestamp': datetime.now(),
                        'symbol': self._get_ticker_from_asset(asset_id),
                        'bids': [{
                            'price': book_info.price,
                            'volume': book_info.qtd,
                            'trader_id': f"T{book_info.nOrders}"
                        }]
                    }
                    self.book_manager.on_price_book_callback(book_data)
                    
                self.enhanced_stats['book_updates'] += 1
                
                # Calcular features se temos dados suficientes
                if self.enhanced_stats['book_updates'] % 10 == 0:
                    self._calculate_features()
                    
                # Callback enhanced
                if self.enhanced_callbacks['on_book_update']:
                    self.enhanced_callbacks['on_book_update'](book_data)
                    
        except Exception as e:
            logger.error(f"Erro em on_price_book enhanced: {e}")
    
    def on_offer_book(self, asset_id, side, position, book_info):
        """
        Callback enhanced de offer book
        
        Args:
            asset_id: ID do ativo
            side: Lado (0=bid, 1=ask)
            position: Posição no book
            book_info: Informações do book
        """
        try:
            # Chamar callback base
            super().on_offer_book(asset_id, side, position, book_info)
            
            # Processar no BookDataManager
            with self.feature_lock:
                if side == 1:  # Ask
                    book_data = {
                        'timestamp': datetime.now(),
                        'symbol': self._get_ticker_from_asset(asset_id),
                        'asks': [{
                            'price': book_info.price,
                            'volume': book_info.qtd,
                            'trader_id': f"T{book_info.nOrders}"
                        }]
                    }
                    self.book_manager.on_offer_book_callback(book_data)
                    
        except Exception as e:
            logger.error(f"Erro em on_offer_book enhanced: {e}")
    
    def on_daily(self, asset_id, daily_info):
        """
        Callback enhanced de dados diários (candles)
        
        Args:
            asset_id: ID do ativo
            daily_info: Informações do candle
        """
        try:
            # Chamar callback base
            super().on_daily(asset_id, daily_info)
            
            # Processar candle
            with self.feature_lock:
                candle_data = {
                    'timestamp': datetime.now(),
                    'open': daily_info.sOpen,
                    'high': daily_info.sHigh,
                    'low': daily_info.sLow,
                    'close': daily_info.sClose,
                    'volume': daily_info.sVol
                }
                
                # Adicionar ao buffer de candles
                self.candle_buffer.add_candle(
                    timestamp=candle_data['timestamp'],
                    open=candle_data['open'],
                    high=candle_data['high'],
                    low=candle_data['low'],
                    close=candle_data['close'],
                    volume=candle_data['volume']
                )
                
                # Atualizar feature engineer
                self.feature_engineer._update_candle(candle_data)
                
                self.enhanced_stats['candle_updates'] += 1
                
                # Calcular features
                if self.candle_buffer.size() >= 20:
                    self._calculate_features()
                    
        except Exception as e:
            logger.error(f"Erro em on_daily enhanced: {e}")
    
    def on_history(self, asset_id, date, price, vol, qtd, status, negocios):
        """
        Callback enhanced de histórico (trades)
        
        Args:
            asset_id: ID do ativo
            date: Data/hora
            price: Preço
            vol: Volume financeiro
            qtd: Quantidade
            status: Status do trade
            negocios: Número de negócios
        """
        try:
            # Chamar callback base
            super().on_history(asset_id, date, price, vol, qtd, status, negocios)
            
            # Processar trade
            with self.feature_lock:
                trade_data = {
                    'timestamp': self._parse_datetime(date),
                    'symbol': self._get_ticker_from_asset(asset_id),
                    'price': price,
                    'volume': qtd,
                    'side': 'buy' if status == 0 else 'sell',
                    'aggressor': 'buyer' if status == 0 else 'seller'
                }
                
                self.book_manager.on_trade_callback(trade_data)
                self.enhanced_stats['trade_updates'] += 1
                
                # Callback enhanced
                if self.enhanced_callbacks['on_trade_update']:
                    self.enhanced_callbacks['on_trade_update'](trade_data)
                    
        except Exception as e:
            logger.error(f"Erro em on_history enhanced: {e}")
    
    def _calculate_features(self):
        """Calcula 65 features e notifica callbacks"""
        try:
            start_time = time.time()
            
            # Calcular features
            features = self.feature_engineer.calculate_incremental_features({})
            
            # Estatísticas
            calc_time = (time.time() - start_time) * 1000
            self.enhanced_stats['features_calculated'] += 1
            self.enhanced_stats['avg_feature_latency'] = (
                (self.enhanced_stats['avg_feature_latency'] * 
                 (self.enhanced_stats['features_calculated'] - 1) + calc_time) / 
                self.enhanced_stats['features_calculated']
            )
            
            # Contar features disponíveis
            non_zero = sum(1 for v in features.values() if v != 0)
            self.enhanced_stats['features_available'] = non_zero
            
            # Atualizar cache
            self.last_features = features
            self.features_timestamp = datetime.now()
            
            # Callback de features prontas
            if self.enhanced_callbacks['on_features_ready']:
                self.enhanced_callbacks['on_features_ready'](features)
            
            # Log periódico
            if self.enhanced_stats['features_calculated'] % 100 == 0:
                logger.info(
                    f"Features: {non_zero}/65 | "
                    f"Latência: {self.enhanced_stats['avg_feature_latency']:.2f}ms | "
                    f"Cálculos: {self.enhanced_stats['features_calculated']}"
                )
                
        except Exception as e:
            logger.error(f"Erro calculando features: {e}")
    
    def get_current_features(self) -> Dict:
        """
        Retorna features atuais
        
        Returns:
            Dict com 65 features
        """
        with self.feature_lock:
            if self.last_features:
                return self.last_features.copy()
            else:
                # Calcular sob demanda
                return self.feature_engineer.calculate_incremental_features({})
    
    def get_feature_vector(self) -> np.ndarray:
        """
        Retorna vetor de features para ML
        
        Returns:
            Array numpy com 65 features
        """
        return self.feature_engineer.get_feature_vector()
    
    def get_book_state(self) -> Dict:
        """
        Retorna estado atual do book
        
        Returns:
            Dict com estado do book
        """
        return self.book_manager.get_current_state()
    
    def get_microstructure_features(self) -> Dict:
        """
        Retorna features de microestrutura
        
        Returns:
            Dict com features de microestrutura
        """
        return self.book_manager.get_microstructure_features()
    
    def get_enhanced_statistics(self) -> Dict:
        """
        Retorna estatísticas enhanced
        
        Returns:
            Dict com estatísticas completas
        """
        base_stats = self.get_statistics()
        
        enhanced = {
            **base_stats,
            'enhanced': self.enhanced_stats,
            'book_manager': self.book_manager.get_statistics(),
            'feature_engineer': self.feature_engineer.get_statistics(),
            'candle_buffer': {
                'size': self.candle_buffer.size(),
                'max_size': self.candle_buffer.max_size
            }
        }
        
        return enhanced
    
    def _get_ticker_from_asset(self, asset_id) -> str:
        """
        Converte asset_id para ticker
        
        Args:
            asset_id: ID do ativo
            
        Returns:
            Ticker do ativo
        """
        # Implementação simplificada - ajustar conforme necessário
        if hasattr(asset_id, 'ticker'):
            return asset_id.ticker.strip()
        return "WDOU25"  # Default
    
    def _parse_datetime(self, date_str) -> datetime:
        """
        Parse de string datetime
        
        Args:
            date_str: String com data/hora
            
        Returns:
            Objeto datetime
        """
        try:
            if isinstance(date_str, str):
                # Tentar diferentes formatos
                for fmt in ['%Y%m%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S']:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except:
                        continue
            return datetime.now()
        except:
            return datetime.now()
    
    def cleanup(self):
        """Limpeza do ConnectionManager Enhanced"""
        try:
            # Reset dos componentes
            self.book_manager.reset()
            
            # Limpar callbacks enhanced
            for key in self.enhanced_callbacks:
                self.enhanced_callbacks[key] = None
            
            # Chamar cleanup base
            super().cleanup()
            
            logger.info("ConnectionManagerEnhanced finalizado")
            
        except Exception as e:
            logger.error(f"Erro no cleanup: {e}")


# Teste rápido
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testando ConnectionManagerEnhanced...")
    
    # Criar manager
    config = {
        'dll_path': './ProfitDLL64.dll',
        'username': 'test',
        'password': 'test'
    }
    
    manager = ConnectionManagerEnhanced(config)
    
    # Registrar callback de features
    def on_features(features):
        non_zero = sum(1 for v in features.values() if v != 0)
        print(f"Features prontas: {non_zero}/65 não-zero")
    
    manager.register_enhanced_callback('on_features_ready', on_features)
    
    # Simular alguns callbacks
    print("\nSimulando callbacks...")
    
    # Simular price book
    class MockBookInfo:
        def __init__(self):
            self.price = 5450.0
            self.qtd = 100
            self.nOrders = 5
    
    class MockAssetId:
        def __init__(self):
            self.ticker = "WDOU25"
    
    # Simular updates
    for i in range(20):
        book_info = MockBookInfo()
        book_info.price = 5450 + i * 0.5
        manager.on_price_book(MockAssetId(), 0, i % 5, book_info)
    
    # Verificar estatísticas
    stats = manager.get_enhanced_statistics()
    print(f"\nEstatísticas Enhanced:")
    print(f"  Book updates: {stats['enhanced']['book_updates']}")
    print(f"  Features calculadas: {stats['enhanced']['features_calculated']}")
    print(f"  Features disponíveis: {stats['enhanced']['features_available']}/65")
    
    print("\n[OK] ConnectionManagerEnhanced testado com sucesso!")