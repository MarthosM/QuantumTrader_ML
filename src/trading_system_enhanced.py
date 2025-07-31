# -*- coding: utf-8 -*-
"""
Trading System Enhanced - Sistema de trading com ZMQ + Valkey
Mantém compatibilidade total com sistema atual
"""

import logging
import threading
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

class TradingSystemEnhanced:
    """
    Sistema de trading aprimorado com ZMQ + Valkey
    Mantém compatibilidade total com sistema atual
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger('TradingSystemEnhanced')
        
        # Sistema original (sem modificação)
        from src.trading_system import TradingSystem
        self.original_system = TradingSystem(config)
        
        # Configurações enhanced
        from src.config.zmq_valkey_config import ZMQValkeyConfig
        self.zmq_config = ZMQValkeyConfig
        
        # Componentes enhanced (opcionais)
        self.zmq_publisher = None
        self.valkey_manager = None
        self.zmq_valkey_bridge = None
        
        # Stats
        self.enhanced_stats = {
            'zmq_enabled': False,
            'valkey_enabled': False,
            'time_travel_enabled': False,
            'bridge_active': False
        }
        
        # Setup componentes enhanced se habilitados
        if self.zmq_config.is_enhanced_enabled():
            self._setup_enhanced_components()
    
    def _setup_enhanced_components(self):
        """Configura componentes aprimorados se habilitados"""
        
        # 1. Valkey Manager (se habilitado)
        if self.zmq_config.VALKEY_ENABLED:
            try:
                from src.integration.valkey_stream_manager import ValkeyStreamManager
                self.valkey_manager = ValkeyStreamManager()
                self.enhanced_stats['valkey_enabled'] = True
                self.logger.info("✅ Valkey Manager inicializado")
                
                # Criar streams para símbolos configurados
                symbols = self.config.get('symbols', ['WDOQ25'])
                if isinstance(symbols, str):
                    symbols = [symbols]
                    
                for symbol in symbols:
                    self.valkey_manager.create_symbol_streams(symbol)
                    self.logger.info(f"Streams criados para {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Erro ao inicializar Valkey: {e}")
                if not self.zmq_config.FALLBACK_ON_ERROR:
                    raise
        
        # 2. ZMQ Publisher (se habilitado)
        if self.zmq_config.ZMQ_ENABLED:
            try:
                from src.integration.zmq_publisher_wrapper import ZMQPublisherWrapper
                
                # Wrapper para publicar dados via ZMQ
                self.zmq_publisher = ZMQPublisherWrapper(
                    self.original_system.connection_manager
                )
                self.enhanced_stats['zmq_enabled'] = True
                self.logger.info("✅ ZMQ Publisher inicializado")
                
            except Exception as e:
                self.logger.error(f"Erro ao inicializar ZMQ: {e}")
                if not self.zmq_config.FALLBACK_ON_ERROR:
                    raise
        
        # 3. Bridge ZMQ → Valkey (se ambos habilitados)
        if self.enhanced_stats['zmq_enabled'] and self.enhanced_stats['valkey_enabled']:
            try:
                from src.integration.zmq_valkey_bridge import ZMQValkeyBridge
                self.zmq_valkey_bridge = ZMQValkeyBridge(self.valkey_manager)
                self.logger.info("✅ Bridge ZMQ-Valkey criada")
                
            except Exception as e:
                self.logger.error(f"Erro ao criar bridge: {e}")
        
        # 4. Time Travel (se habilitado)
        if self.zmq_config.TIME_TRAVEL_ENABLED and self.valkey_manager:
            self.enhanced_stats['time_travel_enabled'] = True
            self.logger.info("✅ Time Travel habilitado")
        
        # Log status final
        self.logger.info(f"Sistema Enhanced configurado: {self.enhanced_stats}")
    
    def start(self):
        """Inicia sistema completo"""
        
        self.logger.info("🚀 Iniciando Trading System Enhanced...")
        
        # 1. Iniciar sistema original
        self.original_system.start()
        self.logger.info("Sistema original iniciado")
        
        # 2. Iniciar bridge se disponível
        if self.zmq_valkey_bridge:
            self.zmq_valkey_bridge.start()
            self.enhanced_stats['bridge_active'] = True
            self.logger.info("Bridge ZMQ-Valkey iniciada")
        
        # 3. Interceptar feature engine se time travel habilitado
        if self.enhanced_stats['time_travel_enabled']:
            self._setup_time_travel_features()
        
        self.logger.info("✅ Sistema Enhanced iniciado com sucesso")
        self.logger.info(f"Status: {self.get_enhanced_status()}")
    
    def _setup_time_travel_features(self):
        """Configura features com time travel se disponível"""
        try:
            # Criar wrapper para feature engine
            original_feature_engine = self.original_system.feature_engine
            
            # Adicionar método time travel
            def calculate_with_time_travel(symbol: str, lookback_minutes: int = 60):
                """Calcula features usando time travel"""
                
                # Buscar dados históricos
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=lookback_minutes)
                
                # Query time travel
                df = self.valkey_manager.time_travel_to_dataframe(
                    symbol, start_time, end_time, 'ticks'
                )
                
                if df.empty:
                    self.logger.warning(f"Sem dados time travel para {symbol}")
                    return None
                
                # Calcular features usando engine original
                features = original_feature_engine.calculate_features(df)
                
                # Adicionar features enhanced
                features['time_travel_used'] = True
                features['lookback_minutes'] = lookback_minutes
                features['data_points'] = len(df)
                
                return features
            
            # Adicionar método ao feature engine
            self.original_system.feature_engine.calculate_with_time_travel = calculate_with_time_travel
            self.logger.info("Time travel features configuradas")
            
        except Exception as e:
            self.logger.error(f"Erro ao configurar time travel: {e}")
    
    def stop(self):
        """Para sistema completo"""
        
        self.logger.info("⏹️ Parando Trading System Enhanced...")
        
        # 1. Parar bridge
        if self.zmq_valkey_bridge:
            self.zmq_valkey_bridge.stop()
            self.enhanced_stats['bridge_active'] = False
        
        # 2. Fechar ZMQ publisher
        if self.zmq_publisher:
            stats = self.zmq_publisher.get_stats()
            self.logger.info(f"ZMQ Publisher stats: {stats}")
            self.zmq_publisher.close()
        
        # 3. Fechar Valkey
        if self.valkey_manager:
            stats = self.valkey_manager.get_stats()
            self.logger.info(f"Valkey stats: {stats}")
            self.valkey_manager.close()
        
        # 4. Parar sistema original
        self.original_system.stop()
        
        self.logger.info("✅ Sistema Enhanced parado")
    
    def get_time_travel_data(self, symbol: str, start_time: datetime, 
                           end_time: datetime, data_type: str = 'ticks'):
        """
        Interface para time travel queries
        Retorna None se Valkey não estiver habilitado
        """
        if self.valkey_manager and self.enhanced_stats['time_travel_enabled']:
            return self.valkey_manager.time_travel_query(
                symbol, start_time, end_time, data_type
            )
        return None
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Retorna status do sistema enhanced"""
        status = {
            'enhanced_features': self.enhanced_stats.copy(),
            'original_system': {
                'connected': self.original_system.connection_manager.connected,
                'ticker': self.original_system.ticker
            }
        }
        
        # Stats ZMQ
        if self.zmq_publisher:
            status['zmq_stats'] = self.zmq_publisher.get_stats()
        
        # Stats Valkey
        if self.valkey_manager:
            status['valkey_stats'] = self.valkey_manager.get_stats()
        
        # Stats Bridge
        if self.zmq_valkey_bridge:
            status['bridge_stats'] = self.zmq_valkey_bridge.get_stats()
        
        return status
    
    def publish_custom_data(self, data_type: str, symbol: str, data: Dict):
        """Permite publicar dados customizados via ZMQ"""
        if self.zmq_publisher:
            if data_type == 'features':
                self.zmq_publisher.publish_feature_update(symbol, data)
            elif data_type == 'signal':
                self.zmq_publisher.publish_signal(symbol, data)
            else:
                self.logger.warning(f"Tipo de dado não suportado: {data_type}")
    
    # Proxy para métodos do sistema original
    def __getattr__(self, name):
        """Proxy para acessar métodos do sistema original"""
        return getattr(self.original_system, name)