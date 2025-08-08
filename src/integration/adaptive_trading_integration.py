"""
Integração do Sistema de Trading Adaptativo
Conecta o sistema adaptativo com o TradingSystem principal
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import logging
import threading
from pathlib import Path

from ..trading_system import TradingSystem
from ..strategies.adaptive_hybrid_strategy import AdaptiveHybridStrategy
from ..monitoring.adaptive_monitor import AdaptiveMonitor
from ..features.ml_features_v3 import MLFeaturesV3
from ..technical_indicators import TechnicalIndicators
from ..data_structure import TradingDataStructure

class AdaptiveTradingIntegration:
    """
    Integra o sistema de trading adaptativo ao TradingSystem principal
    Gerencia o fluxo de dados e sinais entre componentes
    """
    
    def __init__(self, trading_system: TradingSystem, config: dict):
        self.trading_system = trading_system
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Estratégia adaptativa
        adaptive_config = {
            'models_path': config.get('models_path', 'models'),
            'regime_threshold': config.get('regime_threshold', 0.6),
            'tick_weight': config.get('tick_weight', 0.4),
            'book_weight': config.get('book_weight', 0.6),
            'max_position': config.get('max_position', 2),
            'stop_loss': config.get('stop_loss', 0.02),
            'take_profit': config.get('take_profit', 0.03),
            # Online learning
            'online_buffer_size': config.get('online_buffer_size', 50000),
            'retrain_interval': config.get('retrain_interval', 3600),
            'min_samples_retrain': config.get('min_samples_retrain', 5000),
            'validation_window': config.get('validation_window', 500),
            'performance_threshold': config.get('performance_threshold', 0.55),
            # A/B testing
            'ab_testing_enabled': config.get('ab_testing_enabled', True),
            'ab_test_ratio': config.get('ab_test_ratio', 0.2),
            # Adaptação
            'adaptation_rate': config.get('adaptation_rate', 0.1),
            'performance_window': config.get('performance_window', 100)
        }
        
        self.strategy = AdaptiveHybridStrategy(adaptive_config)
        
        # Monitor
        monitor_config = {
            'metrics_window': config.get('metrics_window', 1000),
            'alert_thresholds': config.get('alert_thresholds', {
                'accuracy': 0.45,
                'drawdown': 0.15,
                'latency': 1000,
                'buffer_overflow': 0.9
            })
        }
        
        self.monitor = AdaptiveMonitor(monitor_config)
        
        # Feature calculators
        self.ml_features = MLFeaturesV3()
        self.tech_indicators = TechnicalIndicators()
        
        # Estado
        self.is_active = False
        self.last_signal_time = datetime.now()
        self.signal_cooldown = config.get('signal_cooldown', 30)  # segundos
        
        # Buffer para candles
        self.candle_buffer = []
        self.candle_timeframe = config.get('candle_timeframe', '5min')
        self.lookback_candles = config.get('lookback_candles', 100)
        
        # Thread de processamento
        self.processing_thread = None
        
    def initialize(self):
        """Inicializa a integração adaptativa"""
        
        self.logger.info("="*80)
        self.logger.info("INICIALIZANDO INTEGRAÇÃO ADAPTATIVA")
        self.logger.info("="*80)
        
        try:
            # 1. Carregar modelos na estratégia
            self.strategy.load_models()
            
            # 2. Iniciar sistema adaptativo
            self.strategy.start_learning()
            
            # 3. Iniciar monitor
            self.monitor.start()
            
            # 4. Conectar aos dados do TradingSystem
            self._connect_to_data_sources()
            
            # 5. Iniciar thread de processamento
            self.is_active = True
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                name="AdaptiveProcessing"
            )
            self.processing_thread.start()
            
            self.logger.info("[OK] Integração adaptativa inicializada")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar: {e}")
            return False
            
    def shutdown(self):
        """Desliga a integração adaptativa"""
        
        self.logger.info("Desligando integração adaptativa...")
        
        self.is_active = False
        
        # Parar componentes
        if self.strategy:
            self.strategy.stop_learning()
            
        if self.monitor:
            self.monitor.stop()
            
        # Aguardar thread
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
            
        self.logger.info("[OK] Integração desligada")
        
    def _connect_to_data_sources(self):
        """Conecta aos dados do TradingSystem"""
        
        # Obter referência à estrutura de dados
        self.data_structure = self.trading_system.data_structure
        
        # Verificar conexão
        if not self.data_structure:
            raise ValueError("TradingSystem não possui data_structure")
            
        self.logger.info("[OK] Conectado às fontes de dados")
        
    def _processing_loop(self):
        """Loop principal de processamento"""
        
        self.logger.info("Loop de processamento adaptativo iniciado")
        
        last_candle_time = None
        
        while self.is_active:
            try:
                # 1. Obter dados recentes
                unified_data = self.data_structure.get_unified_data()
                
                if unified_data is None or unified_data.empty:
                    threading.Event().wait(1)
                    continue
                
                # 2. Verificar se há novo candle
                current_time = unified_data.index[-1]
                
                # Formar candles se necessário
                if self._should_form_candle(current_time, last_candle_time):
                    candles = self._form_candles(unified_data)
                    
                    if candles is not None and len(candles) >= self.lookback_candles:
                        # 3. Processar com estratégia adaptativa
                        start_time = datetime.now()
                        
                        signal_info = self._process_adaptive_signal(candles, unified_data)
                        
                        # Calcular latência
                        latency = (datetime.now() - start_time).total_seconds() * 1000
                        signal_info['latency'] = latency
                        
                        # 4. Registrar no monitor
                        self.monitor.record_prediction(signal_info)
                        
                        # 5. Executar sinal se apropriado
                        if self._should_execute_signal(signal_info):
                            self._execute_signal(signal_info)
                        
                        last_candle_time = current_time
                
                # Pequena pausa
                threading.Event().wait(0.1)
                
            except Exception as e:
                self.logger.error(f"Erro no loop de processamento: {e}")
                threading.Event().wait(1)
                
    def _should_form_candle(self, current_time: pd.Timestamp, 
                          last_time: Optional[pd.Timestamp]) -> bool:
        """Verifica se deve formar novo candle"""
        
        if last_time is None:
            return True
            
        # Verificar intervalo baseado no timeframe
        if self.candle_timeframe == '1min':
            return current_time.minute != last_time.minute
        elif self.candle_timeframe == '5min':
            return current_time.minute // 5 != last_time.minute // 5
        elif self.candle_timeframe == '15min':
            return current_time.minute // 15 != last_time.minute // 15
        else:
            # Default: 5 minutos
            return current_time.minute // 5 != last_time.minute // 5
            
    def _form_candles(self, unified_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Forma candles a partir dos dados unificados"""
        
        try:
            # Usar preço close dos dados unificados
            if 'close' not in unified_data.columns:
                return None
                
            # Resample para formar candles
            candles = unified_data['close'].resample(self.candle_timeframe).agg({
                'open': 'first',
                'high': 'max', 
                'low': 'min',
                'close': 'last'
            })
            
            # Volume
            if 'volume' in unified_data.columns:
                candles['volume'] = unified_data['volume'].resample(
                    self.candle_timeframe
                ).sum()
            else:
                candles['volume'] = 100  # Volume dummy
                
            # Remover NaN
            candles = candles.dropna()
            
            return candles
            
        except Exception as e:
            self.logger.error(f"Erro ao formar candles: {e}")
            return None
            
    def _process_adaptive_signal(self, candles: pd.DataFrame,
                               unified_data: pd.DataFrame) -> Dict:
        """Processa sinal com estratégia adaptativa"""
        
        try:
            # 1. Calcular features tick
            tick_features = self._calculate_tick_features(candles)
            
            # 2. Obter dados de book
            book_data = self._get_book_data(unified_data)
            
            # 3. Processar com estratégia adaptativa
            signal_info = self.strategy.process_market_data(
                tick_features,
                book_data
            )
            
            # 4. Adicionar contexto
            signal_info['timestamp'] = datetime.now()
            signal_info['price'] = candles.iloc[-1]['close']
            
            # Log periódico
            if (datetime.now() - self.last_signal_time).total_seconds() > 30:
                self._log_signal_info(signal_info)
                self.last_signal_time = datetime.now()
                
            return signal_info
            
        except Exception as e:
            self.logger.error(f"Erro ao processar sinal adaptativo: {e}")
            return {
                'signal': 0,
                'confidence': 0,
                'regime': 'undefined',
                'timestamp': datetime.now()
            }
            
    def _calculate_tick_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Calcula features para modelo tick"""
        
        # Calcular indicadores técnicos
        for indicator in ['RSI', 'MACD', 'BB', 'ATR', 'EMA']:
            self.tech_indicators.calculate(candles, indicator)
            
        # Calcular features ML
        features = self.ml_features.calculate_all_features(candles)
        
        # Retornar última linha
        return features.tail(1)
        
    def _get_book_data(self, unified_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Obtém dados de book do unified_data"""
        
        # Por enquanto, retornar None se não há dados de book específicos
        # Em produção, isso seria extraído do book real
        return None
        
    def _should_execute_signal(self, signal_info: Dict) -> bool:
        """Verifica se deve executar o sinal"""
        
        # Verificar cooldown
        if (datetime.now() - self.last_signal_time).total_seconds() < self.signal_cooldown:
            return False
            
        # Verificar se há sinal válido
        if signal_info['signal'] == 0:
            return False
            
        # Verificar confiança mínima
        if signal_info['confidence'] < 0.6:
            return False
            
        # Verificar posição atual
        current_position = self._get_current_position()
        
        # Não abrir nova posição se já temos uma
        if current_position != 0 and signal_info['signal'] == np.sign(current_position):
            return False
            
        return True
        
    def _execute_signal(self, signal_info: Dict):
        """Executa sinal de trading"""
        
        try:
            # Preparar ordem
            order = {
                'action': 'BUY' if signal_info['signal'] == 1 else 'SELL',
                'quantity': self._calculate_position_size(signal_info),
                'price': signal_info['price'],
                'type': 'MARKET',
                'signal_info': signal_info
            }
            
            # Enviar para o TradingSystem
            self.trading_system.execute_order(order)
            
            # Registrar no monitor
            self.monitor.record_trade({
                'action': order['action'],
                'price': order['price'],
                'confidence': signal_info['confidence'],
                'regime': signal_info['regime'],
                'model_type': signal_info.get('model_type', 'current')
            })
            
            self.logger.info(f"[TRADE] {order['action']} executado - "
                           f"Confiança: {signal_info['confidence']:.2%}")
            
        except Exception as e:
            self.logger.error(f"Erro ao executar sinal: {e}")
            
    def _calculate_position_size(self, signal_info: Dict) -> int:
        """Calcula tamanho da posição"""
        
        # Por simplicidade, usar tamanho fixo
        # Em produção, isso seria baseado em Kelly Criterion ou similar
        base_size = self.config.get('base_position_size', 1)
        
        # Ajustar por confiança
        if signal_info['confidence'] > 0.8:
            return base_size * 2
        elif signal_info['confidence'] > 0.7:
            return int(base_size * 1.5)
        else:
            return base_size
            
    def _get_current_position(self) -> int:
        """Obtém posição atual do TradingSystem"""
        
        try:
            # Obter do TradingSystem
            position_info = self.trading_system.get_current_position()
            return position_info.get('quantity', 0)
        except:
            return 0
            
    def _log_signal_info(self, signal_info: Dict):
        """Log informações do sinal"""
        
        self.logger.info(f"\n{'-'*60}")
        self.logger.info(f"Sinal: {signal_info['signal']} | "
                        f"Confiança: {signal_info['confidence']:.2%} | "
                        f"Regime: {signal_info['regime']}")
        
        adaptive_info = signal_info.get('adaptive_info', {})
        if adaptive_info:
            self.logger.info(f"Accuracy recente: {adaptive_info.get('recent_accuracy', 0):.2%} | "
                           f"Modelo: {signal_info.get('model_type', 'current')}")
            
    def update_trade_result(self, trade_result: dict):
        """Atualiza estratégia com resultado do trade"""
        
        # Adicionar informações necessárias
        if 'model_type' not in trade_result:
            trade_result['model_type'] = 'current'
            
        # Atualizar estratégia
        self.strategy.update_trade_result(trade_result)
        
        # Registrar no monitor
        self.monitor.record_trade(trade_result)
        
    def get_status(self) -> Dict:
        """Retorna status da integração adaptativa"""
        
        status = {
            'active': self.is_active,
            'strategy': self.strategy.get_adaptive_metrics() if self.strategy else {},
            'monitor': self.monitor.get_dashboard_data() if self.monitor else {},
            'last_signal': self.last_signal_time.isoformat()
        }
        
        return status
        
    def get_performance_report(self) -> Dict:
        """Gera relatório de performance"""
        
        if self.monitor:
            return self.monitor.generate_report()
        else:
            return {}