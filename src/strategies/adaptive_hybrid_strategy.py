"""
Adaptive Hybrid Strategy com Online Learning
Estratégia que se adapta em tempo real com aprendizado contínuo
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
import logging
import threading
from pathlib import Path

from .hybrid_strategy import HybridStrategy
from ..training.online_learning_system import OnlineLearningSystem

class AdaptiveHybridStrategy(HybridStrategy):
    """
    Estratégia híbrida adaptativa que:
    1. Executa trades com modelos atuais
    2. Aprende continuamente com novos dados
    3. Adapta-se automaticamente a mudanças de mercado
    4. Faz A/B testing entre modelos
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Sistema de aprendizado contínuo
        online_config = {
            'buffer_size': config.get('online_buffer_size', 50000),
            'retrain_interval': config.get('retrain_interval', 3600),  # 1 hora
            'min_samples_retrain': config.get('min_samples_retrain', 5000),
            'validation_window': config.get('validation_window', 500),
            'performance_threshold': config.get('performance_threshold', 0.55)
        }
        
        self.online_learning = OnlineLearningSystem(online_config)
        
        # A/B Testing
        self.ab_testing_enabled = config.get('ab_testing_enabled', True)
        self.ab_test_ratio = config.get('ab_test_ratio', 0.2)  # 20% para novos modelos
        self.ab_test_results = {
            'current': {'trades': 0, 'wins': 0, 'pnl': 0},
            'candidate': {'trades': 0, 'wins': 0, 'pnl': 0}
        }
        
        # Adaptive parameters
        self.adaptive_regime_threshold = self.regime_threshold
        self.adaptive_confidence_threshold = 0.55
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        
        # Performance tracking
        self.recent_performance = []
        self.performance_window = config.get('performance_window', 100)
        
        # Estado
        self.is_learning = False
        self.last_data_update = datetime.now()
        self.total_predictions = 0
        self.correct_predictions = 0
        
    def start_learning(self):
        """Inicia o sistema de aprendizado contínuo"""
        
        self.logger.info("="*80)
        self.logger.info("INICIANDO ESTRATÉGIA ADAPTATIVA")
        self.logger.info("="*80)
        
        # Iniciar online learning
        self.online_learning.start()
        self.is_learning = True
        
        # Thread para monitorar performance
        self.monitor_thread = threading.Thread(
            target=self._performance_monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("[OK] Sistema adaptativo iniciado")
        
    def stop_learning(self):
        """Para o sistema de aprendizado contínuo"""
        
        self.is_learning = False
        self.online_learning.stop()
        self.logger.info("[OK] Sistema adaptativo parado")
        
    def process_market_data(self, tick_data: pd.DataFrame, 
                          book_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Processa dados de mercado e gera sinais adaptativos
        
        Args:
            tick_data: Dados tick recentes
            book_data: Dados de book (opcional)
            
        Returns:
            Sinal de trading com informações adaptativas
        """
        
        # 1. Adicionar dados ao sistema de aprendizado
        if self.is_learning:
            self.online_learning.add_tick_data(tick_data)
            if book_data is not None:
                self.online_learning.add_book_data(book_data)
            self.last_data_update = datetime.now()
        
        # 2. Preparar features
        tick_features = self._prepare_features(tick_data, 'tick')
        book_features = self._prepare_features(book_data, 'book') if book_data else None
        
        # 3. Decidir qual modelo usar (A/B testing)
        use_candidate = self._should_use_candidate_model()
        
        # 4. Obter sinal adaptativo
        if use_candidate:
            signal_info = self._get_candidate_signal(tick_features, book_features)
            model_type = 'candidate'
        else:
            signal_info = self.get_hybrid_signal(tick_features, book_features)
            model_type = 'current'
        
        # 5. Aplicar adaptações
        signal_info = self._apply_adaptations(signal_info)
        
        # 6. Adicionar informações de tracking
        signal_info['model_type'] = model_type
        signal_info['adaptive_info'] = {
            'is_learning': self.is_learning,
            'regime_threshold': self.adaptive_regime_threshold,
            'confidence_threshold': self.adaptive_confidence_threshold,
            'recent_accuracy': self._calculate_recent_accuracy(),
            'model_versions': self.online_learning.model_versions.copy()
        }
        
        # 7. Registrar predição para validação futura
        self._record_prediction(signal_info)
        
        return signal_info
        
    def update_trade_result(self, trade_info: dict):
        """
        Atualiza sistema com resultado do trade
        
        Args:
            trade_info: Informações do trade executado
        """
        
        # Atualizar online learning
        if self.is_learning:
            self.online_learning.add_trade_result(trade_info)
        
        # Atualizar A/B testing
        model_type = trade_info.get('model_type', 'current')
        pnl = trade_info.get('pnl', 0)
        
        self.ab_test_results[model_type]['trades'] += 1
        if pnl > 0:
            self.ab_test_results[model_type]['wins'] += 1
        self.ab_test_results[model_type]['pnl'] += pnl
        
        # Atualizar performance recente
        self.recent_performance.append({
            'timestamp': datetime.now(),
            'pnl': pnl,
            'signal': trade_info.get('signal'),
            'confidence': trade_info.get('confidence')
        })
        
        # Manter janela de performance
        if len(self.recent_performance) > self.performance_window:
            self.recent_performance = self.recent_performance[-self.performance_window:]
        
        # Adaptar parâmetros se necessário
        self._adapt_parameters()
        
    def _prepare_features(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Prepara features para predição"""
        
        if data is None or data.empty:
            return self._create_dummy_features(data_type)
        
        # Aqui você implementaria a preparação real de features
        # Por simplicidade, retornando as últimas linhas
        return data.tail(1)
        
    def _create_dummy_features(self, data_type: str) -> pd.DataFrame:
        """Cria features dummy quando não há dados"""
        
        if data_type == 'tick':
            # Features tick esperadas
            features = {
                'returns_1': 0.0,
                'returns_5': 0.0,
                'returns_10': 0.0,
                'volume_ma_10': 1.0,
                'hour': datetime.now().hour,
                'minute': datetime.now().minute
            }
        else:  # book
            features = {
                'position': 10.0,
                'is_top_5': 0.0,
                'ofi': 0.0,
                'quantity_log': 0.0
            }
        
        return pd.DataFrame([features])
        
    def _should_use_candidate_model(self) -> bool:
        """Decide se deve usar modelo candidato (A/B testing)"""
        
        if not self.ab_testing_enabled:
            return False
        
        # Verificar se há modelos candidatos
        candidate_models = self.online_learning.candidate_models
        if not any(candidate_models.values()):
            return False
        
        # Decisão probabilística
        return np.random.random() < self.ab_test_ratio
        
    def _get_candidate_signal(self, tick_features: pd.DataFrame,
                            book_features: Optional[pd.DataFrame]) -> Dict:
        """Obtém sinal usando modelos candidatos"""
        
        # Temporariamente substituir modelos
        original_models = {
            'tick': self.tick_model,
            'book': self.book_model
        }
        
        candidate_models = self.online_learning.candidate_models
        
        try:
            # Usar modelos candidatos
            if candidate_models['tick']:
                self.tick_model = candidate_models['tick']
            if candidate_models['book']:
                self.book_model = candidate_models['book']
            
            # Obter sinal
            signal_info = self.get_hybrid_signal(tick_features, book_features)
            
        finally:
            # Restaurar modelos originais
            self.tick_model = original_models['tick']
            self.book_model = original_models['book']
        
        return signal_info
        
    def _apply_adaptations(self, signal_info: Dict) -> Dict:
        """Aplica adaptações baseadas em performance recente"""
        
        # 1. Ajustar confiança baseado em performance
        recent_accuracy = self._calculate_recent_accuracy()
        
        if recent_accuracy < 0.45:  # Performance ruim
            # Aumentar threshold de confiança
            signal_info['confidence'] *= 0.8
            
        elif recent_accuracy > 0.60:  # Performance boa
            # Pode ser mais agressivo
            signal_info['confidence'] *= 1.1
        
        # 2. Filtrar por threshold adaptativo
        if signal_info['confidence'] < self.adaptive_confidence_threshold:
            signal_info['signal'] = 0  # HOLD
        
        # 3. Ajustar por volatilidade de mercado
        market_volatility = self._estimate_market_volatility()
        if market_volatility > 0.02:  # Alta volatilidade
            # Ser mais conservador
            if abs(signal_info['signal']) == 1:
                signal_info['confidence'] *= 0.9
        
        # Limitar confiança
        signal_info['confidence'] = min(signal_info['confidence'], 1.0)
        
        return signal_info
        
    def _adapt_parameters(self):
        """Adapta parâmetros baseado em performance"""
        
        if len(self.recent_performance) < 20:
            return
        
        # Calcular métricas recentes
        recent_trades = self.recent_performance[-20:]
        win_rate = sum(1 for t in recent_trades if t['pnl'] > 0) / len(recent_trades)
        avg_confidence = np.mean([t['confidence'] for t in recent_trades])
        
        # Adaptar threshold de regime
        if win_rate < 0.4:
            # Performance ruim - ser mais conservador
            self.adaptive_regime_threshold = min(
                self.adaptive_regime_threshold + self.adaptation_rate * 0.05,
                0.8
            )
        elif win_rate > 0.6:
            # Performance boa - pode relaxar
            self.adaptive_regime_threshold = max(
                self.adaptive_regime_threshold - self.adaptation_rate * 0.05,
                0.5
            )
        
        # Adaptar threshold de confiança
        if win_rate < 0.45 and avg_confidence > 0.6:
            # Alta confiança mas baixo win rate - aumentar threshold
            self.adaptive_confidence_threshold = min(
                self.adaptive_confidence_threshold + self.adaptation_rate * 0.02,
                0.7
            )
        elif win_rate > 0.55 and avg_confidence < 0.5:
            # Baixa confiança mas bom win rate - diminuir threshold
            self.adaptive_confidence_threshold = max(
                self.adaptive_confidence_threshold - self.adaptation_rate * 0.02,
                0.45
            )
        
        self.logger.debug(f"Parâmetros adaptados - Regime: {self.adaptive_regime_threshold:.2f}, "
                         f"Confiança: {self.adaptive_confidence_threshold:.2f}")
        
    def _calculate_recent_accuracy(self) -> float:
        """Calcula accuracy recente"""
        
        if not self.recent_performance:
            return 0.5
        
        recent = self.recent_performance[-50:]
        wins = sum(1 for t in recent if t['pnl'] > 0)
        
        return wins / len(recent) if recent else 0.5
        
    def _estimate_market_volatility(self) -> float:
        """Estima volatilidade atual do mercado"""
        
        # Simplificado - em produção seria mais sofisticado
        if not self.recent_performance:
            return 0.01
        
        recent_pnls = [abs(t['pnl']) for t in self.recent_performance[-20:]]
        
        return np.std(recent_pnls) if recent_pnls else 0.01
        
    def _record_prediction(self, signal_info: Dict):
        """Registra predição para validação futura"""
        
        self.total_predictions += 1
        
        # Aqui você salvaria a predição para comparar com resultado real depois
        # Por simplicidade, apenas incrementando contador
        
    def _performance_monitor_loop(self):
        """Loop de monitoramento de performance"""
        
        while self.is_learning:
            try:
                # Verificar performance do A/B testing
                if self.ab_test_results['candidate']['trades'] >= 50:
                    self._evaluate_ab_test()
                
                # Log status
                if self.total_predictions % 100 == 0:
                    self._log_adaptive_status()
                
                # Aguardar
                threading.Event().wait(60)  # 1 minuto
                
            except Exception as e:
                self.logger.error(f"Erro no monitor de performance: {e}")
                
    def _evaluate_ab_test(self):
        """Avalia resultados do A/B testing"""
        
        current = self.ab_test_results['current']
        candidate = self.ab_test_results['candidate']
        
        if current['trades'] == 0 or candidate['trades'] == 0:
            return
        
        # Win rates
        current_wr = current['wins'] / current['trades']
        candidate_wr = candidate['wins'] / candidate['trades']
        
        # Profit per trade
        current_ppt = current['pnl'] / current['trades']
        candidate_ppt = candidate['pnl'] / candidate['trades']
        
        self.logger.info("="*60)
        self.logger.info("RESULTADOS A/B TESTING")
        self.logger.info("="*60)
        self.logger.info(f"Current - WR: {current_wr:.2%}, PPT: ${current_ppt:.2f}")
        self.logger.info(f"Candidate - WR: {candidate_wr:.2%}, PPT: ${candidate_ppt:.2f}")
        
        # Decidir se promove candidato
        if candidate_wr > current_wr * 1.05 and candidate_ppt > current_ppt:
            self.logger.info("[DECISÃO] Promovendo modelos candidatos!")
            self.online_learning._replace_models()
            
            # Reset A/B test
            self.ab_test_results = {
                'current': {'trades': 0, 'wins': 0, 'pnl': 0},
                'candidate': {'trades': 0, 'wins': 0, 'pnl': 0}
            }
        
    def _log_adaptive_status(self):
        """Loga status do sistema adaptativo"""
        
        status = self.online_learning.get_status()
        
        self.logger.info("\n" + "="*60)
        self.logger.info("STATUS DO SISTEMA ADAPTATIVO")
        self.logger.info("="*60)
        self.logger.info(f"Predições totais: {self.total_predictions}")
        self.logger.info(f"Accuracy recente: {self._calculate_recent_accuracy():.2%}")
        self.logger.info(f"Regime threshold: {self.adaptive_regime_threshold:.2f}")
        self.logger.info(f"Confidence threshold: {self.adaptive_confidence_threshold:.2f}")
        self.logger.info(f"Buffer sizes - Tick: {status['buffer_sizes']['tick']}, "
                        f"Book: {status['buffer_sizes']['book']}")
        self.logger.info(f"Model versions - Tick: v{status['model_versions']['tick']}, "
                        f"Book: v{status['model_versions']['book']}")
        
    def get_adaptive_metrics(self) -> Dict:
        """Retorna métricas do sistema adaptativo"""
        
        return {
            'is_learning': self.is_learning,
            'total_predictions': self.total_predictions,
            'recent_accuracy': self._calculate_recent_accuracy(),
            'adaptive_thresholds': {
                'regime': self.adaptive_regime_threshold,
                'confidence': self.adaptive_confidence_threshold
            },
            'ab_test_results': self.ab_test_results.copy(),
            'online_learning_status': self.online_learning.get_status(),
            'last_data_update': self.last_data_update.isoformat()
        }