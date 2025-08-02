"""
Flow-Aware Feedback System - HMARL Fase 1 Semana 4
Sistema de feedback que considera análise de fluxo
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import deque
import time
import sys
import os

# Adicionar diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from infrastructure.valkey_connection import ValkeyConnectionManager
except ImportError:
    ValkeyConnectionManager = None


class FlowAwareRewardCalculator:
    """Calculadora de rewards considerando análise de fluxo"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RewardCalculator")
        
    def calculate_flow_aware_reward(self, decision: Dict, execution: Dict, 
                                  flow_context: Dict) -> Dict:
        """Calcula reward com componentes de fluxo"""
        # Componente tradicional (P&L)
        pnl = execution.get('pnl', 0)
        traditional_reward = pnl * 10  # Scale factor
        
        # Componente de fluxo
        flow_reward = 0
        
        # Reward por ler corretamente o order flow
        if decision.get('signal_type') == 'flow_based' or 'flow' in decision.get('agent_type', ''):
            flow_accuracy = self._calculate_flow_accuracy(decision, execution, flow_context)
            flow_reward += flow_accuracy * 5
            
        # Reward por timing baseado em footprint
        if 'footprint_pattern' in decision.get('metadata', {}):
            timing_quality = self._evaluate_footprint_timing(decision, execution, flow_context)
            flow_reward += timing_quality * 3
            
        # Penalty por ir contra fluxo forte
        if self._went_against_strong_flow(decision, flow_context):
            flow_reward -= 5
            
        # Bonus por confirmar com múltiplos sinais de fluxo
        if self._multiple_flow_confirmations(decision):
            flow_reward += 2
            
        # Reward total
        total_reward = traditional_reward + flow_reward
        
        return {
            'total': total_reward,
            'traditional_component': traditional_reward,
            'flow_component': flow_reward,
            'breakdown': {
                'pnl': pnl,
                'flow_accuracy': flow_accuracy if 'flow_accuracy' in locals() else 0,
                'timing_quality': timing_quality if 'timing_quality' in locals() else 0,
                'flow_alignment': not self._went_against_strong_flow(decision, flow_context)
            }
        }
        
    def _calculate_flow_accuracy(self, decision: Dict, execution: Dict, 
                               flow_context: Dict) -> float:
        """Calcula precisão da leitura do fluxo"""
        # Verificar se a direção predita pelo fluxo se confirmou
        predicted_direction = decision.get('signal', {}).get('action', 'hold')
        actual_movement = execution.get('price_movement', 0)
        
        accuracy = 0.0
        
        if predicted_direction == 'buy' and actual_movement > 0:
            accuracy = min(actual_movement * 10, 1.0)  # Escala baseada no movimento
        elif predicted_direction == 'sell' and actual_movement < 0:
            accuracy = min(abs(actual_movement) * 10, 1.0)
        elif predicted_direction == 'hold' and abs(actual_movement) < 0.001:
            accuracy = 0.5
            
        # Ajustar pela confiança original
        original_confidence = decision.get('signal', {}).get('confidence', 0.5)
        accuracy *= original_confidence
        
        return accuracy
        
    def _evaluate_footprint_timing(self, decision: Dict, execution: Dict, 
                                 flow_context: Dict) -> float:
        """Avalia qualidade do timing baseado em footprint"""
        # Verificar se entrou no momento certo segundo o footprint
        pattern = decision.get('metadata', {}).get('footprint_pattern', '')
        
        if not pattern:
            return 0.0
            
        timing_score = 0.5  # Base
        
        # Se era padrão de reversão e reverteu
        if 'reversal' in pattern.lower():
            if execution.get('captured_reversal', False):
                timing_score = 1.0
            else:
                timing_score = 0.2
                
        # Se era continuação e continuou
        elif 'continuation' in pattern.lower():
            if execution.get('trend_continued', False):
                timing_score = 0.8
                
        # Ajustar pelo slippage
        slippage = execution.get('slippage', 0)
        if slippage < 0.0001:  # Slippage mínimo
            timing_score *= 1.2
        elif slippage > 0.001:  # Slippage alto
            timing_score *= 0.8
            
        return min(timing_score, 1.0)
        
    def _went_against_strong_flow(self, decision: Dict, flow_context: Dict) -> bool:
        """Verifica se foi contra fluxo forte"""
        flow_direction = flow_context.get('dominant_flow_direction', 'neutral')
        flow_strength = flow_context.get('flow_strength', 0)
        decision_direction = decision.get('signal', {}).get('action', 'hold')
        
        # Se o fluxo era forte (> 0.7) e foi na direção oposta
        if flow_strength > 0.7:
            if (flow_direction == 'bullish' and decision_direction == 'sell') or \
               (flow_direction == 'bearish' and decision_direction == 'buy'):
                return True
                
        return False
        
    def _multiple_flow_confirmations(self, decision: Dict) -> bool:
        """Verifica se teve múltiplas confirmações de fluxo"""
        metadata = decision.get('metadata', {})
        confirmations = 0
        
        # Contar diferentes tipos de confirmação
        if metadata.get('ofi_signal'):
            confirmations += 1
        if metadata.get('delta_signal'):
            confirmations += 1
        if metadata.get('footprint_pattern'):
            confirmations += 1
        if metadata.get('tape_pattern'):
            confirmations += 1
            
        return confirmations >= 2


class FlowPerformanceAnalyzer:
    """Analisador de performance com foco em fluxo"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PerformanceAnalyzer")
        self.performance_history = deque(maxlen=1000)
        
    def analyze_flow_decision_performance(self, decision: Dict, execution: Dict, 
                                        flow_context: Dict) -> Dict:
        """Analisa performance de decisão baseada em fluxo"""
        analysis = {
            'decision_quality': 0.0,
            'flow_reading_accuracy': 0.0,
            'timing_precision': 0.0,
            'risk_management': 0.0,
            'insights': []
        }
        
        # Qualidade da decisão
        if execution.get('profitable', False):
            analysis['decision_quality'] = 0.7 + (execution.get('return', 0) * 3)
        else:
            analysis['decision_quality'] = 0.3 - abs(execution.get('return', 0))
            
        # Precisão na leitura do fluxo
        flow_signals = decision.get('metadata', {}).get('flow_signals', {})
        if flow_signals:
            correct_signals = sum(1 for s in flow_signals.values() if s.get('confirmed', False))
            total_signals = len(flow_signals)
            analysis['flow_reading_accuracy'] = correct_signals / total_signals if total_signals > 0 else 0
            
        # Precisão do timing
        entry_quality = self._evaluate_entry_quality(execution)
        exit_quality = self._evaluate_exit_quality(execution)
        analysis['timing_precision'] = (entry_quality + exit_quality) / 2
        
        # Gestão de risco
        analysis['risk_management'] = self._evaluate_risk_management(decision, execution)
        
        # Gerar insights
        analysis['insights'] = self._generate_insights(decision, execution, flow_context, analysis)
        
        # Armazenar no histórico
        self.performance_history.append({
            'timestamp': time.time(),
            'decision': decision,
            'execution': execution,
            'analysis': analysis
        })
        
        return analysis
        
    def _evaluate_entry_quality(self, execution: Dict) -> float:
        """Avalia qualidade da entrada"""
        slippage = execution.get('entry_slippage', 0)
        
        if slippage < 0.0001:
            return 1.0
        elif slippage < 0.0005:
            return 0.8
        elif slippage < 0.001:
            return 0.6
        else:
            return 0.4
            
    def _evaluate_exit_quality(self, execution: Dict) -> float:
        """Avalia qualidade da saída"""
        # Verificar se saiu no alvo ou no stop
        exit_type = execution.get('exit_type', 'unknown')
        
        if exit_type == 'target':
            return 1.0
        elif exit_type == 'trailing_stop':
            return 0.8
        elif exit_type == 'stop_loss':
            return 0.4
        elif exit_type == 'time_stop':
            return 0.6
        else:
            return 0.5
            
    def _evaluate_risk_management(self, decision: Dict, execution: Dict) -> float:
        """Avalia gestão de risco"""
        score = 0.5
        
        # Verificar se respeitou tamanho de posição
        planned_size = decision.get('position_size', 0)
        actual_size = execution.get('actual_size', 0)
        
        if planned_size > 0:
            size_accuracy = 1 - abs(planned_size - actual_size) / planned_size
            score = size_accuracy * 0.5
            
        # Verificar se respeitou stop loss
        if execution.get('stop_triggered', False):
            max_loss = decision.get('max_loss', 0.02)
            actual_loss = abs(execution.get('return', 0))
            
            if actual_loss <= max_loss * 1.1:  # 10% de tolerância
                score += 0.3
            else:
                score -= 0.2
                
        # Bonus por não ter risco excessivo
        if execution.get('max_drawdown', 0) < 0.01:
            score += 0.2
            
        return max(0, min(1, score))
        
    def _generate_insights(self, decision: Dict, execution: Dict, 
                          flow_context: Dict, analysis: Dict) -> List[str]:
        """Gera insights sobre a performance"""
        insights = []
        
        # Insight sobre leitura de fluxo
        if analysis['flow_reading_accuracy'] > 0.8:
            insights.append("Excelente leitura do fluxo de ordens")
        elif analysis['flow_reading_accuracy'] < 0.3:
            insights.append("Leitura de fluxo precisa melhorar")
            
        # Insight sobre timing
        if analysis['timing_precision'] > 0.8:
            insights.append("Timing de entrada/saída preciso")
        elif analysis['timing_precision'] < 0.4:
            insights.append("Melhorar timing de execução")
            
        # Insight sobre direção do mercado
        if self._went_with_dominant_flow(decision, flow_context) and execution.get('profitable', False):
            insights.append("Sucesso ao seguir fluxo dominante")
        elif not self._went_with_dominant_flow(decision, flow_context) and not execution.get('profitable', False):
            insights.append("Evitar ir contra fluxo forte")
            
        # Insight sobre padrões
        pattern = decision.get('metadata', {}).get('pattern')
        if pattern and execution.get('profitable', False):
            insights.append(f"Padrão {pattern} funcionou bem")
            
        return insights
        
    def _went_with_dominant_flow(self, decision: Dict, flow_context: Dict) -> bool:
        """Verifica se foi com o fluxo dominante"""
        flow_direction = flow_context.get('dominant_flow_direction', 'neutral')
        decision_direction = decision.get('signal', {}).get('action', 'hold')
        
        return (flow_direction == 'bullish' and decision_direction == 'buy') or \
               (flow_direction == 'bearish' and decision_direction == 'sell')


class FlowAwareFeedbackSystem:
    """Sistema de feedback que considera análise de fluxo"""
    
    def __init__(self, valkey_config: Optional[Dict] = None):
        self.reward_calculator = FlowAwareRewardCalculator()
        self.performance_analyzer = FlowPerformanceAnalyzer()
        self.logger = logging.getLogger(f"{__name__}.FeedbackSystem")
        
        # Cache de decisões para correlação
        self.decision_cache = {}
        
        # Configurar Valkey
        self.valkey = None
        if ValkeyConnectionManager and valkey_config:
            self._setup_valkey(valkey_config)
        elif valkey_config:
            self.logger.warning("ValkeyConnectionManager não disponível - usando cache local")
            
    def _setup_valkey(self, config: Dict):
        """Configura conexão com Valkey"""
        try:
            self.valkey = ValkeyConnectionManager(
                host=config.get('host', 'localhost'),
                port=config.get('port', 6379),
                db=config.get('db', 0),
                password=config.get('password')
            )
            
            if self.valkey.connect():
                self.logger.info("Conectado ao Valkey para persistência de feedback")
            else:
                self.logger.error("Falha ao conectar com Valkey")
                self.valkey = None
                
        except Exception as e:
            self.logger.error(f"Erro configurando Valkey: {e}")
            self.valkey = None
        
    def process_execution_feedback_with_flow(self, execution_data: Dict) -> Dict:
        """Processa feedback considerando contexto de fluxo"""
        try:
            # Buscar decisão original
            decision = self.find_decision(execution_data['decision_id'])
            if not decision:
                self.logger.warning(f"Decisão não encontrada: {execution_data['decision_id']}")
                return None
                
            # Buscar contexto de fluxo no momento da decisão
            flow_context = self.get_flow_context(
                decision['symbol'],
                decision['timestamp']
            )
            
            # Calcular reward considerando fluxo
            reward = self.reward_calculator.calculate_flow_aware_reward(
                decision=decision,
                execution=execution_data,
                flow_context=flow_context
            )
            
            # Analisar performance
            performance_analysis = self.performance_analyzer.analyze_flow_decision_performance(
                decision=decision,
                execution=execution_data,
                flow_context=flow_context
            )
            
            # Analisar se o fluxo confirmou a direção
            flow_confirmation = self.analyze_flow_confirmation(
                decision, execution_data, flow_context
            )
            
            # Feedback enriquecido
            enhanced_feedback = {
                'agent_id': decision['agent_id'],
                'decision_id': execution_data['decision_id'],
                'reward': reward['total'],
                'flow_reward_component': reward['flow_component'],
                'traditional_reward_component': reward['traditional_component'],
                'flow_confirmation': flow_confirmation,
                'performance_analysis': performance_analysis,
                'execution_details': execution_data,
                'learning_insights': self._generate_learning_insights(
                    decision, execution_data, flow_context, performance_analysis
                )
            }
            
            # Publicar feedback se valkey disponível
            if self.valkey and self.valkey.is_connected():
                self._publish_feedback(enhanced_feedback)
                
            self.logger.info(f"Feedback processado: reward={reward['total']:.2f}, "
                           f"flow_component={reward['flow_component']:.2f}")
                           
            return enhanced_feedback
            
        except Exception as e:
            self.logger.error(f"Erro processando feedback: {e}")
            return None
            
    def find_decision(self, decision_id: str) -> Optional[Dict]:
        """Busca decisão original"""
        # Primeiro verificar cache
        if decision_id in self.decision_cache:
            return self.decision_cache[decision_id]
            
        # Se tiver valkey, buscar lá
        if self.valkey and self.valkey.is_connected():
            decision = self.valkey.get_decision(decision_id)
            if decision:
                # Adicionar ao cache local
                self.decision_cache[decision_id] = decision
                return decision
            
        return None
        
    def cache_decision(self, decision: Dict):
        """Armazena decisão no cache"""
        decision_id = decision.get('decision_id')
        if decision_id:
            # Cache local
            self.decision_cache[decision_id] = decision
            
            # Persistir no Valkey se disponível
            if self.valkey and self.valkey.is_connected():
                if self.valkey.store_decision(decision):
                    self.logger.debug(f"Decisão {decision_id} persistida no Valkey")
            
            # Limpar cache antigo (manter últimas 1000 decisões)
            if len(self.decision_cache) > 1000:
                oldest_key = next(iter(self.decision_cache))
                del self.decision_cache[oldest_key]
                
    def get_flow_context(self, symbol: str, timestamp: datetime) -> Dict:
        """Recupera contexto de fluxo em momento específico"""
        flow_context = {
            'dominant_flow_direction': 'neutral',
            'flow_strength': 0.0,
            'ofi': 0.0,
            'delta': 0,
            'aggression': 0.0,
            'volume_profile': {}
        }
        
        # Se tiver valkey, buscar dados históricos
        if self.valkey and self.valkey.is_connected():
            # Converter datetime para timestamp
            ts = timestamp.timestamp() if isinstance(timestamp, datetime) else timestamp
            
            stored_context = self.valkey.get_flow_state(symbol, ts)
            if stored_context:
                flow_context.update(stored_context)
                return flow_context
        
        # Simulação para desenvolvimento se não houver dados
        flow_context['dominant_flow_direction'] = np.random.choice(['bullish', 'bearish', 'neutral'])
        flow_context['flow_strength'] = np.random.uniform(0, 1)
        flow_context['ofi'] = np.random.uniform(-1, 1)
        flow_context['delta'] = np.random.randint(-500, 500)
        
        return flow_context
        
    def analyze_flow_confirmation(self, decision: Dict, execution_data: Dict, 
                                flow_context: Dict) -> Dict:
        """Analisa se o fluxo confirmou a direção"""
        confirmation = {
            'confirmed': False,
            'strength': 0.0,
            'accuracy': 0.0,
            'details': {}
        }
        
        # Verificar se a direção do trade foi confirmada pelo fluxo subsequente
        decision_direction = decision.get('signal', {}).get('action', 'hold')
        price_movement = execution_data.get('price_movement', 0)
        
        # Confirmação básica: movimento de preço na direção esperada
        if (decision_direction == 'buy' and price_movement > 0) or \
           (decision_direction == 'sell' and price_movement < 0):
            confirmation['confirmed'] = True
            confirmation['strength'] = min(abs(price_movement) * 10, 1.0)
            
        # Calcular accuracy baseado em múltiplos fatores
        factors = []
        
        # Fator 1: OFI confirmou
        if decision_direction == 'buy' and flow_context.get('ofi', 0) > 0.3:
            factors.append(1.0)
        elif decision_direction == 'sell' and flow_context.get('ofi', 0) < -0.3:
            factors.append(1.0)
        else:
            factors.append(0.0)
            
        # Fator 2: Delta confirmou
        if decision_direction == 'buy' and flow_context.get('delta', 0) > 100:
            factors.append(1.0)
        elif decision_direction == 'sell' and flow_context.get('delta', 0) < -100:
            factors.append(1.0)
        else:
            factors.append(0.0)
            
        confirmation['accuracy'] = np.mean(factors) if factors else 0.0
        
        return confirmation
        
    def _generate_learning_insights(self, decision: Dict, execution_data: Dict,
                                  flow_context: Dict, performance_analysis: Dict) -> Dict:
        """Gera insights para aprendizado"""
        insights = {
            'key_lessons': [],
            'pattern_effectiveness': {},
            'suggested_improvements': [],
            'confidence_calibration': 0.0
        }
        
        # Lições sobre padrões
        if 'pattern' in decision.get('metadata', {}):
            pattern = decision['metadata']['pattern']
            if execution_data.get('profitable', False):
                insights['pattern_effectiveness'][pattern] = 'effective'
                insights['key_lessons'].append(f"Padrão {pattern} funcionou neste contexto")
            else:
                insights['pattern_effectiveness'][pattern] = 'ineffective'
                insights['key_lessons'].append(f"Padrão {pattern} falhou - revisar condições")
                
        # Calibração de confiança
        original_confidence = decision.get('signal', {}).get('confidence', 0.5)
        if execution_data.get('profitable', False):
            if original_confidence < 0.5:
                insights['confidence_calibration'] = 0.2  # Aumentar confiança
                insights['suggested_improvements'].append("Aumentar confiança em sinais similares")
        else:
            if original_confidence > 0.7:
                insights['confidence_calibration'] = -0.2  # Reduzir confiança
                insights['suggested_improvements'].append("Reduzir confiança ou adicionar filtros")
                
        # Sugestões baseadas em performance
        if performance_analysis['flow_reading_accuracy'] < 0.5:
            insights['suggested_improvements'].append("Melhorar interpretação de sinais de fluxo")
            
        if performance_analysis['timing_precision'] < 0.5:
            insights['suggested_improvements'].append("Trabalhar timing de entrada/saída")
            
        return insights
        
    def _publish_feedback(self, feedback: Dict):
        """Publica feedback via valkey"""
        if self.valkey and self.valkey.is_connected():
            try:
                # Armazenar feedback
                if self.valkey.store_feedback(feedback):
                    self.logger.debug(f"Feedback para {feedback['decision_id']} persistido")
                    
                # Publicar em canal pub/sub para notificação em tempo real
                channel = f"hmarl:feedback:{feedback['agent_id']}"
                self.valkey.publish_signal(channel, feedback)
                
            except Exception as e:
                self.logger.error(f"Erro publicando feedback: {e}")
                
    def store_flow_context(self, symbol: str, flow_state: Dict):
        """Armazena contexto de fluxo para referência futura"""
        if self.valkey and self.valkey.is_connected():
            try:
                if self.valkey.store_flow_state(symbol, flow_state):
                    self.logger.debug(f"Flow state para {symbol} armazenado")
            except Exception as e:
                self.logger.error(f"Erro armazenando flow state: {e}")
                
    def get_agent_performance_history(self, agent_id: str) -> Optional[Dict]:
        """Recupera histórico de performance do agente"""
        if self.valkey and self.valkey.is_connected():
            try:
                return self.valkey.get_agent_performance(agent_id)
            except Exception as e:
                self.logger.error(f"Erro recuperando performance: {e}")
                
        # Fallback para dados locais
        if hasattr(self.performance_analyzer, 'performance_history'):
            for entry in self.performance_analyzer.performance_history:
                if entry.get('decision', {}).get('agent_id') == agent_id:
                    return entry.get('analysis', {})
                    
        return None
        
    def get_feedback_statistics(self) -> Dict:
        """Retorna estatísticas do sistema de feedback"""
        stats = {
            'cached_decisions': len(self.decision_cache),
            'valkey_connected': self.valkey.is_connected() if self.valkey else False
        }
        
        if self.valkey and self.valkey.is_connected():
            health = self.valkey.health_check()
            stats.update({
                'valkey_health': health,
                'stored_decisions': health.get('keyspace', {}).get('decision', 0),
                'stored_feedback': health.get('keyspace', {}).get('feedback', 0),
                'stored_patterns': health.get('keyspace', {}).get('pattern', 0)
            })
            
        return stats
        
    def cleanup_old_data(self, days: int = 30):
        """Limpa dados antigos do Valkey"""
        if self.valkey and self.valkey.is_connected():
            try:
                deleted = self.valkey.cleanup_old_data(days)
                self.logger.info(f"Limpeza concluída: {deleted} registros removidos")
                return deleted
            except Exception as e:
                self.logger.error(f"Erro na limpeza: {e}")
                
        return 0


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Criar sistema de feedback
    feedback_system = FlowAwareFeedbackSystem()
    
    # Simular uma decisão
    decision = {
        'decision_id': 'dec_12345',
        'agent_id': 'flow_agent_001',
        'symbol': 'WDOH25',
        'timestamp': datetime.now(),
        'signal': {
            'action': 'buy',
            'confidence': 0.75
        },
        'metadata': {
            'pattern': 'p_reversal',
            'footprint_pattern': 'absorption',
            'ofi_signal': True,
            'delta_signal': True
        },
        'position_size': 2,
        'max_loss': 0.02
    }
    
    # Cachear decisão
    feedback_system.cache_decision(decision)
    
    # Simular execução
    execution = {
        'decision_id': 'dec_12345',
        'pnl': 0.015,  # 1.5% de lucro
        'price_movement': 0.012,
        'profitable': True,
        'return': 0.015,
        'entry_slippage': 0.0002,
        'exit_type': 'target',
        'actual_size': 2,
        'max_drawdown': 0.005,
        'captured_reversal': True
    }
    
    # Processar feedback
    feedback = feedback_system.process_execution_feedback_with_flow(execution)
    
    if feedback:
        print(f"\nFeedback processado:")
        print(f"Reward total: {feedback['reward']:.2f}")
        print(f"Componente tradicional: {feedback['traditional_reward_component']:.2f}")
        print(f"Componente de fluxo: {feedback['flow_reward_component']:.2f}")
        print(f"Flow confirmado: {feedback['flow_confirmation']['confirmed']}")
        print(f"Insights: {feedback['learning_insights']['key_lessons']}")