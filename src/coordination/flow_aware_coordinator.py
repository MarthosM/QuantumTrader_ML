"""
Flow-Aware Coordinator - HMARL Fase 2 Semana 6
Coordenador que considera análise de fluxo na tomada de decisão
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from datetime import datetime
import zmq
import orjson
import sys
import os

# Adicionar diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from infrastructure.valkey_connection import ValkeyConnectionManager
except ImportError:
    ValkeyConnectionManager = None


class SignalQualityScorer:
    """Avalia qualidade dos sinais considerando contexto de fluxo"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SignalQualityScorer")
        self.performance_history = defaultdict(lambda: {'successes': 0, 'total': 0})
        
    def score(self, signal: Dict, flow_consensus: Dict, market_state: Dict) -> float:
        """Pontua qualidade do sinal"""
        score = 0.0
        weights = {
            'confidence': 0.25,
            'flow_alignment': 0.30,
            'historical_performance': 0.20,
            'market_conditions': 0.15,
            'signal_clarity': 0.10
        }
        
        # 1. Confiança base do sinal
        confidence_score = signal.get('confidence', 0.5)
        score += confidence_score * weights['confidence']
        
        # 2. Alinhamento com consenso de fluxo
        flow_alignment_score = self._calculate_flow_alignment(signal, flow_consensus)
        score += flow_alignment_score * weights['flow_alignment']
        
        # 3. Performance histórica do agente
        agent_id = signal.get('agent_id', 'unknown')
        historical_score = self._get_historical_performance(agent_id)
        score += historical_score * weights['historical_performance']
        
        # 4. Condições de mercado
        market_score = self._evaluate_market_conditions(signal, market_state)
        score += market_score * weights['market_conditions']
        
        # 5. Clareza do sinal
        clarity_score = self._evaluate_signal_clarity(signal)
        score += clarity_score * weights['signal_clarity']
        
        return min(score, 1.0)
        
    def _calculate_flow_alignment(self, signal: Dict, flow_consensus: Dict) -> float:
        """Calcula alinhamento com consenso de fluxo"""
        if not flow_consensus or flow_consensus.get('strength', 0) == 0:
            return 0.5  # Neutro se não há consenso
            
        signal_action = signal.get('signal', {}).get('action', 'hold')
        flow_direction = flow_consensus.get('direction', 'neutral')
        flow_strength = flow_consensus.get('strength', 0)
        
        # Alinhamento perfeito
        if (signal_action == 'buy' and flow_direction == 'bullish') or \
           (signal_action == 'sell' and flow_direction == 'bearish'):
            return min(flow_strength * 1.2, 1.0)
            
        # Desalinhamento
        elif (signal_action == 'buy' and flow_direction == 'bearish') or \
             (signal_action == 'sell' and flow_direction == 'bullish'):
            return max(0.2 - flow_strength * 0.5, 0)
            
        # Neutro
        return 0.5
        
    def _get_historical_performance(self, agent_id: str) -> float:
        """Retorna score baseado em performance histórica"""
        stats = self.performance_history[agent_id]
        
        if stats['total'] == 0:
            return 0.5  # Score neutro para agentes novos
            
        success_rate = stats['successes'] / stats['total']
        
        # Ajustar por número de trades (mais confiável com mais dados)
        confidence_factor = min(stats['total'] / 100, 1.0)
        
        return success_rate * confidence_factor
        
    def _evaluate_market_conditions(self, signal: Dict, market_state: Dict) -> float:
        """Avalia se condições de mercado favorecem o sinal"""
        score = 0.5  # Base
        
        # Volatilidade
        volatility = market_state.get('volatility', 0.02)
        signal_type = signal.get('metadata', {}).get('signal_type', 'unknown')
        
        if signal_type == 'trend_following' and volatility > 0.03:
            score += 0.2  # Trend following melhor em alta volatilidade
        elif signal_type == 'mean_reversion' and volatility < 0.02:
            score += 0.2  # Mean reversion melhor em baixa volatilidade
            
        # Volume
        volume_ratio = market_state.get('volume_ratio', 1.0)  # vs média
        if volume_ratio > 1.5:
            score += 0.1  # Volume alto é bom para confirmação
            
        # Horário
        hour = datetime.now().hour
        if 10 <= hour <= 16:  # Horário principal
            score += 0.2
            
        return min(score, 1.0)
        
    def _evaluate_signal_clarity(self, signal: Dict) -> float:
        """Avalia clareza e completude do sinal"""
        required_fields = ['action', 'confidence', 'agent_id', 'timestamp']
        optional_fields = ['stop_loss', 'take_profit', 'metadata', 'reasoning']
        
        # Verificar campos obrigatórios
        missing_required = sum(1 for field in required_fields if field not in signal.get('signal', {}))
        if missing_required > 0:
            return 0.0
            
        # Bonus por campos opcionais
        present_optional = sum(1 for field in optional_fields if field in signal.get('signal', {}))
        clarity_score = 0.5 + (present_optional / len(optional_fields)) * 0.5
        
        return clarity_score
        
    def update_performance(self, agent_id: str, success: bool):
        """Atualiza performance histórica do agente"""
        self.performance_history[agent_id]['total'] += 1
        if success:
            self.performance_history[agent_id]['successes'] += 1


class FlowConsensusBuilder:
    """Constrói consenso a partir de sinais de fluxo"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.FlowConsensusBuilder")
        
    def build(self, flow_signals: List[Dict]) -> Dict:
        """Constrói consenso de análise de fluxo"""
        if not flow_signals:
            return {'strength': 0, 'direction': 'neutral', 'confidence': 0}
            
        # Agrupar por direção
        buy_signals = [s for s in flow_signals if s.get('action') == 'buy']
        sell_signals = [s for s in flow_signals if s.get('action') == 'sell']
        
        # Calcular força ponderada
        buy_strength = sum(s.get('confidence', 0) for s in buy_signals)
        sell_strength = sum(s.get('confidence', 0) for s in sell_signals)
        
        total_strength = buy_strength + sell_strength
        
        if total_strength == 0:
            return {'strength': 0, 'direction': 'neutral', 'confidence': 0}
            
        # Determinar direção e força do consenso
        if buy_strength > sell_strength:
            direction = 'bullish'
            strength = buy_strength / total_strength
            net_strength = (buy_strength - sell_strength) / total_strength
        else:
            direction = 'bearish'
            strength = sell_strength / total_strength
            net_strength = (sell_strength - buy_strength) / total_strength
            
        # Calcular confiança baseada em concordância
        agreement_ratio = max(buy_strength, sell_strength) / total_strength
        confidence = agreement_ratio * net_strength
        
        # Detalhes do consenso
        consensus = {
            'strength': strength,
            'direction': direction,
            'confidence': confidence,
            'buy_strength': buy_strength,
            'sell_strength': sell_strength,
            'net_strength': net_strength,
            'participating_agents': len(flow_signals),
            'details': {
                'order_flow_signals': len([s for s in flow_signals if 'order_flow' in s.get('agent_type', '')]),
                'footprint_signals': len([s for s in flow_signals if 'footprint' in s.get('agent_type', '')]),
                'tape_signals': len([s for s in flow_signals if 'tape' in s.get('agent_type', '')]),
                'buy_count': len(buy_signals),
                'sell_count': len(sell_signals)
            }
        }
        
        return consensus


class FlowAwareCoordinator:
    """Coordenador que considera análise de fluxo na tomada de decisão"""
    
    def __init__(self, valkey_config: Optional[Dict] = None):
        self.flow_consensus_builder = FlowConsensusBuilder()
        self.signal_quality_scorer = SignalQualityScorer()
        self.logger = logging.getLogger(f"{__name__}.FlowAwareCoordinator")
        
        # Configurar Valkey
        self.valkey = None
        if ValkeyConnectionManager and valkey_config:
            self._setup_valkey(valkey_config)
        elif valkey_config:
            self.logger.warning("ValkeyConnectionManager não disponível - usando memória local")
        
        # ZMQ para receber sinais
        self.context = zmq.Context()
        self.signal_receiver = self.context.socket(zmq.SUB)
        self.signal_receiver.connect("tcp://localhost:5559")
        self.signal_receiver.setsockopt(zmq.SUBSCRIBE, b"signal_")
        
        # Publisher para decisões - usar porta diferente das já utilizadas
        self.decision_publisher = self.context.socket(zmq.PUB)
        self.decision_publisher.bind("tcp://localhost:5561")
        
        # Cache de sinais recentes
        self.recent_signals = deque(maxlen=100)
        self.signal_buffer = defaultdict(list)  # Buffer por janela de tempo
        
        # Estado do mercado
        self.market_state = {
            'volatility': 0.02,
            'volume_ratio': 1.0,
            'trend': 'neutral'
        }
        
        # Configuração
        self.coordination_window = 1.0  # segundos para agrupar sinais
        self.min_signals = 2  # mínimo de sinais para coordenar
        
        self.logger.info("FlowAwareCoordinator inicializado")
        
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
                self.logger.info("Coordenador conectado ao Valkey")
                
                # Subscrever em canais de feedback
                feedback_channels = [
                    'hmarl:feedback:*',
                    'hmarl:decisions:stream'
                ]
                self.valkey_pubsub = self.valkey.subscribe_signals(feedback_channels)
                
            else:
                self.logger.error("Falha ao conectar com Valkey")
                self.valkey = None
                
        except Exception as e:
            self.logger.error(f"Erro configurando Valkey: {e}")
            self.valkey = None
        
    def collect_agent_signals(self, timeout: float = 0.1) -> List[Dict]:
        """Coleta sinais de todos os agentes"""
        signals = []
        end_time = time.time() + timeout
        
        while time.time() < end_time:
            try:
                # Non-blocking receive
                topic, data = self.signal_receiver.recv_multipart(zmq.NOBLOCK)
                signal = orjson.loads(data)
                signals.append(signal)
                self.recent_signals.append(signal)
                
                # Adicionar ao buffer temporal
                window_key = int(time.time() / self.coordination_window)
                self.signal_buffer[window_key].append(signal)
                
                # Persistir sinal no Valkey se disponível
                if self.valkey and self.valkey.is_connected():
                    try:
                        signal_key = f"signal_{signal.get('agent_id', 'unknown')}_{int(time.time()*1000)}"
                        self.valkey.client.setex(
                            f"hmarl:signal:{signal_key}",
                            300,  # TTL de 5 minutos
                            orjson.dumps(signal)
                        )
                    except Exception as e:
                        self.logger.error(f"Erro persistindo sinal: {e}")
                
            except zmq.Again:
                # Sem mensagens disponíveis
                time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Erro coletando sinais: {e}")
                
        return signals
        
    def coordinate_with_flow_analysis(self) -> Optional[Dict]:
        """Coordena decisões considerando análise de fluxo"""
        # Coletar sinais da janela atual
        current_window = int(time.time() / self.coordination_window)
        all_signals = self.signal_buffer.get(current_window, [])
        
        if len(all_signals) < self.min_signals:
            return None
            
        # Limpar buffer antigo
        self._cleanup_old_signals(current_window)
        
        # Separar por tipo
        flow_signals = [s for s in all_signals if 'flow' in s.get('agent_type', '')]
        traditional_signals = [s for s in all_signals if 'flow' not in s.get('agent_type', '')]
        
        # Construir consenso de fluxo
        flow_consensus = self.flow_consensus_builder.build(flow_signals)
        
        # Avaliar qualidade dos sinais
        scored_signals = []
        for signal in all_signals:
            score = self.signal_quality_scorer.score(
                signal,
                flow_consensus,
                self.get_current_market_state()
            )
            scored_signals.append((score, signal))
            
        # Ordenar por score
        scored_signals.sort(key=lambda x: x[0], reverse=True)
        
        # Selecionar melhor estratégia
        best_strategy = self._select_best_strategy(scored_signals, flow_consensus)
        
        if best_strategy:
            # Publicar decisão
            self._publish_decision(best_strategy)
            
            # Persistir decisão no Valkey
            if self.valkey and self.valkey.is_connected():
                self._persist_decision(best_strategy)
            
        return best_strategy
        
    def _select_best_strategy(self, scored_signals: List[Tuple[float, Dict]], 
                            flow_consensus: Dict) -> Optional[Dict]:
        """Seleciona melhor estratégia considerando fluxo"""
        if not scored_signals:
            return None
            
        # Se há consenso forte no fluxo, priorizar sinais alinhados
        if flow_consensus['strength'] > 0.7:
            # Filtrar apenas sinais alinhados com fluxo
            aligned_signals = [
                (score, signal) for score, signal in scored_signals
                if self._is_aligned_with_flow(signal, flow_consensus)
            ]
            
            if aligned_signals:
                # Pegar o de maior score entre os alinhados
                best_score, best_signal = aligned_signals[0]
                
                return {
                    'decision_id': f"flow_aligned_{int(time.time()*1000)}",
                    'selected_agent': best_signal['agent_id'],
                    'action': best_signal['signal']['action'],
                    'confidence': best_signal['signal']['confidence'] * flow_consensus['strength'],
                    'reasoning': f"flow_aligned_{flow_consensus['direction']}",
                    'flow_consensus': flow_consensus,
                    'quality_score': best_score,
                    'signal': best_signal['signal'],
                    'metadata': {
                        'total_signals': len(scored_signals),
                        'aligned_signals': len(aligned_signals),
                        'flow_strength': flow_consensus['strength']
                    }
                }
                
        # Caso contrário, usar melhor sinal geral
        if scored_signals:
            best_score, best_signal = scored_signals[0]
            
            # Só prosseguir se score mínimo
            if best_score < 0.4:
                self.logger.info(f"Melhor sinal com score {best_score:.2f} abaixo do mínimo")
                return None
                
            return {
                'decision_id': f"best_score_{int(time.time()*1000)}",
                'selected_agent': best_signal['agent_id'],
                'action': best_signal['signal']['action'],
                'confidence': best_signal['signal']['confidence'],
                'reasoning': 'highest_quality_score',
                'quality_score': best_score,
                'signal': best_signal['signal'],
                'flow_consensus': flow_consensus,
                'metadata': {
                    'total_signals': len(scored_signals),
                    'score_distribution': self._get_score_distribution(scored_signals)
                }
            }
            
        return None
        
    def _is_aligned_with_flow(self, signal: Dict, flow_consensus: Dict) -> bool:
        """Verifica se sinal está alinhado com consenso de fluxo"""
        signal_action = signal.get('signal', {}).get('action', 'hold')
        flow_direction = flow_consensus.get('direction', 'neutral')
        
        return (signal_action == 'buy' and flow_direction == 'bullish') or \
               (signal_action == 'sell' and flow_direction == 'bearish')
               
    def _get_score_distribution(self, scored_signals: List[Tuple[float, Dict]]) -> Dict:
        """Calcula distribuição dos scores"""
        scores = [score for score, _ in scored_signals]
        
        return {
            'max': max(scores) if scores else 0,
            'min': min(scores) if scores else 0,
            'mean': np.mean(scores) if scores else 0,
            'std': np.std(scores) if scores else 0,
            'above_0.7': sum(1 for s in scores if s > 0.7),
            'above_0.5': sum(1 for s in scores if s > 0.5)
        }
        
    def _cleanup_old_signals(self, current_window: int):
        """Remove sinais antigos do buffer"""
        old_windows = [w for w in self.signal_buffer.keys() if w < current_window - 5]
        for window in old_windows:
            del self.signal_buffer[window]
            
    def _publish_decision(self, decision: Dict):
        """Publica decisão coordenada"""
        try:
            topic = b"decision_coordinated"
            message = orjson.dumps(decision)
            self.decision_publisher.send_multipart([topic, message])
            
            self.logger.info(f"Decisão publicada: {decision['action']} "
                           f"com confiança {decision['confidence']:.2f}")
        except Exception as e:
            self.logger.error(f"Erro publicando decisão: {e}")
            
    def _persist_decision(self, decision: Dict):
        """Persiste decisão no Valkey"""
        try:
            # Adicionar metadados
            decision['coordinator_id'] = 'flow_aware_coordinator'
            decision['timestamp'] = time.time()
            
            # Armazenar decisão
            if self.valkey.store_decision(decision):
                self.logger.debug(f"Decisão {decision['decision_id']} persistida")
                
                # Armazenar contexto de fluxo associado
                if 'flow_consensus' in decision:
                    symbol = decision.get('metadata', {}).get('symbol', 'UNKNOWN')
                    self.valkey.store_flow_state(symbol, decision['flow_consensus'])
                    
        except Exception as e:
            self.logger.error(f"Erro persistindo decisão: {e}")
            
    def get_current_market_state(self) -> Dict:
        """Retorna estado atual do mercado"""
        # Em produção, isso viria de dados reais
        return self.market_state.copy()
        
    def update_market_state(self, new_state: Dict):
        """Atualiza estado do mercado"""
        self.market_state.update(new_state)
        
        # Persistir estado do mercado se Valkey disponível
        if self.valkey and self.valkey.is_connected():
            try:
                market_key = "hmarl:market_state:current"
                self.valkey.client.setex(
                    market_key,
                    3600,  # TTL de 1 hora
                    orjson.dumps(self.market_state)
                )
            except Exception as e:
                self.logger.error(f"Erro persistindo estado do mercado: {e}")
        
    def run_coordination_loop(self):
        """Loop principal de coordenação"""
        self.logger.info("Iniciando loop de coordenação")
        
        while True:
            try:
                # Coletar sinais
                self.collect_agent_signals(timeout=0.1)
                
                # Coordenar a cada janela
                if time.time() % self.coordination_window < 0.1:
                    decision = self.coordinate_with_flow_analysis()
                    
                    if decision:
                        self.logger.info(f"Decisão coordenada: {decision['decision_id']}")
                        
                # Processar mensagens do Valkey se disponível
                if self.valkey and hasattr(self, 'valkey_pubsub'):
                    self._process_valkey_messages()
                        
                time.sleep(0.05)
                
            except KeyboardInterrupt:
                self.logger.info("Parando coordenador...")
                break
            except Exception as e:
                self.logger.error(f"Erro no loop de coordenação: {e}")
                time.sleep(1)
                
    def get_coordination_stats(self) -> Dict:
        """Retorna estatísticas de coordenação"""
        return {
            'total_signals_received': len(self.recent_signals),
            'active_windows': len(self.signal_buffer),
            'signal_quality_stats': {
                agent_id: {
                    'success_rate': stats['successes'] / stats['total'] if stats['total'] > 0 else 0,
                    'total_signals': stats['total']
                }
                for agent_id, stats in self.signal_quality_scorer.performance_history.items()
            }
        }
        
    def _process_valkey_messages(self):
        """Processa mensagens do Valkey pub/sub"""
        try:
            message = self.valkey_pubsub.get_message(timeout=0.01)
            if message and message['type'] == 'message':
                # Processar feedback de agentes
                data = orjson.loads(message['data'])
                if 'agent_id' in data and 'reward' in data:
                    # Atualizar performance histórica
                    agent_id = data['agent_id']
                    success = data.get('profitable', False)
                    self.signal_quality_scorer.update_performance(agent_id, success)
                    
        except Exception as e:
            self.logger.error(f"Erro processando mensagem Valkey: {e}")
            
    def get_coordination_stats_from_valkey(self) -> Dict:
        """Recupera estatísticas do Valkey"""
        if not self.valkey or not self.valkey.is_connected():
            return {}
            
        try:
            stats = {
                'recent_decisions': self.valkey.get_recent_decisions(limit=50),
                'valkey_health': self.valkey.health_check()
            }
            
            # Buscar performance de agentes
            agent_performance = {}
            for agent_id in self.signal_quality_scorer.performance_history.keys():
                perf = self.valkey.get_agent_performance(agent_id)
                if perf:
                    agent_performance[agent_id] = perf
                    
            stats['agent_performance'] = agent_performance
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Erro recuperando stats do Valkey: {e}")
            return {}
            
    def shutdown(self):
        """Desliga o coordenador"""
        self.logger.info("Desligando coordenador...")
        
        # Fechar conexão Valkey
        if self.valkey:
            self.valkey.disconnect()
            
        self.signal_receiver.close()
        self.decision_publisher.close()
        self.context.term()


# Função auxiliar para criar múltiplos agentes votando
class VotingSystem:
    """Sistema de votação para decisões de múltiplos agentes"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.VotingSystem")
        
    def weighted_vote(self, votes: List[Dict], weights: Optional[Dict] = None) -> Dict:
        """Realiza votação ponderada"""
        if not votes:
            return {'action': 'hold', 'confidence': 0, 'consensus': 0}
            
        # Se não há pesos, usar uniforme
        if weights is None:
            weights = {vote['agent_id']: 1.0 for vote in votes}
            
        # Agrupar votos por ação
        action_votes = defaultdict(float)
        action_confidence = defaultdict(list)
        
        for vote in votes:
            agent_id = vote['agent_id']
            action = vote['signal']['action']
            confidence = vote['signal']['confidence']
            weight = weights.get(agent_id, 1.0)
            
            action_votes[action] += weight * confidence
            action_confidence[action].append(confidence)
            
        # Determinar ação vencedora
        winning_action = max(action_votes.items(), key=lambda x: x[1])[0]
        
        # Calcular consenso
        total_weight = sum(action_votes.values())
        consensus = action_votes[winning_action] / total_weight if total_weight > 0 else 0
        
        # Confiança média dos que votaram na ação vencedora
        avg_confidence = np.mean(action_confidence[winning_action])
        
        return {
            'action': winning_action,
            'confidence': avg_confidence * consensus,  # Ajustado pelo consenso
            'consensus': consensus,
            'vote_distribution': dict(action_votes),
            'participants': len(votes)
        }


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Testar componentes
    
    # 1. Teste do FlowConsensusBuilder
    consensus_builder = FlowConsensusBuilder()
    
    test_signals = [
        {'action': 'buy', 'confidence': 0.8, 'agent_type': 'order_flow'},
        {'action': 'buy', 'confidence': 0.7, 'agent_type': 'footprint'},
        {'action': 'sell', 'confidence': 0.6, 'agent_type': 'tape_reading'},
        {'action': 'buy', 'confidence': 0.9, 'agent_type': 'order_flow'}
    ]
    
    consensus = consensus_builder.build(test_signals)
    print(f"\nConsenso de fluxo: {consensus}")
    
    # 2. Teste do SignalQualityScorer
    scorer = SignalQualityScorer()
    
    test_signal = {
        'agent_id': 'test_agent_001',
        'signal': {
            'action': 'buy',
            'confidence': 0.75,
            'stop_loss': 4950,
            'take_profit': 5050
        },
        'metadata': {
            'signal_type': 'trend_following'
        }
    }
    
    market_state = {
        'volatility': 0.025,
        'volume_ratio': 1.2,
        'trend': 'up'
    }
    
    score = scorer.score(test_signal, consensus, market_state)
    print(f"\nScore de qualidade do sinal: {score:.3f}")
    
    # 3. Teste do VotingSystem
    voting = VotingSystem()
    
    votes = [
        {'agent_id': 'agent1', 'signal': {'action': 'buy', 'confidence': 0.8}},
        {'agent_id': 'agent2', 'signal': {'action': 'buy', 'confidence': 0.7}},
        {'agent_id': 'agent3', 'signal': {'action': 'sell', 'confidence': 0.6}},
        {'agent_id': 'agent4', 'signal': {'action': 'buy', 'confidence': 0.9}}
    ]
    
    result = voting.weighted_vote(votes)
    print(f"\nResultado da votação: {result}")
    
    # 4. Criar coordenador (sem rodar o loop)
    coordinator = FlowAwareCoordinator()
    print(f"\nCoordenador criado. Use coordinator.run_coordination_loop() para iniciar")