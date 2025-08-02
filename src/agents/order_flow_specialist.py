"""
Order Flow Specialist Agent - HMARL Fase 2 Semana 5
Agente especializado em análise de order flow
"""

import numpy as np
import logging
from typing import Dict, Optional, List, Tuple
from collections import deque
from datetime import datetime, timedelta
import time

from src.agents.flow_aware_base_agent import FlowAwareBaseAgent


class DeltaAnalyzer:
    """Analisador de delta (diferença entre volume comprador e vendedor)"""
    
    def __init__(self):
        self.delta_history = deque(maxlen=1000)
        self.cumulative_delta = 0
        
    def update(self, buy_volume: float, sell_volume: float):
        """Atualiza delta com novo volume"""
        delta = buy_volume - sell_volume
        self.delta_history.append({
            'timestamp': time.time(),
            'delta': delta,
            'cumulative': self.cumulative_delta + delta
        })
        self.cumulative_delta += delta
        
    def get_divergence(self, price_history: List[float]) -> float:
        """Calcula divergência entre preço e delta"""
        if len(self.delta_history) < 10 or len(price_history) < 10:
            return 0.0
            
        # Calcular tendência do preço
        price_change = (price_history[-1] - price_history[-10]) / price_history[-10]
        
        # Calcular tendência do delta
        recent_deltas = [d['delta'] for d in list(self.delta_history)[-10:]]
        delta_trend = np.mean(recent_deltas)
        
        # Divergência: preço subindo mas delta negativo ou vice-versa
        if price_change > 0 and delta_trend < 0:
            return -1.0  # Divergência bearish
        elif price_change < 0 and delta_trend > 0:
            return 1.0   # Divergência bullish
            
        return 0.0


class AbsorptionDetector:
    """Detector de absorção (grandes volumes sem movimento de preço)"""
    
    def detect(self, flow_state: Dict, price_state: Dict) -> Dict:
        """Detecta absorção no mercado"""
        result = {
            'detected': False,
            'type': None,
            'strength': 0.0,
            'level': 0.0
        }
        
        # Verificar se há grande volume
        volume = flow_state.get('volume', 0)
        avg_volume = flow_state.get('avg_volume', 1)
        
        if volume < avg_volume * 1.5:
            return result
            
        # Verificar movimento de preço
        price_change = price_state.get('price_change_pct', 0)
        
        # Absorção: muito volume, pouco movimento
        if abs(price_change) < 0.1:  # Menos de 0.1% de movimento
            result['detected'] = True
            result['strength'] = volume / avg_volume
            result['level'] = price_state.get('price', 0)
            
            # Determinar tipo baseado no fluxo
            ofi = flow_state.get('last_ofi', 0)
            if ofi > 0.2:
                result['type'] = 'buying_absorption'
            elif ofi < -0.2:
                result['type'] = 'selling_absorption'
            else:
                result['type'] = 'neutral_absorption'
                
        return result


class SweepDetector:
    """Detector de sweep (movimento agressivo consumindo liquidez)"""
    
    def detect(self, flow_state: Dict, flow_history: deque) -> Dict:
        """Detecta sweep orders"""
        result = {
            'detected': False,
            'direction': None,
            'intensity': 0.0,
            'levels_swept': 0
        }
        
        if len(flow_history) < 5:
            return result
            
        # Analisar últimos 5 períodos
        recent_flows = list(flow_history)[-5:]
        
        # Verificar consistência de direção
        ofis = [f.get('ofi', 0) for f in recent_flows]
        aggression_scores = [f.get('aggression_score', 0) for f in recent_flows]
        
        # Sweep bullish
        if all(ofi > 0.3 for ofi in ofis) and np.mean(aggression_scores) > 0.7:
            result['detected'] = True
            result['direction'] = 'bullish'
            result['intensity'] = np.mean(ofis)
            result['levels_swept'] = sum(1 for a in aggression_scores if a > 0.8)
            
        # Sweep bearish
        elif all(ofi < -0.3 for ofi in ofis) and np.mean(aggression_scores) > 0.7:
            result['detected'] = True
            result['direction'] = 'bearish'
            result['intensity'] = abs(np.mean(ofis))
            result['levels_swept'] = sum(1 for a in aggression_scores if a > 0.8)
            
        return result


class OrderFlowSpecialistAgent(FlowAwareBaseAgent):
    """Agente especializado em order flow analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            'ofi_threshold': 0.3,
            'delta_threshold': 1000,
            'aggression_threshold': 0.6,
            'min_confidence': 0.4
        }
        if config:
            default_config.update(config)
            
        super().__init__('order_flow_specialist', default_config)
        
        # Componentes especializados
        self.delta_analyzer = DeltaAnalyzer()
        self.absorption_detector = AbsorptionDetector()
        self.sweep_detector = SweepDetector()
        
        # Estado específico do agente
        self.flow_history = deque(maxlen=100)
        self.pattern_confidence = {}
        self.price_history = deque(maxlen=100)
        
        self.logger.info(f"OrderFlowSpecialistAgent inicializado com thresholds: OFI={self.config['ofi_threshold']}")
        
    def process_flow_data(self, flow_data: Dict):
        """Processa dados de fluxo com análise especializada"""
        super().process_flow_data(flow_data)
        
        # Adicionar ao histórico
        self.flow_history.append(flow_data)
        
        # Atualizar delta analyzer
        buy_vol = flow_data.get('buy_volume', 0)
        sell_vol = flow_data.get('sell_volume', 0)
        if buy_vol > 0 or sell_vol > 0:
            self.delta_analyzer.update(buy_vol, sell_vol)
            
    def process_market_data(self, market_data: Dict):
        """Processa dados de mercado"""
        super().process_market_data(market_data)
        
        # Manter histórico de preços
        price = market_data.get('price', 0)
        if price > 0:
            self.price_history.append(price)
            
    def generate_signal_with_flow(self, price_state: Dict, flow_state: Dict) -> Dict:
        """Gera sinal baseado em análise de order flow"""
        # Analisar order flow imbalance
        ofi_signal = self._analyze_ofi_signal(flow_state)
        
        # Analisar delta
        delta_signal = self._analyze_delta_signal(flow_state)
        
        # Detectar absorção
        absorption = self.absorption_detector.detect(flow_state, price_state)
        
        # Detectar sweep
        sweep = self.sweep_detector.detect(flow_state, self.flow_history)
        
        # Combinar sinais
        combined_signal = self._combine_flow_signals({
            'ofi': ofi_signal,
            'delta': delta_signal,
            'absorption': absorption,
            'sweep': sweep
        })
        
        # Adicionar contexto de fluxo
        combined_signal['metadata'] = {
            'flow_patterns_detected': self._get_detected_patterns(flow_state),
            'flow_strength': self._calculate_flow_strength(flow_state),
            'signal_source': 'order_flow_analysis',
            'delta_cumulative': self.delta_analyzer.cumulative_delta
        }
        
        return combined_signal
        
    def _analyze_ofi_signal(self, flow_state: Dict) -> Dict:
        """Analisa sinal baseado em order flow imbalance"""
        ofi_1m = flow_state.get('ofi_1m', 0)
        ofi_5m = flow_state.get('ofi_5m', 0)
        ofi_velocity = flow_state.get('ofi_velocity_5m', 0)
        
        signal = {'action': 'hold', 'confidence': 0}
        
        # Sinal de compra: OFI positivo forte e acelerando
        if ofi_1m > self.config['ofi_threshold'] and ofi_5m > self.config['ofi_threshold']:
            if ofi_velocity > 0:  # Acelerando
                signal = {
                    'action': 'buy',
                    'confidence': min(ofi_1m * 2, 1.0),
                    'reason': f'strong_positive_ofi_{ofi_1m:.2f}'
                }
                
        # Sinal de venda: OFI negativo forte e acelerando
        elif ofi_1m < -self.config['ofi_threshold'] and ofi_5m < -self.config['ofi_threshold']:
            if ofi_velocity < 0:  # Acelerando negativamente
                signal = {
                    'action': 'sell',
                    'confidence': min(abs(ofi_1m) * 2, 1.0),
                    'reason': f'strong_negative_ofi_{ofi_1m:.2f}'
                }
                
        return signal
        
    def _analyze_delta_signal(self, flow_state: Dict) -> Dict:
        """Analisa sinal baseado em delta"""
        signal = {'action': 'hold', 'confidence': 0}
        
        # Verificar divergência
        divergence = self.delta_analyzer.get_divergence(list(self.price_history))
        
        if abs(divergence) > 0.5:
            if divergence > 0:  # Divergência bullish
                signal = {
                    'action': 'buy',
                    'confidence': 0.6,
                    'reason': 'bullish_delta_divergence'
                }
            else:  # Divergência bearish
                signal = {
                    'action': 'sell',
                    'confidence': 0.6,
                    'reason': 'bearish_delta_divergence'
                }
                
        # Verificar delta extremo
        cumulative_delta = self.delta_analyzer.cumulative_delta
        if abs(cumulative_delta) > self.config['delta_threshold']:
            if cumulative_delta > 0:
                signal['confidence'] = max(signal['confidence'], 0.7)
                if signal['action'] == 'hold':
                    signal['action'] = 'buy'
                    signal['reason'] = 'extreme_positive_delta'
            else:
                signal['confidence'] = max(signal['confidence'], 0.7)
                if signal['action'] == 'hold':
                    signal['action'] = 'sell'
                    signal['reason'] = 'extreme_negative_delta'
                    
        return signal
        
    def _combine_flow_signals(self, signals: Dict) -> Dict:
        """Combina múltiplos sinais de fluxo"""
        combined = {
            'action': 'hold',
            'confidence': 0.0,
            'reasons': []
        }
        
        # Coletar todas as ações e confidências
        actions = []
        confidences = []
        
        # OFI signal
        if signals['ofi']['confidence'] > 0:
            actions.append(signals['ofi']['action'])
            confidences.append(signals['ofi']['confidence'])
            combined['reasons'].append(signals['ofi'].get('reason', 'ofi'))
            
        # Delta signal
        if signals['delta']['confidence'] > 0:
            actions.append(signals['delta']['action'])
            confidences.append(signals['delta']['confidence'])
            combined['reasons'].append(signals['delta'].get('reason', 'delta'))
            
        # Absorption signal
        if signals['absorption']['detected']:
            absorption_type = signals['absorption']['type']
            if 'buying' in absorption_type:
                actions.append('sell')  # Absorção de compra = resistência
                confidences.append(0.7)
                combined['reasons'].append('buying_absorption_resistance')
            elif 'selling' in absorption_type:
                actions.append('buy')  # Absorção de venda = suporte
                confidences.append(0.7)
                combined['reasons'].append('selling_absorption_support')
                
        # Sweep signal
        if signals['sweep']['detected']:
            if signals['sweep']['direction'] == 'bullish':
                actions.append('buy')
                confidences.append(0.8)
                combined['reasons'].append('bullish_sweep')
            else:
                actions.append('sell')
                confidences.append(0.8)
                combined['reasons'].append('bearish_sweep')
                
        # Determinar ação final
        if actions:
            # Contar votos
            buy_votes = sum(1 for a in actions if a == 'buy')
            sell_votes = sum(1 for a in actions if a == 'sell')
            
            if buy_votes > sell_votes:
                combined['action'] = 'buy'
                combined['confidence'] = np.mean([c for a, c in zip(actions, confidences) if a == 'buy'])
            elif sell_votes > buy_votes:
                combined['action'] = 'sell'
                combined['confidence'] = np.mean([c for a, c in zip(actions, confidences) if a == 'sell'])
            else:
                # Empate - usar a de maior confiança
                max_conf_idx = np.argmax(confidences)
                combined['action'] = actions[max_conf_idx]
                combined['confidence'] = confidences[max_conf_idx]
                
        return combined
        
    def _get_detected_patterns(self, flow_state: Dict) -> List[str]:
        """Retorna lista de padrões detectados"""
        patterns = []
        
        ofi = flow_state.get('last_ofi', 0)
        if abs(ofi) > 0.5:
            patterns.append(f"extreme_ofi_{ofi:.2f}")
            
        if abs(self.delta_analyzer.cumulative_delta) > 500:
            patterns.append(f"extreme_delta_{self.delta_analyzer.cumulative_delta:.0f}")
            
        return patterns
        
    def _calculate_flow_strength(self, flow_state: Dict) -> float:
        """Calcula força geral do fluxo"""
        ofi = abs(flow_state.get('last_ofi', 0))
        aggression = flow_state.get('aggression_score', 0)
        
        # Combinar métricas
        strength = (ofi + aggression) / 2.0
        return min(strength, 1.0)
        
    def learn_from_flow_feedback(self, feedback: Dict):
        """Aprendizado específico para order flow"""
        super().learn_from_feedback(feedback)
        
        # Atualizar confiança em padrões
        if 'flow_patterns_detected' in feedback.get('decision', {}).get('metadata', {}):
            patterns = feedback['decision']['metadata']['flow_patterns_detected']
            reward = feedback['reward']
            
            for pattern in patterns:
                if pattern not in self.pattern_confidence:
                    self.pattern_confidence[pattern] = 0.5
                    
                # Atualizar confiança baseado em reward
                if reward > 0:
                    self.pattern_confidence[pattern] *= 1.02
                else:
                    self.pattern_confidence[pattern] *= 0.98
                    
                # Manter entre 0.1 e 2.0
                self.pattern_confidence[pattern] = max(0.1, min(2.0, self.pattern_confidence[pattern]))
                
        # Ajustar thresholds baseado em performance
        flow_confirmation = feedback.get('flow_confirmation', {})
        if flow_confirmation.get('accuracy', 1.0) < 0.5:
            # Performance ruim - ser mais conservador
            self.config['ofi_threshold'] *= 1.05
            self.config['delta_threshold'] *= 1.05
            self.logger.info(f"Thresholds aumentados: OFI={self.config['ofi_threshold']:.3f}")
        elif flow_confirmation.get('accuracy', 0) > 0.7:
            # Performance boa - pode ser mais agressivo
            self.config['ofi_threshold'] *= 0.98
            self.config['delta_threshold'] *= 0.98
            self.logger.info(f"Thresholds reduzidos: OFI={self.config['ofi_threshold']:.3f}")


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Criar agente
    config = {
        'ofi_threshold': 0.3,
        'delta_threshold': 1000,
        'aggression_threshold': 0.6
    }
    
    agent = OrderFlowSpecialistAgent(config)
    
    # Simular alguns dados
    agent.process_market_data({
        'price': 5000.0,
        'volume': 100,
        'timestamp': time.time()
    })
    
    # Simular fluxo positivo forte
    agent.process_flow_data({
        'ofi_1m': 0.45,
        'ofi_5m': 0.40,
        'ofi_velocity_5m': 0.05,
        'buy_volume': 700,
        'sell_volume': 300,
        'aggression_score': 0.8,
        'last_ofi': 0.45
    })
    
    # Gerar sinal
    signal = agent.generate_signal_with_flow(
        agent.state['price_state'],
        agent.state['flow_state']
    )
    
    print(f"Sinal gerado: {signal}")
    print(f"Delta cumulativo: {agent.delta_analyzer.cumulative_delta}")
    
    agent.shutdown()