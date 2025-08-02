"""
Footprint Pattern Agent - HMARL Fase 2 Semana 5
Agente especializado em padrões de footprint
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, List, Tuple
from collections import deque
from datetime import datetime
import time
import pickle
from pathlib import Path

from src.agents.flow_aware_base_agent import FlowAwareBaseAgent


class FootprintPatternLibrary:
    """Biblioteca de padrões de footprint conhecidos"""
    
    def __init__(self):
        self.patterns = {
            'p_reversal': {
                'name': 'P-Reversal',
                'type': 'reversal',
                'description': 'Forte absorção seguida de reversão',
                'confidence_base': 0.7
            },
            'b_reversal': {
                'name': 'B-Reversal',
                'type': 'reversal',
                'description': 'Rejeição em níveis chave com delta negativo',
                'confidence_base': 0.65
            },
            'continuation_flag': {
                'name': 'Continuation Flag',
                'type': 'continuation',
                'description': 'Pausa na tendência com volume decrescente',
                'confidence_base': 0.6
            },
            'exhaustion': {
                'name': 'Exhaustion Pattern',
                'type': 'exhaustion',
                'description': 'Alto volume sem progresso de preço',
                'confidence_base': 0.75
            },
            'absorption': {
                'name': 'Absorption Pattern',
                'type': 'absorption',
                'description': 'Grande volume em nível específico',
                'confidence_base': 0.7
            },
            'initiative_buying': {
                'name': 'Initiative Buying',
                'type': 'continuation',
                'description': 'Compras agressivas movendo preço',
                'confidence_base': 0.65
            },
            'initiative_selling': {
                'name': 'Initiative Selling',
                'type': 'continuation',
                'description': 'Vendas agressivas movendo preço',
                'confidence_base': 0.65
            }
        }
        
    def get_pattern(self, pattern_id: str) -> Optional[Dict]:
        """Retorna padrão específico"""
        return self.patterns.get(pattern_id)
        
    def get_all_patterns(self) -> Dict:
        """Retorna todos os padrões"""
        return self.patterns.copy()


class FootprintPatternMatcher:
    """Matcher para identificar padrões de footprint"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PatternMatcher")
        
    def match(self, current_footprint: Dict, pattern_library: FootprintPatternLibrary) -> List[Dict]:
        """Identifica padrões no footprint atual"""
        detected_patterns = []
        
        # Extrair features do footprint
        delta = current_footprint.get('delta', 0)
        volume = current_footprint.get('volume', 0)
        price_movement = current_footprint.get('price_movement', 0)
        imbalance = current_footprint.get('imbalance', 0)
        absorption_score = current_footprint.get('absorption_score', 0)
        
        # P-Reversal: Forte absorção + delta positivo + pouco movimento
        if absorption_score > 0.7 and delta > 100 and abs(price_movement) < 0.1:
            pattern = pattern_library.get_pattern('p_reversal')
            confidence = pattern['confidence_base'] * (absorption_score + 0.3)
            detected_patterns.append({
                'name': pattern['name'],
                'type': pattern['type'],
                'confidence': min(confidence, 1.0),
                'direction': 'bullish'
            })
            
        # B-Reversal: Delta negativo forte + rejeição
        if delta < -100 and price_movement < -0.05 and imbalance < -0.3:
            pattern = pattern_library.get_pattern('b_reversal')
            confidence = pattern['confidence_base'] * abs(imbalance * 2)
            detected_patterns.append({
                'name': pattern['name'],
                'type': pattern['type'],
                'confidence': min(confidence, 1.0),
                'direction': 'bearish'
            })
            
        # Exhaustion: Alto volume sem progresso
        avg_volume = current_footprint.get('avg_volume', 1)
        if volume > avg_volume * 2 and abs(price_movement) < 0.05:
            pattern = pattern_library.get_pattern('exhaustion')
            confidence = pattern['confidence_base'] * (volume / (avg_volume * 2))
            detected_patterns.append({
                'name': pattern['name'],
                'type': pattern['type'],
                'confidence': min(confidence, 1.0),
                'direction': 'neutral'
            })
            
        # Initiative Buying/Selling
        if abs(delta) > 50 and abs(price_movement) > 0.1:
            if delta > 0 and price_movement > 0:
                pattern = pattern_library.get_pattern('initiative_buying')
                confidence = pattern['confidence_base'] * (delta / 100)
                detected_patterns.append({
                    'name': pattern['name'],
                    'type': pattern['type'],
                    'confidence': min(confidence, 1.0),
                    'direction': 'bullish'
                })
            elif delta < 0 and price_movement < 0:
                pattern = pattern_library.get_pattern('initiative_selling')
                confidence = pattern['confidence_base'] * (abs(delta) / 100)
                detected_patterns.append({
                    'name': pattern['name'],
                    'type': pattern['type'],
                    'confidence': min(confidence, 1.0),
                    'direction': 'bearish'
                })
                
        return detected_patterns


class FootprintPatternPredictor:
    """Preditor ML para evolução de padrões de footprint"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.logger = logging.getLogger(f"{__name__}.PatternPredictor")
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self.logger.info("Iniciando sem modelo pré-treinado")
            
    def load_model(self, model_path: str):
        """Carrega modelo pré-treinado"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.logger.info(f"Modelo carregado de {model_path}")
        except Exception as e:
            self.logger.error(f"Erro carregando modelo: {e}")
            
    def predict(self, current_footprint: Dict, detected_patterns: List[Dict]) -> Dict:
        """Prediz evolução do padrão"""
        prediction = {
            'next_action': 'hold',
            'confidence': 0.0,
            'expected_movement': 0.0,
            'time_horizon': 5  # minutos
        }
        
        if not detected_patterns:
            return prediction
            
        # Por enquanto, usar lógica simples
        # Em produção, usaria o modelo ML treinado
        strongest_pattern = max(detected_patterns, key=lambda p: p['confidence'])
        
        if strongest_pattern['type'] == 'reversal':
            if strongest_pattern['direction'] == 'bullish':
                prediction['next_action'] = 'buy'
                prediction['expected_movement'] = 0.5
            else:
                prediction['next_action'] = 'sell'
                prediction['expected_movement'] = -0.5
                
        elif strongest_pattern['type'] == 'continuation':
            if strongest_pattern['direction'] == 'bullish':
                prediction['next_action'] = 'buy'
                prediction['expected_movement'] = 0.3
            else:
                prediction['next_action'] = 'sell'
                prediction['expected_movement'] = -0.3
                
        elif strongest_pattern['type'] == 'exhaustion':
            # Exhaustion sugere reversão iminente
            prediction['next_action'] = 'wait'
            prediction['expected_movement'] = 0.0
            
        prediction['confidence'] = strongest_pattern['confidence'] * 0.8
        
        return prediction


class FootprintPatternAgent(FlowAwareBaseAgent):
    """Agente especializado em padrões de footprint"""
    
    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            'min_pattern_confidence': 0.5,
            'prediction_weight': 0.7,
            'pattern_memory_size': 1000
        }
        if config:
            default_config.update(config)
            
        super().__init__('footprint_pattern', default_config)
        
        # Biblioteca de padrões
        self.pattern_library = FootprintPatternLibrary()
        self.pattern_matcher = FootprintPatternMatcher()
        
        # Machine learning para padrões
        model_path = config.get('model_path') if config else None
        self.pattern_predictor = FootprintPatternPredictor(model_path)
        
        # Histórico de padrões
        self.pattern_history = deque(maxlen=self.config['pattern_memory_size'])
        self.pattern_performance = {}
        
        self.logger.info("FootprintPatternAgent inicializado")
        
    def generate_signal_with_flow(self, price_state: Dict, flow_state: Dict) -> Dict:
        """Gera sinal baseado em padrões de footprint"""
        # Obter footprint atual
        current_footprint = self._extract_footprint_data(flow_state, price_state)
        
        # Detectar padrões conhecidos
        detected_patterns = self.pattern_matcher.match(
            current_footprint,
            self.pattern_library
        )
        
        # Filtrar por confiança mínima
        detected_patterns = [
            p for p in detected_patterns 
            if p['confidence'] >= self.config['min_pattern_confidence']
        ]
        
        if not detected_patterns:
            return {
                'action': 'hold',
                'confidence': 0.0,
                'metadata': {'reason': 'no_patterns_detected'}
            }
            
        # Predizer evolução do padrão
        pattern_prediction = self.pattern_predictor.predict(
            current_footprint,
            detected_patterns
        )
        
        # Gerar sinal baseado em padrões
        signal = self._generate_pattern_signal(
            detected_patterns,
            pattern_prediction,
            price_state
        )
        
        # Armazenar no histórico
        self.pattern_history.append({
            'timestamp': time.time(),
            'patterns': detected_patterns,
            'signal': signal,
            'footprint': current_footprint
        })
        
        return signal
        
    def _extract_footprint_data(self, flow_state: Dict, price_state: Dict) -> Dict:
        """Extrai dados de footprint do estado atual"""
        # Calcular movimento de preço
        price_change = 0.0
        if 'price' in price_state and 'prev_price' in price_state:
            if price_state['prev_price'] > 0:
                price_change = (price_state['price'] - price_state['prev_price']) / price_state['prev_price']
                
        footprint = {
            'delta': flow_state.get('delta', 0),
            'volume': flow_state.get('volume', 0),
            'avg_volume': flow_state.get('avg_volume', 1),
            'price_movement': price_change,
            'imbalance': flow_state.get('last_ofi', 0),
            'absorption_score': flow_state.get('absorption', 0),
            'bid_volume': flow_state.get('bid_volume', 0),
            'ask_volume': flow_state.get('ask_volume', 0),
            'footprint_pattern': flow_state.get('footprint_pattern', '')
        }
        
        return footprint
        
    def _generate_pattern_signal(self, patterns: List[Dict], prediction: Dict, 
                                price_state: Dict) -> Dict:
        """Gera sinal baseado em padrões detectados"""
        if not patterns:
            return {'action': 'hold', 'confidence': 0}
            
        # Analisar padrão mais forte
        strongest_pattern = max(patterns, key=lambda p: p['confidence'])
        
        signal = {
            'action': 'hold',
            'confidence': 0,
            'metadata': {
                'pattern': strongest_pattern['name'],
                'pattern_confidence': strongest_pattern['confidence'],
                'all_patterns': [p['name'] for p in patterns],
                'prediction': prediction
            }
        }
        
        # Usar predição se disponível
        if prediction['confidence'] > 0.3:
            signal['action'] = prediction['next_action']
            signal['confidence'] = prediction['confidence'] * self.config['prediction_weight']
            signal['metadata']['expected_movement'] = prediction['expected_movement']
        else:
            # Usar lógica baseada no tipo de padrão
            if strongest_pattern['type'] == 'reversal':
                if strongest_pattern['direction'] == 'bullish':
                    signal['action'] = 'buy'
                else:
                    signal['action'] = 'sell'
                signal['confidence'] = strongest_pattern['confidence'] * 0.8
                
            elif strongest_pattern['type'] == 'continuation':
                current_trend = self._determine_trend(price_state)
                if current_trend == 'up' and strongest_pattern['direction'] == 'bullish':
                    signal['action'] = 'buy'
                    signal['confidence'] = strongest_pattern['confidence'] * 0.9
                elif current_trend == 'down' and strongest_pattern['direction'] == 'bearish':
                    signal['action'] = 'sell'
                    signal['confidence'] = strongest_pattern['confidence'] * 0.9
                else:
                    signal['confidence'] = strongest_pattern['confidence'] * 0.5
                    
            elif strongest_pattern['type'] == 'exhaustion':
                # Exhaustion sugere cautela
                signal['action'] = 'hold'
                signal['confidence'] = 0.2
                signal['metadata']['reason'] = 'exhaustion_detected'
                
        return signal
        
    def _determine_trend(self, price_state: Dict) -> str:
        """Determina tendência atual do preço"""
        # Simplificado - em produção seria mais sofisticado
        if 'price' in price_state and 'sma_20' in price_state:
            if price_state['price'] > price_state['sma_20']:
                return 'up'
            else:
                return 'down'
        return 'neutral'
        
    def learn_from_pattern_performance(self, pattern_id: str, outcome: Dict):
        """Aprende com performance de padrões"""
        if pattern_id not in self.pattern_performance:
            self.pattern_performance[pattern_id] = {
                'occurrences': 0,
                'successful': 0,
                'total_return': 0.0
            }
            
        stats = self.pattern_performance[pattern_id]
        stats['occurrences'] += 1
        
        if outcome.get('profitable', False):
            stats['successful'] += 1
            
        stats['total_return'] += outcome.get('return', 0.0)
        
        # Calcular taxa de sucesso
        success_rate = stats['successful'] / stats['occurrences']
        avg_return = stats['total_return'] / stats['occurrences']
        
        self.logger.info(f"Pattern {pattern_id}: Success={success_rate:.2%}, AvgReturn={avg_return:.4f}")
        
    def get_pattern_statistics(self) -> Dict:
        """Retorna estatísticas dos padrões"""
        stats = {}
        
        for pattern_id, perf in self.pattern_performance.items():
            if perf['occurrences'] > 0:
                stats[pattern_id] = {
                    'occurrences': perf['occurrences'],
                    'success_rate': perf['successful'] / perf['occurrences'],
                    'avg_return': perf['total_return'] / perf['occurrences']
                }
                
        return stats


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Criar agente
    config = {
        'min_pattern_confidence': 0.5,
        'prediction_weight': 0.7
    }
    
    agent = FootprintPatternAgent(config)
    
    # Simular dados
    price_state = {
        'price': 5000.0,
        'prev_price': 4995.0,
        'sma_20': 4980.0
    }
    
    flow_state = {
        'delta': 150,
        'volume': 500,
        'avg_volume': 200,
        'last_ofi': 0.4,
        'absorption': 0.8,
        'bid_volume': 300,
        'ask_volume': 200,
        'footprint_pattern': 'absorption'
    }
    
    # Gerar sinal
    signal = agent.generate_signal_with_flow(price_state, flow_state)
    
    print(f"Sinal gerado: {signal}")
    print(f"Padrões na biblioteca: {list(agent.pattern_library.patterns.keys())}")
    
    # Simular aprendizado
    agent.learn_from_pattern_performance('p_reversal', {
        'profitable': True,
        'return': 0.015
    })
    
    print(f"Estatísticas: {agent.get_pattern_statistics()}")
    
    agent.shutdown()