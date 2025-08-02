"""
Tape Reading Agent - HMARL Fase 1 Semana 3
Agente especializado em leitura de fita (tape reading)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, List, Tuple
from collections import deque, defaultdict
from datetime import datetime, timedelta
import time

from src.agents.flow_aware_base_agent import FlowAwareBaseAgent


class TapeSpeedAnalyzer:
    """Analisador de velocidade da fita"""
    
    def __init__(self, window_sizes: List[int] = [60, 300, 900]):
        self.logger = logging.getLogger(f"{__name__}.SpeedAnalyzer")
        self.window_sizes = window_sizes  # segundos
        self.trade_buffers = {size: deque() for size in window_sizes}
        
    def update(self, trade: Dict):
        """Atualiza buffers com novo trade"""
        current_time = trade.get('timestamp', time.time())
        
        for size in self.window_sizes:
            buffer = self.trade_buffers[size]
            buffer.append(trade)
            
            # Remover trades antigos
            cutoff_time = current_time - size
            while buffer and buffer[0].get('timestamp', 0) < cutoff_time:
                buffer.popleft()
                
    def analyze_speed(self) -> Dict:
        """Analisa velocidade da fita em diferentes janelas"""
        analysis = {
            'speeds': {},
            'acceleration': {},
            'volume_velocity': {},
            'trade_intensity': {}
        }
        
        for size in self.window_sizes:
            buffer = self.trade_buffers[size]
            
            if len(buffer) < 2:
                analysis['speeds'][f'{size}s'] = 0
                analysis['volume_velocity'][f'{size}s'] = 0
                analysis['trade_intensity'][f'{size}s'] = 0
                continue
                
            # Calcular velocidade (trades por segundo)
            time_span = buffer[-1]['timestamp'] - buffer[0]['timestamp']
            if time_span > 0:
                speed = len(buffer) / time_span
                analysis['speeds'][f'{size}s'] = speed
                
                # Volume por segundo
                total_volume = sum(t.get('volume', 0) for t in buffer)
                analysis['volume_velocity'][f'{size}s'] = total_volume / time_span
                
                # Intensidade (trades grandes vs pequenos)
                volumes = [t.get('volume', 0) for t in buffer]
                if volumes:
                    avg_volume = np.mean(volumes)
                    large_trades = sum(1 for v in volumes if v > avg_volume * 2)
                    analysis['trade_intensity'][f'{size}s'] = large_trades / len(buffer)
                    
        # Calcular aceleração
        if len(self.window_sizes) >= 2:
            for i in range(1, len(self.window_sizes)):
                smaller = self.window_sizes[i-1]
                larger = self.window_sizes[i]
                
                speed_small = analysis['speeds'].get(f'{smaller}s', 0)
                speed_large = analysis['speeds'].get(f'{larger}s', 0)
                
                if speed_large > 0:
                    analysis['acceleration'][f'{smaller}s_vs_{larger}s'] = (
                        (speed_small - speed_large) / speed_large
                    )
                    
        return analysis
        
    def detect_speed_patterns(self) -> List[Dict]:
        """Detecta padrões de velocidade"""
        patterns = []
        analysis = self.analyze_speed()
        
        # Padrão: Aceleração súbita
        if '60s' in analysis['speeds'] and '300s' in analysis['speeds']:
            short_speed = analysis['speeds']['60s']
            medium_speed = analysis['speeds']['300s']
            
            if medium_speed > 0 and short_speed / medium_speed > 2:
                patterns.append({
                    'type': 'sudden_acceleration',
                    'confidence': min((short_speed / medium_speed - 1) * 0.5, 1.0),
                    'description': 'Aceleração súbita na velocidade de trades'
                })
                
        # Padrão: Alta intensidade
        for window, intensity in analysis['trade_intensity'].items():
            if intensity > 0.3:  # 30% trades grandes
                patterns.append({
                    'type': 'high_intensity',
                    'window': window,
                    'confidence': min(intensity * 2, 1.0),
                    'description': f'Alta proporção de trades grandes em {window}'
                })
                
        return patterns


class TapePatternDetector:
    """Detector de padrões na fita"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PatternDetector")
        self.pattern_buffer = deque(maxlen=100)
        
    def detect_patterns(self, recent_trades: List[Dict]) -> List[Dict]:
        """Detecta padrões específicos na fita"""
        detected_patterns = []
        
        if len(recent_trades) < 5:
            return detected_patterns
            
        # Padrão: Sweep
        sweep = self._detect_sweep(recent_trades)
        if sweep:
            detected_patterns.append(sweep)
            
        # Padrão: Iceberg
        iceberg = self._detect_iceberg_pattern(recent_trades)
        if iceberg:
            detected_patterns.append(iceberg)
            
        # Padrão: Absorption
        absorption = self._detect_absorption(recent_trades)
        if absorption:
            detected_patterns.append(absorption)
            
        # Padrão: Momentum
        momentum = self._detect_momentum_pattern(recent_trades)
        if momentum:
            detected_patterns.append(momentum)
            
        # Padrão: Exhaustion
        exhaustion = self._detect_exhaustion(recent_trades)
        if exhaustion:
            detected_patterns.append(exhaustion)
            
        return detected_patterns
        
    def _detect_sweep(self, trades: List[Dict]) -> Optional[Dict]:
        """Detecta padrão de sweep (limpeza de níveis)"""
        if len(trades) < 10:
            return None
            
        # Verificar trades consecutivos no mesmo lado com preços crescentes/decrescentes
        last_10 = trades[-10:]
        
        # Agrupar por direção
        buy_streak = 0
        sell_streak = 0
        
        for i in range(1, len(last_10)):
            if last_10[i].get('side') == 'buy' and last_10[i-1].get('side') == 'buy':
                if last_10[i].get('price', 0) >= last_10[i-1].get('price', 0):
                    buy_streak += 1
            elif last_10[i].get('side') == 'sell' and last_10[i-1].get('side') == 'sell':
                if last_10[i].get('price', 0) <= last_10[i-1].get('price', 0):
                    sell_streak += 1
                    
        if buy_streak >= 4:
            return {
                'type': 'buy_sweep',
                'confidence': min(buy_streak * 0.2, 1.0),
                'direction': 'bullish',
                'trades_involved': buy_streak + 1
            }
        elif sell_streak >= 4:
            return {
                'type': 'sell_sweep',
                'confidence': min(sell_streak * 0.2, 1.0),
                'direction': 'bearish',
                'trades_involved': sell_streak + 1
            }
            
        return None
        
    def _detect_iceberg_pattern(self, trades: List[Dict]) -> Optional[Dict]:
        """Detecta padrão iceberg na fita"""
        if len(trades) < 20:
            return None
            
        # Procurar execuções repetidas no mesmo preço com volumes similares
        price_groups = defaultdict(list)
        
        for trade in trades[-50:]:
            price = trade.get('price', 0)
            price_groups[price].append(trade)
            
        for price, group in price_groups.items():
            if len(group) >= 5:  # Múltiplas execuções
                volumes = [t.get('volume', 0) for t in group]
                avg_vol = np.mean(volumes)
                std_vol = np.std(volumes)
                
                if avg_vol > 0 and std_vol / avg_vol < 0.2:  # Volumes similares
                    return {
                        'type': 'iceberg_execution',
                        'confidence': min(len(group) * 0.15, 1.0),
                        'price_level': price,
                        'avg_volume': avg_vol,
                        'executions': len(group)
                    }
                    
        return None
        
    def _detect_absorption(self, trades: List[Dict]) -> Optional[Dict]:
        """Detecta absorção de pressão na fita"""
        if len(trades) < 20:
            return None
            
        # Procurar alto volume sem movimento de preço
        last_20 = trades[-20:]
        
        total_volume = sum(t.get('volume', 0) for t in last_20)
        avg_volume_per_trade = total_volume / len(last_20) if last_20 else 0
        
        # Verificar movimento de preço
        prices = [t.get('price', 0) for t in last_20]
        price_range = max(prices) - min(prices) if prices else 0
        avg_price = np.mean(prices) if prices else 1
        
        if avg_price > 0:
            price_movement_pct = (price_range / avg_price) * 100
            
            # Alto volume com pouco movimento = absorção
            if total_volume > avg_volume_per_trade * 30 and price_movement_pct < 0.1:
                # Determinar lado da absorção
                buy_volume = sum(t.get('volume', 0) for t in last_20 if t.get('side') == 'buy')
                sell_volume = sum(t.get('volume', 0) for t in last_20 if t.get('side') == 'sell')
                
                if buy_volume > sell_volume * 1.5:
                    direction = 'bullish_absorption'
                elif sell_volume > buy_volume * 1.5:
                    direction = 'bearish_absorption'
                else:
                    direction = 'neutral_absorption'
                    
                return {
                    'type': 'absorption',
                    'direction': direction,
                    'confidence': 0.7,
                    'volume_absorbed': total_volume,
                    'price_range_pct': price_movement_pct
                }
                
        return None
        
    def _detect_momentum_pattern(self, trades: List[Dict]) -> Optional[Dict]:
        """Detecta momentum na fita"""
        if len(trades) < 15:
            return None
            
        last_15 = trades[-15:]
        
        # Calcular direção dominante
        buy_count = sum(1 for t in last_15 if t.get('side') == 'buy')
        sell_count = len(last_15) - buy_count
        
        # Calcular momentum de preço
        prices = [t.get('price', 0) for t in last_15]
        if len(prices) >= 2:
            price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            
            # Momentum forte = movimentos consistentes na mesma direção
            positive_moves = sum(1 for c in price_changes if c > 0)
            negative_moves = sum(1 for c in price_changes if c < 0)
            
            if positive_moves > len(price_changes) * 0.7 and buy_count > sell_count:
                return {
                    'type': 'strong_buy_momentum',
                    'confidence': min((positive_moves / len(price_changes)), 1.0),
                    'direction': 'bullish',
                    'price_moves': positive_moves,
                    'total_moves': len(price_changes)
                }
            elif negative_moves > len(price_changes) * 0.7 and sell_count > buy_count:
                return {
                    'type': 'strong_sell_momentum',
                    'confidence': min((negative_moves / len(price_changes)), 1.0),
                    'direction': 'bearish',
                    'price_moves': negative_moves,
                    'total_moves': len(price_changes)
                }
                
        return None
        
    def _detect_exhaustion(self, trades: List[Dict]) -> Optional[Dict]:
        """Detecta exaustão do movimento"""
        if len(trades) < 30:
            return None
            
        # Dividir em duas metades
        first_half = trades[-30:-15]
        second_half = trades[-15:]
        
        # Comparar velocidade e volume
        first_volume = sum(t.get('volume', 0) for t in first_half)
        second_volume = sum(t.get('volume', 0) for t in second_half)
        
        first_speed = len(first_half) / 15  # trades por segundo aproximado
        second_speed = len(second_half) / 15
        
        # Exhaustion = redução de velocidade e volume
        if second_volume < first_volume * 0.5 and second_speed < first_speed * 0.5:
            # Determinar direção da exaustão
            recent_prices = [t.get('price', 0) for t in second_half]
            if recent_prices:
                price_trend = recent_prices[-1] - recent_prices[0]
                
                return {
                    'type': 'exhaustion',
                    'direction': 'bullish_exhaustion' if price_trend > 0 else 'bearish_exhaustion',
                    'confidence': 0.6,
                    'volume_reduction': 1 - (second_volume / first_volume),
                    'speed_reduction': 1 - (second_speed / first_speed)
                }
                
        return None


class TapeMomentumTracker:
    """Rastreador de momentum na fita"""
    
    def __init__(self, momentum_periods: List[int] = [10, 30, 60]):
        self.logger = logging.getLogger(f"{__name__}.MomentumTracker")
        self.periods = momentum_periods
        self.momentum_history = {period: deque(maxlen=period*2) for period in periods}
        
    def update_momentum(self, trades: List[Dict]) -> Dict:
        """Atualiza cálculo de momentum"""
        momentum = {
            'short_term': {},
            'medium_term': {},
            'long_term': {},
            'divergences': []
        }
        
        if not trades:
            return momentum
            
        for period in self.periods:
            if len(trades) < period:
                continue
                
            recent = trades[-period:]
            
            # Momentum de preço
            price_momentum = self._calculate_price_momentum(recent)
            
            # Momentum de volume
            volume_momentum = self._calculate_volume_momentum(recent)
            
            # Momentum direcional
            directional_momentum = self._calculate_directional_momentum(recent)
            
            # Categorizar por período
            if period <= 10:
                category = 'short_term'
            elif period <= 30:
                category = 'medium_term'
            else:
                category = 'long_term'
                
            momentum[category] = {
                'price': price_momentum,
                'volume': volume_momentum,
                'directional': directional_momentum,
                'period': period
            }
            
            # Armazenar histórico
            self.momentum_history[period].append({
                'timestamp': time.time(),
                'price': price_momentum,
                'volume': volume_momentum,
                'directional': directional_momentum
            })
            
        # Detectar divergências
        momentum['divergences'] = self._detect_divergences(momentum)
        
        return momentum
        
    def _calculate_price_momentum(self, trades: List[Dict]) -> float:
        """Calcula momentum de preço"""
        if len(trades) < 2:
            return 0.0
            
        prices = [t.get('price', 0) for t in trades]
        
        # Regressão linear para tendência
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # Normalizar pelo preço médio
        avg_price = np.mean(prices)
        if avg_price > 0:
            return slope / avg_price * 100  # Percentual
            
        return 0.0
        
    def _calculate_volume_momentum(self, trades: List[Dict]) -> float:
        """Calcula momentum de volume"""
        if len(trades) < 2:
            return 0.0
            
        volumes = [t.get('volume', 0) for t in trades]
        
        # Comparar primeira e segunda metade
        mid = len(volumes) // 2
        first_half_avg = np.mean(volumes[:mid])
        second_half_avg = np.mean(volumes[mid:])
        
        if first_half_avg > 0:
            return (second_half_avg - first_half_avg) / first_half_avg
            
        return 0.0
        
    def _calculate_directional_momentum(self, trades: List[Dict]) -> float:
        """Calcula momentum direcional (compras vs vendas)"""
        buy_volume = sum(t.get('volume', 0) for t in trades if t.get('side') == 'buy')
        sell_volume = sum(t.get('volume', 0) for t in trades if t.get('side') == 'sell')
        
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            return (buy_volume - sell_volume) / total_volume
            
        return 0.0
        
    def _detect_divergences(self, momentum: Dict) -> List[Dict]:
        """Detecta divergências entre diferentes períodos"""
        divergences = []
        
        # Comparar curto vs médio prazo
        if momentum['short_term'] and momentum['medium_term']:
            short_price = momentum['short_term'].get('price', 0)
            medium_price = momentum['medium_term'].get('price', 0)
            
            # Divergência: curto prazo oposto ao médio prazo
            if short_price * medium_price < 0 and abs(short_price) > 0.5 and abs(medium_price) > 0.5:
                divergences.append({
                    'type': 'price_divergence',
                    'description': 'Short-term vs medium-term price momentum',
                    'strength': abs(short_price - medium_price)
                })
                
        return divergences


class TapeReadingAgent(FlowAwareBaseAgent):
    """Agente especializado em tape reading"""
    
    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            'min_pattern_confidence': 0.5,
            'speed_weight': 0.3,
            'pattern_weight': 0.4,
            'momentum_weight': 0.3,
            'trade_buffer_size': 1000
        }
        if config:
            default_config.update(config)
            
        super().__init__('tape_reading', default_config)
        
        # Analisadores especializados
        self.speed_analyzer = TapeSpeedAnalyzer()
        self.pattern_detector = TapePatternDetector()
        self.momentum_tracker = TapeMomentumTracker()
        
        # Buffer de trades
        self.trade_buffer = deque(maxlen=self.config['trade_buffer_size'])
        self.signal_history = deque(maxlen=100)
        
        self.logger.info("TapeReadingAgent inicializado")
        
    def generate_signal_with_flow(self, price_state: Dict, flow_state: Dict) -> Dict:
        """Gera sinal baseado em tape reading"""
        # Obter trades recentes
        recent_trades = flow_state.get('recent_trades', [])
        
        if not recent_trades:
            return {
                'action': 'hold',
                'confidence': 0.0,
                'metadata': {'reason': 'no_trade_data'}
            }
            
        # Atualizar buffer e analisadores
        for trade in recent_trades:
            if trade not in self.trade_buffer:
                self.trade_buffer.append(trade)
                self.speed_analyzer.update(trade)
                
        # Análise de velocidade
        speed_analysis = self.speed_analyzer.analyze_speed()
        speed_patterns = self.speed_analyzer.detect_speed_patterns()
        
        # Detectar padrões
        tape_patterns = self.pattern_detector.detect_patterns(list(self.trade_buffer)[-100:])
        
        # Análise de momentum
        momentum_analysis = self.momentum_tracker.update_momentum(list(self.trade_buffer))
        
        # Gerar sinal baseado nas análises
        signal = self._generate_tape_signal(
            speed_analysis,
            speed_patterns,
            tape_patterns,
            momentum_analysis,
            price_state
        )
        
        # Armazenar no histórico
        self.signal_history.append({
            'timestamp': time.time(),
            'signal': signal,
            'speed': speed_analysis,
            'patterns': tape_patterns,
            'momentum': momentum_analysis
        })
        
        return signal
        
    def _generate_tape_signal(self, speed: Dict, speed_patterns: List[Dict],
                            tape_patterns: List[Dict], momentum: Dict,
                            price_state: Dict) -> Dict:
        """Gera sinal baseado em análise da fita"""
        signal = {
            'action': 'hold',
            'confidence': 0,
            'metadata': {
                'tape_speed': speed.get('speeds', {}).get('60s', 0),
                'patterns_detected': len(tape_patterns),
                'momentum_score': momentum.get('short_term', {}).get('directional', 0)
            }
        }
        
        # Avaliar cada componente
        components = []
        
        # 1. Componente de velocidade
        speed_signal = self._evaluate_speed_component(speed, speed_patterns)
        if speed_signal['confidence'] > 0:
            components.append(speed_signal)
            
        # 2. Componente de padrões
        pattern_signal = self._evaluate_pattern_component(tape_patterns)
        if pattern_signal['confidence'] > 0:
            components.append(pattern_signal)
            
        # 3. Componente de momentum
        momentum_signal = self._evaluate_momentum_component(momentum)
        if momentum_signal['confidence'] > 0:
            components.append(momentum_signal)
            
        # Combinar componentes
        if components:
            # Votação ponderada
            weighted_actions = defaultdict(float)
            total_weight = 0
            
            weights = {
                'speed': self.config['speed_weight'],
                'pattern': self.config['pattern_weight'],
                'momentum': self.config['momentum_weight']
            }
            
            for comp in components:
                weight = weights.get(comp['source'], 0.33)
                weighted_actions[comp['action']] += comp['confidence'] * weight
                total_weight += weight
                
            # Ação com maior peso
            if weighted_actions:
                best_action = max(weighted_actions.items(), key=lambda x: x[1])
                signal['action'] = best_action[0]
                signal['confidence'] = best_action[1] / total_weight if total_weight > 0 else 0
                
                # Adicionar detalhes dos componentes
                signal['metadata']['components'] = [
                    {
                        'source': comp['source'],
                        'action': comp['action'],
                        'confidence': comp['confidence'],
                        'reason': comp.get('reason', '')
                    }
                    for comp in components
                ]
                
        return signal
        
    def _evaluate_speed_component(self, speed: Dict, patterns: List[Dict]) -> Dict:
        """Avalia componente de velocidade"""
        component = {
            'source': 'speed',
            'action': 'hold',
            'confidence': 0,
            'reason': ''
        }
        
        # Verificar aceleração súbita
        for pattern in patterns:
            if pattern['type'] == 'sudden_acceleration':
                component['action'] = 'buy'  # Aceleração geralmente bullish
                component['confidence'] = pattern['confidence'] * 0.8
                component['reason'] = 'sudden_acceleration'
                break
            elif pattern['type'] == 'high_intensity':
                # Alta intensidade pode indicar movimento importante
                component['confidence'] = pattern['confidence'] * 0.6
                component['reason'] = 'high_intensity_trading'
                
        # Verificar velocidade absoluta
        current_speed = speed.get('speeds', {}).get('60s', 0)
        avg_speed = speed.get('speeds', {}).get('300s', 0)
        
        if avg_speed > 0 and current_speed > avg_speed * 1.5:
            if component['confidence'] < 0.5:
                component['confidence'] = 0.5
                component['reason'] = 'above_average_speed'
                
        return component
        
    def _evaluate_pattern_component(self, patterns: List[Dict]) -> Dict:
        """Avalia componente de padrões"""
        component = {
            'source': 'pattern',
            'action': 'hold',
            'confidence': 0,
            'reason': ''
        }
        
        if not patterns:
            return component
            
        # Priorizar padrões por tipo
        priority_order = [
            'buy_sweep', 'sell_sweep',
            'strong_buy_momentum', 'strong_sell_momentum',
            'absorption', 'iceberg_execution',
            'exhaustion'
        ]
        
        # Encontrar padrão de maior prioridade
        for pattern_type in priority_order:
            for pattern in patterns:
                if pattern.get('type') == pattern_type:
                    # Mapear padrão para ação
                    if pattern_type in ['buy_sweep', 'strong_buy_momentum']:
                        component['action'] = 'buy'
                    elif pattern_type in ['sell_sweep', 'strong_sell_momentum']:
                        component['action'] = 'sell'
                    elif pattern_type == 'absorption':
                        if 'bullish' in pattern.get('direction', ''):
                            component['action'] = 'buy'
                        elif 'bearish' in pattern.get('direction', ''):
                            component['action'] = 'sell'
                    elif pattern_type == 'exhaustion':
                        # Exhaustion sugere possível reversão
                        if 'bullish' in pattern.get('direction', ''):
                            component['action'] = 'sell'  # Bullish exhaustion = potential reversal down
                        elif 'bearish' in pattern.get('direction', ''):
                            component['action'] = 'buy'   # Bearish exhaustion = potential reversal up
                            
                    component['confidence'] = pattern.get('confidence', 0.5)
                    component['reason'] = pattern_type
                    
                    return component
                    
        return component
        
    def _evaluate_momentum_component(self, momentum: Dict) -> Dict:
        """Avalia componente de momentum"""
        component = {
            'source': 'momentum',
            'action': 'hold',
            'confidence': 0,
            'reason': ''
        }
        
        # Verificar momentum de curto prazo
        short_term = momentum.get('short_term', {})
        
        if not short_term:
            return component
            
        price_momentum = short_term.get('price', 0)
        directional_momentum = short_term.get('directional', 0)
        
        # Sinal forte: momentum de preço e direcional alinhados
        if price_momentum > 0.5 and directional_momentum > 0.3:
            component['action'] = 'buy'
            component['confidence'] = min((price_momentum + directional_momentum) / 2, 1.0)
            component['reason'] = 'strong_buy_momentum'
        elif price_momentum < -0.5 and directional_momentum < -0.3:
            component['action'] = 'sell'
            component['confidence'] = min((abs(price_momentum) + abs(directional_momentum)) / 2, 1.0)
            component['reason'] = 'strong_sell_momentum'
            
        # Verificar divergências
        divergences = momentum.get('divergences', [])
        if divergences and component['confidence'] > 0:
            # Divergências reduzem confiança
            component['confidence'] *= 0.7
            component['reason'] += '_with_divergence'
            
        return component
        
    def get_tape_statistics(self) -> Dict:
        """Retorna estatísticas da leitura de fita"""
        if not self.signal_history:
            return {}
            
        recent = list(self.signal_history)[-50:]
        
        # Calcular estatísticas
        stats = {
            'avg_tape_speed': np.mean([
                s['speed'].get('speeds', {}).get('60s', 0) 
                for s in recent
            ]),
            'patterns_per_signal': np.mean([
                s['metadata'].get('patterns_detected', 0)
                for s in recent
            ]),
            'momentum_distribution': {
                'positive': sum(1 for s in recent if s['momentum'].get('short_term', {}).get('directional', 0) > 0),
                'negative': sum(1 for s in recent if s['momentum'].get('short_term', {}).get('directional', 0) < 0),
                'neutral': sum(1 for s in recent if abs(s['momentum'].get('short_term', {}).get('directional', 0)) <= 0.1)
            },
            'action_distribution': {
                'buy': sum(1 for s in recent if s['signal']['action'] == 'buy'),
                'sell': sum(1 for s in recent if s['signal']['action'] == 'sell'),
                'hold': sum(1 for s in recent if s['signal']['action'] == 'hold')
            }
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
        'speed_weight': 0.3,
        'pattern_weight': 0.4,
        'momentum_weight': 0.3
    }
    
    agent = TapeReadingAgent(config)
    
    # Simular trades
    current_time = time.time()
    trades = []
    
    # Simular sweep de compra
    for i in range(10):
        trades.append({
            'timestamp': current_time - (100 - i*2),
            'price': 5000 + i*0.5,
            'volume': 50 + i*5,
            'side': 'buy'
        })
        
    # Simular alguns trades de venda
    for i in range(5):
        trades.append({
            'timestamp': current_time - (80 - i*3),
            'price': 5005 - i*0.2,
            'volume': 30,
            'side': 'sell'
        })
        
    # Mais trades de compra (momentum)
    for i in range(8):
        trades.append({
            'timestamp': current_time - (60 - i*2),
            'price': 5004 + i*0.3,
            'volume': 60,
            'side': 'buy'
        })
        
    # Estado de fluxo
    flow_state = {
        'recent_trades': trades
    }
    
    # Estado de preço
    price_state = {
        'price': 5006.0,
        'prev_price': 5000.0
    }
    
    # Gerar sinal
    signal = agent.generate_signal_with_flow(price_state, flow_state)
    
    print(f"\nSinal gerado: {signal}")
    print(f"\nEstatísticas: {agent.get_tape_statistics()}")
    
    # Testar analisadores individualmente
    speed_analyzer = TapeSpeedAnalyzer()
    for trade in trades:
        speed_analyzer.update(trade)
    speed_analysis = speed_analyzer.analyze_speed()
    print(f"\nAnálise de velocidade: {speed_analysis}")
    
    pattern_detector = TapePatternDetector()
    patterns = pattern_detector.detect_patterns(trades)
    print(f"\nPadrões detectados: {patterns}")
    
    agent.shutdown()