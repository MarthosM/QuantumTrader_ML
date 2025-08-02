"""
Liquidity Agent - HMARL Fase 1 Semana 3
Agente especializado em análise de liquidez e profundidade de mercado
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, List, Tuple
from collections import deque
from datetime import datetime, timedelta
import time

from src.agents.flow_aware_base_agent import FlowAwareBaseAgent


class LiquidityDepthAnalyzer:
    """Analisador de profundidade de liquidez"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DepthAnalyzer")
        
    def analyze_book_depth(self, orderbook: Dict) -> Dict:
        """Analisa profundidade do livro de ofertas"""
        analysis = {
            'bid_depth': 0,
            'ask_depth': 0,
            'total_depth': 0,
            'depth_imbalance': 0,
            'weighted_mid_price': 0,
            'liquidity_score': 0
        }
        
        # Extrair níveis de bid e ask
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return analysis
            
        # Calcular profundidade total
        bid_depth = sum(level.get('volume', 0) for level in bids[:10])  # Top 10 níveis
        ask_depth = sum(level.get('volume', 0) for level in asks[:10])
        
        analysis['bid_depth'] = bid_depth
        analysis['ask_depth'] = ask_depth
        analysis['total_depth'] = bid_depth + ask_depth
        
        # Calcular imbalance
        if analysis['total_depth'] > 0:
            analysis['depth_imbalance'] = (bid_depth - ask_depth) / analysis['total_depth']
            
        # Calcular preço médio ponderado
        if bids and asks:
            best_bid = bids[0].get('price', 0)
            best_ask = asks[0].get('price', 0)
            
            if best_bid > 0 and best_ask > 0:
                analysis['weighted_mid_price'] = (
                    best_bid * ask_depth + best_ask * bid_depth
                ) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else (best_bid + best_ask) / 2
                
        # Score de liquidez (0-1)
        # Baseado em profundidade total e spread
        if bids and asks:
            spread = asks[0].get('price', 0) - bids[0].get('price', 0)
            mid_price = (asks[0].get('price', 0) + bids[0].get('price', 0)) / 2
            
            if mid_price > 0:
                spread_bps = (spread / mid_price) * 10000  # basis points
                
                # Score alto = muita liquidez e spread pequeno
                depth_score = min(analysis['total_depth'] / 1000, 1.0)  # Normalizar
                spread_score = max(0, 1 - spread_bps / 50)  # 50 bps = score 0
                
                analysis['liquidity_score'] = (depth_score * 0.7 + spread_score * 0.3)
                
        return analysis
        
    def detect_liquidity_levels(self, orderbook: Dict, price_history: List[float]) -> Dict:
        """Detecta níveis importantes de liquidez"""
        levels = {
            'major_bid_levels': [],
            'major_ask_levels': [],
            'liquidity_gaps': [],
            'concentration_zones': []
        }
        
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        # Detectar níveis com grande volume
        if bids:
            avg_bid_volume = np.mean([b.get('volume', 0) for b in bids[:20]])
            for bid in bids[:20]:
                if bid.get('volume', 0) > avg_bid_volume * 2:
                    levels['major_bid_levels'].append({
                        'price': bid['price'],
                        'volume': bid['volume'],
                        'significance': bid['volume'] / avg_bid_volume
                    })
                    
        if asks:
            avg_ask_volume = np.mean([a.get('volume', 0) for a in asks[:20]])
            for ask in asks[:20]:
                if ask.get('volume', 0) > avg_ask_volume * 2:
                    levels['major_ask_levels'].append({
                        'price': ask['price'],
                        'volume': ask['volume'],
                        'significance': ask['volume'] / avg_ask_volume
                    })
                    
        # Detectar gaps de liquidez
        for i in range(1, min(len(bids), 10)):
            price_gap = bids[i-1]['price'] - bids[i]['price']
            expected_gap = bids[0]['price'] * 0.0001  # 0.01%
            
            if price_gap > expected_gap * 3:
                levels['liquidity_gaps'].append({
                    'side': 'bid',
                    'price_from': bids[i]['price'],
                    'price_to': bids[i-1]['price'],
                    'gap_size': price_gap
                })
                
        for i in range(1, min(len(asks), 10)):
            price_gap = asks[i]['price'] - asks[i-1]['price']
            expected_gap = asks[0]['price'] * 0.0001  # 0.01%
            
            if price_gap > expected_gap * 3:
                levels['liquidity_gaps'].append({
                    'side': 'ask',
                    'price_from': asks[i-1]['price'],
                    'price_to': asks[i]['price'],
                    'gap_size': price_gap
                })
                
        return levels


class HiddenLiquidityDetector:
    """Detector de liquidez oculta (icebergs, etc)"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.HiddenLiquidityDetector")
        self.execution_history = deque(maxlen=1000)
        
    def detect_iceberg_orders(self, trades: List[Dict], orderbook: Dict) -> Dict:
        """Detecta ordens iceberg"""
        detection = {
            'iceberg_probability': 0.0,
            'detected_levels': [],
            'refresh_patterns': []
        }
        
        if not trades:
            return detection
            
        # Agrupar trades por nível de preço
        price_groups = {}
        for trade in trades[-100:]:  # Últimos 100 trades
            price = trade.get('price', 0)
            if price not in price_groups:
                price_groups[price] = []
            price_groups[price].append(trade)
            
        # Detectar padrões de iceberg
        for price, group_trades in price_groups.items():
            if len(group_trades) < 3:
                continue
                
            # Verificar se há múltiplas execuções no mesmo preço
            volumes = [t.get('volume', 0) for t in group_trades]
            times = [t.get('timestamp', 0) for t in group_trades]
            
            # Padrão iceberg: volumes similares, intervalos regulares
            if len(volumes) >= 3:
                volume_std = np.std(volumes)
                avg_volume = np.mean(volumes)
                
                if avg_volume > 0 and volume_std / avg_volume < 0.2:  # Volumes similares
                    # Verificar intervalos de tempo
                    if len(times) >= 3:
                        intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
                        if intervals:
                            interval_std = np.std(intervals)
                            avg_interval = np.mean(intervals)
                            
                            if avg_interval > 0 and interval_std / avg_interval < 0.3:
                                # Provável iceberg detectado
                                detection['detected_levels'].append({
                                    'price': price,
                                    'avg_volume': avg_volume,
                                    'executions': len(group_trades),
                                    'confidence': 0.8
                                })
                                
        # Calcular probabilidade geral
        if detection['detected_levels']:
            detection['iceberg_probability'] = min(len(detection['detected_levels']) * 0.2, 1.0)
            
        return detection
        
    def estimate_hidden_liquidity(self, orderbook: Dict, trades: List[Dict]) -> Dict:
        """Estima liquidez oculta"""
        estimation = {
            'hidden_bid_liquidity': 0,
            'hidden_ask_liquidity': 0,
            'dark_pool_activity': 0,
            'hidden_ratio': 0
        }
        
        if not trades or not orderbook:
            return estimation
            
        # Analisar execuções vs liquidez visível
        recent_trades = trades[-50:]
        
        bid_executions = sum(
            t.get('volume', 0) for t in recent_trades 
            if t.get('side', '') == 'sell'  # Vendas executam contra bids
        )
        
        ask_executions = sum(
            t.get('volume', 0) for t in recent_trades 
            if t.get('side', '') == 'buy'  # Compras executam contra asks
        )
        
        # Liquidez visível
        visible_bid_liquidity = sum(
            b.get('volume', 0) for b in orderbook.get('bids', [])[:5]
        )
        visible_ask_liquidity = sum(
            a.get('volume', 0) for a in orderbook.get('asks', [])[:5]
        )
        
        # Se execuções > liquidez visível, há liquidez oculta
        if bid_executions > visible_bid_liquidity * 1.5:
            estimation['hidden_bid_liquidity'] = bid_executions - visible_bid_liquidity
            
        if ask_executions > visible_ask_liquidity * 1.5:
            estimation['hidden_ask_liquidity'] = ask_executions - visible_ask_liquidity
            
        # Estimar atividade dark pool (trades grandes sem impacto)
        large_trades = [
            t for t in recent_trades 
            if t.get('volume', 0) > np.mean([t.get('volume', 0) for t in recent_trades]) * 3
        ]
        
        if large_trades:
            # Verificar se houve pouco impacto no preço
            price_impact = np.std([t.get('price', 0) for t in large_trades])
            avg_price = np.mean([t.get('price', 0) for t in large_trades])
            
            if avg_price > 0 and price_impact / avg_price < 0.001:  # < 0.1% impacto
                estimation['dark_pool_activity'] = len(large_trades) / len(recent_trades)
                
        # Calcular ratio oculto/visível
        total_hidden = estimation['hidden_bid_liquidity'] + estimation['hidden_ask_liquidity']
        total_visible = visible_bid_liquidity + visible_ask_liquidity
        
        if total_visible > 0:
            estimation['hidden_ratio'] = total_hidden / total_visible
            
        return estimation


class LiquidityConsumptionTracker:
    """Rastreador de consumo de liquidez"""
    
    def __init__(self, window_size: int = 100):
        self.logger = logging.getLogger(f"{__name__}.ConsumptionTracker")
        self.consumption_history = deque(maxlen=window_size)
        self.replenishment_history = deque(maxlen=window_size)
        
    def track_consumption(self, orderbook_before: Dict, orderbook_after: Dict, 
                         trades: List[Dict]) -> Dict:
        """Rastreia consumo de liquidez"""
        metrics = {
            'consumption_rate': 0,
            'replenishment_speed': 0,
            'net_liquidity_change': 0,
            'aggressive_consumption': 0
        }
        
        if not orderbook_before or not orderbook_after:
            return metrics
            
        # Calcular liquidez total antes e depois
        liquidity_before = (
            sum(b.get('volume', 0) for b in orderbook_before.get('bids', [])[:10]) +
            sum(a.get('volume', 0) for a in orderbook_before.get('asks', [])[:10])
        )
        
        liquidity_after = (
            sum(b.get('volume', 0) for b in orderbook_after.get('bids', [])[:10]) +
            sum(a.get('volume', 0) for a in orderbook_after.get('asks', [])[:10])
        )
        
        # Volume executado
        executed_volume = sum(t.get('volume', 0) for t in trades)
        
        # Taxa de consumo
        if liquidity_before > 0:
            metrics['consumption_rate'] = executed_volume / liquidity_before
            
        # Mudança líquida
        metrics['net_liquidity_change'] = liquidity_after - liquidity_before + executed_volume
        
        # Velocidade de reposição
        if executed_volume > 0:
            replenishment = max(0, metrics['net_liquidity_change'])
            metrics['replenishment_speed'] = replenishment / executed_volume
            
        # Consumo agressivo (trades que moveram o preço)
        if trades:
            price_moves = []
            for i in range(1, len(trades)):
                price_change = abs(trades[i]['price'] - trades[i-1]['price'])
                price_moves.append(price_change)
                
            if price_moves:
                avg_move = np.mean(price_moves)
                aggressive_trades = sum(1 for move in price_moves if move > avg_move * 2)
                metrics['aggressive_consumption'] = aggressive_trades / len(trades)
                
        # Armazenar histórico
        self.consumption_history.append({
            'timestamp': time.time(),
            'consumption_rate': metrics['consumption_rate'],
            'executed_volume': executed_volume
        })
        
        return metrics
        
    def get_consumption_trend(self) -> Dict:
        """Analisa tendência de consumo de liquidez"""
        if len(self.consumption_history) < 10:
            return {'trend': 'neutral', 'strength': 0}
            
        recent_rates = [h['consumption_rate'] for h in list(self.consumption_history)[-20:]]
        
        # Calcular tendência
        if len(recent_rates) >= 2:
            trend_slope = np.polyfit(range(len(recent_rates)), recent_rates, 1)[0]
            
            if trend_slope > 0.001:
                return {'trend': 'increasing', 'strength': min(abs(trend_slope) * 100, 1.0)}
            elif trend_slope < -0.001:
                return {'trend': 'decreasing', 'strength': min(abs(trend_slope) * 100, 1.0)}
                
        return {'trend': 'neutral', 'strength': 0}


class LiquidityAgent(FlowAwareBaseAgent):
    """Agente especializado em análise de liquidez"""
    
    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            'min_liquidity_score': 0.3,
            'depth_weight': 0.4,
            'hidden_weight': 0.3,
            'consumption_weight': 0.3,
            'imbalance_threshold': 0.3
        }
        if config:
            default_config.update(config)
            
        super().__init__('liquidity_specialist', default_config)
        
        # Analisadores especializados
        self.depth_analyzer = LiquidityDepthAnalyzer()
        self.hidden_detector = HiddenLiquidityDetector()
        self.consumption_tracker = LiquidityConsumptionTracker()
        
        # Histórico de análises
        self.liquidity_history = deque(maxlen=1000)
        self.signal_performance = {}
        
        self.logger.info("LiquidityAgent inicializado")
        
    def generate_signal_with_flow(self, price_state: Dict, flow_state: Dict) -> Dict:
        """Gera sinal baseado em análise de liquidez"""
        # Obter dados do orderbook
        orderbook = flow_state.get('orderbook', {})
        trades = flow_state.get('recent_trades', [])
        
        if not orderbook:
            return {
                'action': 'hold',
                'confidence': 0.0,
                'metadata': {'reason': 'no_orderbook_data'}
            }
            
        # Análise de profundidade
        depth_analysis = self.depth_analyzer.analyze_book_depth(orderbook)
        
        # Detectar níveis de liquidez
        price_history = self._get_price_history(price_state)
        liquidity_levels = self.depth_analyzer.detect_liquidity_levels(
            orderbook, price_history
        )
        
        # Detectar liquidez oculta
        iceberg_detection = self.hidden_detector.detect_iceberg_orders(trades, orderbook)
        hidden_estimation = self.hidden_detector.estimate_hidden_liquidity(orderbook, trades)
        
        # Rastrear consumo
        consumption_metrics = self._get_consumption_metrics(flow_state)
        
        # Gerar sinal baseado em liquidez
        signal = self._generate_liquidity_signal(
            depth_analysis,
            liquidity_levels,
            iceberg_detection,
            hidden_estimation,
            consumption_metrics,
            price_state
        )
        
        # Armazenar análise
        self.liquidity_history.append({
            'timestamp': time.time(),
            'depth': depth_analysis,
            'levels': liquidity_levels,
            'hidden': hidden_estimation,
            'consumption': consumption_metrics,
            'signal': signal
        })
        
        return signal
        
    def _get_price_history(self, price_state: Dict) -> List[float]:
        """Obtém histórico de preços"""
        # Simplificado - em produção viria do data structure
        history = price_state.get('price_history', [])
        if not history and 'price' in price_state:
            history = [price_state['price']]
        return history
        
    def _get_consumption_metrics(self, flow_state: Dict) -> Dict:
        """Obtém métricas de consumo"""
        # Em produção, manteria orderbook anterior para comparação
        # Por ora, usar estimativas
        trades = flow_state.get('recent_trades', [])
        
        if trades:
            volume = sum(t.get('volume', 0) for t in trades[-10:])
            return {
                'consumption_rate': min(volume / 1000, 1.0),  # Normalizado
                'replenishment_speed': 0.5,  # Placeholder
                'net_liquidity_change': 0,
                'aggressive_consumption': 0.3
            }
            
        return self.consumption_tracker.get_consumption_trend()
        
    def _generate_liquidity_signal(self, depth: Dict, levels: Dict, 
                                  iceberg: Dict, hidden: Dict, 
                                  consumption: Dict, price_state: Dict) -> Dict:
        """Gera sinal baseado em análise de liquidez"""
        signal = {
            'action': 'hold',
            'confidence': 0,
            'metadata': {
                'liquidity_score': depth.get('liquidity_score', 0),
                'depth_imbalance': depth.get('depth_imbalance', 0),
                'iceberg_probability': iceberg.get('iceberg_probability', 0),
                'hidden_ratio': hidden.get('hidden_ratio', 0),
                'consumption_rate': consumption.get('consumption_rate', 0)
            }
        }
        
        # Verificar score mínimo de liquidez
        if depth['liquidity_score'] < self.config['min_liquidity_score']:
            signal['metadata']['reason'] = 'insufficient_liquidity'
            return signal
            
        # Análise de imbalance
        imbalance = depth.get('depth_imbalance', 0)
        
        # Sinal de compra: mais liquidez no bid (suporte)
        if imbalance > self.config['imbalance_threshold']:
            signal['action'] = 'buy'
            base_confidence = 0.5 + (imbalance * 0.3)
            
            # Ajustar por liquidez oculta
            if hidden.get('hidden_bid_liquidity', 0) > 0:
                base_confidence *= 1.2
                
            signal['confidence'] = min(base_confidence, 0.9)
            signal['metadata']['signal_type'] = 'liquidity_support'
            
        # Sinal de venda: mais liquidez no ask (resistência)
        elif imbalance < -self.config['imbalance_threshold']:
            signal['action'] = 'sell'
            base_confidence = 0.5 + (abs(imbalance) * 0.3)
            
            # Ajustar por liquidez oculta
            if hidden.get('hidden_ask_liquidity', 0) > 0:
                base_confidence *= 1.2
                
            signal['confidence'] = min(base_confidence, 0.9)
            signal['metadata']['signal_type'] = 'liquidity_resistance'
            
        # Verificar níveis importantes
        if signal['action'] != 'hold':
            signal = self._adjust_for_liquidity_levels(signal, levels, price_state)
            
        # Ajustar por consumo
        if consumption.get('consumption_rate', 0) > 0.7:
            # Alto consumo - mercado agressivo
            signal['confidence'] *= 1.1
            signal['metadata']['high_consumption'] = True
            
        # Penalizar se há muitos icebergs
        if iceberg.get('iceberg_probability', 0) > 0.5:
            signal['confidence'] *= 0.8
            signal['metadata']['iceberg_warning'] = True
            
        # Aplicar pesos finais
        weighted_confidence = (
            signal['confidence'] * self.config['depth_weight'] +
            depth['liquidity_score'] * self.config['hidden_weight'] +
            (1 - consumption.get('consumption_rate', 0)) * self.config['consumption_weight']
        )
        
        signal['confidence'] = min(weighted_confidence, 1.0)
        
        return signal
        
    def _adjust_for_liquidity_levels(self, signal: Dict, levels: Dict, 
                                   price_state: Dict) -> Dict:
        """Ajusta sinal baseado em níveis de liquidez"""
        current_price = price_state.get('price', 0)
        
        if not current_price:
            return signal
            
        # Verificar proximidade a níveis importantes
        if signal['action'] == 'buy':
            # Verificar suporte de liquidez
            major_bids = levels.get('major_bid_levels', [])
            for bid_level in major_bids:
                distance = (current_price - bid_level['price']) / current_price
                if 0 < distance < 0.002:  # Dentro de 0.2%
                    signal['confidence'] *= (1 + bid_level['significance'] * 0.1)
                    signal['metadata']['near_support'] = bid_level['price']
                    break
                    
        elif signal['action'] == 'sell':
            # Verificar resistência de liquidez
            major_asks = levels.get('major_ask_levels', [])
            for ask_level in major_asks:
                distance = (ask_level['price'] - current_price) / current_price
                if 0 < distance < 0.002:  # Dentro de 0.2%
                    signal['confidence'] *= (1 + ask_level['significance'] * 0.1)
                    signal['metadata']['near_resistance'] = ask_level['price']
                    break
                    
        # Verificar gaps de liquidez
        gaps = levels.get('liquidity_gaps', [])
        for gap in gaps:
            if gap['gap_size'] > current_price * 0.001:  # Gap > 0.1%
                signal['metadata']['liquidity_gap_warning'] = True
                signal['confidence'] *= 0.9
                
        return signal
        
    def analyze_liquidity_state(self) -> Dict:
        """Analisa estado geral da liquidez"""
        if len(self.liquidity_history) < 10:
            return {'state': 'insufficient_data'}
            
        recent_analyses = list(self.liquidity_history)[-50:]
        
        # Calcular médias
        avg_liquidity_score = np.mean([
            a['depth']['liquidity_score'] for a in recent_analyses
        ])
        
        avg_imbalance = np.mean([
            a['depth']['depth_imbalance'] for a in recent_analyses
        ])
        
        avg_consumption = np.mean([
            a['consumption'].get('consumption_rate', 0) for a in recent_analyses
        ])
        
        # Determinar estado
        state = 'normal'
        
        if avg_liquidity_score < 0.3:
            state = 'low_liquidity'
        elif avg_liquidity_score > 0.7:
            state = 'high_liquidity'
            
        if abs(avg_imbalance) > 0.5:
            state += '_imbalanced'
            
        if avg_consumption > 0.6:
            state += '_high_consumption'
            
        return {
            'state': state,
            'avg_liquidity_score': avg_liquidity_score,
            'avg_imbalance': avg_imbalance,
            'avg_consumption': avg_consumption,
            'samples': len(recent_analyses)
        }
        
    def get_liquidity_metrics(self) -> Dict:
        """Retorna métricas de liquidez"""
        if not self.liquidity_history:
            return {}
            
        recent = list(self.liquidity_history)[-100:]
        
        return {
            'current_liquidity_score': recent[-1]['depth']['liquidity_score'],
            'avg_liquidity_score': np.mean([r['depth']['liquidity_score'] for r in recent]),
            'max_imbalance': max([abs(r['depth']['depth_imbalance']) for r in recent]),
            'iceberg_detections': sum([
                1 for r in recent 
                if r['hidden'].get('iceberg_probability', 0) > 0.5
            ]),
            'high_consumption_events': sum([
                1 for r in recent 
                if r['consumption'].get('consumption_rate', 0) > 0.7
            ])
        }


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Criar agente
    config = {
        'min_liquidity_score': 0.3,
        'imbalance_threshold': 0.3
    }
    
    agent = LiquidityAgent(config)
    
    # Simular dados
    price_state = {
        'price': 5000.0,
        'price_history': [4995, 4998, 5000]
    }
    
    # Simular orderbook
    orderbook = {
        'bids': [
            {'price': 4999.5, 'volume': 100},
            {'price': 4999.0, 'volume': 150},
            {'price': 4998.5, 'volume': 200},
            {'price': 4998.0, 'volume': 80},
            {'price': 4997.5, 'volume': 120}
        ],
        'asks': [
            {'price': 5000.5, 'volume': 80},
            {'price': 5001.0, 'volume': 120},
            {'price': 5001.5, 'volume': 90},
            {'price': 5002.0, 'volume': 150},
            {'price': 5002.5, 'volume': 100}
        ]
    }
    
    # Simular trades recentes
    trades = [
        {'price': 5000.0, 'volume': 50, 'side': 'buy', 'timestamp': time.time() - 10},
        {'price': 5000.5, 'volume': 30, 'side': 'buy', 'timestamp': time.time() - 8},
        {'price': 5000.0, 'volume': 45, 'side': 'sell', 'timestamp': time.time() - 5},
        {'price': 4999.5, 'volume': 60, 'side': 'sell', 'timestamp': time.time() - 2}
    ]
    
    flow_state = {
        'orderbook': orderbook,
        'recent_trades': trades
    }
    
    # Gerar sinal
    signal = agent.generate_signal_with_flow(price_state, flow_state)
    
    print(f"\nSinal gerado: {signal}")
    print(f"\nEstado da liquidez: {agent.analyze_liquidity_state()}")
    print(f"\nMétricas: {agent.get_liquidity_metrics()}")
    
    # Testar analisadores individualmente
    depth_analyzer = LiquidityDepthAnalyzer()
    depth_analysis = depth_analyzer.analyze_book_depth(orderbook)
    print(f"\nAnálise de profundidade: {depth_analysis}")
    
    hidden_detector = HiddenLiquidityDetector()
    iceberg_detection = hidden_detector.detect_iceberg_orders(trades, orderbook)
    print(f"\nDetecção iceberg: {iceberg_detection}")
    
    agent.shutdown()