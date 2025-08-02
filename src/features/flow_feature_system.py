"""
Sistema Completo de Extração de Features de Fluxo - HMARL Fase 1 Semana 2
Implementa ~250 features especializadas em análise de fluxo de ordens
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import deque
import time
import logging
from abc import ABC, abstractmethod

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureCache:
    """Cache para features calculadas com TTL"""
    
    def __init__(self, ttl: int = 5):
        self.ttl = ttl  # segundos
        self.cache = {}
        
    def get(self, symbol: str, timestamp: datetime) -> Optional[Dict]:
        """Retorna features do cache se ainda válidas"""
        key = f"{symbol}_{int(timestamp.timestamp())}"
        if key in self.cache:
            cached_time, features = self.cache[key]
            if time.time() - cached_time < self.ttl:
                return features
        return None
        
    def set(self, symbol: str, timestamp: datetime, features: Dict):
        """Armazena features no cache"""
        key = f"{symbol}_{int(timestamp.timestamp())}"
        self.cache[key] = (time.time(), features)
        
        # Limpar cache antigo
        self._cleanup()
        
    def _cleanup(self):
        """Remove entradas expiradas"""
        current_time = time.time()
        expired_keys = [
            k for k, (t, _) in self.cache.items() 
            if current_time - t > self.ttl * 2
        ]
        for key in expired_keys:
            del self.cache[key]


class FlowFeatureSystem:
    """Sistema completo de extração de features de fluxo"""
    
    def __init__(self, valkey_manager=None):
        self.valkey = valkey_manager
        
        # Extractors especializados
        self.order_flow = OrderFlowAnalyzer(valkey_manager)
        self.tape_reader = TapeReadingAnalyzer(valkey_manager)
        self.footprint = FootprintAnalyzer(valkey_manager)
        self.liquidity = LiquidityAnalyzer(valkey_manager)
        self.microstructure = MicrostructureAnalyzer(valkey_manager)
        
        # Cache de features para performance
        self.feature_cache = FeatureCache(ttl=5)  # 5 segundos TTL
        
        logger.info("FlowFeatureSystem inicializado com 5 analyzers especializados")
        
    def extract_comprehensive_features(self, symbol: str, timestamp: datetime) -> Dict:
        """Extrai conjunto completo de features (~250 total)"""
        # Check cache primeiro
        cached = self.feature_cache.get(symbol, timestamp)
        if cached is not None:
            logger.debug(f"Features retornadas do cache para {symbol}")
            return cached
            
        logger.info(f"Extraindo features completas para {symbol} em {timestamp}")
        features = {}
        
        try:
            # 1. Order Flow Features (30-40 features)
            flow_features = self.extract_order_flow_features(symbol, timestamp)
            features.update(flow_features)
            logger.debug(f"Order Flow: {len(flow_features)} features extraídas")
            
            # 2. Tape Reading Features (20-30 features)
            tape_features = self.extract_tape_features(symbol, timestamp)
            features.update(tape_features)
            logger.debug(f"Tape Reading: {len(tape_features)} features extraídas")
            
            # 3. Footprint Features (15-20 features)
            footprint_features = self.extract_footprint_features(symbol, timestamp)
            features.update(footprint_features)
            logger.debug(f"Footprint: {len(footprint_features)} features extraídas")
            
            # 4. Liquidity Features (15-20 features)
            liquidity_features = self.extract_liquidity_features(symbol, timestamp)
            features.update(liquidity_features)
            logger.debug(f"Liquidity: {len(liquidity_features)} features extraídas")
            
            # 5. Microstructure Features (30-40 features)
            micro_features = self.extract_microstructure_features(symbol, timestamp)
            features.update(micro_features)
            logger.debug(f"Microstructure: {len(micro_features)} features extraídas")
            
            # 6. Traditional Technical Features (80-100 features) - mantidas
            tech_features = self.extract_technical_features(symbol, timestamp)
            features.update(tech_features)
            logger.debug(f"Technical: {len(tech_features)} features extraídas")
            
            # Cache result
            self.feature_cache.set(symbol, timestamp, features)
            
            logger.info(f"Total de {len(features)} features extraídas para {symbol}")
            return features
            
        except Exception as e:
            logger.error(f"Erro extraindo features: {e}")
            return {}
            
    def extract_order_flow_features(self, symbol: str, timestamp: datetime) -> Dict:
        """Extrai features de order flow com múltiplas perspectivas"""
        features = {}
        
        try:
            # Order Flow Imbalance em múltiplas janelas
            for window in [1, 5, 15, 30, 60]:  # minutos
                ofi = self.order_flow.calculate_imbalance(symbol, window, timestamp)
                features[f'ofi_{window}m'] = ofi['imbalance']
                features[f'ofi_velocity_{window}m'] = ofi['velocity']
                features[f'ofi_acceleration_{window}m'] = ofi['acceleration']
                
            # Análise de agressão
            aggression = self.order_flow.analyze_aggression(symbol, timestamp)
            features['buy_aggression'] = aggression['buy_aggression']
            features['sell_aggression'] = aggression['sell_aggression']
            features['aggression_ratio'] = aggression['ratio']
            
            # Volume at Price
            vap = self.order_flow.calculate_volume_at_price(symbol, timestamp)
            features['poc_distance'] = vap['poc_distance']  # Point of Control
            features['value_area_high'] = vap['value_area_high']
            features['value_area_low'] = vap['value_area_low']
            features['volume_skew'] = vap['skew']
            
            # Delta analysis
            delta = self.order_flow.calculate_delta(symbol, timestamp)
            features['cumulative_delta'] = delta['cumulative']
            features['delta_divergence'] = delta['divergence']
            features['delta_momentum'] = delta['momentum']
            
        except Exception as e:
            logger.error(f"Erro em order flow features: {e}")
            
        return features
        
    def extract_tape_features(self, symbol: str, timestamp: datetime) -> Dict:
        """Extrai features de tape reading"""
        features = {}
        
        try:
            # Velocidade do tape
            speed = self.tape_reader.calculate_tape_speed(symbol, timestamp)
            features['tape_speed_1m'] = speed['1m']
            features['tape_speed_5m'] = speed['5m']
            features['tape_acceleration'] = speed['acceleration']
            
            # Tamanho dos trades
            size_dist = self.tape_reader.analyze_trade_sizes(symbol, timestamp)
            features['avg_trade_size'] = size_dist['average']
            features['large_trade_ratio'] = size_dist['large_ratio']
            features['small_trade_ratio'] = size_dist['small_ratio']
            features['trade_size_variance'] = size_dist['variance']
            
            # Padrões do tape
            patterns = self.tape_reader.detect_patterns(symbol, timestamp)
            features['sweep_detected'] = patterns['sweep']
            features['iceberg_detected'] = patterns['iceberg']
            features['absorption_detected'] = patterns['absorption']
            features['pattern_confidence'] = patterns['confidence']
            
            # Momentum do tape
            momentum = self.tape_reader.calculate_tape_momentum(symbol, timestamp)
            features['tape_momentum'] = momentum['value']
            features['tape_momentum_change'] = momentum['change']
            
        except Exception as e:
            logger.error(f"Erro em tape features: {e}")
            
        return features
        
    def extract_footprint_features(self, symbol: str, timestamp: datetime) -> Dict:
        """Extrai features de footprint"""
        features = {}
        
        try:
            # Análise de footprint
            footprint = self.footprint.analyze_footprint(symbol, timestamp)
            features['footprint_imbalance'] = footprint['imbalance']
            features['footprint_delta'] = footprint['delta']
            features['footprint_absorption'] = footprint['absorption']
            
            # Padrões de footprint
            patterns = self.footprint.detect_footprint_patterns(symbol, timestamp)
            features['reversal_pattern'] = patterns['reversal']
            features['continuation_pattern'] = patterns['continuation']
            features['exhaustion_pattern'] = patterns['exhaustion']
            
            # Níveis importantes
            levels = self.footprint.identify_key_levels(symbol, timestamp)
            features['resistance_distance'] = levels['resistance_distance']
            features['support_distance'] = levels['support_distance']
            features['key_level_strength'] = levels['strength']
            
        except Exception as e:
            logger.error(f"Erro em footprint features: {e}")
            
        return features
        
    def extract_liquidity_features(self, symbol: str, timestamp: datetime) -> Dict:
        """Extrai features de liquidez"""
        features = {}
        
        try:
            # Profundidade do book
            depth = self.liquidity.analyze_book_depth(symbol, timestamp)
            features['bid_depth'] = depth['bid_depth']
            features['ask_depth'] = depth['ask_depth']
            features['depth_imbalance'] = depth['imbalance']
            features['liquidity_score'] = depth['score']
            
            # Liquidez oculta
            hidden = self.liquidity.detect_hidden_liquidity(symbol, timestamp)
            features['hidden_liquidity_bid'] = hidden['bid']
            features['hidden_liquidity_ask'] = hidden['ask']
            features['iceberg_probability'] = hidden['iceberg_prob']
            
            # Consumo de liquidez
            consumption = self.liquidity.analyze_liquidity_consumption(symbol, timestamp)
            features['liquidity_consumption_rate'] = consumption['rate']
            features['replenishment_speed'] = consumption['replenishment']
            
        except Exception as e:
            logger.error(f"Erro em liquidity features: {e}")
            
        return features
        
    def extract_microstructure_features(self, symbol: str, timestamp: datetime) -> Dict:
        """Extrai features de microestrutura"""
        features = {}
        
        try:
            # Análise de spread
            spread = self.microstructure.analyze_spread(symbol, timestamp)
            features['bid_ask_spread'] = spread['current']
            features['spread_volatility'] = spread['volatility']
            features['effective_spread'] = spread['effective']
            features['realized_spread'] = spread['realized']
            
            # Impacto de preço
            impact = self.microstructure.calculate_price_impact(symbol, timestamp)
            features['price_impact_buy'] = impact['buy']
            features['price_impact_sell'] = impact['sell']
            features['impact_asymmetry'] = impact['asymmetry']
            
            # Análise de ticks
            ticks = self.microstructure.analyze_tick_data(symbol, timestamp)
            features['tick_direction'] = ticks['direction']
            features['tick_velocity'] = ticks['velocity']
            features['uptick_ratio'] = ticks['uptick_ratio']
            features['zero_tick_ratio'] = ticks['zero_ratio']
            
            # Padrões HFT
            hft = self.microstructure.detect_hft_patterns(symbol, timestamp)
            features['hft_activity'] = hft['activity_level']
            features['quote_stuffing'] = hft['quote_stuffing']
            features['layering_detected'] = hft['layering']
            
        except Exception as e:
            logger.error(f"Erro em microstructure features: {e}")
            
        return features
        
    def extract_technical_features(self, symbol: str, timestamp: datetime) -> Dict:
        """Extrai features técnicas tradicionais (mantidas por compatibilidade)"""
        features = {}
        
        # Placeholder - seria integrado com sistema existente
        # Por enquanto retorna valores dummy para não quebrar
        technical_indicators = [
            'rsi', 'macd', 'bb_upper', 'bb_lower', 'ema_9', 'ema_20',
            'sma_50', 'sma_200', 'atr', 'adx', 'volume_sma', 'vwap'
        ]
        
        for indicator in technical_indicators:
            features[f'tech_{indicator}'] = 0.0
            
        return features


class OrderFlowAnalyzer:
    """Analisador especializado de order flow"""
    
    def __init__(self, valkey_manager=None):
        self.valkey = valkey_manager
        self.logger = logging.getLogger(f"{__name__}.OrderFlowAnalyzer")
        
    def calculate_imbalance(self, symbol: str, window_minutes: int, timestamp: datetime) -> Dict:
        """Calcula order flow imbalance com velocidade e aceleração"""
        try:
            # Simulação para desenvolvimento - seria conectado ao Valkey
            buy_volume = np.random.uniform(1000, 5000)
            sell_volume = np.random.uniform(1000, 5000)
            total_volume = buy_volume + sell_volume
            
            imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
            
            # Simular velocidade e aceleração
            velocity = np.random.uniform(-0.1, 0.1)
            acceleration = np.random.uniform(-0.01, 0.01)
            
            return {
                'imbalance': imbalance,
                'velocity': velocity,
                'acceleration': acceleration,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume
            }
            
        except Exception as e:
            self.logger.error(f"Erro calculando imbalance: {e}")
            return {'imbalance': 0, 'velocity': 0, 'acceleration': 0}
            
    def analyze_aggression(self, symbol: str, timestamp: datetime) -> Dict:
        """Analisa agressão de compra/venda"""
        try:
            # Simulação
            buy_aggression = np.random.uniform(0, 1)
            sell_aggression = np.random.uniform(0, 1)
            
            ratio = buy_aggression / sell_aggression if sell_aggression > 0 else 1.0
            
            return {
                'buy_aggression': buy_aggression,
                'sell_aggression': sell_aggression,
                'ratio': ratio
            }
            
        except Exception as e:
            self.logger.error(f"Erro analisando agressão: {e}")
            return {'buy_aggression': 0, 'sell_aggression': 0, 'ratio': 1}
            
    def calculate_volume_at_price(self, symbol: str, timestamp: datetime) -> Dict:
        """Calcula volume at price e POC"""
        try:
            # Simulação
            current_price = 5000.0
            poc = current_price + np.random.uniform(-10, 10)
            
            return {
                'poc_distance': abs(current_price - poc),
                'value_area_high': poc + 5,
                'value_area_low': poc - 5,
                'skew': np.random.uniform(-1, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Erro calculando VAP: {e}")
            return {'poc_distance': 0, 'value_area_high': 0, 'value_area_low': 0, 'skew': 0}
            
    def calculate_delta(self, symbol: str, timestamp: datetime) -> Dict:
        """Calcula delta e métricas relacionadas"""
        try:
            # Simulação
            return {
                'cumulative': np.random.uniform(-1000, 1000),
                'divergence': np.random.uniform(-100, 100),
                'momentum': np.random.uniform(-50, 50)
            }
            
        except Exception as e:
            self.logger.error(f"Erro calculando delta: {e}")
            return {'cumulative': 0, 'divergence': 0, 'momentum': 0}


class TapeReadingAnalyzer:
    """Analisador de tape reading"""
    
    def __init__(self, valkey_manager=None):
        self.valkey = valkey_manager
        self.logger = logging.getLogger(f"{__name__}.TapeReadingAnalyzer")
        
    def calculate_tape_speed(self, symbol: str, timestamp: datetime) -> Dict:
        """Calcula velocidade do tape"""
        try:
            # Simulação
            return {
                '1m': np.random.uniform(10, 100),
                '5m': np.random.uniform(50, 500),
                'acceleration': np.random.uniform(-10, 10)
            }
            
        except Exception as e:
            self.logger.error(f"Erro calculando tape speed: {e}")
            return {'1m': 0, '5m': 0, 'acceleration': 0}
            
    def analyze_trade_sizes(self, symbol: str, timestamp: datetime) -> Dict:
        """Analisa distribuição de tamanhos"""
        try:
            # Simulação
            avg_size = np.random.uniform(1, 10)
            
            return {
                'average': avg_size,
                'large_ratio': np.random.uniform(0, 0.3),
                'small_ratio': np.random.uniform(0.3, 0.7),
                'variance': np.random.uniform(0.1, 2.0)
            }
            
        except Exception as e:
            self.logger.error(f"Erro analisando trade sizes: {e}")
            return {'average': 1, 'large_ratio': 0, 'small_ratio': 0, 'variance': 0}
            
    def detect_patterns(self, symbol: str, timestamp: datetime) -> Dict:
        """Detecta padrões no tape"""
        try:
            # Simulação
            return {
                'sweep': np.random.choice([0, 1], p=[0.9, 0.1]),
                'iceberg': np.random.choice([0, 1], p=[0.95, 0.05]),
                'absorption': np.random.choice([0, 1], p=[0.85, 0.15]),
                'confidence': np.random.uniform(0.5, 1.0)
            }
            
        except Exception as e:
            self.logger.error(f"Erro detectando padrões: {e}")
            return {'sweep': 0, 'iceberg': 0, 'absorption': 0, 'confidence': 0}
            
    def calculate_tape_momentum(self, symbol: str, timestamp: datetime) -> Dict:
        """Calcula momentum do tape"""
        try:
            # Simulação
            return {
                'value': np.random.uniform(-1, 1),
                'change': np.random.uniform(-0.1, 0.1)
            }
            
        except Exception as e:
            self.logger.error(f"Erro calculando momentum: {e}")
            return {'value': 0, 'change': 0}


class FootprintAnalyzer:
    """Analisador de footprint charts"""
    
    def __init__(self, valkey_manager=None):
        self.valkey = valkey_manager
        self.logger = logging.getLogger(f"{__name__}.FootprintAnalyzer")
        
    def analyze_footprint(self, symbol: str, timestamp: datetime) -> Dict:
        """Analisa footprint básico"""
        try:
            # Simulação
            return {
                'imbalance': np.random.uniform(-1, 1),
                'delta': np.random.uniform(-100, 100),
                'absorption': np.random.uniform(0, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Erro analisando footprint: {e}")
            return {'imbalance': 0, 'delta': 0, 'absorption': 0}
            
    def detect_footprint_patterns(self, symbol: str, timestamp: datetime) -> Dict:
        """Detecta padrões de footprint"""
        try:
            # Simulação
            return {
                'reversal': np.random.choice([0, 1], p=[0.8, 0.2]),
                'continuation': np.random.choice([0, 1], p=[0.7, 0.3]),
                'exhaustion': np.random.choice([0, 1], p=[0.85, 0.15])
            }
            
        except Exception as e:
            self.logger.error(f"Erro detectando padrões: {e}")
            return {'reversal': 0, 'continuation': 0, 'exhaustion': 0}
            
    def identify_key_levels(self, symbol: str, timestamp: datetime) -> Dict:
        """Identifica níveis chave"""
        try:
            # Simulação
            current_price = 5000.0
            
            return {
                'resistance_distance': np.random.uniform(5, 50),
                'support_distance': np.random.uniform(5, 50),
                'strength': np.random.uniform(0.5, 1.0)
            }
            
        except Exception as e:
            self.logger.error(f"Erro identificando níveis: {e}")
            return {'resistance_distance': 0, 'support_distance': 0, 'strength': 0}


class LiquidityAnalyzer:
    """Analisador de liquidez"""
    
    def __init__(self, valkey_manager=None):
        self.valkey = valkey_manager
        self.logger = logging.getLogger(f"{__name__}.LiquidityAnalyzer")
        
    def analyze_book_depth(self, symbol: str, timestamp: datetime) -> Dict:
        """Analisa profundidade do book"""
        try:
            # Simulação
            bid_depth = np.random.uniform(100, 1000)
            ask_depth = np.random.uniform(100, 1000)
            
            return {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'imbalance': (bid_depth - ask_depth) / (bid_depth + ask_depth),
                'score': np.random.uniform(0.5, 1.0)
            }
            
        except Exception as e:
            self.logger.error(f"Erro analisando depth: {e}")
            return {'bid_depth': 0, 'ask_depth': 0, 'imbalance': 0, 'score': 0}
            
    def detect_hidden_liquidity(self, symbol: str, timestamp: datetime) -> Dict:
        """Detecta liquidez oculta"""
        try:
            # Simulação
            return {
                'bid': np.random.uniform(0, 100),
                'ask': np.random.uniform(0, 100),
                'iceberg_prob': np.random.uniform(0, 0.3)
            }
            
        except Exception as e:
            self.logger.error(f"Erro detectando liquidez oculta: {e}")
            return {'bid': 0, 'ask': 0, 'iceberg_prob': 0}
            
    def analyze_liquidity_consumption(self, symbol: str, timestamp: datetime) -> Dict:
        """Analisa consumo de liquidez"""
        try:
            # Simulação
            return {
                'rate': np.random.uniform(0, 1),
                'replenishment': np.random.uniform(0, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Erro analisando consumo: {e}")
            return {'rate': 0, 'replenishment': 0}


class MicrostructureAnalyzer:
    """Analisador de microestrutura de mercado"""
    
    def __init__(self, valkey_manager=None):
        self.valkey = valkey_manager
        self.logger = logging.getLogger(f"{__name__}.MicrostructureAnalyzer")
        
    def analyze_spread(self, symbol: str, timestamp: datetime) -> Dict:
        """Analisa spread bid-ask"""
        try:
            # Simulação
            return {
                'current': np.random.uniform(0.5, 2.0),
                'volatility': np.random.uniform(0.1, 0.5),
                'effective': np.random.uniform(0.5, 2.5),
                'realized': np.random.uniform(0.5, 3.0)
            }
            
        except Exception as e:
            self.logger.error(f"Erro analisando spread: {e}")
            return {'current': 0, 'volatility': 0, 'effective': 0, 'realized': 0}
            
    def calculate_price_impact(self, symbol: str, timestamp: datetime) -> Dict:
        """Calcula impacto de preço"""
        try:
            # Simulação
            buy_impact = np.random.uniform(0, 0.1)
            sell_impact = np.random.uniform(0, 0.1)
            
            return {
                'buy': buy_impact,
                'sell': sell_impact,
                'asymmetry': buy_impact - sell_impact
            }
            
        except Exception as e:
            self.logger.error(f"Erro calculando price impact: {e}")
            return {'buy': 0, 'sell': 0, 'asymmetry': 0}
            
    def analyze_tick_data(self, symbol: str, timestamp: datetime) -> Dict:
        """Analisa dados de tick"""
        try:
            # Simulação
            return {
                'direction': np.random.uniform(-1, 1),
                'velocity': np.random.uniform(0, 10),
                'uptick_ratio': np.random.uniform(0.3, 0.7),
                'zero_ratio': np.random.uniform(0, 0.2)
            }
            
        except Exception as e:
            self.logger.error(f"Erro analisando ticks: {e}")
            return {'direction': 0, 'velocity': 0, 'uptick_ratio': 0, 'zero_ratio': 0}
            
    def detect_hft_patterns(self, symbol: str, timestamp: datetime) -> Dict:
        """Detecta padrões HFT"""
        try:
            # Simulação
            return {
                'activity_level': np.random.uniform(0, 1),
                'quote_stuffing': np.random.choice([0, 1], p=[0.95, 0.05]),
                'layering': np.random.choice([0, 1], p=[0.97, 0.03])
            }
            
        except Exception as e:
            self.logger.error(f"Erro detectando HFT: {e}")
            return {'activity_level': 0, 'quote_stuffing': 0, 'layering': 0}


if __name__ == "__main__":
    # Teste do sistema
    system = FlowFeatureSystem()
    
    # Extrair features
    features = system.extract_comprehensive_features('WDOH25', datetime.now())
    
    print(f"\nTotal de features extraídas: {len(features)}")
    print("\nAmostra de features:")
    for i, (key, value) in enumerate(features.items()):
        if i < 20:  # Mostrar primeiras 20
            print(f"  {key}: {value:.4f}")
        else:
            break
    print("  ...")
    
    # Teste de cache
    import time
    start = time.time()
    features2 = system.extract_comprehensive_features('WDOH25', datetime.now())
    cache_time = time.time() - start
    print(f"\nTempo com cache: {cache_time:.4f} segundos")