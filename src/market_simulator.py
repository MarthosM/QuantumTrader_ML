# src/backtesting/market_simulator.py
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging

class MarketSimulator:
    """Simula condições realistas de mercado para backtesting"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Estado do mercado
        self.current_timestamp = None
        self.current_data = None
        self.market_state = 'normal'
        
        # Histórico para cálculos
        self.price_history = []
        self.volume_history = []
        
        # Parâmetros de simulação
        self.spread_model = SpreadModel()
        self.liquidity_model = LiquidityModel()
        self.volatility_model = VolatilityModel()
        
    def update(self, timestamp: datetime, market_data: pd.Series):
        """Atualiza estado do simulador"""
        self.current_timestamp = timestamp
        self.current_data = market_data
        
        # Atualizar histórico
        self.price_history.append(market_data['close'])
        self.volume_history.append(market_data.get('volume', 0))
        
        # Limitar tamanho do histórico
        if len(self.price_history) > 1000:
            self.price_history.pop(0)
            self.volume_history.pop(0)
        
        # Detectar estado do mercado
        self._detect_market_state()
        
        # Atualizar modelos
        self.spread_model.update(market_data)
        self.liquidity_model.update(market_data)
        self.volatility_model.update(self.price_history)
    
    def _detect_market_state(self):
        """Detecta estado atual do mercado"""
        if len(self.price_history) < 20:
            return
        
        # Volatilidade recente
        recent_returns = pd.Series(self.price_history[-20:]).pct_change().dropna()
        current_volatility = recent_returns.std()
        
        # Volume
        if len(self.volume_history) > 20:
            avg_volume = np.mean(self.volume_history[-20:])
            current_volume = self.volume_history[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        else:
            volume_ratio = 1
        
        # Determinar estado
        if current_volatility > 0.02:  # Alta volatilidade (>2%)
            self.market_state = 'high_volatility'
        elif volume_ratio < 0.5:  # Baixa liquidez
            self.market_state = 'low_liquidity'
        elif volume_ratio > 2.0:  # Alto volume
            self.market_state = 'high_activity'
        else:
            self.market_state = 'normal'
    
    def get_execution_price(self, side: str, size: int, 
                          base_price: float) -> Tuple[float, float]:
        """
        Calcula preço de execução realista
        
        Returns:
            Tuple (preço_execução, impacto_mercado)
        """
        # Spread base
        spread = self.spread_model.get_spread(self.market_state)
        
        # Impacto de mercado
        market_impact = self.liquidity_model.calculate_impact(
            size, self.current_data.get('volume', 1000)
        )
        
        # Ajustar por volatilidade
        volatility_adjustment = self.volatility_model.get_adjustment()
        
        # Calcular preço final
        if side == 'buy':
            execution_price = base_price + spread/2 + market_impact + volatility_adjustment
        else:
            execution_price = base_price - spread/2 - market_impact - volatility_adjustment
        
        total_impact = abs(execution_price - base_price)
        
        return execution_price, total_impact
    
    def can_execute_size(self, size: int) -> bool:
        """Verifica se o tamanho pode ser executado"""
        if self.current_data is None:
            return False
        
        current_volume = self.current_data.get('volume', 0)
        
        # Regra: não pode ser mais que 10% do volume
        return size <= current_volume * 0.1
    
    def get_market_conditions(self) -> Dict:
        """Retorna condições atuais do mercado"""
        return {
            'state': self.market_state,
            'spread': self.spread_model.current_spread,
            'liquidity': self.liquidity_model.current_liquidity,
            'volatility': self.volatility_model.current_volatility,
            'can_trade': self.market_state != 'low_liquidity'
        }


class SpreadModel:
    """Modelo de spread bid-ask"""
    
    def __init__(self):
        self.base_spread = 1.0  # 1 tick
        self.current_spread = self.base_spread
        
    def update(self, market_data: pd.Series):
        """Atualiza modelo de spread"""
        # Usar dados reais de spread se disponíveis
        if 'bid' in market_data and 'ask' in market_data:
            self.current_spread = market_data['ask'] - market_data['bid']
        else:
            # Estimar baseado em volatilidade
            if 'high' in market_data and 'low' in market_data:
                range_pct = (market_data['high'] - market_data['low']) / market_data['close']
                self.current_spread = max(self.base_spread, range_pct * 100)
    
    def get_spread(self, market_state: str) -> float:
        """Retorna spread ajustado pelo estado do mercado"""
        multipliers = {
            'normal': 1.0,
            'high_volatility': 2.0,
            'low_liquidity': 3.0,
            'high_activity': 0.8
        }
        
        return self.current_spread * multipliers.get(market_state, 1.0)


class LiquidityModel:
    """Modelo de liquidez e impacto de mercado"""
    
    def __init__(self):
        self.current_liquidity = 'normal'
        self.impact_factor = 0.0001  # 0.01% por contrato
        
    def update(self, market_data: pd.Series):
        """Atualiza estado de liquidez"""
        volume = market_data.get('volume', 0)
        
        if volume < 1000:
            self.current_liquidity = 'low'
        elif volume > 10000:
            self.current_liquidity = 'high'
        else:
            self.current_liquidity = 'normal'
    
    def calculate_impact(self, size: int, current_volume: float) -> float:
        """Calcula impacto de mercado da ordem"""
        if current_volume == 0:
            return size * self.impact_factor * 10  # Alto impacto
        
        # Impacto proporcional ao tamanho relativo
        size_ratio = size / current_volume
        
        # Modelo quadrático para grandes ordens
        if size_ratio > 0.05:  # Mais de 5% do volume
            impact = self.impact_factor * size * (1 + size_ratio ** 2)
        else:
            impact = self.impact_factor * size
        
        return impact


class VolatilityModel:
    """Modelo de volatilidade para ajustes de preço"""
    
    def __init__(self):
        self.current_volatility = 0.01  # 1%
        self.volatility_window = 20
        
    def update(self, price_history: list):
        """Atualiza volatilidade atual"""
        if len(price_history) < self.volatility_window:
            return
        
        # Calcular volatilidade realizada
        returns = pd.Series(price_history[-self.volatility_window:]).pct_change().dropna()
        self.current_volatility = returns.std()
    
    def get_adjustment(self) -> float:
        """Retorna ajuste de preço baseado em volatilidade"""
        # Quanto maior a volatilidade, maior o ajuste potencial
        return np.random.normal(0, self.current_volatility) * 100  # Convertido para pontos