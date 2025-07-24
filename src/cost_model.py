# src/backtesting/cost_model.py
import numpy as np
from typing import Dict, Optional
import pandas as pd
from enum import Enum

class BacktestMode(Enum):
    """Modos de backtesting com diferentes níveis de realismo"""
    SIMPLE = "simple"
    REALISTIC = "realistic"
    CONSERVATIVE = "conservative"
    STRESS = "stress"

class CostModel:
    """Modelo de custos para backtesting realista"""
    
    def __init__(self, config):
        self.config = config
        
        # Custos base
        self.commission_per_contract = config.commission_per_contract
        self.slippage_ticks = config.slippage_ticks
        self.tick_value = config.tick_value
        
        # Modelos de custo por modo
        self.cost_multipliers = {
            BacktestMode.SIMPLE: {
                'commission': 1.0,
                'slippage': 0.0,
                'spread': 0.0
            },
            BacktestMode.REALISTIC: {
                'commission': 1.0,
                'slippage': 1.0,
                'spread': 1.0
            },
            BacktestMode.CONSERVATIVE: {
                'commission': 1.2,
                'slippage': 1.5,
                'spread': 1.5
            },
            BacktestMode.STRESS: {
                'commission': 1.5,
                'slippage': 2.0,
                'spread': 2.0
            }
        }
    
    def calculate_commission(self, quantity: int) -> float:
        """Calcula comissão da operação"""
        multiplier = self.cost_multipliers[self.config.mode]['commission']
        return self.commission_per_contract * quantity * multiplier
    
    def calculate_slippage(self, base_price: float, side: str,
                         market_data: pd.Series) -> float:
        """
        Calcula slippage em pontos
        
        Args:
            base_price: Preço base
            side: 'buy' ou 'sell'
            market_data: Dados de mercado atuais
            
        Returns:
            Slippage em pontos de preço
        """
        multiplier = self.cost_multipliers[self.config.mode]['slippage']
        
        # Slippage base em ticks
        base_slippage = self.slippage_ticks * self.tick_value * multiplier
        
        # Ajustar por condições de mercado
        if self.config.include_market_impact:
            # Volatilidade aumenta slippage
            if 'high' in market_data and 'low' in market_data:
                volatility = (market_data['high'] - market_data['low']) / market_data['close']
                volatility_factor = 1 + volatility * 10  # Até 10x em alta volatilidade
                base_slippage *= volatility_factor
            
            # Volume baixo aumenta slippage
            if 'volume' in market_data:
                if market_data['volume'] < 1000:
                    base_slippage *= 2.0
                elif market_data['volume'] < 5000:
                    base_slippage *= 1.5
        
        # Adicionar componente aleatório
        random_factor = np.random.uniform(0.8, 1.2)
        
        return base_slippage * random_factor
    
    def calculate_spread_cost(self, quantity: int, spread: float) -> float:
        """Calcula custo do spread"""
        multiplier = self.cost_multipliers[self.config.mode]['spread']
        return (spread / 2) * quantity * multiplier
    
    def calculate_total_cost(self, trade_params: Dict) -> Dict:
        """
        Calcula todos os custos de uma operação
        
        Args:
            trade_params: Parâmetros do trade
            
        Returns:
            Dicionário com breakdown dos custos
        """
        quantity = trade_params['quantity']
        
        # Comissão (entrada + saída)
        commission = self.calculate_commission(quantity) * 2
        
        # Slippage estimado
        slippage_entry = self.calculate_slippage(
            trade_params['entry_price'],
            trade_params['side'],
            trade_params.get('market_data', pd.Series())
        )
        
        # Assumir slippage similar na saída
        slippage_exit = slippage_entry
        
        total_slippage = (slippage_entry + slippage_exit) * quantity
        
        # Spread (se disponível)
        spread_cost = 0
        if 'spread' in trade_params:
            spread_cost = self.calculate_spread_cost(quantity, trade_params['spread'])
        
        # Total
        total_cost = commission + total_slippage + spread_cost
        
        return {
            'commission': commission,
            'slippage': total_slippage,
            'spread': spread_cost,
            'total': total_cost,
            'cost_per_contract': total_cost / quantity if quantity > 0 else 0
        }
    
    def adjust_for_time_of_day(self, base_cost: float, hour: int) -> float:
        """Ajusta custos por hora do dia"""
        # Custos maiores em horários de menor liquidez
        if hour < 9 or hour > 17:
            return base_cost * 1.5
        elif hour == 9 or hour == 17:  # Abertura/Fechamento
            return base_cost * 1.2
        else:
            return base_cost
    
    def estimate_breakeven_points(self, entry_price: float, 
                                quantity: int) -> Dict[str, float]:
        """Estima pontos de breakeven considerando custos"""
        # Custos totais estimados
        costs = self.calculate_total_cost({
            'quantity': quantity,
            'entry_price': entry_price,
            'side': 'buy'
        })
        
        # Breakeven por contrato
        cost_per_point = costs['total'] / quantity
        
        return {
            'long_breakeven': entry_price + cost_per_point,
            'short_breakeven': entry_price - cost_per_point,
            'total_cost': costs['total'],
            'cost_in_points': cost_per_point
        }