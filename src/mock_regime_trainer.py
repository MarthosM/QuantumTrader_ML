"""
Mock do Regime Trainer para testes
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


class MockRegimeTrainer:
    """Mock do MarketRegimeTrainer para testes"""
    
    def __init__(self):
        self.models = {
            'regime_classifier': MockRegimeClassifier(),
            'trend': {'default': MockTrendModel()},
            'range': {'default': MockRangeModel()}
        }
        
    def analyze_market(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analisa mercado e retorna regime"""
        if market_data.empty:
            return {'regime': 'undefined', 'confidence': 0}
        
        # Simular análise baseada em médias móveis
        close_prices = market_data['close'] if 'close' in market_data.columns else market_data.iloc[:, 0]
        
        # Calcular tendência simples
        if len(close_prices) >= 20:
            sma_20 = close_prices.rolling(20).mean().iloc[-1]
            current_price = close_prices.iloc[-1]
            
            price_vs_sma = (current_price - sma_20) / sma_20
            
            # Calcular volatilidade
            returns = close_prices.pct_change().dropna()
            volatility = returns.std() if len(returns) > 0 else 0.01
            
            # Determinar regime
            if abs(price_vs_sma) > 0.02:  # 2% longe da média
                if price_vs_sma > 0:
                    regime = 'trend_up'
                else:
                    regime = 'trend_down'
                confidence = min(0.9, 0.6 + abs(price_vs_sma) * 10)
            else:
                regime = 'range'
                confidence = 0.8
                
            return {
                'regime': regime,
                'confidence': confidence,
                'trend_strength': abs(price_vs_sma),
                'volatility': volatility
            }
        
        return {'regime': 'undefined', 'confidence': 0.5}


class MockRegimeClassifier:
    """Mock do classificador de regime"""
    
    def predict(self, X):
        # Retorna regime baseado em alguma lógica simples
        return np.array([0])  # 0=range, 1=trend_up, 2=trend_down
    
    def predict_proba(self, X):
        # Retorna probabilidades mock
        return np.array([[0.7, 0.2, 0.1]])


class MockTrendModel:
    """Mock do modelo de tendência"""
    
    def predict_proba(self, X):
        # Simula probabilidade de continuação da tendência
        return np.array([[0.3, 0.7]])  # 70% de probabilidade de sinal


class MockRangeModel:
    """Mock do modelo de range"""
    
    def predict_proba(self, X):
        # Simula probabilidade de reversão
        return np.array([[0.4, 0.6]])  # 60% de probabilidade de sinal