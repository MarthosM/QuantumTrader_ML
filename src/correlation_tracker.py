import numpy as np
import pandas as pd
from typing import Dict, List
import logging

class DynamicCorrelationTracker:
    """Rastreador de correlação dinâmica para gestão de risco"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.correlation_window = 60  # minutos
        self.correlation_history = {}
        
    def assess_risk(self, signal: Dict, portfolio_state: Dict) -> Dict:
        """Avalia risco de correlação para novo sinal"""
        
        # Extrair posições atuais
        current_positions = portfolio_state.get('positions', {})
        
        if not current_positions:
            return {
                'max_correlation': 0,
                'correlated_positions': [],
                'risk_level': 'low'
            }
        
        # Calcular correlações
        correlations = self._calculate_correlations(
            signal['symbol'], 
            [pos['symbol'] for pos in current_positions.values()]
        )
        
        # Identificar alta correlação
        high_corr_threshold = 0.7
        correlated = [
            (symbol, corr) for symbol, corr in correlations.items() 
            if abs(corr) > high_corr_threshold
        ]
        
        # Avaliar risco
        max_correlation = max(abs(c) for _, c in correlations.items()) if correlations else 0
        
        risk_level = 'low'
        if max_correlation > 0.9:
            risk_level = 'critical'
        elif max_correlation > 0.7:
            risk_level = 'high'
        elif max_correlation > 0.5:
            risk_level = 'medium'
            
        return {
            'max_correlation': max_correlation,
            'correlated_positions': correlated,
            'risk_level': risk_level,
            'correlations': correlations
        }
    
    def _calculate_correlations(self, new_symbol: str, 
                              existing_symbols: List[str]) -> Dict[str, float]:
        """Calcula correlações entre símbolos"""
        
        correlations = {}
        
        for symbol in existing_symbols:
            # Em produção, usar dados reais de preços
            # Por enquanto, simulação baseada em características dos ativos
            if symbol == new_symbol:
                corr = 1.0
            elif symbol[:3] == new_symbol[:3]:  # Mesmo underlying
                corr = 0.85
            elif 'FUT' in symbol and 'FUT' in new_symbol:  # Ambos futuros
                corr = 0.3
            else:
                corr = 0.1
                
            correlations[symbol] = corr
            
        return correlations