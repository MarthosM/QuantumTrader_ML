"""
Adaptive Threshold Manager - Ajusta thresholds dinamicamente
"""

import json
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from collections import deque


class AdaptiveThresholdManager:
    """Gerencia e ajusta thresholds baseado em performance"""
    
    def __init__(self, config_path: str = "config/improved_thresholds.json"):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        
        # Carregar thresholds iniciais
        self.thresholds = self._load_thresholds()
        self.original_thresholds = self.thresholds.copy()
        
        # Histórico de performance
        self.trade_history = deque(maxlen=100)
        self.regime_performance = {
            'trend_up': {'trades': 0, 'wins': 0, 'total_return': 0},
            'trend_down': {'trades': 0, 'wins': 0, 'total_return': 0},
            'range': {'trades': 0, 'wins': 0, 'total_return': 0}
        }
        
        # Parâmetros de adaptação
        self.min_trades_for_adaptation = 20
        self.adaptation_rate = 0.1
        self.performance_window = 50
        
    def _load_thresholds(self) -> Dict:
        """Carrega thresholds do arquivo"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except:
            # Fallback para valores padrão
            return {
                "trend_thresholds": {
                    "confidence": 0.55,
                    "direction": 0.25,
                    "magnitude": 0.0015
                },
                "range_thresholds": {
                    "confidence": 0.50,
                    "direction": 0.20,
                    "magnitude": 0.001
                }
            }
    
    def get_thresholds_for_regime(self, regime: str) -> Dict:
        """Retorna thresholds ajustados para o regime"""
        if regime in ['trend_up', 'trend_down']:
            return self.thresholds.get('trend_thresholds', {})
        elif regime == 'range':
            return self.thresholds.get('range_thresholds', {})
        else:
            return self.thresholds.get('undefined_thresholds', self.thresholds['trend_thresholds'])
    
    def update_trade_result(self, trade_info: Dict):
        """Atualiza histórico com resultado do trade"""
        self.trade_history.append({
            'timestamp': datetime.now(),
            'regime': trade_info.get('regime', 'unknown'),
            'win': trade_info.get('win', False),
            'return': trade_info.get('return', 0),
            'confidence': trade_info.get('confidence', 0),
            'direction': trade_info.get('direction', 0),
            'magnitude': trade_info.get('magnitude', 0)
        })
        
        # Atualizar estatísticas por regime
        regime = trade_info.get('regime', 'unknown')
        if regime in self.regime_performance:
            self.regime_performance[regime]['trades'] += 1
            if trade_info.get('win', False):
                self.regime_performance[regime]['wins'] += 1
            self.regime_performance[regime]['total_return'] += trade_info.get('return', 0)
    
    def adapt_thresholds(self):
        """Adapta thresholds baseado em performance recente"""
        if len(self.trade_history) < self.min_trades_for_adaptation:
            return
        
        # Analisar performance por regime
        for regime in ['trend', 'range']:
            regime_trades = [t for t in self.trade_history 
                           if regime in t.get('regime', '')]
            
            if len(regime_trades) < 10:
                continue
            
            # Calcular métricas
            win_rate = sum(1 for t in regime_trades if t['win']) / len(regime_trades)
            avg_confidence = np.mean([t['confidence'] for t in regime_trades])
            
            # Ajustar thresholds
            threshold_key = f"{regime}_thresholds"
            if threshold_key in self.thresholds:
                # Se win rate baixo, aumentar thresholds (ser mais seletivo)
                if win_rate < 0.45:
                    self._increase_thresholds(threshold_key, 0.05)
                # Se win rate alto mas poucos trades, diminuir thresholds
                elif win_rate > 0.60 and len(regime_trades) < 20:
                    self._decrease_thresholds(threshold_key, 0.03)
                
                self.logger.info(f"[ADAPTIVE] {regime} - WR: {win_rate:.2%}, Trades: {len(regime_trades)}")
    
    def _increase_thresholds(self, threshold_key: str, factor: float):
        """Aumenta thresholds (mais conservador)"""
        thresholds = self.thresholds[threshold_key]
        for key in ['confidence', 'direction']:
            if key in thresholds:
                thresholds[key] = min(0.8, thresholds[key] * (1 + factor))
        
        self.logger.info(f"[ADAPTIVE] Aumentando thresholds {threshold_key}: {thresholds}")
    
    def _decrease_thresholds(self, threshold_key: str, factor: float):
        """Diminui thresholds (mais agressivo)"""
        thresholds = self.thresholds[threshold_key]
        for key in ['confidence', 'direction']:
            if key in thresholds:
                # Não deixar ficar muito baixo
                min_value = 0.15 if key == 'direction' else 0.40
                thresholds[key] = max(min_value, thresholds[key] * (1 - factor))
        
        self.logger.info(f"[ADAPTIVE] Diminuindo thresholds {threshold_key}: {thresholds}")
    
    def get_performance_summary(self) -> Dict:
        """Retorna resumo de performance"""
        summary = {
            'total_trades': len(self.trade_history),
            'regime_stats': {}
        }
        
        for regime, stats in self.regime_performance.items():
            if stats['trades'] > 0:
                summary['regime_stats'][regime] = {
                    'trades': stats['trades'],
                    'win_rate': stats['wins'] / stats['trades'],
                    'avg_return': stats['total_return'] / stats['trades']
                }
        
        return summary
    
    def suggest_threshold_adjustments(self) -> List[str]:
        """Sugere ajustes manuais baseados em análise"""
        suggestions = []
        
        # Analisar trades recentes
        recent_trades = list(self.trade_history)[-30:]
        if not recent_trades:
            return ["Sem dados suficientes para sugestões"]
        
        # Analisar por regime
        for regime in ['trend', 'range']:
            regime_trades = [t for t in recent_trades if regime in t.get('regime', '')]
            if len(regime_trades) < 5:
                continue
            
            win_rate = sum(1 for t in regime_trades if t['win']) / len(regime_trades)
            avg_conf = np.mean([t['confidence'] for t in regime_trades])
            
            # Sugestões específicas
            if win_rate < 0.4:
                if avg_conf < 0.6:
                    suggestions.append(f"{regime}: Aumentar threshold de confiança (atual: {self.thresholds[f'{regime}_thresholds']['confidence']:.2f})")
                else:
                    suggestions.append(f"{regime}: Revisar features - boa confiança mas baixo win rate")
            
            if len(regime_trades) < 10 and win_rate > 0.5:
                suggestions.append(f"{regime}: Considerar reduzir thresholds para mais sinais")
        
        # Análise geral
        total_trades = len(recent_trades)
        if total_trades < 15:
            suggestions.append("Poucos trades gerados - considerar reduzir thresholds gerais em 10-15%")
        
        return suggestions