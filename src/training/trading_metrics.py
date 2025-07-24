# src/training/metrics/trading_metrics.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

class TradingMetricsAnalyzer:
    """Analisador de métricas específicas para trading"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_trading_metrics(self, predictions: np.ndarray,
                                actual_returns: np.ndarray,
                                timestamps: Optional[pd.DatetimeIndex] = None,
                                position_size: float = 1.0) -> Dict:
        """
        Calcula métricas completas de trading
        
        Args:
            predictions: Array com predições (0=sell, 1=hold, 2=buy)
            actual_returns: Retornos reais do período
            timestamps: Timestamps das predições
            position_size: Tamanho da posição para cálculo
            
        Returns:
            Dicionário com métricas de trading
        """
        # Converter predições para posições
        positions = self._predictions_to_positions(predictions)
        
        # Calcular retornos da estratégia
        strategy_returns = positions[:-1] * actual_returns[1:]
        
        # Métricas básicas
        metrics = {
            'total_return': self._calculate_total_return(strategy_returns),
            'annualized_return': self._calculate_annualized_return(strategy_returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(strategy_returns),
            'sortino_ratio': self._calculate_sortino_ratio(strategy_returns),
            'max_drawdown': self._calculate_max_drawdown(strategy_returns),
            'calmar_ratio': self._calculate_calmar_ratio(strategy_returns),
            'win_rate': self._calculate_win_rate(strategy_returns),
            'profit_factor': self._calculate_profit_factor(strategy_returns),
            'recovery_factor': self._calculate_recovery_factor(strategy_returns),
            'payoff_ratio': self._calculate_payoff_ratio(strategy_returns)
        }
        
        # Análise de trades
        trades_analysis = self._analyze_trades(positions, actual_returns)
        metrics.update(trades_analysis)
        
        # Métricas de risco
        risk_metrics = self._calculate_risk_metrics(strategy_returns)
        metrics.update(risk_metrics)
        
        # Métricas de consistência
        consistency_metrics = self._calculate_consistency_metrics(
            strategy_returns, timestamps
        )
        metrics.update(consistency_metrics)
        
        return metrics
    
    def _predictions_to_positions(self, predictions: np.ndarray) -> np.ndarray:
        """Converte predições em posições"""
        # 0 = sell → -1
        # 1 = hold → 0
        # 2 = buy → 1
        positions = np.zeros_like(predictions, dtype=float)
        positions[predictions == 0] = -1
        positions[predictions == 2] = 1
        
        return positions
    
    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """Calcula retorno total acumulado"""
        return (1 + returns).prod() - 1
    
    def _calculate_annualized_return(self, returns: np.ndarray,
                                   periods_per_year: int = 252) -> float:
        """Calcula retorno anualizado"""
        total_return = self._calculate_total_return(returns)
        n_periods = len(returns)
        
        if n_periods == 0:
            return 0
        
        years = n_periods / periods_per_year
        annualized = (1 + total_return) ** (1 / years) - 1
        
        return annualized
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray,
                              risk_free_rate: float = 0.02,
                              periods_per_year: int = 252) -> float:
        """Calcula Sharpe Ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
        
        return sharpe
    
    def _calculate_sortino_ratio(self, returns: np.ndarray,
                               target_return: float = 0,
                               periods_per_year: int = 252) -> float:
        """Calcula Sortino Ratio (penaliza apenas volatilidade negativa)"""
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - target_return / periods_per_year
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = downside_returns.std()
        if downside_std == 0:
            return np.inf
        
        sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
        
        return sortino
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calcula drawdown máximo"""
        if len(returns) == 0:
            return 0
        
        # Calcula valor acumulado - converter para pandas Series para ter cummax
        cum_returns = pd.Series((1 + returns).cumprod())
        
        # Calcula running maximum
        running_max = cum_returns.cummax()
        
        # Calcula drawdown
        drawdown = (cum_returns - running_max) / running_max
        
        return abs(drawdown.min())
    
    def _calculate_calmar_ratio(self, returns: np.ndarray,
                              periods_per_year: int = 252) -> float:
        """Calcula Calmar Ratio (retorno anualizado / max drawdown)"""
        annual_return = self._calculate_annualized_return(returns, periods_per_year)
        max_dd = self._calculate_max_drawdown(returns)
        
        if max_dd == 0:
            return np.inf
        
        return annual_return / max_dd
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calcula taxa de acerto"""
        if len(returns) == 0:
            return 0
        
        winning_trades = returns > 0
        return winning_trades.sum() / len(returns)
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calcula fator de lucro (ganhos totais / perdas totais)"""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return np.inf
        
        return gains / losses
    
    def _calculate_recovery_factor(self, returns: np.ndarray) -> float:
        """Calcula fator de recuperação"""
        total_return = self._calculate_total_return(returns)
        max_dd = self._calculate_max_drawdown(returns)
        
        if max_dd == 0:
            return np.inf
        
        return total_return / max_dd
    
    def _calculate_payoff_ratio(self, returns: np.ndarray) -> float:
        """Calcula relação média ganho/perda"""
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(losses) == 0:
            return np.inf
        
        avg_gain = gains.mean() if len(gains) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
        
        return avg_gain / avg_loss
    
    def _analyze_trades(self, positions: np.ndarray, 
                       returns: np.ndarray) -> Dict:
        """Analisa trades individuais"""
        # Identificar mudanças de posição (trades)
        position_changes = np.diff(positions)
        trade_indices = np.where(position_changes != 0)[0]
        
        trades = []
        for i in range(len(trade_indices) - 1):
            entry_idx = trade_indices[i] + 1
            exit_idx = trade_indices[i + 1] + 1
            
            if exit_idx <= len(returns):
                trade_return = returns[entry_idx:exit_idx].sum()
                trade_duration = exit_idx - entry_idx
                
                trades.append({
                    'return': trade_return,
                    'duration': trade_duration,
                    'type': 'long' if positions[entry_idx] > 0 else 'short'
                })
        
        if not trades:
            return {
                'n_trades': 0,
                'avg_trade_return': 0,
                'avg_trade_duration': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'long_trades': 0,
                'short_trades': 0
            }
        
        trade_returns = [t['return'] for t in trades]
        
        return {
            'n_trades': len(trades),
            'avg_trade_return': np.mean(trade_returns),
            'avg_trade_duration': np.mean([t['duration'] for t in trades]),
            'best_trade': max(trade_returns),
            'worst_trade': min(trade_returns),
            'long_trades': sum(1 for t in trades if t['type'] == 'long'),
            'short_trades': sum(1 for t in trades if t['type'] == 'short'),
            'consecutive_wins': self._max_consecutive_wins(trade_returns),
            'consecutive_losses': self._max_consecutive_losses(trade_returns)
        }
    
    def _calculate_risk_metrics(self, returns: np.ndarray) -> Dict:
        """Calcula métricas de risco"""
        return {
            'volatility': returns.std() * np.sqrt(252),
            'downside_volatility': returns[returns < 0].std() * np.sqrt(252),
            'var_95': np.percentile(returns, 5),
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
            'skewness': self._calculate_skewness(returns),
            'kurtosis': self._calculate_kurtosis(returns),
            'omega_ratio': self._calculate_omega_ratio(returns)
        }
    
    def _calculate_consistency_metrics(self, returns: np.ndarray,
                                     timestamps: Optional[pd.DatetimeIndex] = None) -> Dict:
        """Calcula métricas de consistência"""
        metrics = {
            'positive_months': 0,
            'negative_months': 0,
            'best_month': 0.0,
            'worst_month': 0.0,
            'monthly_sharpe': 0.0
        }
        
        if timestamps is not None and len(returns) > 20:
            # Agregar por mês
            returns_series = pd.Series(returns, index=timestamps[:-1])
            monthly_returns = returns_series.groupby(pd.Grouper(freq='M')).sum()
            
            if len(monthly_returns) > 0:
                metrics['positive_months'] = int((monthly_returns > 0).sum())
                metrics['negative_months'] = int((monthly_returns < 0).sum())
                metrics['best_month'] = float(monthly_returns.max())
                metrics['worst_month'] = float(monthly_returns.min())
                metrics['monthly_sharpe'] = self._calculate_sharpe_ratio(
                    np.array(monthly_returns.values), periods_per_year=12
                )
        
        return metrics
    
    def _max_consecutive_wins(self, returns: List[float]) -> int:
        """Calcula máximo de vitórias consecutivas"""
        max_wins = 0
        current_wins = 0
        
        for r in returns:
            if r > 0:
                current_wins += 1
                max_wins = max(max_wins, current_wins)
            else:
                current_wins = 0
        
        return max_wins
    
    def _max_consecutive_losses(self, returns: List[float]) -> int:
        """Calcula máximo de perdas consecutivas"""
        max_losses = 0
        current_losses = 0
        
        for r in returns:
            if r < 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        
        return max_losses
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calcula assimetria dos retornos"""
        if len(returns) < 3:
            return 0
        
        mean = returns.mean()
        std = returns.std()
        
        if std == 0:
            return 0
        
        return ((returns - mean) ** 3).mean() / (std ** 3)
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calcula curtose dos retornos"""
        if len(returns) < 4:
            return 0
        
        mean = returns.mean()
        std = returns.std()
        
        if std == 0:
            return 0
        
        return ((returns - mean) ** 4).mean() / (std ** 4) - 3
    
    def _calculate_omega_ratio(self, returns: np.ndarray,
                             threshold: float = 0) -> float:
        """Calcula Omega ratio"""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if losses.sum() == 0:
            return np.inf
        
        return gains.sum() / losses.sum()
    
    def calculate_regime_performance(self, predictions: np.ndarray,
                                   actual_returns: np.ndarray,
                                   market_regimes: np.ndarray) -> Dict:
        """Calcula performance por regime de mercado"""
        regimes = ['trending_up', 'trending_down', 'ranging', 'high_volatility']
        regime_metrics = {}
        
        for regime in regimes:
            mask = market_regimes == regime
            
            if mask.sum() > 0:
                regime_preds = predictions[mask]
                regime_returns = actual_returns[mask]
                
                regime_metrics[regime] = self.calculate_trading_metrics(
                    regime_preds, regime_returns
                )
        
        return regime_metrics