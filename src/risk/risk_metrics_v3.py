"""
Risk Metrics V3 - Sistema de Cálculo de Métricas de Risco e P&L
===============================================================

Este módulo implementa cálculos avançados de métricas de risco:
- P&L (Profit and Loss) detalhado
- Value at Risk (VaR)
- Sharpe Ratio e outras métricas de risco-retorno
- Maximum Drawdown e recovery time
- Risk-adjusted returns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats


@dataclass
class RiskMetrics:
    """Container para métricas de risco"""
    # P&L Metrics
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    
    # Return Metrics
    total_return: float
    annualized_return: float
    volatility: float
    downside_volatility: float
    
    # Risk Metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Drawdown Metrics
    max_drawdown: float
    max_drawdown_duration: int
    current_drawdown: float
    recovery_time: Optional[int]
    
    # VaR Metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    
    # Trade Statistics
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    expectancy: float
    
    # Other Metrics
    kelly_criterion: float
    risk_reward_ratio: float
    profit_per_day: float
    trades_per_day: float


class RiskMetricsCalculator:
    """Calculadora de métricas de risco"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Inicializa calculadora
        
        Args:
            risk_free_rate: Taxa livre de risco anualizada (default 5%)
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
        
    def calculate_all_metrics(self, 
                            trades: List[Dict],
                            equity_curve: List[float],
                            initial_capital: float,
                            trading_days: Optional[int] = None) -> RiskMetrics:
        """
        Calcula todas as métricas de risco
        
        Args:
            trades: Lista de trades executados
            equity_curve: Curva de equity
            initial_capital: Capital inicial
            trading_days: Número de dias de trading
            
        Returns:
            RiskMetrics com todas as métricas calculadas
        """
        # Converter para arrays numpy
        equity_array = np.array(equity_curve)
        
        # Calcular returns
        returns = self._calculate_returns(equity_array)
        
        # P&L Metrics
        pnl_metrics = self._calculate_pnl_metrics(trades, equity_array, initial_capital)
        
        # Return Metrics
        return_metrics = self._calculate_return_metrics(returns, trading_days)
        
        # Risk Metrics
        risk_metrics = self._calculate_risk_metrics(returns, return_metrics['volatility'])
        
        # Drawdown Metrics
        drawdown_metrics = self._calculate_drawdown_metrics(equity_array)
        
        # VaR Metrics
        var_metrics = self._calculate_var_metrics(returns)
        
        # Trade Statistics
        trade_stats = self._calculate_trade_statistics(trades)
        
        # Other Metrics
        other_metrics = self._calculate_other_metrics(
            trade_stats, return_metrics, trading_days or 1
        )
        
        # Combinar todas as métricas
        return RiskMetrics(
            # P&L
            total_pnl=pnl_metrics['total_pnl'],
            realized_pnl=pnl_metrics['realized_pnl'],
            unrealized_pnl=pnl_metrics['unrealized_pnl'],
            gross_profit=pnl_metrics['gross_profit'],
            gross_loss=pnl_metrics['gross_loss'],
            profit_factor=pnl_metrics['profit_factor'],
            
            # Returns
            total_return=return_metrics['total_return'],
            annualized_return=return_metrics['annualized_return'],
            volatility=return_metrics['volatility'],
            downside_volatility=return_metrics['downside_volatility'],
            
            # Risk
            sharpe_ratio=risk_metrics['sharpe_ratio'],
            sortino_ratio=risk_metrics['sortino_ratio'],
            calmar_ratio=risk_metrics['calmar_ratio'],
            information_ratio=risk_metrics['information_ratio'],
            
            # Drawdown
            max_drawdown=drawdown_metrics['max_drawdown'],
            max_drawdown_duration=drawdown_metrics['max_drawdown_duration'],
            current_drawdown=drawdown_metrics['current_drawdown'],
            recovery_time=drawdown_metrics['recovery_time'],
            
            # VaR
            var_95=var_metrics['var_95'],
            var_99=var_metrics['var_99'],
            cvar_95=var_metrics['cvar_95'],
            cvar_99=var_metrics['cvar_99'],
            
            # Trade Stats
            win_rate=trade_stats['win_rate'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            avg_trade=trade_stats['avg_trade'],
            expectancy=trade_stats['expectancy'],
            
            # Other
            kelly_criterion=other_metrics['kelly_criterion'],
            risk_reward_ratio=other_metrics['risk_reward_ratio'],
            profit_per_day=other_metrics['profit_per_day'],
            trades_per_day=other_metrics['trades_per_day']
        )
    
    def _calculate_returns(self, equity_curve: np.ndarray) -> np.ndarray:
        """Calcula retornos da curva de equity"""
        if len(equity_curve) < 2:
            return np.array([])
        
        # Retornos percentuais
        returns = np.diff(equity_curve) / equity_curve[:-1]
        return returns[~np.isnan(returns)]  # Remover NaN
    
    def _calculate_pnl_metrics(self, 
                              trades: List[Dict], 
                              equity_curve: np.ndarray,
                              initial_capital: float) -> Dict:
        """Calcula métricas de P&L"""
        if not trades:
            return {
                'total_pnl': 0.0,
                'realized_pnl': 0.0,
                'unrealized_pnl': 0.0,
                'gross_profit': 0.0,
                'gross_loss': 0.0,
                'profit_factor': 0.0
            }
        
        # P&L realizado
        realized_pnl = sum(t.get('pnl', 0) for t in trades)
        
        # P&L total
        final_equity = equity_curve[-1] if len(equity_curve) > 0 else initial_capital
        total_pnl = final_equity - initial_capital
        
        # P&L não realizado
        unrealized_pnl = total_pnl - realized_pnl
        
        # Lucro e perda brutos
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        
        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_pnl': total_pnl,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor
        }
    
    def _calculate_return_metrics(self, 
                                 returns: np.ndarray,
                                 trading_days: Optional[int]) -> Dict:
        """Calcula métricas de retorno"""
        if len(returns) == 0:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'downside_volatility': 0.0
            }
        
        # Retorno total
        total_return = np.prod(1 + returns) - 1
        
        # Retorno anualizado
        if trading_days and trading_days > 0:
            annualized_factor = 252 / trading_days
            annualized_return = (1 + total_return) ** annualized_factor - 1
        else:
            annualized_return = total_return
        
        # Volatilidade
        volatility = np.std(returns) * np.sqrt(252)
        
        # Downside volatility (apenas retornos negativos)
        negative_returns = returns[returns < 0]
        downside_volatility = (
            np.std(negative_returns) * np.sqrt(252) 
            if len(negative_returns) > 0 else 0
        )
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'downside_volatility': downside_volatility
        }
    
    def _calculate_risk_metrics(self, 
                               returns: np.ndarray,
                               volatility: float) -> Dict:
        """Calcula métricas de risco-retorno"""
        if len(returns) == 0 or volatility == 0:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'information_ratio': 0.0
            }
        
        # Média dos retornos
        mean_return = np.mean(returns)
        annualized_mean = mean_return * 252
        
        # Sharpe Ratio
        sharpe_ratio = (annualized_mean - self.risk_free_rate) / volatility
        
        # Sortino Ratio
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 1e-6
        sortino_ratio = (annualized_mean - self.risk_free_rate) / (downside_std * np.sqrt(252))
        
        # Calmar Ratio (return / max drawdown)
        # Será calculado depois com drawdown
        calmar_ratio = 0.0
        
        # Information Ratio (simplificado)
        information_ratio = annualized_mean / volatility if volatility > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio
        }
    
    def _calculate_drawdown_metrics(self, equity_curve: np.ndarray) -> Dict:
        """Calcula métricas de drawdown"""
        if len(equity_curve) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'current_drawdown': 0.0,
                'recovery_time': None
            }
        
        # Calcular running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calcular drawdown em cada ponto
        drawdown = (running_max - equity_curve) / running_max
        
        # Maximum drawdown
        max_drawdown = np.max(drawdown)
        
        # Current drawdown
        current_drawdown = drawdown[-1]
        
        # Drawdown duration
        max_dd_duration = self._calculate_max_drawdown_duration(drawdown)
        
        # Recovery time (se aplicável)
        recovery_time = self._calculate_recovery_time(drawdown)
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'current_drawdown': current_drawdown,
            'recovery_time': recovery_time
        }
    
    def _calculate_max_drawdown_duration(self, drawdown: np.ndarray) -> int:
        """Calcula duração máxima de drawdown"""
        if len(drawdown) == 0:
            return 0
        
        # Encontrar períodos de drawdown
        in_drawdown = drawdown > 0
        
        # Calcular duração de cada período
        max_duration = 0
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _calculate_recovery_time(self, drawdown: np.ndarray) -> Optional[int]:
        """Calcula tempo de recuperação do último drawdown"""
        if len(drawdown) == 0 or drawdown[-1] > 0:
            return None  # Ainda em drawdown
        
        # Encontrar último ponto em drawdown
        for i in range(len(drawdown) - 1, -1, -1):
            if drawdown[i] > 0:
                return len(drawdown) - i - 1
        
        return None
    
    def _calculate_var_metrics(self, returns: np.ndarray) -> Dict:
        """Calcula Value at Risk e Conditional VaR"""
        if len(returns) == 0:
            return {
                'var_95': 0.0,
                'var_99': 0.0,
                'cvar_95': 0.0,
                'cvar_99': 0.0
            }
        
        # VaR percentis
        var_95 = np.percentile(returns, 5)  # 5% pior caso
        var_99 = np.percentile(returns, 1)  # 1% pior caso
        
        # CVaR (Expected Shortfall)
        cvar_95 = np.mean(returns[returns <= var_95])
        cvar_99 = np.mean(returns[returns <= var_99])
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99
        }
    
    def _calculate_trade_statistics(self, trades: List[Dict]) -> Dict:
        """Calcula estatísticas dos trades"""
        if not trades:
            return {
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_trade': 0.0,
                'expectancy': 0.0
            }
        
        # Separar trades vencedores e perdedores
        pnls = [t.get('pnl', 0) for t in trades]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        # Win rate
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Médias
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        avg_trade = np.mean(pnls) if pnls else 0
        
        # Expectancy (valor esperado por trade)
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'expectancy': expectancy
        }
    
    def _calculate_other_metrics(self,
                               trade_stats: Dict,
                               return_metrics: Dict,
                               trading_days: int) -> Dict:
        """Calcula outras métricas úteis"""
        # Kelly Criterion
        if trade_stats['avg_loss'] != 0 and trade_stats['win_rate'] > 0:
            win_loss_ratio = abs(trade_stats['avg_win'] / trade_stats['avg_loss'])
            kelly = (trade_stats['win_rate'] * win_loss_ratio - (1 - trade_stats['win_rate'])) / win_loss_ratio
            kelly = max(0, min(0.25, kelly))  # Limitar entre 0 e 25%
        else:
            kelly = 0.0
        
        # Risk Reward Ratio
        risk_reward = (
            abs(trade_stats['avg_win'] / trade_stats['avg_loss']) 
            if trade_stats['avg_loss'] != 0 else 0
        )
        
        # Profit per day
        profit_per_day = (
            return_metrics.get('total_return', 0) / trading_days 
            if trading_days > 0 else 0
        )
        
        # Trades per day (placeholder - needs trade count)
        trades_per_day = 0  # Será calculado com dados reais
        
        return {
            'kelly_criterion': kelly,
            'risk_reward_ratio': risk_reward,
            'profit_per_day': profit_per_day,
            'trades_per_day': trades_per_day
        }
    
    def generate_risk_report(self, metrics: RiskMetrics) -> str:
        """Gera relatório de risco formatado"""
        report = f"""
RELATÓRIO DE MÉTRICAS DE RISCO
==============================

P&L SUMMARY
-----------
Total P&L: R$ {metrics.total_pnl:,.2f}
Realized P&L: R$ {metrics.realized_pnl:,.2f}
Unrealized P&L: R$ {metrics.unrealized_pnl:,.2f}
Gross Profit: R$ {metrics.gross_profit:,.2f}
Gross Loss: R$ {metrics.gross_loss:,.2f}
Profit Factor: {metrics.profit_factor:.2f}

RETURN METRICS
--------------
Total Return: {metrics.total_return:.2%}
Annualized Return: {metrics.annualized_return:.2%}
Volatility: {metrics.volatility:.2%}
Downside Volatility: {metrics.downside_volatility:.2%}

RISK-ADJUSTED RETURNS
--------------------
Sharpe Ratio: {metrics.sharpe_ratio:.2f}
Sortino Ratio: {metrics.sortino_ratio:.2f}
Calmar Ratio: {metrics.calmar_ratio:.2f}
Information Ratio: {metrics.information_ratio:.2f}

DRAWDOWN ANALYSIS
-----------------
Maximum Drawdown: {metrics.max_drawdown:.2%}
Max DD Duration: {metrics.max_drawdown_duration} periods
Current Drawdown: {metrics.current_drawdown:.2%}
Recovery Time: {metrics.recovery_time or 'N/A'} periods

VALUE AT RISK
-------------
VaR 95%: {metrics.var_95:.2%}
VaR 99%: {metrics.var_99:.2%}
CVaR 95%: {metrics.cvar_95:.2%}
CVaR 99%: {metrics.cvar_99:.2%}

TRADE STATISTICS
----------------
Win Rate: {metrics.win_rate:.2%}
Average Win: R$ {metrics.avg_win:,.2f}
Average Loss: R$ {metrics.avg_loss:,.2f}
Average Trade: R$ {metrics.avg_trade:,.2f}
Expectancy: R$ {metrics.expectancy:,.2f}

POSITION SIZING
---------------
Kelly Criterion: {metrics.kelly_criterion:.2%}
Risk/Reward Ratio: {metrics.risk_reward_ratio:.2f}

PERFORMANCE
-----------
Profit per Day: {metrics.profit_per_day:.2%}
Trades per Day: {metrics.trades_per_day:.1f}
"""
        return report


def validate_risk_limits(metrics: RiskMetrics, 
                        risk_limits: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Valida se as métricas estão dentro dos limites de risco
    
    Args:
        metrics: Métricas calculadas
        risk_limits: Dicionário com limites de risco
        
    Returns:
        (is_valid, violations): Tupla com status e lista de violações
    """
    violations = []
    
    # Verificar cada limite
    if metrics.max_drawdown > risk_limits.get('max_drawdown', 0.20):
        violations.append(f"Max Drawdown ({metrics.max_drawdown:.2%}) excede limite ({risk_limits['max_drawdown']:.2%})")
    
    if metrics.sharpe_ratio < risk_limits.get('min_sharpe', 0.5):
        violations.append(f"Sharpe Ratio ({metrics.sharpe_ratio:.2f}) abaixo do mínimo ({risk_limits['min_sharpe']:.2f})")
    
    if metrics.win_rate < risk_limits.get('min_win_rate', 0.40):
        violations.append(f"Win Rate ({metrics.win_rate:.2%}) abaixo do mínimo ({risk_limits['min_win_rate']:.2%})")
    
    if metrics.profit_factor < risk_limits.get('min_profit_factor', 1.2):
        violations.append(f"Profit Factor ({metrics.profit_factor:.2f}) abaixo do mínimo ({risk_limits['min_profit_factor']:.2f})")
    
    return len(violations) == 0, violations


if __name__ == "__main__":
    # Teste das métricas
    import json
    
    # Carregar dados de exemplo
    with open('backtest_results.json', 'r') as f:
        backtest_data = json.load(f)
    
    # Criar calculadora
    calculator = RiskMetricsCalculator()
    
    # Calcular métricas
    metrics = calculator.calculate_all_metrics(
        trades=backtest_data['trades'],
        equity_curve=backtest_data['equity_curve'],
        initial_capital=backtest_data['config']['initial_capital'],
        trading_days=5
    )
    
    # Gerar relatório
    report = calculator.generate_risk_report(metrics)
    print(report)
    
    # Validar limites de risco
    risk_limits = {
        'max_drawdown': 0.10,
        'min_sharpe': 1.0,
        'min_win_rate': 0.50,
        'min_profit_factor': 1.5
    }
    
    is_valid, violations = validate_risk_limits(metrics, risk_limits)
    
    if not is_valid:
        print("\nVIOLAÇÕES DE RISCO DETECTADAS:")
        for violation in violations:
            print(f"- {violation}")
    else:
        print("\nTODAS AS MÉTRICAS DENTRO DOS LIMITES DE RISCO")