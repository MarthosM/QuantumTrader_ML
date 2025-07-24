from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import deque

class PerformanceAnalyzer:
    """Analyzador básico de performance"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.pnl_history = deque(maxlen=1000)
        self.trade_history = deque(maxlen=1000)
        
    def add_trade(self, trade: Dict[str, Any]):
        """Adiciona um trade ao histórico"""
        self.trade_history.append({
            'timestamp': datetime.now(),
            'pnl': trade.get('pnl', 0),
            'side': trade.get('side', 'unknown'),
            'size': trade.get('size', 0)
        })
        
    def analyze_trade(self, trade: Dict[str, Any]):
        """Analisa um trade (alias para add_trade)"""
        self.add_trade(trade)
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de performance"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_trade': 0.0,
                'max_drawdown': 0.0
            }
            
        trades = list(self.trade_history)
        total_pnl = sum(t['pnl'] for t in trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        
        return {
            'total_trades': len(trades),
            'win_rate': len(winning_trades) / len(trades) if trades else 0,
            'total_pnl': total_pnl,
            'avg_trade': total_pnl / len(trades) if trades else 0,
            'max_drawdown': self._calculate_max_drawdown(trades),
            'status': 'active'
        }
        
    def _calculate_max_drawdown(self, trades: List[Dict[str, Any]]) -> float:
        """Calcula drawdown máximo"""
        if not trades:
            return 0.0
            
        running_total = 0
        max_peak = 0
        max_drawdown = 0
        
        for trade in trades:
            running_total += trade['pnl']
            max_peak = max(max_peak, running_total)
            drawdown = max_peak - running_total
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown
        
    def get_daily_pnl(self) -> float:
        """Retorna P&L do dia"""
        today_trades = [
            t for t in self.trade_history 
            if t['timestamp'].date() == datetime.now().date()
        ]
        return sum(t['pnl'] for t in today_trades)
        
    def reset_analysis(self):
        """Reset da análise"""
        self.pnl_history.clear()
        self.trade_history.clear()
