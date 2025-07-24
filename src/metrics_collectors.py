import psutil
import os
from typing import Dict, Any
from datetime import datetime

class SystemMetricsCollector:
    """Coletor simples de métricas do system"""
    
    def collect(self) -> Dict[str, Any]:
        return {
            'cpu_percent': psutil.cpu_percent() if hasattr(psutil, 'cpu_percent') else 0.0,
            'memory_percent': psutil.virtual_memory().percent if hasattr(psutil, 'virtual_memory') else 0.0,
            'memory_mb': psutil.virtual_memory().used / (1024 * 1024) if hasattr(psutil, 'virtual_memory') else 0.0,
            'disk_usage': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') and os.path.exists('/') else 0.0,
            'timestamp': datetime.now()
        }

class TradingMetricsCollector:
    """Coletor simples de métricas de trading"""
    
    def collect(self) -> Dict[str, Any]:
        return {
            'active_positions': 0,
            'daily_pnl': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'timestamp': datetime.now()
        }

class ModelMetricsCollector:
    """Coletor simples de métricas de modelo"""
    
    def collect(self) -> Dict[str, Any]:
        return {
            'predictions_made': 0,
            'avg_confidence': 0.0,
            'model_accuracy': 0.0,
            'ensemble_agreement': 0.0,
            'timestamp': datetime.now()
        }

class RiskMetricsCollector:
    """Coletor simples de métricas de risco"""
    
    def collect(self) -> Dict[str, Any]:
        return {
            'current_drawdown': 0.0,
            'risk_per_trade': 0.02,
            'exposure': 0.0,
            'var': 0.0,
            'timestamp': datetime.now()
        }
