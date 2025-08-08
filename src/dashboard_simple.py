import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from collections import deque
import threading
import time

# Import dos coletores de métricas
from src.metrics_collectors import (
    SystemMetricsCollector,
    TradingMetricsCollector, 
    ModelMetricsCollector,
    RiskMetricsCollector
)

class RealTimeDashboard:
    """Dashboard simplificado para monitoramento"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_buffer = deque(maxlen=1000)
        self.alerts_buffer = deque(maxlen=100)
        self.running = False
        
        # Inicializar coletores
        self._initialize_metrics_collectors()
        
    def _initialize_metrics_collectors(self):
        """Inicializa coletores de métricas"""
        self.collectors = {
            'system': SystemMetricsCollector(),
            'trading': TradingMetricsCollector(),
            'model': ModelMetricsCollector(),
            'risk': RiskMetricsCollector()
        }
        
    def start(self):
        """Inicia o dashboard"""
        self.running = True
        print("Dashboard iniciado com sucesso!")
        
    def stop(self):
        """Para o dashboard"""
        self.running = False
        print("Dashboard parado.")
        
    def collect_metrics(self):
        """Coleta métricas de todos os coletores"""
        current_metrics = {
            'timestamp': datetime.now()
        }
        
        for name, collector in self.collectors.items():
            try:
                metrics = collector.collect()
                current_metrics[name] = metrics
            except Exception as e:
                print(f"Erro coletando métricas {name}: {e}")
                current_metrics[name] = {'error': str(e)}
        
        self.metrics_buffer.append(current_metrics)
        return current_metrics
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Retorna dados para o dashboard"""
        if not self.metrics_buffer:
            return {'status': 'No data available'}
        
        latest = self.metrics_buffer[-1]
        
        return {
            'status': 'active',
            'timestamp': latest.get('timestamp', datetime.now()),
            'system': latest.get('system', {}),
            'trading': latest.get('trading', {}),
            'model': latest.get('model', {}),
            'risk': latest.get('risk', {}),
            'metrics_count': len(self.metrics_buffer)
        }
    
    def add_alert(self, alert: Dict[str, Any]):
        """Adiciona um alerta"""
        alert['timestamp'] = datetime.now()
        self.alerts_buffer.append(alert)
        print(f"ALERT: {alert.get('type', 'Unknown')} - {alert.get('message', 'No message')}")
        
    def get_alerts(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Retorna alertas recentes"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts_buffer 
            if alert.get('timestamp', datetime.min) > cutoff
        ]
        
    def is_running(self) -> bool:
        """Verifica se o dashboard está rodando"""
        return self.running
