import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from collections import deque
import threading
import time

# Import dos coletores de métricas
from metrics_collectors import (
    SystemMetricsCollector,
    TradingMetricsCollector, 
    ModelMetricsCollector,
    RiskMetricsCollector
)

class RealTimeDashboard:
    """Dashboard interativo para monitoramento em tempo real"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_buffer = deque(maxlen=1000)  # Buffer circular para métricas
        self.alerts_buffer = deque(maxlen=100)
        self.performance_history = {}
        self.model_metrics = {}
        self.system_health = {}
        
        # WebSocket para comunicação em tempo real
        self.websocket_server = None
        self.connected_clients = []
        
        # Thread para atualização contínua
        self.update_thread = None
        self.running = False
        
        # Inicializar componentes
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
        """Inicia o dashboard e servidor WebSocket"""
        self.running = True
        
        # Iniciar servidor WebSocket
        self._start_websocket_server()
        
        # Iniciar thread de atualização
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        print("Dashboard iniciado em http://localhost:8080")
        
    def _update_loop(self):
        """Loop principal de atualização de métricas"""
        while self.running:
            try:
                # Coletar métricas de todos os coletores
                current_metrics = self._collect_all_metrics()
                
                # Adicionar timestamp
                current_metrics['timestamp'] = datetime.now().isoformat()
                
                # Armazenar no buffer
                self.metrics_buffer.append(current_metrics)
                
                # Processar alertas
                alerts = self._check_alerts(current_metrics)
                if alerts:
                    self._process_alerts(alerts)
                
                # Enviar para clientes conectados
                self._broadcast_metrics(current_metrics)
                
                # Aguardar intervalo de atualização
                time.sleep(self.config.get('update_interval', 1))
                
            except Exception as e:
                print(f"Erro no loop de atualização: {e}")
                
    def _collect_all_metrics(self) -> Dict[str, Any]:
        """Coleta métricas de todos os coletores"""
        all_metrics = {}
        
        for name, collector in self.collectors.items():
            try:
                metrics = collector.collect()
                all_metrics[name] = metrics
            except Exception as e:
                print(f"Erro coletando métricas de {name}: {e}")
                all_metrics[name] = {}
                
        return all_metrics
        
    def _check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verifica condições de alerta"""
        alerts = []
        
        # Verificar performance de trading
        if 'trading' in metrics:
            trading = metrics['trading']
            
            # Alerta de drawdown
            if trading.get('current_drawdown', 0) > self.config.get('max_drawdown_alert', 0.05):
                alerts.append({
                    'level': 'critical',
                    'type': 'drawdown',
                    'message': f"Drawdown crítico: {trading['current_drawdown']:.2%}",
                    'value': trading['current_drawdown']
                })
            
            # Alerta de win rate baixo
            if trading.get('win_rate_today', 1) < self.config.get('min_win_rate_alert', 0.45):
                alerts.append({
                    'level': 'warning',
                    'type': 'win_rate',
                    'message': f"Win rate baixo: {trading['win_rate_today']:.2%}",
                    'value': trading['win_rate_today']
                })
        
        # Verificar saúde do sistema
        if 'system' in metrics:
            system = metrics['system']
            
            # Alerta de memória
            if system.get('memory_usage_percent', 0) > 80:
                alerts.append({
                    'level': 'warning',
                    'type': 'memory',
                    'message': f"Uso de memória alto: {system['memory_usage_percent']:.1f}%",
                    'value': system['memory_usage_percent']
                })
            
            # Alerta de latência
            if system.get('latency_ms', 0) > self.config.get('max_latency_alert', 100):
                alerts.append({
                    'level': 'warning',
                    'type': 'latency',
                    'message': f"Latência alta: {system['latency_ms']}ms",
                    'value': system['latency_ms']
                })
        
        # Verificar modelos ML
        if 'model' in metrics:
            model = metrics['model']
            
            # Alerta de drift
            if model.get('drift_score', 0) > self.config.get('max_drift_alert', 0.1):
                alerts.append({
                    'level': 'warning',
                    'type': 'model_drift',
                    'message': f"Drift detectado: score {model['drift_score']:.3f}",
                    'value': model['drift_score']
                })
        
        return alerts
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Retorna resumo de performance"""
        if not self.metrics_buffer:
            return {}
            
        # Converter buffer para DataFrame para análise
        df = pd.DataFrame(list(self.metrics_buffer))
        
        # Calcular estatísticas
        summary = {
            'period': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'trading': self._calculate_trading_summary(df),
            'system': self._calculate_system_summary(df),
            'models': self._calculate_model_summary(df),
            'alerts': self._get_alert_summary()
        }
        
        return summary