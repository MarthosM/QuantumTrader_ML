"""
SystemMonitorV3 - Sistema de monitoramento em tempo real
Baseado no DEVELOPER_GUIDE_V3_REFACTORING.md
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import threading
import time
import json
import os
from collections import deque, defaultdict


class SystemMonitorV3:
    """
    Monitor completo do sistema de trading
    
    Features:
    - Monitora performance em tempo real
    - Rastreia latências e throughput
    - Detecta anomalias e problemas
    - Gera alertas automáticos
    - Mantém histórico de métricas
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o monitor
        
        Args:
            config: Configurações opcionais
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configurações
        self.metrics_window = self.config.get('metrics_window', 3600)  # 1 hora
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'latency_ms': 100,
            'error_rate': 0.05,
            'memory_usage_pct': 80,
            'queue_usage_pct': 70
        })
        
        # Métricas em tempo real
        self.metrics = {
            'system': {
                'uptime_seconds': 0,
                'start_time': None,
                'memory_usage_mb': 0,
                'cpu_usage_pct': 0
            },
            'data_flow': {
                'trades_per_second': 0,
                'books_per_second': 0,
                'features_per_second': 0,
                'predictions_per_second': 0
            },
            'latencies': {
                'trade_processing_ms': deque(maxlen=1000),
                'feature_calculation_ms': deque(maxlen=1000),
                'prediction_generation_ms': deque(maxlen=1000),
                'end_to_end_ms': deque(maxlen=1000)
            },
            'errors': defaultdict(int),
            'predictions': {
                'total': 0,
                'by_regime': defaultdict(int),
                'by_action': defaultdict(int),
                'confidence_histogram': defaultdict(int)
            },
            'trading': {
                'signals_generated': 0,
                'orders_sent': 0,
                'orders_filled': 0,
                'positions_opened': 0,
                'positions_closed': 0
            }
        }
        
        # Histórico de métricas
        self.metrics_history = []
        self.alerts_history = []
        
        # Estado do monitor
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Callbacks para componentes
        self.component_callbacks = {}
        
        # Lock para thread safety
        self._lock = threading.RLock()
        
    def start(self):
        """Inicia monitoramento"""
        self.logger.info("Iniciando SystemMonitorV3...")
        
        with self._lock:
            self.monitoring_active = True
            self.metrics['system']['start_time'] = datetime.now()
            
        # Iniciar thread de monitoramento
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Monitoramento iniciado")
        
    def stop(self):
        """Para monitoramento"""
        self.logger.info("Parando SystemMonitorV3...")
        
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
            
        self.logger.info("Monitoramento parado")
        
    def register_component(self, component_name: str, callback: callable):
        """
        Registra componente para monitoramento
        
        Args:
            component_name: Nome do componente
            callback: Função que retorna métricas do componente
        """
        self.component_callbacks[component_name] = callback
        self.logger.info(f"Componente registrado: {component_name}")
        
    def record_latency(self, operation: str, latency_ms: float):
        """
        Registra latência de uma operação
        
        Args:
            operation: Nome da operação
            latency_ms: Latência em milissegundos
        """
        with self._lock:
            if operation in self.metrics['latencies']:
                self.metrics['latencies'][operation].append(latency_ms)
                
                # Verificar threshold
                if latency_ms > self.alert_thresholds['latency_ms']:
                    self._create_alert(
                        'high_latency',
                        f"Latência alta em {operation}: {latency_ms:.1f}ms",
                        severity='warning'
                    )
                    
    def record_prediction(self, prediction: Dict):
        """
        Registra uma predição realizada
        
        Args:
            prediction: Dados da predição
        """
        with self._lock:
            self.metrics['predictions']['total'] += 1
            
            # Por regime
            regime = prediction.get('regime', 'unknown')
            self.metrics['predictions']['by_regime'][regime] += 1
            
            # Por ação
            action = prediction.get('action', 'unknown')
            self.metrics['predictions']['by_action'][action] += 1
            
            # Histograma de confiança
            confidence = prediction.get('confidence', 0)
            bucket = int(confidence * 10) / 10  # Buckets de 0.1
            self.metrics['predictions']['confidence_histogram'][bucket] += 1
            
    def record_error(self, component: str, error_type: str):
        """
        Registra um erro
        
        Args:
            component: Componente onde ocorreu o erro
            error_type: Tipo do erro
        """
        with self._lock:
            error_key = f"{component}_{error_type}"
            self.metrics['errors'][error_key] += 1
            
            # Calcular taxa de erro
            total_operations = max(1, self.metrics['predictions']['total'])
            error_rate = sum(self.metrics['errors'].values()) / total_operations
            
            if error_rate > self.alert_thresholds['error_rate']:
                self._create_alert(
                    'high_error_rate',
                    f"Taxa de erro alta: {error_rate:.1%}",
                    severity='critical'
                )
                
    def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        
        while self.monitoring_active:
            try:
                # Atualizar métricas do sistema
                self._update_system_metrics()
                
                # Coletar métricas dos componentes
                self._collect_component_metrics()
                
                # Calcular métricas derivadas
                self._calculate_derived_metrics()
                
                # Verificar alertas
                self._check_alerts()
                
                # Salvar snapshot
                self._save_metrics_snapshot()
                
                # Aguardar próximo ciclo
                time.sleep(1)  # Atualizar a cada segundo
                
            except Exception as e:
                self.logger.error(f"Erro no loop de monitoramento: {e}")
                
    def _update_system_metrics(self):
        """Atualiza métricas do sistema"""
        
        with self._lock:
            # Uptime
            if self.metrics['system']['start_time']:
                uptime = (datetime.now() - self.metrics['system']['start_time']).total_seconds()
                self.metrics['system']['uptime_seconds'] = uptime
                
            # Memória (simplificado)
            try:
                import psutil
                process = psutil.Process()
                self.metrics['system']['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
                self.metrics['system']['cpu_usage_pct'] = process.cpu_percent(interval=0.1)
            except:
                pass
                
    def _collect_component_metrics(self):
        """Coleta métricas dos componentes registrados"""
        
        for component_name, callback in self.component_callbacks.items():
            try:
                component_metrics = callback()
                
                # Processar métricas do componente
                if component_name == 'realtime_processor':
                    self._process_realtime_metrics(component_metrics)
                elif component_name == 'prediction_engine':
                    self._process_prediction_metrics(component_metrics)
                    
            except Exception as e:
                self.logger.error(f"Erro coletando métricas de {component_name}: {e}")
                
    def _process_realtime_metrics(self, metrics: Dict):
        """Processa métricas do RealTimeProcessor"""
        
        with self._lock:
            # Throughput
            if 'trades_processed' in metrics:
                # Calcular taxa aproximada
                self.metrics['data_flow']['trades_per_second'] = metrics.get('trades_processed', 0) / max(1, self.metrics['system']['uptime_seconds'])
                
            # Latências
            if 'avg_latency_ms' in metrics:
                self.record_latency('trade_processing_ms', metrics['avg_latency_ms'])
                
    def _process_prediction_metrics(self, metrics: Dict):
        """Processa métricas do PredictionEngine"""
        
        with self._lock:
            if 'predictions_made' in metrics:
                self.metrics['data_flow']['predictions_per_second'] = metrics.get('predictions_made', 0) / max(1, self.metrics['system']['uptime_seconds'])
                
    def _calculate_derived_metrics(self):
        """Calcula métricas derivadas"""
        
        with self._lock:
            # Latências médias
            for operation, latencies in self.metrics['latencies'].items():
                if latencies:
                    avg_latency = np.mean(list(latencies))
                    max_latency = np.max(list(latencies))
                    
                    # Adicionar às métricas
                    self.metrics[f'{operation}_avg'] = avg_latency
                    self.metrics[f'{operation}_max'] = max_latency
                    
    def _check_alerts(self):
        """Verifica condições de alerta"""
        
        with self._lock:
            # Verificar memória
            memory_usage_mb = self.metrics['system'].get('memory_usage_mb', 0)
            if memory_usage_mb > 1000:  # Mais de 1GB
                memory_pct = (memory_usage_mb / 2048) * 100  # Assumindo 2GB disponível
                if memory_pct > self.alert_thresholds['memory_usage_pct']:
                    self._create_alert(
                        'high_memory',
                        f"Uso de memória alto: {memory_usage_mb:.0f}MB ({memory_pct:.1f}%)",
                        severity='warning'
                    )
                    
    def _create_alert(self, alert_type: str, message: str, severity: str = 'info'):
        """Cria um alerta"""
        
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        
        self.alerts_history.append(alert)
        
        # Log baseado na severidade
        if severity == 'critical':
            self.logger.critical(f"ALERTA: {message}")
        elif severity == 'warning':
            self.logger.warning(f"ALERTA: {message}")
        else:
            self.logger.info(f"ALERTA: {message}")
            
    def _save_metrics_snapshot(self):
        """Salva snapshot das métricas"""
        
        with self._lock:
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'metrics': self._serialize_metrics(self.metrics)
            }
            
            self.metrics_history.append(snapshot)
            
            # Limitar histórico
            max_history = 3600  # 1 hora de snapshots
            if len(self.metrics_history) > max_history:
                self.metrics_history = self.metrics_history[-max_history:]
                
    def _serialize_metrics(self, metrics: Dict) -> Dict:
        """Serializa métricas para JSON"""
        
        serialized = {}
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                serialized[key] = self._serialize_metrics(value)
            elif isinstance(value, deque):
                serialized[key] = {
                    'values': list(value)[-10:],  # Últimos 10 valores
                    'avg': np.mean(list(value)) if value else 0,
                    'max': np.max(list(value)) if value else 0
                }
            elif isinstance(value, defaultdict):
                serialized[key] = dict(value)
            elif isinstance(value, datetime):
                serialized[key] = value.isoformat()
            else:
                serialized[key] = value
                
        return serialized
        
    def get_current_metrics(self) -> Dict:
        """Retorna métricas atuais"""
        with self._lock:
            return self._serialize_metrics(self.metrics)
            
    def get_alerts(self, hours: int = 1) -> List[Dict]:
        """Retorna alertas recentes"""
        
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_alerts = [
                alert for alert in self.alerts_history
                if alert['timestamp'] > cutoff
            ]
            
        return recent_alerts
        
    def generate_report(self) -> Dict:
        """Gera relatório completo"""
        
        with self._lock:
            report = {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': self.metrics['system']['uptime_seconds'],
                'current_metrics': self.get_current_metrics(),
                'recent_alerts': self.get_alerts(hours=1),
                'performance_summary': self._generate_performance_summary()
            }
            
        return report
        
    def _generate_performance_summary(self) -> Dict:
        """Gera resumo de performance"""
        
        summary = {
            'data_throughput': {
                'trades_per_second': self.metrics['data_flow'].get('trades_per_second', 0),
                'predictions_per_second': self.metrics['data_flow'].get('predictions_per_second', 0)
            },
            'latencies': {},
            'error_rate': 0,
            'prediction_distribution': {}
        }
        
        # Latências médias
        for operation in ['trade_processing_ms', 'prediction_generation_ms']:
            if f'{operation}_avg' in self.metrics:
                summary['latencies'][operation] = {
                    'avg': self.metrics[f'{operation}_avg'],
                    'max': self.metrics[f'{operation}_max']
                }
                
        # Taxa de erro
        total_errors = sum(self.metrics['errors'].values())
        total_operations = max(1, self.metrics['predictions']['total'])
        summary['error_rate'] = total_errors / total_operations
        
        # Distribuição de predições
        summary['prediction_distribution'] = {
            'by_regime': dict(self.metrics['predictions']['by_regime']),
            'by_action': dict(self.metrics['predictions']['by_action'])
        }
        
        return summary


def main():
    """Teste do SystemMonitorV3"""
    
    print("="*60)
    print("TESTE DO SYSTEM MONITOR V3")
    print("="*60)
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Criar monitor
    monitor = SystemMonitorV3()
    
    # Iniciar monitoramento
    monitor.start()
    
    # Simular atividade
    print("\nSimulando atividade do sistema...")
    
    # Simular latências
    for i in range(50):
        monitor.record_latency('trade_processing_ms', np.random.uniform(5, 50))
        monitor.record_latency('prediction_generation_ms', np.random.uniform(10, 100))
        time.sleep(0.1)
    
    # Simular predições
    for i in range(20):
        prediction = {
            'regime': np.random.choice(['trend_up', 'trend_down', 'range']),
            'action': np.random.choice(['buy', 'sell', 'hold']),
            'confidence': np.random.uniform(0.5, 0.9)
        }
        monitor.record_prediction(prediction)
        time.sleep(0.1)
    
    # Simular alguns erros
    monitor.record_error('prediction_engine', 'model_not_found')
    monitor.record_error('realtime_processor', 'data_timeout')
    
    # Aguardar um pouco
    time.sleep(2)
    
    # Gerar relatório
    report = monitor.generate_report()
    
    print("\nRELATÓRIO DO SISTEMA:")
    print(f"Uptime: {report['uptime_seconds']:.1f} segundos")
    
    # Performance
    perf = report['performance_summary']
    print(f"\nThroughput:")
    print(f"  Trades/s: {perf['data_throughput']['trades_per_second']:.2f}")
    print(f"  Predições/s: {perf['data_throughput']['predictions_per_second']:.2f}")
    
    print(f"\nLatências:")
    for op, stats in perf['latencies'].items():
        print(f"  {op}: avg={stats['avg']:.1f}ms, max={stats['max']:.1f}ms")
    
    print(f"\nTaxa de erro: {perf['error_rate']:.1%}")
    
    # Alertas
    alerts = report['recent_alerts']
    if alerts:
        print(f"\nAlertas recentes: {len(alerts)}")
        for alert in alerts[-3:]:  # Últimos 3
            print(f"  [{alert['severity']}] {alert['message']}")
    
    # Parar monitor
    monitor.stop()
    
    print("\n[OK] Teste concluído!")


if __name__ == "__main__":
    main()