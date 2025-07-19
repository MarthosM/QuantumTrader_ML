import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from collections import deque
import threading
import json

class RealTimePerformanceMonitor:
    """Monitor de performance em tempo real para ML Trading"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Métricas em tempo real
        self.metrics = {
            'trades': deque(maxlen=1000),
            'predictions': deque(maxlen=1000),
            'latencies': deque(maxlen=1000),
            'errors': deque(maxlen=100)
        }
        
        # Agregações
        self.hourly_stats = {}
        self.daily_stats = {}
        
        # Estado
        self.monitoring_thread = None
        self.running = False
        self.update_interval = config.get('update_interval', 60)  # segundos
        
    def start_monitoring(self):
        """Inicia monitoramento em tempo real"""
        
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Monitor de performance iniciado")
        
    def record_trade(self, trade_data: Dict):
        """Registra dados de uma operação"""
        
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': trade_data['symbol'],
            'side': trade_data['side'],
            'price': trade_data['price'],
            'quantity': trade_data['quantity'],
            'pnl': trade_data.get('pnl', 0),
            'prediction_confidence': trade_data.get('confidence', 0)
        }
        
        self.metrics['trades'].append(trade_record)
        
    def record_prediction(self, prediction_data: Dict):
        """Registra dados de uma predição"""
        
        prediction_record = {
            'timestamp': datetime.now(),
            'model': prediction_data['model'],
            'direction': prediction_data['direction'],
            'confidence': prediction_data['confidence'],
            'features_used': prediction_data.get('features_used', []),
            'latency_ms': prediction_data.get('latency_ms', 0)
        }
        
        self.metrics['predictions'].append(prediction_record)
        self.metrics['latencies'].append(prediction_data.get('latency_ms', 0))
        
    def get_current_metrics(self) -> Dict:
        """Retorna métricas atuais do sistema"""
        
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        # Filtrar trades recentes
        recent_trades = [t for t in self.metrics['trades'] 
                        if t['timestamp'] > hour_ago]
        daily_trades = [t for t in self.metrics['trades'] 
                       if t['timestamp'] > day_ago]
        
        # Calcular métricas
        metrics = {
            'timestamp': now,
            'hourly_metrics': self._calculate_period_metrics(recent_trades),
            'daily_metrics': self._calculate_period_metrics(daily_trades),
            'system_metrics': self._calculate_system_metrics(),
            'model_metrics': self._calculate_model_metrics()
        }
        
        return metrics
        
    def _calculate_period_metrics(self, trades: List[Dict]) -> Dict:
        """Calcula métricas para um período"""
        
        if not trades:
            return {
                'trade_count': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
            
        # Extrair PnLs
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        
        # Calcular métricas
        return {
            'trade_count': len(trades),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(pnls),
            'max_drawdown': self._calculate_max_drawdown(pnls)
        }
        
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calcula Sharpe Ratio"""
        
        if len(returns) < 2:
            return 0
            
        returns_array = np.array(returns)
        if returns_array.std() == 0:
            return 0
            
        # Assumindo 252 dias de trading por ano
        return np.sqrt(252) * returns_array.mean() / returns_array.std()
        
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calcula drawdown máximo"""
        
        if not returns:
            return 0
            
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / np.maximum(running_max, 1)
        
        return abs(np.min(drawdown))
        
    def _calculate_system_metrics(self) -> Dict:
        """Calcula métricas do sistema"""
        
        recent_predictions = [p for p in self.metrics['predictions'] 
                            if p['timestamp'] > datetime.now() - timedelta(minutes=30)]
        recent_latencies = list(self.metrics['latencies'])[-100:]
        recent_errors = list(self.metrics['errors'])[-10:]
        
        return {
            'prediction_rate_per_min': len(recent_predictions) / 30 if recent_predictions else 0,
            'avg_latency_ms': np.mean(recent_latencies) if recent_latencies else 0,
            'max_latency_ms': np.max(recent_latencies) if recent_latencies else 0,
            'error_count_last_10': len(recent_errors),
            'memory_usage_mb': 0,  # Placeholder - pode ser implementado com psutil
            'cpu_usage_percent': 0  # Placeholder - pode ser implementado com psutil
        }
        
    def _calculate_model_metrics(self) -> Dict:
        """Calcula métricas por modelo"""
        
        model_stats = {}
        
        for prediction in self.metrics['predictions']:
            model_name = prediction['model']
            if model_name not in model_stats:
                model_stats[model_name] = {
                    'prediction_count': 0,
                    'avg_confidence': 0,
                    'avg_latency': 0,
                    'confidences': [],
                    'latencies': []
                }
            
            model_stats[model_name]['prediction_count'] += 1
            model_stats[model_name]['confidences'].append(prediction['confidence'])
            model_stats[model_name]['latencies'].append(prediction['latency_ms'])
        
        # Calcular médias
        for model_name, stats in model_stats.items():
            stats['avg_confidence'] = np.mean(stats['confidences']) if stats['confidences'] else 0
            stats['avg_latency'] = np.mean(stats['latencies']) if stats['latencies'] else 0
            # Remove listas temporárias
            del stats['confidences']
            del stats['latencies']
        
        return model_stats
        
    def _check_alerts(self, current_metrics: Dict):
        """Verifica alertas baseados nas métricas atuais"""
        
        try:
            hourly = current_metrics['hourly_metrics']
            daily = current_metrics['daily_metrics']
            system = current_metrics['system_metrics']
            
            # Alerta para win rate baixo
            if hourly['trade_count'] > 5 and hourly['win_rate'] < 0.4:
                self.logger.warning(f"Win rate baixo na última hora: {hourly['win_rate']:.2%}")
            
            # Alerta para drawdown alto
            if daily['max_drawdown'] > 0.05:  # 5%
                self.logger.warning(f"Drawdown alto no dia: {daily['max_drawdown']:.2%}")
            
            # Alerta para latência alta
            if system['avg_latency_ms'] > 1000:  # 1 segundo
                self.logger.warning(f"Latência média alta: {system['avg_latency_ms']:.1f}ms")
            
            # Alerta para muitos erros
            if system['error_count_last_10'] > 5:
                self.logger.warning(f"Muitos erros recentes: {system['error_count_last_10']}")
                
        except Exception as e:
            self.logger.error(f"Erro ao verificar alertas: {e}")
    
    def _save_metrics_snapshot(self, metrics: Dict):
        """Salva snapshot das métricas"""
        
        try:
            # Implementar salvamento se necessário
            # Por exemplo, em arquivo JSON ou banco de dados
            pass
        except Exception as e:
            self.logger.error(f"Erro ao salvar métricas: {e}")
        
    def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        
        while self.running:
            try:
                # Atualizar métricas
                current_metrics = self.get_current_metrics()
                
                # Verificar alertas
                self._check_alerts(current_metrics)
                
                # Salvar snapshot
                self._save_metrics_snapshot(current_metrics)
                
                # Aguardar próxima atualização
                threading.Event().wait(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Erro no loop de monitoramento: {e}")
                self.metrics['errors'].append({
                    'timestamp': datetime.now(),
                    'error': str(e)
                })