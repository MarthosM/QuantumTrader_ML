"""
Sistema de Monitoramento para Trading Adaptativo
Monitora performance e saúde do sistema de aprendizado contínuo
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import queue
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict

class AdaptiveMonitor:
    """
    Monitor avançado para sistema de trading adaptativo
    Rastreia métricas, performance e saúde do sistema
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configurações
        self.metrics_window = config.get('metrics_window', 1000)
        self.alert_thresholds = config.get('alert_thresholds', {
            'accuracy': 0.45,
            'drawdown': 0.15,
            'latency': 1000,  # ms
            'buffer_overflow': 0.9
        })
        
        # Buffers de métricas
        self.performance_metrics = deque(maxlen=self.metrics_window)
        self.system_metrics = deque(maxlen=self.metrics_window)
        self.model_metrics = defaultdict(lambda: deque(maxlen=self.metrics_window))
        
        # Estado
        self.alerts = []
        self.is_monitoring = False
        self.start_time = datetime.now()
        
        # Estatísticas agregadas
        self.daily_stats = defaultdict(dict)
        self.regime_stats = defaultdict(lambda: defaultdict(int))
        
        # Threading
        self.metrics_queue = queue.Queue(maxsize=10000)
        self.monitor_thread = None
        
    def start(self):
        """Inicia monitoramento"""
        
        self.logger.info("Iniciando sistema de monitoramento adaptativo")
        self.is_monitoring = True
        
        # Thread de processamento
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="AdaptiveMonitor"
        )
        self.monitor_thread.start()
        
        self.logger.info("[OK] Monitor iniciado")
        
    def stop(self):
        """Para monitoramento"""
        
        self.logger.info("Parando monitor...")
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
        # Salvar estatísticas finais
        self._save_final_report()
        
    def record_prediction(self, prediction_info: dict):
        """Registra predição para análise"""
        
        metric = {
            'timestamp': datetime.now(),
            'type': 'prediction',
            'signal': prediction_info.get('signal'),
            'confidence': prediction_info.get('confidence'),
            'regime': prediction_info.get('regime'),
            'model_type': prediction_info.get('model_type', 'current'),
            'latency': prediction_info.get('latency', 0)
        }
        
        if not self.metrics_queue.full():
            self.metrics_queue.put(metric)
            
    def record_trade(self, trade_info: dict):
        """Registra resultado de trade"""
        
        metric = {
            'timestamp': datetime.now(),
            'type': 'trade',
            'action': trade_info.get('action'),
            'price': trade_info.get('price'),
            'pnl': trade_info.get('pnl', 0),
            'signal_confidence': trade_info.get('confidence'),
            'regime': trade_info.get('regime'),
            'model_type': trade_info.get('model_type')
        }
        
        if not self.metrics_queue.full():
            self.metrics_queue.put(metric)
            
    def record_system_metric(self, metric_name: str, value: float, 
                           context: Optional[dict] = None):
        """Registra métrica do sistema"""
        
        metric = {
            'timestamp': datetime.now(),
            'type': 'system',
            'metric_name': metric_name,
            'value': value,
            'context': context or {}
        }
        
        if not self.metrics_queue.full():
            self.metrics_queue.put(metric)
            
    def record_model_update(self, model_info: dict):
        """Registra atualização de modelo"""
        
        metric = {
            'timestamp': datetime.now(),
            'type': 'model_update',
            'model_type': model_info.get('model_type'),
            'version': model_info.get('version'),
            'performance': model_info.get('performance', {}),
            'reason': model_info.get('reason')
        }
        
        if not self.metrics_queue.full():
            self.metrics_queue.put(metric)
            
    def _monitor_loop(self):
        """Loop principal de monitoramento"""
        
        last_analysis = datetime.now()
        analysis_interval = 60  # segundos
        
        while self.is_monitoring:
            try:
                # Processar métricas da fila
                while not self.metrics_queue.empty():
                    metric = self.metrics_queue.get(timeout=0.1)
                    self._process_metric(metric)
                
                # Análise periódica
                if (datetime.now() - last_analysis).total_seconds() > analysis_interval:
                    self._analyze_performance()
                    self._check_alerts()
                    last_analysis = datetime.now()
                
                # Pequena pausa
                threading.Event().wait(0.1)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Erro no monitor: {e}")
                
    def _process_metric(self, metric: dict):
        """Processa métrica individual"""
        
        metric_type = metric.get('type')
        
        if metric_type == 'prediction':
            self.performance_metrics.append(metric)
            self._update_regime_stats(metric)
            
        elif metric_type == 'trade':
            self.performance_metrics.append(metric)
            self._update_daily_stats(metric)
            
        elif metric_type == 'system':
            self.system_metrics.append(metric)
            
        elif metric_type == 'model_update':
            model_type = metric.get('model_type')
            self.model_metrics[model_type].append(metric)
            
    def _update_regime_stats(self, metric: dict):
        """Atualiza estatísticas por regime"""
        
        regime = metric.get('regime', 'undefined')
        signal = metric.get('signal', 0)
        
        self.regime_stats[regime]['total'] += 1
        
        if signal == 1:
            self.regime_stats[regime]['buy'] += 1
        elif signal == -1:
            self.regime_stats[regime]['sell'] += 1
        else:
            self.regime_stats[regime]['hold'] += 1
            
    def _update_daily_stats(self, metric: dict):
        """Atualiza estatísticas diárias"""
        
        date = metric['timestamp'].date()
        
        if 'trades' not in self.daily_stats[date]:
            self.daily_stats[date] = {
                'trades': 0,
                'wins': 0,
                'pnl': 0,
                'volume': 0
            }
        
        self.daily_stats[date]['trades'] += 1
        
        pnl = metric.get('pnl', 0)
        if pnl > 0:
            self.daily_stats[date]['wins'] += 1
        self.daily_stats[date]['pnl'] += pnl
        
    def _analyze_performance(self):
        """Analisa performance recente"""
        
        if not self.performance_metrics:
            return
            
        # Converter para DataFrame
        recent_trades = [m for m in self.performance_metrics if m['type'] == 'trade']
        
        if recent_trades:
            # Calcular métricas
            total_trades = len(recent_trades)
            wins = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            # Calcular drawdown
            pnls = [t.get('pnl', 0) for t in recent_trades]
            cumulative = np.cumsum(pnls)
            if len(cumulative) > 0:
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / (running_max + 1e-10)
                max_drawdown = abs(drawdown.min())
            else:
                max_drawdown = 0
            
            # Log estatísticas
            self.logger.info(f"\n{'='*60}")
            self.logger.info("ANÁLISE DE PERFORMANCE")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Trades recentes: {total_trades}")
            self.logger.info(f"Win rate: {win_rate:.2%}")
            self.logger.info(f"Max drawdown: {max_drawdown:.2%}")
            
            # Análise por regime
            regime_performance = self._analyze_by_regime(recent_trades)
            for regime, stats in regime_performance.items():
                self.logger.info(f"\n{regime.upper()}:")
                self.logger.info(f"  Trades: {stats['trades']}")
                self.logger.info(f"  Win rate: {stats['win_rate']:.2%}")
                self.logger.info(f"  Avg P&L: ${stats['avg_pnl']:.2f}")
                
    def _analyze_by_regime(self, trades: List[dict]) -> Dict:
        """Analisa performance por regime"""
        
        regime_perf = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': []})
        
        for trade in trades:
            regime = trade.get('regime', 'undefined')
            regime_perf[regime]['trades'] += 1
            
            pnl = trade.get('pnl', 0)
            if pnl > 0:
                regime_perf[regime]['wins'] += 1
            regime_perf[regime]['pnl'].append(pnl)
        
        # Calcular estatísticas
        results = {}
        for regime, stats in regime_perf.items():
            if stats['trades'] > 0:
                results[regime] = {
                    'trades': stats['trades'],
                    'win_rate': stats['wins'] / stats['trades'],
                    'avg_pnl': np.mean(stats['pnl'])
                }
        
        return results
        
    def _check_alerts(self):
        """Verifica condições de alerta"""
        
        alerts = []
        
        # Verificar accuracy
        recent_trades = [m for m in self.performance_metrics 
                        if m['type'] == 'trade'][-100:]
        
        if recent_trades:
            wins = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
            accuracy = wins / len(recent_trades)
            
            if accuracy < self.alert_thresholds['accuracy']:
                alerts.append({
                    'level': 'WARNING',
                    'type': 'LOW_ACCURACY',
                    'message': f'Accuracy baixa: {accuracy:.2%}',
                    'timestamp': datetime.now()
                })
        
        # Verificar latência
        recent_predictions = [m for m in self.performance_metrics 
                            if m['type'] == 'prediction'][-50:]
        
        if recent_predictions:
            latencies = [p.get('latency', 0) for p in recent_predictions]
            avg_latency = np.mean(latencies)
            
            if avg_latency > self.alert_thresholds['latency']:
                alerts.append({
                    'level': 'WARNING',
                    'type': 'HIGH_LATENCY',
                    'message': f'Latência alta: {avg_latency:.0f}ms',
                    'timestamp': datetime.now()
                })
        
        # Verificar buffers do sistema
        system_metrics = list(self.system_metrics)
        buffer_metrics = [m for m in system_metrics 
                         if m.get('metric_name') == 'buffer_usage']
        
        if buffer_metrics:
            latest_buffer = buffer_metrics[-1].get('value', 0)
            if latest_buffer > self.alert_thresholds['buffer_overflow']:
                alerts.append({
                    'level': 'CRITICAL',
                    'type': 'BUFFER_OVERFLOW',
                    'message': f'Buffer próximo do limite: {latest_buffer:.0%}',
                    'timestamp': datetime.now()
                })
        
        # Processar alertas
        for alert in alerts:
            self._handle_alert(alert)
            
    def _handle_alert(self, alert: dict):
        """Processa alerta"""
        
        self.alerts.append(alert)
        
        # Log baseado no nível
        if alert['level'] == 'CRITICAL':
            self.logger.error(f"[ALERTA CRÍTICO] {alert['message']}")
        elif alert['level'] == 'WARNING':
            self.logger.warning(f"[ALERTA] {alert['message']}")
        else:
            self.logger.info(f"[INFO] {alert['message']}")
            
    def get_dashboard_data(self) -> Dict:
        """Retorna dados para dashboard"""
        
        # Métricas recentes
        recent_trades = [m for m in self.performance_metrics 
                        if m['type'] == 'trade'][-100:]
        
        if recent_trades:
            wins = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
            total_pnl = sum(t.get('pnl', 0) for t in recent_trades)
            win_rate = wins / len(recent_trades)
        else:
            wins = total_pnl = win_rate = 0
        
        # Tempo de execução
        uptime = datetime.now() - self.start_time
        
        # Status dos modelos
        model_status = {}
        for model_type, metrics in self.model_metrics.items():
            if metrics:
                latest = metrics[-1]
                model_status[model_type] = {
                    'version': latest.get('version', 0),
                    'last_update': latest['timestamp'].isoformat()
                }
        
        return {
            'system': {
                'uptime': str(uptime),
                'start_time': self.start_time.isoformat(),
                'total_predictions': len([m for m in self.performance_metrics 
                                        if m['type'] == 'prediction']),
                'total_trades': len(recent_trades)
            },
            'performance': {
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'recent_wins': wins,
                'recent_trades': len(recent_trades)
            },
            'models': model_status,
            'alerts': self.alerts[-10:],  # Últimos 10 alertas
            'regime_distribution': dict(self.regime_stats)
        }
        
    def generate_report(self, output_path: Optional[str] = None) -> Dict:
        """Gera relatório detalhado"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_dashboard_data(),
            'daily_performance': dict(self.daily_stats),
            'regime_analysis': dict(self.regime_stats),
            'alerts_history': self.alerts
        }
        
        # Análise detalhada de trades
        trades = [m for m in self.performance_metrics if m['type'] == 'trade']
        if trades:
            trades_df = pd.DataFrame(trades)
            
            # Estatísticas por modelo
            model_stats = {}
            for model_type in trades_df['model_type'].unique():
                model_trades = trades_df[trades_df['model_type'] == model_type]
                wins = (model_trades['pnl'] > 0).sum()
                
                model_stats[model_type] = {
                    'trades': len(model_trades),
                    'win_rate': wins / len(model_trades) if len(model_trades) > 0 else 0,
                    'total_pnl': model_trades['pnl'].sum(),
                    'avg_pnl': model_trades['pnl'].mean()
                }
            
            report['model_comparison'] = model_stats
        
        # Salvar se path fornecido
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Relatório salvo em: {output_path}")
        
        return report
        
    def plot_performance(self, save_path: Optional[str] = None):
        """Gera gráficos de performance"""
        
        if not self.performance_metrics:
            self.logger.warning("Sem dados para plotar")
            return
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dashboard de Performance - Trading Adaptativo', fontsize=16)
        
        # 1. Equity curve
        trades = [m for m in self.performance_metrics if m['type'] == 'trade']
        if trades:
            pnls = [t.get('pnl', 0) for t in trades]
            cumulative = np.cumsum(pnls)
            
            ax = axes[0, 0]
            ax.plot(cumulative, label='P&L Cumulativo')
            ax.fill_between(range(len(cumulative)), 0, cumulative, alpha=0.3)
            ax.set_title('Curva de Equity')
            ax.set_xlabel('Trades')
            ax.set_ylabel('P&L ($)')
            ax.legend()
        
        # 2. Win rate por regime
        regime_perf = self._analyze_by_regime(trades)
        if regime_perf:
            ax = axes[0, 1]
            regimes = list(regime_perf.keys())
            win_rates = [regime_perf[r]['win_rate'] for r in regimes]
            
            bars = ax.bar(regimes, win_rates)
            ax.set_title('Win Rate por Regime')
            ax.set_ylabel('Win Rate (%)')
            ax.set_ylim(0, 1)
            
            # Adicionar valores nas barras
            for bar, wr in zip(bars, win_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{wr:.1%}', ha='center', va='bottom')
        
        # 3. Distribuição de P&L
        if trades:
            ax = axes[1, 0]
            pnls = [t.get('pnl', 0) for t in trades]
            
            ax.hist(pnls, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', label='Break-even')
            ax.set_title('Distribuição de P&L')
            ax.set_xlabel('P&L ($)')
            ax.set_ylabel('Frequência')
            ax.legend()
        
        # 4. Performance diária
        if self.daily_stats:
            ax = axes[1, 1]
            dates = sorted(self.daily_stats.keys())[-30:]  # Últimos 30 dias
            daily_pnls = [self.daily_stats[d].get('pnl', 0) for d in dates]
            
            ax.bar(range(len(dates)), daily_pnls, 
                  color=['green' if p > 0 else 'red' for p in daily_pnls])
            ax.set_title('P&L Diário (Últimos 30 dias)')
            ax.set_xlabel('Dias')
            ax.set_ylabel('P&L ($)')
            
            # Rotacionar labels se muitas datas
            if len(dates) > 15:
                ax.set_xticks(range(0, len(dates), 5))
                ax.set_xticklabels([d.strftime('%d/%m') for d in dates[::5]], 
                                  rotation=45)
        
        plt.tight_layout()
        
        # Salvar ou mostrar
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Gráficos salvos em: {save_path}")
        else:
            plt.show()
            
    def _save_final_report(self):
        """Salva relatório final ao parar"""
        
        try:
            # Criar diretório de reports
            reports_dir = Path('reports/adaptive_monitoring')
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Nome do arquivo com timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = reports_dir / f"final_report_{timestamp}.json"
            
            # Gerar e salvar relatório
            self.generate_report(str(report_path))
            
            # Salvar gráficos
            plot_path = reports_dir / f"performance_plots_{timestamp}.png"
            self.plot_performance(str(plot_path))
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar relatório final: {e}")