from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import deque

class AlertingSystem:
    """Sistema inteligente de alertas"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_history = deque(maxlen=1000)
        self.alert_rules = self._load_alert_rules()
        self.notification_channels = self._setup_notification_channels()
        self.alert_aggregator = AlertAggregator()
        
    def _load_alert_rules(self) -> Dict[str, AlertRule]:
        """Carrega regras de alerta configuradas"""
        rules = {}
        
        # Regras de trading
        rules['max_drawdown'] = AlertRule(
            name='max_drawdown',
            condition=lambda m: m.get('trading', {}).get('drawdown', 0) > 0.05,
            severity='critical',
            message_template='Drawdown excedeu limite: {value:.2%}',
            cooldown_minutes=5
        )
        
        rules['consecutive_losses'] = AlertRule(
            name='consecutive_losses',
            condition=lambda m: m.get('trading', {}).get('consecutive_losses', 0) > 5,
            severity='high',
            message_template='Múltiplas perdas consecutivas: {value}',
            cooldown_minutes=15
        )
        
        # Regras de sistema
        rules['high_latency'] = AlertRule(
            name='high_latency',
            condition=lambda m: m.get('system', {}).get('latency_ms', 0) > 100,
            severity='medium',
            message_template='Latência alta detectada: {value}ms',
            cooldown_minutes=5
        )
        
        rules['connection_lost'] = AlertRule(
            name='connection_lost',
            condition=lambda m: not m.get('system', {}).get('connection_status', False),
            severity='critical',
            message_template='Conexão perdida com {source}',
            cooldown_minutes=1
        )
        
        # Regras de ML
        rules['model_drift'] = AlertRule(
            name='model_drift',
            condition=lambda m: m.get('model', {}).get('drift_score', 0) > 0.15,
            severity='high',
            message_template='Drift significativo detectado: score {value:.3f}',
            cooldown_minutes=30
        )
        
        rules['low_confidence'] = AlertRule(
            name='low_confidence',
            condition=lambda m: m.get('model', {}).get('avg_confidence', 1) < 0.5,
            severity='medium',
            message_template='Confiança do modelo baixa: {value:.2%}',
            cooldown_minutes=10
        )
        
        return rules
        
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Verifica todas as regras de alerta"""
        triggered_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            if rule.should_trigger(metrics):
                alert = Alert(
                    rule=rule,
                    metrics=metrics,
                    timestamp=datetime.now()
                )
                triggered_alerts.append(alert)
                
        # Agregar alertas similares
        aggregated = self.alert_aggregator.aggregate(triggered_alerts)
        
        # Processar e enviar alertas
        for alert in aggregated:
            self._process_alert(alert)
            
        return aggregated
        
    def _process_alert(self, alert: Alert):
        """Processa e envia alerta"""
        # Adicionar ao histórico
        self.alert_history.append(alert)
        
        # Determinar canais baseado na severidade
        channels = self._get_channels_for_severity(alert.severity)
        
        # Enviar para cada canal
        for channel in channels:
            try:
                channel.send(alert)
            except Exception as e:
                print(f"Erro enviando alerta para {channel.name}: {e}")
                
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Retorna resumo de alertas"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alert_history 
                        if a.timestamp > cutoff_time]
        
        if not recent_alerts:
            return {'total': 0, 'by_severity': {}, 'by_type': {}}
            
        summary = {
            'total': len(recent_alerts),
            'by_severity': {},
            'by_type': {},
            'most_frequent': [],
            'trend': self._calculate_alert_trend(recent_alerts)
        }
        
        # Agrupar por severidade
        for alert in recent_alerts:
            severity = alert.severity
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            alert_type = alert.rule.name
            summary['by_type'][alert_type] = summary['by_type'].get(alert_type, 0) + 1
            
        # Top 5 alertas mais frequentes
        summary['most_frequent'] = sorted(
            summary['by_type'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return summary