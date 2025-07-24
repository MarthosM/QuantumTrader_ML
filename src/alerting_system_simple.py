from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import deque

class AlertingSystem:
    """Sistema básico de alertas"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alert_history = deque(maxlen=1000)
        
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verifica e retorna alertas baseados nas métricas"""
        alerts = []
        
        # Verificação básica de drawdown
        drawdown = metrics.get('drawdown', 0)
        if drawdown > 0.05:  # 5% drawdown
            alerts.append({
                'type': 'max_drawdown',
                'severity': 'high',
                'message': f'Drawdown elevado: {drawdown:.2%}',
                'timestamp': datetime.now()
            })
        
        # Verificação de posições
        positions = metrics.get('positions_count', 0)
        max_positions = self.config.get('max_positions', 3)
        if positions > max_positions:
            alerts.append({
                'type': 'max_positions',
                'severity': 'medium', 
                'message': f'Muitas posições: {positions}/{max_positions}',
                'timestamp': datetime.now()
            })
            
        # Adicionar alertas ao histórico
        for alert in alerts:
            self.alert_history.append(alert)
            
        return alerts
        
    def send_alert(self, alert: Dict[str, Any]):
        """Processa e envia um alerta"""
        print(f"[{alert['timestamp']}] {alert['severity'].upper()}: {alert['message']}")
        
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Retorna alertas das últimas horas"""
        cutoff = datetime.now().timestamp() - (hours * 3600)
        return [a for a in self.alert_history if a['timestamp'].timestamp() > cutoff]
        
    def get_alert_summary(self) -> Dict[str, Any]:
        """Retorna resumo dos alertas"""
        recent = self.get_recent_alerts()
        return {
            'total_alerts': len(self.alert_history),
            'recent_alerts': len(recent),
            'high_severity': len([a for a in recent if a['severity'] == 'high']),
            'medium_severity': len([a for a in recent if a['severity'] == 'medium']),
            'status': 'active'
        }
        
    def clear_alerts(self):
        """Limpa histórico de alertas"""
        self.alert_history.clear()
