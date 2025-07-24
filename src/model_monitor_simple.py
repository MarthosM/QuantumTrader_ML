import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import deque

class MLModelMonitor:
    """Monitor básico para modelos de Machine Learning"""
    
    def __init__(self, model_manager=None, feature_engine=None):
        self.model_manager = model_manager
        self.feature_engine = feature_engine
        self.prediction_history = deque(maxlen=1000)
        
    def monitor_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Monitora uma predição individual"""
        monitor_data = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'confidence': prediction.get('confidence', 0),
            'action': prediction.get('action', 'HOLD')
        }
        
        self.prediction_history.append(monitor_data)
        return monitor_data
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Retorna resumo de performance"""
        if not self.prediction_history:
            return {'status': 'No data available'}
            
        predictions = list(self.prediction_history)
        
        return {
            'total_predictions': len(predictions),
            'avg_confidence': sum(p.get('confidence', 0) for p in predictions) / len(predictions),
            'last_prediction': predictions[-1] if predictions else None,
            'status': 'active'
        }
        
    def reset_monitoring(self):
        """Reset do monitoring"""
        self.prediction_history.clear()
        
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Status do monitoramento"""
        return {
            'active': True,
            'predictions_monitored': len(self.prediction_history),
            'last_update': datetime.now().isoformat()
        }
