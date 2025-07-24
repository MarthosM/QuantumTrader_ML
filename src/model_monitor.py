import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import deque

class ModelDriftDetector:
    """Detector simples de drift de modelo"""
    def detect_drift(self, current_features, historical_features):
        return False

class MLModelMonitor:
    """Monitor específico para modelos de Machine Learning"""
    
    def __init__(self, model_manager, feature_engine):
        self.model_manager = model_manager
        self.feature_engine = feature_engine
        self.performance_history = deque(maxlen=1000)
        self.prediction_history = deque(maxlen=1000)
        self.feature_importance_history = {}
        self.drift_detector = ModelDriftDetector()
        
    def monitor_prediction(self, features: pd.DataFrame, prediction: Dict[str, Any], 
                         actual_outcome: Optional[Dict[str, Any]] = None):
        """Monitora uma predição individual"""
        
        monitor_data = {
            'timestamp': datetime.now(),
            'features_snapshot': features.iloc[-1].to_dict(),
            'prediction': prediction,
            'model_confidence': prediction.get('confidence', 0),
            'ensemble_agreement': self._calculate_ensemble_agreement(prediction),
            'feature_quality': self._assess_feature_quality(features)
        }
        
        # Se temos o resultado real, calcular accuracy
        if actual_outcome:
            monitor_data['actual'] = actual_outcome
            monitor_data['accuracy'] = self._calculate_accuracy(prediction, actual_outcome)
            
        self.prediction_history.append(monitor_data)
        
        # Verificar drift
        if len(self.prediction_history) > 100:
            drift_score = self.drift_detector.calculate_drift(
                self.prediction_history,
                features
            )
            monitor_data['drift_score'] = drift_score
            
        return monitor_data
        
    def _calculate_ensemble_agreement(self, prediction: Dict[str, Any]) -> float:
        """Calcula concordância entre modelos do ensemble"""
        if 'model_predictions' not in prediction:
            return 1.0
            
        model_preds = prediction['model_predictions']
        if len(model_preds) < 2:
            return 1.0
            
        # Calcular desvio padrão das predições
        directions = [p.get('direction', 0) for p in model_preds.values()]
        agreement = 1 - np.std(directions)
        
        return max(0, min(1, agreement))
        
    def _assess_feature_quality(self, features: pd.DataFrame) -> Dict[str, float]:
        """Avalia qualidade das features"""
        quality_metrics = {
            'missing_ratio': features.isna().sum().sum() / features.size,
            'zero_ratio': (features == 0).sum().sum() / features.size,
            'outlier_ratio': self._calculate_outlier_ratio(features),
            'correlation_stability': self._check_correlation_stability(features)
        }
        
        # Score geral de qualidade
        quality_metrics['overall_score'] = 1 - np.mean([
            quality_metrics['missing_ratio'],
            quality_metrics['zero_ratio'],
            quality_metrics['outlier_ratio'] / 10  # Normalizar
        ])
        
        return quality_metrics
        
    def get_model_performance_report(self) -> Dict[str, Any]:
        """Gera relatório completo de performance dos modelos"""
        if not self.prediction_history:
            return {}
            
        # Converter histórico para DataFrame
        df = pd.DataFrame(list(self.prediction_history))
        
        report = {
            'summary': {
                'total_predictions': len(df),
                'period': {
                    'start': df['timestamp'].min(),
                    'end': df['timestamp'].max()
                },
                'average_confidence': df['model_confidence'].mean(),
                'ensemble_agreement': df['ensemble_agreement'].mean()
            },
            'accuracy_metrics': self._calculate_accuracy_metrics(df),
            'feature_analysis': self._analyze_feature_importance(df),
            'drift_analysis': self._analyze_drift_patterns(df),
            'model_specific': self._get_model_specific_metrics()
        }
        
        return report
        
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detecta anomalias no comportamento dos modelos"""
        anomalies = []
        
        if len(self.prediction_history) < 50:
            return anomalies
            
        recent_predictions = list(self.prediction_history)[-50:]
        
        # Verificar queda súbita de confidence
        confidences = [p['model_confidence'] for p in recent_predictions]
        if np.mean(confidences[-10:]) < np.mean(confidences[:-10]) * 0.7:
            anomalies.append({
                'type': 'confidence_drop',
                'severity': 'high',
                'description': 'Queda significativa na confiança do modelo',
                'value': np.mean(confidences[-10:])
            })
            
        # Verificar divergência no ensemble
        agreements = [p['ensemble_agreement'] for p in recent_predictions]
        if np.mean(agreements[-10:]) < 0.6:
            anomalies.append({
                'type': 'ensemble_divergence',
                'severity': 'medium',
                'description': 'Modelos do ensemble estão divergindo',
                'value': np.mean(agreements[-10:])
            })
            
        # Verificar feature quality
        quality_scores = [p['feature_quality']['overall_score'] 
                         for p in recent_predictions if 'feature_quality' in p]
        if quality_scores and np.mean(quality_scores[-10:]) < 0.8:
            anomalies.append({
                'type': 'feature_quality',
                'severity': 'medium',
                'description': 'Qualidade das features está baixa',
                'value': np.mean(quality_scores[-10:])
            })
            
        return anomalies