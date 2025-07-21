# src/training/metrics/performance_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

class PerformanceAnalyzer:
    """Analisador detalhado de performance do sistema"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_model_performance(self, predictions: Dict[str, np.ndarray],
                                actuals: pd.Series,
                                timestamps: pd.DatetimeIndex) -> Dict:
        """
        Analisa performance detalhada dos modelos
        
        Args:
            predictions: Dicionário com predições de cada modelo
            actuals: Valores reais
            timestamps: Timestamps das predições
            
        Returns:
            Análise completa de performance
        """
        analysis = {
            'individual_models': {},
            'ensemble_analysis': {},
            'temporal_analysis': {},
            'error_analysis': {},
            'feature_importance': {}
        }
        
        # Análise individual de cada modelo
        for model_name, preds in predictions.items():
            analysis['individual_models'][model_name] = self._analyze_single_model(
                preds, actuals
            )
        
        # Análise do ensemble
        if len(predictions) > 1:
            analysis['ensemble_analysis'] = self._analyze_ensemble_performance(
                predictions, actuals
            )
        
        # Análise temporal
        analysis['temporal_analysis'] = self._analyze_temporal_performance(
            predictions, actuals, timestamps
        )
        
        # Análise de erros
        analysis['error_analysis'] = self._analyze_prediction_errors(
            predictions, actuals
        )
        
        return analysis
    
    def _analyze_single_model(self, predictions: np.ndarray,
                            actuals: pd.Series) -> Dict:
        """Analisa performance de um único modelo"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
        
        # Converter para classes se necessário
        if len(predictions.shape) > 1:
            y_pred = np.argmax(predictions, axis=1)
            y_proba = predictions
        else:
            y_pred = predictions
            y_proba = None
        
        metrics = {
            'accuracy': accuracy_score(actuals, y_pred),
            'precision': precision_score(actuals, y_pred, average='weighted'),
            'recall': recall_score(actuals, y_pred, average='weighted'),
            'f1_score': f1_score(actuals, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(actuals, y_pred).tolist()
        }
        
        # Métricas por classe
        for class_idx in range(3):
            class_mask = actuals == class_idx
            if class_mask.sum() > 0:
                metrics[f'class_{class_idx}_accuracy'] = accuracy_score(
                    actuals[class_mask], y_pred[class_mask]
                )
        
        # Análise de probabilidades
        if y_proba is not None:
            metrics['avg_confidence'] = np.mean(np.max(y_proba, axis=1))
            metrics['confidence_distribution'] = {
                'mean': float(np.mean(np.max(y_proba, axis=1))),
                'std': float(np.std(np.max(y_proba, axis=1))),
                'min': float(np.min(np.max(y_proba, axis=1))),
                'max': float(np.max(np.max(y_proba, axis=1)))
            }
        
        return metrics
    
    def _analyze_ensemble_performance(self, predictions: Dict[str, np.ndarray],
                                    actuals: pd.Series) -> Dict:
        """Analisa performance do ensemble e diversidade"""
        # Converter todas as predições para classes
        pred_classes = {}
        for name, preds in predictions.items():
            if len(preds.shape) > 1:
                pred_classes[name] = np.argmax(preds, axis=1)
            else:
                pred_classes[name] = preds
        
        # Criar matriz de predições
        pred_matrix = np.array(list(pred_classes.values())).T
        
        # Diversidade do ensemble
        diversity_metrics = {
            'disagreement_rate': self._calculate_disagreement_rate(pred_matrix),
            'correlation_matrix': self._calculate_prediction_correlations(pred_classes),
            'q_statistic': self._calculate_q_statistic(pred_classes, actuals),
            'kappa_statistic': self._calculate_kappa_statistic(pred_classes, actuals)
        }
        
        # Performance por votação
        voting_analysis = {
            'majority_vote': self._analyze_majority_voting(pred_matrix, actuals),
            'weighted_vote': self._analyze_weighted_voting(predictions, actuals),
            'oracle_accuracy': self._calculate_oracle_accuracy(pred_matrix, actuals)
        }
        
        return {
            'diversity': diversity_metrics,
            'voting': voting_analysis
        }
    
    def _analyze_temporal_performance(self, predictions: Dict[str, np.ndarray],
                                    actuals: pd.Series,
                                    timestamps: pd.DatetimeIndex) -> Dict:
        """Analisa performance ao longo do tempo"""
        # Usar primeiro modelo para análise temporal
        first_model = list(predictions.keys())[0]
        preds = predictions[first_model]
        
        if len(preds.shape) > 1:
            y_pred = np.argmax(preds, axis=1)
        else:
            y_pred = preds
        
        # Criar DataFrame temporal
        df = pd.DataFrame({
            'timestamp': timestamps,
            'actual': actuals,
            'predicted': y_pred,
            'correct': y_pred == actuals
        })
        
        # Análise por período
        temporal_analysis = {
            'hourly': self._analyze_by_period(df, 'H'),
            'daily': self._analyze_by_period(df, 'D'),
            'weekly': self._analyze_by_period(df, 'W'),
            'by_hour_of_day': self._analyze_by_hour_of_day(df),
            'by_day_of_week': self._analyze_by_day_of_week(df)
        }
        
        # Análise de tendência
        temporal_analysis['performance_trend'] = self._analyze_performance_trend(df)
        
        return temporal_analysis
    
    def _analyze_prediction_errors(self, predictions: Dict[str, np.ndarray],
                                 actuals: pd.Series) -> Dict:
        """Analisa padrões de erro nas predições"""
        error_analysis = {}
        
        for model_name, preds in predictions.items():
            if len(preds.shape) > 1:
                y_pred = np.argmax(preds, axis=1)
                y_proba = preds
            else:
                y_pred = preds
                y_proba = None
            
            # Tipos de erro
            errors = y_pred != actuals
            error_mask = errors
            
            # Análise de erros por tipo
            error_types = {
                'false_buys': ((y_pred == 2) & (actuals != 2)).sum(),
                'false_sells': ((y_pred == 0) & (actuals != 0)).sum(),
                'missed_buys': ((y_pred != 2) & (actuals == 2)).sum(),
                'missed_sells': ((y_pred != 0) & (actuals == 0)).sum()
            }
            
            # Análise de confiança nos erros
            confidence_analysis = {}
            if y_proba is not None:
                max_proba = np.max(y_proba, axis=1)
                confidence_analysis = {
                    'avg_confidence_when_wrong': max_proba[error_mask].mean(),
                    'avg_confidence_when_right': max_proba[~error_mask].mean(),
                    'high_confidence_errors': (max_proba[error_mask] > 0.8).sum()
                }
            
            error_analysis[model_name] = {
                'total_errors': errors.sum(),
                'error_rate': errors.mean(),
                'error_types': error_types,
                'confidence_analysis': confidence_analysis
            }
        
        return error_analysis
    
    def _calculate_disagreement_rate(self, pred_matrix: np.ndarray) -> float:
        """Calcula taxa de discordância entre modelos"""
        n_samples = pred_matrix.shape[0]
        disagreements = 0
        
        for i in range(n_samples):
            unique_preds = len(np.unique(pred_matrix[i]))
            if unique_preds > 1:
                disagreements += 1
        
        return disagreements / n_samples
    
    def _calculate_prediction_correlations(self, predictions: Dict) -> Dict:
        """Calcula correlações entre predições dos modelos"""
        model_names = list(predictions.keys())
        n_models = len(model_names)
        
        correlation_matrix = np.zeros((n_models, n_models))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i <= j:
                    corr = np.corrcoef(predictions[model1], predictions[model2])[0, 1]
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr
        
        return {
            'matrix': correlation_matrix.tolist(),
            'model_names': model_names,
            'avg_correlation': np.mean(correlation_matrix[np.triu_indices(n_models, k=1)])
        }
    
    def _analyze_majority_voting(self, pred_matrix: np.ndarray,
                               actuals: pd.Series) -> Dict:
        """Analisa performance de votação por maioria"""
        from scipy import stats
        
        # Votação por maioria
        majority_pred = stats.mode(pred_matrix, axis=1)[0].flatten()
        
        from sklearn.metrics import accuracy_score, f1_score
        
        return {
            'accuracy': accuracy_score(actuals, majority_pred),
            'f1_score': f1_score(actuals, majority_pred, average='weighted')
        }
    
    def _calculate_oracle_accuracy(self, pred_matrix: np.ndarray,
                                 actuals: pd.Series) -> float:
        """Calcula accuracy do oracle (melhor possível do ensemble)"""
        oracle_correct = 0
        
        for i in range(len(actuals)):
            # Oracle acerta se algum modelo acertou
            if actuals.iloc[i] in pred_matrix[i]:
                oracle_correct += 1
        
        return oracle_correct / len(actuals)
    
    def _analyze_by_period(self, df: pd.DataFrame, period: str) -> Dict:
        """Analisa performance por período temporal"""
        df['period'] = df['timestamp'].dt.to_period(period)
        
        period_stats = df.groupby('period').agg({
            'correct': ['sum', 'count', 'mean']
        })
        
        return {
            'accuracy_by_period': period_stats['correct']['mean'].to_dict(),
            'trades_by_period': period_stats['correct']['count'].to_dict()
        }
    
    def _analyze_performance_trend(self, df: pd.DataFrame) -> Dict:
        """Analisa tendência de performance ao longo do tempo"""
        # Calcular média móvel de acurácia
        df['rolling_accuracy'] = df['correct'].rolling(window=100, min_periods=20).mean()
        
        # Detectar tendência
        if len(df) > 100:
            # Regressão linear simples
            x = np.arange(len(df))
            y = df['rolling_accuracy'].fillna(method='bfill').fillna(method='ffill')
            
            slope, intercept = np.polyfit(x, y, 1)
            
            trend = 'improving' if slope > 0 else 'declining'
            trend_strength = abs(slope) * 1000  # Normalizar
        else:
            trend = 'insufficient_data'
            trend_strength = 0
        
        return {
            'trend': trend,
            'trend_strength': trend_strength,
            'recent_performance': df['correct'].tail(100).mean() if len(df) > 100 else None
        }
    
    def generate_performance_report(self, analysis: Dict,
                                  output_path: str = 'performance_report.html'):
        """Gera relatório HTML detalhado de performance"""
        # Similar ao relatório de treinamento, mas focado em análise de performance
        pass