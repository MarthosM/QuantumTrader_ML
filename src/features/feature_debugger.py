"""
Feature Debugger - Analisa qualidade das features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime


class FeatureDebugger:
    """Debug e análise de qualidade das features"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.feature_stats = {}
        
    def analyze_features(self, features_df: pd.DataFrame) -> Dict:
        """Analisa qualidade das features"""
        if features_df.empty:
            return {'status': 'error', 'message': 'No features'}
            
        analysis = {
            'timestamp': datetime.now(),
            'total_features': len(features_df.columns),
            'total_rows': len(features_df),
            'nan_analysis': {},
            'variance_analysis': {},
            'correlation_analysis': {},
            'quality_score': 0.0
        }
        
        # 1. Análise de NaN
        nan_counts = features_df.isna().sum()
        nan_percentages = (nan_counts / len(features_df) * 100).round(2)
        
        analysis['nan_analysis'] = {
            'high_nan_features': nan_percentages[nan_percentages > 20].to_dict(),
            'total_nan_features': len(nan_counts[nan_counts > 0]),
            'avg_nan_percentage': nan_percentages.mean()
        }
        
        # 2. Análise de variância
        numeric_features = features_df.select_dtypes(include=[np.number])
        variances = numeric_features.var()
        low_variance = variances[variances < 0.0001]
        
        analysis['variance_analysis'] = {
            'low_variance_features': low_variance.index.tolist(),
            'zero_variance_features': variances[variances == 0].index.tolist(),
            'avg_variance': variances.mean()
        }
        
        # 3. Features mais importantes (por variação)
        if not numeric_features.empty:
            # Normalizar e calcular importância
            normalized = (numeric_features - numeric_features.mean()) / (numeric_features.std() + 1e-8)
            feature_importance = normalized.abs().mean().sort_values(ascending=False)
            
            analysis['top_features'] = feature_importance.head(20).to_dict()
            analysis['bottom_features'] = feature_importance.tail(10).to_dict()
        
        # 4. Correlação com retornos (se disponível)
        if 'returns_1' in features_df.columns:
            correlations = numeric_features.corrwith(features_df['returns_1']).abs()
            analysis['correlation_analysis'] = {
                'high_correlation': correlations[correlations > 0.3].to_dict(),
                'avg_correlation': correlations.mean()
            }
        
        # 5. Quality Score
        quality_factors = []
        
        # Penalizar por NaN
        nan_penalty = max(0, 1 - (analysis['nan_analysis']['avg_nan_percentage'] / 50))
        quality_factors.append(nan_penalty)
        
        # Penalizar por baixa variância
        low_var_ratio = len(low_variance) / len(variances) if len(variances) > 0 else 1
        variance_score = max(0, 1 - low_var_ratio)
        quality_factors.append(variance_score)
        
        # Bonus por correlação
        if 'avg_correlation' in analysis['correlation_analysis']:
            corr_bonus = min(1, analysis['correlation_analysis']['avg_correlation'] * 3)
            quality_factors.append(corr_bonus)
        
        analysis['quality_score'] = np.mean(quality_factors) if quality_factors else 0
        
        # Log summary
        self._log_analysis_summary(analysis)
        
        return analysis
    
    def suggest_improvements(self, analysis: Dict) -> List[str]:
        """Sugere melhorias baseadas na análise"""
        suggestions = []
        
        # NaN issues
        if analysis['nan_analysis']['avg_nan_percentage'] > 10:
            suggestions.append("Implementar melhor forward-fill ou interpolação para features com alto NaN")
            high_nan = analysis['nan_analysis']['high_nan_features']
            if high_nan:
                suggestions.append(f"Features críticas com NaN: {list(high_nan.keys())[:5]}")
        
        # Variance issues
        zero_var = analysis['variance_analysis']['zero_variance_features']
        if zero_var:
            suggestions.append(f"Remover features sem variância: {zero_var[:5]}")
        
        low_var = analysis['variance_analysis']['low_variance_features']
        if len(low_var) > 10:
            suggestions.append("Considerar normalização ou transformação para features de baixa variância")
        
        # Correlation
        if 'avg_correlation' in analysis['correlation_analysis']:
            if analysis['correlation_analysis']['avg_correlation'] < 0.05:
                suggestions.append("Features têm baixa correlação com retornos - revisar seleção")
        
        # Quality score
        if analysis['quality_score'] < 0.5:
            suggestions.append("Quality score baixo - revisar pipeline de features")
        
        return suggestions
    
    def _log_analysis_summary(self, analysis: Dict):
        """Log resumo da análise"""
        self.logger.info("="*60)
        self.logger.info("[FEATURE DEBUG] Análise de Qualidade")
        self.logger.info(f"Total features: {analysis['total_features']}")
        self.logger.info(f"NaN médio: {analysis['nan_analysis']['avg_nan_percentage']:.1f}%")
        self.logger.info(f"Features baixa variância: {len(analysis['variance_analysis']['low_variance_features'])}")
        self.logger.info(f"Quality Score: {analysis['quality_score']:.3f}")
        
        if 'top_features' in analysis:
            self.logger.info("\nTop 5 features por importância:")
            for feat, score in list(analysis['top_features'].items())[:5]:
                self.logger.info(f"  {feat}: {score:.3f}")
        
        self.logger.info("="*60)
    
    def create_feature_report(self, features_df: pd.DataFrame, save_path: str = None) -> str:
        """Cria relatório detalhado das features"""
        analysis = self.analyze_features(features_df)
        suggestions = self.suggest_improvements(analysis)
        
        report = f"""
# Feature Quality Report
Generated: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Features: {analysis['total_features']}
- Total Samples: {analysis['total_rows']}
- Quality Score: {analysis['quality_score']:.3f}/1.0

## NaN Analysis
- Average NaN: {analysis['nan_analysis']['avg_nan_percentage']:.1f}%
- Features with NaN: {analysis['nan_analysis']['total_nan_features']}

## Variance Analysis
- Low variance features: {len(analysis['variance_analysis']['low_variance_features'])}
- Zero variance features: {len(analysis['variance_analysis']['zero_variance_features'])}

## Top Features by Importance
"""
        if 'top_features' in analysis:
            for i, (feat, score) in enumerate(analysis['top_features'].items()):
                if i >= 10:
                    break
                report += f"{i+1}. {feat}: {score:.3f}\n"
        
        report += "\n## Improvement Suggestions\n"
        for i, suggestion in enumerate(suggestions, 1):
            report += f"{i}. {suggestion}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
                
        return report