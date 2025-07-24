"""
Sistema Integrado de Treinamento ML para Trading
Combina preparação de dados, engenharia de features e treinamento de modelos
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# Imports locais
from .robust_nan_handler import RobustNaNHandler
from ..feature_engine import FeatureEngine
from ..technical_indicators import TechnicalIndicators
from ..ml_features import MLFeatures

logger = logging.getLogger(__name__)

class MLTrainingSystem:
    """
    Sistema completo de treinamento ML para trading
    Integra todas as etapas desde dados brutos até modelos prontos
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        
        # Configuração padrão
        self.config = {
            'test_size': 0.2,
            'validation_size': 0.15,
            'min_data_points': 1000,
            'max_features': 100,
            'cv_folds': 5,
            'random_state': 42,
            'models_dir': Path('src/training/models'),
            'reports_dir': Path('training_results'),
            'target_column': 'target',
            'feature_importance_threshold': 0.001
        }
        
        if config:
            self.config.update(config)
        
        # Inicializar componentes
        self.nan_handler = RobustNaNHandler()
        self.feature_engine = FeatureEngine()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Métricas de treinamento
        self.training_metrics = {}
        self.feature_importance = {}
        
        # Criar diretórios necessários
        self.config['models_dir'].mkdir(parents=True, exist_ok=True)
        self.config['reports_dir'].mkdir(parents=True, exist_ok=True)
    
    def prepare_training_data(self, raw_data: pd.DataFrame, 
                            target_data: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara dados completos para treinamento
        
        Args:
            raw_data: Dados OHLCV brutos
            target_data: Targets para supervised learning (opcional)
            
        Returns:
            Tuple[Features preparadas, Targets]
        """
        self.logger.info("🔄 Iniciando preparação de dados para treinamento")
        self.logger.info(f"Dados de entrada: {len(raw_data)} linhas, {len(raw_data.columns)} colunas")
        
        # 1. Validar dados de entrada
        if len(raw_data) < self.config['min_data_points']:
            raise ValueError(f"Dados insuficientes: {len(raw_data)} < {self.config['min_data_points']}")
        
        # 2. Gerar todas as features
        self.logger.info("📊 Gerando features técnicas...")
        features_df = self._generate_comprehensive_features(raw_data)
        
        # 3. Tratar valores NaN de forma inteligente
        self.logger.info("🧹 Tratando valores NaN...")
        clean_features, nan_stats = self.nan_handler.handle_nans(features_df, raw_data)
        
        # 4. Validar qualidade dos dados limpos
        validation_result = self.nan_handler.validate_nan_handling(clean_features)
        
        # 5. Gerar targets se não fornecidos
        if target_data is None:
            self.logger.info("🎯 Gerando targets automáticos...")
            target_data = self._generate_auto_targets(raw_data, clean_features.index)
        else:
            # Alinhar targets com features limpas
            target_data = target_data.loc[clean_features.index]
        
        # 6. Seleção de features importantes
        self.logger.info("🎛️ Selecionando features mais relevantes...")
        selected_features = self._select_important_features(clean_features, target_data)
        
        self.logger.info(f"✅ Preparação concluída:")
        self.logger.info(f"  • Features finais: {len(selected_features.columns)}")
        self.logger.info(f"  • Amostras: {len(selected_features)}")
        self.logger.info(f"  • Score qualidade: {validation_result['quality_score']:.3f}")
        
        return selected_features, target_data
    
    def _generate_comprehensive_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Gera conjunto abrangente de features"""
        
        # Usar o FeatureEngine para gerar features
        tech_indicators = TechnicalIndicators()
        ml_features = MLFeatures()
        
        # Calcular indicadores técnicos (método correto: calculate_all)
        indicators_df = tech_indicators.calculate_all(raw_data)
        
        # Calcular features de ML (método correto: calculate_all)
        ml_features_df = ml_features.calculate_all(
            candles=raw_data, 
            indicators=indicators_df
        )
        
        # Combinar todas as features
        all_features = pd.concat([indicators_df, ml_features_df], axis=1)
        
        # Remover duplicatas de colunas
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        self.logger.info(f"Features geradas: {len(all_features.columns)}")
        self.logger.info(f"  • Indicadores técnicos: {len(indicators_df.columns)}")
        self.logger.info(f"  • Features ML: {len(ml_features_df.columns)}")
        
        return all_features
    
    def _generate_auto_targets(self, raw_data: pd.DataFrame, 
                             feature_index: pd.Index) -> pd.Series:
        """Gera targets automáticos baseados em movimento de preço"""
        
        # Usar apenas dados alinhados com features
        aligned_data = raw_data.loc[feature_index]
        
        # Calcular retorno futuro (próximos N períodos)
        future_periods = 5
        future_return = aligned_data['close'].shift(-future_periods) / aligned_data['close'] - 1
        
        # Classificar em 3 classes baseado em thresholds
        threshold_buy = 0.002   # 0.2%
        threshold_sell = -0.002  # -0.2%
        
        targets = pd.Series(index=feature_index, dtype=int)
        targets[future_return > threshold_buy] = 2  # BUY
        targets[future_return < threshold_sell] = 0  # SELL  
        targets[(future_return >= threshold_sell) & (future_return <= threshold_buy)] = 1  # HOLD
        
        # Remover NaN no final
        targets = targets.dropna()
        
        self.logger.info(f"Targets gerados: {targets.value_counts().to_dict()}")
        
        return targets
    
    def _select_important_features(self, features: pd.DataFrame, 
                                 targets: pd.Series) -> pd.DataFrame:
        """Seleciona features mais importantes usando LightGBM"""
        
        # Alinhar features e targets
        common_index = features.index.intersection(targets.index)
        features_aligned = features.loc[common_index]
        targets_aligned = targets.loc[common_index]
        
        if len(features_aligned) == 0:
            raise ValueError("Nenhuma amostra comum entre features e targets")
        
        # Treinar modelo simples para importância
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=self.config['random_state'],
            verbose=-1
        )
        
        model.fit(features_aligned, targets_aligned)
        
        # Obter importância das features
        feature_importance = pd.Series(
            model.feature_importances_,
            index=features_aligned.columns
        ).sort_values(ascending=False)
        
        # Selecionar top features
        top_features = feature_importance.head(self.config['max_features'])
        selected_columns = top_features[top_features > self.config['feature_importance_threshold']].index
        
        self.feature_importance = feature_importance.to_dict()
        
        self.logger.info(f"Features selecionadas: {len(selected_columns)}/{len(features.columns)}")
        
        return features_aligned[selected_columns]
    
    def train_models(self, features: pd.DataFrame, 
                    targets: pd.Series) -> Dict[str, Any]:
        """
        Treina modelos ML com validação temporal
        
        Returns:
            Dict com modelos treinados e métricas
        """
        self.logger.info("🚀 Iniciando treinamento de modelos ML")
        
        # Alinhar dados
        common_index = features.index.intersection(targets.index)
        X = features.loc[common_index]
        y = targets.loc[common_index]
        
        # Split temporal (importante para dados financeiros)
        split_point = int(len(X) * (1 - self.config['test_size']))
        
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        self.logger.info(f"Divisão temporal: {len(X_train)} treino, {len(X_test)} teste")
        
        # Normalizar features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Treinar modelo principal - LightGBM
        model = self._train_lightgbm_model(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Salvar modelo e componentes
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.config['models_dir'] / f"training_{timestamp}"
        model_dir.mkdir(exist_ok=True)
        
        # Salvar modelo
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)
        
        # Salvar scaler
        scaler_path = model_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        
        # Salvar metadados
        metadata = {
            'timestamp': timestamp,
            'features': list(X.columns),
            'target_classes': list(np.unique(y)),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics
        }
        
        metadata_path = model_dir / "metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"✅ Modelo salvo em: {model_dir}")
        
        return {
            'model': model,
            'scaler': self.scaler,
            'metadata': metadata,
            'model_dir': model_dir,
            'training_metrics': self.training_metrics
        }
    
    def _train_lightgbm_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series) -> LGBMClassifier:
        """Treina modelo LightGBM otimizado"""
        
        # Configuração otimizada para trading
        model_params = {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'num_leaves': 64,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'random_state': self.config['random_state'],
            'class_weight': 'balanced',
            'verbose': -1
        }
        
        model = LGBMClassifier(**model_params)
        
        # Treinar com early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Calcular métricas
        y_pred_train = np.array(model.predict(X_train))
        y_pred_test = np.array(model.predict(X_test))
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        self.training_metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'n_estimators_used': model.n_estimators_,
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True)
        }
        
        self.logger.info(f"📊 Métricas do modelo:")
        self.logger.info(f"  • Acurácia treino: {train_accuracy:.3f}")
        self.logger.info(f"  • Acurácia teste: {test_accuracy:.3f}")
        self.logger.info(f"  • Estimadores usados: {model.n_estimators_}")
        
        return model
    
    def create_training_report(self, training_result: Dict[str, Any]) -> str:
        """Cria relatório detalhado do treinamento"""
        
        metadata = training_result['metadata']
        metrics = training_result['training_metrics']
        
        report_lines = [
            "=" * 60,
            "🤖 RELATÓRIO DE TREINAMENTO ML",
            "=" * 60,
            f"📅 Data/Hora: {metadata['timestamp']}",
            f"🎯 Modelo: LightGBM Classifier",
            "",
            "📊 DADOS DE TREINAMENTO:",
            f"  • Features: {len(metadata['features'])}",
            f"  • Amostras treino: {metadata['train_samples']:,}",
            f"  • Amostras teste: {metadata['test_samples']:,}",
            f"  • Classes: {metadata['target_classes']}",
            "",
            "🎯 PERFORMANCE:",
            f"  • Acurácia Treino: {metrics['train_accuracy']:.3f}",
            f"  • Acurácia Teste: {metrics['test_accuracy']:.3f}",
            f"  • Overfitting: {abs(metrics['train_accuracy'] - metrics['test_accuracy']):.3f}",
            "",
            "🏆 TOP 10 FEATURES IMPORTANTES:",
        ]
        
        # Top features
        sorted_importance = sorted(
            metadata['feature_importance'].items(), 
            key=lambda x: x[1], reverse=True
        )[:10]
        
        for i, (feature, importance) in enumerate(sorted_importance, 1):
            report_lines.append(f"  {i:2d}. {feature:<25} {importance:.4f}")
        
        # Métricas por classe
        if 'classification_report' in metrics:
            report_lines.extend([
                "",
                "📈 MÉTRICAS POR CLASSE:",
            ])
            
            for class_name, class_metrics in metrics['classification_report'].items():
                if isinstance(class_metrics, dict) and 'precision' in class_metrics:
                    report_lines.append(
                        f"  {class_name}: Precision={class_metrics['precision']:.3f}, "
                        f"Recall={class_metrics['recall']:.3f}, F1={class_metrics['f1-score']:.3f}"
                    )
        
        report_lines.extend([
            "",
            "=" * 60,
            f"Status: {'✅ TREINAMENTO CONCLUÍDO' if metrics['test_accuracy'] > 0.5 else '⚠️  REVISAR MODELO'}",
            "=" * 60
        ])
        
        return "\n".join(report_lines)

def main():
    """Exemplo de uso do sistema de treinamento"""
    logging.basicConfig(level=logging.INFO)
    
    # Configuração
    config = {
        'models_dir': Path('src/training/models'),
        'reports_dir': Path('training_results'),
        'max_features': 80,
        'test_size': 0.2
    }
    
    # Inicializar sistema
    training_system = MLTrainingSystem(config)
    
    print("🤖 Sistema de Treinamento ML Inicializado")
    print("Aguardando dados para treinamento...")

if __name__ == "__main__":
    main()
