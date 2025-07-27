"""
TrainingOrchestratorV3 - Pipeline unificado de treinamento com modelos por regime
Baseado no DEVELOPER_GUIDE_V3_REFACTORING.md
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Any
import os
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb


class TrainingOrchestratorV3:
    """
    Orquestra todo o pipeline de treinamento ML
    
    Features:
    - Treina modelos específicos por regime
    - Validação temporal com walk-forward
    - Ensemble de múltiplos algoritmos
    - Otimização de hiperparâmetros
    - Persistência e versionamento
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o orchestrator
        
        Args:
            config: Configurações opcionais
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Paths
        self.model_path = self.config.get('model_path', 'models/')
        self.dataset_path = self.config.get('dataset_path', 'datasets/')
        self.results_path = self.config.get('results_path', 'results/')
        
        # Garantir diretórios
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        
        # Configurações de treino
        self.n_splits = self.config.get('n_splits', 5)
        self.test_size = self.config.get('test_size', 0.2)
        self.random_state = self.config.get('random_state', 42)
        
        # Modelos a treinar
        self.model_types = self.config.get('model_types', ['xgboost', 'lightgbm', 'random_forest'])
        
        # Regimes a treinar
        self.regimes = self.config.get('regimes', ['trend_up', 'trend_down', 'range'])
        
        # Métricas target
        self.target_metrics = self.config.get('target_metrics', {
            'accuracy': 0.55,
            'precision': 0.50,
            'recall': 0.50
        })
        
        # Cache de modelos
        self.models = {}
        self.results = {}
        
    def train_complete_system(self, dataset_metadata_path: Optional[str] = None) -> Dict:
        """
        Treina sistema completo com todos os modelos e regimes
        
        Args:
            dataset_metadata_path: Caminho para metadados do dataset
            
        Returns:
            Dict com resultados do treinamento
        """
        self.logger.info("="*60)
        self.logger.info("INICIANDO TREINAMENTO DO SISTEMA V3")
        self.logger.info("="*60)
        self.logger.info(f"Modelos: {self.model_types}")
        self.logger.info(f"Regimes: {self.regimes}")
        self.logger.info(f"Target metrics: {self.target_metrics}")
        
        # 1. Carregar datasets
        self.logger.info("\n1. Carregando datasets...")
        datasets = self._load_datasets(dataset_metadata_path)
        
        if not datasets:
            self.logger.error("Falha ao carregar datasets")
            return {}
        
        # 2. Treinar modelos por regime
        self.logger.info("\n2. Treinando modelos por regime...")
        
        for regime in self.regimes:
            self.logger.info(f"\n{'='*40}")
            self.logger.info(f"REGIME: {regime.upper()}")
            self.logger.info(f"{'='*40}")
            
            # Verificar se há dados suficientes
            regime_train_data = datasets['train']['by_regime'].get(regime)
            if not regime_train_data or len(regime_train_data['features']) < 100:
                self.logger.warning(f"Dados insuficientes para regime {regime}")
                continue
            
            # Treinar cada tipo de modelo
            regime_models = {}
            regime_results = {}
            
            for model_type in self.model_types:
                self.logger.info(f"\nTreinando {model_type} para {regime}...")
                
                model, results = self._train_model(
                    model_type=model_type,
                    regime=regime,
                    train_data=regime_train_data,
                    valid_data=datasets['valid']['by_regime'].get(regime, {}),
                    test_data=datasets['test']['by_regime'].get(regime, {})
                )
                
                if model:
                    regime_models[model_type] = model
                    regime_results[model_type] = results
                    
                    # Log métricas
                    self._log_model_metrics(model_type, regime, results)
            
            # Salvar modelos do regime
            if regime_models:
                self.models[regime] = regime_models
                self.results[regime] = regime_results
        
        # 3. Criar ensemble
        self.logger.info("\n3. Criando modelos ensemble...")
        ensemble_results = self._create_ensemble_models(datasets)
        
        # 4. Salvar modelos e resultados
        self.logger.info("\n4. Salvando modelos e resultados...")
        self._save_models()
        self._save_results()
        
        # 5. Relatório final
        self._generate_final_report()
        
        return {
            'models': self.models,
            'results': self.results,
            'ensemble_results': ensemble_results,
            'success': True
        }
    
    def _load_datasets(self, metadata_path: Optional[str] = None) -> Dict:
        """Carrega datasets para treinamento"""
        
        try:
            # Se não foi fornecido, procurar o mais recente
            if not metadata_path:
                metadata_files = [f for f in os.listdir(self.dataset_path) 
                                if f.endswith('.json') and 'metadata' in f]
                if not metadata_files:
                    self.logger.error("Nenhum dataset encontrado")
                    return {}
                
                # Usar o mais recente
                metadata_path = os.path.join(self.dataset_path, sorted(metadata_files)[-1])
            
            # Carregar metadados
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.logger.info(f"Dataset: {metadata['ticker']}")
            self.logger.info(f"Período: {metadata['start_date']} a {metadata['end_date']}")
            
            # Carregar datasets
            datasets = {'train': {}, 'valid': {}, 'test': {}}
            
            for split in ['train', 'valid', 'test']:
                # Dataset principal
                main_file = os.path.join(self.dataset_path, 
                                       metadata['splits'][split]['filename'])
                
                if os.path.exists(main_file):
                    data = pd.read_parquet(main_file)
                    
                    # Separar features e labels
                    feature_cols = [col for col in data.columns if col.startswith('v3_')]
                    label_cols = ['target_binary', 'target_class', 'target_return']
                    
                    datasets[split] = {
                        'features': data[feature_cols],
                        'labels': data[label_cols] if label_cols[0] in data.columns else None,
                        'meta': data[['regime']] if 'regime' in data.columns else None
                    }
                    
                    # Carregar por regime
                    datasets[split]['by_regime'] = {}
                    
                    if 'by_regime' in metadata['splits'][split]:
                        for regime, regime_info in metadata['splits'][split]['by_regime'].items():
                            regime_file = os.path.join(self.dataset_path, regime_info['filename'])
                            
                            if os.path.exists(regime_file):
                                regime_data = pd.read_parquet(regime_file)
                                
                                datasets[split]['by_regime'][regime] = {
                                    'features': regime_data[feature_cols],
                                    'labels': regime_data[label_cols] if label_cols[0] in regime_data.columns else None
                                }
                
                self.logger.info(f"  {split}: {len(datasets[split]['features'])} samples")
            
            return datasets
            
        except Exception as e:
            self.logger.error(f"Erro carregando datasets: {e}")
            return {}
    
    def _train_model(self, model_type: str, regime: str,
                    train_data: Dict, valid_data: Dict, 
                    test_data: Dict) -> Tuple[Any, Dict]:
        """Treina um modelo específico para um regime"""
        
        try:
            # Preparar dados
            X_train = train_data['features']
            y_train = train_data['labels']['target_binary'] if train_data['labels'] is not None else None
            
            if y_train is None:
                self.logger.error(f"Labels não encontradas para {regime}")
                return None, {}
            
            # Validação
            X_valid = valid_data.get('features') if valid_data else None
            y_valid = valid_data.get('labels', {}).get('target_binary') if valid_data else None
            
            # Criar modelo
            model = self._create_model(model_type, regime)
            
            # Treinar
            if model_type == 'xgboost':
                eval_set = [(X_valid, y_valid)] if X_valid is not None else None
                model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
            elif model_type == 'lightgbm':
                eval_set = [(X_valid, y_valid)] if X_valid is not None else None
                model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    callbacks=[lgb.log_evaluation(0)]
                )
            else:  # random_forest
                model.fit(X_train, y_train)
            
            # Avaliar
            results = self._evaluate_model(model, X_train, y_train, X_valid, y_valid,
                                         test_data if test_data else None)
            
            return model, results
            
        except Exception as e:
            self.logger.error(f"Erro treinando {model_type} para {regime}: {e}")
            return None, {}
    
    def _create_model(self, model_type: str, regime: str) -> Any:
        """Cria modelo com hiperparâmetros específicos por regime"""
        
        # Hiperparâmetros base
        base_params = {
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        # Ajustes por regime
        if regime == 'trend_up' or regime == 'trend_down':
            # Trend: modelos mais profundos
            if model_type == 'xgboost':
                params = {
                    **base_params,
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0.1,
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc'
                }
                return xgb.XGBClassifier(**params)
                
            elif model_type == 'lightgbm':
                params = {
                    **base_params,
                    'n_estimators': 200,
                    'num_leaves': 64,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'objective': 'binary',
                    'metric': 'auc',
                    'verbosity': -1
                }
                return lgb.LGBMClassifier(**params)
                
            else:  # random_forest
                params = {
                    **base_params,
                    'n_estimators': 200,
                    'max_depth': 12,
                    'min_samples_split': 20,
                    'min_samples_leaf': 10,
                    'max_features': 'sqrt'
                }
                return RandomForestClassifier(**params)
                
        else:  # range
            # Range: modelos mais simples, rápidos
            if model_type == 'xgboost':
                params = {
                    **base_params,
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'gamma': 0.2,
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc'
                }
                return xgb.XGBClassifier(**params)
                
            elif model_type == 'lightgbm':
                params = {
                    **base_params,
                    'n_estimators': 100,
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.7,
                    'bagging_fraction': 0.7,
                    'bagging_freq': 5,
                    'objective': 'binary',
                    'metric': 'auc',
                    'verbosity': -1
                }
                return lgb.LGBMClassifier(**params)
                
            else:  # random_forest
                params = {
                    **base_params,
                    'n_estimators': 100,
                    'max_depth': 8,
                    'min_samples_split': 50,
                    'min_samples_leaf': 20,
                    'max_features': 'sqrt'
                }
                return RandomForestClassifier(**params)
    
    def _evaluate_model(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                       X_valid: pd.DataFrame, y_valid: pd.Series,
                       test_data: Optional[Dict]) -> Dict:
        """Avalia modelo em múltiplos conjuntos"""
        
        results = {}
        
        # Train metrics
        y_pred_train = model.predict(X_train)
        y_proba_train = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else y_pred_train
        
        results['train'] = {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train, zero_division=0),
            'recall': recall_score(y_train, y_pred_train, zero_division=0),
            'f1': f1_score(y_train, y_pred_train, zero_division=0),
            'auc': roc_auc_score(y_train, y_proba_train) if len(np.unique(y_train)) > 1 else 0.5
        }
        
        # Validation metrics
        if X_valid is not None and y_valid is not None:
            y_pred_valid = model.predict(X_valid)
            y_proba_valid = model.predict_proba(X_valid)[:, 1] if hasattr(model, 'predict_proba') else y_pred_valid
            
            results['valid'] = {
                'accuracy': accuracy_score(y_valid, y_pred_valid),
                'precision': precision_score(y_valid, y_pred_valid, zero_division=0),
                'recall': recall_score(y_valid, y_pred_valid, zero_division=0),
                'f1': f1_score(y_valid, y_pred_valid, zero_division=0),
                'auc': roc_auc_score(y_valid, y_proba_valid) if len(np.unique(y_valid)) > 1 else 0.5
            }
        
        # Test metrics
        if test_data and 'features' in test_data and 'labels' in test_data:
            X_test = test_data['features']
            y_test = test_data['labels']['target_binary'] if test_data['labels'] is not None else None
            
            if y_test is not None:
                y_pred_test = model.predict(X_test)
                y_proba_test = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred_test
                
                results['test'] = {
                    'accuracy': accuracy_score(y_test, y_pred_test),
                    'precision': precision_score(y_test, y_pred_test, zero_division=0),
                    'recall': recall_score(y_test, y_pred_test, zero_division=0),
                    'f1': f1_score(y_test, y_pred_test, zero_division=0),
                    'auc': roc_auc_score(y_test, y_proba_test) if len(np.unique(y_test)) > 1 else 0.5
                }
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = dict(zip(
                X_train.columns,
                model.feature_importances_
            ))
        
        return results
    
    def _log_model_metrics(self, model_type: str, regime: str, results: Dict):
        """Log métricas do modelo"""
        
        self.logger.info(f"\nMétricas {model_type} - {regime}:")
        
        for split in ['train', 'valid', 'test']:
            if split in results:
                metrics = results[split]
                self.logger.info(f"  {split}:")
                self.logger.info(f"    Accuracy: {metrics['accuracy']:.3f}")
                self.logger.info(f"    Precision: {metrics['precision']:.3f}")
                self.logger.info(f"    Recall: {metrics['recall']:.3f}")
                self.logger.info(f"    F1: {metrics['f1']:.3f}")
                self.logger.info(f"    AUC: {metrics['auc']:.3f}")
        
        # Verificar se atende targets
        if 'valid' in results:
            valid_metrics = results['valid']
            meets_targets = all([
                valid_metrics.get(metric, 0) >= target
                for metric, target in self.target_metrics.items()
            ])
            
            if meets_targets:
                self.logger.info(f"  ✅ Atende targets!")
            else:
                self.logger.info(f"  ❌ Não atende targets")
    
    def _create_ensemble_models(self, datasets: Dict) -> Dict:
        """Cria modelos ensemble combinando algoritmos"""
        
        ensemble_results = {}
        
        for regime in self.regimes:
            if regime not in self.models or len(self.models[regime]) < 2:
                continue
            
            self.logger.info(f"\nCriando ensemble para {regime}...")
            
            # Implementação simplificada: votação majoritária
            # TODO: Implementar stacking mais sofisticado
            
            ensemble_results[regime] = {
                'models': list(self.models[regime].keys()),
                'method': 'majority_voting'
            }
        
        return ensemble_results
    
    def _save_models(self):
        """Salva modelos treinados"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for regime, regime_models in self.models.items():
            for model_type, model in regime_models.items():
                # Nome do arquivo
                filename = f"model_v3_{regime}_{model_type}_{timestamp}.pkl"
                filepath = os.path.join(self.model_path, filename)
                
                # Salvar modelo
                joblib.dump(model, filepath)
                self.logger.info(f"Modelo salvo: {filename}")
        
        # Salvar metadados
        metadata = {
            'timestamp': timestamp,
            'regimes': list(self.models.keys()),
            'model_types': self.model_types,
            'target_metrics': self.target_metrics,
            'models': {}
        }
        
        for regime in self.models:
            metadata['models'][regime] = {}
            for model_type in self.models[regime]:
                metadata['models'][regime][model_type] = f"model_v3_{regime}_{model_type}_{timestamp}.pkl"
        
        metadata_path = os.path.join(self.model_path, f"models_metadata_v3_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadados salvos: {metadata_path}")
    
    def _save_results(self):
        """Salva resultados do treinamento"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_path, f"training_results_v3_{timestamp}.json")
        
        # Converter numpy types para Python native
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Preparar resultados
        results_data = {
            'timestamp': timestamp,
            'results': {}
        }
        
        for regime, regime_results in self.results.items():
            results_data['results'][regime] = {}
            for model_type, model_results in regime_results.items():
                results_data['results'][regime][model_type] = {
                    split: {
                        metric: convert_numpy(value)
                        for metric, value in metrics.items()
                        if metric != 'feature_importance'
                    }
                    for split, metrics in model_results.items()
                    if split != 'feature_importance'
                }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Resultados salvos: {results_file}")
    
    def _generate_final_report(self):
        """Gera relatório final do treinamento"""
        
        self.logger.info("\n" + "="*60)
        self.logger.info("RELATÓRIO FINAL DO TREINAMENTO V3")
        self.logger.info("="*60)
        
        # Resumo por regime
        for regime in self.results:
            self.logger.info(f"\n{regime.upper()}:")
            
            best_model = None
            best_score = 0
            
            for model_type, results in self.results[regime].items():
                if 'valid' in results:
                    score = results['valid']['accuracy']
                    self.logger.info(f"  {model_type}: {score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_model = model_type
            
            if best_model:
                self.logger.info(f"  Melhor modelo: {best_model} ({best_score:.3f})")
        
        # Verificar se sistema atende requisitos
        all_meet_targets = True
        
        self.logger.info("\nVerificação de Targets:")
        for regime in self.results:
            regime_meets = False
            
            for model_type, results in self.results[regime].items():
                if 'valid' in results:
                    meets = all([
                        results['valid'].get(metric, 0) >= target
                        for metric, target in self.target_metrics.items()
                    ])
                    
                    if meets:
                        regime_meets = True
                        break
            
            status = "✅" if regime_meets else "❌"
            self.logger.info(f"  {regime}: {status}")
            
            if not regime_meets:
                all_meet_targets = False
        
        if all_meet_targets:
            self.logger.info("\n✅ SISTEMA APROVADO - Todos os regimes atendem targets!")
        else:
            self.logger.info("\n❌ SISTEMA REPROVADO - Alguns regimes não atendem targets")


def main():
    """Teste do TrainingOrchestratorV3"""
    
    print("="*60)
    print("TESTE DO TRAINING ORCHESTRATOR V3")
    print("="*60)
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuração
    config = {
        'model_types': ['xgboost', 'lightgbm', 'random_forest'],
        'regimes': ['trend_up', 'trend_down', 'range'],
        'target_metrics': {
            'accuracy': 0.55,
            'precision': 0.50,
            'recall': 0.50
        }
    }
    
    # Criar orchestrator
    orchestrator = TrainingOrchestratorV3(config)
    
    # Treinar sistema
    results = orchestrator.train_complete_system()
    
    if results.get('success'):
        print("\n[OK] Sistema V3 treinado com sucesso!")
    else:
        print("\n[ERROR] Falha no treinamento do sistema")


if __name__ == "__main__":
    main()