## 8. Hyperparameter Optimizer - Otimização de Hiperparâmetros
# src/training/hyperopt_engine.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.model_selection import cross_val_score
import optuna
from optuna.samplers import TPESampler
from concurrent.futures import ProcessPoolExecutor

class HyperparameterOptimizer:
    """Otimizador de hiperparâmetros usando Optuna"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.study_results = {}
        
    def optimize_ensemble_hyperparameters(self, 
                                        X_train: pd.DataFrame,
                                        y_train: pd.Series,
                                        X_val: pd.DataFrame,
                                        y_val: pd.Series,
                                        n_trials: int = 50,
                                        n_jobs: int = 1) -> Dict:
        """
        Otimiza hiperparâmetros de todos os modelos do ensemble
        
        Args:
            X_train: Features de treino
            y_train: Targets de treino
            X_val: Features de validação
            y_val: Targets de validação
            n_trials: Número de trials por modelo
            n_jobs: Número de jobs paralelos
            
        Returns:
            Dicionário com melhores hiperparâmetros por modelo
        """
        self.logger.info(f"Iniciando otimização de hiperparâmetros com {n_trials} trials")
        
        best_params = {}
        
        # Modelos para otimizar
        models_to_optimize = [
            'xgboost',
            'lightgbm',
            'random_forest'
        ]
        
        for model_type in models_to_optimize:
            self.logger.info(f"Otimizando {model_type}...")
            
            # Criar estudo Optuna
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42),
                study_name=f'{model_type}_optimization'
            )
            
            # Função objetivo
            objective = self._create_objective_function(
                model_type, X_train, y_train, X_val, y_val
            )
            
            # Executar otimização
            study.optimize(
                objective,
                n_trials=n_trials,
                n_jobs=n_jobs,
                show_progress_bar=True
            )
            
            # Salvar resultados
            best_params[model_type] = study.best_params
            self.study_results[model_type] = {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'n_trials': len(study.trials),
                'study': study
            }
            
            self.logger.info(
                f"{model_type} - Melhor score: {study.best_value:.4f}"
            )
        
        # Otimizar parâmetros específicos de deep learning se necessário
        # (LSTM e Transformer geralmente requerem mais recursos)
        
        return best_params
    
    def _create_objective_function(self, model_type: str,
                                 X_train: pd.DataFrame,
                                 y_train: pd.Series,
                                 X_val: pd.DataFrame,
                                 y_val: pd.Series):
        """Cria função objetivo para otimização"""
        
        def objective(trial):
            # Sugerir hiperparâmetros baseado no tipo de modelo
            if model_type == 'xgboost':
                params = self._suggest_xgboost_params(trial)
            elif model_type == 'lightgbm':
                params = self._suggest_lightgbm_params(trial)
            elif model_type == 'random_forest':
                params = self._suggest_rf_params(trial)
            else:
                raise ValueError(f"Tipo de modelo desconhecido: {model_type}")
            
            # Treinar e avaliar modelo
            score = self._evaluate_model(
                model_type, params, X_train, y_train, X_val, y_val
            )
            
            return score
        
        return objective
    
    def _suggest_xgboost_params(self, trial: optuna.Trial) -> Dict:
        """Sugere hiperparâmetros para XGBoost"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'objective': 'multi:softprob',
            'num_class': 3,
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1
        }
    
    def _suggest_lightgbm_params(self, trial: optuna.Trial) -> Dict:
        """Sugere hiperparâmetros para LightGBM"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.5),
            'objective': 'multiclass',
            'num_class': 3,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
    def _suggest_rf_params(self, trial: optuna.Trial) -> Dict:
        """Sugere hiperparâmetros para Random Forest"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42,
            'n_jobs': -1
        }
    
    def _evaluate_model(self, model_type: str, params: Dict,
                       X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Avalia modelo com parâmetros específicos"""
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score
        
        # Criar modelo
        if model_type == 'xgboost':
            model = XGBClassifier(**params)
        elif model_type == 'lightgbm':
            model = LGBMClassifier(**params)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(**params)
        
        # Treinar
        if model_type == 'xgboost':
            try:
                # Tentar com early_stopping_rounds (versões mais antigas)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
            except (TypeError, AttributeError):
                # Fallback simples sem early stopping
                self.logger.warning("Early stopping não disponível, treinando sem")
                model.fit(X_train, y_train, verbose=False)
        elif model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(10),
                        lgb.log_evaluation(0)
                    ]
                )
            except Exception as e:
                self.logger.warning(f"Early stopping não disponível para LightGBM: {e}, treinando sem")
                model.fit(X_train, y_train, verbose=False)
        else:
            model.fit(X_train, y_train)
        
        # Avaliar
        y_pred = model.predict(X_val)
        
        # Usar F1 score como métrica principal
        score = f1_score(y_val, y_pred, average='weighted')
        
        return score
    
    def optimize_deep_learning_architecture(self, 
                                          model_type: str,
                                          X_train: pd.DataFrame,
                                          y_train: pd.Series,
                                          X_val: pd.DataFrame,
                                          y_val: pd.Series,
                                          n_trials: int = 20) -> Dict:
        """
        Otimiza arquitetura de modelos deep learning
        
        Nota: Requer mais recursos computacionais
        """
        self.logger.info(f"Otimizando arquitetura {model_type}")
        
        study = optuna.create_study(
            direction='maximize',
            study_name=f'{model_type}_architecture'
        )
        
        if model_type == 'lstm':
            objective = self._create_lstm_objective(X_train, y_train, X_val, y_val)
        elif model_type == 'transformer':
            objective = self._create_transformer_objective(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Tipo de modelo DL desconhecido: {model_type}")
        
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def _create_lstm_objective(self, X_train, y_train, X_val, y_val):
        """Função objetivo para LSTM"""
        
        def objective(trial):
            # Arquitetura
            n_layers = trial.suggest_int('n_layers', 1, 3)
            units = []
            for i in range(n_layers):
                units.append(trial.suggest_int(f'units_l{i}', 32, 128))
            
            # Hiperparâmetros
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            recurrent_dropout = trial.suggest_float('recurrent_dropout', 0.0, 0.3)
            learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # Construir e treinar modelo (simplificado)
            # Em produção, usar ModelTrainer._train_lstm
            
            # Retornar score simulado por enquanto
            return np.random.uniform(0.5, 0.8)
        
        return objective
    
    def analyze_hyperparameter_importance(self, model_type: str) -> Dict:
        """Analisa importância dos hiperparâmetros"""
        if model_type not in self.study_results:
            raise ValueError(f"Nenhum estudo encontrado para {model_type}")
        
        study = self.study_results[model_type]['study']
        
        # Importância dos parâmetros
        importance = optuna.importance.get_param_importances(study)
        
        # Análise de correlação
        correlations = {}
        for param in study.best_params.keys():
            values = [t.params.get(param) for t in study.trials if param in t.params]
            scores = [t.value for t in study.trials if param in t.params]
            
            if len(values) > 1 and isinstance(values[0], (int, float)):
                correlations[param] = np.corrcoef(values, scores)[0, 1]
        
        return {
            'importance': importance,
            'correlations': correlations,
            'best_params': study.best_params,
            'best_score': study.best_value
        }
    
    def save_optimization_results(self, save_path: str = 'optimization_results'):
        """Salva resultados da otimização"""
        from pathlib import Path
        import json
        import pickle
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Salvar resumo em JSON
        summary = {}
        for model_type, results in self.study_results.items():
            summary[model_type] = {
                'best_params': results['best_params'],
                'best_value': results['best_value'],
                'n_trials': results['n_trials']
            }
        
        with open(save_path / 'optimization_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Salvar estudos Optuna
        for model_type, results in self.study_results.items():
            study_path = save_path / f'{model_type}_study.pkl'
            with open(study_path, 'wb') as f:
                pickle.dump(results['study'], f)
        
        self.logger.info(f"Resultados de otimização salvos em {save_path}")