# src/training/ensemble_trainer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from datetime import datetime
import joblib
from concurrent.futures import ProcessPoolExecutor

class EnsembleTrainer:
    """Treinador de ensemble de modelos para trading"""
    
    def __init__(self, model_trainer):
        self.model_trainer = model_trainer
        self.logger = logging.getLogger(__name__)
        self.ensemble_config = self._get_ensemble_config()
        
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series,
                      model_types: Optional[List[str]] = None,
                      parallel: bool = True) -> Dict:
        """
        Treina ensemble completo de modelos
        
        Args:
            X_train: Features de treino
            y_train: Targets de treino
            X_val: Features de validação
            y_val: Targets de validação
            model_types: Lista de modelos para treinar
            parallel: Se deve treinar em paralelo
            
        Returns:
            Dicionário com modelos treinados e métricas
        """
        if model_types is None:
            model_types = list(self.ensemble_config.keys())
        
        self.logger.info(f"Treinando ensemble com {len(model_types)} modelos")
        
        # Treinar modelos
        if parallel and len(model_types) > 2:
            trained_models = self._parallel_training(
                X_train, y_train, X_val, y_val, model_types
            )
        else:
            trained_models = self._sequential_training(
                X_train, y_train, X_val, y_val, model_types
            )
        
        # Calcular pesos do ensemble
        if not trained_models:
            self.logger.error("Nenhum modelo foi treinado com sucesso")
            raise ValueError("Nenhum modelo foi treinado com sucesso")
            
        ensemble_weights = self._calculate_ensemble_weights(
            trained_models, X_val, y_val
        )
        
        # Avaliar ensemble completo
        ensemble_metrics = self._evaluate_ensemble(
            trained_models, ensemble_weights, X_val, y_val
        )
        
        # Análise de diversidade
        diversity_metrics = self._analyze_model_diversity(
            trained_models, X_val
        )
        
        return {
            'models': trained_models,
            'weights': ensemble_weights,
            'ensemble_metrics': ensemble_metrics,
            'diversity_metrics': diversity_metrics,
            'training_summary': self._create_training_summary(trained_models)
        }
    
    def _get_ensemble_config(self) -> Dict:
        """Configuração dos modelos do ensemble"""
        return {
            'xgboost_fast': {
                'model_type': 'xgboost',
                'hyperparams': {
                    'n_estimators': 50,
                    'max_depth': 4,
                    'learning_rate': 0.15
                },
                'weight_factor': 1.2  # Peso base no ensemble
            },
            
            'lightgbm_balanced': {
                'model_type': 'lightgbm',
                'hyperparams': {
                    'n_estimators': 75,
                    'max_depth': 5,
                    'learning_rate': 0.12
                },
                'weight_factor': 1.1
            },
            
            'random_forest_stable': {
                'model_type': 'random_forest',
                'hyperparams': {
                    'n_estimators': 150,
                    'max_depth': 8
                },
                'weight_factor': 0.9
            },
            
            'lstm_temporal': {
                'model_type': 'lstm',
                'hyperparams': {
                    'units': [64, 32],
                    'epochs': 30
                },
                'weight_factor': 1.0
            },
            
            'transformer_attention': {
                'model_type': 'transformer',
                'hyperparams': {
                    'd_model': 64,
                    'num_heads': 8,
                    'epochs': 30
                },
                'weight_factor': 0.8
            }
        }
    
    def _sequential_training(self, X_train, y_train, X_val, y_val, model_types):
        """Treinamento sequencial dos modelos"""
        trained_models = {}
        
        for model_name in model_types:
            if model_name not in self.ensemble_config:
                self.logger.warning(f"Configuração não encontrada para {model_name}")
                continue
            
            config = self.ensemble_config[model_name]
            
            try:
                # Treinar modelo
                model, metrics = self.model_trainer.train_model(
                    model_type=config['model_type'],
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    hyperparams=config['hyperparams']
                )
                
                trained_models[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'config': config,
                    'feature_names': list(X_train.columns)
                }
                
                self.logger.info(
                    f"{model_name} treinado - Val Accuracy: {metrics['val_accuracy']:.4f}"
                )
                
            except Exception as e:
                self.logger.error(f"Erro treinando {model_name}: {e}")
        
        return trained_models
    
    def _parallel_training(self, X_train, y_train, X_val, y_val, model_types):
        """Treinamento paralelo dos modelos"""
        trained_models = {}
        
        # Preparar tarefas
        training_tasks = []
        for model_name in model_types:
            if model_name not in self.ensemble_config:
                continue
            
            config = self.ensemble_config[model_name]
            training_tasks.append((
                model_name,
                config,
                X_train,
                y_train,
                X_val,
                y_val
            ))
        
        # Executar em paralelo
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = []
            
            for task in training_tasks:
                future = executor.submit(self._train_single_model, *task)
                futures.append((task[0], future))
            
            # Coletar resultados
            for model_name, future in futures:
                try:
                    result = future.result(timeout=3600)  # 1 hora timeout
                    if result:
                        trained_models[model_name] = result
                        self.logger.info(
                            f"{model_name} treinado - Val Accuracy: "
                            f"{result['metrics']['val_accuracy']:.4f}"
                        )
                except Exception as e:
                    self.logger.error(f"Erro treinando {model_name}: {e}")
        
        return trained_models
    
    def _train_single_model(self, model_name, config, X_train, y_train, X_val, y_val):
        """Função auxiliar para treinar um único modelo (para paralelização)"""
        try:
            model, metrics = self.model_trainer.train_model(
                model_type=config['model_type'],
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                hyperparams=config['hyperparams']
            )
            
            return {
                'model': model,
                'metrics': metrics,
                'config': config,
                'feature_names': list(X_train.columns)
            }
        except Exception as e:
            self.logger.error(f"Erro em _train_single_model para {model_name}: {e}")
            return None
    
    def _calculate_ensemble_weights(self, trained_models, X_val, y_val):
        """Calcula pesos ótimos para o ensemble"""
        from scipy.optimize import minimize
        
        # Obter predições de cada modelo
        predictions = {}
        for name, model_data in trained_models.items():
            model = model_data['model']
            
            # Predições de probabilidade
            if hasattr(model, 'predict_proba'):
                predictions[name] = model.predict_proba(X_val)
            else:
                # Para modelos deep learning
                predictions[name] = model.predict(X_val)
        
        # Função objetivo para otimização
        def ensemble_loss(weights):
            # Normalizar pesos
            weights = weights / np.sum(weights)
            
            # Combinar predições
            ensemble_pred = np.zeros_like(list(predictions.values())[0])
            for i, (name, pred) in enumerate(predictions.items()):
                ensemble_pred += weights[i] * pred
            
            # Calcular loss (negative accuracy)
            y_pred = np.argmax(ensemble_pred, axis=1)
            accuracy = np.mean(y_pred == y_val)
            
            return -accuracy  # Minimizar negative accuracy
        
        # Pesos iniciais baseados na configuração
        initial_weights = np.array([
            trained_models[name]['config']['weight_factor']
            for name in trained_models.keys()
        ])
        
        # Constraints: pesos devem somar 1 e ser não-negativos
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(trained_models))]
        
        # Otimizar
        result = minimize(
            ensemble_loss,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Pesos otimizados
        optimal_weights = result.x / np.sum(result.x)
        
        weights_dict = {
            name: float(weight)
            for name, weight in zip(trained_models.keys(), optimal_weights)
        }
        
        self.logger.info(f"Pesos do ensemble otimizados: {weights_dict}")
        
        return weights_dict
    
    def _evaluate_ensemble(self, trained_models, weights, X_val, y_val):
        """Avalia performance do ensemble completo"""
        # Obter predições ponderadas
        ensemble_pred = self._get_ensemble_predictions(
            trained_models, weights, X_val
        )
        
        # Classe predita
        y_pred = np.argmax(ensemble_pred, axis=1)
        
        # Calcular métricas
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import confusion_matrix
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted'),
            'recall': recall_score(y_val, y_pred, average='weighted'),
            'f1_score': f1_score(y_val, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist(),
            'confidence_mean': np.mean(np.max(ensemble_pred, axis=1)),
            'confidence_std': np.std(np.max(ensemble_pred, axis=1))
        }
        
        # Métricas por classe
        for i in range(3):
            mask = y_val == i
            if mask.sum() > 0:
                metrics[f'class_{i}_accuracy'] = np.mean(y_pred[mask] == i)
        
        # Análise de confiança por acerto/erro
        correct_mask = y_pred == y_val
        metrics['confidence_correct'] = np.mean(np.max(ensemble_pred[correct_mask], axis=1))
        metrics['confidence_incorrect'] = np.mean(np.max(ensemble_pred[~correct_mask], axis=1))
        
        return metrics
    
    def _get_ensemble_predictions(self, trained_models, weights, X):
        """Obtém predições ponderadas do ensemble"""
        predictions = []
        model_names = []
        
        for name, model_data in trained_models.items():
            model = model_data['model']
            weight = weights[name]
            
            # Predições
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X)
            
            predictions.append(pred * weight)
            model_names.append(name)
        
        # Combinar predições
        ensemble_pred = np.sum(predictions, axis=0)
        
        return ensemble_pred
    
    def _analyze_model_diversity(self, trained_models, X_val):
        """Analisa diversidade entre modelos do ensemble"""
        predictions = {}
        
        # Obter predições de cada modelo
        for name, model_data in trained_models.items():
            model = model_data['model']
            
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_val)
                predictions[name] = np.argmax(pred, axis=1)
            else:
                pred = model.predict(X_val)
                predictions[name] = np.argmax(pred, axis=1)
        
        # Calcular correlações entre predições
        pred_matrix = np.array(list(predictions.values())).T
        correlations = np.corrcoef(pred_matrix.T)
        
        # Métricas de diversidade
        diversity_metrics = {
            'avg_correlation': np.mean(correlations[np.triu_indices_from(correlations, k=1)]),
            'min_correlation': np.min(correlations[np.triu_indices_from(correlations, k=1)]),
            'max_correlation': np.max(correlations[np.triu_indices_from(correlations, k=1)]),
            'disagreement_rate': self._calculate_disagreement_rate(predictions),
            'correlation_matrix': correlations.tolist()
        }
        
        return diversity_metrics
    
    def _calculate_disagreement_rate(self, predictions):
        """Calcula taxa de discordância entre modelos"""
        pred_array = np.array(list(predictions.values()))
        n_models = len(predictions)
        n_samples = pred_array.shape[1]
        
        disagreements = 0
        for i in range(n_samples):
            unique_preds = len(np.unique(pred_array[:, i]))
            if unique_preds > 1:
                disagreements += 1
        
        return disagreements / n_samples
    
    def _create_training_summary(self, trained_models):
        """Cria resumo do treinamento"""
        summary = {
            'total_models': len(trained_models),
            'model_types': list(trained_models.keys()),
            'avg_val_accuracy': np.mean([
                m['metrics']['val_accuracy'] 
                for m in trained_models.values()
            ]),
            'best_model': max(
                trained_models.items(),
                key=lambda x: x[1]['metrics']['val_accuracy']
            )[0],
            'training_time': datetime.now().isoformat()
        }
        
        # Adicionar métricas individuais
        summary['individual_metrics'] = {
            name: {
                'val_accuracy': model['metrics']['val_accuracy'],
                'val_f1': model['metrics']['val_f1'],
                'trade_accuracy': model['metrics'].get('trade_accuracy', 0)
            }
            for name, model in trained_models.items()
        }
        
        return summary
    
    def save_ensemble(self, ensemble_data: Dict, save_path: str = 'src/training/models/ensemble'):
        """Salva ensemble completo"""
        save_path_obj = Path(save_path)
        save_path_obj.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ensemble_dir = save_path_obj / f"ensemble_{timestamp}"
        ensemble_dir.mkdir(exist_ok=True)
        
        # Salvar cada modelo
        model_paths = {}
        for name, model_data in ensemble_data['models'].items():
            model_path = ensemble_dir / f"{name}.pkl"
            
            # Salvar modelo
            if name in ['lstm_temporal', 'transformer_attention']:
                model_data['model'].save(model_path.with_suffix('.h5'))
                model_paths[name] = str(model_path.with_suffix('.h5'))
            else:
                joblib.dump(model_data['model'], model_path)
                model_paths[name] = str(model_path)
        
        # Salvar metadados do ensemble
        ensemble_metadata = {
            'timestamp': timestamp,
            'model_paths': model_paths,
            'weights': ensemble_data['weights'],
            'ensemble_metrics': ensemble_data['ensemble_metrics'],
            'diversity_metrics': ensemble_data['diversity_metrics'],
            'training_summary': ensemble_data['training_summary']
        }
        
        metadata_path = ensemble_dir / 'ensemble_metadata.json'
        with open(metadata_path, 'w') as f:
            import json
            json.dump(ensemble_metadata, f, indent=2)
        
        self.logger.info(f"Ensemble salvo em {ensemble_dir}")
        
        return str(ensemble_dir)