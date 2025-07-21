# src/training/training_orchestrator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from datetime import datetime
import json
import yaml

class TrainingOrchestrator:
    """Orquestrador central do sistema de treinamento"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Importações dinâmicas para evitar problemas de path
        try:
            # Tentar importações do sistema de treinamento
            from data_loader import TrainingDataLoader
            from preprocessor import DataPreprocessor
            from feature_pipeline import FeatureEngineeringPipeline
            from model_trainer import ModelTrainer
            from ensemble_trainer import EnsembleTrainer
            from validation_engine import ValidationEngine
            from hyperopt_engine import HyperparameterOptimizer
            from trading_metrics import TradingMetricsAnalyzer
            from performance_analyzer import PerformanceAnalyzer
            
        except ImportError:
            # Fallback para importações relativas
            try:
                from .data_loader import TrainingDataLoader
                from .preprocessor import DataPreprocessor
                from .feature_pipeline import FeatureEngineeringPipeline
                from .model_trainer import ModelTrainer
                from .ensemble_trainer import EnsembleTrainer
                from .validation_engine import ValidationEngine
                from .hyperopt_engine import HyperparameterOptimizer
                from .trading_metrics import TradingMetricsAnalyzer
                from .performance_analyzer import PerformanceAnalyzer
                
            except ImportError:
                # Último recurso - importações absolutas
                from training.data_loader import TrainingDataLoader
                from training.preprocessor import DataPreprocessor
                from training.feature_pipeline import FeatureEngineeringPipeline
                from training.model_trainer import ModelTrainer
                from training.ensemble_trainer import EnsembleTrainer
                from training.validation_engine import ValidationEngine
                from training.hyperopt_engine import HyperparameterOptimizer
                from training.trading_metrics import TradingMetricsAnalyzer
                from training.performance_analyzer import PerformanceAnalyzer
        
        # Componentes existentes do sistema - com fallbacks
        try:
            from src.model_manager import ModelManager
            from src.feature_engine import FeatureEngine
        except ImportError:
            # Tentar sem src.
            try:
                from model_manager import ModelManager
                from feature_engine import FeatureEngine
            except ImportError:
                # Mock classes para desenvolvimento
                self.logger.warning("Usando classes mock para ModelManager e FeatureEngine")
                
                class MockModelManager:
                    def __init__(self, *args, **kwargs): pass
                    def load_models(self): return True
                    
                class MockFeatureEngine:
                    def __init__(self, *args, **kwargs): pass
                    def calculate(self, *args, **kwargs): return {}
                
                ModelManager = MockModelManager
                FeatureEngine = MockFeatureEngine
        
        # Inicializar componentes
        self.data_loader = TrainingDataLoader(config.get('data_path', 'data/'))
        self.preprocessor = DataPreprocessor()
        self.feature_pipeline = FeatureEngineeringPipeline()
        self.model_trainer = ModelTrainer(config.get('model_save_path', 'src/training/models/'))
        self.ensemble_trainer = EnsembleTrainer(self.model_trainer)
        self.validation_engine = ValidationEngine()
        self.hyperopt = HyperparameterOptimizer()
        
        # Componentes existentes
        self.model_manager = ModelManager(config.get('models_dir', 'saved_models/'))
        self.feature_engine = FeatureEngine()
        
        # Métricas e resultados
        self.training_results = {}
        
    def train_complete_system(self, 
                            start_date: datetime,
                            end_date: datetime,
                            symbols: List[str],
                            target_metrics: Dict = None,
                            validation_method: str = 'walk_forward') -> Dict:
        """
        Treina sistema completo end-to-end
        
        Args:
            start_date: Data inicial dos dados
            end_date: Data final dos dados
            symbols: Lista de símbolos para treinar
            target_metrics: Métricas alvo para validação
            validation_method: Método de validação temporal
            
        Returns:
            Dicionário com resultados completos do treinamento
        """
        self.logger.info("Iniciando treinamento completo do sistema")
        self.logger.info(f"Período: {start_date} até {end_date}")
        self.logger.info(f"Símbolos: {symbols}")
        
        # 1. Carregar e validar dados
        self.logger.info("Etapa 1: Carregamento de dados")
        raw_data = self.data_loader.load_historical_data(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            validate_realtime=True
        )
        
        # 2. Preprocessamento com RobustNaNHandler
        self.logger.info("Etapa 2: Preprocessamento")
        processed_data, targets = self.preprocessor.preprocess_training_data(
            raw_data,
            target_col='target' if 'target' in raw_data.columns else None,
            scale_features=False,  # Escalar depois da seleção
            raw_ohlcv=raw_data  # 🔧 Passar dados brutos para tratamento robusto de NaN
        )
        
        # 3. Feature engineering
        self.logger.info("Etapa 3: Feature Engineering")
        features_df = self.feature_pipeline.create_training_features(
            processed_data,
            feature_groups=['technical', 'momentum', 'volatility', 'microstructure'],
            parallel=True
        )
        
        # 4. Seleção de features
        self.logger.info("Etapa 4: Seleção de Features")
        top_features = self.feature_pipeline.select_top_features(
            features_df, 
            targets,
            n_features=30,
            method='mutual_info'
        )
        
        # Filtrar apenas top features
        selected_features = features_df[top_features]
        
        # 5. Escalar features selecionadas
        scaled_features = self.preprocessor._scale_features(selected_features)
        
        # 6. Criar splits de validação
        self.logger.info("Etapa 5: Criação de Splits de Validação")
        
        # Verificar se há dados suficientes para validação
        min_samples_required = 100  # Mínimo para treinamento básico
        if len(scaled_features) < min_samples_required:
            self.logger.error(f"Dados insuficientes: {len(scaled_features)} amostras (mínimo: {min_samples_required})")
            
            # Retornar resultado com erro mas não falhar completamente
            error_result = {
                'status': 'failed',
                'error': 'insufficient_data',
                'message': f'Apenas {len(scaled_features)} amostras disponíveis. Mínimo: {min_samples_required}',
                'timestamp': datetime.now().isoformat(),
                'data_period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'symbols': symbols,
                'aggregated_metrics': self._get_default_metrics()
            }
            
            self.logger.warning("⚠️ Treinamento executado com dados insuficientes - métricas serão zeradas")
            return error_result

        validation_results = []
        splits_created = 0
        
        try:
            # Contar splits primeiro
            splits_list = list(self.validation_engine.create_temporal_splits(
                scaled_features, 
                method=validation_method
            ))
            splits_created = len(splits_list)
            
            self.logger.info(f"Walk-forward validation: {splits_created} folds criados")
            
            # Se não há splits, usar dados completos como um único fold
            if splits_created == 0:
                self.logger.warning("Nenhum split criado - usando dados completos como fold único")
                splits_list = [(scaled_features, scaled_features.iloc[-10:], {'fold': 0})]  # Usar últimos 10 como teste
                
        except Exception as e:
            self.logger.error(f"Erro ao criar splits: {e}")
            # Fallback - usar dados completos
            splits_list = [(scaled_features, scaled_features.iloc[-5:], {'fold': 0})]  # Usar últimos 5 como teste
            self.logger.warning("Usando fallback - dados completos com split mínimo")
        
        for fold_idx, (train_data, test_data, split_info) in enumerate(splits_list):
            self.logger.info(f"\nProcessando fold {fold_idx + 1}")
            
            # Alinhar targets
            train_targets = targets.loc[train_data.index]
            test_targets = targets.loc[test_data.index]
            
            # 7. Otimização de hiperparâmetros (apenas no primeiro fold)
            if fold_idx == 0:
                self.logger.info("Etapa 6: Otimização de Hiperparâmetros")
                best_hyperparams = self.hyperopt.optimize_ensemble_hyperparameters(
                    train_data, train_targets,
                    test_data, test_targets,
                    n_trials=20
                )
            else:
                # Usar hiperparâmetros do primeiro fold
                best_hyperparams = validation_results[0].get('hyperparams', {})
            
            # 8. Treinar ensemble
            self.logger.info("Etapa 7: Treinamento do Ensemble")
            ensemble_result = self.ensemble_trainer.train_ensemble(
                X_train=train_data,
                y_train=train_targets,
                X_val=test_data,
                y_val=test_targets,
                model_types=['xgboost_fast', 'lightgbm_balanced', 'random_forest_stable'],
                parallel=False
            )
            
            # 9. Validar resultados
            self.logger.info("Etapa 8: Validação")
            validation_metrics = self.validation_engine.calculate_validation_metrics(
                test_targets,
                self._get_ensemble_predictions(ensemble_result, test_data),
                self._get_ensemble_probabilities(ensemble_result, test_data)
            )
            
            # Armazenar resultados do fold
            fold_results = {
                'fold': fold_idx,
                'split_info': split_info,
                'ensemble_result': ensemble_result,
                'validation_metrics': validation_metrics,
                'hyperparams': best_hyperparams
            }
            
            validation_results.append(fold_results)
            
            # Log métricas principais
            self.logger.info(f"Accuracy: {validation_metrics['accuracy']:.4f}")
            self.logger.info(f"F1 Score: {validation_metrics['f1_score']:.4f}")
        
        # 10. Análise agregada
        self.logger.info("\nEtapa 9: Análise Agregada")
        aggregated_metrics = self._aggregate_validation_results(validation_results)
        
        # 11. Treinar modelo final com todos os dados
        self.logger.info("Etapa 10: Treinamento Final")
        final_ensemble = self._train_final_ensemble(
            scaled_features, 
            targets,
            best_hyperparams,
            top_features
        )
        
        # 12. Salvar resultados
        self.logger.info("Etapa 11: Salvando Resultados")
        save_paths = self._save_training_results(
            final_ensemble,
            validation_results,
            aggregated_metrics,
            top_features
        )
        
        # Resultados finais
        training_results = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'data_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'symbols': symbols,
            'validation_method': validation_method,
            'n_folds': len(validation_results),
            'selected_features': top_features,
            'aggregated_metrics': aggregated_metrics,
            'final_ensemble': final_ensemble,
            'save_paths': save_paths,
            'validation_results': validation_results
        }
        
        # Verificar se atingiu métricas alvo
        if target_metrics:
            metrics_achieved = self._check_target_metrics(
                aggregated_metrics, target_metrics
            )
            training_results['target_metrics_achieved'] = metrics_achieved
        
        self.training_results = training_results
        
        self.logger.info("\n✅ Treinamento completo finalizado com sucesso!")
        self._print_summary(training_results)
        
        return training_results
    
    def _get_ensemble_predictions(self, ensemble_result: Dict, X: pd.DataFrame) -> np.ndarray:
        """Obtém predições do ensemble como classes discretas"""
        predictions = []
        weights = ensemble_result['weights']
        
        for name, model_data in ensemble_result['models'].items():
            model = model_data['model']
            weight = weights[name]
            
            if hasattr(model, 'predict_proba'):
                # Usar probabilidades se disponível
                proba = model.predict_proba(X)
                predictions.append(proba * weight)
            elif hasattr(model, 'predict'):
                # Para modelos que só retornam classes, converter para probabilidade
                pred_classes = model.predict(X)
                # Converter classes para one-hot encoding
                n_classes = 3  # Venda, Neutro, Compra
                proba = np.zeros((len(pred_classes), n_classes))
                proba[np.arange(len(pred_classes)), pred_classes.astype(int)] = 1.0
                predictions.append(proba * weight)
        
        if not predictions:
            # Fallback se não há modelos
            return np.zeros(len(X), dtype=int)
        
        # Combinar probabilidades ponderadas
        ensemble_proba = np.sum(predictions, axis=0)
        
        # Normalizar para somar 1 (garantir probabilidades válidas)
        row_sums = ensemble_proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Evitar divisão por zero
        ensemble_proba = ensemble_proba / row_sums
        
        # Converter probabilidades para classes (argmax)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        return ensemble_pred.astype(int)
    
    def _get_ensemble_probabilities(self, ensemble_result: Dict, X: pd.DataFrame) -> np.ndarray:
        """Obtém probabilidades do ensemble"""
        probabilities = []
        weights = ensemble_result['weights']
        
        for name, model_data in ensemble_result['models'].items():
            model = model_data['model']
            weight = weights[name]
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
            else:
                # Para modelos deep learning
                proba = model.predict(X)
            
            probabilities.append(proba * weight)
        
        # Combinar probabilidades ponderadas
        ensemble_proba = np.sum(probabilities, axis=0)
        
        # Normalizar para somar 1
        ensemble_proba = ensemble_proba / ensemble_proba.sum(axis=1, keepdims=True)
        
        return ensemble_proba
    
    def _aggregate_validation_results(self, validation_results: List[Dict]) -> Dict:
        """Agrega resultados de múltiplos folds"""
        # Se não há resultados, retornar métricas padrão
        if not validation_results:
            self.logger.warning("Nenhum resultado de validação para agregar")
            return self._get_default_metrics()
        
        # Coletar métricas de cada fold
        all_metrics = []
        for result in validation_results:
            if 'validation_metrics' in result:
                all_metrics.append(result['validation_metrics'])
        
        # Se ainda não há métricas, retornar padrão
        if not all_metrics:
            self.logger.warning("Nenhuma métrica de validação encontrada")
            return self._get_default_metrics()

        # Calcular estatísticas
        aggregated = {}

        # Métricas numéricas
        numeric_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'avg_confidence', 'confidence_when_correct', 'confidence_when_wrong'
        ]

        for metric in numeric_metrics:
            values = [m.get(metric, 0) for m in all_metrics]
            if values:  # Verificar se há valores
                aggregated[f'{metric}_mean'] = np.mean(values) if values else 0.0
                aggregated[f'{metric}_std'] = np.std(values) if len(values) > 1 else 0.0
                aggregated[f'{metric}_min'] = np.min(values) if values else 0.0
                aggregated[f'{metric}_max'] = np.max(values) if values else 0.0
            else:
                aggregated[f'{metric}_mean'] = 0.0
                aggregated[f'{metric}_std'] = 0.0
                aggregated[f'{metric}_min'] = 0.0
                aggregated[f'{metric}_max'] = 0.0        # Matriz de confusão agregada
        conf_matrices = [m.get('confusion_matrix', [[0,0,0],[0,0,0],[0,0,0]]) for m in all_metrics]
        if conf_matrices and len(conf_matrices) > 0:
            try:
                aggregated['confusion_matrix_avg'] = np.mean(conf_matrices, axis=0).tolist()
            except Exception as e:
                self.logger.warning(f"Erro ao calcular matriz de confusão: {e}")
                aggregated['confusion_matrix_avg'] = [[0,0,0],[0,0,0],[0,0,0]]
        else:
            aggregated['confusion_matrix_avg'] = [[0,0,0],[0,0,0],[0,0,0]]

        # Métricas por classe
        for class_idx in range(3):
            for metric_type in ['precision', 'recall']:
                metric_name = f'class_{class_idx}_{metric_type}'
                values = [m.get(metric_name, 0) for m in all_metrics if metric_name in m]
                if values:
                    aggregated[f'{metric_name}_mean'] = np.mean(values)
                else:
                    aggregated[f'{metric_name}_mean'] = 0.0

        return aggregated
    
    def _get_default_metrics(self) -> Dict:
        """Retorna métricas padrão quando não há dados suficientes"""
        default_metrics = {}
        
        # Métricas numéricas básicas
        numeric_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'avg_confidence', 'confidence_when_correct', 'confidence_when_wrong'
        ]
        
        for metric in numeric_metrics:
            default_metrics[f'{metric}_mean'] = 0.0
            default_metrics[f'{metric}_std'] = 0.0
            default_metrics[f'{metric}_min'] = 0.0
            default_metrics[f'{metric}_max'] = 0.0
        
        # Matriz de confusão padrão
        default_metrics['confusion_matrix_avg'] = [[0,0,0],[0,0,0],[0,0,0]]
        
        # Métricas por classe
        for class_idx in range(3):
            for metric_type in ['precision', 'recall']:
                metric_name = f'class_{class_idx}_{metric_type}'
                default_metrics[f'{metric_name}_mean'] = 0.0
        
        return default_metrics
    
    def _json_serialize_helper(self, obj):
        """Auxiliar para serializar objetos NumPy em JSON"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    def _train_final_ensemble(self, features: pd.DataFrame, targets: pd.Series,
                            hyperparams: Dict, feature_names: List[str]) -> Dict:
        """Treina ensemble final com todos os dados"""
        # Alinhar features e targets antes da divisão
        min_len = min(len(features), len(targets))
        features = features.iloc[:min_len]
        targets = targets.iloc[:min_len]
        
        self.logger.info(f"Dados alinhados para treinamento final: {min_len} amostras")
        
        # Dividir dados para validação interna (80/20)
        split_idx = int(len(features) * 0.8)
        
        X_train = features.iloc[:split_idx]
        y_train = targets.iloc[:split_idx]
        X_val = features.iloc[split_idx:]
        y_val = targets.iloc[split_idx:]
        
        # Verificar alinhamento de dados
        self.logger.info(f"Treinamento final - X_train: {X_train.shape}, y_train: {y_train.shape}")
        self.logger.info(f"Validação final - X_val: {X_val.shape}, y_val: {y_val.shape}")
        
        if len(X_train) != len(y_train):
            self.logger.error(f"Desalinhamento X_train ({len(X_train)}) vs y_train ({len(y_train)})")
            return None
            
        if len(X_val) != len(y_val):
            self.logger.error(f"Desalinhamento X_val ({len(X_val)}) vs y_val ({len(y_val)})")
            return None
        
        # Treinar ensemble final
        final_ensemble = self.ensemble_trainer.train_ensemble(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_types=['xgboost_fast', 'lightgbm_balanced', 'random_forest_stable'],
            parallel=False
        )
        
        # Adicionar metadados
        final_ensemble['feature_names'] = feature_names
        final_ensemble['training_date'] = datetime.now()
        final_ensemble['data_size'] = len(features)
        
        return final_ensemble
    
    def _save_training_results(self, final_ensemble: Dict, validation_results: List,
                             aggregated_metrics: Dict, feature_names: List[str]) -> Dict:
        """Salva todos os resultados do treinamento"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Diretório principal
        results_dir = Path(self.config.get('results_path', 'src/training/models'))
        results_dir = results_dir / f"training_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Salvar ensemble
        ensemble_path = self.ensemble_trainer.save_ensemble(
            final_ensemble,
            save_path=str(results_dir / 'ensemble')
        )
        
        # 2. Salvar métricas agregadas
        metrics_path = results_dir / 'aggregated_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(aggregated_metrics, f, indent=2)
        
        # 3. Salvar lista de features
        features_path = results_dir / 'selected_features.json'
        with open(features_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        # 4. Salvar configuração de treinamento
        config_path = results_dir / 'training_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        # 5. Salvar resumo de validação
        validation_summary = []
        for result in validation_results:
            summary = {
                'fold': result['fold'],
                'split_info': result['split_info'],
                'metrics': result['validation_metrics']
            }
            validation_summary.append(summary)
        
        validation_path = results_dir / 'validation_summary.json'
        with open(validation_path, 'w') as f:
            json.dump(validation_summary, f, indent=2, default=self._json_serialize_helper)
        
        # 6. Criar arquivo README
        readme_path = results_dir / 'README.md'
        self._create_training_readme(
            readme_path, 
            timestamp, 
            aggregated_metrics,
            final_ensemble
        )
        
        return {
            'results_dir': str(results_dir),
            'ensemble_path': ensemble_path,
            'metrics_path': str(metrics_path),
            'features_path': str(features_path),
            'config_path': str(config_path),
            'validation_path': str(validation_path)
        }
    
    def _check_target_metrics(self, achieved: Dict, targets: Dict) -> Dict:
        """Verifica se métricas alvo foram atingidas"""
        results = {}
        
        for metric, target_value in targets.items():
            metric_mean = f'{metric}_mean'
            if metric_mean in achieved:
                achieved_value = achieved[metric_mean]
                results[metric] = {
                    'target': target_value,
                    'achieved': achieved_value,
                    'success': achieved_value >= target_value
                }
        
        return results
    
    # src/training/training_orchestrator.py (continuação)

    def _print_summary(self, results: Dict):
        """Imprime resumo dos resultados"""
        print("\n" + "="*60)
        print("RESUMO DO TREINAMENTO")
        print("="*60)
        
        # Período e dados
        print(f"\nPeríodo: {results['data_period']['start']} até {results['data_period']['end']}")
        print(f"Símbolos: {', '.join(results['symbols'])}")
        print(f"Método de validação: {results['validation_method']}")
        print(f"Número de folds: {results['n_folds']}")
        print(f"Features selecionadas: {len(results['selected_features'])}")
        
        # Métricas principais
        metrics = results['aggregated_metrics']
        print("\nMÉTRICAS DE PERFORMANCE:")
        print(f"Accuracy: {metrics['accuracy_mean']:.4f} (±{metrics['accuracy_std']:.4f})")
        print(f"F1 Score: {metrics['f1_score_mean']:.4f} (±{metrics['f1_score_std']:.4f})")
        print(f"Precision: {metrics['precision_mean']:.4f} (±{metrics['precision_std']:.4f})")
        print(f"Recall: {metrics['recall_mean']:.4f} (±{metrics['recall_std']:.4f})")
        
        # Métricas por classe
        print("\nPERFORMANCE POR CLASSE:")
        for i in range(3):
            class_name = ['Venda', 'Neutro', 'Compra'][i]
            prec_key = f'class_{i}_precision_mean'
            rec_key = f'class_{i}_recall_mean'
            if prec_key in metrics:
                print(f"{class_name}: Precision={metrics[prec_key]:.4f}, Recall={metrics[rec_key]:.4f}")
        
        # Confiança
        print(f"\nCONFIANÇA:")
        print(f"Média geral: {metrics['avg_confidence_mean']:.4f}")
        print(f"Quando correto: {metrics['confidence_when_correct_mean']:.4f}")
        print(f"Quando errado: {metrics['confidence_when_wrong_mean']:.4f}")
        
        # Métricas alvo
        if 'target_metrics_achieved' in results:
            print("\nMÉTRICAS ALVO:")
            for metric, info in results['target_metrics_achieved'].items():
                status = "✅" if info['success'] else "❌"
                print(f"{metric}: {info['achieved']:.4f} / {info['target']:.4f} {status}")
        
        # Caminhos salvos
        print("\nARQUIVOS SALVOS:")
        for key, path in results['save_paths'].items():
            print(f"{key}: {path}")
        
        print("="*60)

    def generate_training_report(self) -> str:
        """Gera relatório de treinamento e retorna caminho"""
        if not self.training_results:
            raise ValueError("Nenhum resultado de treinamento disponível")
        
        # O relatório já foi salvo em save_paths
        if 'save_paths' in self.training_results:
            readme_path = Path(self.training_results['save_paths']['results_dir']) / 'README.md'
            if readme_path.exists():
                return str(readme_path)
        
        return "Relatório não encontrado"
    
    def _create_training_readme(self, readme_path: Path, timestamp: str,
                              metrics: Dict, ensemble: Dict):
        """Cria arquivo README com informações do treinamento"""
        content = f"""# Resultados do Treinamento - {timestamp}

## Resumo

Este diretório contém os resultados completos do treinamento do sistema ML de trading.

## Métricas de Performance

### Métricas Gerais
- **Accuracy**: {metrics['accuracy_mean']:.4f} (±{metrics['accuracy_std']:.4f})
- **F1 Score**: {metrics['f1_score_mean']:.4f} (±{metrics['f1_score_std']:.4f})
- **Precision**: {metrics['precision_mean']:.4f} (±{metrics['precision_std']:.4f})
- **Recall**: {metrics['recall_mean']:.4f} (±{metrics['recall_std']:.4f})

### Confiança do Modelo
- **Confiança média**: {metrics['avg_confidence_mean']:.4f}
- **Confiança quando correto**: {metrics['confidence_when_correct_mean']:.4f}
- **Confiança quando errado**: {metrics['confidence_when_wrong_mean']:.4f}

## Ensemble

O ensemble final contém os seguintes modelos:
"""
        
        # Adicionar informações dos modelos
        for model_name, model_data in ensemble['models'].items():
            model_metrics = model_data['metrics']
            content += f"\n### {model_name}"
            content += f"\n- Accuracy de validação: {model_metrics['val_accuracy']:.4f}"
            content += f"\n- F1 Score: {model_metrics['val_f1']:.4f}"
            content += f"\n- Peso no ensemble: {ensemble['weights'][model_name]:.3f}\n"
        
        content += """
## Estrutura de Arquivos

- `ensemble/`: Modelos treinados do ensemble
- `aggregated_metrics.json`: Métricas agregadas de todos os folds
- `selected_features.json`: Lista de features selecionadas
- `training_config.yaml`: Configuração completa do treinamento
- `validation_summary.json`: Resumo detalhado da validação

## Como Usar

Para carregar o ensemble treinado:

```python
from src.models.model_manager import ModelManager

model_manager = ModelManager('path/to/ensemble')
model_manager.load_saved_models()
```
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)