"""
Pipeline Completo de Treinamento Book-Enhanced
Treina modelos especializados em microestrutura usando dados de book
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from src.features.book_features import BookFeatureEngineer
from src.training.validation_engine import ValidationEngine
from src.training.performance_analyzer import PerformanceAnalyzer


class BookTrainingPipeline:
    """
    Pipeline completo para treinamento de modelos book-enhanced
    Características:
    - Usa dados de book de ofertas (30-60 dias)
    - Foco em microestrutura e timing preciso
    - Múltiplos targets: spread, imbalance, price moves
    - Validação específica para HFT
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('BookTrainingPipeline')
        
        # Componentes
        self.book_engineer = BookFeatureEngineer()
        self.validation_engine = ValidationEngine()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Paths
        self.book_data_path = Path(config.get('book_data_path', 'data/realtime/book'))
        self.output_path = Path(config.get('models_path', 'models/book_enhanced'))
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Configurações de microestrutura
        self.microstructure_config = {
            'spread_bins': [0.0001, 0.0002, 0.0005, 0.001, 0.002],  # Bins para classificação de spread
            'imbalance_thresholds': [-0.3, -0.1, 0.1, 0.3],        # Thresholds de imbalance
            'price_move_horizons': [1, 5, 10, 30, 60],             # Horizontes em segundos
            'min_samples_per_target': 1000                          # Mínimo de amostras por target
        }
        
        # Targets disponíveis
        self.available_targets = {
            'spread_next': 'Predição do spread no próximo tick',
            'spread_class': 'Classificação do spread (tight/normal/wide)',
            'imbalance_next': 'Predição do book imbalance',
            'imbalance_direction': 'Direção do imbalance (buy/neutral/sell)',
            'price_move_1s': 'Movimento de preço em 1 segundo',
            'price_move_5s': 'Movimento de preço em 5 segundos',
            'liquidity_consumption': 'Taxa de consumo de liquidez',
            'sweep_probability': 'Probabilidade de sweep no próximo período'
        }
        
    def train_complete_pipeline(self,
                               symbol: str,
                               lookback_days: int = 30,
                               targets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Executa pipeline completo de treinamento book-enhanced
        
        Args:
            symbol: Símbolo para treinar
            lookback_days: Dias de histórico de book (default: 30)
            targets: Lista de targets específicos (default: todos)
            
        Returns:
            Resultados completos do treinamento
        """
        self.logger.info(f"=== PIPELINE BOOK-ENHANCED: {symbol} ===")
        self.logger.info(f"Período: últimos {lookback_days} dias")
        
        # 1. Carregar dados de book
        self.logger.info("1. Carregando dados de book...")
        book_data = self._load_book_data(symbol, lookback_days)
        
        if book_data.empty:
            self.logger.error(f"Sem dados de book disponíveis para {symbol}")
            return {'status': 'failed', 'error': 'no_book_data'}
            
        self.logger.info(f"Dados carregados: {len(book_data)} registros")
        
        # 2. Engenharia de features
        self.logger.info("2. Calculando features de microestrutura...")
        features_df = self.book_engineer.calculate_all_features(book_data)
        
        # 3. Preparar targets
        self.logger.info("3. Preparando targets...")
        if not targets:
            targets = list(self.available_targets.keys())
            
        targets_df = self._prepare_targets(book_data, features_df, targets)
        
        # 4. Treinar modelos para cada target
        self.logger.info("4. Treinando modelos especializados...")
        trained_models = {}
        
        for target_name in targets:
            if target_name not in targets_df.columns:
                self.logger.warning(f"Target {target_name} não disponível")
                continue
                
            self.logger.info(f"\n--- Target: {target_name} ---")
            
            # Preparar dados
            X, y = self._prepare_training_data(features_df, targets_df[target_name])
            
            if len(X) < self.microstructure_config['min_samples_per_target']:
                self.logger.warning(f"Dados insuficientes para {target_name}: {len(X)} amostras")
                continue
                
            # Treinar modelo
            model_result = self._train_microstructure_model(
                X, y, target_name, symbol
            )
            
            trained_models[target_name] = model_result
            
        # 5. Análise de performance
        self.logger.info("5. Analisando performance dos modelos...")
        performance_summary = self._analyze_microstructure_performance(trained_models)
        
        # 6. Criar modelo ensemble para execução
        self.logger.info("6. Criando modelo ensemble de execução...")
        execution_model = self._create_execution_ensemble(trained_models)
        
        # 7. Salvar resultados
        self.logger.info("7. Salvando resultados...")
        save_paths = self._save_pipeline_results(
            symbol, trained_models, execution_model, performance_summary
        )
        
        # Resultado final
        result = {
            'status': 'completed',
            'symbol': symbol,
            'period': {
                'days': lookback_days,
                'end_date': datetime.now().isoformat()
            },
            'data_stats': {
                'total_samples': len(book_data),
                'features_generated': len(features_df.columns),
                'targets_trained': len(trained_models)
            },
            'models': trained_models,
            'execution_ensemble': execution_model,
            'performance': performance_summary,
            'save_paths': save_paths,
            'timestamp': datetime.now().isoformat()
        }
        
        self._print_summary(result)
        
        return result
        
    def _load_book_data(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Carrega dados de book históricos"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        all_data = []
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            date_dir = self.book_data_path / date_str
            
            if date_dir.exists():
                # Carregar offer book
                for file_path in date_dir.glob(f'offer_book_{symbol}_*.parquet'):
                    try:
                        df = pd.read_parquet(file_path)
                        df['book_type'] = 'offer'
                        all_data.append(df)
                    except Exception as e:
                        self.logger.error(f"Erro ao ler {file_path}: {e}")
                        
                # Carregar price book
                for file_path in date_dir.glob(f'price_book_{symbol}_*.parquet'):
                    try:
                        df = pd.read_parquet(file_path)
                        df['book_type'] = 'price'
                        all_data.append(df)
                    except Exception as e:
                        self.logger.error(f"Erro ao ler {file_path}: {e}")
                        
            current_date += timedelta(days=1)
            
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            
            # Ordenar por timestamp
            if 'timestamp' in combined.columns:
                combined['timestamp'] = pd.to_datetime(combined['timestamp'])
                combined = combined.sort_values('timestamp')
                combined = combined.set_index('timestamp')
                
            return combined
            
        return pd.DataFrame()
        
    def _prepare_targets(self, book_data: pd.DataFrame, features: pd.DataFrame, 
                        target_names: List[str]) -> pd.DataFrame:
        """Prepara múltiplos targets para treinamento"""
        targets = pd.DataFrame(index=features.index)
        
        # Spread targets
        if 'spread' in features.columns:
            if 'spread_next' in target_names:
                targets['spread_next'] = features['spread'].shift(-1)
                
            if 'spread_class' in target_names:
                # Classificar spread em categorias
                targets['spread_class'] = pd.cut(
                    features['spread'],
                    bins=[-np.inf] + self.microstructure_config['spread_bins'] + [np.inf],
                    labels=range(len(self.microstructure_config['spread_bins']) + 1)
                )
                
        # Imbalance targets
        if 'book_imbalance' in features.columns:
            if 'imbalance_next' in target_names:
                targets['imbalance_next'] = features['book_imbalance'].shift(-1)
                
            if 'imbalance_direction' in target_names:
                # Direção do imbalance
                targets['imbalance_direction'] = pd.cut(
                    features['book_imbalance'],
                    bins=[-np.inf] + self.microstructure_config['imbalance_thresholds'] + [np.inf],
                    labels=['strong_sell', 'sell', 'neutral', 'buy', 'strong_buy']
                )
                
        # Price move targets
        if 'price' in book_data.columns:
            price = book_data['price']
            
            for horizon in self.microstructure_config['price_move_horizons']:
                target_name = f'price_move_{horizon}s'
                if target_name in target_names:
                    # Resample para o horizonte desejado
                    future_price = price.shift(-horizon, freq='s')
                    targets[target_name] = (future_price - price) / price
                    
        # Liquidity consumption
        if 'liquidity_consumption' in target_names:
            if 'total_depth' in features.columns:
                # Taxa de mudança na profundidade
                targets['liquidity_consumption'] = features['total_depth'].pct_change().shift(-1)
                
        # Sweep probability
        if 'sweep_probability' in target_names:
            if 'bid_sweep_signal' in features.columns and 'ask_sweep_signal' in features.columns:
                # Probabilidade de sweep nos próximos N ticks
                sweep_window = 10
                targets['sweep_probability'] = (
                    (features['bid_sweep_signal'].rolling(sweep_window).sum() > 0) |
                    (features['ask_sweep_signal'].rolling(sweep_window).sum() > 0)
                ).shift(-sweep_window).astype(float)
                
        return targets
        
    def _prepare_training_data(self, features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara dados para treinamento removendo NaN"""
        # Remover colunas com muitos NaN
        nan_threshold = 0.3
        nan_ratio = features.isna().sum() / len(features)
        valid_cols = nan_ratio[nan_ratio < nan_threshold].index
        
        X = features[valid_cols]
        y = target
        
        # Remover linhas com NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        return X_clean, y_clean
        
    def _train_microstructure_model(self, X: pd.DataFrame, y: pd.Series,
                                   target_name: str, symbol: str) -> Dict[str, Any]:
        """Treina modelo especializado em microestrutura"""
        # Determinar tipo de problema
        is_regression = y.dtype in ['float64', 'float32', 'int64', 'int32']
        
        # Configurar modelos
        if is_regression:
            models = {
                'xgboost': xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                ),
                'lightgbm': lgb.LGBMRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_split=10,
                    random_state=42
                )
            }
        else:
            # Classificação
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
            
            models = {
                'xgboost': xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                ),
                'lightgbm': lgb.LGBMClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            }
            
        # Time series split para validação
        tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X) * 0.1))
        
        model_results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Treinando {model_name} para {target_name}...")
            
            # Cross-validation scores
            cv_scores = []
            feature_importances = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Treinar
                model.fit(X_train, y_train)
                
                # Validar
                y_pred = model.predict(X_val)
                
                if is_regression:
                    mae = mean_absolute_error(y_val, y_pred)
                    mse = mean_squared_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    
                    fold_score = {
                        'mae': mae,
                        'mse': mse,
                        'rmse': np.sqrt(mse),
                        'r2': r2
                    }
                else:
                    from sklearn.metrics import accuracy_score, f1_score
                    
                    fold_score = {
                        'accuracy': accuracy_score(y_val, y_pred),
                        'f1_score': f1_score(y_val, y_pred, average='weighted')
                    }
                    
                cv_scores.append(fold_score)
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_importances.append(model.feature_importances_)
                    
            # Treinar modelo final com todos os dados
            model.fit(X, y)
            
            # Calcular métricas agregadas
            avg_scores = {}
            for metric in cv_scores[0].keys():
                values = [s[metric] for s in cv_scores]
                avg_scores[f'{metric}_mean'] = np.mean(values)
                avg_scores[f'{metric}_std'] = np.std(values)
                
            # Feature importance média
            if feature_importances:
                avg_importance = np.mean(feature_importances, axis=0)
                feature_importance_dict = dict(zip(X.columns, avg_importance))
                # Top 10 features
                top_features = dict(sorted(
                    feature_importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10])
            else:
                top_features = {}
                
            model_results[model_name] = {
                'model': model,
                'cv_scores': avg_scores,
                'feature_importance': top_features,
                'is_regression': is_regression
            }
            
        # Selecionar melhor modelo
        if is_regression:
            best_model_name = min(
                model_results.keys(),
                key=lambda x: model_results[x]['cv_scores'].get('mae_mean', float('inf'))
            )
        else:
            best_model_name = max(
                model_results.keys(),
                key=lambda x: model_results[x]['cv_scores'].get('accuracy_mean', 0)
            )
            
        best_model_data = model_results[best_model_name]
        
        # Salvar modelo
        model_path = self.output_path / symbol / target_name / 'model.pkl'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model_data['model'], model_path)
        
        # Salvar metadata
        metadata = {
            'target': target_name,
            'symbol': symbol,
            'model_type': best_model_name,
            'is_regression': is_regression,
            'features': list(X.columns),
            'n_samples': len(X),
            'cv_scores': best_model_data['cv_scores'],
            'top_features': best_model_data['feature_importance'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(model_path.parent / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return {
            'model_path': str(model_path),
            'best_model': best_model_name,
            'all_models': model_results,
            'metadata': metadata
        }
        
    def _create_execution_ensemble(self, trained_models: Dict[str, Dict]) -> Dict[str, Any]:
        """Cria ensemble para execução em tempo real"""
        # Componentes do ensemble
        ensemble_components = {
            'spread_predictor': None,
            'imbalance_analyzer': None,
            'price_predictor': None,
            'execution_optimizer': None
        }
        
        # Mapear modelos para componentes
        for target_name, model_data in trained_models.items():
            if 'spread' in target_name and ensemble_components['spread_predictor'] is None:
                ensemble_components['spread_predictor'] = {
                    'model_path': model_data['model_path'],
                    'target': target_name,
                    'performance': model_data['metadata']['cv_scores']
                }
            elif 'imbalance' in target_name and ensemble_components['imbalance_analyzer'] is None:
                ensemble_components['imbalance_analyzer'] = {
                    'model_path': model_data['model_path'],
                    'target': target_name,
                    'performance': model_data['metadata']['cv_scores']
                }
            elif 'price_move' in target_name and ensemble_components['price_predictor'] is None:
                ensemble_components['price_predictor'] = {
                    'model_path': model_data['model_path'],
                    'target': target_name,
                    'performance': model_data['metadata']['cv_scores']
                }
                
        # Criar estratégia de execução
        execution_strategy = {
            'components': ensemble_components,
            'decision_rules': {
                'entry': {
                    'spread_threshold': 0.0002,  # Entrar apenas com spread < 2 bps
                    'imbalance_min': 0.3,        # Imbalance mínimo para direção
                    'confidence_min': 0.7         # Confiança mínima
                },
                'exit': {
                    'profit_target': 0.0005,      # 5 bps de lucro
                    'stop_loss': 0.0003,          # 3 bps de perda
                    'time_limit': 60              # Máximo 60 segundos
                }
            },
            'risk_params': {
                'max_position': 100,
                'max_orders_per_second': 10,
                'min_edge': 0.0001            # Edge mínimo de 1 bp
            }
        }
        
        return execution_strategy
        
    def _analyze_microstructure_performance(self, trained_models: Dict[str, Dict]) -> Dict[str, Any]:
        """Analisa performance específica para microestrutura"""
        performance = {
            'by_target': {},
            'overall': {
                'n_models': len(trained_models),
                'regression_models': 0,
                'classification_models': 0
            }
        }
        
        regression_scores = []
        classification_scores = []
        
        for target_name, model_data in trained_models.items():
            cv_scores = model_data['metadata']['cv_scores']
            is_regression = model_data['metadata']['is_regression']
            
            if is_regression:
                performance['overall']['regression_models'] += 1
                perf_summary = {
                    'mae': cv_scores.get('mae_mean', 0),
                    'rmse': cv_scores.get('rmse_mean', 0),
                    'r2': cv_scores.get('r2_mean', 0)
                }
                regression_scores.append(perf_summary)
            else:
                performance['overall']['classification_models'] += 1
                perf_summary = {
                    'accuracy': cv_scores.get('accuracy_mean', 0),
                    'f1_score': cv_scores.get('f1_score_mean', 0)
                }
                classification_scores.append(perf_summary)
                
            performance['by_target'][target_name] = {
                'type': 'regression' if is_regression else 'classification',
                'best_model': model_data['best_model'],
                'performance': perf_summary,
                'top_features': list(model_data['metadata']['top_features'].keys())[:5]
            }
            
        # Calcular médias gerais
        if regression_scores:
            performance['overall']['avg_regression_r2'] = np.mean([s['r2'] for s in regression_scores])
            performance['overall']['avg_regression_mae'] = np.mean([s['mae'] for s in regression_scores])
            
        if classification_scores:
            performance['overall']['avg_classification_accuracy'] = np.mean([s['accuracy'] for s in classification_scores])
            performance['overall']['avg_classification_f1'] = np.mean([s['f1_score'] for s in classification_scores])
            
        return performance
        
    def _save_pipeline_results(self, symbol: str, trained_models: Dict,
                              execution_ensemble: Dict, performance: Dict) -> Dict[str, str]:
        """Salva todos os resultados do pipeline"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = self.output_path / symbol / f'pipeline_{timestamp}'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Salvar resumo do pipeline
        summary = {
            'symbol': symbol,
            'timestamp': timestamp,
            'models_trained': list(trained_models.keys()),
            'performance': performance,
            'execution_ensemble': execution_ensemble
        }
        
        summary_path = results_dir / 'pipeline_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # 2. Salvar estratégia de execução
        execution_path = results_dir / 'execution_strategy.json'
        with open(execution_path, 'w') as f:
            json.dump(execution_ensemble, f, indent=2)
            
        # 3. Criar README
        readme_content = f"""# Pipeline Book-Enhanced - {symbol}

## Resumo

Data: {timestamp}
Período: Últimos 30 dias de dados de book

## Modelos Treinados

Total: {len(trained_models)} modelos especializados

### Targets de Regressão
"""
        
        for target, perf in performance['by_target'].items():
            if perf['type'] == 'regression':
                readme_content += f"\n#### {target}
- Modelo: {perf['best_model']}
- MAE: {perf['performance'].get('mae', 0):.6f}
- R²: {perf['performance'].get('r2', 0):.4f}
- Top Features: {', '.join(perf['top_features'])}
"
                
        readme_content += "\n### Targets de Classificação\n"
        
        for target, perf in performance['by_target'].items():
            if perf['type'] == 'classification':
                readme_content += f"\n#### {target}
- Modelo: {perf['best_model']}
- Accuracy: {perf['performance'].get('accuracy', 0):.4f}
- F1 Score: {perf['performance'].get('f1_score', 0):.4f}
- Top Features: {', '.join(perf['top_features'])}
"
                
        readme_content += f"\n## Estratégia de Execução

### Componentes
- Spread Predictor: {'✓' if execution_ensemble['components']['spread_predictor'] else '✗'}
- Imbalance Analyzer: {'✓' if execution_ensemble['components']['imbalance_analyzer'] else '✗'}
- Price Predictor: {'✓' if execution_ensemble['components']['price_predictor'] else '✗'}

### Parâmetros de Risco
- Posição Máxima: {execution_ensemble['risk_params']['max_position']}
- Orders/segundo: {execution_ensemble['risk_params']['max_orders_per_second']}
- Edge Mínimo: {execution_ensemble['risk_params']['min_edge']*10000:.1f} bps

## Como Usar

```python
from src.trading.book_execution import BookExecutionEngine

engine = BookExecutionEngine('{str(results_dir)}')
engine.start_execution('WDOU25')
```
"""
        
        readme_path = results_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
            
        return {
            'results_dir': str(results_dir),
            'summary': str(summary_path),
            'execution_strategy': str(execution_path),
            'readme': str(readme_path)
        }
        
    def _print_summary(self, result: Dict):
        """Imprime resumo dos resultados"""
        print("\n" + "="*60)
        print(f"PIPELINE BOOK-ENHANCED COMPLETO: {result['symbol']}")
        print("="*60)
        
        print(f"\nPeríodo: {result['period']['days']} dias")
        print(f"Total de amostras: {result['data_stats']['total_samples']:,}")
        print(f"Features geradas: {result['data_stats']['features_generated']}")
        print(f"Modelos treinados: {result['data_stats']['targets_trained']}")
        
        print("\nPerformance Geral:")
        overall = result['performance']['overall']
        
        if 'avg_regression_r2' in overall:
            print(f"  Regressão - R² médio: {overall['avg_regression_r2']:.4f}")
            print(f"  Regressão - MAE médio: {overall['avg_regression_mae']:.6f}")
            
        if 'avg_classification_accuracy' in overall:
            print(f"  Classificação - Accuracy médio: {overall['avg_classification_accuracy']:.4f}")
            print(f"  Classificação - F1 médio: {overall['avg_classification_f1']:.4f}")
            
        print("\nModelos por Target:")
        for target, perf in result['performance']['by_target'].items():
            if perf['type'] == 'regression':
                print(f"  {target}: R²={perf['performance']['r2']:.3f}, MAE={perf['performance']['mae']:.5f}")
            else:
                print(f"  {target}: Acc={perf['performance']['accuracy']:.3f}, F1={perf['performance']['f1_score']:.3f}")
                
        print(f"\nResultados salvos em: {result['save_paths']['results_dir']}")
        print("="*60)


# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuração
    config = {
        'book_data_path': 'data/realtime/book',
        'models_path': 'models/book_enhanced'
    }
    
    # Criar pipeline
    pipeline = BookTrainingPipeline(config)
    
    # Treinar para símbolo com targets específicos
    result = pipeline.train_complete_pipeline(
        symbol='WDOU25',
        lookback_days=30,
        targets=['spread_next', 'imbalance_direction', 'price_move_5s']
    )
    
    print(f"\nStatus: {result['status']}")