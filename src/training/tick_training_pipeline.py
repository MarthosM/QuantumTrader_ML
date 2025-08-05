"""
Pipeline Completo de Treinamento Tick-Only
Integra todos os componentes para treinar modelos com dados tick-a-tick
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import joblib

from src.training.training_orchestrator import TrainingOrchestrator
from src.training.regime_analyzer import RegimeAnalyzer
from src.training.feature_pipeline import FeatureEngineeringPipeline
from src.training.validation_engine import ValidationEngine
from src.training.performance_analyzer import PerformanceAnalyzer


class TickTrainingPipeline:
    """
    Pipeline completo para treinamento de modelos tick-only
    Características:
    - Usa 1 ano de dados históricos tick-a-tick
    - Treina modelos separados por regime de mercado
    - Seleção automática de features
    - Validação temporal (walk-forward)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('TickTrainingPipeline')
        
        # Componentes
        self.orchestrator = TrainingOrchestrator(config)
        self.regime_analyzer = RegimeAnalyzer()
        self.feature_pipeline = FeatureEngineeringPipeline()
        self.validation_engine = ValidationEngine()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Paths
        self.data_path = Path(config.get('tick_data_path', 'data/historical'))
        self.output_path = Path(config.get('models_path', 'models/tick_only'))
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Feature sets por regime
        self.regime_features = {
            'trend_up': {
                'primary': ['momentum_5', 'momentum_10', 'roc_5', 'roc_10', 
                           'sma_5', 'sma_10', 'macd', 'adx_14'],
                'secondary': ['volume_ratio', 'buy_pressure', 'higher_highs']
            },
            'trend_down': {
                'primary': ['momentum_5', 'momentum_10', 'roc_5', 'roc_10',
                           'sma_5', 'sma_10', 'macd', 'adx_14'],
                'secondary': ['volume_ratio', 'sell_pressure', 'lower_lows']
            },
            'range': {
                'primary': ['rsi_14', 'bollinger_upper', 'bollinger_lower',
                           'volatility_5', 'volatility_20'],
                'secondary': ['volume_imbalance', 'spread_mean', 'atr_14']
            }
        }
        
    def train_complete_pipeline(self, 
                               symbol: str,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               lookback_days: int = 365) -> Dict[str, Any]:
        """
        Executa pipeline completo de treinamento tick-only
        
        Args:
            symbol: Símbolo para treinar
            start_date: Data inicial (default: 1 ano atrás)
            end_date: Data final (default: hoje)
            lookback_days: Dias de histórico (default: 365)
            
        Returns:
            Resultados completos do treinamento
        """
        self.logger.info(f"=== PIPELINE TICK-ONLY: {symbol} ===")
        
        # Configurar datas
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=lookback_days)
            
        # 1. Carregar dados históricos
        self.logger.info("1. Carregando dados históricos...")
        tick_data = self._load_historical_data(symbol, start_date, end_date)
        
        if tick_data.empty:
            self.logger.error(f"Sem dados disponíveis para {symbol}")
            return {'status': 'failed', 'error': 'no_data'}
            
        self.logger.info(f"Dados carregados: {len(tick_data)} registros")
        
        # 2. Análise de regimes
        self.logger.info("2. Analisando regimes de mercado...")
        regime_analysis = self._analyze_regimes(tick_data)
        
        # 3. Preparar features
        self.logger.info("3. Preparando features...")
        all_features = self._prepare_all_features(tick_data)
        
        # 4. Treinar modelos por regime
        self.logger.info("4. Treinando modelos por regime...")
        regime_models = {}
        
        for regime in ['trend_up', 'trend_down', 'range']:
            self.logger.info(f"\n--- Regime: {regime} ---")
            
            # Filtrar dados do regime
            regime_data = self._filter_by_regime(all_features, regime_analysis, regime)
            
            if len(regime_data) < 1000:
                self.logger.warning(f"Dados insuficientes para {regime}: {len(regime_data)} amostras")
                continue
                
            # Selecionar features específicas do regime
            selected_features = self._select_regime_features(regime_data, regime)
            
            # Treinar modelo
            model_result = self._train_regime_model(
                regime_data[selected_features], 
                regime, 
                symbol
            )
            
            regime_models[regime] = model_result
            
        # 5. Análise de performance agregada
        self.logger.info("5. Analisando performance agregada...")
        performance_summary = self._analyze_performance(regime_models)
        
        # 6. Salvar resultados
        self.logger.info("6. Salvando resultados...")
        save_paths = self._save_pipeline_results(
            symbol, regime_models, performance_summary, regime_analysis
        )
        
        # Resultado final
        result = {
            'status': 'completed',
            'symbol': symbol,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': (end_date - start_date).days
            },
            'data_stats': {
                'total_samples': len(tick_data),
                'features_generated': len(all_features.columns),
                'regime_distribution': regime_analysis['distribution']
            },
            'models': regime_models,
            'performance': performance_summary,
            'save_paths': save_paths,
            'timestamp': datetime.now().isoformat()
        }
        
        self._print_summary(result)
        
        return result
        
    def _load_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Carrega dados históricos tick-a-tick"""
        all_data = []
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            
            # Tentar múltiplos caminhos possíveis
            possible_paths = [
                self.data_path / symbol / date_str / 'trades.parquet',
                self.data_path / symbol / f'{date_str}_trades.parquet',
                self.data_path / f'{symbol}_{date_str}.parquet'
            ]
            
            for file_path in possible_paths:
                if file_path.exists():
                    try:
                        df = pd.read_parquet(file_path)
                        all_data.append(df)
                        break
                    except Exception as e:
                        self.logger.error(f"Erro ao ler {file_path}: {e}")
                        
            current_date += timedelta(days=1)
            
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            
            # Garantir ordenação temporal
            if 'timestamp' in combined.columns:
                combined = combined.sort_values('timestamp')
                
            return combined
            
        return pd.DataFrame()
        
    def _analyze_regimes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analisa distribuição de regimes no período"""
        regime_counts = {
            'trend_up': 0,
            'trend_down': 0,
            'range': 0,
            'undefined': 0
        }
        
        regime_series = []
        window_size = 50
        
        # Analisar regime para cada janela
        for i in range(window_size, len(data), 10):  # Passo de 10 para eficiência
            window = data.iloc[i-window_size:i]
            
            # Preparar dados para RegimeAnalyzer
            candles_df = pd.DataFrame({
                'open': window.get('open', window.get('price', 0)),
                'high': window.get('high', window.get('price', 0)),
                'low': window.get('low', window.get('price', 0)),
                'close': window.get('close', window.get('price', 0)),
                'volume': window.get('volume', 0)
            })
            
            regime_info = self.regime_analyzer.analyze_market(candles_df)
            regime = regime_info['regime']
            
            regime_counts[regime] += 1
            regime_series.append({
                'index': i,
                'regime': regime,
                'confidence': regime_info['confidence'],
                'adx': regime_info.get('adx', 0)
            })
            
        # Calcular distribuição percentual
        total = sum(regime_counts.values())
        distribution = {k: v/total*100 for k, v in regime_counts.items()}
        
        return {
            'counts': regime_counts,
            'distribution': distribution,
            'series': regime_series,
            'dominant_regime': max(regime_counts, key=regime_counts.get)
        }
        
    def _prepare_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepara todas as features disponíveis"""
        # Usar FeatureEngineeringPipeline
        features = self.feature_pipeline.create_training_features(
            data,
            feature_groups=['technical', 'momentum', 'volatility', 'microstructure'],
            parallel=True
        )
        
        # Adicionar target
        if 'price' in data.columns:
            # Target: direção do preço em 5 períodos
            future_returns = data['price'].pct_change(5).shift(-5)
            features['target'] = pd.cut(
                future_returns,
                bins=[-np.inf, -0.001, 0.001, np.inf],
                labels=[0, 1, 2]  # 0: venda, 1: neutro, 2: compra
            )
            
        return features
        
    def _filter_by_regime(self, data: pd.DataFrame, regime_analysis: Dict, target_regime: str) -> pd.DataFrame:
        """Filtra dados por regime específico"""
        regime_indices = []
        
        for regime_point in regime_analysis['series']:
            if regime_point['regime'] == target_regime:
                # Adicionar índices ao redor do ponto
                start = max(0, regime_point['index'] - 25)
                end = min(len(data), regime_point['index'] + 25)
                regime_indices.extend(range(start, end))
                
        # Remover duplicatas e ordenar
        regime_indices = sorted(list(set(regime_indices)))
        
        if regime_indices:
            return data.iloc[regime_indices]
        else:
            return pd.DataFrame()
            
    def _select_regime_features(self, data: pd.DataFrame, regime: str) -> List[str]:
        """Seleciona features específicas para o regime"""
        # Features base do regime
        base_features = (self.regime_features[regime]['primary'] + 
                        self.regime_features[regime]['secondary'])
        
        # Filtrar features disponíveis
        available_features = [f for f in base_features if f in data.columns]
        
        # Adicionar seleção automática se necessário
        if len(available_features) < 20:
            # Usar seleção automática para completar
            if 'target' in data.columns:
                additional_features = self.feature_pipeline.select_top_features(
                    data.drop('target', axis=1),
                    data['target'],
                    n_features=30,
                    method='mutual_info'
                )
                
                # Combinar com features existentes
                all_features = list(set(available_features + additional_features))[:30]
                return all_features
                
        return available_features
        
    def _train_regime_model(self, data: pd.DataFrame, regime: str, symbol: str) -> Dict[str, Any]:
        """Treina modelo específico para um regime"""
        # Configuração específica por regime
        regime_config = {
            'trend_up': {
                'model_types': ['xgboost_fast', 'lightgbm_balanced'],
                'hyperparams': {
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'n_estimators': 200
                }
            },
            'trend_down': {
                'model_types': ['xgboost_fast', 'lightgbm_balanced'],
                'hyperparams': {
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'n_estimators': 200
                }
            },
            'range': {
                'model_types': ['random_forest_stable', 'lightgbm_balanced'],
                'hyperparams': {
                    'learning_rate': 0.05,
                    'max_depth': 4,
                    'n_estimators': 300
                }
            }
        }
        
        # Preparar dados
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Dividir dados temporalmente
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Treinar ensemble usando orchestrator
        from src.training.ensemble_trainer import EnsembleTrainer
        from src.training.model_trainer import ModelTrainer
        
        model_trainer = ModelTrainer(str(self.output_path / symbol / regime))
        ensemble_trainer = EnsembleTrainer(model_trainer)
        
        # Treinar
        ensemble_result = ensemble_trainer.train_ensemble(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_types=regime_config[regime]['model_types'],
            parallel=False
        )
        
        # Validar com walk-forward
        validation_results = self._walk_forward_validation(X, y, ensemble_result)
        
        # Salvar modelo
        model_path = self.output_path / symbol / regime / 'model.pkl'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Salvar melhor modelo do ensemble
        best_model_name = max(ensemble_result['weights'], key=ensemble_result['weights'].get)
        best_model = ensemble_result['models'][best_model_name]['model']
        joblib.dump(best_model, model_path)
        
        # Salvar metadata
        metadata = {
            'regime': regime,
            'symbol': symbol,
            'features': list(X.columns),
            'n_samples': len(X),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'ensemble_weights': ensemble_result['weights'],
            'validation_results': validation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(model_path.parent / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return {
            'model_path': str(model_path),
            'ensemble': ensemble_result,
            'validation': validation_results,
            'metadata': metadata
        }
        
    def _walk_forward_validation(self, X: pd.DataFrame, y: pd.Series, 
                                ensemble: Dict) -> Dict[str, Any]:
        """Validação walk-forward para séries temporais"""
        results = []
        window_size = int(len(X) * 0.6)  # 60% para treino inicial
        step_size = int(len(X) * 0.1)    # 10% de step
        
        for i in range(window_size, len(X) - step_size, step_size):
            # Treino: todos os dados até i
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            
            # Teste: próximo step
            X_test = X.iloc[i:i+step_size]
            y_test = y.iloc[i:i+step_size]
            
            # Fazer predições
            predictions = []
            for name, model_data in ensemble['models'].items():
                model = model_data['model']
                pred = model.predict(X_test)
                predictions.append(pred)
                
            # Ensemble prediction (maioria)
            ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
            
            # Calcular métricas
            from sklearn.metrics import accuracy_score, f1_score
            
            fold_result = {
                'fold': len(results),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'accuracy': accuracy_score(y_test, ensemble_pred),
                'f1_score': f1_score(y_test, ensemble_pred, average='weighted')
            }
            
            results.append(fold_result)
            
        # Agregar resultados
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        avg_f1 = np.mean([r['f1_score'] for r in results])
        
        return {
            'method': 'walk_forward',
            'n_folds': len(results),
            'avg_accuracy': avg_accuracy,
            'avg_f1_score': avg_f1,
            'fold_results': results
        }
        
    def _analyze_performance(self, regime_models: Dict[str, Dict]) -> Dict[str, Any]:
        """Analisa performance agregada dos modelos"""
        overall_metrics = {
            'accuracy': [],
            'f1_score': [],
            'samples': []
        }
        
        regime_performance = {}
        
        for regime, model_data in regime_models.items():
            val_results = model_data['validation']
            
            regime_performance[regime] = {
                'accuracy': val_results['avg_accuracy'],
                'f1_score': val_results['avg_f1_score'],
                'n_samples': model_data['metadata']['n_samples'],
                'model_path': model_data['model_path']
            }
            
            # Adicionar às métricas gerais
            overall_metrics['accuracy'].append(val_results['avg_accuracy'])
            overall_metrics['f1_score'].append(val_results['avg_f1_score'])
            overall_metrics['samples'].append(model_data['metadata']['n_samples'])
            
        # Calcular médias ponderadas
        total_samples = sum(overall_metrics['samples'])
        weights = [s/total_samples for s in overall_metrics['samples']]
        
        weighted_accuracy = sum(a*w for a, w in zip(overall_metrics['accuracy'], weights))
        weighted_f1 = sum(f*w for f, w in zip(overall_metrics['f1_score'], weights))
        
        return {
            'overall': {
                'weighted_accuracy': weighted_accuracy,
                'weighted_f1_score': weighted_f1,
                'total_samples': total_samples,
                'n_regimes': len(regime_models)
            },
            'by_regime': regime_performance
        }
        
    def _save_pipeline_results(self, symbol: str, regime_models: Dict,
                              performance: Dict, regime_analysis: Dict) -> Dict[str, str]:
        """Salva todos os resultados do pipeline"""
        # Diretório principal
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = self.output_path / symbol / f'pipeline_{timestamp}'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Salvar resumo do pipeline
        summary = {
            'symbol': symbol,
            'timestamp': timestamp,
            'regime_distribution': regime_analysis['distribution'],
            'performance': performance,
            'models': {
                regime: {
                    'path': model_data['model_path'],
                    'features': model_data['metadata']['features'][:10],  # Top 10
                    'validation': {
                        'accuracy': model_data['validation']['avg_accuracy'],
                        'f1_score': model_data['validation']['avg_f1_score']
                    }
                }
                for regime, model_data in regime_models.items()
            }
        }
        
        summary_path = results_dir / 'pipeline_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # 2. Criar README
        readme_content = f"""# Pipeline de Treinamento Tick-Only - {symbol}

## Resumo

Data: {timestamp}
Período: 365 dias de dados tick-a-tick

## Distribuição de Regimes

"""
        for regime, pct in regime_analysis['distribution'].items():
            readme_content += f"- **{regime}**: {pct:.1f}%\n"
            
        readme_content += f"\n## Performance Geral

- **Accuracy Ponderada**: {performance['overall']['weighted_accuracy']:.4f}
- **F1 Score Ponderado**: {performance['overall']['weighted_f1_score']:.4f}
- **Total de Amostras**: {performance['overall']['total_samples']:,}

## Performance por Regime

"""
        for regime, perf in performance['by_regime'].items():
            readme_content += f"### {regime.upper()}
- Accuracy: {perf['accuracy']:.4f}
- F1 Score: {perf['f1_score']:.4f}
- Amostras: {perf['n_samples']:,}
- Modelo: `{perf['model_path']}`

"""
        
        readme_path = results_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
            
        return {
            'results_dir': str(results_dir),
            'summary': str(summary_path),
            'readme': str(readme_path)
        }
        
    def _print_summary(self, result: Dict):
        """Imprime resumo dos resultados"""
        print("\n" + "="*60)
        print(f"PIPELINE TICK-ONLY COMPLETO: {result['symbol']}")
        print("="*60)
        
        print(f"\nPeríodo: {result['period']['days']} dias")
        print(f"Total de amostras: {result['data_stats']['total_samples']:,}")
        print(f"Features geradas: {result['data_stats']['features_generated']}")
        
        print("\nDistribuição de Regimes:")
        for regime, pct in result['data_stats']['regime_distribution'].items():
            print(f"  {regime}: {pct:.1f}%")
            
        print("\nPerformance Geral:")
        overall = result['performance']['overall']
        print(f"  Accuracy: {overall['weighted_accuracy']:.4f}")
        print(f"  F1 Score: {overall['weighted_f1_score']:.4f}")
        
        print("\nModelos Treinados:")
        for regime, perf in result['performance']['by_regime'].items():
            print(f"  {regime}: Acc={perf['accuracy']:.3f}, F1={perf['f1_score']:.3f}")
            
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
        'tick_data_path': 'data/historical',
        'models_path': 'models/tick_only',
        'model_save_path': 'models/tick_only'
    }
    
    # Criar pipeline
    pipeline = TickTrainingPipeline(config)
    
    # Treinar para símbolo
    result = pipeline.train_complete_pipeline(
        symbol='WDOU25',
        lookback_days=365
    )
    
    print(f"\nStatus: {result['status']}")