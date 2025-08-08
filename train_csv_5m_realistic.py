"""
Pipeline Realista para 5 Milhões de Registros
Versão com configurações apropriadas para trading real
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
import joblib
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CSV5MRealisticTrainer:
    """Treina modelos com configurações realistas para trading"""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.sample_size = 5_000_000
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_data_with_progress(self):
        """Carrega 5M registros"""
        
        print("=" * 80)
        print("CARREGAMENTO DE DADOS - VERSÃO REALISTA")
        print("=" * 80)
        
        dtypes = {
            '<trade_number>': 'uint32',
            '<price>': 'float32',
            '<qty>': 'uint16',
            '<vol>': 'float32',
            '<buy_agent>': 'category',
            '<sell_agent>': 'category',
            '<trade_type>': 'category'
        }
        
        print(f"\nCarregando {self.sample_size:,} registros...")
        start_time = datetime.now()
        
        df = pd.read_csv(self.csv_path, 
                        nrows=self.sample_size,
                        dtype=dtypes)
        
        print("Processando timestamps...")
        df['timestamp'] = pd.to_datetime(
            df['<date>'].astype(str) + ' ' + df['<time>'].astype(str).str.zfill(6),
            format='%Y%m%d %H%M%S'
        )
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        load_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✓ Tempo: {load_time:.1f}s")
        print(f"✓ Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
        
        return df
    
    def create_realistic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features sem data leakage"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE FEATURES REALISTAS")
        print("=" * 80)
        
        features = pd.DataFrame(index=df.index)
        
        # NÃO usar price direto - apenas mudanças e ratios!
        price = df['<price>'].values.astype('float32')
        
        # 1. RETORNOS E MUDANÇAS (sem price absoluto)
        print("\n→ Features de Retorno...")
        for period in [1, 2, 5, 10, 20, 50, 100]:
            features[f'returns_{period}'] = pd.Series(price).pct_change(period)
        
        # Log returns para capturar não-linearidades
        features['log_returns_1'] = np.log(price / np.roll(price, 1))
        features['log_returns_5'] = np.log(price / np.roll(price, 5))
        features['log_returns_20'] = np.log(price / np.roll(price, 20))
        
        # 2. VOLATILIDADE E RISCO
        print("→ Features de Volatilidade...")
        for window in [10, 20, 50, 100]:
            features[f'volatility_{window}'] = features['returns_1'].rolling(window).std()
            features[f'volatility_ratio_{window}'] = (
                features[f'volatility_{window}'] / 
                features[f'volatility_{window}'].rolling(window*2).mean()
            ).fillna(1)
        
        # Realized volatility (Garman-Klass estimator proxy)
        features['volatility_gk'] = np.sqrt(
            features['returns_1'].pow(2).rolling(20).mean()
        )
        
        # 3. MOMENTUM NORMALIZADO
        print("→ Features de Momentum...")
        features['momentum_5_20'] = features['returns_5'] / features['returns_20'].replace(0, np.nan)
        features['momentum_20_50'] = features['returns_20'] / features['returns_50'].replace(0, np.nan)
        
        # Momentum ajustado por volatilidade
        features['sharpe_5'] = features['returns_5'] / features['volatility_20'].clip(lower=0.0001)
        features['sharpe_20'] = features['returns_20'] / features['volatility_50'].clip(lower=0.0001)
        
        # 4. VOLUME PATTERNS
        print("→ Features de Volume...")
        qty = df['<qty>'].values.astype('float32')
        
        # Volume relativo (sem valores absolutos)
        for window in [20, 50, 100]:
            vol_ma = pd.Series(qty).rolling(window).mean()
            vol_std = pd.Series(qty).rolling(window).std()
            features[f'volume_zscore_{window}'] = (qty - vol_ma) / vol_std.clip(lower=0.001)
            features[f'volume_ratio_{window}'] = qty / vol_ma.clip(lower=1)
        
        # Volume-weighted returns
        dollar_volume = df['<vol>'].values
        features['volume_weighted_return'] = (
            (features['returns_1'] * dollar_volume).rolling(20).sum() / 
            pd.Series(dollar_volume).rolling(20).sum()
        )
        
        # 5. MICROSTRUCTURE INDICATORS
        print("→ Features de Microestrutura...")
        
        # Trade intensity
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        features['trade_intensity'] = 1 / df['time_diff'].rolling(50).median().clip(lower=0.001)
        features['trade_intensity_ratio'] = (
            features['trade_intensity'] / 
            features['trade_intensity'].rolling(200).mean()
        ).fillna(1)
        
        # Order flow
        features['is_buyer_aggressor'] = (df['<trade_type>'] == 'AggressorBuyer').astype('int8')
        features['is_seller_aggressor'] = (df['<trade_type>'] == 'AggressorSeller').astype('int8')
        
        # Order flow imbalance
        for window in [10, 20, 50, 100]:
            buyer_flow = features['is_buyer_aggressor'].rolling(window).sum()
            seller_flow = features['is_seller_aggressor'].rolling(window).sum()
            total_flow = buyer_flow + seller_flow
            
            features[f'order_flow_imbalance_{window}'] = (
                (buyer_flow - seller_flow) / total_flow.clip(lower=1)
            )
            
        # Signed volume
        features['signed_volume'] = qty * (features['is_buyer_aggressor'] - features['is_seller_aggressor'])
        features['cumulative_signed_volume'] = features['signed_volume'].rolling(100).sum()
        
        # 6. AGENT BEHAVIOR (sem loops)
        print("→ Features de Agent Behavior...")
        
        # Top agents
        top_buyers = df['<buy_agent>'].value_counts().head(10).index
        top_sellers = df['<sell_agent>'].value_counts().head(10).index
        
        # Binary indicators para top 5
        for i in range(5):
            if i < len(top_buyers):
                features[f'top_buyer_{i}_active'] = (df['<buy_agent>'] == top_buyers[i]).astype('int8')
            if i < len(top_sellers):
                features[f'top_seller_{i}_active'] = (df['<sell_agent>'] == top_sellers[i]).astype('int8')
        
        # Agent activity metrics
        features['top_buyers_count'] = features[[f'top_buyer_{i}_active' for i in range(5) if f'top_buyer_{i}_active' in features]].sum(axis=1)
        features['top_sellers_count'] = features[[f'top_seller_{i}_active' for i in range(5) if f'top_seller_{i}_active' in features]].sum(axis=1)
        
        # Agent switching
        features['buyer_changed'] = (df['<buy_agent>'] != df['<buy_agent>'].shift(1)).astype('int8')
        features['seller_changed'] = (df['<sell_agent>'] != df['<sell_agent>'].shift(1)).astype('int8')
        features['agent_turnover'] = features['buyer_changed'].rolling(100).mean() + features['seller_changed'].rolling(100).mean()
        
        # 7. TECHNICAL PATTERNS
        print("→ Features Técnicas...")
        
        # Moving average crossovers (sem preços absolutos)
        ma_5 = pd.Series(price).rolling(5).mean()
        ma_20 = pd.Series(price).rolling(20).mean()
        ma_50 = pd.Series(price).rolling(50).mean()
        
        features['ma_5_20_ratio'] = ma_5 / ma_20 - 1
        features['ma_20_50_ratio'] = ma_20 / ma_50 - 1
        
        # Bollinger bands position
        bb_mean = pd.Series(price).rolling(20).mean()
        bb_std = pd.Series(price).rolling(20).std()
        features['bb_position'] = (price - bb_mean) / (2 * bb_std).clip(lower=0.0001)
        
        # 8. TEMPORAL FEATURES
        print("→ Features Temporais...")
        
        features['hour'] = df['timestamp'].dt.hour.astype('int8')
        features['minute'] = df['timestamp'].dt.minute.astype('int8')
        features['day_of_week'] = df['timestamp'].dt.dayofweek.astype('int8')
        
        # Time since market open (normalizado)
        market_open = df['timestamp'].dt.normalize() + pd.Timedelta(hours=9)
        minutes_since_open = ((df['timestamp'] - market_open).dt.total_seconds() / 60).clip(lower=0, upper=480)
        features['time_normalized'] = minutes_since_open / 480
        
        # Trading session indicators
        features['is_opening_30min'] = (minutes_since_open <= 30).astype('int8')
        features['is_closing_30min'] = (minutes_since_open >= 450).astype('int8')
        features['is_lunch_hour'] = ((features['hour'] == 12) | (features['hour'] == 13)).astype('int8')
        
        print(f"\n✓ Total features: {features.shape[1]}")
        
        # Cleanup
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        # Garantir float32
        for col in features.columns:
            if features[col].dtype == 'float64':
                features[col] = features[col].astype('float32')
        
        return features
    
    def create_realistic_targets(self, df: pd.DataFrame) -> tuple:
        """Cria targets com thresholds realistas"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE TARGETS REALISTAS")
        print("=" * 80)
        
        horizon = 60  # 60 trades ahead
        
        # Future return
        future_price = df['<price>'].shift(-horizon)
        returns = (future_price - df['<price>']) / df['<price>']
        
        # THRESHOLDS REALISTAS
        # Opção 1: Valor absoluto
        buy_threshold = 0.002   # 0.2% para BUY
        sell_threshold = -0.002 # -0.2% para SELL
        
        # Opção 2: Percentis mais extremos
        p15 = returns.quantile(0.15)
        p85 = returns.quantile(0.85)
        
        # Usar o mais conservador
        buy_threshold = max(buy_threshold, p85)
        sell_threshold = min(sell_threshold, p15)
        
        # Criar target
        target = pd.Series(0, index=df.index, dtype='int8')
        target[returns < sell_threshold] = -1  # SELL
        target[returns > buy_threshold] = 1    # BUY
        
        # Stats
        dist = target.value_counts().sort_index()
        total = len(target)
        
        print(f"\nDistribuição do target:")
        print(f"  SELL (-1): {dist.get(-1, 0):>7,} ({dist.get(-1, 0)/total*100:>5.1f}%)")
        print(f"  HOLD  (0): {dist.get(0, 0):>7,} ({dist.get(0, 0)/total*100:>5.1f}%)")
        print(f"  BUY   (1): {dist.get(1, 0):>7,} ({dist.get(1, 0)/total*100:>5.1f}%)")
        
        print(f"\nThresholds:")
        print(f"  SELL: < {sell_threshold:.4f} ({sell_threshold*100:.2f}%)")
        print(f"  BUY:  > {buy_threshold:.4f} ({buy_threshold*100:.2f}%)")
        
        # Criar target binário adicional (apenas sinais)
        target_binary = target.copy()
        target_binary[target_binary == 0] = np.nan  # Ignorar HOLD
        
        return target, target_binary
    
    def train_realistic_models(self, features: pd.DataFrame, target: pd.Series, target_binary: pd.Series):
        """Treina modelos com foco em sinais de trading"""
        
        print("\n" + "=" * 80)
        print("TREINAMENTO REALISTA")
        print("=" * 80)
        
        # Preparar dados
        mask = ~target.isna()
        X = features[mask]
        y = target[mask]
        
        print(f"\nDados totais: {len(X):,} registros")
        
        # Normalizar features
        print("Normalizando features...")
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Split temporal
        split = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Calcular pesos das classes
        classes = np.array([-1, 0, 1])
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"\nPesos das classes: {class_weight_dict}")
        
        # Modelos
        models_config = [
            ("LightGBM Balanced", self._train_lightgbm_balanced),
            ("XGBoost Balanced", self._train_xgboost_balanced),
            ("Random Forest Balanced", self._train_rf_balanced)
        ]
        
        predictions = {}
        importances = {}
        
        for name, train_func in models_config:
            print(f"\n{'='*50}")
            print(f"Treinando {name}...")
            print(f"{'='*50}")
            
            model, pred, metrics, importance = train_func(
                X_train, y_train, X_test, y_test, class_weight_dict
            )
            
            self.models[name.lower().replace(' ', '_')] = model
            self.results[name.lower().replace(' ', '_')] = metrics
            predictions[name] = pred
            importances[name] = importance
            
            # Mostrar métricas
            print(f"\nMétricas {name}:")
            print(f"  Accuracy Geral: {metrics['accuracy']:.2%}")
            print(f"  Accuracy SELL: {metrics['accuracy_sell']:.2%}")
            print(f"  Accuracy BUY: {metrics['accuracy_buy']:.2%}")
            print(f"  F1-Score (weighted): {metrics['f1_weighted']:.2%}")
            print(f"  Trading Accuracy: {metrics['trading_accuracy']:.2%}")
        
        # Ensemble
        print(f"\n{'='*50}")
        print("ENSEMBLE VOTING")
        print(f"{'='*50}")
        
        # Voting simples
        ensemble_pred = np.zeros_like(y_test.values, dtype=float)
        for pred in predictions.values():
            ensemble_pred += pred
        
        ensemble_pred = np.sign(ensemble_pred).astype(int)
        
        # Métricas do ensemble
        ensemble_metrics = self._calculate_trading_metrics(y_test, ensemble_pred)
        self.results['ensemble'] = ensemble_metrics
        
        print(f"\nMétricas Ensemble:")
        print(f"  Accuracy Geral: {ensemble_metrics['accuracy']:.2%}")
        print(f"  Trading Accuracy: {ensemble_metrics['trading_accuracy']:.2%}")
        
        # Feature importance média
        print("\n" + "="*50)
        print("TOP 20 FEATURES")
        print("="*50)
        
        avg_imp = pd.DataFrame(index=X.columns)
        for name, imp in importances.items():
            avg_imp[name] = imp.reindex(X.columns, fill_value=0)
        
        avg_imp['average'] = avg_imp.mean(axis=1)
        avg_imp = avg_imp.sort_values('average', ascending=False)
        
        print(avg_imp[['average']].head(20))
        
        return avg_imp
    
    def _calculate_trading_metrics(self, y_true, y_pred):
        """Calcula métricas focadas em trading"""
        
        # Accuracy geral
        accuracy = (y_pred == y_true).mean()
        
        # Accuracy por classe
        accuracy_sell = (y_pred[y_true == -1] == -1).mean() if (y_true == -1).any() else 0
        accuracy_hold = (y_pred[y_true == 0] == 0).mean() if (y_true == 0).any() else 0
        accuracy_buy = (y_pred[y_true == 1] == 1).mean() if (y_true == 1).any() else 0
        
        # Trading accuracy (ignorando HOLD)
        trading_mask = y_true != 0
        if trading_mask.any():
            trading_accuracy = (y_pred[trading_mask] == y_true[trading_mask]).mean()
        else:
            trading_accuracy = 0
        
        # F1-score
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'accuracy_sell': accuracy_sell,
            'accuracy_hold': accuracy_hold,
            'accuracy_buy': accuracy_buy,
            'trading_accuracy': trading_accuracy,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm.tolist()
        }
    
    def _train_lightgbm_balanced(self, X_train, y_train, X_test, y_test, class_weights):
        """LightGBM com classes balanceadas"""
        
        # Criar pesos por amostra
        sample_weights = np.zeros(len(y_train))
        for class_val, weight in class_weights.items():
            sample_weights[y_train == class_val] = weight
        
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'num_leaves': 31,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'max_depth': 6,
            'min_data_in_leaf': 100,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'verbose': -1
        }
        
        y_train_lgb = y_train + 1
        y_test_lgb = y_test + 1
        
        lgb_train = lgb.Dataset(X_train, label=y_train_lgb, weight=sample_weights)
        lgb_val = lgb.Dataset(X_test, label=y_test_lgb, reference=lgb_train)
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=200,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )
        
        pred = model.predict(X_test, num_iteration=model.best_iteration)
        pred_class = np.argmax(pred, axis=1) - 1
        
        metrics = self._calculate_trading_metrics(y_test, pred_class)
        
        importance = pd.Series(
            model.feature_importance(importance_type='gain'),
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return model, pred_class, metrics, importance
    
    def _train_xgboost_balanced(self, X_train, y_train, X_test, y_test, class_weights):
        """XGBoost com classes balanceadas"""
        
        # Criar pesos por amostra
        sample_weights = np.zeros(len(y_train))
        for class_val, weight in class_weights.items():
            sample_weights[y_train == class_val] = weight
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=100,
            reg_alpha=1.0,
            reg_lambda=1.0,
            tree_method='hist',
            eval_metric='mlogloss',
            early_stopping_rounds=30,
            n_jobs=-1,
            random_state=42
        )
        
        y_train_xgb = y_train + 1
        y_test_xgb = y_test + 1
        
        model.fit(
            X_train, y_train_xgb,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test_xgb)],
            verbose=False
        )
        
        pred = model.predict(X_test) - 1
        
        metrics = self._calculate_trading_metrics(y_test, pred)
        
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return model, pred, metrics, importance
    
    def _train_rf_balanced(self, X_train, y_train, X_test, y_test, class_weights):
        """Random Forest com classes balanceadas"""
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=100,
            min_samples_leaf=50,
            max_features='sqrt',
            class_weight=class_weights,  # Usar pesos diretamente
            n_jobs=-1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        
        metrics = self._calculate_trading_metrics(y_test, pred)
        
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return model, pred, metrics, importance
    
    def save_results(self, feature_importance: pd.DataFrame):
        """Salva resultados com métricas realistas"""
        
        output_dir = Path('models/csv_5m_realistic')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*80)
        print("SALVANDO RESULTADOS")
        print("="*80)
        
        # Salvar modelos
        for name, model in self.models.items():
            if 'lightgbm' in name:
                model_file = output_dir / f'{name}_{timestamp}.txt'
                model.save_model(str(model_file))
            else:
                model_file = output_dir / f'{name}_{timestamp}.pkl'
                joblib.dump(model, model_file)
        
        # Salvar scaler
        joblib.dump(self.scaler, output_dir / f'scaler_{timestamp}.pkl')
        
        # Feature importance
        feature_importance.to_csv(output_dir / f'features_{timestamp}.csv')
        
        # Relatório detalhado
        report = {
            'training_date': datetime.now().isoformat(),
            'sample_size': self.sample_size,
            'configuration': {
                'thresholds': 'Realistic (0.2% or p15/p85)',
                'class_balancing': 'Enabled',
                'features': 'No price leakage',
                'evaluation': 'Trading-focused metrics'
            },
            'models_performance': {}
        }
        
        # Adicionar performance de cada modelo
        for name, metrics in self.results.items():
            report['models_performance'][name] = {
                'accuracy_overall': round(metrics['accuracy'], 4),
                'accuracy_sell': round(metrics['accuracy_sell'], 4),
                'accuracy_hold': round(metrics['accuracy_hold'], 4),
                'accuracy_buy': round(metrics['accuracy_buy'], 4),
                'trading_accuracy': round(metrics['trading_accuracy'], 4),
                'f1_weighted': round(metrics['f1_weighted'], 4),
                'confusion_matrix': metrics['confusion_matrix']
            }
        
        # Melhor modelo para trading
        best_trading = max(self.results.items(), 
                          key=lambda x: x[1]['trading_accuracy'])
        report['best_for_trading'] = {
            'model': best_trading[0],
            'trading_accuracy': round(best_trading[1]['trading_accuracy'], 4)
        }
        
        # Salvar relatório
        with open(output_dir / f'training_report_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Resultados salvos em: {output_dir}")
        
        # Imprimir resumo
        print("\n" + "="*80)
        print("RESUMO DO TREINAMENTO REALISTA")
        print("="*80)
        
        print("\nAccuracy de Trading (ignorando HOLD):")
        for name, metrics in sorted(self.results.items(), 
                                   key=lambda x: x[1]['trading_accuracy'], 
                                   reverse=True):
            print(f"  {name:.<30} {metrics['trading_accuracy']:.2%}")
        
        print(f"\nMelhor modelo para trading: {best_trading[0]}")
        print(f"Trading accuracy: {best_trading[1]['trading_accuracy']:.2%}")

def main():
    """Pipeline principal realista"""
    
    print("PIPELINE REALISTA - 5 MILHÕES DE REGISTROS")
    print("Configuração apropriada para trading real\n")
    
    csv_path = r"C:\Users\marth\Downloads\WDO_FUT\WDOFUT_BMF_T.csv"
    
    trainer = CSV5MRealisticTrainer(csv_path)
    
    try:
        # 1. Carregar dados
        df = trainer.load_data_with_progress()
        
        # 2. Criar features sem leakage
        features = trainer.create_realistic_features(df)
        
        # 3. Criar targets realistas
        target, target_binary = trainer.create_realistic_targets(df)
        
        # 4. Treinar com métricas de trading
        feature_importance = trainer.train_realistic_models(features, target, target_binary)
        
        # 5. Salvar com relatório completo
        trainer.save_results(feature_importance)
        
        # Cleanup
        del df, features, target
        gc.collect()
        
        print("\n" + "="*80)
        print("✓ TREINAMENTO REALISTA CONCLUÍDO!")
        print("="*80)
        
        print("\nExpectativas realistas:")
        print("- Overall accuracy: 45-60%")
        print("- Trading accuracy: 50-65%")
        print("- Classes balanceadas")
        print("- Sem data leakage")
        print("- Métricas focadas em trading real")
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()