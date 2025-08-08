"""
Pipeline Corrigido e Otimizado para 5 Milhões de Registros
Versão com cálculos vetorizados para velocidade
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CSV5MFastCorrectedTrainer:
    """Treina modelos com correções e otimizações de velocidade"""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.sample_size = 5_000_000
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_data_with_progress(self):
        """Carrega 5M registros com barra de progresso"""
        
        print("=" * 80)
        print("CARREGAMENTO DE DADOS - 5 MILHÕES DE REGISTROS")
        print("=" * 80)
        
        # Tipos otimizados
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
        
        # Carregar direto (mais rápido)
        df = pd.read_csv(self.csv_path, 
                        nrows=self.sample_size,
                        dtype=dtypes)
        
        # Criar timestamp
        print("Processando timestamps...")
        df['timestamp'] = pd.to_datetime(
            df['<date>'].astype(str) + ' ' + df['<time>'].astype(str).str.zfill(6),
            format='%Y%m%d %H%M%S'
        )
        
        # Ordenar
        print("Ordenando dados...")
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        load_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✓ Tempo total: {load_time:.1f}s")
        print(f"✓ Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
        
        return df
    
    def create_fast_corrected_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features corrigidas com cálculos vetorizados"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE FEATURES OTIMIZADAS")
        print("=" * 80)
        
        features = pd.DataFrame(index=df.index)
        
        # 1. FEATURES DE PREÇO
        print("\n→ Features de Preço...")
        price = df['<price>'].values.astype('float32')
        features['price'] = price
        features['log_price'] = np.log(price)
        
        # Retornos
        for period in [1, 5, 20, 50]:
            features[f'returns_{period}'] = pd.Series(price).pct_change(period)
        
        # Volatilidade
        features['volatility_20'] = features['returns_1'].rolling(20).std()
        features['volatility_50'] = features['returns_1'].rolling(50).std()
        
        # Momentum normalizado
        features['momentum_fast'] = features['returns_5'] / features['volatility_20'].clip(lower=0.0001)
        features['momentum_slow'] = features['returns_20'] / features['volatility_50'].clip(lower=0.0001)
        
        # 2. FEATURES DE VOLUME
        print("→ Features de Volume...")
        qty = df['<qty>'].values.astype('float32')
        features['qty'] = qty
        features['log_qty'] = np.log1p(qty)
        
        # Volume Z-score
        for window in [50, 200]:
            vol_ma = pd.Series(qty).rolling(window).mean()
            vol_std = pd.Series(qty).rolling(window).std()
            features[f'volume_zscore_{window}'] = (qty - vol_ma) / vol_std.clip(lower=0.001)
        
        # VWAP
        cumvol = df['<vol>'].expanding().sum()
        cumqty = df['<qty>'].expanding().sum()
        vwap = (cumvol / cumqty).astype('float32')
        features['price_vwap_ratio'] = (price / vwap - 1) * 100
        
        # 3. AGENT BEHAVIOR SIMPLIFICADO
        print("→ Agent Behavior (Vetorizado)...")
        
        # Top agents
        top_buyers = df['<buy_agent>'].value_counts().head(10).index
        top_sellers = df['<sell_agent>'].value_counts().head(10).index
        
        # Agent presence binário
        for i, agent in enumerate(top_buyers[:5]):
            features[f'mega_buyer_{i}'] = (df['<buy_agent>'] == agent).astype('int8')
            
        for i, agent in enumerate(top_sellers[:5]):
            features[f'mega_seller_{i}'] = (df['<sell_agent>'] == agent).astype('int8')
        
        # Agent activity scores (vetorizado)
        features['top_buyers_active'] = features[[f'mega_buyer_{i}' for i in range(5)]].sum(axis=1)
        features['top_sellers_active'] = features[[f'mega_seller_{i}' for i in range(5)]].sum(axis=1)
        
        # Agent switching (vetorizado)
        features['buyer_switch'] = (df['<buy_agent>'] != df['<buy_agent>'].shift(1)).astype('int8')
        features['seller_switch'] = (df['<sell_agent>'] != df['<sell_agent>'].shift(1)).astype('int8')
        
        features['buyer_switch_rate'] = features['buyer_switch'].rolling(100).mean()
        features['seller_switch_rate'] = features['seller_switch'].rolling(100).mean()
        
        # Agent concentration simplificado (usando top agents como proxy)
        features['buyer_concentration_proxy'] = features[[f'mega_buyer_{i}' for i in range(5)]].max(axis=1).rolling(100).mean()
        features['seller_concentration_proxy'] = features[[f'mega_seller_{i}' for i in range(5)]].max(axis=1).rolling(100).mean()
        
        # 4. TRADE FLOW
        print("→ Trade Flow...")
        
        # Trade types
        features['is_buyer_aggressor'] = (df['<trade_type>'] == 'AggressorBuyer').astype('int8')
        features['is_seller_aggressor'] = (df['<trade_type>'] == 'AggressorSeller').astype('int8')
        
        # Flow imbalance (EWM mais rápido)
        for window in [20, 50, 100]:
            buyer_flow = features['is_buyer_aggressor'].ewm(span=window, adjust=False).mean()
            seller_flow = features['is_seller_aggressor'].ewm(span=window, adjust=False).mean()
            
            features[f'flow_imbalance_{window}'] = buyer_flow - seller_flow
            features[f'flow_intensity_{window}'] = buyer_flow + seller_flow
        
        # Trade speed
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        features['trade_speed'] = 1 / df['time_diff'].rolling(50).median().clip(lower=0.001)
        
        # 5. MICROSTRUCTURE PROXIES
        print("→ Microstructure...")
        
        # Price impact
        features['price_impact'] = features['returns_1'].abs() / features['log_qty'].clip(lower=0.1)
        features['price_impact_ma'] = features['price_impact'].rolling(100).mean()
        
        # Signed volume
        signed_volume = features['qty'] * (features['is_buyer_aggressor'] - features['is_seller_aggressor'])
        features['signed_volume_sum'] = signed_volume.rolling(50).sum()
        
        # 6. INTERAÇÕES SIMPLIFICADAS
        print("→ Feature Interactions...")
        
        # Volume-volatility
        features['volume_volatility'] = features['volume_zscore_50'] * features['volatility_20']
        
        # Flow-momentum
        features['flow_momentum'] = features['flow_imbalance_50'] * features['momentum_fast']
        
        # Agent-volume
        features['agent_volume'] = features['top_buyers_active'] * features['volume_zscore_50']
        
        # 7. TEMPORAL
        print("→ Features Temporais...")
        
        features['hour'] = df['timestamp'].dt.hour.astype('int8')
        features['minute'] = df['timestamp'].dt.minute.astype('int8')
        
        # Time of day normalizado
        market_open = df['timestamp'].dt.normalize() + pd.Timedelta(hours=9)
        minutes_since_open = ((df['timestamp'] - market_open).dt.total_seconds() / 60).clip(lower=0, upper=480)
        features['time_normalized'] = minutes_since_open / 480
        
        # Períodos
        features['is_morning'] = (features['hour'] < 12).astype('int8')
        features['is_closing'] = (features['hour'] >= 16).astype('int8')
        
        # 8. LARGE TRADES
        print("→ Large Trades...")
        
        # Percentil dinâmico
        features['trade_size_percentile'] = pd.Series(qty).rolling(500).rank(pct=True)
        features['is_large_trade'] = (features['trade_size_percentile'] > 0.90).astype('int8')
        features['large_trade_ratio'] = features['is_large_trade'].rolling(100).mean()
        
        print(f"\n✓ Total features: {features.shape[1]}")
        
        # Cleanup
        features = features.ffill().fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        # Float32
        for col in features.columns:
            if features[col].dtype == 'float64':
                features[col] = features[col].astype('float32')
        
        return features
    
    def create_better_targets(self, df: pd.DataFrame) -> dict:
        """Cria targets com thresholds mais extremos"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE TARGETS")
        print("=" * 80)
        
        targets = {}
        
        # Horizonte principal
        horizon = 60
        
        # Future return
        future_price = df['<price>'].shift(-horizon)
        returns = (future_price - df['<price>']) / df['<price>']
        
        # Thresholds extremos (20/80)
        p20 = returns.quantile(0.20)
        p80 = returns.quantile(0.80)
        
        target = pd.Series(0, index=df.index, dtype='int8')
        target[returns < p20] = -1
        target[returns > p80] = 1
        
        targets['target_main'] = target
        
        # Stats
        dist = target.value_counts().sort_index()
        print(f"\nDistribuição:")
        for val in [-1, 0, 1]:
            count = dist.get(val, 0)
            print(f"  {val}: {count:,} ({count/len(target)*100:.1f}%)")
        
        print(f"\nThresholds: [{p20:.5f}, {p80:.5f}]")
        
        return targets
    
    def train_fast_ensemble(self, features: pd.DataFrame, targets: dict):
        """Treina ensemble otimizado"""
        
        print("\n" + "=" * 80)
        print("TREINAMENTO DE ENSEMBLE")
        print("=" * 80)
        
        # Target
        target = targets['target_main']
        
        # Dados válidos
        mask = ~target.isna()
        X = features[mask]
        y = target[mask]
        
        print(f"\nDados: {len(X):,} registros")
        
        # Normalização
        print("Normalizando features...")
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Split
        split = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Modelos
        models_to_train = [
            ("LightGBM", self._train_lightgbm),
            ("XGBoost", self._train_xgboost),
            ("Random Forest", self._train_rf)
        ]
        
        predictions = {}
        importances = {}
        
        for name, train_func in models_to_train:
            print(f"\n{'='*40}")
            print(f"Treinando {name}...")
            
            model, pred, acc, imp = train_func(X_train, y_train, X_test, y_test)
            
            self.models[name.lower().replace(' ', '_')] = model
            self.results[name.lower().replace(' ', '_')] = acc
            predictions[name] = pred
            importances[name] = imp
            
            print(f"Accuracy: {acc:.2%}")
        
        # Ensemble
        print(f"\n{'='*40}")
        print("ENSEMBLE VOTING")
        
        # Voting simples
        ensemble_pred = np.zeros_like(y_test.values, dtype=float)
        for pred in predictions.values():
            ensemble_pred += pred
        
        ensemble_pred = np.sign(ensemble_pred)
        ensemble_acc = (ensemble_pred == y_test).mean()
        
        print(f"Ensemble Accuracy: {ensemble_acc:.2%}")
        self.results['ensemble'] = ensemble_acc
        
        # Top features
        print("\n" + "="*40)
        print("TOP 15 FEATURES")
        
        # Média das importâncias
        all_features = list(X.columns)
        avg_imp = pd.DataFrame(index=all_features)
        
        for name, imp in importances.items():
            avg_imp[name] = imp.reindex(all_features, fill_value=0)
        
        avg_imp['average'] = avg_imp.mean(axis=1)
        avg_imp = avg_imp.sort_values('average', ascending=False)
        
        print(avg_imp[['average']].head(15))
        
        return avg_imp
    
    def _train_lightgbm(self, X_train, y_train, X_test, y_test):
        """LightGBM simplificado"""
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'max_depth': 6,
            'min_data_in_leaf': 300,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'verbose': -1
        }
        
        y_train_lgb = y_train + 1
        y_test_lgb = y_test + 1
        
        lgb_train = lgb.Dataset(X_train, label=y_train_lgb)
        lgb_val = lgb.Dataset(X_test, label=y_test_lgb, reference=lgb_train)
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=150,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        pred = model.predict(X_test, num_iteration=model.best_iteration)
        pred_class = np.argmax(pred, axis=1) - 1
        
        accuracy = (pred_class == y_test).mean()
        
        importance = pd.Series(
            model.feature_importance(importance_type='gain'),
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return model, pred_class, accuracy, importance
    
    def _train_xgboost(self, X_train, y_train, X_test, y_test):
        """XGBoost simplificado"""
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=300,
            reg_alpha=0.5,
            reg_lambda=0.5,
            tree_method='hist',
            eval_metric='mlogloss',
            early_stopping_rounds=20,
            n_jobs=-1,
            random_state=42
        )
        
        y_train_xgb = y_train + 1
        y_test_xgb = y_test + 1
        
        model.fit(
            X_train, y_train_xgb,
            eval_set=[(X_test, y_test_xgb)],
            verbose=False
        )
        
        pred = model.predict(X_test) - 1
        accuracy = (pred == y_test).mean()
        
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return model, pred, accuracy, importance
    
    def _train_rf(self, X_train, y_train, X_test, y_test):
        """Random Forest simplificado"""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=300,
            min_samples_leaf=150,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        accuracy = (pred == y_test).mean()
        
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return model, pred, accuracy, importance
    
    def save_results(self, feature_importance: pd.DataFrame):
        """Salva resultados"""
        
        output_dir = Path('models/csv_5m_fast_corrected')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*80)
        print("SALVANDO RESULTADOS")
        
        # Modelos
        for name, model in self.models.items():
            if 'lightgbm' in name:
                model_file = output_dir / f'{name}_{timestamp}.txt'
                model.save_model(str(model_file))
            else:
                model_file = output_dir / f'{name}_{timestamp}.pkl'
                joblib.dump(model, model_file)
        
        # Scaler
        joblib.dump(self.scaler, output_dir / f'scaler_{timestamp}.pkl')
        
        # Feature importance
        feature_importance.to_csv(output_dir / f'features_{timestamp}.csv')
        
        # Metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'sample_size': self.sample_size,
            'results': {name: float(acc) for name, acc in self.results.items()},
            'best_accuracy': float(max(self.results.values())),
            'corrections': [
                'No RSI (overfitting)',
                'Extreme thresholds (20/80)',
                'Normalized features',
                'Simplified agent features',
                'Fast vectorized calculations'
            ]
        }
        
        with open(output_dir / f'metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Salvos em: {output_dir}")

def main():
    """Pipeline principal"""
    
    print("PIPELINE OTIMIZADO E CORRIGIDO - 5M REGISTROS")
    print("Versão rápida com cálculos vetorizados\n")
    
    csv_path = r"C:\Users\marth\Downloads\WDO_FUT\WDOFUT_BMF_T.csv"
    
    trainer = CSV5MFastCorrectedTrainer(csv_path)
    
    try:
        # 1. Carregar
        df = trainer.load_data_with_progress()
        
        # 2. Features
        features = trainer.create_fast_corrected_features(df)
        
        # 3. Targets
        targets = trainer.create_better_targets(df)
        
        # 4. Treinar
        importance = trainer.train_fast_ensemble(features, targets)
        
        # 5. Salvar
        trainer.save_results(importance)
        
        # Cleanup
        del df, features, targets
        gc.collect()
        
        print("\n" + "="*80)
        print("✓ CONCLUÍDO!")
        print("="*80)
        
        print("\nRESULTADOS:")
        for name, acc in sorted(trainer.results.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:.<20} {acc:.2%}")
        
        best = max(trainer.results.values())
        print(f"\n🎯 MELHOR: {best:.2%}")
        
        if best > 0.55:
            print("\n🎉 META ALCANÇADA! > 55%")
        elif best > 0.50:
            print("\n📈 BOA MELHORIA! > 50%")
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()