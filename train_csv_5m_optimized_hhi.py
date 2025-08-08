"""
Pipeline Completo para 5 Milhões de Registros
Versão com HHI otimizado (janelas menores)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import joblib
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CSV5MOptimizedTrainer:
    """Treina modelos com 5M registros e HHI otimizado"""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.sample_size = 5_000_000
        self.models = {}
        self.results = {}
        
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
        
        # Carregar com progresso
        chunk_size = 100_000
        chunks = []
        total_chunks = self.sample_size // chunk_size
        
        with tqdm(total=total_chunks, desc="Carregando chunks") as pbar:
            for i, chunk in enumerate(pd.read_csv(self.csv_path, 
                                                chunksize=chunk_size,
                                                dtype=dtypes)):
                chunks.append(chunk)
                pbar.update(1)
                
                if len(chunks) * chunk_size >= self.sample_size:
                    break
        
        # Combinar chunks
        print("\nCombinando chunks...")
        df = pd.concat(chunks, ignore_index=True)
        df = df.head(self.sample_size)
        
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
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        
        print(f"\n✓ Tempo total: {load_time:.1f}s")
        print(f"✓ Memória: {memory_mb:.1f} MB")
        print(f"✓ Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
        
        # Análise de cobertura
        df['date'] = df['timestamp'].dt.date
        unique_dates = df['date'].nunique()
        print(f"✓ Dias únicos: {unique_dates}")
        print(f"✓ Registros por dia: ~{len(df) // unique_dates:,}")
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features avançadas com HHI otimizado"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE FEATURES AVANÇADAS")
        print("=" * 80)
        
        features = pd.DataFrame(index=df.index)
        
        # Lista de features para criar
        feature_tasks = [
            ("Preço e Retornos", self._create_price_features),
            ("Volume e Liquidez", self._create_volume_features),
            ("Trade Flow", self._create_flow_features),
            ("Agent Behavior (Otimizado)", self._create_agent_features_optimized),
            ("Temporal", self._create_temporal_features),
            ("Padrões", self._create_pattern_features)
        ]
        
        # Criar features com progresso
        for task_name, task_func in tqdm(feature_tasks, desc="Grupos de features"):
            print(f"\n→ {task_name}...")
            task_features = task_func(df)
            features = pd.concat([features, task_features], axis=1)
            print(f"  ✓ {task_features.shape[1]} features criadas")
        
        print(f"\n✓ Total de features: {features.shape[1]}")
        
        # Cleanup
        print("\nLimpeza e otimização...")
        features = features.ffill().fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        # Converter para float32
        for col in tqdm(features.columns, desc="Otimizando tipos"):
            if features[col].dtype == 'float64':
                features[col] = features[col].astype('float32')
        
        return features
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de preço e retornos"""
        features = pd.DataFrame(index=df.index)
        
        # Preço base
        features['price'] = df['<price>'].astype('float32')
        features['log_price'] = np.log(features['price'])
        
        # Retornos múltiplos
        with tqdm([1, 2, 5, 10, 20, 30, 50, 100], desc="  Retornos") as pbar:
            for period in pbar:
                features[f'returns_{period}'] = features['price'].pct_change(period)
        
        # Volatilidade
        with tqdm([10, 20, 30, 50], desc="  Volatilidade") as pbar:
            for window in pbar:
                features[f'volatility_{window}'] = features['returns_1'].rolling(window).std()
        
        # Momentum
        features['momentum_5_20'] = features['price'].rolling(5).mean() / features['price'].rolling(20).mean() - 1
        features['momentum_20_50'] = features['price'].rolling(20).mean() / features['price'].rolling(50).mean() - 1
        
        # RSI simplificado
        gains = features['returns_1'].clip(lower=0)
        losses = -features['returns_1'].clip(upper=0)
        avg_gains = gains.rolling(14).mean()
        avg_losses = losses.rolling(14).mean()
        rs = avg_gains / avg_losses
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        return features
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de volume e liquidez"""
        features = pd.DataFrame(index=df.index)
        
        # Volume base
        features['qty'] = df['<qty>'].astype('float32')
        features['log_qty'] = np.log1p(features['qty'])
        features['volume_usd'] = df['<vol>'].astype('float32')
        
        # Volume profiles
        with tqdm([20, 50, 100, 200], desc="  Volume MA") as pbar:
            for window in pbar:
                features[f'volume_ma_{window}'] = features['qty'].rolling(window).mean()
                features[f'volume_ratio_{window}'] = features['qty'] / features[f'volume_ma_{window}']
        
        # VWAP múltiplo
        with tqdm([100, 500], desc="  VWAP") as pbar:
            for window in pbar:
                cumvol = df['<vol>'].rolling(window).sum()
                cumqty = df['<qty>'].rolling(window).sum()
                vwap = cumvol / cumqty
                features[f'vwap_{window}'] = vwap.astype('float32')
                features[f'price_to_vwap_{window}'] = (df['<price>'] / vwap - 1) * 100
        
        # Volume concentration
        features['volume_concentration'] = features['qty'].rolling(100).std() / features['qty'].rolling(100).mean()
        
        return features
    
    def _create_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de trade flow"""
        features = pd.DataFrame(index=df.index)
        
        # Trade types
        features['is_buyer_aggressor'] = (df['<trade_type>'] == 'AggressorBuyer').astype('int8')
        features['is_seller_aggressor'] = (df['<trade_type>'] == 'AggressorSeller').astype('int8')
        features['is_auction'] = (df['<trade_type>'] == 'Auction').astype('int8')
        features['is_rlp'] = (df['<trade_type>'] == 'RLP').astype('int8')
        
        # Flow imbalance
        with tqdm([20, 50, 100, 200], desc="  Flow imbalance") as pbar:
            for window in pbar:
                buyer_sum = features['is_buyer_aggressor'].rolling(window).sum()
                seller_sum = features['is_seller_aggressor'].rolling(window).sum()
                total = buyer_sum + seller_sum
                features[f'flow_imbalance_{window}'] = (buyer_sum - seller_sum) / total.clip(lower=1)
                features[f'buyer_ratio_{window}'] = buyer_sum / total.clip(lower=1)
        
        # Trade intensity
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        features['trade_intensity'] = 1 / df['time_diff'].rolling(100).median().clip(lower=0.001)
        features['trade_intensity_std'] = features['trade_intensity'].rolling(100).std()
        
        return features
    
    def _create_agent_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de agent behavior com HHI otimizado"""
        features = pd.DataFrame(index=df.index)
        
        # Top agents
        print("    Identificando top agents...")
        top_buyers = df['<buy_agent>'].value_counts().head(20).index
        top_sellers = df['<sell_agent>'].value_counts().head(20).index
        
        # Binary features para top 10
        for i in tqdm(range(10), desc="  Top buyers"):
            if i < len(top_buyers):
                features[f'top_buyer_{i}'] = (df['<buy_agent>'] == top_buyers[i]).astype('int8')
        
        for i in tqdm(range(10), desc="  Top sellers"):
            if i < len(top_sellers):
                features[f'top_seller_{i}'] = (df['<sell_agent>'] == top_sellers[i]).astype('int8')
        
        # Activity scores
        features['top_buyers_active'] = features[[f'top_buyer_{i}' for i in range(10) if f'top_buyer_{i}' in features]].sum(axis=1)
        features['top_sellers_active'] = features[[f'top_seller_{i}' for i in range(10) if f'top_seller_{i}' in features]].sum(axis=1)
        
        # HHI OTIMIZADO - janelas menores e amostragem mais esparsa
        print("    Calculando HHI otimizado...")
        
        # Usar apenas janela de 50 e 100 (ao invés de 100 e 500)
        hhi_windows = [50, 100]
        
        for window in tqdm(hhi_windows, desc="  HHI (otimizado)"):
            # Amostragem muito mais esparsa - a cada 1000 pontos
            sample_rate = 1000
            sample_indices = list(range(window, len(df), sample_rate))
            
            buyer_hhi = []
            seller_hhi = []
            
            # Limitar número de amostras
            max_samples = min(len(sample_indices), 1000)
            sample_indices = sample_indices[:max_samples]
            
            for i in tqdm(sample_indices, desc=f"    Calculando HHI-{window}", leave=False):
                # Calcular HHI na janela
                window_buyers = df['<buy_agent>'].iloc[i-window:i]
                window_sellers = df['<sell_agent>'].iloc[i-window:i]
                
                # Usar value_counts normalizado diretamente
                buyer_shares = window_buyers.value_counts(normalize=True)
                seller_shares = window_sellers.value_counts(normalize=True)
                
                # HHI = soma dos quadrados das participações
                buyer_hhi.append((buyer_shares ** 2).sum())
                seller_hhi.append((seller_shares ** 2).sum())
            
            # Criar série e interpolar
            buyer_series = pd.Series(buyer_hhi, index=sample_indices)
            seller_series = pd.Series(seller_hhi, index=sample_indices)
            
            # Interpolar com método mais rápido
            features[f'buyer_hhi_{window}'] = buyer_series.reindex(range(len(df))).interpolate(method='linear', limit_direction='both').fillna(0)
            features[f'seller_hhi_{window}'] = seller_series.reindex(range(len(df))).interpolate(method='linear', limit_direction='both').fillna(0)
        
        # Agent switching (mais simples)
        features['buyer_changed'] = (df['<buy_agent>'] != df['<buy_agent>'].shift(1)).astype('int8')
        features['seller_changed'] = (df['<sell_agent>'] != df['<sell_agent>'].shift(1)).astype('int8')
        
        # Agent repetition count (simplificado)
        features['buyer_repeat_count'] = features['buyer_changed'].groupby(
            features['buyer_changed'].cumsum()
        ).cumcount()
        
        features['seller_repeat_count'] = features['seller_changed'].groupby(
            features['seller_changed'].cumsum()
        ).cumcount()
        
        return features
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features temporais"""
        features = pd.DataFrame(index=df.index)
        
        # Básicas
        features['hour'] = df['timestamp'].dt.hour.astype('int8')
        features['minute'] = df['timestamp'].dt.minute.astype('int8')
        features['day_of_week'] = df['timestamp'].dt.dayofweek.astype('int8')
        
        # Períodos
        features['is_opening'] = ((features['hour'] == 9) & (features['minute'] < 30)).astype('int8')
        features['is_morning'] = ((features['hour'] >= 9) & (features['hour'] < 12)).astype('int8')
        features['is_lunch'] = ((features['hour'] >= 12) & (features['hour'] < 14)).astype('int8')
        features['is_afternoon'] = ((features['hour'] >= 14) & (features['hour'] < 16)).astype('int8')
        features['is_closing'] = ((features['hour'] >= 16) & (features['hour'] < 17)).astype('int8')
        
        # Time since open
        market_open = df['timestamp'].dt.normalize() + pd.Timedelta(hours=9)
        features['minutes_since_open'] = ((df['timestamp'] - market_open).dt.total_seconds() / 60).clip(lower=0)
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        
        return features
    
    def _create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de padrões"""
        features = pd.DataFrame(index=df.index)
        
        # Large trades
        with tqdm([75, 90, 95], desc="  Large trades") as pbar:
            for percentile in pbar:
                threshold = df['<qty>'].quantile(percentile/100)
                features[f'large_trade_p{percentile}'] = (df['<qty>'] > threshold).astype('int8')
                features[f'large_trade_p{percentile}_ratio_100'] = features[f'large_trade_p{percentile}'].rolling(100).mean()
        
        # Price levels
        features['price_percentile_500'] = df['<price>'].rolling(500).rank(pct=True)
        features['volume_percentile_500'] = df['<qty>'].rolling(500).rank(pct=True)
        
        # Trade size stats
        features['avg_trade_size'] = df['<qty>'].rolling(100).mean()
        features['trade_size_std'] = df['<qty>'].rolling(100).std()
        
        return features
    
    def create_multi_targets(self, df: pd.DataFrame) -> dict:
        """Cria múltiplos targets com progresso"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE TARGETS MULTI-HORIZONTE")
        print("=" * 80)
        
        targets = {}
        horizons = [30, 60, 120]  # Removendo 300 para acelerar
        
        for horizon in tqdm(horizons, desc="Horizontes"):
            # Future return
            future_price = df['<price>'].shift(-horizon)
            returns = (future_price - df['<price>']) / df['<price>']
            
            # Percentis para balance
            p30 = returns.quantile(0.30)
            p70 = returns.quantile(0.70)
            
            target = pd.Series(0, index=df.index, dtype='int8')
            target[returns < p30] = -1
            target[returns > p70] = 1
            
            targets[f'target_{horizon}'] = target
            
            # Stats
            dist = target.value_counts()
            print(f"\n  Target {horizon}: {dict(dist)}")
            print(f"  Thresholds: [{p30:.5f}, {p70:.5f}]")
        
        return targets
    
    def train_ensemble_models(self, features: pd.DataFrame, targets: dict):
        """Treina ensemble de modelos com progresso"""
        
        print("\n" + "=" * 80)
        print("TREINAMENTO DE ENSEMBLE")
        print("=" * 80)
        
        # Target principal
        target = targets['target_60']
        
        # Dados válidos
        mask = ~target.isna()
        X = features[mask]
        y = target[mask]
        
        print(f"\nDados: {len(X):,} registros")
        
        # Split 80/20
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Lista de modelos - removendo Gradient Boosting para acelerar
        model_configs = [
            ("LightGBM", self._train_lightgbm),
            ("XGBoost", self._train_xgboost),
            ("Random Forest", self._train_random_forest)
        ]
        
        # Treinar modelos
        predictions = {}
        feature_importances = {}
        
        for model_name, train_func in model_configs:
            print(f"\n{'='*40}")
            print(f"Treinando {model_name}...")
            print(f"{'='*40}")
            
            model, pred, accuracy, importance = train_func(X_train, y_train, X_test, y_test)
            
            self.models[model_name.lower().replace(' ', '_')] = model
            self.results[model_name.lower().replace(' ', '_')] = accuracy
            predictions[model_name] = pred
            feature_importances[model_name] = importance
            
            print(f"✓ {model_name} Accuracy: {accuracy:.2%}")
        
        # Ensemble voting
        print(f"\n{'='*40}")
        print("ENSEMBLE VOTING")
        print(f"{'='*40}")
        
        # Weighted voting
        weights = {'LightGBM': 0.4, 'XGBoost': 0.4, 'Random Forest': 0.2}
        
        ensemble_pred = np.zeros_like(y_test.values, dtype=float)
        for model_name, pred in predictions.items():
            ensemble_pred += weights.get(model_name, 0.33) * pred
        
        ensemble_pred = np.round(ensemble_pred).astype(int)
        ensemble_accuracy = (ensemble_pred == y_test).mean()
        
        print(f"\n✓ Ensemble Accuracy: {ensemble_accuracy:.2%}")
        self.results['ensemble'] = ensemble_accuracy
        
        # Feature importance média
        print("\n" + "="*40)
        print("TOP 20 FEATURES (MÉDIA)")
        print("="*40)
        
        # Combinar importâncias
        all_features = set()
        for imp in feature_importances.values():
            all_features.update(imp.index)
        
        avg_importance = pd.DataFrame(index=list(all_features))
        for model_name, imp in feature_importances.items():
            avg_importance[model_name] = imp.reindex(avg_importance.index, fill_value=0)
        
        avg_importance['average'] = avg_importance.mean(axis=1)
        avg_importance = avg_importance.sort_values('average', ascending=False)
        
        print(avg_importance[['average']].head(20))
        
        return avg_importance
    
    def _train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Treina LightGBM"""
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 8,
            'min_data_in_leaf': 200,
            'verbose': -1
        }
        
        # Ajustar labels
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
        
        # Feature importance
        importance = pd.Series(
            model.feature_importance(importance_type='gain'),
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return model, pred_class, accuracy, importance
    
    def _train_xgboost(self, X_train, y_train, X_test, y_test):
        """Treina XGBoost"""
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=200,
            tree_method='hist',
            eval_metric='mlogloss',
            early_stopping_rounds=20,
            n_jobs=-1,
            random_state=42
        )
        
        # Ajustar labels
        y_train_xgb = y_train + 1
        y_test_xgb = y_test + 1
        
        model.fit(
            X_train, y_train_xgb,
            eval_set=[(X_test, y_test_xgb)],
            verbose=False
        )
        
        pred = model.predict(X_test) - 1
        accuracy = (pred == y_test).mean()
        
        # Feature importance
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return model, pred, accuracy, importance
    
    def _train_random_forest(self, X_train, y_train, X_test, y_test):
        """Treina Random Forest"""
        model = RandomForestClassifier(
            n_estimators=50,  # Reduzido para velocidade
            max_depth=10,
            min_samples_split=200,
            min_samples_leaf=100,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        
        print("  Treinando Random Forest...")
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        accuracy = (pred == y_test).mean()
        
        # Feature importance
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return model, pred, accuracy, importance
    
    def save_results(self, feature_importance: pd.DataFrame):
        """Salva todos os resultados"""
        
        output_dir = Path('models/csv_5m_optimized')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*80)
        print("SALVANDO RESULTADOS")
        print("="*80)
        
        # Salvar modelos
        for name, model in tqdm(self.models.items(), desc="Salvando modelos"):
            if 'lightgbm' in name:
                model_file = output_dir / f'{name}_5m_{timestamp}.txt'
                model.save_model(str(model_file))
            else:
                model_file = output_dir / f'{name}_5m_{timestamp}.pkl'
                joblib.dump(model, model_file)
        
        # Feature importance
        importance_file = output_dir / f'feature_importance_5m_{timestamp}.csv'
        feature_importance.to_csv(importance_file)
        
        # Metadados completos
        metadata = {
            'training_date': datetime.now().isoformat(),
            'sample_size': self.sample_size,
            'results': {name: float(acc) for name, acc in self.results.items()},
            'best_model': max(self.results.items(), key=lambda x: x[1])[0],
            'best_accuracy': float(max(self.results.values())),
            'progression': {
                '300k': 0.46,
                '1M': 0.5143,
                '5M': float(max(self.results.values()))
            },
            'improvement': {
                'from_300k': f"+{(max(self.results.values()) - 0.46)*100:.1f}%",
                'from_1M': f"+{(max(self.results.values()) - 0.5143)*100:.1f}%"
            },
            'top_30_features': feature_importance.head(30).to_dict()
        }
        
        metadata_file = output_dir / f'metadata_5m_{timestamp}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Todos os arquivos salvos em: {output_dir}")
        
        # Criar relatório resumido
        report_file = output_dir / f'training_report_5m_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO DE TREINAMENTO - 5 MILHÕES DE REGISTROS\n")
            f.write("HHI OTIMIZADO (janelas 50 e 100)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Registros utilizados: {self.sample_size:,}\n\n")
            
            f.write("RESULTADOS POR MODELO:\n")
            f.write("-"*40 + "\n")
            for name, acc in sorted(self.results.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{name:.<30} {acc:.2%}\n")
            
            f.write("\n\nPROGRESSÃO DE ACCURACY:\n")
            f.write("-"*40 + "\n")
            f.write(f"300k registros:............... 46.00%\n")
            f.write(f"1M registros:................. 51.43% (+5.43%)\n")
            f.write(f"5M registros:................. {max(self.results.values()):.2%} (+{(max(self.results.values()) - 0.5143)*100:.2f}%)\n")
            
            f.write("\n\nTOP 15 FEATURES:\n")
            f.write("-"*40 + "\n")
            for idx, row in feature_importance.head(15).iterrows():
                f.write(f"{idx:.<30} {row['average']:.4f}\n")
        
        print(f"✓ Relatório salvo: {report_file.name}")

def main():
    """Pipeline principal com HHI otimizado"""
    
    print("PIPELINE OTIMIZADO - 5 MILHÕES DE REGISTROS")
    print("Com HHI em janelas menores (50 e 100)\n")
    
    csv_path = r"C:\Users\marth\Downloads\WDO_FUT\WDOFUT_BMF_T.csv"
    
    # Criar trainer
    trainer = CSV5MOptimizedTrainer(csv_path)
    
    try:
        # 1. Carregar dados
        df = trainer.load_data_with_progress()
        
        # 2. Criar features
        features = trainer.create_advanced_features(df)
        
        # 3. Criar targets
        targets = trainer.create_multi_targets(df)
        
        # 4. Treinar ensemble
        feature_importance = trainer.train_ensemble_models(features, targets)
        
        # 5. Salvar resultados
        trainer.save_results(feature_importance)
        
        # Limpar memória
        del df, features, targets
        gc.collect()
        
        print("\n" + "="*80)
        print("✓ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*80)
        
        print("\nRESUMO FINAL:")
        print("-"*40)
        for name, accuracy in sorted(trainer.results.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:.<25} {accuracy:.2%}")
        
        print(f"\n🎯 MELHOR RESULTADO: {max(trainer.results.values()):.2%}")
        
        if max(trainer.results.values()) > 0.55:
            print("\n🎉 META ALCANÇADA! Accuracy > 55%")
        
        print("\nPRÓXIMOS PASSOS:")
        print("1. Executar backtesting com os modelos treinados")
        print("2. Integrar com Book Collector para microestrutura")
        print("3. Implementar paper trading para validação")
        print("4. Deploy em produção com monitoramento")
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n" + "="*80)
        print("Processo finalizado.")

if __name__ == "__main__":
    main()