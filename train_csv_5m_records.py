"""
Pipeline Otimizado para 5 Milhões de Registros
Com processamento em chunks e features avançadas
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CSV5MTrainer:
    """Treina modelos com 5M registros usando chunks"""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.sample_size = 5_000_000
        self.chunk_size = 500_000  # Processar em chunks de 500k
        self.models = {}
        self.results = {}
        
    def load_data_chunked(self):
        """Carrega 5M registros em chunks para economia de memória"""
        
        print("=" * 80)
        print("CARREGAMENTO EM CHUNKS - 5 MILHÕES DE REGISTROS")
        print("=" * 80)
        
        # Colunas necessárias
        usecols = ['<date>', '<time>', '<price>', '<qty>', '<vol>', 
                   '<buy_agent>', '<sell_agent>', '<trade_type>', '<trade_number>']
        
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
        
        print(f"\nCarregando {self.sample_size:,} registros em chunks de {self.chunk_size:,}...")
        start_time = datetime.now()
        
        # Ler em chunks
        chunks = []
        total_read = 0
        
        for chunk in pd.read_csv(self.csv_path, 
                                chunksize=self.chunk_size,
                                usecols=usecols,
                                dtype=dtypes):
            
            chunks.append(chunk)
            total_read += len(chunk)
            print(f"  Chunk {len(chunks)}: {len(chunk):,} registros (Total: {total_read:,})")
            
            if total_read >= self.sample_size:
                break
        
        # Combinar chunks
        print("\nCombinando chunks...")
        df = pd.concat(chunks, ignore_index=True)
        
        # Limitar ao tamanho desejado
        df = df.head(self.sample_size)
        
        # Criar timestamp
        print("Criando timestamps...")
        df['timestamp'] = pd.to_datetime(
            df['<date>'].astype(str) + ' ' + df['<time>'].astype(str).str.zfill(6),
            format='%Y%m%d %H%M%S'
        )
        
        # Remover colunas temporárias
        df = df.drop(['<date>', '<time>'], axis=1)
        
        # Ordenar por timestamp
        print("Ordenando dados...")
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        load_time = (datetime.now() - start_time).total_seconds()
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        
        print(f"\nTempo total: {load_time:.1f}s")
        print(f"Memória utilizada: {memory_mb:.1f} MB")
        print(f"Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
        print(f"Duração: {df['timestamp'].max() - df['timestamp'].min()}")
        
        # Análise de dias
        df['date'] = df['timestamp'].dt.date
        unique_dates = df['date'].nunique()
        print(f"Dias únicos: {unique_dates}")
        
        return df
    
    def create_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features abrangentes para 5M registros"""
        
        print("\n=== CRIAÇÃO DE FEATURES ABRANGENTES ===")
        features = pd.DataFrame(index=df.index)
        
        # 1. FEATURES DE PREÇO AVANÇADAS
        print("1. Features de preço avançadas...")
        features['price'] = df['<price>'].astype('float32')
        features['log_price'] = np.log(features['price'])
        
        # Retornos múltiplos
        returns_periods = [1, 2, 5, 10, 20, 30, 50, 100]
        for period in returns_periods:
            features[f'returns_{period}'] = features['price'].pct_change(period)
        
        # Volatilidade em múltiplas janelas
        vol_windows = [10, 20, 30, 50, 100]
        for window in vol_windows:
            features[f'volatility_{window}'] = features['returns_1'].rolling(window).std()
            features[f'volatility_ratio_{window}'] = (
                features[f'volatility_{window}'] / 
                features[f'volatility_{window}'].rolling(window*2).mean()
            )
        
        # Momentum indicators
        features['momentum_short'] = features['price'].rolling(10).mean() / features['price'].rolling(30).mean() - 1
        features['momentum_medium'] = features['price'].rolling(30).mean() / features['price'].rolling(100).mean() - 1
        features['momentum_long'] = features['price'].rolling(100).mean() / features['price'].rolling(300).mean() - 1
        
        # Price acceleration
        features['price_acceleration'] = features['returns_1'].diff()
        
        # 2. FEATURES DE VOLUME E LIQUIDEZ
        print("2. Features de volume e liquidez...")
        features['qty'] = df['<qty>'].astype('float32')
        features['log_qty'] = np.log1p(features['qty'])
        features['volume_usd'] = df['<vol>'].astype('float32')
        
        # Volume profiles
        for window in [20, 50, 100, 200]:
            features[f'volume_ma_{window}'] = features['qty'].rolling(window).mean()
            features[f'volume_ratio_{window}'] = features['qty'] / features[f'volume_ma_{window}']
            features[f'volume_zscore_{window}'] = (
                (features['qty'] - features[f'volume_ma_{window}']) / 
                features['qty'].rolling(window).std()
            )
        
        # VWAP e desvios
        cumvol = df['<vol>'].expanding().sum()
        cumqty = df['<qty>'].expanding().sum()
        features['vwap'] = (cumvol / cumqty).astype('float32')
        features['price_to_vwap'] = (features['price'] / features['vwap'] - 1) * 100
        
        # Rolling VWAP
        for window in [100, 500, 1000]:
            roll_cumvol = df['<vol>'].rolling(window).sum()
            roll_cumqty = df['<qty>'].rolling(window).sum()
            features[f'vwap_{window}'] = (roll_cumvol / roll_cumqty).astype('float32')
            features[f'price_to_vwap_{window}'] = (features['price'] / features[f'vwap_{window}'] - 1) * 100
        
        # 3. TRADE FLOW E MICROESTRUTURA
        print("3. Features de trade flow e microestrutura...")
        
        # Trade types
        features['is_buyer_aggressor'] = (df['<trade_type>'] == 'AggressorBuyer').astype('int8')
        features['is_seller_aggressor'] = (df['<trade_type>'] == 'AggressorSeller').astype('int8')
        features['is_auction'] = (df['<trade_type>'] == 'Auction').astype('int8')
        features['is_rlp'] = (df['<trade_type>'] == 'RLP').astype('int8')
        
        # Imbalance em múltiplas janelas
        for window in [20, 50, 100, 200, 500]:
            buyer_flow = features['is_buyer_aggressor'].rolling(window).sum()
            seller_flow = features['is_seller_aggressor'].rolling(window).sum()
            total_flow = buyer_flow + seller_flow
            
            features[f'flow_imbalance_{window}'] = (buyer_flow - seller_flow) / total_flow
            features[f'buyer_ratio_{window}'] = buyer_flow / total_flow
            features[f'auction_ratio_{window}'] = features['is_auction'].rolling(window).mean()
        
        # Trade intensity
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        features['trade_intensity'] = 1 / df['time_diff'].rolling(100).median()
        features['trade_intensity_ma'] = features['trade_intensity'].rolling(500).mean()
        
        # 4. AGENT BEHAVIOR DETALHADO
        print("4. Features de agent behavior...")
        
        # Top agents globais
        top_20_buyers = df['<buy_agent>'].value_counts().head(20).index
        top_20_sellers = df['<sell_agent>'].value_counts().head(20).index
        
        # Agent presence (binário para top 10)
        for i, agent in enumerate(top_20_buyers[:10]):
            features[f'buyer_{i}_present'] = (df['<buy_agent>'] == agent).astype('int8')
            
        for i, agent in enumerate(top_20_sellers[:10]):
            features[f'seller_{i}_present'] = (df['<sell_agent>'] == agent).astype('int8')
        
        # Agent activity scores
        features['top10_buyers_active'] = features[[f'buyer_{i}_present' for i in range(10)]].sum(axis=1)
        features['top10_sellers_active'] = features[[f'seller_{i}_present' for i in range(10)]].sum(axis=1)
        
        # Agent concentration (HHI)
        print("   Calculando concentração de agentes...")
        
        for window in [100, 500, 1000]:
            buyer_concentration = []
            seller_concentration = []
            
            # Usar amostragem para acelerar
            sample_indices = range(0, len(df), max(1, window // 100))
            
            for i in sample_indices:
                if i < window:
                    buyer_concentration.append(0)
                    seller_concentration.append(0)
                else:
                    # Calcular HHI na janela
                    window_buyers = df['<buy_agent>'].iloc[max(0, i-window):i]
                    window_sellers = df['<sell_agent>'].iloc[max(0, i-window):i]
                    
                    buyer_counts = window_buyers.value_counts()
                    seller_counts = window_sellers.value_counts()
                    
                    buyer_hhi = (buyer_counts / len(window_buyers)).pow(2).sum()
                    seller_hhi = (seller_counts / len(window_sellers)).pow(2).sum()
                    
                    buyer_concentration.append(buyer_hhi)
                    seller_concentration.append(seller_hhi)
            
            # Interpolar para todos os índices
            buyer_conc_series = pd.Series(buyer_concentration, index=sample_indices)
            seller_conc_series = pd.Series(seller_concentration, index=sample_indices)
            
            features[f'buyer_hhi_{window}'] = buyer_conc_series.reindex(range(len(df))).interpolate().fillna(0)
            features[f'seller_hhi_{window}'] = seller_conc_series.reindex(range(len(df))).interpolate().fillna(0)
        
        # 5. FEATURES TEMPORAIS E SAZONALIDADE
        print("5. Features temporais e sazonalidade...")
        
        features['hour'] = df['timestamp'].dt.hour.astype('int8')
        features['minute'] = df['timestamp'].dt.minute.astype('int8')
        features['day_of_week'] = df['timestamp'].dt.dayofweek.astype('int8')
        features['day_of_month'] = df['timestamp'].dt.day.astype('int8')
        
        # Períodos do dia
        features['is_opening'] = ((features['hour'] == 9) & (features['minute'] < 30)).astype('int8')
        features['is_morning'] = ((features['hour'] >= 9) & (features['hour'] < 12)).astype('int8')
        features['is_lunch'] = ((features['hour'] >= 12) & (features['hour'] < 14)).astype('int8')
        features['is_afternoon'] = ((features['hour'] >= 14) & (features['hour'] < 16)).astype('int8')
        features['is_closing'] = ((features['hour'] >= 16) & (features['hour'] < 17)).astype('int8')
        
        # Time since market open
        market_open = df['timestamp'].dt.normalize() + pd.Timedelta(hours=9)
        features['minutes_since_open'] = (df['timestamp'] - market_open).dt.total_seconds() / 60
        
        # 6. FEATURES DE PADRÕES E ANOMALIAS
        print("6. Features de padrões e anomalias...")
        
        # Large trades
        for percentile in [80, 90, 95, 99]:
            threshold = df['<qty>'].quantile(percentile/100)
            features[f'large_trade_p{percentile}'] = (df['<qty>'] > threshold).astype('int8')
            
            # Ratio em janelas
            for window in [100, 500]:
                features[f'large_trade_p{percentile}_ratio_{window}'] = (
                    features[f'large_trade_p{percentile}'].rolling(window).mean()
                )
        
        # Price/Volume ranks
        for window in [500, 1000]:
            features[f'price_rank_{window}'] = features['price'].rolling(window).rank(pct=True)
            features[f'volume_rank_{window}'] = features['qty'].rolling(window).rank(pct=True)
        
        print(f"\nTotal features criadas: {features.shape[1]}")
        
        # Cleanup
        features = features.ffill().fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        # Garantir float32
        for col in features.columns:
            if features[col].dtype == 'float64':
                features[col] = features[col].astype('float32')
        
        return features
    
    def create_multi_horizon_targets(self, df: pd.DataFrame) -> dict:
        """Cria targets para múltiplos horizontes"""
        
        print("\n=== CRIANDO TARGETS MULTI-HORIZONTE ===")
        targets = {}
        
        horizons = [30, 60, 120, 300]  # trades ahead
        
        for horizon in horizons:
            print(f"\nTarget horizonte {horizon} trades...")
            
            # Future return
            future_price = df['<price>'].shift(-horizon)
            returns = (future_price - df['<price>']) / df['<price>']
            
            # Thresholds balanceados
            upper = returns.quantile(0.67)
            lower = returns.quantile(0.33)
            
            target = pd.Series(0, index=df.index, dtype='int8')
            target[returns > upper] = 1
            target[returns < lower] = -1
            
            targets[f'target_{horizon}'] = target
            
            print(f"Distribuição: {dict(target.value_counts())}")
            print(f"Thresholds: [{lower:.5f}, {upper:.5f}]")
        
        return targets
    
    def train_advanced_ensemble(self, features: pd.DataFrame, targets: dict):
        """Treina ensemble avançado com múltiplos modelos"""
        
        print("\n" + "=" * 80)
        print("TREINAMENTO DE ENSEMBLE AVANÇADO")
        print("=" * 80)
        
        # Usar target principal
        main_target = targets['target_60']
        
        # Remover NaN
        mask = ~main_target.isna()
        X = features[mask]
        y = main_target[mask]
        
        print(f"\nDados disponíveis: {len(X):,} registros")
        
        # Split temporal 80/20
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # 1. LightGBM (principal)
        print("\n=== LightGBM ===")
        lgb_params = {
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
            'verbose': -1,
            'num_threads': -1
        }
        
        # Ajustar labels para LightGBM
        y_train_lgb = y_train + 1
        y_test_lgb = y_test + 1
        
        print("Treinando LightGBM...")
        lgb_train = lgb.Dataset(X_train, label=y_train_lgb)
        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=150,
            valid_sets=[lgb_train],
            callbacks=[lgb.log_evaluation(0)]
        )
        
        lgb_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
        lgb_pred_class = np.argmax(lgb_pred, axis=1) - 1
        
        lgb_accuracy = (lgb_pred_class == y_test).mean()
        print(f"LightGBM Accuracy: {lgb_accuracy:.2%}")
        
        self.models['lightgbm'] = lgb_model
        self.results['lightgbm'] = lgb_accuracy
        
        # 2. XGBoost
        print("\n=== XGBoost ===")
        xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=200,
            tree_method='hist',
            n_jobs=-1,
            random_state=42
        )
        
        print("Treinando XGBoost...")
        xgb_model.fit(X_train, y_train_lgb)
        xgb_pred = xgb_model.predict(X_test) - 1
        
        xgb_accuracy = (xgb_pred == y_test).mean()
        print(f"XGBoost Accuracy: {xgb_accuracy:.2%}")
        
        self.models['xgboost'] = xgb_model
        self.results['xgboost'] = xgb_accuracy
        
        # 3. Random Forest
        print("\n=== Random Forest ===")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=200,
            min_samples_leaf=100,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        
        print("Treinando Random Forest...")
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        
        rf_accuracy = (rf_pred == y_test).mean()
        print(f"Random Forest Accuracy: {rf_accuracy:.2%}")
        
        self.models['random_forest'] = rf
        self.results['random_forest'] = rf_accuracy
        
        # 4. Ensemble Voting
        print("\n=== ENSEMBLE VOTING ===")
        
        # Weighted voting
        weights = [0.4, 0.4, 0.2]  # LGB, XGB, RF
        ensemble_pred = (
            weights[0] * lgb_pred_class + 
            weights[1] * xgb_pred + 
            weights[2] * rf_pred
        )
        ensemble_pred = np.round(ensemble_pred).astype(int)
        
        ensemble_accuracy = (ensemble_pred == y_test).mean()
        print(f"Ensemble Accuracy: {ensemble_accuracy:.2%}")
        
        self.results['ensemble'] = ensemble_accuracy
        
        # Feature importance
        print("\n=== TOP 20 FEATURES ===")
        
        feature_imp = pd.DataFrame({
            'feature': X.columns,
            'lgb_importance': lgb_model.feature_importance(importance_type='gain'),
            'xgb_importance': xgb_model.feature_importances_,
            'rf_importance': rf.feature_importances_
        })
        
        # Normalizar importâncias
        for col in ['lgb_importance', 'xgb_importance', 'rf_importance']:
            feature_imp[col] = feature_imp[col] / feature_imp[col].sum()
        
        # Média ponderada
        feature_imp['avg_importance'] = (
            weights[0] * feature_imp['lgb_importance'] +
            weights[1] * feature_imp['xgb_importance'] +
            weights[2] * feature_imp['rf_importance']
        )
        
        feature_imp = feature_imp.sort_values('avg_importance', ascending=False)
        print(feature_imp[['feature', 'avg_importance']].head(20))
        
        return feature_imp
    
    def save_models_and_results(self, feature_importance: pd.DataFrame):
        """Salva modelos e resultados"""
        
        output_dir = Path('models/csv_5m_ensemble')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n=== SALVANDO MODELOS E RESULTADOS ===")
        
        # Salvar modelos
        for name, model in self.models.items():
            if name == 'lightgbm':
                model_file = output_dir / f'{name}_5m_{timestamp}.txt'
                model.save_model(str(model_file))
            else:
                model_file = output_dir / f'{name}_5m_{timestamp}.pkl'
                joblib.dump(model, model_file)
            
            print(f"  {name} salvo")
        
        # Salvar feature importance
        feature_file = output_dir / f'features_5m_{timestamp}.csv'
        feature_importance.to_csv(feature_file, index=False)
        
        # Salvar metadados
        metadata = {
            'training_date': datetime.now().isoformat(),
            'sample_size': self.sample_size,
            'results': {name: float(acc) for name, acc in self.results.items()},
            'top_30_features': feature_importance.head(30).to_dict('records'),
            'improvements': {
                'from_300k_to_1m': '+5.43%',
                'from_1m_to_5m': f"+{max(self.results.values()) - 0.5143:.2%}"
            }
        }
        
        metadata_file = output_dir / f'metadata_5m_{timestamp}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[OK] Todos os arquivos salvos em: {output_dir}")

def main():
    """Pipeline principal para 5M registros"""
    
    print("TREINAMENTO AVANÇADO - 5 MILHÕES DE REGISTROS")
    print("Objetivo: Alcançar 55%+ accuracy\n")
    
    csv_path = r"C:\Users\marth\Downloads\WDO_FUT\WDOFUT_BMF_T.csv"
    
    # Criar trainer
    trainer = CSV5MTrainer(csv_path)
    
    # 1. Carregar dados
    df = trainer.load_data_chunked()
    
    # 2. Criar features
    features = trainer.create_comprehensive_features(df)
    
    # 3. Criar targets
    targets = trainer.create_multi_horizon_targets(df)
    
    # 4. Treinar ensemble
    feature_importance = trainer.train_advanced_ensemble(features, targets)
    
    # 5. Salvar resultados
    trainer.save_models_and_results(feature_importance)
    
    # Limpar memória
    del df, features, targets
    gc.collect()
    
    print("\n" + "=" * 80)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 80)
    
    print("\nRESUMO FINAL:")
    for name, accuracy in trainer.results.items():
        print(f"  {name}: {accuracy:.2%}")
    
    print("\nPROGRESSÃO DE ACCURACY:")
    print("  300k registros: 46.00%")
    print("  1M registros:   51.43% (+5.43%)")
    print(f"  5M registros:   {max(trainer.results.values()):.2%} (+{max(trainer.results.values()) - 0.5143:.2%})")
    
    print("\nPRÓXIMOS PASSOS:")
    print("1. Integrar com Book Collector para microestrutura")
    print("2. Implementar backtesting completo")
    print("3. Deploy em produção com data real-time")

if __name__ == "__main__":
    main()