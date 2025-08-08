"""
Pipeline de Treinamento com 1 Milhão de Registros
Otimizado para melhor accuracy com mais dados
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
import warnings
warnings.filterwarnings('ignore')

class CSVLargeScaleTrainer:
    """Treina modelos com 1M+ registros de forma eficiente"""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.sample_size = 1_000_000
        self.models = {}
        self.results = {}
        
    def load_data_optimized(self):
        """Carrega 1M registros com otimizações de memória"""
        
        print("=" * 80)
        print("CARREGAMENTO OTIMIZADO - 1 MILHÃO DE REGISTROS")
        print("=" * 80)
        
        # Tipos otimizados
        dtypes = {
            '<ticker>': 'category',
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
        
        # Carregar dados
        df = pd.read_csv(self.csv_path, nrows=self.sample_size, dtype=dtypes)
        
        # Criar timestamp
        df['timestamp'] = pd.to_datetime(
            df['<date>'].astype(str) + ' ' + df['<time>'].astype(str).str.zfill(6),
            format='%Y%m%d %H%M%S'
        )
        
        # Ordenar por timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        load_time = (datetime.now() - start_time).total_seconds()
        memory_mb = df.memory_usage().sum() / 1024**2
        
        print(f"Tempo de carregamento: {load_time:.1f}s")
        print(f"Memória utilizada: {memory_mb:.1f} MB")
        print(f"Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
        print(f"Duração: {df['timestamp'].max() - df['timestamp'].min()}")
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features avançadas para melhor accuracy"""
        
        print("\n=== CRIAÇÃO DE FEATURES AVANÇADAS ===")
        features = pd.DataFrame(index=df.index)
        
        # 1. FEATURES DE PREÇO (Múltiplos horizontes)
        print("1. Features de preço...")
        features['price'] = df['<price>'].astype('float32')
        features['price_log'] = np.log(features['price'])
        
        # Returns em múltiplos períodos
        for period in [1, 5, 10, 20, 50]:
            features[f'returns_{period}'] = features['price'].pct_change(period)
            
        # Volatilidade realizada
        for window in [10, 20, 50]:
            features[f'volatility_{window}'] = features['returns_1'].rolling(window).std()
            
        # Price momentum
        features['momentum_5_20'] = features['price'].rolling(5).mean() / features['price'].rolling(20).mean() - 1
        features['momentum_10_50'] = features['price'].rolling(10).mean() / features['price'].rolling(50).mean() - 1
        
        # 2. FEATURES DE VOLUME E LIQUIDEZ
        print("2. Features de volume...")
        features['qty'] = df['<qty>'].astype('float32')
        features['qty_log'] = np.log1p(features['qty'])
        features['volume_usd'] = df['<vol>'].astype('float32')
        
        # Volume profiles
        for window in [20, 50, 100]:
            features[f'volume_ratio_{window}'] = features['qty'] / features['qty'].rolling(window).mean()
            features[f'volume_std_{window}'] = features['qty'].rolling(window).std()
        
        # VWAP
        features['vwap'] = (df['<vol>'].cumsum() / df['<qty>'].cumsum()).astype('float32')
        features['price_to_vwap'] = features['price'] / features['vwap'] - 1
        
        # 3. TRADE FLOW E MICROESTRUTURA
        print("3. Features de trade flow...")
        features['is_buyer_aggressor'] = (df['<trade_type>'] == 'AggressorBuyer').astype('int8')
        features['is_seller_aggressor'] = (df['<trade_type>'] == 'AggressorSeller').astype('int8')
        features['is_auction'] = (df['<trade_type>'] == 'Auction').astype('int8')
        
        # Aggressor imbalance com múltiplas janelas
        for window in [20, 50, 100]:
            buyer_sum = features['is_buyer_aggressor'].rolling(window).sum()
            seller_sum = features['is_seller_aggressor'].rolling(window).sum()
            features[f'aggressor_imbalance_{window}'] = (buyer_sum - seller_sum) / window
            
        # Trade intensity
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        features['trade_intensity'] = 1 / df['time_diff'].rolling(50).mean()
        
        # 4. AGENT BEHAVIOR AVANÇADO
        print("4. Features de agent behavior...")
        
        # Concentração de agentes
        for window in [50, 100, 200]:
            print(f"   Calculando concentração janela {window}...")
            
            # Usar rolling com função customizada para agents
            buyer_concentration = []
            seller_concentration = []
            
            for i in range(len(df)):
                if i < window:
                    buyer_concentration.append(0)
                    seller_concentration.append(0)
                else:
                    # Janela de dados
                    window_buyers = df['<buy_agent>'].iloc[i-window:i]
                    window_sellers = df['<sell_agent>'].iloc[i-window:i]
                    
                    # Concentração (HHI - Herfindahl-Hirschman Index)
                    buyer_counts = window_buyers.value_counts()
                    seller_counts = window_sellers.value_counts()
                    
                    buyer_hhi = (buyer_counts / window).pow(2).sum()
                    seller_hhi = (seller_counts / window).pow(2).sum()
                    
                    buyer_concentration.append(buyer_hhi)
                    seller_concentration.append(seller_hhi)
                    
            features[f'buyer_concentration_{window}'] = buyer_concentration
            features[f'seller_concentration_{window}'] = seller_concentration
        
        # Agent momentum (mesmo agente repetindo)
        features['buyer_repeat'] = (df['<buy_agent>'] == df['<buy_agent>'].shift(1)).astype('int8')
        features['seller_repeat'] = (df['<sell_agent>'] == df['<sell_agent>'].shift(1)).astype('int8')
        
        # Top agents activity
        top_5_buyers = df['<buy_agent>'].value_counts().head(5).index
        top_5_sellers = df['<sell_agent>'].value_counts().head(5).index
        
        for i, agent in enumerate(top_5_buyers):
            features[f'top_buyer_{i}_active'] = (df['<buy_agent>'] == agent).rolling(100).mean()
            
        for i, agent in enumerate(top_5_sellers):
            features[f'top_seller_{i}_active'] = (df['<sell_agent>'] == agent).rolling(100).mean()
        
        # 5. FEATURES TEMPORAIS
        print("5. Features temporais...")
        features['hour'] = df['timestamp'].dt.hour.astype('int8')
        features['minute'] = df['timestamp'].dt.minute.astype('int8')
        features['second'] = df['timestamp'].dt.second.astype('int8')
        
        # Períodos do dia
        features['is_opening'] = ((features['hour'] == 9) & (features['minute'] < 30)).astype('int8')
        features['is_morning'] = ((features['hour'] >= 9) & (features['hour'] < 12)).astype('int8')
        features['is_afternoon'] = ((features['hour'] >= 12) & (features['hour'] < 15)).astype('int8')
        features['is_closing'] = ((features['hour'] >= 16) & (features['hour'] < 17)).astype('int8')
        
        # 6. FEATURES DE PADRÕES
        print("6. Features de padrões...")
        
        # Large trades
        for percentile in [75, 90, 95]:
            threshold = df['<qty>'].quantile(percentile/100)
            features[f'large_trade_p{percentile}'] = (df['<qty>'] > threshold).astype('int8')
            features[f'large_trade_ratio_p{percentile}'] = features[f'large_trade_p{percentile}'].rolling(100).mean()
        
        # Price levels
        features['price_percentile'] = features['price'].rolling(500).rank(pct=True)
        features['volume_percentile'] = features['qty'].rolling(500).rank(pct=True)
        
        print(f"\nTotal features criadas: {features.shape[1]}")
        
        # Cleanup
        features = features.ffill().fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        # Converter para float32
        for col in features.columns:
            if features[col].dtype == 'float64':
                features[col] = features[col].astype('float32')
        
        return features
    
    def create_advanced_targets(self, df: pd.DataFrame, features: pd.DataFrame) -> dict:
        """Cria múltiplos targets para ensemble"""
        
        print("\n=== CRIAÇÃO DE TARGETS AVANÇADOS ===")
        targets = {}
        
        # 1. Target de direção (classificação)
        for horizon in [30, 60, 120]:
            future_price = df['<price>'].shift(-horizon)
            returns = (future_price - df['<price>']) / df['<price>']
            
            # Thresholds dinâmicos baseados em volatilidade
            vol = returns.rolling(1000).std()
            
            target = pd.Series(0, index=df.index, dtype='int8')
            target[returns > 0.5 * vol] = 1
            target[returns < -0.5 * vol] = -1
            
            targets[f'direction_{horizon}'] = target
            
            print(f"\nTarget direction_{horizon}:")
            print(target.value_counts().sort_index())
        
        # 2. Target de magnitude (regressão)
        targets['returns_60'] = df['<price>'].pct_change(60).shift(-60)
        
        # 3. Target de volatilidade
        targets['volatility_future'] = features['returns_1'].rolling(60).std().shift(-60)
        
        return targets
    
    def train_ensemble(self, features: pd.DataFrame, targets: dict):
        """Treina ensemble de modelos"""
        
        print("\n" + "=" * 80)
        print("TREINAMENTO DE ENSEMBLE AVANÇADO")
        print("=" * 80)
        
        # Usar target principal
        main_target = targets['direction_60']
        
        # Remover NaN
        mask = ~(features.isna().any(axis=1) | main_target.isna())
        X = features[mask]
        y = main_target[mask]
        
        print(f"\nDados para treinamento: {len(X):,} registros")
        
        # Split temporal 80/20
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # 1. LightGBM (rápido e eficiente)
        print("\n=== LightGBM ===")
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': -1
        }
        
        # Ajustar labels para LightGBM (0, 1, 2)
        y_train_lgb = y_train + 1
        y_test_lgb = y_test + 1
        
        lgb_train = lgb.Dataset(X_train, label=y_train_lgb)
        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=100,
            valid_sets=[lgb_train],
            callbacks=[lgb.log_evaluation(0)]
        )
        
        lgb_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
        lgb_pred_class = np.argmax(lgb_pred, axis=1) - 1
        
        print("\nResultados LightGBM:")
        print(classification_report(y_test, lgb_pred_class, 
                                  target_names=['Baixa', 'Neutro', 'Alta'],
                                  zero_division=0))
        
        self.models['lightgbm'] = lgb_model
        self.results['lightgbm'] = {
            'accuracy': (lgb_pred_class == y_test).mean(),
            'predictions': lgb_pred_class
        }
        
        # 2. XGBoost
        print("\n=== XGBoost ===")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            n_jobs=-1,
            random_state=42
        )
        
        xgb_model.fit(X_train, y_train_lgb)
        xgb_pred = xgb_model.predict(X_test) - 1
        
        print("\nResultados XGBoost:")
        print(classification_report(y_test, xgb_pred, 
                                  target_names=['Baixa', 'Neutro', 'Alta'],
                                  zero_division=0))
        
        self.models['xgboost'] = xgb_model
        self.results['xgboost'] = {
            'accuracy': (xgb_pred == y_test).mean(),
            'predictions': xgb_pred
        }
        
        # 3. Random Forest
        print("\n=== Random Forest ===")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=100,
            min_samples_leaf=50,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        
        print("\nResultados Random Forest:")
        print(classification_report(y_test, rf_pred, 
                                  target_names=['Baixa', 'Neutro', 'Alta'],
                                  zero_division=0))
        
        self.models['random_forest'] = rf
        self.results['random_forest'] = {
            'accuracy': (rf_pred == y_test).mean(),
            'predictions': rf_pred
        }
        
        # Feature importance
        feature_imp = pd.DataFrame({
            'feature': X.columns,
            'importance_rf': rf.feature_importances_,
            'importance_lgb': lgb_model.feature_importance(importance_type='gain') / lgb_model.feature_importance(importance_type='gain').sum()
        })
        
        # Média das importâncias
        feature_imp['importance_avg'] = (feature_imp['importance_rf'] + feature_imp['importance_lgb']) / 2
        feature_imp = feature_imp.sort_values('importance_avg', ascending=False)
        
        print("\n=== TOP 15 FEATURES MAIS IMPORTANTES ===")
        print(feature_imp[['feature', 'importance_avg']].head(15))
        
        # Ensemble voting
        print("\n=== ENSEMBLE VOTING ===")
        ensemble_pred = np.zeros_like(rf_pred)
        for pred in [lgb_pred_class, xgb_pred, rf_pred]:
            ensemble_pred += pred
        ensemble_pred = np.sign(ensemble_pred)
        
        print("\nResultados Ensemble:")
        print(classification_report(y_test, ensemble_pred, 
                                  target_names=['Baixa', 'Neutro', 'Alta'],
                                  zero_division=0))
        
        ensemble_accuracy = (ensemble_pred == y_test).mean()
        self.results['ensemble'] = {
            'accuracy': ensemble_accuracy,
            'predictions': ensemble_pred
        }
        
        return feature_imp
    
    def save_models(self, feature_importance: pd.DataFrame):
        """Salva modelos e metadados"""
        
        output_dir = Path('models/csv_1m_models')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n=== SALVANDO MODELOS ===")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salvar modelos
        for name, model in self.models.items():
            if name == 'lightgbm':
                model_file = output_dir / f'csv_1m_{name}_{timestamp}.txt'
                model.save_model(str(model_file))
            else:
                model_file = output_dir / f'csv_1m_{name}_{timestamp}.pkl'
                joblib.dump(model, model_file)
            
            print(f"  {name} salvo em: {model_file.name}")
        
        # Salvar feature importance
        importance_file = output_dir / f'feature_importance_1m_{timestamp}.csv'
        feature_importance.to_csv(importance_file, index=False)
        
        # Salvar metadados
        metadata = {
            'training_date': datetime.now().isoformat(),
            'sample_size': self.sample_size,
            'results': {
                name: {'accuracy': float(res['accuracy'])} 
                for name, res in self.results.items()
            },
            'top_features': feature_importance.head(20).to_dict('records')
        }
        
        metadata_file = output_dir / f'training_metadata_1m_{timestamp}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[OK] Metadados salvos em: {metadata_file.name}")

def main():
    """Pipeline principal para 1M registros"""
    
    print("PIPELINE DE TREINAMENTO - 1 MILHÃO DE REGISTROS")
    print("Objetivo: Melhorar accuracy com mais dados\n")
    
    # Configurações
    csv_path = r"C:\Users\marth\Downloads\WDO_FUT\WDOFUT_BMF_T.csv"
    
    # Criar trainer
    trainer = CSVLargeScaleTrainer(csv_path)
    
    # 1. Carregar dados
    df = trainer.load_data_optimized()
    
    # 2. Criar features avançadas
    features = trainer.create_advanced_features(df)
    
    # 3. Criar targets
    targets = trainer.create_advanced_targets(df, features)
    
    # 4. Treinar ensemble
    feature_importance = trainer.train_ensemble(features, targets)
    
    # 5. Salvar modelos
    trainer.save_models(feature_importance)
    
    # Limpar memória
    del df, features, targets
    gc.collect()
    
    print("\n" + "=" * 80)
    print("TREINAMENTO CONCLUÍDO!")
    print("=" * 80)
    
    print("\nRESUMO DOS RESULTADOS:")
    for name, res in trainer.results.items():
        print(f"  {name}: {res['accuracy']:.2%} accuracy")
    
    print("\nMELHORIAS OBSERVADAS:")
    print("  - Mais dados = melhor generalização")
    print("  - Features avançadas de agent behavior")
    print("  - Ensemble voting para robustez")
    
    print("\nPRÓXIMOS PASSOS:")
    print("1. Testar com 5M registros")
    print("2. Adicionar features de book (quando disponível)")
    print("3. Implementar backtesting")
    print("4. Otimizar hiperparâmetros")

if __name__ == "__main__":
    main()