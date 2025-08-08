"""
Pipeline de Treinamento com Dados CSV Tick-a-Tick
Treina modelos HMARL com foco em comportamento de agentes e fluxo de trades
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')


class CSVTickDataTrainer:
    """
    Treina modelos com dados tick-a-tick do CSV
    Foco em agent behavior e trade flow patterns
    """
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.models = {}
        self.encoders = {}
        self.feature_importance = {}
        
    def load_and_prepare_data(self, nrows=None):
        """Carrega e prepara dados do CSV"""
        
        print("=" * 80)
        print("CARREGANDO DADOS CSV TICK-A-TICK")
        print("=" * 80)
        
        # Carregar dados
        print(f"\nCarregando dados de: {self.csv_path.name}")
        
        # Para economia de memória, ler em chunks se nrows não especificado
        if nrows is None:
            # Ler primeiro chunk para estimar
            chunk = pd.read_csv(self.csv_path, nrows=100000)
            print(f"Lendo arquivo completo em chunks...")
            
            chunks = []
            for i, chunk in enumerate(pd.read_csv(self.csv_path, chunksize=100000)):
                chunks.append(chunk)
                if i >= 9:  # Limitar a 1M de linhas por enquanto
                    break
                print(f"  Chunk {i+1} carregado...")
                
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(self.csv_path, nrows=nrows)
            
        print(f"Dados carregados: {len(df):,} registros")
        
        # Criar timestamp
        df['timestamp'] = pd.to_datetime(df['<date>'].astype(str) + ' ' + 
                                       df['<time>'].astype(str).str.zfill(6), 
                                       format='%Y%m%d %H%M%S')
        
        # Ordenar por timestamp
        df = df.sort_values('timestamp')
        
        print(f"Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
        
        return df
        
    def create_agent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features baseadas em comportamento de agentes"""
        
        print("\n=== CRIANDO FEATURES DE AGENT BEHAVIOR ===")
        
        features = pd.DataFrame(index=df.index)
        
        # 1. Concentração de agentes (últimos N trades)
        window_sizes = [10, 50, 100]
        
        for window in window_sizes:
            # Concentração de compradores
            buyer_conc = []
            seller_conc = []
            
            for i in range(len(df)):
                if i < window:
                    buyer_conc.append(0)
                    seller_conc.append(0)
                else:
                    # Calcular concentração no window
                    buyer_window = df['<buy_agent>'].iloc[i-window:i]
                    seller_window = df['<sell_agent>'].iloc[i-window:i]
                    
                    # Agente mais frequente / total
                    buyer_counts = buyer_window.value_counts()
                    seller_counts = seller_window.value_counts()
                    
                    buyer_conc.append(buyer_counts.iloc[0] / window if len(buyer_counts) > 0 else 0)
                    seller_conc.append(seller_counts.iloc[0] / window if len(seller_counts) > 0 else 0)
                    
            features[f'buyer_concentration_{window}'] = buyer_conc
            features[f'seller_concentration_{window}'] = seller_conc
            
        # 2. Agent Momentum (mesmo agente repetindo)
        features['buyer_streak'] = (df['<buy_agent>'] == df['<buy_agent>'].shift(1)).astype(int)
        features['seller_streak'] = (df['<sell_agent>'] == df['<sell_agent>'].shift(1)).astype(int)
        
        # 3. Agent Diversity
        for window in [20, 50]:
            unique_buyers = []
            unique_sellers = []
            
            for i in range(len(df)):
                if i < window:
                    unique_buyers.append(0)
                    unique_sellers.append(0)
                else:
                    unique_buyers.append(df['<buy_agent>'].iloc[i-window:i].nunique())
                    unique_sellers.append(df['<sell_agent>'].iloc[i-window:i].nunique())
                    
            features[f'unique_buyers_{window}'] = unique_buyers
            features[f'unique_sellers_{window}'] = unique_sellers
            
        # 4. Institutional vs Retail (baseado em volume)
        features['avg_trade_size'] = df['<qty>'].rolling(20).mean()
        features['large_trade_ratio'] = (df['<qty>'] > df['<qty>'].rolling(100).mean() * 2).astype(int)
        
        # 5. Agent Imbalance
        # Encode agents para análise numérica
        buy_encoder = LabelEncoder()
        sell_encoder = LabelEncoder()
        
        df['buy_agent_encoded'] = buy_encoder.fit_transform(df['<buy_agent>'])
        df['sell_agent_encoded'] = sell_encoder.fit_transform(df['<sell_agent>'])
        
        self.encoders['buy_agent'] = buy_encoder
        self.encoders['sell_agent'] = sell_encoder
        
        # Calcular dominância de agentes específicos
        top_buyers = df['<buy_agent>'].value_counts().head(5).index
        top_sellers = df['<sell_agent>'].value_counts().head(5).index
        
        for agent in top_buyers[:3]:
            features[f'buyer_{agent}_active'] = (df['<buy_agent>'] == agent).rolling(50).sum()
            
        for agent in top_sellers[:3]:
            features[f'seller_{agent}_active'] = (df['<sell_agent>'] == agent).rolling(50).sum()
            
        print(f"Features de agentes criadas: {features.shape[1]}")
        
        return features
        
    def create_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de fluxo de trades"""
        
        print("\n=== CRIANDO FEATURES DE TRADE FLOW ===")
        
        features = pd.DataFrame(index=df.index)
        
        # 1. Trade Intensity
        features['trade_count_1min'] = df.groupby(pd.Grouper(key='timestamp', freq='1min')).cumcount()
        features['trade_count_5min'] = df.groupby(pd.Grouper(key='timestamp', freq='5min')).cumcount()
        
        # 2. Volume Features
        features['volume_1min'] = df['<vol>'].rolling('1min', on='timestamp').sum()
        features['volume_5min'] = df['<vol>'].rolling('5min', on='timestamp').sum()
        features['volume_acceleration'] = features['volume_1min'] / features['volume_5min']
        
        # 3. Price Movement
        features['price_change'] = df['<price>'].pct_change()
        features['price_volatility'] = df['<price>'].rolling(50).std()
        features['price_momentum'] = df['<price>'].rolling(20).apply(lambda x: (x[-1] - x[0]) / x[0])
        
        # 4. Trade Type Analysis
        features['aggressor_buy_ratio'] = (df['<trade_type>'] == 'AggressorBuyer').rolling(50).mean()
        features['aggressor_sell_ratio'] = (df['<trade_type>'] == 'AggressorSeller').rolling(50).mean()
        features['aggressor_imbalance'] = features['aggressor_buy_ratio'] - features['aggressor_sell_ratio']
        
        # 5. Auction Activity
        features['auction_ratio'] = (df['<trade_type>'] == 'Auction').rolling(100).mean()
        features['rlp_ratio'] = (df['<trade_type>'] == 'RLP').rolling(100).mean()
        
        # 6. Trade Size Distribution
        features['small_trades'] = (df['<qty>'] <= 5).rolling(50).sum()
        features['medium_trades'] = ((df['<qty>'] > 5) & (df['<qty>'] <= 50)).rolling(50).sum()
        features['large_trades'] = (df['<qty>'] > 50).rolling(50).sum()
        
        # 7. Time-based features
        features['hour'] = df['timestamp'].dt.hour
        features['minute'] = df['timestamp'].dt.minute
        features['is_opening'] = (features['hour'] == 9).astype(int)
        features['is_closing'] = (features['hour'] >= 16).astype(int)
        
        print(f"Features de fluxo criadas: {features.shape[1]}")
        
        return features
        
    def create_targets(self, df: pd.DataFrame, horizon_seconds=60) -> pd.Series:
        """Cria targets para predição"""
        
        print(f"\n=== CRIANDO TARGETS (horizonte: {horizon_seconds}s) ===")
        
        # Target: Direção do preço no próximo período
        future_price = df['<price>'].shift(-horizon_seconds)  # Aproximado
        price_change = (future_price - df['<price>']) / df['<price>']
        
        # Classificação em 3 classes
        targets = pd.Series(index=df.index, dtype=int)
        targets[price_change > 0.0002] = 1   # Alta
        targets[price_change < -0.0002] = -1  # Baixa
        targets.fillna(0, inplace=True)       # Neutro
        
        print(f"Distribuição do target:")
        print(targets.value_counts())
        
        return targets
        
    def train_models(self, features: pd.DataFrame, targets: pd.Series):
        """Treina múltiplos modelos"""
        
        print("\n" + "=" * 80)
        print("TREINAMENTO DOS MODELOS")
        print("=" * 80)
        
        # Remover NaN
        mask = ~(features.isna().any(axis=1) | targets.isna())
        X = features[mask]
        y = targets[mask]
        
        print(f"\nDados para treinamento: {len(X):,} registros")
        
        # Split temporal
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # 1. Random Forest
        print("\n=== Random Forest ===")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            n_jobs=-1,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        print("Performance:")
        print(classification_report(y_test, rf_pred, target_names=['Baixa', 'Neutro', 'Alta']))
        
        self.models['random_forest'] = rf_model
        self.feature_importance['random_forest'] = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 2. XGBoost
        print("\n=== XGBoost ===")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X_train, y_train + 1)  # XGBoost precisa de labels 0, 1, 2
        xgb_pred = xgb_model.predict(X_test) - 1
        
        print("Performance:")
        print(classification_report(y_test, xgb_pred, target_names=['Baixa', 'Neutro', 'Alta']))
        
        self.models['xgboost'] = xgb_model
        
        # 3. Gradient Boosting
        print("\n=== Gradient Boosting ===")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        
        print("Performance:")
        print(classification_report(y_test, gb_pred, target_names=['Baixa', 'Neutro', 'Alta']))
        
        self.models['gradient_boosting'] = gb_model
        
        return X_test, y_test
        
    def save_models(self, output_dir: str):
        """Salva modelos treinados"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n=== SALVANDO MODELOS ===")
        
        for model_name, model in self.models.items():
            model_file = output_path / f'csv_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
            joblib.dump(model, model_file)
            print(f"  {model_name} salvo em: {model_file.name}")
            
        # Salvar encoders
        encoders_file = output_path / 'agent_encoders.pkl'
        joblib.dump(self.encoders, encoders_file)
        
        # Salvar feature importance
        for model_name, importance_df in self.feature_importance.items():
            importance_file = output_path / f'feature_importance_{model_name}.csv'
            importance_df.to_csv(importance_file, index=False)
            
            print(f"\nTop 10 features - {model_name}:")
            print(importance_df.head(10))
            
        # Salvar metadados
        metadata = {
            'training_date': datetime.now().isoformat(),
            'csv_file': str(self.csv_path),
            'models': list(self.models.keys()),
            'features_created': {
                'agent_features': True,
                'flow_features': True
            }
        }
        
        metadata_file = output_path / 'csv_models_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    """Pipeline principal"""
    
    # Path do CSV
    csv_path = r"C:\Users\marth\Downloads\WDO_FUT\WDOFUT_BMF_T.csv"
    
    # Criar trainer
    trainer = CSVTickDataTrainer(csv_path)
    
    # 1. Carregar dados (limitado para teste rápido)
    df = trainer.load_and_prepare_data(nrows=500000)  # 500k linhas para início
    
    # 2. Criar features
    agent_features = trainer.create_agent_features(df)
    flow_features = trainer.create_flow_features(df)
    
    # Combinar features
    all_features = pd.concat([agent_features, flow_features], axis=1)
    
    # 3. Criar targets
    targets = trainer.create_targets(df)
    
    # 4. Treinar modelos
    X_test, y_test = trainer.train_models(all_features, targets)
    
    # 5. Salvar modelos
    trainer.save_models('models/csv_tick_models')
    
    print("\n" + "=" * 80)
    print("TREINAMENTO CONCLUÍDO!")
    print("=" * 80)
    print("\nPróximos passos:")
    print("1. Treinar com dataset completo (17GB)")
    print("2. Criar ensemble com modelos do Book Collector")
    print("3. Implementar estratégia de trading híbrida")
    print("4. Backtest com ambas fontes de dados")


if __name__ == "__main__":
    main()