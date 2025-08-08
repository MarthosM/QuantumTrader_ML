"""
Pipeline Otimizado de Treinamento com Dados CSV Tick-a-Tick
Versão mais eficiente para grandes volumes de dados
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import gc


def prepare_csv_data(csv_path: str, sample_size: int = 100000):
    """Prepara dados do CSV de forma eficiente"""
    
    print("=" * 80)
    print("PREPARAÇÃO DE DADOS CSV TICK-A-TICK")
    print("=" * 80)
    
    # Ler amostra do CSV
    print(f"\nCarregando {sample_size:,} registros...")
    
    df = pd.read_csv(csv_path, nrows=sample_size)
    
    # Criar timestamp
    df['timestamp'] = pd.to_datetime(df['<date>'].astype(str) + ' ' + 
                                   df['<time>'].astype(str).str.zfill(6), 
                                   format='%Y%m%d %H%M%S')
    
    print(f"Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
    
    return df


def create_simplified_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features simplificadas mas eficazes"""
    
    print("\n=== CRIANDO FEATURES SIMPLIFICADAS ===")
    
    features = pd.DataFrame()
    
    # 1. Features de Preço
    features['price'] = df['<price>']
    features['price_log'] = np.log(df['<price>'])
    features['price_change'] = df['<price>'].pct_change()
    features['price_volatility'] = df['<price>'].rolling(20).std()
    
    # 2. Features de Volume
    features['qty'] = df['<qty>']
    features['qty_log'] = np.log(df['<qty>'] + 1)
    features['volume'] = df['<vol>']
    features['avg_price'] = df['<vol>'] / df['<qty>']
    
    # 3. Features de Trade Flow
    features['trade_size_ratio'] = df['<qty>'] / df['<qty>'].rolling(50).mean()
    features['volume_ratio'] = df['<vol>'] / df['<vol>'].rolling(50).mean()
    
    # 4. Aggressor Analysis
    features['is_buyer_aggressor'] = (df['<trade_type>'] == 'AggressorBuyer').astype(int)
    features['is_seller_aggressor'] = (df['<trade_type>'] == 'AggressorSeller').astype(int)
    features['is_auction'] = (df['<trade_type>'] == 'Auction').astype(int)
    features['is_rlp'] = (df['<trade_type>'] == 'RLP').astype(int)
    
    # Aggressor imbalance (rolling)
    features['aggressor_imbalance'] = (
        features['is_buyer_aggressor'].rolling(50).sum() - 
        features['is_seller_aggressor'].rolling(50).sum()
    )
    
    # 5. Agent Features (simplificado)
    # Top agents encoding
    top_buyers = df['<buy_agent>'].value_counts().head(10).index
    top_sellers = df['<sell_agent>'].value_counts().head(10).index
    
    for i, agent in enumerate(top_buyers):
        features[f'is_top_buyer_{i}'] = (df['<buy_agent>'] == agent).astype(int)
        
    for i, agent in enumerate(top_sellers):
        features[f'is_top_seller_{i}'] = (df['<sell_agent>'] == agent).astype(int)
    
    # 6. Temporal Features
    features['hour'] = df['timestamp'].dt.hour
    features['minute'] = df['timestamp'].dt.minute
    features['seconds_since_open'] = (df['timestamp'] - df['timestamp'].dt.normalize()).dt.total_seconds()
    
    # 7. Trade Intensity
    features['trade_count_1min'] = 1  # Será agregado depois
    features['trade_count_5min'] = 1
    
    # Calcular trades por minuto
    df['minute_group'] = df['timestamp'].dt.floor('1min')
    minute_counts = df.groupby('minute_group').size()
    df['trades_per_minute'] = df['minute_group'].map(minute_counts)
    features['trades_per_minute'] = df['trades_per_minute']
    
    # 8. Large Trade Indicator
    qty_threshold = df['<qty>'].quantile(0.9)
    features['is_large_trade'] = (df['<qty>'] > qty_threshold).astype(int)
    
    print(f"Features criadas: {features.shape[1]}")
    
    # Remover infinitos e NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)
    
    return features


def create_target(df: pd.DataFrame, horizon: int = 60) -> pd.Series:
    """Cria target de classificação"""
    
    print(f"\n=== CRIANDO TARGET (horizonte: {horizon} trades) ===")
    
    # Future return
    future_price = df['<price>'].shift(-horizon)
    returns = (future_price - df['<price>']) / df['<price>']
    
    # Classificar em 3 categorias
    target = pd.Series(0, index=df.index)  # 0 = Neutro
    
    # Definir thresholds baseados em percentis
    upper_threshold = returns.quantile(0.7)
    lower_threshold = returns.quantile(0.3)
    
    target[returns > upper_threshold] = 1   # Alta
    target[returns < lower_threshold] = -1  # Baixa
    
    print(f"Distribuição do target:")
    print(target.value_counts().sort_index())
    print(f"\nThresholds: Baixa < {lower_threshold:.4f} < Neutro < {upper_threshold:.4f} < Alta")
    
    return target


def train_agent_behavior_model(features: pd.DataFrame, target: pd.Series, output_dir: str):
    """Treina modelo focado em comportamento de agentes"""
    
    print("\n" + "=" * 80)
    print("TREINAMENTO DO MODELO DE AGENT BEHAVIOR")
    print("=" * 80)
    
    # Preparar dados
    mask = ~target.isna()
    X = features[mask]
    y = target[mask]
    
    print(f"\nDados disponíveis: {len(X):,} registros")
    
    # Split temporal (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Random Forest (robusto e interpretável)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=100,
        min_samples_leaf=50,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    
    print("\nTreinando Random Forest...")
    model.fit(X_train, y_train)
    
    # Avaliar
    y_pred = model.predict(X_test)
    
    print("\n=== RESULTADOS ===")
    print(classification_report(y_test, y_pred, 
                              target_names=['Baixa (-1)', 'Neutro (0)', 'Alta (1)']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Features Mais Importantes:")
    print(feature_importance.head(15))
    
    # Salvar modelo
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_file = output_path / f'agent_behavior_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    joblib.dump(model, model_file)
    
    # Salvar metadados
    metadata = {
        'model_type': 'RandomForest - Agent Behavior',
        'training_date': datetime.now().isoformat(),
        'features': list(X.columns),
        'feature_importance': feature_importance.head(20).to_dict('records'),
        'performance': {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'target_distribution': y_train.value_counts().to_dict()
        }
    }
    
    metadata_file = output_path / f'agent_model_metadata_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[OK] Modelo salvo em: {model_file}")
    print(f"[OK] Metadados em: {metadata_file}")
    
    return model, feature_importance


def main():
    """Pipeline principal otimizado"""
    
    # Configurações
    csv_path = r"C:\Users\marth\Downloads\WDO_FUT\WDOFUT_BMF_T.csv"
    sample_size = 200000  # Começar com 200k registros
    
    print("PIPELINE DE TREINAMENTO COM DADOS CSV")
    print(f"Amostra: {sample_size:,} registros")
    
    # 1. Carregar dados
    df = prepare_csv_data(csv_path, sample_size)
    
    # 2. Criar features
    features = create_simplified_features(df)
    
    # 3. Criar target
    target = create_target(df, horizon=60)
    
    # 4. Treinar modelo
    model, importance = train_agent_behavior_model(
        features, 
        target,
        'models/csv_agent_models'
    )
    
    # Limpar memória
    del df
    gc.collect()
    
    print("\n" + "=" * 80)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 80)
    
    print("\nPRÓXIMOS PASSOS:")
    print("1. Aumentar sample_size gradualmente (500k, 1M, etc)")
    print("2. Adicionar mais features de agent behavior")
    print("3. Criar ensemble com modelos do Book Collector")
    print("4. Implementar backtesting com ambas fontes")


if __name__ == "__main__":
    main()