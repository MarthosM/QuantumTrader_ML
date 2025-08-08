"""
Pipeline Rápido de Treinamento com Dados CSV
Versão otimizada para performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import gc

def load_csv_sample(csv_path: str, sample_size: int = 100000):
    """Carrega amostra do CSV com tipos otimizados"""
    
    print("=" * 80)
    print("CARREGAMENTO OTIMIZADO DE DADOS CSV")
    print("=" * 80)
    
    # Tipos de dados otimizados para economia de memória
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
    
    print(f"\nCarregando {sample_size:,} registros...")
    df = pd.read_csv(csv_path, nrows=sample_size, dtype=dtypes)
    
    # Criar timestamp
    df['timestamp'] = pd.to_datetime(df['<date>'].astype(str) + ' ' + 
                                   df['<time>'].astype(str).str.zfill(6), 
                                   format='%Y%m%d %H%M%S')
    
    # Ordenar por timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Dados carregados: {len(df):,} registros")
    print(f"Memória utilizada: {df.memory_usage().sum() / 1024**2:.1f} MB")
    print(f"Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
    
    return df

def create_fast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features otimizadas para performance"""
    
    print("\n=== CRIAÇÃO RÁPIDA DE FEATURES ===")
    
    features = pd.DataFrame(index=df.index)
    
    # 1. Features de Preço (vetorizadas)
    features['price'] = df['<price>'].astype('float32')
    features['price_log'] = np.log(features['price'])
    features['returns_1'] = features['price'].pct_change(1)
    features['returns_5'] = features['price'].pct_change(5)
    features['returns_20'] = features['price'].pct_change(20)
    
    # Volatilidade rolling
    features['volatility_20'] = features['returns_1'].rolling(20).std()
    features['volatility_50'] = features['returns_1'].rolling(50).std()
    
    # 2. Features de Volume (simples)
    features['qty'] = df['<qty>'].astype('float32')
    features['qty_log'] = np.log1p(features['qty'])
    features['volume_ratio'] = features['qty'] / features['qty'].rolling(50).mean()
    
    # 3. Trade Flow (vetorizado)
    features['is_buyer_aggressor'] = (df['<trade_type>'] == 'AggressorBuyer').astype('int8')
    features['is_seller_aggressor'] = (df['<trade_type>'] == 'AggressorSeller').astype('int8')
    
    # Imbalance com EWM (mais rápido que rolling)
    features['aggressor_imbalance'] = (
        features['is_buyer_aggressor'].ewm(span=50).mean() - 
        features['is_seller_aggressor'].ewm(span=50).mean()
    )
    
    # 4. Agent Features Simplificadas
    # Top 5 agentes mais ativos
    top_buyers = df['<buy_agent>'].value_counts().head(5).index
    top_sellers = df['<sell_agent>'].value_counts().head(5).index
    
    # One-hot encoding apenas dos top agents
    for i, agent in enumerate(top_buyers):
        features[f'top_buyer_{i}'] = (df['<buy_agent>'] == agent).astype('int8')
    
    for i, agent in enumerate(top_sellers):
        features[f'top_seller_{i}'] = (df['<sell_agent>'] == agent).astype('int8')
    
    # 5. Agent Activity Score (quanto os top agents estão ativos)
    features['top_buyers_active'] = features[[f'top_buyer_{i}' for i in range(5)]].sum(axis=1)
    features['top_sellers_active'] = features[[f'top_seller_{i}' for i in range(5)]].sum(axis=1)
    
    # 6. Trade Intensity
    # Contar trades por janela temporal
    df['time_bin'] = df['timestamp'].dt.floor('1min')
    trades_per_min = df.groupby('time_bin').size()
    df['trades_per_minute'] = df['time_bin'].map(trades_per_min)
    features['trade_intensity'] = df['trades_per_minute'].astype('float32')
    
    # 7. Features Temporais
    features['hour'] = df['timestamp'].dt.hour.astype('int8')
    features['minute'] = df['timestamp'].dt.minute.astype('int8')
    features['is_opening'] = ((features['hour'] == 9) & (features['minute'] < 30)).astype('int8')
    features['is_closing'] = (features['hour'] >= 16).astype('int8')
    
    # 8. Large Trade Indicator
    qty_p90 = df['<qty>'].quantile(0.9)
    features['is_large_trade'] = (df['<qty>'] > qty_p90).astype('int8')
    
    print(f"Features criadas: {features.shape[1]}")
    
    # Cleanup NaN
    features = features.ffill().fillna(0)
    
    # Converter para float32 para economia de memória
    for col in features.columns:
        if features[col].dtype == 'float64':
            features[col] = features[col].astype('float32')
    
    print(f"Memória das features: {features.memory_usage().sum() / 1024**2:.1f} MB")
    
    return features

def create_targets(df: pd.DataFrame, features: pd.DataFrame, horizon: int = 60) -> pd.Series:
    """Cria targets baseados em movimento de preço"""
    
    print(f"\n=== CRIANDO TARGETS (horizonte: {horizon} trades) ===")
    
    # Future return
    future_price = df['<price>'].shift(-horizon)
    returns = (future_price - df['<price>']) / df['<price>']
    
    # Usar desvio padrão para thresholds dinâmicos
    std_returns = returns.std()
    
    # 3 classes com thresholds baseados em desvio padrão
    targets = pd.Series(0, index=df.index, dtype='int8')  # 0 = Neutro
    targets[returns > 0.5 * std_returns] = 1   # Alta
    targets[returns < -0.5 * std_returns] = -1  # Baixa
    
    print(f"\nDistribuição do target:")
    print(targets.value_counts().sort_index())
    print(f"\nThresholds utilizados:")
    print(f"  Baixa: < {-0.5 * std_returns:.5f}")
    print(f"  Alta: > {0.5 * std_returns:.5f}")
    
    return targets

def train_models(features: pd.DataFrame, targets: pd.Series, output_dir: str):
    """Treina modelos de forma eficiente"""
    
    print("\n" + "=" * 80)
    print("TREINAMENTO RÁPIDO DOS MODELOS")
    print("=" * 80)
    
    # Preparar dados
    mask = ~targets.isna()
    X = features[mask]
    y = targets[mask]
    
    print(f"\nDados disponíveis: {len(X):,} registros")
    
    # Split temporal 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    models = {}
    results = {}
    
    # 1. Random Forest (rápido e robusto)
    print("\n=== Random Forest ===")
    rf = RandomForestClassifier(
        n_estimators=50,  # Menos árvores para velocidade
        max_depth=8,      # Profundidade limitada
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
    
    models['random_forest'] = rf
    results['random_forest'] = {
        'accuracy': (rf_pred == y_test).mean(),
        'predictions': rf_pred
    }
    
    # Feature importance
    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(feature_imp.head(10))
    
    # 2. XGBoost (opcional, mais lento)
    print("\n=== XGBoost ===")
    xgb_model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        tree_method='hist',  # Mais rápido
        n_jobs=-1,
        random_state=42
    )
    
    # XGBoost precisa de labels 0, 1, 2
    y_train_xgb = y_train + 1
    y_test_xgb = y_test + 1
    
    xgb_model.fit(X_train, y_train_xgb)
    xgb_pred = xgb_model.predict(X_test) - 1
    
    print("\nResultados XGBoost:")
    print(classification_report(y_test, xgb_pred, 
                              target_names=['Baixa', 'Neutro', 'Alta'],
                              zero_division=0))
    
    models['xgboost'] = xgb_model
    results['xgboost'] = {
        'accuracy': (xgb_pred == y_test).mean(),
        'predictions': xgb_pred
    }
    
    # Salvar modelos
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        model_file = output_path / f'csv_{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        joblib.dump(model, model_file)
        print(f"\n[OK] {name} salvo em: {model_file.name}")
    
    # Salvar metadados
    metadata = {
        'training_date': datetime.now().isoformat(),
        'sample_size': len(X),
        'features': list(X.columns),
        'feature_importance': feature_imp.head(20).to_dict('records'),
        'results': {
            name: {'accuracy': float(res['accuracy'])} 
            for name, res in results.items()
        }
    }
    
    metadata_file = output_path / f'training_metadata_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[OK] Metadados salvos em: {metadata_file.name}")
    
    return models, results, feature_imp

def main():
    """Pipeline principal otimizado"""
    
    print("PIPELINE RÁPIDO DE TREINAMENTO COM DADOS CSV")
    print("Otimizado para performance e memória\n")
    
    # Configurações
    csv_path = r"C:\Users\marth\Downloads\WDO_FUT\WDOFUT_BMF_T.csv"
    sample_size = 300000  # 300k para teste rápido
    
    # 1. Carregar dados
    df = load_csv_sample(csv_path, sample_size)
    
    # 2. Criar features
    features = create_fast_features(df)
    
    # 3. Criar targets
    targets = create_targets(df, features, horizon=60)
    
    # 4. Treinar modelos
    models, results, feature_importance = train_models(
        features, 
        targets,
        'models/csv_fast_models'
    )
    
    # Limpar memória
    del df
    gc.collect()
    
    print("\n" + "=" * 80)
    print("TREINAMENTO CONCLUÍDO!")
    print("=" * 80)
    
    print("\nResumo dos Resultados:")
    for name, res in results.items():
        print(f"  {name}: {res['accuracy']:.2%} accuracy")
    
    print("\nPróximos passos:")
    print("1. Aumentar sample_size para 1M+ registros")
    print("2. Adicionar mais features de microestrutura")
    print("3. Criar ensemble com Book Collector")
    print("4. Implementar backtesting")

if __name__ == "__main__":
    main()