"""
Pipeline Otimizado para 1M de Registros
Versão com processamento em chunks e features simplificadas
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import gc
import warnings
warnings.filterwarnings('ignore')

def load_1m_records_fast(csv_path: str):
    """Carrega 1M registros de forma otimizada"""
    
    print("=" * 80)
    print("CARREGAMENTO RÁPIDO - 1 MILHÃO DE REGISTROS")
    print("=" * 80)
    
    # Colunas essenciais apenas
    usecols = ['<date>', '<time>', '<price>', '<qty>', '<vol>', 
               '<buy_agent>', '<sell_agent>', '<trade_type>']
    
    # Tipos otimizados
    dtypes = {
        '<price>': 'float32',
        '<qty>': 'uint16',
        '<vol>': 'float32',
        '<buy_agent>': 'category',
        '<sell_agent>': 'category',
        '<trade_type>': 'category'
    }
    
    print("\nCarregando 1,000,000 registros...")
    start_time = datetime.now()
    
    # Carregar dados
    df = pd.read_csv(csv_path, 
                     nrows=1_000_000,
                     usecols=usecols,
                     dtype=dtypes)
    
    # Criar timestamp simples
    df['timestamp'] = pd.to_datetime(
        df['<date>'].astype(str) + ' ' + df['<time>'].astype(str).str.zfill(6),
        format='%Y%m%d %H%M%S'
    )
    
    # Remover colunas não necessárias
    df = df.drop(['<date>', '<time>'], axis=1)
    
    # Ordenar por timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    load_time = (datetime.now() - start_time).total_seconds()
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    
    print(f"Tempo de carregamento: {load_time:.1f}s")
    print(f"Memória utilizada: {memory_mb:.1f} MB")
    print(f"Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
    
    return df

def create_essential_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria apenas features essenciais para velocidade"""
    
    print("\n=== CRIAÇÃO DE FEATURES ESSENCIAIS ===")
    features = pd.DataFrame(index=df.index)
    
    # 1. Preço e retornos
    print("1. Features de preço...")
    features['price'] = df['<price>']
    features['returns_1'] = features['price'].pct_change(1)
    features['returns_5'] = features['price'].pct_change(5)
    features['returns_20'] = features['price'].pct_change(20)
    features['returns_50'] = features['price'].pct_change(50)
    
    # Volatilidade simples
    features['volatility_20'] = features['returns_1'].rolling(20).std()
    features['volatility_50'] = features['returns_1'].rolling(50).std()
    
    # 2. Volume
    print("2. Features de volume...")
    features['qty'] = df['<qty>'].astype('float32')
    features['qty_ma_50'] = features['qty'].rolling(50).mean()
    features['volume_ratio'] = features['qty'] / features['qty_ma_50']
    
    # VWAP simples
    cumvol = df['<vol>'].cumsum()
    cumqty = df['<qty>'].cumsum()
    features['vwap'] = (cumvol / cumqty).astype('float32')
    features['price_to_vwap'] = features['price'] / features['vwap'] - 1
    
    # 3. Trade Flow
    print("3. Features de trade flow...")
    features['is_buyer_aggressor'] = (df['<trade_type>'] == 'AggressorBuyer').astype('int8')
    features['is_seller_aggressor'] = (df['<trade_type>'] == 'AggressorSeller').astype('int8')
    
    # Imbalance simples
    features['aggressor_imbalance_50'] = (
        features['is_buyer_aggressor'].rolling(50).sum() - 
        features['is_seller_aggressor'].rolling(50).sum()
    ) / 50
    
    features['aggressor_imbalance_200'] = (
        features['is_buyer_aggressor'].rolling(200).sum() - 
        features['is_seller_aggressor'].rolling(200).sum()
    ) / 200
    
    # 4. Agent Features Simples
    print("4. Features de agentes...")
    
    # Top 10 agentes
    top_buyers = df['<buy_agent>'].value_counts().head(10).index
    top_sellers = df['<sell_agent>'].value_counts().head(10).index
    
    # Criar dummies apenas para top agents
    for i, agent in enumerate(top_buyers[:5]):
        features[f'top_buyer_{i}'] = (df['<buy_agent>'] == agent).astype('int8')
    
    for i, agent in enumerate(top_sellers[:5]):
        features[f'top_seller_{i}'] = (df['<sell_agent>'] == agent).astype('int8')
    
    # Activity score dos top agents
    features['top_buyers_activity'] = features[[f'top_buyer_{i}' for i in range(5)]].sum(axis=1)
    features['top_sellers_activity'] = features[[f'top_seller_{i}' for i in range(5)]].sum(axis=1)
    
    # 5. Features temporais básicas
    print("5. Features temporais...")
    features['hour'] = df['timestamp'].dt.hour.astype('int8')
    features['minute'] = df['timestamp'].dt.minute.astype('int8')
    
    # Trade size
    qty_p90 = df['<qty>'].quantile(0.9)
    features['is_large_trade'] = (df['<qty>'] > qty_p90).astype('int8')
    features['large_trade_ratio'] = features['is_large_trade'].rolling(100).mean()
    
    print(f"\nTotal features: {features.shape[1]}")
    
    # Cleanup
    features = features.ffill().fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    
    # Garantir float32
    for col in features.columns:
        if features[col].dtype == 'float64':
            features[col] = features[col].astype('float32')
    
    return features

def create_target(df: pd.DataFrame, horizon: int = 60) -> pd.Series:
    """Cria target simples e eficiente"""
    
    print(f"\n=== CRIANDO TARGET (horizonte: {horizon} trades) ===")
    
    # Future return
    future_price = df['<price>'].shift(-horizon)
    returns = (future_price - df['<price>']) / df['<price>']
    
    # Usar percentis para thresholds balanceados
    upper = returns.quantile(0.67)
    lower = returns.quantile(0.33)
    
    target = pd.Series(0, index=df.index, dtype='int8')
    target[returns > upper] = 1
    target[returns < lower] = -1
    
    print(f"\nDistribuição do target:")
    print(target.value_counts().sort_index())
    print(f"\nThresholds: Baixa < {lower:.5f} < Neutro < {upper:.5f} < Alta")
    
    return target

def train_fast_models(features: pd.DataFrame, target: pd.Series):
    """Treina modelos de forma eficiente"""
    
    print("\n" + "=" * 80)
    print("TREINAMENTO RÁPIDO DE MODELOS")
    print("=" * 80)
    
    # Preparar dados
    mask = ~target.isna()
    X = features[mask]
    y = target[mask]
    
    print(f"\nDados disponíveis: {len(X):,} registros")
    
    # Split 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    models = {}
    results = {}
    
    # 1. XGBoost (principal)
    print("\n=== XGBoost ===")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',  # Mais rápido
        n_jobs=-1,
        random_state=42
    )
    
    # XGBoost precisa labels 0, 1, 2
    y_train_xgb = y_train + 1
    y_test_xgb = y_test + 1
    
    print("Treinando XGBoost...")
    xgb_model.fit(X_train, y_train_xgb)
    xgb_pred = xgb_model.predict(X_test) - 1
    
    xgb_accuracy = (xgb_pred == y_test).mean()
    print(f"Accuracy: {xgb_accuracy:.2%}")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, xgb_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    models['xgboost'] = xgb_model
    results['xgboost'] = xgb_accuracy
    
    # 2. Random Forest (backup)
    print("\n=== Random Forest ===")
    rf = RandomForestClassifier(
        n_estimators=50,  # Menos árvores
        max_depth=8,
        min_samples_split=200,
        min_samples_leaf=100,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    
    print("Treinando Random Forest...")
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    rf_accuracy = (rf_pred == y_test).mean()
    print(f"Accuracy: {rf_accuracy:.2%}")
    
    models['random_forest'] = rf
    results['random_forest'] = rf_accuracy
    
    # Feature importance
    print("\n=== TOP 10 FEATURES ===")
    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'xgb_importance': xgb_model.feature_importances_,
        'rf_importance': rf.feature_importances_
    })
    
    feature_imp['avg_importance'] = (feature_imp['xgb_importance'] + feature_imp['rf_importance']) / 2
    feature_imp = feature_imp.sort_values('avg_importance', ascending=False)
    
    print(feature_imp[['feature', 'avg_importance']].head(10))
    
    return models, results, feature_imp

def save_results(models, results, feature_importance):
    """Salva modelos e resultados"""
    
    output_dir = Path('models/csv_1m_fast')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n=== SALVANDO MODELOS ===")
    
    # Salvar modelos
    for name, model in models.items():
        model_file = output_dir / f'{name}_1m_{timestamp}.pkl'
        joblib.dump(model, model_file)
        print(f"  {name} salvo")
    
    # Salvar metadados
    metadata = {
        'training_date': datetime.now().isoformat(),
        'sample_size': 1_000_000,
        'results': {name: float(acc) for name, acc in results.items()},
        'top_features': feature_importance.head(15).to_dict('records')
    }
    
    metadata_file = output_dir / f'metadata_1m_{timestamp}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Salvar feature importance
    feature_importance.to_csv(output_dir / f'features_1m_{timestamp}.csv', index=False)
    
    print(f"\n[OK] Resultados salvos em: {output_dir}")

def main():
    """Pipeline principal otimizado"""
    
    print("TREINAMENTO OTIMIZADO - 1 MILHÃO DE REGISTROS")
    print("Versão rápida com features essenciais\n")
    
    csv_path = r"C:\Users\marth\Downloads\WDO_FUT\WDOFUT_BMF_T.csv"
    
    # 1. Carregar dados
    df = load_1m_records_fast(csv_path)
    
    # 2. Criar features
    features = create_essential_features(df)
    
    # 3. Criar target
    target = create_target(df)
    
    # 4. Treinar modelos
    models, results, feature_importance = train_fast_models(features, target)
    
    # 5. Salvar resultados
    save_results(models, results, feature_importance)
    
    # Limpar memória
    del df, features, target
    gc.collect()
    
    print("\n" + "=" * 80)
    print("TREINAMENTO CONCLUÍDO!")
    print("=" * 80)
    
    print("\nRESUMO:")
    for name, accuracy in results.items():
        print(f"  {name}: {accuracy:.2%}")
    
    print("\nMELHORIA ESPERADA:")
    print("  - De 46% (300k registros) para ~50%+ (1M registros)")
    print("  - Melhor generalização com mais dados")
    print("  - Features de agentes mais representativas")

if __name__ == "__main__":
    main()