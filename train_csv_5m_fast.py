"""
Pipeline Ultra-Otimizado para 5M Registros
Foco em velocidade e eficiência
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import xgboost as xgb
import joblib
import gc
import warnings
warnings.filterwarnings('ignore')

def load_5m_records_ultra_fast(csv_path: str):
    """Carrega 5M registros com máxima eficiência"""
    
    print("=" * 80)
    print("CARREGAMENTO ULTRA-RÁPIDO - 5 MILHÕES DE REGISTROS")
    print("=" * 80)
    
    # Apenas colunas essenciais
    usecols = ['<date>', '<time>', '<price>', '<qty>', '<vol>', 
               '<buy_agent>', '<sell_agent>', '<trade_type>']
    
    # Tipos mínimos
    dtypes = {
        '<price>': 'float32',
        '<qty>': 'uint16',
        '<vol>': 'float32'
    }
    
    print("\nCarregando 5,000,000 registros...")
    start_time = datetime.now()
    
    # Carregar direto sem chunks
    df = pd.read_csv(csv_path, 
                     nrows=5_000_000,
                     usecols=usecols,
                     dtype=dtypes,
                     low_memory=False)
    
    # Timestamp simplificado
    df['timestamp'] = pd.to_datetime(
        df['<date>'].astype(str) + df['<time>'].astype(str).str.zfill(6),
        format='%Y%m%d%H%M%S'
    )
    
    # Categoricals para economia de memória
    df['<buy_agent>'] = df['<buy_agent>'].astype('category')
    df['<sell_agent>'] = df['<sell_agent>'].astype('category')
    df['<trade_type>'] = df['<trade_type>'].astype('category')
    
    # Ordenar
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Remover colunas temporárias
    df = df.drop(['<date>', '<time>'], axis=1)
    
    load_time = (datetime.now() - start_time).total_seconds()
    print(f"Tempo: {load_time:.1f}s")
    print(f"Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
    
    # Análise rápida
    days = (df['timestamp'].max() - df['timestamp'].min()).days
    print(f"Dias de dados: {days}")
    
    return df

def create_core_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Cria apenas features core que realmente importam"""
    
    print("\n=== FEATURES CORE (RÁPIDAS) ===")
    features = pd.DataFrame(index=df.index)
    
    # 1. Retornos (mais importantes segundo análise anterior)
    print("1. Retornos...")
    price = df['<price>'].values
    features['price'] = price
    
    # Vetorizado para velocidade
    for period in [1, 5, 10, 20, 50]:
        features[f'returns_{period}'] = pd.Series(price).pct_change(period)
    
    # 2. Volatilidade básica
    print("2. Volatilidade...")
    features['volatility_20'] = features['returns_1'].rolling(20, min_periods=1).std()
    features['volatility_50'] = features['returns_1'].rolling(50, min_periods=1).std()
    
    # 3. Volume essencial
    print("3. Volume...")
    features['qty'] = df['<qty>'].astype('float32')
    features['qty_ma_50'] = features['qty'].rolling(50, min_periods=1).mean()
    features['volume_ratio'] = features['qty'] / features['qty_ma_50']
    
    # 4. Trade flow básico
    print("4. Trade flow...")
    features['is_buyer'] = (df['<trade_type>'] == 'AggressorBuyer').astype('int8')
    features['is_seller'] = (df['<trade_type>'] == 'AggressorSeller').astype('int8')
    
    # Imbalance com EWM (mais rápido)
    features['imbalance_50'] = (
        features['is_buyer'].ewm(span=50, adjust=False).mean() - 
        features['is_seller'].ewm(span=50, adjust=False).mean()
    )
    
    features['imbalance_200'] = (
        features['is_buyer'].ewm(span=200, adjust=False).mean() - 
        features['is_seller'].ewm(span=200, adjust=False).mean()
    )
    
    # 5. Top agents simplificado
    print("5. Top agents...")
    
    # Apenas top 5 de cada lado
    top_buyers = df['<buy_agent>'].value_counts().head(5).index
    top_sellers = df['<sell_agent>'].value_counts().head(5).index
    
    # Criar variáveis binárias
    for i, agent in enumerate(top_buyers):
        features[f'buyer_{i}'] = (df['<buy_agent>'] == agent).astype('int8')
    
    for i, agent in enumerate(top_sellers):
        features[f'seller_{i}'] = (df['<sell_agent>'] == agent).astype('int8')
    
    # Score agregado
    features['top_buyers_score'] = features[[f'buyer_{i}' for i in range(5)]].sum(axis=1)
    features['top_sellers_score'] = features[[f'seller_{i}' for i in range(5)]].sum(axis=1)
    
    # 6. Temporal básico
    print("6. Temporal...")
    features['hour'] = df['timestamp'].dt.hour.astype('int8')
    features['minute'] = df['timestamp'].dt.minute.astype('int8')
    
    # Large trades
    qty_p90 = df['<qty>'].quantile(0.9)
    features['large_trade'] = (df['<qty>'] > qty_p90).astype('int8')
    
    print(f"\nTotal features: {features.shape[1]}")
    
    # Cleanup rápido
    features = features.fillna(0)
    
    # Float32 para tudo
    for col in features.columns:
        if features[col].dtype == 'float64':
            features[col] = features[col].astype('float32')
    
    return features

def create_balanced_target(df: pd.DataFrame, horizon: int = 60) -> pd.Series:
    """Target balanceado para melhor performance"""
    
    print(f"\n=== TARGET BALANCEADO (horizonte: {horizon}) ===")
    
    # Future return
    returns = df['<price>'].pct_change(horizon).shift(-horizon)
    
    # Percentis para balance
    p33 = returns.quantile(0.33)
    p67 = returns.quantile(0.67)
    
    target = pd.Series(0, index=df.index, dtype='int8')
    target[returns < p33] = -1
    target[returns > p67] = 1
    
    print(f"Distribuição: {dict(target.value_counts())}")
    print(f"Limites: [{p33:.5f}, {p67:.5f}]")
    
    return target

def train_single_model_fast(features: pd.DataFrame, target: pd.Series):
    """Treina um único modelo otimizado"""
    
    print("\n" + "=" * 80)
    print("TREINAMENTO RÁPIDO - XGBOOST")
    print("=" * 80)
    
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
    
    # XGBoost otimizado
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        n_jobs=-1,
        eval_metric='mlogloss',
        random_state=42
    )
    
    # Labels 0,1,2 para XGBoost
    y_train_xgb = y_train + 1
    y_test_xgb = y_test + 1
    
    print("\nTreinando...")
    start = datetime.now()
    
    model.fit(
        X_train, y_train_xgb,
        eval_set=[(X_test, y_test_xgb)],
        verbose=False
    )
    
    train_time = (datetime.now() - start).total_seconds()
    print(f"Tempo de treino: {train_time:.1f}s")
    
    # Predict
    y_pred = model.predict(X_test) - 1
    
    # Métricas
    accuracy = (y_pred == y_test).mean()
    print(f"\nAccuracy: {accuracy:.2%}")
    
    # Por classe
    for class_val in [-1, 0, 1]:
        class_mask = y_test == class_val
        class_acc = (y_pred[class_mask] == class_val).mean() if class_mask.any() else 0
        print(f"Classe {class_val}: {class_acc:.2%}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== TOP 15 FEATURES ===")
    print(importance.head(15))
    
    return model, accuracy, importance

def save_5m_results(model, accuracy, importance):
    """Salva resultados do treino 5M"""
    
    output_dir = Path('models/csv_5m_fast')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar modelo
    model_file = output_dir / f'xgboost_5m_{timestamp}.pkl'
    joblib.dump(model, model_file)
    print(f"\n[OK] Modelo salvo: {model_file.name}")
    
    # Metadados
    metadata = {
        'training_date': datetime.now().isoformat(),
        'sample_size': 5_000_000,
        'accuracy': float(accuracy),
        'progression': {
            '300k': 0.46,
            '1M': 0.5143,
            '5M': float(accuracy)
        },
        'improvement_total': f"+{(accuracy - 0.46)*100:.1f}%",
        'top_features': importance.head(20).to_dict('records')
    }
    
    metadata_file = output_dir / f'metadata_5m_{timestamp}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Feature importance
    importance.to_csv(output_dir / f'features_5m_{timestamp}.csv', index=False)
    
    print(f"[OK] Metadados salvos")

def main():
    """Pipeline principal ultra-otimizado"""
    
    print("TREINAMENTO ULTRA-RÁPIDO - 5 MILHÕES DE REGISTROS")
    print("Foco: Velocidade e Eficiência\n")
    
    csv_path = r"C:\Users\marth\Downloads\WDO_FUT\WDOFUT_BMF_T.csv"
    
    # 1. Carregar
    df = load_5m_records_ultra_fast(csv_path)
    
    # 2. Features
    features = create_core_features_fast(df)
    
    # 3. Target
    target = create_balanced_target(df)
    
    # 4. Treinar
    model, accuracy, importance = train_single_model_fast(features, target)
    
    # 5. Salvar
    save_5m_results(model, accuracy, importance)
    
    # Cleanup
    del df, features, target
    gc.collect()
    
    print("\n" + "=" * 80)
    print("SUCESSO!")
    print("=" * 80)
    
    print(f"\nRESULTADO FINAL: {accuracy:.2%} accuracy")
    
    print("\nPROGRESSÃO:")
    print(f"  300k → 1M:  +5.43% (46.00% → 51.43%)")
    print(f"  1M → 5M:    +{(accuracy - 0.5143)*100:.2f}% (51.43% → {accuracy:.2%})")
    print(f"  TOTAL:      +{(accuracy - 0.46)*100:.2f}% (46.00% → {accuracy:.2%})")
    
    if accuracy > 0.55:
        print("\n🎉 META ALCANÇADA! Accuracy > 55%")
    
    print("\nPRÓXIMOS PASSOS:")
    print("1. Integrar com Book Collector")
    print("2. Implementar ensemble final")
    print("3. Backtest com dados reais")

if __name__ == "__main__":
    main()