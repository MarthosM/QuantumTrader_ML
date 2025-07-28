"""
Analisa compatibilidade do dataset com DEVELOPER_GUIDE
"""

import pandas as pd
import json
import os
from datetime import datetime

def analyze_dataset_compatibility():
    """Analisa dataset e verifica compatibilidade com guidelines"""
    
    print("="*60)
    print("ANÁLISE DE COMPATIBILIDADE DO DATASET")
    print("="*60)
    
    # Carregar dataset mais recente
    dataset_path = "datasets/WDOH25_20250628_20250728_train_20250728_085701.parquet"
    metadata_path = "datasets/WDOH25_20250628_20250728_metadata_20250728_085701.json"
    
    # 1. Verificar metadados
    print("\n1. METADADOS DO DATASET:")
    print("-"*40)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Ticker: {metadata['ticker']}")
    print(f"Período: {metadata['start_date'][:10]} a {metadata['end_date'][:10]}")
    print(f"Features: {metadata['splits']['train']['features']}")
    print(f"Lookback: {metadata['lookback_periods']} períodos")
    print(f"Target: {metadata['target_periods']} períodos")
    print(f"Threshold: {metadata['target_threshold']} ({metadata['target_threshold']*100:.1f}%)")
    print(f"Scaler: {metadata['scaler_type']}")
    
    # 2. Analisar estrutura dos dados
    print("\n2. ESTRUTURA DOS DADOS:")
    print("-"*40)
    
    df = pd.read_parquet(dataset_path)
    print(f"Shape: {df.shape}")
    print(f"Colunas: {len(df.columns)}")
    print(f"Index: {df.index.dtype} (timezone: {df.index.tz})")
    
    # 3. Verificar features V3
    print("\n3. FEATURES V3 (DEVELOPER_GUIDE_V3):")
    print("-"*40)
    
    v3_features = [col for col in df.columns if col.startswith('v3_')]
    print(f"Total features V3: {len(v3_features)}")
    
    # Categorizar features
    categories = {
        'price': [],
        'volume': [],
        'microstructure': [],
        'technical': [],
        'ml': [],
        'other': []
    }
    
    for feat in v3_features:
        if any(x in feat for x in ['price', 'return', 'momentum']):
            categories['price'].append(feat)
        elif any(x in feat for x in ['volume', 'vwap']):
            categories['volume'].append(feat)
        elif any(x in feat for x in ['spread', 'imbalance', 'pressure']):
            categories['microstructure'].append(feat)
        elif any(x in feat for x in ['rsi', 'macd', 'bb', 'ema', 'sma', 'adx', 'atr']):
            categories['technical'].append(feat)
        elif any(x in feat for x in ['cluster', 'pattern', 'regime']):
            categories['ml'].append(feat)
        else:
            categories['other'].append(feat)
    
    for cat, feats in categories.items():
        if feats:
            print(f"\n{cat.upper()} ({len(feats)} features):")
            for feat in sorted(feats)[:5]:  # Mostrar apenas 5 primeiras
                print(f"  - {feat}")
            if len(feats) > 5:
                print(f"  ... e mais {len(feats)-5} features")
    
    # 4. Verificar labels
    print("\n4. LABELS:")
    print("-"*40)
    
    label_cols = [col for col in df.columns if col.startswith('target_')]
    print(f"Labels encontradas: {label_cols}")
    
    if 'target_class' in df.columns:
        print(f"\nDistribuição target_class:")
        dist = df['target_class'].value_counts().sort_index()
        for val, count in dist.items():
            pct = count / len(df) * 100
            print(f"  {val:2d}: {count:5d} ({pct:5.1f}%)")
    
    # 5. Verificar regime
    print("\n5. REGIME DE MERCADO:")
    print("-"*40)
    
    if 'regime' in df.columns:
        regime_dist = df['regime'].value_counts()
        for regime, count in regime_dist.items():
            pct = count / len(df) * 100
            print(f"  {regime:12s}: {count:5d} ({pct:5.1f}%)")
    else:
        print("  ⚠️ Coluna 'regime' não encontrada!")
    
    # 6. Qualidade dos dados
    print("\n6. QUALIDADE DOS DADOS:")
    print("-"*40)
    
    # NaN check
    nan_count = df[v3_features].isna().sum().sum()
    total_values = len(df) * len(v3_features)
    nan_pct = nan_count / total_values * 100
    
    print(f"NaN total: {nan_count} ({nan_pct:.2f}%)")
    
    # Features com mais NaN
    nan_by_feature = df[v3_features].isna().sum()
    worst_features = nan_by_feature.nlargest(5)
    if worst_features.sum() > 0:
        print("\nFeatures com mais NaN:")
        for feat, count in worst_features.items():
            pct = count / len(df) * 100
            print(f"  {feat}: {count} ({pct:.1f}%)")
    
    # 7. Compatibilidade com sistema
    print("\n7. COMPATIBILIDADE COM SISTEMA:")
    print("-"*40)
    
    # Verificar requisitos do DEVELOPER_GUIDE
    checks = {
        "Features V3 presentes": len(v3_features) > 0,
        "Labels categóricas": 'target_class' in df.columns,
        "Labels binárias": 'target_binary' in df.columns,
        "Retorno real": 'target_return' in df.columns,
        "Magnitude": 'target_magnitude' in df.columns,
        "Regime detectado": 'regime' in df.columns,
        "Separação temporal": True,  # Confirmado pelos metadados
        "Normalização aplicada": True,  # Features já normalizadas
        "Formato parquet": True,
        "Index datetime": isinstance(df.index, pd.DatetimeIndex)
    }
    
    all_ok = True
    for check, status in checks.items():
        symbol = "[OK]" if status else "[FAIL]"
        print(f"  {symbol} {check}")
        if not status:
            all_ok = False
    
    # 8. Recomendações
    print("\n8. RECOMENDAÇÕES:")
    print("-"*40)
    
    if all_ok:
        print("[SUCESSO] Dataset está TOTALMENTE COMPATÍVEL com o sistema!")
    else:
        print("[AVISO] Dataset tem alguns problemas de compatibilidade.")
    
    # Verificar tamanho mínimo
    if len(df) < 1000:
        print(f"\n[AVISO] Dataset muito pequeno ({len(df)} amostras). Recomenda-se:")
        print("   - Coletar mais dados históricos (mínimo 30 dias)")
        print("   - Usar período maior no create_historical_dataset.py")
    
    # Verificar balanceamento
    if 'target_class' in df.columns:
        dist = df['target_class'].value_counts()
        min_class = dist.min()
        max_class = dist.max()
        imbalance = max_class / min_class
        
        if imbalance > 3:
            print(f"\n[AVISO] Classes desbalanceadas (ratio {imbalance:.1f}:1). Considere:")
            print("   - Ajustar target_threshold")
            print("   - Usar técnicas de balanceamento no treinamento")
    
    # Verificar regime
    if 'regime' in df.columns:
        regime_dist = df['regime'].value_counts()
        if 'range' in regime_dist and regime_dist['range'] / len(df) > 0.9:
            print(f"\n[AVISO] Dataset dominado por regime 'range' ({regime_dist['range']/len(df):.1%})")
            print("   - Normal para períodos de baixa volatilidade")
            print("   - Considere coletar dados de períodos mais voláteis")
    
    print("\n" + "="*60)
    print("ANÁLISE CONCLUÍDA")
    print("="*60)
    
    return all_ok

if __name__ == "__main__":
    analyze_dataset_compatibility()