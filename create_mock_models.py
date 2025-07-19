"""
Criador de modelos mock para testes
"""

import os
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def create_mock_models(models_dir: str):
    """Cria modelos mock para teste do sistema"""
    
    # Criar diretório se não existir
    os.makedirs(models_dir, exist_ok=True)
    
    # Features padrão
    features = [
        'open', 'high', 'low', 'close', 'volume',
        'ema_5', 'ema_20', 'ema_50', 'ema_200',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper_20', 'bb_lower_20', 'bb_middle_20',
        'atr', 'adx',
        'momentum_5', 'momentum_10', 'momentum_20',
        'momentum_pct_5', 'momentum_pct_10', 'momentum_pct_20',
        'volatility_5', 'volatility_10', 'volatility_20', 'volatility_50',
        'return_5', 'return_10', 'return_20', 'return_50'
    ]
    
    # Dados sintéticos para treino
    X = np.random.random((100, len(features)))
    y = np.random.choice([0, 1], 100)
    
    # Modelo 1: Random Forest para Trend Detection
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_model.fit(X, y)
    
    # Salvar modelo
    rf_path = os.path.join(models_dir, 'trend_detector_rf.pkl')
    joblib.dump(rf_model, rf_path)
    
    # Salvar features
    rf_features_path = os.path.join(models_dir, 'trend_detector_rf_features.json')
    with open(rf_features_path, 'w') as f:
        json.dump(features, f)
    
    # Salvar metadata
    rf_meta_path = os.path.join(models_dir, 'trend_detector_rf_metadata.json')
    with open(rf_meta_path, 'w') as f:
        json.dump({
            'accuracy': 0.75,
            'training_date': '2025-07-18',
            'model_type': 'trend_detection',
            'target': 'trend_direction'
        }, f)
    
    # Modelo 2: Logistic Regression para Range Detection
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X, y)
    
    # Salvar modelo
    lr_path = os.path.join(models_dir, 'range_detector_lr.pkl')
    joblib.dump(lr_model, lr_path)
    
    # Salvar features
    lr_features_path = os.path.join(models_dir, 'range_detector_lr_features.json')
    with open(lr_features_path, 'w') as f:
        json.dump(features, f)
    
    # Salvar metadata
    lr_meta_path = os.path.join(models_dir, 'range_detector_lr_metadata.json')
    with open(lr_meta_path, 'w') as f:
        json.dump({
            'accuracy': 0.68,
            'training_date': '2025-07-18',
            'model_type': 'range_detection',
            'target': 'range_probability'
        }, f)
    
    print(f"✅ Modelos mock criados em: {models_dir}")
    print(f"📊 Modelo 1: trend_detector_rf.pkl ({len(features)} features)")
    print(f"📊 Modelo 2: range_detector_lr.pkl ({len(features)} features)")
    

if __name__ == '__main__':
    import sys
    models_dir = sys.argv[1] if len(sys.argv) > 1 else 'src/models'
    create_mock_models(models_dir)
