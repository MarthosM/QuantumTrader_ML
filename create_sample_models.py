"""
Script para criar modelos ML de exemplo para teste
"""

import os
import joblib
import json
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Diretório dos modelos
models_dir = os.path.join(os.path.dirname(__file__), 'src', 'training', 'models')
os.makedirs(models_dir, exist_ok=True)

# Features de exemplo baseadas no sistema
sample_features = [
    "ema_diff", "volume_ratio_10", "high_low_range_20", "momentum",
    "range_percent", "momentum_1", "volume_ratio_50", "momentum_10",
    "momentum_5", "volume_ratio_5", "high_low_range_50", "return_20",
    "bb_width_10", "high_low_range_5", "adx_substitute", "return_50",
    "return_5", "rsi", "momentum_3", "high_low_range_10", "adx",
    "volume_ratio_20", "ema_diff_fast", "bb_width_20", 
    "ichimoku_conversion_line", "momentum_15", "return_10", "bb_width",
    "volume_ratio", "momentum_20", "volatility_ratio"
]

# Criar dados de exemplo
n_samples = 1000
n_features = len(sample_features)
X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, 3, n_samples)  # 0: SELL, 1: HOLD, 2: BUY

print(f"Criando modelos de exemplo em: {models_dir}")

# 1. Random Forest
print("\n1. Criando Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Salvar modelo
model_path = os.path.join(models_dir, 'random_forest_model.pkl')
joblib.dump(rf_model, model_path)

# Salvar features
features_path = os.path.join(models_dir, 'random_forest_model_features.json')
with open(features_path, 'w') as f:
    json.dump(sample_features, f)

# Salvar metadata
metadata = {
    'model_type': 'RandomForestClassifier',
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_features': n_features,
    'n_samples': n_samples,
    'best_score': 0.75,
    'feature_importance': dict(zip(sample_features, rf_model.feature_importances_))
}
metadata_path = os.path.join(models_dir, 'random_forest_model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Random Forest salvo em: {model_path}")

# 2. LightGBM
print("\n2. Criando LightGBM...")
lgb_model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
lgb_model.fit(X, y)

# Salvar modelo
model_path = os.path.join(models_dir, 'lightgbm_model.pkl')
joblib.dump(lgb_model, model_path)

# Salvar features
features_path = os.path.join(models_dir, 'lightgbm_model_features.json')
with open(features_path, 'w') as f:
    json.dump(sample_features, f)

# Salvar metadata
metadata = {
    'model_type': 'LGBMClassifier',
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_features': n_features,
    'n_samples': n_samples,
    'best_score': 0.78,
    'feature_importance': dict(zip(sample_features, lgb_model.feature_importances_))
}
metadata_path = os.path.join(models_dir, 'lightgbm_model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ LightGBM salvo em: {model_path}")

# 3. XGBoost
print("\n3. Criando XGBoost...")
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X, y)

# Salvar modelo
model_path = os.path.join(models_dir, 'xgboost_model.pkl')
joblib.dump(xgb_model, model_path)

# Salvar features
features_path = os.path.join(models_dir, 'xgboost_model_features.json')
with open(features_path, 'w') as f:
    json.dump(sample_features, f)

# Salvar metadata
metadata = {
    'model_type': 'XGBClassifier',
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_features': n_features,
    'n_samples': n_samples,
    'best_score': 0.80,
    'feature_importance': dict(zip(sample_features, xgb_model.feature_importances_))
}
metadata_path = os.path.join(models_dir, 'xgboost_model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ XGBoost salvo em: {model_path}")

# Criar arquivo de features requeridas
all_features_path = os.path.join(models_dir, 'all_required_features.json')
with open(all_features_path, 'w') as f:
    json.dump(sample_features, f, indent=2)

print(f"\n✅ Arquivo de features salvo em: {all_features_path}")

print(f"\n🎉 {3} modelos criados com sucesso!")
print(f"📁 Diretório: {models_dir}")
print(f"📋 Features: {len(sample_features)} features por modelo")