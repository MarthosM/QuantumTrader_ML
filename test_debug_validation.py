"""
Debug da validação para entender por que está falhando com 100% cobertura
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def debug_validation():
    print("DEBUG: Validacao de Features")
    print("=" * 50)
    
    os.environ['TRADING_ENV'] = 'development'
    
    from src.feature_validator import FeatureValidator
    from src.technical_indicators import TechnicalIndicators
    from src.ml_features import MLFeatures
    
    # Criar dados pequenos
    dates = pd.date_range(start=datetime.now() - timedelta(hours=2), end=datetime.now(), freq='1min')
    np.random.seed(42)
    
    data = []
    base_price = 130000
    for i, date in enumerate(dates):
        price = base_price + i * 10
        data.append({
            'open': price,
            'high': price + 50,
            'low': price - 30,
            'close': price + 20,
            'volume': 100
        })
    
    candles_df = pd.DataFrame(data, index=dates)
    
    # Calcular features
    tech_indicators = TechnicalIndicators()
    ml_features = MLFeatures()
    
    indicators = tech_indicators.calculate_all(candles_df)
    
    microstructure = pd.DataFrame(index=candles_df.index)
    microstructure['buy_volume'] = candles_df['volume'] * 0.5
    microstructure['sell_volume'] = candles_df['volume'] * 0.5
    microstructure['buy_trades'] = 10
    microstructure['sell_trades'] = 10
    microstructure['imbalance'] = 0
    
    features = ml_features.calculate_all(candles_df, microstructure, indicators)
    all_features = pd.concat([candles_df, indicators, features], axis=1)
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]
    
    # Debug validação
    validator = FeatureValidator()
    
    for model_name in ['fallback_model', 'ensemble_production']:
        print(f"\nTESTE: {model_name}")
        print("-" * 30)
        
        is_valid, result = validator.validate_dataframe(all_features, model_name)
        
        print(f"is_valid: {is_valid}")
        print(f"coverage: {result.get('coverage_percentage', 0):.1f}%")
        print(f"missing: {result.get('missing_features', [])}")
        
        # Debug qualidade
        quality = result.get('quality_validation', {})
        print(f"quality_valid: {quality.get('is_valid', 'N/A')}")
        print(f"quality_issues: {quality.get('issues', [])}")
        
        # Debug dataframe
        df_validation = result.get('dataframe_validation', {})
        print(f"has_nan: {df_validation.get('has_nan', 'N/A')}")
        print(f"nan_count: {df_validation.get('nan_count', 'N/A')}")
        
        # Debug overall
        print(f"overall_valid: {result.get('overall_valid', 'N/A')}")

if __name__ == "__main__":
    debug_validation()