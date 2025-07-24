#!/usr/bin/env python3
"""
Teste específico para verificar o acesso ao DataFrame no stress_test_engine
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
from src.stress_test_engine import StressTestEngine

def test_dataframe_access():
    """Testa o acesso seguro a valores do DataFrame"""
    
    # Mock do backtester
    class MockBacktester:
        pass
    
    mock_backtester = MockBacktester()
    engine = StressTestEngine(mock_backtester)
    
    # Criar DataFrame de teste similar ao usado no trading
    dates = pd.date_range('2024-01-01', periods=5, freq='1H')
    test_data = pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [100.5, 101.5, 102.5, 103.5, 104.5],
        'low': [99.5, 100.5, 101.5, 102.5, 103.5],
        'close': [101.0, 102.0, 103.0, 104.0, 105.0],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }, index=dates)
    
    print("Testando acesso seguro ao DataFrame...")
    print(f"DataFrame shape: {test_data.shape}")
    print(f"DataFrame dtypes:\n{test_data.dtypes}")
    print()
    
    # Testar acesso aos valores usando .loc como no código original
    for i in range(len(test_data)):
        idx = test_data.index[i]
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            try:
                # Simular o acesso problemático do código original
                raw_value = test_data.loc[idx, col]
                converted_value = engine._safe_float_conversion(raw_value)
                
                print(f"Index {i}, Col '{col}': {type(raw_value).__name__} -> float({converted_value})")
                
            except Exception as e:
                print(f"Index {i}, Col '{col}': ERRO - {e}")
    
    print("\nTestando cenário de baixa liquidez...")
    
    try:
        # Executar o método que estava com problema
        result = engine._create_low_liquidity_scenario(test_data)
        print("✓ Cenário de baixa liquidez executado com sucesso")
        print(f"Resultado shape: {result.shape}")
        
        # Verificar se os valores fazem sentido
        original_close = engine._safe_float_conversion(test_data.iloc[-1]['close'])
        result_close = engine._safe_float_conversion(result.iloc[-1]['close'])
        print(f"Close original: {original_close}, Close resultado: {result_close}")
        
    except Exception as e:
        print(f"✗ Erro no cenário de baixa liquidez: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataframe_access()
