#!/usr/bin/env python3
"""
Teste para verificar a conversão de timestamps no ml_backtester.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from datetime import datetime
from src.ml_backtester import AdvancedMLBacktester, BacktestConfig, BacktestMode

def test_timestamp_conversion():
    """Testa a conversão de diferentes tipos de timestamp"""
    
    # Criar configuração simples
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),
        mode=BacktestMode.SIMPLE
    )
    
    # Criar backtester
    backtester = AdvancedMLBacktester(config)
    
    # Testar diferentes tipos de timestamp
    test_cases = [
        datetime(2024, 1, 1, 10, 30, 0),  # datetime nativo
        pd.Timestamp('2024-01-01 10:30:00'),  # pd.Timestamp
        '2024-01-01 10:30:00',  # string
        pd.to_datetime('2024-01-01 10:30:00'),  # resultado de pd.to_datetime
    ]
    
    print("Testando conversão de timestamps...")
    
    for i, timestamp in enumerate(test_cases):
        try:
            result = backtester._ensure_datetime(timestamp)
            print(f"Teste {i+1}: {type(timestamp).__name__} -> {type(result).__name__}")
            print(f"  Input:  {timestamp}")
            print(f"  Output: {result}")
            print(f"  É datetime Python: {isinstance(result, datetime)}")
            print(f"  ✓ Sucesso\n")
        except Exception as e:
            print(f"Teste {i+1}: {type(timestamp).__name__} -> ERRO")
            print(f"  Input:  {timestamp}")
            print(f"  Erro:   {e}")
            print(f"  ✗ Falhou\n")
    
    # Testar com DataFrame
    print("Testando com DataFrame iterrows()...")
    
    # Criar DataFrame com diferentes tipos de index
    dates = pd.date_range('2024-01-01 09:00:00', periods=3, freq='1H')
    df = pd.DataFrame({
        'close': [100.0, 101.0, 102.0],
        'volume': [1000, 1100, 1200]
    }, index=dates)
    
    for timestamp, data in df.iterrows():
        try:
            converted = backtester._ensure_datetime(timestamp)
            print(f"DataFrame iterrows: {type(timestamp).__name__} -> {type(converted).__name__}")
            print(f"  Original: {timestamp}")
            print(f"  Convertido: {converted}")
            print(f"  É datetime Python: {isinstance(converted, datetime)}")
            print(f"  ✓ Sucesso\n")
        except Exception as e:
            print(f"DataFrame iterrows: ERRO - {e}\n")

if __name__ == "__main__":
    test_timestamp_conversion()
