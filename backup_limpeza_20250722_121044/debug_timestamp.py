#!/usr/bin/env python3
"""
Teste específico para debug da conversão de pandas Timestamp
"""

import pandas as pd
from datetime import datetime

def test_pandas_timestamp_conversion():
    """Testa a conversão específica de pandas Timestamp"""
    
    # Criar um pandas Timestamp
    ts = pd.Timestamp('2024-01-01 10:30:00')
    print(f"Original: {ts} (tipo: {type(ts)})")
    print(f"Has to_pydatetime: {hasattr(ts, 'to_pydatetime')}")
    
    if hasattr(ts, 'to_pydatetime'):
        converted = ts.to_pydatetime()
        print(f"Convertido: {converted} (tipo: {type(converted)})")
        print(f"É datetime Python: {isinstance(converted, datetime)}")
    
    # Testar com diferentes criações de Timestamp
    test_cases = [
        pd.Timestamp('2024-01-01 10:30:00'),
        pd.to_datetime('2024-01-01 10:30:00'),
        pd.date_range('2024-01-01', periods=1)[0]
    ]
    
    for i, ts in enumerate(test_cases):
        print(f"\nTeste {i+1}:")
        print(f"  Original: {ts} (tipo: {type(ts)})")
        print(f"  Has to_pydatetime: {hasattr(ts, 'to_pydatetime')}")
        
        if hasattr(ts, 'to_pydatetime'):
            converted = ts.to_pydatetime()
            print(f"  Convertido: {converted} (tipo: {type(converted)})")
            print(f"  É datetime Python: {isinstance(converted, datetime)}")

if __name__ == "__main__":
    test_pandas_timestamp_conversion()
