#!/usr/bin/env python3
"""
Teste para verificar a função _safe_float_conversion no stress_test_engine.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from src.stress_test_engine import StressTestEngine

def test_safe_float_conversion():
    """Testa a conversão segura de diferentes tipos para float"""
    
    # Mock do backtester para criar o StressTestEngine
    class MockBacktester:
        pass
    
    mock_backtester = MockBacktester()
    engine = StressTestEngine(mock_backtester)
    
    # Testar diferentes tipos de entrada
    test_cases = [
        (42.5, 42.5),                                    # float normal
        (42, 42.0),                                      # int
        ("42.5", 42.5),                                  # string
        (np.float64(42.5), 42.5),                        # numpy float64
        (np.int32(42), 42.0),                            # numpy int32
        (pd.Series([42.5]).iloc[0], 42.5),               # pandas Series element
        (np.array([42.5]).item(), 42.5),                 # numpy array item
    ]
    
    print("Testando conversão segura para float...")
    
    for i, (input_val, expected) in enumerate(test_cases):
        try:
            result = engine._safe_float_conversion(input_val)
            print(f"Teste {i+1}: {type(input_val).__name__} -> float")
            print(f"  Input:    {input_val}")
            print(f"  Esperado: {expected}")
            print(f"  Resultado: {result}")
            print(f"  Tipo resultado: {type(result)}")
            
            if abs(result - expected) < 1e-10:  # Tolerância para floats
                print(f"  ✓ Sucesso\n")
            else:
                print(f"  ✗ Falhou - valores diferentes\n")
                
        except Exception as e:
            print(f"Teste {i+1}: {type(input_val).__name__} -> ERRO")
            print(f"  Input: {input_val}")
            print(f"  Erro:  {e}")
            print(f"  ✗ Falhou\n")
    
    # Testar casos extremos
    print("Testando casos extremos...")
    
    extreme_cases = [
        (None, "None"),
        (complex(1, 2), "complex"),
        ([], "lista vazia"),
        ("não é número", "string inválida"),
    ]
    
    for input_val, description in extreme_cases:
        try:
            result = engine._safe_float_conversion(input_val)
            print(f"Caso extremo ({description}): {result}")
            if result == 0.0:
                print(f"  ✓ Fallback correto para 0.0\n")
            else:
                print(f"  ? Resultado inesperado: {result}\n")
        except Exception as e:
            print(f"Caso extremo ({description}): ERRO - {e}\n")

if __name__ == "__main__":
    test_safe_float_conversion()
