#!/usr/bin/env python3
"""
Teste simples de imports para diagnosticar problemas
"""

import sys
import os
import traceback

print("🔍 Iniciando teste de imports...")

# Adicionar src ao path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f"✅ Path atualizado: {src_path}")

# Teste 1: Imports básicos
try:
    import pandas as pd
    import numpy as np
    print("✅ Pandas e NumPy importados")
except Exception as e:
    print(f"❌ Erro imports básicos: {e}")

# Teste 2: Import do TradingSystem
try:
    print("🔄 Tentando importar TradingSystem...")
    from trading_system import TradingSystem
    print("✅ TradingSystem importado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao importar TradingSystem: {e}")
    print("📋 Traceback completo:")
    traceback.print_exc()

print("🏁 Teste de imports finalizado")
