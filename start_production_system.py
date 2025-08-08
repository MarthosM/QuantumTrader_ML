"""
Script para iniciar o sistema de produção com o ticker correto
"""

import subprocess
import os
from datetime import datetime

print("\n" + "="*60)
print("INICIANDO SISTEMA DE PRODUÇÃO - QUANTUMTRADER ML")
print("="*60)
print(f"Hora: {datetime.now()}")
print("="*60)

# Configurar ticker correto
os.environ['TICKER'] = 'WDOU25'

print(f"\n✓ Ticker configurado: WDOU25 (Setembro)")
print("\nIniciando sistema de produção...")
print("Para parar: CTRL+C")
print("\n" + "="*60)

try:
    # Executar sistema de produção
    subprocess.run(["python", "production_fixed.py"])
except KeyboardInterrupt:
    print("\n\nSistema finalizado pelo usuário.")
except Exception as e:
    print(f"\nErro ao executar: {e}")