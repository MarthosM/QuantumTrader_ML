#!/usr/bin/env python
"""
SCRIPT PRINCIPAL DE PRODUCAO - BOOK_CLEAN
Sistema corrigido e testado - Pronto para uso
"""

import sys
import os

print("""
================================================================================
                    QUANTUM TRADER ML - PRODUCAO
================================================================================
Modelo: book_clean (79.23% Trading Accuracy)
Features: 14 completas
Latencia: < 50ms por predicao

ATENCAO: Este sistema executara trades reais!
================================================================================
""")

# Verificar confirmacao
response = input("Deseja iniciar o sistema de trading? (s/N): ")
if response.lower() != 's':
    print("Sistema cancelado.")
    sys.exit(0)

print("\nIniciando sistema...")

# Executar o sistema corrigido
os.system("python start_production_book_clean.py")