#!/usr/bin/env python
"""
SISTEMA HMARL COM SHUTDOWN RAPIDO
Corrigido para nao travar ao pressionar Ctrl+C
"""

import os
import sys

print("""
================================================================================
         QUANTUM TRADER ML - HMARL PRODUCAO (SHUTDOWN RAPIDO)
================================================================================
Sistema: HMARL + Captura de Dados + Enhanced Monitor
Modelo: book_clean (79.23% accuracy)

MELHORIAS IMPLEMENTADAS:
- Ctrl+C finaliza em menos de 2 segundos
- Nao trava em "Finalizando agentes HMARL"
- Cleanup otimizado com timeouts curtos
- Threads daemon para morte automatica

PARA PARAR: Pressione Ctrl+C (finaliza rapidamente)
================================================================================
""")

response = input("Iniciar sistema? (s/N): ")
if response.lower() == 's':
    print("\nIniciando sistema com shutdown rapido...")
    print("Pressione Ctrl+C para parar rapidamente\n")
    os.system("python start_hmarl_production_with_capture.py")
else:
    print("Cancelado.")
    sys.exit(0)