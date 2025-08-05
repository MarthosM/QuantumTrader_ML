"""
Script para copiar a DLL ProfitDLL64.dll
"""

import shutil
from pathlib import Path

# Caminho de origem
source = Path(r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll")

# Caminho de destino (renomeando para ProfitDLL64.dll)
destination = Path("ProfitDLL64.dll")

try:
    if source.exists():
        shutil.copy2(source, destination)
        print(f"[OK] DLL copiada com sucesso!")
        print(f"De: {source}")
        print(f"Para: {destination.absolute()}")
    else:
        print(f"[ERRO] Arquivo não encontrado: {source}")
        print("\nVerifique se o caminho está correto.")
except Exception as e:
    print(f"[ERRO] Falha ao copiar: {e}")