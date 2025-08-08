"""
Script para reiniciar o sistema com todas as correções
"""

import subprocess
import os
import time
from datetime import datetime

print("\n" + "="*60)
print("REINICIANDO SISTEMA QUANTUMTRADER ML")
print("="*60)
print(f"Hora: {datetime.now()}")
print("="*60)

# 1. Matar processos existentes
print("\n1. Finalizando processos existentes...")
subprocess.run("taskkill /F /IM python.exe", shell=True, capture_output=True)
time.sleep(2)

# 2. Configurar ambiente
print("\n2. Configurando ambiente...")
os.environ['TICKER'] = 'WDOU25'
print("   ✓ Ticker: WDOU25")

# 3. Iniciar sistema de produção
print("\n3. Iniciando sistema de produção...")
prod_process = subprocess.Popen(
    ["python", "production_fixed.py"],
    creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
)
print("   ✓ Sistema de produção iniciado")

# 4. Aguardar sistema estabilizar
print("\n4. Aguardando sistema estabilizar...")
time.sleep(10)

# 5. Iniciar monitor v2
print("\n5. Iniciando monitor GUI v2...")
monitor_process = subprocess.Popen(
    ["python", "monitor_gui_v2.py"],
    creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
)
print("   ✓ Monitor GUI v2 iniciado")

print("\n" + "="*60)
print("SISTEMA COMPLETO INICIADO!")
print("="*60)
print("\n✓ Sistema de produção rodando com WDOU25")
print("✓ Monitor GUI v2 mostrando dados em tempo real")
print("\nPara parar: Feche as janelas ou CTRL+C em cada console")
print("="*60)

try:
    # Manter script rodando
    while True:
        time.sleep(60)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sistema rodando...")
        
except KeyboardInterrupt:
    print("\n\nFinalizando sistema...")
    try:
        prod_process.terminate()
        monitor_process.terminate()
    except:
        pass
    print("Sistema finalizado.")