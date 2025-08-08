"""
Script para iniciar o monitor web
"""

import webbrowser
import time
import subprocess
import sys

print("\n" + "="*60)
print("INICIANDO MONITOR WEB - QUANTUM TRADER ML")
print("="*60)

# Iniciar servidor Flask
print("\nIniciando servidor web...")
process = subprocess.Popen([sys.executable, "monitor_web.py"])

# Aguardar servidor iniciar
print("Aguardando servidor iniciar...")
time.sleep(3)

# Abrir navegador
url = "http://localhost:5000"
print(f"\nAbrindo navegador em: {url}")
webbrowser.open(url)

print("\nMonitor web iniciado!")
print("Para parar: CTRL+C")
print("="*60)

try:
    # Manter processo rodando
    process.wait()
except KeyboardInterrupt:
    print("\nParando monitor...")
    process.terminate()