#!/usr/bin/env python
"""
SISTEMA HMARL COM APENAS ENHANCED MONITOR
Monitor colorido com informacoes completas
"""

import sys
import os
import time
import psutil

def kill_old_monitors():
    """Remove monitores antigos silenciosamente"""
    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = proc.info.get('cmdline', [])
                if cmdline and 'monitor_gui.py' in ' '.join(cmdline):
                    try:
                        proc.terminate()
                        killed += 1
                    except:
                        try:
                            proc.kill()
                            killed += 1
                        except:
                            pass
        except:
            pass
    return killed

print("""
================================================================================
         QUANTUM TRADER ML - PRODUCAO COM ENHANCED MONITOR
================================================================================
Sistema: HMARL com captura de dados
Monitor: Enhanced Monitor (colorido) com informacoes HMARL
Modelo: book_clean (79.23% accuracy)

CARACTERISTICAS DO MONITOR:
- Interface colorida moderna
- Metricas em tempo real
- Status dos 4 agentes HMARL
- Graficos de performance
- Log de eventos colorido
================================================================================
""")

# Limpar monitores antigos
print("Verificando monitores...")
killed = kill_old_monitors()
if killed > 0:
    print(f"[OK] {killed} monitor(es) antigo(s) fechado(s)")
    time.sleep(1)

print("\nIniciando sistema com Enhanced Monitor...")
print("O monitor abrira em uma nova janela colorida")
print("="*80)

# Executar sistema com Enhanced Monitor
os.system("python start_hmarl_production_with_capture.py")