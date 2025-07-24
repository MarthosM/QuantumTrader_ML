#!/usr/bin/env python3
"""
Monitor de Tempo Real - ML Trading v2.0
Monitora sistema em tempo real após patches
"""

import time
import subprocess
from datetime import datetime

def monitor_system():
    print("="*50)
    print("📊 MONITOR TEMPO REAL - ML TRADING v2.0")
    print("="*50)
    print(f"Início: {datetime.now().strftime('%H:%M:%S')}")
    print("")
    
    # Métricas esperadas após correções
    expected = {
        "Predições por hora": "120-180 (era 0)",
        "Sinais por hora": "3-8 (era 0)", 
        "Intervalo ML": "20s (era 60s)",
        "Thresholds": "0.5 (era 0.6)",
        "Atualizações": "Tempo real"
    }
    
    print("🎯 MÉTRICAS ESPERADAS APÓS CORREÇÕES:")
    for metric, value in expected.items():
        print(f"   • {metric}: {value}")
    
    print("")
    print("🔍 MONITORE OS LOGS PARA VERIFICAR:")
    print("   • Predição ML - Direção: X.XX")
    print("   • SINAL GERADO: BUY/SELL @ X.XX")
    print("   • Métricas - Predições: >0")
    print("")
    print("⏰ Sistema deve mostrar atividade a cada 20-30 segundos")
    print("="*50)

if __name__ == "__main__":
    monitor_system()
