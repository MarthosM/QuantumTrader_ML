#!/usr/bin/env python3
"""
🚀 QUICK START - ML TRADING v2.0
Inicia sistema com configurações otimizadas
"""

import os
import sys
import subprocess
from datetime import datetime

def quick_start():
    print("🚀 INICIANDO ML TRADING v2.0 - MODO AGRESSIVO")
    print("="*50)
    print(f"Início: {datetime.now().strftime('%H:%M:%S')}")
    print("")
    
    print("⚙️ CONFIGURAÇÕES ATIVAS:")
    print("   • ML_INTERVAL: 15 segundos")
    print("   • THRESHOLDS: Reduzidos (0.45)")
    print("   • MONITORAMENTO: Tempo real") 
    print("   • PREDIÇÕES: 240/hora esperadas")
    print("")
    
    print("🔍 MONITORE ESTAS MÉTRICAS:")
    print("   • Predição ML - Direção: X.XX")
    print("   • SINAL GERADO: BUY/SELL")
    print("   • Métricas - Predições: >0")
    print("")
    
    print("⏰ Aguarde predições a cada 15-20 segundos...")
    print("="*50)
    print("")
    
    # Executar sistema
    try:
        print("🏃 Executando sistema de trading...")
        subprocess.run([sys.executable, "run_training.py"])
    except KeyboardInterrupt:
        print("\n⛔ Sistema interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")

if __name__ == "__main__":
    quick_start()
