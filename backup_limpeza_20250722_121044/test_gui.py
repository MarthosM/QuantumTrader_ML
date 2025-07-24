#!/usr/bin/env python3
"""
🧪 TESTE GUI - ML TRADING v2.0
==============================
Testa se o GUI pode ser inicializado
"""

import sys
import os
import tkinter as tk
from datetime import datetime

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_gui():
    """Testa inicialização básica do GUI"""
    print("🧪 TESTANDO GUI...")
    print("="*30)
    
    try:
        # Teste básico do tkinter
        print("1. Testando Tkinter...")
        root = tk.Tk()
        root.title("Teste GUI")
        root.geometry("300x200")
        
        label = tk.Label(root, text="GUI Funcionando!", font=('Arial', 16))
        label.pack(expand=True)
        
        print("✅ Tkinter OK")
        
        # Mostrar por 3 segundos
        root.after(3000, root.quit)
        root.mainloop()
        root.destroy()
        
        # Teste do módulo de monitor
        print("2. Testando módulo de monitor...")
        try:
            from trading_monitor_gui import TradingMonitorGUI
            print("✅ Módulo TradingMonitorGUI importado")
        except ImportError as e:
            print(f"❌ Erro importando monitor: {e}")
            return False
            
        print("")
        print("✅ TESTE GUI CONCLUÍDO - SISTEMA PRONTO!")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

if __name__ == "__main__":
    test_gui()
