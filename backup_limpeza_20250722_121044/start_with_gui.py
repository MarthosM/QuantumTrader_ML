#!/usr/bin/env python3
"""
🖥️ LAUNCHER GUI - ML TRADING v2.0
=================================
Inicia sistema com GUI garantido
"""

import sys
import os
import time
import threading
from datetime import datetime

# Adicionar diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def start_with_gui():
    """Inicia sistema com GUI garantido"""
    print("🚀 INICIANDO SISTEMA COM GUI GARANTIDO")
    print("="*50)
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print("")
    
    try:
        # Import com fallback
        try:
            from trading_system import TradingSystem
            from trading_monitor_gui import TradingMonitorGUI
            print("✅ Módulos importados com sucesso")
        except ImportError as e:
            print(f"❌ Erro no import: {e}")
            return False
        
        # Configuração com GUI forçado
        config = {
            'dll_path': os.getenv('PROFIT_DLL_PATH', ''),
            'username': os.getenv('PROFIT_USER', ''),
            'password': os.getenv('PROFIT_PASSWORD', ''),
            'account_id': os.getenv('PROFIT_ACCOUNT_ID', ''),
            'broker_id': os.getenv('PROFIT_BROKER_ID', ''),
            'trading_password': os.getenv('PROFIT_TRADING_PASSWORD', ''),
            'key': os.getenv('PROFIT_KEY', ''),
            'models_dir': os.getenv('MODELS_DIR', ''),
            'ml_interval': int(os.getenv('ML_INTERVAL', 15)),
            'use_gui': True,  # FORÇAR GUI
            'historical_days': int(os.getenv('HISTORICAL_DAYS', 10))
        }
        
        print("⚙️ CONFIGURAÇÕES:")
        print(f"   • GUI: {config['use_gui']}")
        print(f"   • ML_INTERVAL: {config['ml_interval']}s")
        print("")
        
        # Criar sistema
        print("🔧 Criando sistema de trading...")
        system = TradingSystem(config)
        
        # Inicializar
        print("🔄 Inicializando sistema...")
        if not system.initialize():
            print("❌ Falha na inicialização")
            return False
            
        print("✅ Sistema inicializado")
        
        # FORÇAR CRIAÇÃO DO GUI
        print("🖥️ FORÇANDO CRIAÇÃO DO GUI...")
        try:
            gui = TradingMonitorGUI(system)
            print("✅ GUI criado com sucesso")
            
            # Iniciar GUI em thread separada
            def run_gui():
                try:
                    gui.run()
                except Exception as e:
                    print(f"❌ Erro no GUI: {e}")
                    
            gui_thread = threading.Thread(target=run_gui, daemon=False)
            gui_thread.start()
            print("✅ GUI iniciado em thread")
            
            time.sleep(2)  # Aguardar GUI inicializar
            
        except Exception as e:
            print(f"❌ Erro criando GUI: {e}")
            print("⚠️ Continuando sem GUI...")
        
        # Iniciar sistema
        print("▶️ Iniciando operação...")
        if system.start():
            print("✅ Sistema operacional!")
            
            # Manter vivo
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n⏹️ Parando sistema...")
                system.stop()
        else:
            print("❌ Falha ao iniciar sistema")
            return False
            
    except Exception as e:
        print(f"❌ Erro geral: {e}")
        return False
        
    return True

if __name__ == "__main__":
    start_with_gui()
