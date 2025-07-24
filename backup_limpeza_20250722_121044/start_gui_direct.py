#!/usr/bin/env python3
"""
🚀 START GUI DIRETO - ML TRADING v2.0
====================================
Inicia sistema garantindo que o GUI abra

DIFERENCIAL:
✅ Força abertura do GUI
✅ Não depende de configurações externas
✅ Mostra status de inicialização
"""

import os
import sys
import time
import threading
from datetime import datetime

# Configurar path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def load_config():
    """Carrega configuração do .env"""
    config = {}
    
    # Carregar do .env
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    config[key] = value
    
    # Converter para formato esperado
    return {
        'dll_path': config.get('PROFIT_DLL_PATH', ''),
        'username': config.get('PROFIT_USER', ''),
        'password': config.get('PROFIT_PASSWORD', ''),
        'account_id': config.get('PROFIT_ACCOUNT_ID', ''),
        'broker_id': config.get('PROFIT_BROKER_ID', ''),
        'trading_password': config.get('PROFIT_TRADING_PASSWORD', ''),
        'key': config.get('PROFIT_KEY', ''),
        'models_dir': config.get('MODELS_DIR', ''),
        'ml_interval': int(config.get('ML_INTERVAL', 15)),
        'use_gui': True,  # SEMPRE TRUE
        'historical_days': int(config.get('HISTORICAL_DAYS', 10)),
        'initial_balance': float(config.get('INITIAL_BALANCE', 100000))
    }

def start_gui_forced():
    """Inicia sistema com GUI forçado"""
    print("🚀 INICIANDO ML TRADING v2.0 COM GUI GARANTIDO")
    print("="*55)
    print(f"🕐 Início: {datetime.now().strftime('%H:%M:%S')}")
    print("")
    
    try:
        # 1. Carregar configuração
        print("⚙️ Carregando configuração...")
        config = load_config()
        print(f"   ✅ ML_INTERVAL: {config['ml_interval']}s")
        print(f"   ✅ GUI: {config['use_gui']}")
        print("")
        
        # 2. Importar módulos
        print("📦 Importando módulos...")
        from trading_system import TradingSystem
        from trading_monitor_gui import TradingMonitorGUI
        print("   ✅ Módulos carregados")
        print("")
        
        # 3. Criar sistema
        print("🔧 Criando sistema de trading...")
        system = TradingSystem(config)
        print("   ✅ Sistema criado")
        
        # 4. Inicializar sistema
        print("🔄 Inicializando componentes...")
        if not system.initialize():
            print("   ❌ Falha na inicialização do sistema")
            return False
        print("   ✅ Sistema inicializado")
        print("")
        
        # 5. CRIAR GUI IMEDIATAMENTE
        print("🖥️ INICIANDO MONITOR GUI...")
        gui = TradingMonitorGUI(system)
        print("   ✅ GUI criado")
        
        # 6. Iniciar GUI em thread não-daemon (para manter vivo)
        def run_gui():
            try:
                print("   🚀 Executando GUI...")
                gui.run()
            except Exception as e:
                print(f"   ❌ Erro no GUI: {e}")
        
        gui_thread = threading.Thread(target=run_gui, daemon=False)
        gui_thread.start()
        print("   ✅ GUI thread iniciada")
        
        # Pequena pausa para GUI inicializar
        time.sleep(2)
        print("")
        
        # 7. Iniciar sistema de trading
        print("▶️ INICIANDO OPERAÇÃO...")
        if system.start():
            print("   ✅ Sistema operacional!")
            print("")
            print("📊 MONITOR GUI ATIVO - AGUARDE DADOS...")
            print("="*55)
            print("🎯 O que observar:")
            print("   • Predições ML a cada 15-20 segundos")
            print("   • Preços em tempo real")
            print("   • Sinais de trading")
            print("   • Métricas de performance")
            print("")
            
            # Manter sistema vivo
            try:
                # Aguardar thread do GUI terminar
                gui_thread.join()
            except KeyboardInterrupt:
                print("\n⏹️ Interrupção detectada - parando sistema...")
                system.stop()
                return True
                
        else:
            print("   ❌ Falha ao iniciar sistema")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Função principal"""
    success = start_gui_forced()
    
    if success:
        print("\n✅ Sistema finalizado com sucesso!")
    else:
        print("\n❌ Sistema finalizado com erro!")
        
    print(f"🕐 Fim: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
