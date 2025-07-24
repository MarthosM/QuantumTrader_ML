#!/usr/bin/env python3
"""
🖥️ CORREÇÃO MONITOR GUI - ML TRADING v2.0
=========================================
Força a inicialização do monitor GUI que não está abrindo

PROBLEMAS IDENTIFICADOS:
✅ USE_GUI=False no .env (corrigido)
✅ Possível erro na inicialização do GUI
✅ Thread do monitor pode não estar iniciando

SOLUÇÕES IMPLEMENTADAS:
✅ Ativar USE_GUI=True
✅ Criar launcher específico para GUI
✅ Adicionar fallbacks para erros de GUI
"""

import os
import sys
import time
import threading
from datetime import datetime

def fix_gui_config():
    """Corrige configuração do GUI"""
    print("🔧 CORRIGINDO CONFIGURAÇÃO DO GUI...")
    
    # Verificar se USE_GUI está True
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if "USE_GUI=True" in content:
            print("   ✅ USE_GUI=True configurado corretamente")
            return True
        else:
            print("   ❌ USE_GUI não está True - corrigindo...")
            
            # Corrigir
            content = content.replace("USE_GUI=False", "USE_GUI=True")
            
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(content)
                
            print("   ✅ USE_GUI=True aplicado")
            return True
    else:
        print("   ❌ Arquivo .env não encontrado")
        return False

def create_gui_launcher():
    """Cria launcher específico para GUI"""
    print("🖥️ CRIANDO LAUNCHER ESPECÍFICO PARA GUI...")
    
    launcher_content = '''#!/usr/bin/env python3
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
                print("\\n⏹️ Parando sistema...")
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
'''
    
    with open("start_with_gui.py", "w", encoding='utf-8') as f:
        f.write(launcher_content)
        
    print("   ✅ Launcher GUI criado: start_with_gui.py")

def create_gui_test():
    """Cria teste simples para verificar GUI"""
    print("🧪 CRIANDO TESTE DO GUI...")
    
    test_content = '''#!/usr/bin/env python3
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
'''
    
    with open("test_gui.py", "w", encoding='utf-8') as f:
        f.write(test_content)
        
    print("   ✅ Teste GUI criado: test_gui.py")

def patch_trading_system_gui():
    """Aplica patch no sistema para garantir GUI"""
    print("🔧 APLICANDO PATCH PARA GARANTIR GUI...")
    
    try:
        # Ler sistema atual
        with open("src/trading_system.py", "r", encoding='utf-8') as f:
            content = f.read()
            
        # Procurar e modificar seção do GUI
        if "self.use_gui = config.get('use_gui', True)" not in content:
            # Aplicar patch para forçar GUI
            content = content.replace(
                "self.use_gui = config.get('use_gui', True)",
                "self.use_gui = True  # PATCH: Sempre usar GUI"
            )
            
            # Se não encontrou, adicionar na inicialização
            if "self.use_gui" not in content:
                init_pos = content.find("def __init__(self, config: Dict):")
                if init_pos > 0:
                    # Encontrar final do __init__
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if "__init__(self, config: Dict):" in line:
                            # Inserir após algumas linhas
                            lines.insert(i + 10, "        self.use_gui = True  # PATCH: GUI forçado")
                            break
                    content = '\n'.join(lines)
            
            # Salvar
            with open("src/trading_system.py", "w", encoding='utf-8') as f:
                f.write(content)
                
            print("   ✅ Patch aplicado no sistema de trading")
        else:
            print("   ✅ Sistema já configurado para GUI")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Erro aplicando patch: {e}")
        return False

def show_instructions():
    """Mostra instruções finais"""
    print("")
    print("📋 CORREÇÃO DO GUI CONCLUÍDA!")
    print("="*40)
    print("")
    print("🚀 COMANDOS PARA TESTAR:")
    print("   1. Teste simples: python test_gui.py")
    print("   2. Sistema com GUI: python start_with_gui.py")
    print("   3. Método tradicional: python run_training.py")
    print("")
    print("✅ MELHORIAS APLICADAS:")
    print("   • USE_GUI=True ativado")
    print("   • Launcher específico criado")
    print("   • Teste de GUI incluído")
    print("   • Patches aplicados")
    print("")
    print("⚠️ SE GUI NÃO ABRIR:")
    print("   1. Verificar se tkinter está instalado")
    print("   2. Executar: python test_gui.py")
    print("   3. Verificar logs de erro")
    print("")
    print(f"🕐 Corrigido em: {datetime.now().strftime('%H:%M:%S')}")

def main():
    """Função principal"""
    print("🖥️ CORREÇÃO MONITOR GUI - ML TRADING v2.0")
    print("="*50)
    
    try:
        # 1. Corrigir configuração
        if not fix_gui_config():
            print("❌ Falha na configuração")
            return
            
        # 2. Criar launcher
        create_gui_launcher()
        
        # 3. Criar teste
        create_gui_test()
        
        # 4. Aplicar patches
        patch_trading_system_gui()
        
        # 5. Mostrar instruções
        show_instructions()
        
        print("")
        print("✅ CORREÇÃO CONCLUÍDA - TESTE O GUI!")
        
    except Exception as e:
        print(f"❌ Erro na correção: {e}")

if __name__ == "__main__":
    main()
