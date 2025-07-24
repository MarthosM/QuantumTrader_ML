"""
Teste para debugar problemas de GUI monitor
"""

import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configurar logging para debug
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_gui_debug():
    print("=" * 80)
    print("TESTE DEBUG: MONITOR GUI")
    print("=" * 80)
    
    # Set environment
    os.environ['TRADING_ENV'] = 'development'
    os.environ['USE_GUI'] = 'true'
    
    print("1. Verificando configuração de GUI...")
    print(f"   USE_GUI: {os.getenv('USE_GUI')}")
    print(f"   TRADING_ENV: {os.getenv('TRADING_ENV')}")
    
    # Test GUI module import
    print("\n2. Testando importação do módulo GUI...")
    try:
        from src.trading_monitor_gui import create_monitor_gui, TradingMonitorGUI
        print("   OK Modulo trading_monitor_gui importado com sucesso")
    except Exception as e:
        print(f"   ERRO na importacao do GUI: {e}")
        return
    
    # Test config loading
    print("\n3. Testando carregamento de configuração...")
    try:
        from src.main import load_config
        config = load_config()
        print(f"   use_gui na config: {config.get('use_gui')}")
        print(f"   OK Configuracao carregada com sucesso")
    except Exception as e:
        print(f"   ERRO no carregamento da config: {e}")
        return
    
    # Test TradingSystem initialization (without GUI)
    print("\n4. Testando inicialização do TradingSystem...")
    try:
        # Temporarily disable GUI for testing
        config['use_gui'] = False
        
        from src.trading_system import TradingSystem
        trading_system = TradingSystem(config)
        print("   OK TradingSystem inicializado com sucesso")
        
        # Test GUI creation
        print("\n5. Testando criação do monitor GUI...")
        config['use_gui'] = True  # Re-enable
        trading_system.use_gui = True
        
        # Try to create GUI manually
        gui = create_monitor_gui(trading_system)
        print("   OK Monitor GUI criado com sucesso")
        print(f"   GUI class: {type(gui)}")
        
        # Check if GUI window was created
        if hasattr(gui, 'root') and gui.root:
            print("   OK Janela GUI criada")
            print(f"   Window title: {gui.root.title()}")
        else:
            print("   AVISO Janela GUI nao foi criada")
            
    except Exception as e:
        print(f"   ERRO na criacao do GUI: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n6. Verificando componentes do sistema...")
    
    # Check feature calculation
    if hasattr(trading_system, 'feature_engine'):
        print("   OK FeatureEngine disponivel")
    else:
        print("   AVISO FeatureEngine nao encontrado")
    
    # Check ML coordinator
    if hasattr(trading_system, 'ml_coordinator'):
        print("   OK MLCoordinator disponivel")
    else:
        print("   AVISO MLCoordinator nao encontrado")
    
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO COMPLETO")
    print("=" * 80)
    
    # Summary
    print("OK Modulos importados corretamente")
    print("OK Configuracao carregada")
    print("OK TradingSystem inicializado")
    print("OK Monitor GUI criado")
    
    print("\nPROVÁVEL CAUSA:")
    print("O sistema está funcionando, mas pode haver um problema na")
    print("execução da thread principal ou no loop de eventos do GUI.")
    
    print("\nRECOMENDAÇÃO:")
    print("Execute o sistema principal e verifique os logs para")
    print("mensagens específicas sobre a inicialização do GUI.")

if __name__ == "__main__":
    test_gui_debug()