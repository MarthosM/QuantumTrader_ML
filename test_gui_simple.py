"""
Teste simples do GUI isolado
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_gui_simple():
    print("=" * 80)
    print("TESTE SIMPLES: GUI ISOLADO")
    print("=" * 80)
    
    # Configurar ambiente
    os.environ['TRADING_ENV'] = 'development'
    
    try:
        # Criar sistema mock para teste
        class MockTradingSystem:
            def __init__(self):
                self.data_structure = None
                self.is_running = True
                self.model_manager = None
                self.ticker = "WDOQ25"
                
        mock_system = MockTradingSystem()
        
        print("1. Sistema mock criado")
        
        # Testar criação do GUI
        from src.trading_monitor_gui import create_monitor_gui
        
        print("2. Criando monitor GUI...")
        gui = create_monitor_gui(mock_system)
        
        print("3. GUI criado com sucesso!")
        print(f"   Type: {type(gui)}")
        print(f"   Title: {gui.root.title()}")
        
        # Testar inicialização do GUI
        print("4. Testando funcionalidade do GUI...")
        
        # Simular dados de teste
        test_data = {
            'candle': {
                'datetime': '2025-07-24 11:55:00',
                'open': 130000,
                'high': 130100,
                'low': 129900, 
                'close': 130050,
                'volume': 1000
            },
            'prediction': {
                'action': 'BUY',
                'confidence': 0.75,
                'probability': 0.68
            }
        }
        
        # Atualizar GUI com dados de teste
        if hasattr(gui, 'update_data'):
            gui.update_data(test_data)
            print("   Dados de teste enviados")
        
        # Testar por alguns segundos
        print("5. Rodando GUI por 5 segundos...")
        
        def run_gui_test():
            try:
                # Configurar GUI para não bloquear
                gui.root.after(5000, lambda: gui.root.quit())  # Sair após 5 segundos
                gui.run()
                return True
            except Exception as e:
                print(f"   Erro no GUI: {e}")
                return False
        
        success = run_gui_test()
        
        if success:
            print("6. SUCESSO: GUI funcionou corretamente!")
            return True
        else:
            print("6. FALHA: Problema no GUI")
            return False
            
    except Exception as e:
        print(f"ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gui_simple()
    print("\n" + "=" * 80)
    if success:
        print("RESULTADO: GUI funciona isoladamente!")
        print("CONCLUSÃO: O problema não é no GUI, mas na inicialização do sistema.")
    else:
        print("RESULTADO: Problema no próprio GUI")
    print("=" * 80)