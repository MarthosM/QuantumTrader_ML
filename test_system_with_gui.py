"""
Teste do sistema completo com GUI monitor
"""

import os
import sys
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_system_with_gui():
    print("=" * 80)
    print("TESTE: SISTEMA COMPLETO COM GUI MONITOR")
    print("=" * 80)
    
    # Configurar ambiente
    os.environ['TRADING_ENV'] = 'development'
    os.environ['USE_GUI'] = 'true'
    
    try:
        # Importar e configurar sistema
        from src.main import load_config
        from src.trading_system import TradingSystem
        
        config = load_config()
        print(f"1. Configuração carregada - GUI: {config.get('use_gui')}")
        
        # Inicializar sistema
        print("2. Inicializando TradingSystem...")
        trading_system = TradingSystem(config)
        
        print("3. Iniciando sistema...")
        
        # Simular inicialização do sistema em thread separada
        def run_system():
            try:
                trading_system.start()
            except Exception as e:
                print(f"Erro no sistema: {e}")
        
        system_thread = threading.Thread(target=run_system, daemon=True)
        system_thread.start()
        
        # Aguardar inicialização
        print("4. Aguardando inicialização...")
        time.sleep(5)
        
        # Verificar se GUI foi criado
        if hasattr(trading_system, 'monitor') and trading_system.monitor:
            print("5. Monitor GUI encontrado!")
            print(f"   GUI Type: {type(trading_system.monitor)}")
            
            if hasattr(trading_system.monitor, 'root'):
                print("   Janela GUI criada com sucesso")
                print(f"   Title: {trading_system.monitor.root.title()}")
                
                # Verificar se GUI está rodando
                if hasattr(trading_system.monitor, 'running'):
                    print(f"   Status: {'Rodando' if trading_system.monitor.running else 'Parado'}")
                
                # Testar atualização do GUI por alguns segundos
                print("6. Testando GUI por 10 segundos...")
                
                # Simular dados para atualização
                test_data = {
                    'candle': {
                        'datetime': '2025-07-24 11:50:00',
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
                if hasattr(trading_system.monitor, 'update_data'):
                    trading_system.monitor.update_data(test_data)
                    print("   Dados de teste enviados para GUI")
                
                # Dar tempo para o GUI processar
                for i in range(10):
                    print(f"   Aguardando... {i+1}/10s")
                    time.sleep(1)
                    
                    # Verificar se janela ainda existe
                    try:
                        if trading_system.monitor.root.winfo_exists():
                            continue
                        else:
                            print("   AVISO: Janela GUI foi fechada")
                            break
                    except Exception:
                        print("   AVISO: Erro verificando janela GUI")
                        break
                
                print("7. RESULTADO: GUI funcionando corretamente!")
                return True
                
            else:
                print("   ERRO: Janela GUI não foi criada")
                return False
        else:
            print("5. ERRO: Monitor GUI não foi encontrado")
            print("   Verificando configuração...")
            
            if hasattr(trading_system, 'use_gui'):
                print(f"   use_gui: {trading_system.use_gui}")
            
            if hasattr(trading_system, '_gui_ready'):
                print(f"   _gui_ready: {trading_system._gui_ready}")
            
            return False
            
    except Exception as e:
        print(f"ERRO GERAL: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            if 'trading_system' in locals():
                trading_system.stop()
        except:
            pass

if __name__ == "__main__":
    success = test_system_with_gui()
    print("\n" + "=" * 80)
    if success:
        print("SUCESSO: Sistema com GUI funcionando!")
    else:
        print("FALHA: Problema na inicialização do GUI")
    print("=" * 80)