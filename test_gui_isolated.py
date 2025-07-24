#!/usr/bin/env python3
"""
Teste isolado do GUI - diagnóstico de visibilidade
"""

import os
import sys
import logging

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_gui_isolated():
    """Teste simples do GUI isoladamente"""
    print("🔍 Testando GUI isoladamente...")
    
    try:
        # Configurar logging simples
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('GUITest')
        
        # Simular sistema de trading mínimo
        class MockTradingSystem:
            def __init__(self):
                self.is_running = True
                self.data_structure = None
                self.connection_manager = None
                self.logger = logger
                
            def get_candles_df(self):
                import pandas as pd
                # Mock de dados de candle
                return pd.DataFrame({
                    'open': [5577.0],
                    'high': [5578.0], 
                    'low': [5576.0],
                    'close': [5577.5],
                    'volume': [1000000],
                    'trades': [100],
                    'buy_volume': [600000],
                    'sell_volume': [400000]
                })
                
            def get_current_price(self):
                return 5577.5
                
            def get_day_statistics(self):
                return {
                    'day_open': 5570.0,
                    'day_high': 5580.0,
                    'day_low': 5565.0,
                    'day_variation': 7.5,
                    'day_variation_pct': 0.13
                }
        
        # Criar sistema mock
        mock_system = MockTradingSystem()
        
        # Importar e criar GUI
        from trading_monitor_gui import create_monitor_gui
        
        print("✓ Importação do GUI bem-sucedida")
        
        # Criar monitor GUI
        monitor = create_monitor_gui(mock_system)
        print("✓ Monitor GUI criado com sucesso")
        
        # Testar se janela é criada
        if hasattr(monitor, 'root'):
            print("✓ Janela Tkinter criada")
            
            # Tentar mostrar janela
            monitor.root.title("TESTE - Monitor Trading ML v2.0")
            monitor.root.geometry("800x600+100+100")  # Força posição específica
            monitor.root.lift()  # Traz para frente
            monitor.root.attributes('-topmost', True)  # Sempre no topo
            monitor.root.focus_force()  # Força foco
            
            print("✓ Configurações de visibilidade aplicadas")
            print("🚀 Iniciando GUI em modo de teste...")
            
            # Executar mainloop
            monitor.run()
            
        else:
            print("❌ Janela Tkinter não foi criada")
            return False
            
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro geral: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == '__main__':
    print("=" * 60)
    print("TESTE ISOLADO DO GUI - DIAGNÓSTICO DE VISIBILIDADE")
    print("=" * 60)
    
    success = test_gui_isolated()
    
    if success:
        print("\n✅ Teste concluído - GUI funcionando")
    else:
        print("\n❌ Teste falhou - Problemas com GUI")
        
    print("=" * 60)
