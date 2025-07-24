#!/usr/bin/env python3
"""
Teste Simples do Monitor GUI
"""

import sys
import os
import time
sys.path.insert(0, 'src')

def test_simple_gui():
    """Teste básico do monitor GUI"""
    
    print("🧪 TESTE SIMPLES DO MONITOR GUI")
    print("=" * 40)
    
    try:
        from trading_monitor_gui import TradingMonitorGUI, create_monitor_gui
        print("✓ Importação bem sucedida")
        
        # Sistema mock simples
        class SimpleMockSystem:
            def __init__(self):
                from datetime import datetime
                
                self.last_prediction = {
                    'direction': 0.75,
                    'confidence': 0.82,
                    'magnitude': 0.0045,
                    'action': 'BUY',
                    'regime': 'trend_up',
                    'timestamp': datetime.now()
                }
                
                self.is_running = True
                self.active_positions = {}
                self.account_info = {
                    'balance': 100000.0,
                    'available': 95000.0,
                    'daily_pnl': 250.0
                }
                
            def _get_trading_metrics_safe(self):
                return {'trades_count': 3, 'win_rate': 0.67, 'pnl': 250.0, 'positions': 0}
                
            def _get_system_metrics_safe(self):
                return {'cpu_percent': 15.2, 'memory_mb': 245.8, 'threads': 8, 'uptime': 3665}
        
        # Criar sistema mock
        mock_system = SimpleMockSystem()
        print("✓ Sistema mock criado")
        
        # Criar monitor
        monitor = create_monitor_gui(mock_system)
        print("✓ Monitor GUI criado com sucesso")
        
        print("\n🎯 TESTE BEM SUCEDIDO!")
        print("🖥️  A interface será aberta...")
        print("🟢 Clique em 'Iniciar Monitor' para ver dados")
        print("🔴 Feche a janela quando terminar")
        print("=" * 40)
        
        # Executar interface
        monitor.run()
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple_gui()
