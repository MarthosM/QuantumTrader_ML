#!/usr/bin/env python3
"""
Exemplo de integração do Monitor GUI com o Trading System
Demonstra como inicializar e usar o monitor visual
"""

import sys
import os
import time
from datetime import datetime
import logging

# Adicionar o diretório src ao Python path
sys.path.insert(0, 'src')

def main():
    """Exemplo principal de uso do Monitor GUI"""
    
    print("🎯 Sistema de Trading ML v2.0 - Monitor GUI")
    print("=" * 50)
    
    # Configurar logging básico
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Importar o monitor GUI
    try:
        from trading_monitor_gui import TradingMonitorGUI, create_monitor_gui
        print("✓ Monitor GUI importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro importando monitor GUI: {e}")
        return
    
    # Criar sistema mock simplificado para teste
    class MockTradingSystem:
        """Sistema simulado para demonstração"""
        
        def __init__(self):
            self.last_prediction = {
                'direction': 0.82,
                'confidence': 0.91,
                'magnitude': 0.0078,
                'action': 'BUY',
                'regime': 'trend_up',
                'timestamp': datetime.now()
            }
            
            self.active_positions = {
                'WDOQ25': {
                    'side': 'long',
                    'entry_price': 123450.75,
                    'current_price': 123678.25,
                    'size': 2,
                    'timestamp': datetime.now()
                }
            }
            
            self.account_info = {
                'balance': 150000.0,
                'available': 142300.0,
                'daily_pnl': 1250.75
            }
            
            self.is_running = True
            
            # Mock data structure com candles
            self.data_structure = MockDataStructure()
            
            # Mock metrics
            self.metrics = MockMetrics()
            
        def _get_trading_metrics_safe(self):
            """Retorna métricas de trading simuladas"""
            return {
                'trades_count': 8,
                'win_rate': 0.75,
                'pnl': 1250.75,
                'positions': len(self.active_positions)
            }
            
        def _get_system_metrics_safe(self):
            """Retorna métricas de sistema simuladas"""
            return {
                'cpu_percent': 22.5,
                'memory_mb': 387.2,
                'threads': 12,
                'uptime': 14325  # ~4 horas
            }
    
    class MockDataStructure:
        """Mock da estrutura de dados"""
        def __init__(self):
            import pandas as pd
            
            # Simular candle atual
            current_time = datetime.now()
            self.candles = pd.DataFrame([{
                'open': 123456.50,
                'high': 123789.25,
                'low': 123234.75,
                'close': 123678.25,
                'volume': 15750,
                'timestamp': current_time
            }])
    
    class MockMetrics:
        """Mock de métricas ML"""
        def __init__(self):
            self.metrics = {
                'predictions_made': 156,
                'signals_generated': 89,
                'trades_total': 8,
                'win_rate': 0.75
            }
    
    print("\n📊 Criando sistema simulado...")
    trading_system = MockTradingSystem()
    print("✓ Sistema mock criado")
    
    print("\n🖥️  Iniciando Monitor GUI...")
    try:
        # Criar monitor usando factory function
        monitor = create_monitor_gui(trading_system)
        print("✓ Monitor GUI criado")
        
        # Instruções para o usuário
        print("\n" + "="*50)
        print("🎯 MONITOR GUI INICIADO COM SUCESSO!")
        print("="*50)
        print("📍 O que você verá no monitor:")
        print("  • Predições ML em tempo real")
        print("  • Dados do último candle (OHLCV)")
        print("  • Métricas de trading (P&L, win rate)")
        print("  • Posições ativas")
        print("  • Métricas do sistema (CPU, memória)")
        print("  • Alertas em tempo real")
        print("\n🔴 Para parar: Feche a janela ou pressione Ctrl+C aqui")
        print("🟢 Para iniciar monitoramento: Clique em 'Iniciar Monitor'")
        print("="*50)
        
        # Executar interface gráfica
        monitor.run()
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupção do usuário detectada")
    except Exception as e:
        print(f"❌ Erro executando monitor: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔚 Monitor GUI finalizado")
        

def test_integration_with_real_system():
    """
    Exemplo de integração com sistema real
    (Descomente e adapte conforme necessário)
    """
    '''
    # Importar sistema real
    from trading_system import TradingSystem
    from trading_monitor_gui import create_monitor_gui
    
    # Configuração do sistema
    config = {
        'dll_path': r'C:\\Path\\To\\ProfitDLL\\ProfitDLL.dll',
        'username': 'seu_usuario',
        'password': 'sua_senha',
        'models_dir': 'models/',
        'use_gui': True,  # Importante!
        'ticker': 'WDOQ25',
        'historical_days': 10
    }
    
    # Criar sistema de trading
    system = TradingSystem(config)
    
    # Inicializar sistema
    if system.initialize():
        print("Sistema inicializado - GUI será aberta automaticamente")
        
        # Iniciar sistema (GUI será criada internamente)
        system.start()
    else:
        print("Erro na inicialização do sistema")
    '''
    pass


if __name__ == "__main__":
    print("Escolha uma opção:")
    print("1. Demo com dados simulados (recomendado)")
    print("2. Ver exemplo de integração com sistema real")
    print("0. Sair")
    
    choice = input("\nOpção: ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        print("\n📝 Exemplo de integração:")
        print("="*50)
        
        example_code = '''
# Exemplo de integração no seu sistema real:

from trading_system import TradingSystem
from trading_monitor_gui import create_monitor_gui

# Configuração
config = {
    'use_gui': True,  # 🔑 Importante!
    'dll_path': 'caminho_para_dll',
    'username': 'usuario',
    'password': 'senha',
    'models_dir': 'models/',
    # ... outras configurações
}

# Criar e inicializar sistema
system = TradingSystem(config)
if system.initialize():
    system.start()  # GUI será criada automaticamente
'''
        print(example_code)
        print("="*50)
        
    elif choice == "0":
        print("👋 Até logo!")
    else:
        print("❌ Opção inválida")
