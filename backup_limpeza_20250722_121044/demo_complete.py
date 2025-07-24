#!/usr/bin/env python3
"""
Demo Completa - Trading System ML v2.0 + Monitor GUI
Mostra como integrar e usar o monitor GUI em um ambiente completo
"""

import sys
import os
import time
from datetime import datetime
import logging

# Adicionar src ao Python path
sys.path.insert(0, 'src')

def configurar_logging():
    """Configura system de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Usar logging customizado para o trading system
    logger = logging.getLogger('TradingDemo')
    return logger

def demo_gui_standalone():
    """Demo do monitor GUI standalone"""
    
    print("🎯 DEMO - Monitor GUI Standalone")
    print("=" * 50)
    
    from trading_monitor_gui import create_monitor_gui
    
    # Sistema simulado realista
    class RealisticMockSystem:
        """Sistema mock realista para demonstração"""
        
        def __init__(self):
            # Dados de trading simulados
            self.is_running = True
            
            # Estado inicial da conta
            self.account_info = {
                'balance': 150000.0,
                'available': 142300.0,
                'daily_pnl': 1250.75
            }
            
            # Posições ativas simuladas  
            self.active_positions = {
                'WDOQ25': {
                    'side': 'long',
                    'entry_price': 123456.75,
                    'current_price': 123679.25,
                    'size': 2,
                    'timestamp': datetime.now()
                }
            }
            
            # Última predição ML
            self.last_prediction = {
                'direction': 0.847,
                'confidence': 0.923,
                'magnitude': 0.0047,
                'action': 'BUY',
                'regime': 'trend_up',
                'timestamp': datetime.now()
            }
            
            # Simular data structure com candles
            self.data_structure = self._create_mock_data_structure()
            
            # Mock metrics
            self.metrics = self._create_mock_metrics()
            
        def _create_mock_data_structure(self):
            """Cria estrutura de dados simulada"""
            import pandas as pd
            
            class MockDataStructure:
                def __init__(self):
                    # Candle atual
                    current_time = datetime.now()
                    self.candles = pd.DataFrame([{
                        'open': 123456.50,
                        'high': 123789.25,
                        'low': 123234.75,
                        'close': 123679.25,
                        'volume': 15750,
                        'timestamp': current_time
                    }])
                    
            return MockDataStructure()
            
        def _create_mock_metrics(self):
            """Cria métricas ML simuladas"""
            class MockMetrics:
                def __init__(self):
                    self.metrics = {
                        'predictions_made': 156,
                        'signals_generated': 89,
                        'trades_total': 8,
                        'win_rate': 0.75
                    }
                    
            return MockMetrics()
            
        def _get_trading_metrics_safe(self):
            """Métricas de trading simuladas"""
            return {
                'trades_count': 8,
                'win_rate': 0.75,
                'pnl': 1250.75,
                'positions': len(self.active_positions)
            }
            
        def _get_system_metrics_safe(self):
            """Métricas do sistema simuladas"""
            return {
                'cpu_percent': 22.5,
                'memory_mb': 387.2,
                'threads': 12,
                'uptime': 14325  # ~4 horas
            }
    
    print("📊 Criando sistema realista...")
    system = RealisticMockSystem()
    print("✓ Sistema mock realista criado")
    
    print("🖥️  Criando monitor GUI...")
    monitor = create_monitor_gui(system)
    print("✓ Monitor GUI criado")
    
    print("\n" + "="*60)
    print("🎯 MONITOR GUI ATIVO - DADOS REALISTAS")
    print("="*60)
    print("📈 Dados sendo exibidos:")
    print(f"  • Predição: {system.last_prediction['action']} com {system.last_prediction['confidence']:.1%} de confiança")
    print(f"  • P&L Diário: R$ {system.account_info['daily_pnl']:,.2f}")
    print(f"  • Posições: {len(system.active_positions)} aberta(s)")
    print(f"  • Win Rate: {system._get_trading_metrics_safe()['win_rate']:.1%}")
    
    print("\n📋 Como usar:")
    print("  1. 🟢 Clique 'Iniciar Monitor' para começar")
    print("  2. 👀 Observe os dados em tempo real")
    print("  3. 📊 Navegue pelas abas (Trading/Sistema/Posições)")
    print("  4. 🔴 Feche a janela para terminar")
    print("="*60)
    
    # Executar interface
    monitor.run()

def demo_integration_example():
    """Exemplo de integração com sistema real"""
    
    print("\n🔧 EXEMPLO DE INTEGRAÇÃO COM SISTEMA REAL")
    print("=" * 60)
    
    print("📝 Código para integrar o monitor ao seu sistema:")
    print("-" * 40)
    
    integration_code = '''
# No seu trading_system.py

def __init__(self, config):
    # ... outras inicializações ...
    
    # 🔑 CHAVE: Habilitar GUI
    self.use_gui = config.get('use_gui', False)
    self.monitor = None

def start(self):
    # ... inicialização do sistema ...
    
    # Auto-iniciar monitor GUI se habilitado
    if self.use_gui:
        from trading_monitor_gui import create_monitor_gui
        self.monitor = create_monitor_gui(self)
        
        # Executar em thread separada  
        monitor_thread = threading.Thread(
            target=self.monitor.run,
            daemon=True
        )
        monitor_thread.start()
        
        # Auto-iniciar monitoramento
        time.sleep(1)  # Aguardar GUI carregar
        if self.is_running:
            self.monitor.start_monitoring()

def stop(self):
    # Para monitor primeiro
    if self.monitor:
        self.monitor.stop()
    
    # ... parar outros componentes ...

# No seu main.py
config = {
    'use_gui': True,  # 🔑 ESSENCIAL!
    'dll_path': 'caminho/para/ProfitDLL.dll',
    'username': 'seu_usuario',
    'password': 'sua_senha',
    'models_dir': 'models/',
    'ticker': 'WDOQ25',
    # ... outras configs ...
}

system = TradingSystem(config)
if system.initialize():
    system.start()  # GUI abre automaticamente!
'''
    
    print(integration_code)
    print("-" * 40)
    print("✅ Integração simples e automática!")
    print("✅ Zero configuração adicional necessária")
    print("✅ Threading automático (não bloqueia sistema)")
    print("✅ Coleta automática de dados em tempo real")

def demo_features_showcase():
    """Demonstração das funcionalidades"""
    
    print("\n🎨 FUNCIONALIDADES DO MONITOR GUI")
    print("=" * 50)
    
    features = [
        ("🎯 Predições ML", [
            "Direção do preço (-1 a +1)",
            "Confiança do modelo (0% a 100%)",
            "Magnitude da predição",
            "Ação recomendada (BUY/SELL/HOLD)",
            "Regime de mercado detectado",
            "Timestamp da última predição"
        ]),
        ("📈 Dados de Candle", [
            "OHLC (Open, High, Low, Close)",
            "Volume negociado",
            "Variação percentual",
            "Timestamp do último candle",
            "Cores intuitivas (verde/vermelho)"
        ]),
        ("💹 Métricas Trading", [
            "P&L diário em tempo real",
            "Número de trades executados",
            "Win rate atualizado",
            "Posições ativas",
            "Saldo e disponível da conta"
        ]),
        ("⚡ Métricas Sistema", [
            "Uso de CPU (%)",
            "Consumo de memória (MB)",
            "Threads ativas",
            "Uptime do sistema",
            "Predições ML realizadas",
            "Sinais gerados"
        ]),
        ("📊 Posições Ativas", [
            "Símbolo do ativo",
            "Lado (Long/Short)",
            "Preço de entrada",
            "Preço atual",
            "P&L em tempo real",
            "Tamanho da posição"
        ]),
        ("🚨 Sistema de Alertas", [
            "Drawdown crítico (>5%)",
            "Win rate baixo (<45%)",
            "Memória alta (>80%)",
            "Latência alta (>100ms)",
            "Model drift detectado"
        ]),
        ("🛠️ Controles", [
            "Botão Iniciar/Parar monitoramento",
            "Abas organizadas (Trading/Sistema/Posições)",
            "Timestamp da última atualização",
            "Interface responsiva",
            "Fechamento seguro"
        ])
    ]
    
    for title, items in features:
        print(f"\n{title}:")
        for item in items:
            print(f"  • {item}")
    
    print(f"\n🎉 Total: {sum(len(items) for _, items in features)} funcionalidades!")

def main():
    """Função principal da demo"""
    
    print("🎯 TRADING SYSTEM ML v2.0 - DEMO COMPLETA")
    print("=" * 60)
    print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    logger = configurar_logging()
    logger.info("Demo iniciada")
    
    print("\nEscolha uma opção:")
    print("1. 🖥️  Demo Monitor GUI (Recomendado)")
    print("2. 🔧 Ver exemplo de integração")
    print("3. 🎨 Listar todas as funcionalidades")
    print("4. 🎯 Demo completa (GUI + Exemplos)")
    print("0. ❌ Sair")
    
    choice = input("\nOpção: ").strip()
    
    if choice == "1":
        demo_gui_standalone()
    elif choice == "2":
        demo_integration_example()
    elif choice == "3":
        demo_features_showcase()
    elif choice == "4":
        demo_features_showcase()
        demo_integration_example()
        
        if input("\n🎯 Executar demo GUI? (s/n): ").lower().startswith('s'):
            demo_gui_standalone()
    elif choice == "0":
        print("👋 Até logo!")
    else:
        print("❌ Opção inválida")
        
    logger.info("Demo finalizada")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro na demo: {e}")
        import traceback
        traceback.print_exc()
