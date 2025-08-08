"""
Exemplo de execução do sistema de trading adaptativo
Integra o sistema adaptativo com o TradingSystem principal
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime
import time
import signal
from pathlib import Path

from src.trading_system import TradingSystem
from src.integration.adaptive_trading_integration import AdaptiveTradingIntegration

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('adaptive_trading.log')
    ]
)

# Variável global para controle de shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handler para shutdown graceful"""
    global shutdown_requested
    print("\n\nShutdown solicitado... Aguarde finalização.")
    shutdown_requested = True

def main():
    """Executa sistema de trading adaptativo integrado"""
    
    print("="*80)
    print("SISTEMA DE TRADING ADAPTATIVO INTEGRADO")
    print("="*80)
    print("\nEste sistema combina:")
    print("- Trading em tempo real com ProfitDLL")
    print("- Estratégia híbrida (tick + book)")
    print("- Aprendizado contínuo (online learning)")
    print("- A/B testing automático")
    print("- Monitoramento avançado")
    
    # Registrar handler de sinal
    signal.signal(signal.SIGINT, signal_handler)
    
    # Configuração do TradingSystem
    trading_config = {
        'symbols': ['WDOU25'],
        'data_path': 'data/',
        'model_path': 'models/',
        'max_positions': 2,
        'position_size': 1,
        'stop_loss': 0.02,
        'take_profit': 0.03,
        'log_level': 'INFO'
    }
    
    # Configuração do sistema adaptativo
    adaptive_config = {
        # Paths
        'models_path': 'models',
        
        # Estratégia híbrida
        'regime_threshold': 0.6,
        'tick_weight': 0.4,
        'book_weight': 0.6,
        'max_position': 2,
        'stop_loss': 0.02,
        'take_profit': 0.03,
        
        # Online learning
        'online_buffer_size': 50000,
        'retrain_interval': 1800,  # 30 minutos
        'min_samples_retrain': 5000,
        'validation_window': 500,
        'performance_threshold': 0.52,
        
        # A/B testing
        'ab_testing_enabled': True,
        'ab_test_ratio': 0.2,  # 20% para novos modelos
        
        # Adaptação
        'adaptation_rate': 0.1,
        'performance_window': 100,
        
        # Trading
        'signal_cooldown': 30,  # segundos entre sinais
        'candle_timeframe': '5min',
        'lookback_candles': 100,
        'base_position_size': 1,
        
        # Monitoramento
        'metrics_window': 1000,
        'alert_thresholds': {
            'accuracy': 0.45,
            'drawdown': 0.15,
            'latency': 1000,  # ms
            'buffer_overflow': 0.9
        }
    }
    
    try:
        # 1. Criar TradingSystem
        print("\n1. Inicializando TradingSystem...")
        trading_system = TradingSystem(trading_config)
        
        # Inicializar sistema principal
        if not trading_system.initialize():
            print("Erro ao inicializar TradingSystem")
            return
            
        print("[OK] TradingSystem inicializado")
        
        # 2. Criar integração adaptativa
        print("\n2. Criando integração adaptativa...")
        adaptive_integration = AdaptiveTradingIntegration(
            trading_system,
            adaptive_config
        )
        
        # 3. Inicializar integração
        print("\n3. Inicializando sistema adaptativo...")
        if not adaptive_integration.initialize():
            print("Erro ao inicializar sistema adaptativo")
            trading_system.shutdown()
            return
            
        print("[OK] Sistema adaptativo inicializado")
        
        # 4. Iniciar trading
        print("\n4. Iniciando trading adaptativo...")
        trading_system.start_trading()
        
        print("\n" + "="*80)
        print("SISTEMA RODANDO")
        print("="*80)
        print("\nO sistema está agora:")
        print("✓ Coletando dados em tempo real")
        print("✓ Executando estratégia híbrida adaptativa")
        print("✓ Aprendendo continuamente com novos dados")
        print("✓ Testando novos modelos automaticamente")
        print("✓ Monitorando performance em tempo real")
        print("\nPressione Ctrl+C para parar...")
        
        # Loop principal
        last_status_time = datetime.now()
        status_interval = 60  # segundos
        
        while not shutdown_requested:
            try:
                # Verificar se deve mostrar status
                if (datetime.now() - last_status_time).total_seconds() > status_interval:
                    print_system_status(adaptive_integration)
                    last_status_time = datetime.now()
                
                # Pequena pausa
                time.sleep(1)
                
            except KeyboardInterrupt:
                break
                
    except Exception as e:
        logging.error(f"Erro crítico: {e}")
        
    finally:
        # Shutdown graceful
        print("\n\nFinalizando sistema...")
        
        if 'adaptive_integration' in locals():
            print("- Desligando sistema adaptativo...")
            adaptive_integration.shutdown()
            
            # Gerar relatório final
            print("\n- Gerando relatório final...")
            generate_final_report(adaptive_integration)
            
        if 'trading_system' in locals():
            print("- Desligando TradingSystem...")
            trading_system.stop_trading()
            trading_system.shutdown()
            
        print("\n[OK] Sistema finalizado com sucesso")

def print_system_status(integration: AdaptiveTradingIntegration):
    """Imprime status do sistema"""
    
    try:
        status = integration.get_status()
        
        print("\n" + "="*60)
        print("STATUS DO SISTEMA")
        print("="*60)
        
        # Status geral
        print(f"\nAtivo: {status['active']}")
        print(f"Último sinal: {status['last_signal']}")
        
        # Métricas da estratégia
        strategy_metrics = status['strategy']
        if strategy_metrics:
            print(f"\nESTRATÉGIA ADAPTATIVA:")
            print(f"- Predições totais: {strategy_metrics.get('total_predictions', 0)}")
            print(f"- Accuracy recente: {strategy_metrics.get('recent_accuracy', 0):.2%}")
            print(f"- Aprendizado ativo: {strategy_metrics.get('is_learning', False)}")
            
            # Thresholds adaptativos
            thresholds = strategy_metrics.get('adaptive_thresholds', {})
            print(f"- Threshold regime: {thresholds.get('regime', 0):.2f}")
            print(f"- Threshold confiança: {thresholds.get('confidence', 0):.2f}")
        
        # Dashboard do monitor
        dashboard = status['monitor']
        if dashboard:
            # Performance
            perf = dashboard.get('performance', {})
            print(f"\nPERFORMANCE:")
            print(f"- Win rate: {perf.get('win_rate', 0):.2%}")
            print(f"- P&L total: ${perf.get('total_pnl', 0):.2f}")
            print(f"- Trades recentes: {perf.get('recent_trades', 0)}")
            
            # Alertas
            alerts = dashboard.get('alerts', [])
            if alerts:
                print(f"\nALERTAS RECENTES:")
                for alert in alerts[-3:]:  # Últimos 3 alertas
                    print(f"- [{alert['level']}] {alert['message']}")
                    
    except Exception as e:
        logging.error(f"Erro ao imprimir status: {e}")

def generate_final_report(integration: AdaptiveTradingIntegration):
    """Gera relatório final do sistema"""
    
    try:
        # Criar diretório de resultados
        results_dir = Path('results/adaptive_trading')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp para arquivos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Relatório JSON
        report_path = results_dir / f"adaptive_report_{timestamp}.json"
        report = integration.get_performance_report()
        
        if report:
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"✓ Relatório salvo: {report_path}")
        
        # 2. Gráficos de performance
        if hasattr(integration.monitor, 'plot_performance'):
            plot_path = results_dir / f"adaptive_performance_{timestamp}.png"
            integration.monitor.plot_performance(str(plot_path))
            print(f"✓ Gráficos salvos: {plot_path}")
        
        # 3. Resumo textual
        summary_path = results_dir / f"adaptive_summary_{timestamp}.txt"
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RESUMO DO SISTEMA DE TRADING ADAPTATIVO\n")
            f.write("="*80 + "\n\n")
            
            if report and 'summary' in report:
                summary = report['summary']
                
                # Sistema
                system = summary.get('system', {})
                f.write(f"Tempo de execução: {system.get('uptime', 'N/A')}\n")
                f.write(f"Total de predições: {system.get('total_predictions', 0)}\n")
                f.write(f"Total de trades: {system.get('total_trades', 0)}\n\n")
                
                # Performance
                perf = summary.get('performance', {})
                f.write(f"Win rate: {perf.get('win_rate', 0):.2%}\n")
                f.write(f"P&L total: ${perf.get('total_pnl', 0):.2f}\n\n")
                
                # Modelos
                models = summary.get('models', {})
                if models:
                    f.write("VERSÕES DOS MODELOS:\n")
                    for model_type, info in models.items():
                        f.write(f"- {model_type}: v{info.get('version', 0)}\n")
        
        print(f"✓ Resumo salvo: {summary_path}")
        
    except Exception as e:
        logging.error(f"Erro ao gerar relatório final: {e}")

if __name__ == "__main__":
    main()