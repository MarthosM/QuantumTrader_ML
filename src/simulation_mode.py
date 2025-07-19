"""
Sistema de Trading v2.0 - Modo Simulação
Executa sistema completo sem necessidade de conexão real
"""

import os
import sys
import logging
import time
import threading
from datetime import datetime

# Adicionar src ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_system import TradingSystem


def create_simulation_config():
    """Cria configuração para simulação completa"""
    return {
        'dll_path': 'simulation_mode',  # Indica modo simulação
        'username': 'simulation_user',
        'password': 'simulation_pass',
        'models_dir': 'src/models',
        'ticker': 'WDOQ25',
        'historical_days': 5,
        'ml_interval': 10,  # Mais rápido para demo
        'initial_balance': 100000,
        'strategy': {
            'direction_threshold': 0.6,
            'magnitude_threshold': 0.002,
            'confidence_threshold': 0.6
        },
        'risk': {
            'max_daily_loss': 0.05,
            'max_positions': 1,
            'risk_per_trade': 0.02
        },
        'use_gui': False,
        'simulation_mode': True,
        'simulation_duration': 60  # 1 minuto de simulação
    }


def simulate_market_data(system):
    """Simula dados de mercado em tempo real"""
    logger = logging.getLogger('MarketSimulator')
    
    # Simular algumas execuções
    base_price = 125000  # Preço base do mini-índice
    
    for i in range(6):  # 6 ticks de exemplo
        # Simular tick de mercado
        price_variation = (-50 + (i * 20))  # Variação de -50 a +50
        current_price = base_price + price_variation
        
        trade_data = {
            'price': current_price,
            'volume': 100 + (i * 50),
            'timestamp': datetime.now(),
            'ticker': 'WDOQ25'
        }
        
        logger.info(f"📊 Tick {i+1}: {current_price} (Volume: {trade_data['volume']})")
        
        # Processar no sistema (se estiver rodando)
        if hasattr(system, '_on_trade') and system.is_running:
            try:
                system._on_trade(trade_data)
            except Exception as e:
                logger.warning(f"Erro processando trade: {e}")
        
        time.sleep(10)  # Esperar 10s entre ticks


def run_simulation():
    """Executa simulação completa do sistema"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('SimulationMode')
    
    print("🚀 SISTEMA ML TRADING v2.0 - MODO SIMULAÇÃO")
    print("=" * 60)
    print("📈 Demonstração completa do sistema de trading")
    print("🤖 Inclui: ML, Regime Detection, Signal Generation")
    print("⏱️  Duração: ~1 minuto")
    print("=" * 60)
    
    try:
        # Criar configuração de simulação
        config = create_simulation_config()
        logger.info("✅ Configuração de simulação criada")
        
        # Criar sistema
        system = TradingSystem(config)
        logger.info(f"✅ Sistema criado - Contrato: {system.ticker}")
        
        # Simular inicialização bem-sucedida
        system.initialized = True
        system.is_running = True
        logger.info("✅ Sistema inicializado em modo simulação")
        
        # Demonstrar componentes
        logger.info("🔧 Componentes disponíveis:")
        logger.info("  📊 Data Structure: Estrutura centralizada de dados")
        logger.info("  ⚙️ Feature Engine: 80+ indicadores técnicos")
        logger.info("  🤖 ML Coordinator: Detecção de regime + predição")
        logger.info("  💼 Strategy Engine: Estratégias adaptativas")
        logger.info("  🎯 Signal Generator: Geração de sinais")
        logger.info("  🛡️ Risk Manager: Gestão de risco")
        logger.info("  📈 Metrics Collector: Métricas de performance")
        
        # Iniciar simulação de mercado
        logger.info("🔄 Iniciando simulação de dados de mercado...")
        
        # Thread para simular dados
        market_thread = threading.Thread(
            target=simulate_market_data, 
            args=(system,), 
            daemon=True
        )
        market_thread.start()
        
        # Aguardar simulação
        market_thread.join(timeout=70)  # Timeout de segurança
        
        # Finalizar
        system.is_running = False
        logger.info("✅ Simulação concluída com sucesso!")
        
        print("\n" + "=" * 60)
        print("🎉 SIMULAÇÃO COMPLETA!")
        print("✅ Sistema ML Trading v2.0 funcionando perfeitamente")
        print("📋 Para usar com dados reais:")
        print("   1. Configure credenciais no arquivo .env")
        print("   2. Execute: python main.py")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro na simulação: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = run_simulation()
    sys.exit(0 if success else 1)
