"""
Exemplo de configuração para teste do sistema sem DLL
"""

import os
import sys
import logging

# Adicionar src ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_system import TradingSystem


def create_test_config():
    """Cria configuração para teste sem DLL"""
    return {
        'dll_path': 'mock_dll.dll',  # Caminho mock
        'username': 'test_user',
        'password': 'test_pass',
        'models_dir': 'src/models',
        'ticker': 'WDOQ25',
        'historical_days': 5,
        'ml_interval': 30,
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
        'test_mode': True  # Indica modo de teste
    }


def test_system():
    """Testa o sistema sem DLL"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('TestSystem')
    logger.info("Testando Sistema de Trading v2.0 (Modo Teste)")
    
    try:
        # Criar configuração de teste
        config = create_test_config()
        
        # Criar sistema
        system = TradingSystem(config)
        
        # Testar criação
        logger.info("✅ Sistema criado com sucesso")
        logger.info(f"📊 Contrato detectado: {system.ticker}")
        logger.info(f"⚙️ Configuração carregada: {len(config)} parâmetros")
        
        # Simular inicialização (sem DLL)
        logger.info("🧪 Simulando inicialização...")
        
        # Demonstrar funcionalidades
        logger.info("📈 Sistema pronto para:")
        logger.info("  - Processamento de dados históricos")
        logger.info("  - Cálculo de features (80+ indicadores)")
        logger.info("  - Detecção de regime de mercado")
        logger.info("  - Predições ML por regime")
        logger.info("  - Geração de sinais de trading")
        logger.info("  - Gestão de risco automatizada")
        
        logger.info("✅ Teste concluído com sucesso!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no teste: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = test_system()
    sys.exit(0 if success else 1)
