"""
Ponto de entrada principal do sistema v2.0
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Adicionar src ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_system import TradingSystem


def load_config():
    """Carrega configuração do sistema"""
    # Tentar diferentes caminhos para o .env
    possible_env_paths = [
        # Diretório atual
        '.env',
        # Diretório pai do script
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'),
        # Caminho absoluto conhecido
        r'C:\Users\marth\OneDrive\Programacao\Python\Projetos\ML_Tradingv2.0\.env'
    ]
    
    env_loaded = False
    for env_path in possible_env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            env_loaded = True
            break
    
    config = {
        'dll_path': os.getenv("PROFIT_DLL_PATH", r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"),
        'key': os.getenv('PROFIT_KEY'),
        'username': os.getenv('PROFIT_USER'),
        'password': os.getenv('PROFIT_PASSWORD'),
        'account_id': os.getenv('PROFIT_ACCOUNT_ID'),
        'broker_id': os.getenv('PROFIT_BROKER_ID'),
        'trading_password': os.getenv('PROFIT_TRADING_PASSWORD'),
        'models_dir': os.getenv('MODELS_DIR', r'C:\Users\marth\OneDrive\Programacao\Python\Projetos\ML_Tradingv2.0\src\models\models_regime3'),
        'ticker': os.getenv('TICKER'),
        'historical_days': int(os.getenv('HISTORICAL_DAYS', '1')),
        'ml_interval': int(os.getenv('ML_INTERVAL', '60')),
        'initial_balance': float(os.getenv('INITIAL_BALANCE', '100000')),
        'strategy': {
            'direction_threshold': float(os.getenv('DIRECTION_THRESHOLD', '0.45')),
            'magnitude_threshold': float(os.getenv('MAGNITUDE_THRESHOLD', '0.00015')),
            'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.15'))
        },
        'risk': {
            'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '0.05')),
            'max_positions': int(os.getenv('MAX_POSITIONS', '1')),
            'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.02'))
        },
        'use_gui': os.getenv('USE_GUI', 'true').lower() == 'true'
    }
    
    return config


def main():
    """Função principal"""
    # Configurar logging apenas uma vez
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    logger = logging.getLogger('Main')
    logger.info("Iniciando Sistema de Trading v2.0")
    
    try:
        # Carregar configuração
        config = load_config()
        
        # Verificar credenciais obrigatórias
        required_fields = ['key', 'username', 'password']
        missing_fields = [field for field in required_fields if not config.get(field)]
        
        if missing_fields:
            logger.error(f"Campos obrigatórios não configurados no .env: {', '.join(missing_fields)}")
            return 1
            
        # Log das configurações (sem senhas)
        logger.info(f"Usuário: {config['username']}")
        logger.info(f"Conta: {config.get('account_id', 'Não especificada')}")
        logger.info(f"Corretora: {config.get('broker_id', 'Não especificada')}")
        logger.info(f"Ticker: {config.get('ticker', 'Auto-detectado')}")
            
        # Criar sistema
        system = TradingSystem(config)
        
        # Inicializar
        if not system.initialize():
            logger.error("Falha na inicialização do sistema")
            return 1
            
        # Iniciar operação
        if not system.start():
            logger.error("Falha ao iniciar operação")
            return 1
            
        return 0
        
    except KeyboardInterrupt:
        logger.info("Sistema interrompido pelo usuário")
        return 0
        
    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)
        return 1
        

if __name__ == '__main__':
    sys.exit(main())