"""
Ponto de entrada principal do sistema v2.0 - Enhanced Version
Detecta automaticamente se deve usar sistema enhanced baseado nas configurações
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Adicionar src ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar configuração ZMQ/Valkey primeiro
from config.zmq_valkey_config import ZMQValkeyConfig

# Importar sistema apropriado
if ZMQValkeyConfig.is_enhanced_enabled():
    from trading_system_enhanced import TradingSystemEnhanced as TradingSystem
    SYSTEM_TYPE = "Enhanced"
else:
    from trading_system import TradingSystem
    SYSTEM_TYPE = "Original"


def load_config():
    """Carrega configuração do sistema"""
    # Tentar diferentes caminhos para o .env
    possible_env_paths = [
        # Diretório atual
        '.env',
        # Diretório pai do script
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'),
        # Caminho absoluto conhecido
        r'C:\Users\marth\OneDrive\Programacao\Python\Projetos\QuantumTrader_ML\.env'
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
        'ml_interval': int(os.getenv('ML_INTERVAL', '10')),
        'initial_balance': float(os.getenv('INITIAL_BALANCE', '100000')),
        'strategy': {
            'direction_threshold': float(os.getenv('DIRECTION_THRESHOLD', '0.45')),
            'magnitude_threshold': float(os.getenv('MAGNITUDE_THRESHOLD', '0.00015')),
            'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.45'))
        },
        'risk': {
            'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '0.02')),
            'max_positions': int(os.getenv('MAX_POSITIONS', '1')),
            'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.005'))
        },
        'use_gui': os.getenv('USE_GUI', 'False').lower() == 'true',
        'env_loaded': env_loaded
    }
    
    # Adicionar símbolos para criar streams (enhanced)
    ticker = config.get('ticker', 'WDOQ25')
    config['symbols'] = [ticker] if ticker else ['WDOQ25']
    
    return config


def setup_logging():
    """Configura sistema de logging"""
    # Configurar nível baseado em variável de ambiente
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/trading_main.log', mode='a')
        ]
    )
    
    # Desabilitar logs excessivos de bibliotecas
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return logging.getLogger('Main')


def main():
    """Função principal do sistema"""
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info(f"ML Trading System v2.0 - {SYSTEM_TYPE}")
    logger.info("="*60)
    
    # Mostrar status enhanced
    if SYSTEM_TYPE == "Enhanced":
        logger.info("🚀 Sistema Enhanced detectado!")
        status = ZMQValkeyConfig.get_status()
        logger.info(f"ZMQ: {'✅' if status['zmq'] else '❌'}")
        logger.info(f"Valkey: {'✅' if status['valkey'] else '❌'}")
        logger.info(f"Time Travel: {'✅' if status['time_travel'] else '❌'}")
        logger.info(f"Enhanced ML: {'✅' if status['enhanced_ml'] else '❌'}")
    else:
        logger.info("Sistema original em uso (enhanced desabilitado)")
    
    try:
        # Carregar configurações
        config = load_config()
        
        # Verificar se .env foi carregado
        if not config['env_loaded']:
            logger.warning("Arquivo .env não encontrado, usando valores padrão")
        
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
        
        # Log adicional para enhanced
        if SYSTEM_TYPE == "Enhanced" and hasattr(system, 'get_enhanced_status'):
            enhanced_status = system.get_enhanced_status()
            logger.info(f"Enhanced features ativas: {enhanced_status['enhanced_features']}")
            
        return 0
        
    except KeyboardInterrupt:
        logger.info("Sistema interrompido pelo usuário")
        
        # Parar sistema adequadamente se enhanced
        if 'system' in locals() and hasattr(system, 'stop'):
            logger.info("Parando sistema...")
            system.stop()
            
        return 0
        
    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)
        return 1
        

if __name__ == '__main__':
    sys.exit(main())