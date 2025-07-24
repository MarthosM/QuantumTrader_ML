"""
Ponto de entrada principal do sistema v2.0 - Versão Universal
Funciona tanto executado da raiz quanto do src
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv


def setup_paths():
    """Configura paths de forma inteligente"""
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    
    # Detectar se estamos em src/ ou na raiz
    if current_dir.name == 'src':
        # Estamos em src/, adicionar src ao path
        project_root = current_dir.parent
        src_path = current_dir
    else:
        # Estamos na raiz, adicionar src ao path
        project_root = current_dir
        src_path = current_dir / 'src'
    
    # Adicionar src ao sys.path se não estiver
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Mudar para diretório src para importações
    os.chdir(str(src_path))
    
    return project_root, src_path


def load_config():
    """Carrega configuração do sistema"""
    project_root, src_path = setup_paths()
    
    # Tentar diferentes caminhos para o .env
    possible_env_paths = [
        # Na raiz do projeto
        project_root / '.env',
        # No diretório atual
        Path('.env'),
        # Caminho absoluto conhecido
        Path(r'C:\Users\marth\OneDrive\Programacao\Python\Projetos\ML_Tradingv2.0\.env')
    ]
    
    env_loaded = False
    for env_path in possible_env_paths:
        if env_path.exists():
            load_dotenv(str(env_path))
            env_loaded = True
            print(f"✅ .env carregado: {env_path}")
            break
    
    if not env_loaded:
        print("⚠️ Arquivo .env não encontrado!")
        print("Caminhos testados:")
        for path in possible_env_paths:
            print(f"  - {path}")
    
    config = {
        'dll_path': os.getenv("PROFIT_DLL_PATH", r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"),
        'key': os.getenv('PROFIT_KEY'),
        'username': os.getenv('PROFIT_USER'),
        'password': os.getenv('PROFIT_PASSWORD'),
        'account_id': os.getenv('PROFIT_ACCOUNT_ID'),
        'broker_id': os.getenv('PROFIT_BROKER_ID'),
        'trading_password': os.getenv('PROFIT_TRADING_PASSWORD'),
        'models_dir': os.getenv('MODELS_DIR', str(src_path / 'models' / 'models_regime3')),
        'ticker': os.getenv('TICKER'),
        'historical_days': int(os.getenv('HISTORICAL_DAYS', '1')),
        'ml_interval': int(os.getenv('ML_INTERVAL', '60')),
        'initial_balance': float(os.getenv('INITIAL_BALANCE', '100000')),
        'strategy': {
            'direction_threshold': float(os.getenv('DIRECTION_THRESHOLD', '0.45')),
            'magnitude_threshold': float(os.getenv('MAGNITUDE_THRESHOLD', '0.00015')),
            'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.6')),
            'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.02')),
            'max_positions': int(os.getenv('MAX_POSITIONS', '1')),
            'stop_loss_points': int(os.getenv('STOP_LOSS_POINTS', '10')),
            'take_profit_points': int(os.getenv('TAKE_PROFIT_POINTS', '20'))
        }
    }
    
    return config


def main():
    """Função principal"""
    print("=" * 60)
    print("    ML TRADING v2.0 - MAIN.PY UNIVERSAL")
    print("=" * 60)
    
    try:
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('ml_trading_main.log', encoding='utf-8')
            ]
        )
        
        logger = logging.getLogger('Main')
        logger.info("Iniciando Sistema de Trading v2.0")
        
        # Configurar paths
        project_root, src_path = setup_paths()
        logger.info(f"Projeto: {project_root}")
        logger.info(f"Src: {src_path}")
        logger.info(f"Working dir: {os.getcwd()}")
        
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
        
        # Importar TradingSystem (agora que o path está configurado)
        try:
            from trading_system import TradingSystem
            logger.info("✅ TradingSystem importado com sucesso")
        except ImportError as e:
            logger.error(f"❌ Erro importando TradingSystem: {e}")
            logger.error(f"Sys.path: {sys.path}")
            logger.error(f"Current dir: {os.getcwd()}")
            return 1
            
        # Criar sistema
        logger.info("Criando sistema de trading...")
        system = TradingSystem(config)
        
        # Inicializar
        logger.info("Inicializando sistema...")
        if not system.initialize():
            logger.error("Falha na inicialização do sistema")
            return 1
            
        # Iniciar operação
        logger.info("Iniciando operação...")
        if not system.start():
            logger.error("Falha ao iniciar operação")
            return 1
            
        logger.info("✅ Sistema iniciado com sucesso!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Sistema interrompido pelo usuário")
        return 0
        
    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)
        return 1
        

if __name__ == '__main__':
    sys.exit(main())
