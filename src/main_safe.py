"""
Main Safe - Versão segura do sistema principal
Evita o callback problemático que causa segmentation fault
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import signal
import time

# Adicionar o diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/main_safe_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('MainSafe')

# Handler global para CTRL+C
running = True

def signal_handler(sig, frame):
    global running
    logger.info("\nSinal de interrupção recebido. Finalizando...")
    running = False

def main():
    """Função principal"""
    global running
    
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("="*60)
    logger.info("Sistema de Trading v2.0 - Versão Segura")
    logger.info("="*60)
    
    # Verificar campos obrigatórios
    required_fields = ['PROFIT_KEY', 'PROFIT_USER', 'PROFIT_PASSWORD', 
                      'PROFIT_ACCOUNT_ID', 'PROFIT_BROKER_ID']
    
    missing_fields = []
    for field in required_fields:
        if not os.getenv(field):
            # Tentar com PROFIT_USERNAME também
            if field == 'PROFIT_USER' and os.getenv('PROFIT_USERNAME'):
                continue
            missing_fields.append(field.lower().replace('PROFIT_', ''))
    
    if missing_fields:
        logger.error(f"Campos obrigatórios não configurados no .env: {', '.join(missing_fields)}")
        return 1
    
    # Configuração
    config = {
        'dll_path': os.getenv("PROFIT_DLL_PATH", "ProfitDLL64.dll"),
        'key': os.getenv('PROFIT_KEY'),
        'username': os.getenv('PROFIT_USER') or os.getenv('PROFIT_USERNAME'),
        'password': os.getenv('PROFIT_PASSWORD'),
        'account_id': os.getenv('PROFIT_ACCOUNT_ID'),
        'broker_id': os.getenv('PROFIT_BROKER_ID'),
        'trading_password': os.getenv('PROFIT_TRADING_PASSWORD'),
        'initial_capital': float(os.getenv('INITIAL_BALANCE', '50000')),
        'ticker': os.getenv('TICKER', 'WDOU25'),
        'models_dir': 'models',
        'use_gui': os.getenv('USE_GUI', 'False').lower() == 'true',
        'historical_days': int(os.getenv('HISTORICAL_DAYS', '1')),
        'ml_interval': int(os.getenv('ML_INTERVAL', '60')),
        'disable_account_callback': True  # IMPORTANTE: Desabilitar callback problemático
    }
    
    # Log de configuração
    logger.info(f"Usuário: {config['username']}")
    logger.info(f"Conta: {config['account_id']}")
    logger.info(f"Corretora: {config['broker_id']}")
    logger.info(f"Ticker: {config['ticker']}")
    logger.info("Callback de conta: DESABILITADO (prevenção de segfault)")
    
    try:
        # Usar versão modificada do TradingSystem
        from trading_system_v2_safe import TradingSystemV2Safe
        
        # Criar sistema
        system = TradingSystemV2Safe(config)
        
        # Inicializar
        logger.info("\nInicializando sistema...")
        if not system.initialize():
            logger.error("Falha na inicialização do sistema")
            return 1
        
        logger.info("Sistema inicializado com sucesso!")
        
        # Aguardar estabilização
        logger.info("Aguardando estabilização...")
        time.sleep(3)
        
        # Iniciar sistema
        logger.info("Iniciando sistema de trading...")
        if not system.start():
            logger.error("Falha ao iniciar sistema")
            return 1
        
        logger.info("\n" + "="*60)
        logger.info("SISTEMA OPERACIONAL")
        logger.info("Para parar: CTRL+C")
        logger.info("="*60 + "\n")
        
        # Loop principal
        start_time = time.time()
        last_status = time.time()
        
        while running and system.is_running():
            try:
                current_time = time.time()
                
                # Status a cada 30 segundos
                if current_time - last_status > 30:
                    status = system.get_status()
                    runtime = (current_time - start_time) / 60
                    
                    logger.info(f"\n[STATUS] Runtime: {runtime:.1f}min")
                    logger.info(f"Conexão: {'OK' if status.get('connected') else 'ERRO'}")
                    logger.info(f"Candles: {status.get('candles_count', 0)}")
                    logger.info(f"Última predição: {status.get('last_prediction_time', 'N/A')}")
                    logger.info(f"Posições ativas: {status.get('active_positions', 0)}")
                    
                    last_status = current_time
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Erro no loop principal: {e}")
                time.sleep(5)
        
        # Finalizar
        logger.info("\nFinalizando sistema...")
        system.stop()
        
        # Estatísticas finais
        final_status = system.get_final_statistics()
        if final_status:
            logger.info("\n" + "="*60)
            logger.info("ESTATÍSTICAS FINAIS")
            logger.info("="*60)
            logger.info(f"Tempo total: {final_status.get('runtime_minutes', 0):.1f} minutos")
            logger.info(f"Trades executados: {final_status.get('total_trades', 0)}")
            logger.info(f"P&L Total: R$ {final_status.get('total_pnl', 0):.2f}")
            logger.info(f"Win Rate: {final_status.get('win_rate', 0):.1%}")
            logger.info(f"Predições ML: {final_status.get('ml_predictions', 0)}")
            logger.info("="*60)
        
        logger.info("Sistema finalizado com sucesso!")
        return 0
        
    except ImportError:
        # Se não existir versão safe, criar uma mínima
        logger.warning("Versão safe não encontrada. Usando sistema simplificado...")
        
        # Importar sistema simplificado
        from trading_system import TradingSystem
        
        # Adicionar flag para desabilitar callback
        config['safe_mode'] = True
        
        system = TradingSystem(config)
        
        if not system.initialize():
            logger.error("Falha na inicialização")
            return 1
            
        if not system.start():
            logger.error("Falha ao iniciar")
            return 1
            
        logger.info("Sistema iniciado em modo seguro!")
        
        # Loop simplificado
        while running:
            time.sleep(1)
            
        system.stop()
        return 0
        
    except Exception as e:
        logger.error(f"Erro crítico: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())