"""
Script Correto para Produção
Inicializa e inicia o sistema corretamente
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/production/correct_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Inicialização correta do sistema"""
    
    print("\n" + "="*60)
    print("QUANTUM TRADER ML - PRODUÇÃO CORRETA")
    print("="*60)
    print(f"Data: {datetime.now()}")
    print("="*60)
    
    # Criar diretórios
    Path("logs/production").mkdir(parents=True, exist_ok=True)
    
    try:
        # Importar e criar sistema de trading
        from src.trading_system import TradingSystem
        
        # Configuração completa
        config = {
            # Conexão ProfitDLL
            'dll_path': os.getenv('PROFIT_DLL_PATH', 'ProfitDLL64.dll'),
            'key': os.getenv('PROFIT_KEY', ''),
            'username': os.getenv('PROFIT_USERNAME', ''),
            'password': os.getenv('PROFIT_PASSWORD', ''),
            'account_id': os.getenv('PROFIT_ACCOUNT_ID', ''),
            'broker_id': os.getenv('PROFIT_BROKER_ID', ''),
            'trading_password': os.getenv('PROFIT_TRADING_PASSWORD', ''),
            
            # Configurações do sistema
            'models_dir': 'models',
            'ticker': 'WDOU25',
            'symbols': ['WDOU25'],
            'historical_days': 1,
            
            # Risco
            'initial_capital': 50000.0,
            'max_position_size': 1,
            'max_daily_loss': 500.0,
            'stop_loss_pct': 0.005,
            
            # Estratégia
            'strategy': {
                'confidence_threshold': 0.75,
                'direction_threshold': 0.6,
                'magnitude_threshold': 0.002,
            },
            
            # Risk management
            'risk': {
                'max_position_size': 1,
                'stop_loss_pct': 0.005,
                'take_profit_pct': 0.01,
            }
        }
        
        logger.info("Criando sistema de trading...")
        system = TradingSystem(config)
        
        logger.info("INICIALIZANDO sistema (passo importante!)...")
        if not system.initialize():
            logger.error("Falha na inicialização do sistema")
            return 1
            
        logger.info("[OK] Sistema inicializado!")
        
        logger.info("INICIANDO sistema...")
        if not system.start():
            logger.error("Falha ao iniciar sistema")
            return 1
            
        logger.info("[OK] Sistema iniciado!")
        logger.info("Para parar: CTRL+C ou crie arquivo STOP_TRADING.txt")
        
        # Loop principal
        stop_file = Path("STOP_TRADING.txt")
        
        while True:
            # Verificar arquivo de parada
            if stop_file.exists():
                logger.info("Arquivo de parada detectado. Finalizando...")
                break
            
            # Status a cada 30 segundos
            import time
            time.sleep(30)
            
            try:
                status = system.get_status()
                logger.info(f"Status: running={status['running']}, initialized={status['initialized']}, candles={status['candles']}")
                
                # Mostrar métricas se disponíveis
                if status['metrics']:
                    logger.info(f"Métricas: {status['metrics']}")
                    
                # Mostrar posições se houver
                if status['active_positions']:
                    logger.info(f"Posições ativas: {status['active_positions']}")
                    
            except Exception as e:
                logger.error(f"Erro ao obter status: {e}")
                
    except KeyboardInterrupt:
        logger.info("\nInterrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro: {e}", exc_info=True)
    finally:
        logger.info("Finalizando sistema...")
        try:
            if 'system' in locals():
                system.stop()
        except:
            pass
        logger.info("Sistema finalizado")

if __name__ == "__main__":
    main()