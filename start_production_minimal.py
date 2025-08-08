"""
Script Mínimo para Iniciar Produção
Versão simplificada para teste inicial
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
        logging.FileHandler(f'logs/production/minimal_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Inicialização mínima do sistema"""
    
    print("\n" + "="*60)
    print("QUANTUM TRADER ML - PRODUÇÃO MÍNIMA")
    print("="*60)
    print(f"Data: {datetime.now()}")
    print("="*60)
    
    # Criar diretórios
    Path("logs/production").mkdir(parents=True, exist_ok=True)
    
    try:
        # Importar e criar sistema de trading
        from src.trading_system import TradingSystem
        
        # Configuração mínima
        config = {
            'connection': {
                'host': '127.0.0.1',
                'port': 8001,
                'username': os.getenv('PROFIT_USERNAME', ''),
                'password': os.getenv('PROFIT_PASSWORD', ''),
                'symbols': ['WDOU25'],
            },
            'risk': {
                'initial_capital': 50000.0,
                'max_position_size': 1,
                'max_daily_loss': 500.0,
                'stop_loss_pct': 0.005,
            },
            'strategy': {
                'confidence_threshold': 0.75,
            },
            'ml': {
                'models_path': 'models',
            }
        }
        
        logger.info("Criando sistema de trading...")
        system = TradingSystem(config)
        
        logger.info("Inicializando sistema...")
        system.start()
        
        logger.info("[OK] Sistema iniciado!")
        logger.info("Para parar: CTRL+C ou crie arquivo STOP_TRADING.txt")
        
        # Loop principal simplificado
        stop_file = Path("STOP_TRADING.txt")
        
        while True:
            # Verificar arquivo de parada
            if stop_file.exists():
                logger.info("Arquivo de parada detectado. Finalizando...")
                break
            
            # Status básico a cada 30 segundos
            import time
            time.sleep(30)
            
            try:
                logger.info("Sistema rodando...")
                # Tentar obter status se disponível
                if hasattr(system, 'get_status'):
                    status = system.get_status()
                    logger.info(f"Status: {status}")
            except Exception as e:
                logger.debug(f"Erro ao obter status: {e}")
                
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