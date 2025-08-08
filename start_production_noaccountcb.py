"""
Script de Produção - Sem Callback de Conta
Versão temporária que desabilita callback problemático
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import time

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/production/noaccountcb_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Patch temporário para desabilitar callback de conta
def patch_connection_manager():
    """Aplica patch temporário no connection manager"""
    try:
        import src.connection_manager_v4 as cm
        
        # Salvar método original
        original_configure_callbacks = cm.ConnectionManagerV4._configure_callbacks_v4
        
        def patched_configure_callbacks(self):
            """Versão patchada sem callback de conta"""
            logger.info("[PATCH] Configurando callbacks sem account callback...")
            
            # Chamar original mas interceptar
            original_configure_callbacks(self)
            
            # Limpar callback de conta se existir
            if hasattr(self, 'account_callbacks'):
                self.account_callbacks = []
            
            logger.info("[PATCH] Callbacks configurados (account callback desabilitado)")
            
        # Aplicar patch
        cm.ConnectionManagerV4._configure_callbacks_v4 = patched_configure_callbacks
        logger.info("[PATCH] Connection manager patchado com sucesso")
        
    except Exception as e:
        logger.error(f"Erro ao aplicar patch: {e}")

def main():
    """Inicialização do sistema sem callback de conta"""
    
    print("\n" + "="*60)
    print("QUANTUM TRADER ML - PRODUÇÃO (SEM ACCOUNT CALLBACK)")
    print("="*60)
    print(f"Data: {datetime.now()}")
    print("="*60)
    
    # Aplicar patch antes de importar sistema
    patch_connection_manager()
    
    # Criar diretórios
    Path("logs/production").mkdir(parents=True, exist_ok=True)
    
    try:
        # Agora importar sistema
        from src.trading_system import TradingSystem
        
        # Configuração
        config = {
            'dll_path': os.getenv('PROFIT_DLL_PATH', 'ProfitDLL64.dll'),
            'key': os.getenv('PROFIT_KEY', ''),
            'username': os.getenv('PROFIT_USERNAME', ''),
            'password': os.getenv('PROFIT_PASSWORD', ''),
            'account_id': os.getenv('PROFIT_ACCOUNT_ID', ''),
            'broker_id': os.getenv('PROFIT_BROKER_ID', ''),
            'trading_password': os.getenv('PROFIT_TRADING_PASSWORD', ''),
            'models_dir': 'models',
            'ticker': 'WDOU25',
            'symbols': ['WDOU25'],
            'historical_days': 1,
            'initial_capital': 50000.0,
            'max_position_size': 1,
            'max_daily_loss': 500.0,
            'stop_loss_pct': 0.005,
            'strategy': {
                'confidence_threshold': 0.75,
                'direction_threshold': 0.6,
                'magnitude_threshold': 0.002,
            },
            'risk': {
                'max_position_size': 1,
                'stop_loss_pct': 0.005,
                'take_profit_pct': 0.01,
            }
        }
        
        logger.info("Criando sistema de trading...")
        system = TradingSystem(config)
        
        logger.info("INICIALIZANDO sistema...")
        if not system.initialize():
            logger.error("Falha na inicialização")
            return 1
            
        logger.info("[OK] Sistema inicializado!")
        
        logger.info("INICIANDO sistema...")
        if not system.start():
            logger.error("Falha ao iniciar")
            return 1
            
        logger.info("[OK] Sistema iniciado!")
        logger.info("Sistema operacional!")
        logger.info("Para parar: CTRL+C ou crie arquivo STOP_TRADING.txt")
        
        # Loop principal
        stop_file = Path("STOP_TRADING.txt")
        
        while True:
            if stop_file.exists():
                logger.info("Arquivo de parada detectado.")
                break
                
            # Status a cada 30 segundos
            time.sleep(30)
            
            try:
                status = system.get_status()
                logger.info(f"Sistema rodando - candles: {status.get('candles', 0)}")
                
                # Se houver dados, mostrar
                if status.get('candles', 0) > 0:
                    logger.info("[DATA] Recebendo dados de mercado!")
                    
            except Exception as e:
                logger.debug(f"Erro status: {e}")
                
    except KeyboardInterrupt:
        logger.info("\nInterrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro: {e}", exc_info=True)
    finally:
        logger.info("Finalizando...")
        try:
            if 'system' in locals():
                system.stop()
        except:
            pass
        logger.info("Sistema finalizado")

if __name__ == "__main__":
    main()