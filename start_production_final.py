"""
Script Final de Produção
Versão estável para executar o sistema de trading
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import time
import signal

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
log_file = f'logs/production/trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
Path("logs/production").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Flag para controle
running = True

def signal_handler(sig, frame):
    """Handler para CTRL+C"""
    global running
    logger.info("\n[INFO] Sinal de interrupção recebido. Finalizando...")
    running = False

def main():
    """Função principal"""
    global running
    
    # Registrar handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"\nLogs salvos em: {log_file}")
    print("\n" + "="*60)
    print("QUANTUM TRADER ML - SISTEMA DE PRODUÇÃO")
    print("="*60)
    print("MODO: CONTA SIMULADOR")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        # Importar sistema
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
            
            # Sistema
            'models_dir': 'models',
            'ticker': 'WDOU25',
            'symbols': ['WDOU25'],
            'historical_days': 1,
            
            # Capital e Risco
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
        
        logger.info("="*60)
        logger.info("INICIALIZANDO SISTEMA")
        logger.info("="*60)
        
        # Usar timeout para evitar travamento
        import threading
        init_success = False
        
        def init_system():
            nonlocal init_success
            try:
                init_success = system.initialize()
            except Exception as e:
                logger.error(f"Erro na inicialização: {e}")
        
        # Executar em thread separada
        init_thread = threading.Thread(target=init_system)
        init_thread.start()
        init_thread.join(timeout=30)
        
        if init_thread.is_alive():
            logger.error("Timeout na inicialização (30s)")
            return 1
            
        if not init_success:
            logger.error("Falha na inicialização")
            return 1
            
        logger.info("[OK] Sistema inicializado com sucesso!")
        
        logger.info("="*60)
        logger.info("INICIANDO SISTEMA")
        logger.info("="*60)
        
        # Iniciar sistema
        start_success = False
        
        def start_system():
            nonlocal start_success
            try:
                start_success = system.start()
            except Exception as e:
                logger.error(f"Erro ao iniciar: {e}")
        
        start_thread = threading.Thread(target=start_system)
        start_thread.start()
        start_thread.join(timeout=15)
        
        if start_thread.is_alive():
            logger.error("Timeout ao iniciar (15s)")
            return 1
            
        if not start_success:
            logger.error("Falha ao iniciar sistema")
            return 1
            
        logger.info("[OK] Sistema iniciado com sucesso!")
        logger.info("="*60)
        logger.info("SISTEMA OPERACIONAL")
        logger.info("="*60)
        logger.info("- Máximo 1 contrato por vez")
        logger.info("- Stop loss: 0.5%")
        logger.info("- Limite diário: R$ 500")
        logger.info("- Para parar: CTRL+C ou crie STOP_TRADING.txt")
        logger.info("="*60)
        
        # Loop principal
        stop_file = Path("STOP_TRADING.txt")
        last_status = time.time()
        last_data_check = time.time()
        data_received = False
        
        while running:
            # Verificar arquivo de parada
            if stop_file.exists():
                logger.info("[STOP] Arquivo STOP_TRADING.txt detectado")
                break
            
            # Status a cada 30 segundos
            if time.time() - last_status > 30:
                try:
                    status = system.get_status()
                    candles = status.get('candles', 0)
                    
                    logger.info(f"[STATUS] Sistema rodando | Candles: {candles} | Initialized: {status.get('initialized')}")
                    
                    # Verificar se está recebendo dados
                    if candles > 0 and not data_received:
                        data_received = True
                        logger.info("[DATA] Sistema está recebendo dados de mercado!")
                        
                    # Mostrar posições se houver
                    if status.get('active_positions'):
                        logger.info(f"[POSITIONS] {status['active_positions']}")
                        
                except Exception as e:
                    logger.debug(f"Erro ao obter status: {e}")
                    
                last_status = time.time()
            
            # Verificar dados a cada 5 minutos
            if time.time() - last_data_check > 300:
                if not data_received:
                    logger.warning("[WARNING] Sistema não está recebendo dados. Verifique se o mercado está aberto.")
                last_data_check = time.time()
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("\n[STOP] Interrompido pelo usuário (CTRL+C)")
    except Exception as e:
        logger.error(f"[ERROR] Erro crítico: {e}", exc_info=True)
    finally:
        running = False
        logger.info("="*60)
        logger.info("FINALIZANDO SISTEMA")
        logger.info("="*60)
        
        # Tentar parar sistema
        try:
            if 'system' in locals():
                logger.info("Parando sistema de trading...")
                
                def stop_system():
                    try:
                        system.stop()
                    except:
                        pass
                
                stop_thread = threading.Thread(target=stop_system)
                stop_thread.start()
                stop_thread.join(timeout=10)
                
                if stop_thread.is_alive():
                    logger.warning("Timeout ao parar sistema")
                else:
                    logger.info("[OK] Sistema parado com sucesso")
        except:
            pass
            
        logger.info("="*60)
        logger.info("SISTEMA FINALIZADO")
        logger.info(f"Logs salvos em: {log_file}")
        logger.info("="*60)

if __name__ == "__main__":
    sys.exit(main())