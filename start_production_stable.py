"""
Script de Produção Estável
Baseado na abordagem do book_collector_continuous.py
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import time
import threading
import signal

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
log_file = f'logs/production/stable_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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

# Flag global
running = True
system = None

def signal_handler(sig, frame):
    """Handler para CTRL+C"""
    global running, system
    logger.info("\n[STOP] Sinal de interrupção recebido")
    running = False
    if system:
        try:
            system.stop()
        except:
            pass

def main():
    """Função principal"""
    global running, system
    
    # Registrar handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"\nLogs: {log_file}")
    print("\n" + "="*60)
    print("QUANTUM TRADER ML - VERSÃO ESTÁVEL")
    print("="*60)
    print(f"Data: {datetime.now()}")
    print("="*60)
    
    try:
        # Importar sistema
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
        
        logger.info("Criando sistema...")
        system = TradingSystem(config)
        
        # Aguardar um pouco para estabilizar
        logger.info("Aguardando estabilização...")
        time.sleep(2)
        
        logger.info("="*60)
        logger.info("INICIALIZANDO SISTEMA")
        logger.info("="*60)
        
        # Inicializar sem timeout - deixar falhar naturalmente se houver problema
        logger.info("Inicializando componentes...")
        if not system.initialize():
            logger.error("Falha na inicialização")
            return 1
            
        logger.info("[OK] Sistema inicializado!")
        
        # Aguardar conexões estabilizarem
        logger.info("Aguardando conexões estabilizarem...")
        time.sleep(3)
        
        logger.info("="*60)
        logger.info("INICIANDO SISTEMA DE TRADING")
        logger.info("="*60)
        
        if not system.start():
            logger.error("Falha ao iniciar")
            return 1
            
        logger.info("[OK] Sistema iniciado!")
        
        # Aguardar início dos dados
        logger.info("Aguardando recepção de dados...")
        time.sleep(5)
        
        logger.info("="*60)
        logger.info("SISTEMA OPERACIONAL")
        logger.info("="*60)
        logger.info("Configurações:")
        logger.info("- Máximo 1 contrato por vez")
        logger.info("- Stop loss: 0.5%")
        logger.info("- Limite diário: R$ 500")
        logger.info("- Para parar: CTRL+C")
        logger.info("="*60)
        
        # Estatísticas
        stats = {
            'start_time': time.time(),
            'last_status': time.time(),
            'last_data_check': time.time(),
            'data_received': False,
            'trades': 0,
            'errors': 0
        }
        
        # Loop principal
        while running:
            try:
                current_time = time.time()
                
                # Status a cada 30 segundos
                if current_time - stats['last_status'] > 30:
                    try:
                        status = system.get_status()
                        candles = status.get('candles', 0)
                        
                        # Primeira vez recebendo dados
                        if candles > 0 and not stats['data_received']:
                            stats['data_received'] = True
                            logger.info("[DATA] Sistema recebendo dados!")
                        
                        # Log periódico
                        runtime = int((current_time - stats['start_time']) / 60)
                        logger.info(f"[STATUS] Runtime: {runtime}min | Candles: {candles} | Running: {status.get('running')}")
                        
                        # Posições
                        if status.get('active_positions'):
                            logger.info(f"[POSITIONS] {status['active_positions']}")
                            
                    except Exception as e:
                        logger.debug(f"Erro status: {e}")
                        stats['errors'] += 1
                        
                    stats['last_status'] = current_time
                
                # Verificar dados a cada 5 minutos
                if current_time - stats['last_data_check'] > 300:
                    if not stats['data_received']:
                        logger.warning("[WARN] Sem dados ainda. Mercado pode estar fechado.")
                    else:
                        logger.info(f"[INFO] Sistema operacional há {int((current_time - stats['start_time'])/60)} minutos")
                    stats['last_data_check'] = current_time
                
                # Pequena pausa
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Erro no loop: {e}")
                stats['errors'] += 1
                if stats['errors'] > 10:
                    logger.error("Muitos erros. Parando sistema.")
                    break
                    
        logger.info("\n[STOP] Loop principal finalizado")
        
    except KeyboardInterrupt:
        logger.info("\n[STOP] Interrompido por CTRL+C")
    except Exception as e:
        logger.error(f"[ERROR] Erro crítico: {e}", exc_info=True)
    finally:
        running = False
        
        logger.info("="*60)
        logger.info("FINALIZANDO SISTEMA")
        logger.info("="*60)
        
        # Estatísticas finais
        if 'stats' in locals():
            runtime = (time.time() - stats['start_time']) / 60
            logger.info(f"Tempo total: {runtime:.1f} minutos")
            logger.info(f"Dados recebidos: {'SIM' if stats.get('data_received') else 'NÃO'}")
            logger.info(f"Erros: {stats.get('errors', 0)}")
        
        # Parar sistema
        if system:
            try:
                logger.info("Parando sistema...")
                system.stop()
                logger.info("[OK] Sistema parado")
            except Exception as e:
                logger.error(f"Erro ao parar: {e}")
                
        logger.info("="*60)
        logger.info("SISTEMA FINALIZADO")
        logger.info(f"Logs: {log_file}")
        logger.info("="*60)

if __name__ == "__main__":
    sys.exit(main())