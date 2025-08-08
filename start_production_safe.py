"""
Script de Produção Seguro - Sem Callback de Conta
Usa versão modificada do connection manager para evitar segfault
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import time
import signal

# IMPORTANTE: Substituir temporariamente o connection_manager_v4
# Fazer backup do original primeiro
import shutil

# Fazer backup se ainda não existe
if not os.path.exists("src/connection_manager_v4_original.py"):
    shutil.copy2("src/connection_manager_v4.py", "src/connection_manager_v4_original.py")

# Substituir pelo safe
shutil.copy2("src/connection_manager_v4_safe.py", "src/connection_manager_v4.py")

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
log_file = f'logs/production/safe_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
    global running
    logger.info("\n[STOP] Sinal de interrupção recebido")
    running = False
    # Restaurar arquivo original
    restore_original_connection_manager()

def restore_original_connection_manager():
    """Restaura o connection manager original"""
    try:
        if os.path.exists("src/connection_manager_v4_original.py"):
            shutil.copy2("src/connection_manager_v4_original.py", "src/connection_manager_v4.py")
            logger.info("Connection manager original restaurado")
    except:
        pass

def main():
    """Função principal"""
    global running, system
    
    # Registrar handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"\nLogs: {log_file}")
    print("\n" + "="*60)
    print("QUANTUM TRADER ML - VERSÃO SEGURA")
    print("="*60)
    print(f"Data: {datetime.now()}")
    print("="*60)
    print("\n[INFO] Usando connection manager sem callback de conta")
    print("[INFO] Isso evita o segmentation fault conhecido")
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
        
        logger.info("Criando sistema de trading...")
        system = TradingSystem(config)
        
        logger.info("Aguardando estabilização...")
        time.sleep(2)
        
        logger.info("="*60)
        logger.info("INICIALIZANDO SISTEMA")
        logger.info("="*60)
        
        if not system.initialize():
            logger.error("Falha na inicialização")
            return 1
            
        logger.info("[OK] Sistema inicializado!")
        
        # Aguardar conexões
        logger.info("Aguardando conexões estabilizarem...")
        time.sleep(3)
        
        logger.info("="*60)
        logger.info("INICIANDO SISTEMA DE TRADING")
        logger.info("="*60)
        
        if not system.start():
            logger.error("Falha ao iniciar")
            return 1
            
        logger.info("[OK] Sistema iniciado com sucesso!")
        
        # Aguardar dados
        logger.info("Aguardando dados de mercado...")
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
            'data_received': False,
            'candles': 0,
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
                        
                        # Verificar se está recebendo dados
                        if candles > stats['candles']:
                            stats['candles'] = candles
                            if not stats['data_received']:
                                stats['data_received'] = True
                                logger.info("[DATA] Sistema está recebendo dados!")
                        
                        # Log de status
                        runtime = int((current_time - stats['start_time']) / 60)
                        logger.info(f"[STATUS] Runtime: {runtime}min | Candles: {candles} | Running: {status.get('running')}")
                        
                        # Mostrar posições se houver
                        if status.get('active_positions'):
                            logger.info(f"[POSITIONS] {status['active_positions']}")
                            
                    except Exception as e:
                        logger.debug(f"Erro ao obter status: {e}")
                        stats['errors'] += 1
                        
                    stats['last_status'] = current_time
                
                # Pequena pausa
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Erro no loop: {e}")
                stats['errors'] += 1
                if stats['errors'] > 10:
                    logger.error("Muitos erros. Parando sistema.")
                    break
                    
        logger.info("\n[STOP] Finalizando sistema...")
        
    except KeyboardInterrupt:
        logger.info("\n[STOP] Interrompido por CTRL+C")
    except Exception as e:
        logger.error(f"[ERROR] Erro crítico: {e}", exc_info=True)
    finally:
        running = False
        
        # Estatísticas finais
        if 'stats' in locals():
            runtime = (time.time() - stats['start_time']) / 60
            logger.info(f"\nTempo total: {runtime:.1f} minutos")
            logger.info(f"Dados recebidos: {'SIM' if stats.get('data_received') else 'NÃO'}")
            logger.info(f"Total de candles: {stats.get('candles', 0)}")
            logger.info(f"Erros: {stats.get('errors', 0)}")
        
        # Parar sistema
        if system:
            try:
                logger.info("Parando sistema de trading...")
                system.stop()
                logger.info("[OK] Sistema parado")
            except Exception as e:
                logger.error(f"Erro ao parar: {e}")
        
        # Restaurar arquivo original
        restore_original_connection_manager()
        
        logger.info("\n" + "="*60)
        logger.info("SISTEMA FINALIZADO")
        logger.info(f"Logs: {log_file}")
        logger.info("="*60)

if __name__ == "__main__":
    sys.exit(main())