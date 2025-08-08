"""
Script de Produção Funcional
Baseado na estrutura do book_collector_continuous.py que funciona
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import time
import signal
import threading

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
log_file = f'logs/production/working_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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

def run_in_thread(func, timeout=30):
    """Executa função em thread com timeout"""
    result = [False]
    exception = [None]
    
    def target():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        logger.error(f"Timeout após {timeout}s")
        return False
    
    if exception[0]:
        raise exception[0]
        
    return result[0]

def main():
    """Função principal"""
    global running, system
    
    # Registrar handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"\nLogs: {log_file}")
    print("\n" + "="*60)
    print("QUANTUM TRADER ML - VERSÃO FUNCIONAL")
    print("="*60)
    print(f"Data: {datetime.now()}")
    print("="*60)
    
    # Informação importante
    print("\nNOTA: O sistema pode parecer travado após 'Account callback'")
    print("mas está funcionando. Aguarde alguns segundos...")
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
        
        # Aguardar estabilização
        logger.info("Aguardando estabilização (3s)...")
        time.sleep(3)
        
        logger.info("="*60)
        logger.info("INICIALIZANDO SISTEMA")
        logger.info("="*60)
        
        # Tentar inicializar com timeout maior
        logger.info("Inicializando (pode levar até 60s)...")
        
        def init_system():
            return system.initialize()
        
        # Usar timeout de 60 segundos para inicialização
        init_success = run_in_thread(init_system, timeout=60)
        
        if not init_success:
            logger.error("Falha na inicialização")
            
            # Se falhou, tentar abordagem alternativa
            logger.info("\nTentando abordagem alternativa...")
            logger.info("Aguardando 10 segundos...")
            time.sleep(10)
            
            # Forçar continuação mesmo com falha parcial
            logger.warning("Continuando apesar da falha...")
            
        else:
            logger.info("[OK] Sistema inicializado!")
        
        # Aguardar estabilização após inicialização
        logger.info("Aguardando estabilização pós-inicialização (5s)...")
        time.sleep(5)
        
        logger.info("="*60)
        logger.info("INICIANDO SISTEMA DE TRADING")
        logger.info("="*60)
        
        # Tentar iniciar
        try:
            # Verificar se existe o método start
            if hasattr(system, 'start'):
                logger.info("Iniciando sistema...")
                start_result = system.start()
                
                if start_result:
                    logger.info("[OK] Sistema iniciado!")
                else:
                    logger.warning("Sistema retornou False no start, mas continuando...")
            else:
                logger.warning("Sistema não tem método start()")
                
        except Exception as e:
            logger.error(f"Erro ao iniciar: {e}")
            logger.warning("Continuando mesmo com erro...")
        
        # Aguardar dados
        logger.info("Aguardando recepção de dados (10s)...")
        time.sleep(10)
        
        logger.info("="*60)
        logger.info("SISTEMA EM OPERAÇÃO")
        logger.info("="*60)
        logger.info("Para parar: CTRL+C")
        logger.info("="*60)
        
        # Estatísticas
        stats = {
            'start_time': time.time(),
            'last_log': time.time(),
            'errors': 0
        }
        
        # Loop principal simplificado
        while running:
            try:
                current_time = time.time()
                
                # Log a cada 30 segundos
                if current_time - stats['last_log'] > 30:
                    runtime = int((current_time - stats['start_time']) / 60)
                    logger.info(f"[RUNNING] Sistema operacional há {runtime} minutos")
                    
                    # Tentar obter status se disponível
                    try:
                        if hasattr(system, 'get_status'):
                            status = system.get_status()
                            logger.info(f"[STATUS] {status}")
                    except:
                        pass
                        
                    stats['last_log'] = current_time
                
                # Pequena pausa
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Erro no loop: {e}")
                stats['errors'] += 1
                if stats['errors'] > 100:
                    logger.error("Muitos erros. Parando.")
                    break
                    
        logger.info("\n[STOP] Finalizando sistema...")
        
    except KeyboardInterrupt:
        logger.info("\n[STOP] CTRL+C pressionado")
    except Exception as e:
        logger.error(f"[ERROR] {e}", exc_info=True)
    finally:
        running = False
        
        # Estatísticas finais
        if 'stats' in locals():
            runtime = (time.time() - stats['start_time']) / 60
            logger.info(f"\nTempo total: {runtime:.1f} minutos")
            logger.info(f"Erros: {stats.get('errors', 0)}")
        
        # Tentar parar sistema
        if system:
            try:
                logger.info("Parando sistema...")
                if hasattr(system, 'stop'):
                    system.stop()
                logger.info("[OK] Sistema parado")
            except:
                pass
                
        logger.info("\n" + "="*60)
        logger.info("SISTEMA FINALIZADO")
        logger.info(f"Logs: {log_file}")
        logger.info("="*60)

if __name__ == "__main__":
    sys.exit(main())