"""
Script principal para executar HMARL com processos isolados
Previne Segmentation Fault executando ProfitDLL em processo separado
"""

import os
import sys
import time
import logging
import multiprocessing
import signal
from datetime import datetime
from dotenv import load_dotenv

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.integration.profit_dll_server import run_server
from src.integration.hmarl_client import HMARLClient

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HMARLIsolated')


class HMARLProcessManager:
    """
    Gerencia processos servidor e cliente HMARL
    """
    
    def __init__(self):
        self.server_process = None
        self.client = None
        self.is_running = False
        
        # Configurações
        self.server_config = {
            'dll_path': r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll",
            'username': os.getenv("PROFIT_USERNAME"),
            'password': os.getenv("PROFIT_PASSWORD"),
            'key': os.getenv("PROFIT_KEY"),
            'port': 6789
        }
        
        self.client_config = {
            'ticker': os.getenv('TICKER', 'WDOQ25'),
            'valkey': {
                'host': 'localhost',
                'port': 6379
            }
        }
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handler para sinais de interrupção"""
        logger.info("\nSinal de interrupção recebido")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Inicia servidor e cliente em processos separados"""
        try:
            self.is_running = True
            
            # 1. Iniciar servidor ProfitDLL em processo separado
            logger.info("🚀 Iniciando servidor ProfitDLL em processo isolado...")
            self.server_process = multiprocessing.Process(
                target=run_server,
                args=(self.server_config,),
                name="ProfitDLLServer"
            )
            self.server_process.daemon = True
            self.server_process.start()
            
            # 2. Aguardar servidor iniciar
            logger.info("⏳ Aguardando servidor inicializar...")
            
            # Verificar se servidor está vivo por até 30 segundos
            start_wait = time.time()
            server_ready = False
            
            while time.time() - start_wait < 30:
                if self.server_process.is_alive():
                    # Aguardar mais um pouco para servidor estar pronto
                    time.sleep(3)
                    server_ready = True
                    break
                time.sleep(1)
            
            # 3. Verificar se servidor está rodando
            if not server_ready or not self.server_process.is_alive():
                logger.error("❌ Servidor ProfitDLL falhou ao iniciar ou morreu")
                if self.server_process.exitcode is not None:
                    logger.error(f"Exit code do servidor: {self.server_process.exitcode}")
                return False
            
            logger.info("✅ Servidor ProfitDLL rodando no processo PID: {}".format(
                self.server_process.pid
            ))
            
            # 4. Iniciar cliente HMARL no processo principal
            logger.info("🤖 Iniciando cliente HMARL...")
            self.client = HMARLClient(
                server_address=('localhost', self.server_config['port']),
                config=self.client_config
            )
            
            # 5. Inicializar componentes HMARL
            if not self.client.initialize():
                logger.error("❌ Falha ao inicializar HMARL")
                return False
            
            # 6. Conectar ao servidor
            if not self.client.connect_to_server():
                logger.error("❌ Falha ao conectar ao servidor")
                return False
            
            # 7. Subscrever ticker
            ticker = self.client_config['ticker']
            logger.info(f"📊 Subscrevendo para {ticker}...")
            if self.client.subscribe_ticker(ticker):
                logger.info(f"✅ Subscrito para {ticker}")
            else:
                logger.warning(f"⚠️ Falha ao subscrever {ticker}")
            
            # 8. Iniciar processamento HMARL
            logger.info("🎯 Sistema HMARL iniciado com sucesso!")
            logger.info("="*60)
            logger.info("Arquitetura de processos isolados:")
            logger.info(f"  Processo 1 (PID {self.server_process.pid}): ProfitDLL Server")
            logger.info(f"  Processo 2 (PID {os.getpid()}): HMARL Client + Agentes")
            logger.info("="*60)
            
            # Executar cliente
            self.client.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Erro iniciando sistema: {e}", exc_info=True)
            return False
    
    def monitor(self):
        """Monitora status dos processos"""
        logger.info("\n📊 === MONITORAMENTO DO SISTEMA ===")
        
        while self.is_running:
            try:
                # Status do servidor
                server_alive = self.server_process.is_alive() if self.server_process else False
                logger.info(f"Servidor ProfitDLL: {'✅ Ativo' if server_alive else '❌ Inativo'}")
                
                # Métricas do cliente
                if self.client:
                    metrics = self.client.get_metrics()
                    logger.info("Métricas HMARL:")
                    logger.info(f"  - Mensagens recebidas: {metrics['messages_received']}")
                    logger.info(f"  - Trades processados: {metrics['trades_processed']}")
                    logger.info(f"  - Features calculadas: {metrics['features_calculated']}")
                    logger.info(f"  - Sinais de agentes: {metrics['agent_signals']}")
                    logger.info(f"  - Erros: {metrics['errors']}")
                    logger.info(f"  - Uptime: {metrics['uptime_seconds']:.0f}s")
                
                logger.info("="*50)
                
                # Verificar saúde do sistema
                if not server_alive:
                    logger.error("⚠️ Servidor caiu! Tentando reiniciar...")
                    self._restart_server()
                
                time.sleep(30)  # Monitorar a cada 30 segundos
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Erro no monitoramento: {e}")
                time.sleep(5)
    
    def _restart_server(self):
        """Tenta reiniciar o servidor"""
        try:
            if self.server_process:
                self.server_process.terminate()
                self.server_process.join(timeout=5)
            
            # Reiniciar
            self.server_process = multiprocessing.Process(
                target=run_server,
                args=(self.server_config,),
                name="ProfitDLLServer"
            )
            self.server_process.daemon = True
            self.server_process.start()
            
            time.sleep(5)
            
            if self.server_process.is_alive():
                logger.info("✅ Servidor reiniciado com sucesso")
                # Reconectar cliente
                if self.client:
                    self.client.connect_to_server()
            else:
                logger.error("❌ Falha ao reiniciar servidor")
                
        except Exception as e:
            logger.error(f"Erro reiniciando servidor: {e}")
    
    def stop(self):
        """Para todos os processos de forma limpa"""
        logger.info("\n🛑 Parando sistema HMARL...")
        
        self.is_running = False
        
        # Parar cliente
        if self.client:
            try:
                self.client.stop()
                logger.info("✅ Cliente HMARL parado")
            except Exception as e:
                logger.error(f"Erro parando cliente: {e}")
        
        # Parar servidor
        if self.server_process and self.server_process.is_alive():
            try:
                self.server_process.terminate()
                self.server_process.join(timeout=10)
                
                if self.server_process.is_alive():
                    logger.warning("Forçando término do servidor...")
                    self.server_process.kill()
                    self.server_process.join()
                
                logger.info("✅ Servidor ProfitDLL parado")
            except Exception as e:
                logger.error(f"Erro parando servidor: {e}")
        
        logger.info("✨ Sistema HMARL finalizado")


def main():
    """Função principal"""
    logger.info("="*60)
    logger.info("🚀 HMARL com Processos Isolados")
    logger.info("Solução para Segmentation Fault")
    logger.info("="*60)
    
    # Verificar pré-requisitos
    logger.info("🔍 Verificando pré-requisitos...")
    
    # Verificar credenciais
    required_vars = ['PROFIT_USERNAME', 'PROFIT_PASSWORD', 'PROFIT_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"❌ Variáveis de ambiente faltando: {missing}")
        logger.info("Configure as credenciais no arquivo .env")
        return
    
    # Verificar horário
    now = datetime.now()
    if now.weekday() >= 5:
        logger.warning("⚠️ AVISO: Mercado fechado (fim de semana)")
    elif now.hour < 9 or now.hour >= 18:
        logger.warning(f"⚠️ AVISO: Fora do horário de pregão ({now.hour}h)")
    
    logger.info("✅ Pré-requisitos verificados")
    
    # Criar e iniciar gerenciador
    manager = HMARLProcessManager()
    
    try:
        if manager.start():
            # Sistema iniciado com sucesso
            # O cliente está rodando no processo principal
            pass
        else:
            logger.error("❌ Falha ao iniciar sistema")
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Interrupção do usuário")
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}", exc_info=True)
    finally:
        manager.stop()


if __name__ == "__main__":
    # Necessário para Windows
    multiprocessing.freeze_support()
    
    main()