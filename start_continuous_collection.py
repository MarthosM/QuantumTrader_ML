"""
Script de inicialização para coleta contínua
Gerencia o processo de coleta com recuperação automática
"""

import subprocess
import time
import logging
from datetime import datetime, time as dtime
import sys
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_collection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('CollectionManager')

class CollectionManager:
    def __init__(self):
        self.process = None
        self.market_open = dtime(9, 0)
        self.market_close = dtime(18, 0)
        self.restart_count = 0
        self.max_restarts = 10
        
    def is_market_open(self):
        """Verifica se o mercado está aberto"""
        now = datetime.now()
        current_time = now.time()
        weekday = now.weekday()
        
        # Fim de semana
        if weekday >= 5:
            return False
            
        # Horário de mercado
        return self.market_open <= current_time <= self.market_close
        
    def start_collector(self):
        """Inicia o processo de coleta"""
        try:
            logger.info("Iniciando book_collector_continuous.py...")
            
            # Executar em subprocess com output em tempo real
            self.process = subprocess.Popen(
                [sys.executable, 'book_collector_continuous.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combinar stderr com stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            logger.info(f"Processo iniciado com PID: {self.process.pid}")
            
            # Thread para ler output em tempo real
            import threading
            self.output_thread = threading.Thread(
                target=self._read_output,
                daemon=True
            )
            self.output_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao iniciar coletor: {e}")
            return False
            
    def _read_output(self):
        """Lê output do processo em tempo real"""
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    # Mostrar output do collector
                    print(f"[COLLECTOR] {line.strip()}")
                    
                if self.process.poll() is not None:
                    break
                    
        except Exception as e:
            logger.error(f"Erro ao ler output: {e}")
            
    def monitor_process(self):
        """Monitora o processo e reinicia se necessário"""
        while True:
            try:
                # Verificar se processo está rodando
                if self.process and self.process.poll() is not None:
                    # Processo terminou
                    exit_code = self.process.returncode
                    logger.warning(f"Processo terminou com código: {exit_code}")
                    
                    # Output já está sendo lido pela thread
                    
                    # Verificar se deve reiniciar
                    if self.restart_count < self.max_restarts and self.is_market_open():
                        self.restart_count += 1
                        logger.info(f"Reiniciando coletor ({self.restart_count}/{self.max_restarts})...")
                        time.sleep(10)  # Aguardar antes de reiniciar
                        
                        if self.start_collector():
                            logger.info("Coletor reiniciado com sucesso")
                        else:
                            logger.error("Falha ao reiniciar coletor")
                            break
                    else:
                        logger.info("Limite de reinicializações atingido ou mercado fechado")
                        break
                        
                # Verificar se mercado fechou
                if not self.is_market_open():
                    logger.info("Mercado fechado. Encerrando coleta...")
                    if self.process and self.process.poll() is None:
                        self.process.terminate()
                        time.sleep(5)
                        if self.process.poll() is None:
                            self.process.kill()
                    break
                    
                # Aguardar antes de próxima verificação
                time.sleep(30)  # Verificar a cada 30 segundos
                
            except KeyboardInterrupt:
                logger.info("Interrompido pelo usuário")
                if self.process and self.process.poll() is None:
                    self.process.terminate()
                break
                
            except Exception as e:
                logger.error(f"Erro no monitoramento: {e}")
                time.sleep(60)  # Aguardar 1 minuto em caso de erro
                
    def run(self):
        """Executa o gerenciador de coleta"""
        logger.info("="*70)
        logger.info("GERENCIADOR DE COLETA CONTÍNUA")
        logger.info(f"Horário: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        logger.info("="*70)
        
        # Aguardar mercado abrir se necessário
        while not self.is_market_open():
            logger.info("Aguardando abertura do mercado...")
            time.sleep(300)  # Verificar a cada 5 minutos
            
        # Iniciar coletor
        if self.start_collector():
            # Monitorar processo
            self.monitor_process()
        else:
            logger.error("Falha ao iniciar coletor")
            
        logger.info("Gerenciador finalizado")

def main():
    # Verificar se já existe um processo rodando
    pid_file = 'collection_manager.pid'
    
    if os.path.exists(pid_file):
        with open(pid_file, 'r') as f:
            old_pid = int(f.read().strip())
            
        # Verificar se processo ainda está rodando
        try:
            os.kill(old_pid, 0)
            print(f"Gerenciador já está rodando (PID: {old_pid})")
            return
        except OSError:
            # Processo não existe mais
            os.remove(pid_file)
            
    # Salvar PID atual
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))
        
    try:
        manager = CollectionManager()
        manager.run()
    finally:
        # Remover arquivo PID
        if os.path.exists(pid_file):
            os.remove(pid_file)

if __name__ == "__main__":
    main()