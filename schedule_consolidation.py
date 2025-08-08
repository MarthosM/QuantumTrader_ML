"""
Agendador para consolidação automática dos dados do Book Collector
Pode ser configurado para rodar em diferentes momentos
"""

import schedule
import time
import subprocess
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

class ConsolidationScheduler:
    def __init__(self):
        self.logger = logging.getLogger('ConsolidationScheduler')
        
    def consolidate_today(self):
        """Consolida dados de hoje"""
        date = datetime.now().strftime('%Y%m%d')
        self.logger.info(f"Iniciando consolidação automática para {date}")
        
        try:
            result = subprocess.run(
                ['python', 'auto_consolidate_book_data.py', date],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info("Consolidação concluída com sucesso")
            else:
                self.logger.error(f"Erro na consolidação: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Erro ao executar consolidação: {e}")
            
    def check_and_consolidate(self):
        """Verifica se há novos dados para consolidar"""
        date = datetime.now().strftime('%Y%m%d')
        data_dir = Path(f'data/realtime/book/{date}')
        
        if not data_dir.exists():
            self.logger.info("Sem dados para consolidar hoje")
            return
            
        # Verificar se já foi consolidado
        consolidated_dir = data_dir / 'consolidated'
        if consolidated_dir.exists():
            # Verificar hora do último arquivo consolidado
            metadata_file = consolidated_dir / f'consolidation_metadata_{date}.json'
            if metadata_file.exists():
                last_modified = datetime.fromtimestamp(metadata_file.stat().st_mtime)
                hours_since = (datetime.now() - last_modified).total_seconds() / 3600
                
                if hours_since < 1:  # Consolidado há menos de 1 hora
                    self.logger.info(f"Dados já consolidados há {hours_since:.1f} horas")
                    return
                    
        self.consolidate_today()
        
    def run_scheduler(self, mode='market_close'):
        """
        Executa o agendador em diferentes modos:
        - 'market_close': Consolida após fechamento do mercado (18:05)
        - 'hourly': Consolida a cada hora
        - 'every_30min': Consolida a cada 30 minutos
        - 'continuous': Verifica a cada 5 minutos se há novos dados
        """
        
        self.logger.info(f"Agendador iniciado no modo: {mode}")
        
        if mode == 'market_close':
            # Consolidar após fechamento do mercado
            schedule.every().day.at("18:05").do(self.consolidate_today)
            self.logger.info("Consolidação agendada para 18:05 (após fechamento)")
            
        elif mode == 'hourly':
            # Consolidar a cada hora
            schedule.every().hour.at(":05").do(self.check_and_consolidate)
            self.logger.info("Consolidação agendada para cada hora")
            
        elif mode == 'every_30min':
            # Consolidar a cada 30 minutos
            schedule.every(30).minutes.do(self.check_and_consolidate)
            self.logger.info("Consolidação agendada a cada 30 minutos")
            
        elif mode == 'continuous':
            # Verificar a cada 5 minutos
            schedule.every(5).minutes.do(self.check_and_consolidate)
            self.logger.info("Verificação contínua a cada 5 minutos")
            
        # Loop principal
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Verificar a cada minuto
                
        except KeyboardInterrupt:
            self.logger.info("Agendador interrompido pelo usuário")
            
def main():
    import sys
    
    # Modo padrão
    mode = 'market_close'
    
    # Verificar argumentos
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
    print(f"""
╔══════════════════════════════════════════════════════════╗
║        AGENDADOR DE CONSOLIDAÇÃO DO BOOK COLLECTOR       ║
╚══════════════════════════════════════════════════════════╝

Modos disponíveis:
- market_close : Consolida após fechamento (18:05)
- hourly       : Consolida a cada hora
- every_30min  : Consolida a cada 30 minutos  
- continuous   : Verifica a cada 5 minutos

Modo selecionado: {mode}

Pressione Ctrl+C para parar
""")
    
    scheduler = ConsolidationScheduler()
    scheduler.run_scheduler(mode)
    
if __name__ == "__main__":
    main()