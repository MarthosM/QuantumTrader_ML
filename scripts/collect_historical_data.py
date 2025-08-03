"""
Script de Coleta Automatizada de Dados Históricos
===============================================

Este script automatiza a coleta de dados históricos considerando:
- Limitações do ProfitDLL (3 meses, 9 dias por vez)
- Múltiplas fontes de dados
- Validação e armazenamento otimizado
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
from pathlib import Path
import schedule
import time
import multiprocessing
from dotenv import load_dotenv

from src.database.historical_data_collector import HistoricalDataCollector
from src.integration.profit_dll_server import run_server

# Carregar variáveis de ambiente
load_dotenv()


class AutomatedDataCollector:
    """Sistema automatizado de coleta de dados com processo isolado"""
    
    def __init__(self, config_path: str):
        """
        Inicializa coletor automatizado
        
        Args:
            config_path: Caminho para arquivo de configuração
        """
        # Carregar configuração
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Configurar logging
        self._setup_logging()
        
        # Inicializar componentes
        self.collector = HistoricalDataCollector(self.config)
        self.server_process = None
        
        # Estado
        self.collection_status = {
            'last_run': None,
            'last_success': None,
            'errors': [],
            'symbols_collected': {}
        }
    
    def _setup_logging(self):
        """Configura sistema de logging"""
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"data_collection_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def collect_all_symbols(self, 
                          start_date: datetime = None,
                          end_date: datetime = None):
        """
        Coleta dados para todos os símbolos configurados
        
        Args:
            start_date: Data inicial (padrão: 6 meses atrás)
            end_date: Data final (padrão: ontem)
        """
        # Datas padrão
        if not end_date:
            end_date = datetime.now() - timedelta(days=1)  # Ontem
        if not start_date:
            start_date = end_date - timedelta(days=180)  # 6 meses
        
        self.logger.info(f"Iniciando coleta de {start_date.date()} a {end_date.date()}")
        
        symbols = self.config.get('symbols', ['WDOU25', 'WDOV25'])
        data_types = self.config.get('data_types', ['trades', 'candles'])
        
        self.collection_status['last_run'] = datetime.now().isoformat()
        
        for symbol in symbols:
            try:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Coletando dados para {symbol}")
                self.logger.info(f"{'='*60}")
                
                collection_start = start_date
                
                # Coletar dados
                success = self._collect_symbol_data(
                    symbol=symbol,
                    start_date=collection_start,
                    end_date=end_date,
                    data_types=data_types
                )
                
                if success:
                    self.collection_status['symbols_collected'][symbol] = {
                        'last_date': end_date.isoformat(),
                        'status': 'success'
                    }
                else:
                    self.collection_status['symbols_collected'][symbol] = {
                        'last_date': collection_start.isoformat(),
                        'status': 'partial'
                    }
                
            except Exception as e:
                self.logger.error(f"Erro coletando {symbol}: {e}")
                self.collection_status['errors'].append({
                    'symbol': symbol,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Salvar estado
        self.collection_status['last_success'] = datetime.now().isoformat()
        
        # Gerar relatório
        self._generate_collection_report()
    
    def _collect_symbol_data(self,
                           symbol: str,
                           start_date: datetime,
                           end_date: datetime,
                           data_types: List[str]) -> bool:
        """Coleta dados para um símbolo específico"""
        try:
            # Coletar dados usando a estratégia otimizada
            raw_data = self.collector.collect_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_types=data_types
            )
            
            # Verificar se coletou dados
            if 'trades' in raw_data and not raw_data['trades'].empty:
                self.logger.info(f"Coletados {len(raw_data['trades'])} trades")
                return True
            elif 'candles' in raw_data and not raw_data['candles'].empty:
                self.logger.info(f"Coletados {len(raw_data['candles'])} candles")
                return True
            else:
                self.logger.warning(f"Nenhum dado coletado para {symbol}")
                return False
            
        except Exception as e:
            self.logger.error(f"Erro no processamento: {e}")
            return False
    
    def start_isolated_server(self):
        """Inicia servidor ProfitDLL em processo isolado"""
        import multiprocessing
        
        server_config = {
            'dll_path': self.config.get('dll_path', r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"),
            'username': os.getenv("PROFIT_USERNAME"),
            'password': os.getenv("PROFIT_PASSWORD"),
            'key': os.getenv("PROFIT_KEY"),
            'port': self.config.get('profitdll_server_port', 6790)
        }
        
        self.logger.info("Iniciando servidor ProfitDLL isolado...")
        self.server_process = multiprocessing.Process(
            target=run_server,
            args=(server_config,),
            name="ProfitDLLServer"
        )
        self.server_process.daemon = True
        self.server_process.start()
        
        # Aguardar servidor inicializar
        time.sleep(8)  # Dar mais tempo para conectar ao ProfitDLL
        
        if self.server_process.is_alive():
            self.logger.info(f"✅ Servidor rodando (PID: {self.server_process.pid})")
            return True
        else:
            self.logger.error("❌ Servidor falhou ao iniciar")
            return False
    
    def stop_server(self):
        """Para servidor ProfitDLL"""
        if self.server_process and self.server_process.is_alive():
            self.logger.info("Parando servidor ProfitDLL...")
            self.server_process.terminate()
            self.server_process.join(timeout=10)
            
            if self.server_process.is_alive():
                self.server_process.kill()
                self.server_process.join()
            
            self.logger.info("Servidor parado")
    
    def update_recent_data(self, days_back: int = 7):
        """
        Atualiza dados recentes (última semana por padrão)
        
        Args:
            days_back: Número de dias para atualizar
        """
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=days_back)
        
        self.logger.info(f"Atualizando dados dos últimos {days_back} dias")
        
        # Iniciar servidor se necessário
        if not self.server_process or not self.server_process.is_alive():
            if not self.start_isolated_server():
                self.logger.error("Falha ao iniciar servidor")
                return
        
        try:
            self.collect_all_symbols(start_date, end_date)
        finally:
            self.stop_server()
    
    def show_data_summary(self):
        """Mostra resumo dos dados disponíveis"""
        self.logger.info("\nResumo dos dados disponíveis:")
        
        for symbol in self.config.get('symbols', []):
            summary = self.collector.get_data_summary(symbol)
            if 'dates' in summary:
                self.logger.info(f"\n{symbol}:")
                self.logger.info(f"  Datas: {len(summary['dates'])} dias")
                self.logger.info(f"  Tipos: {summary.get('data_types', [])}")
                self.logger.info(f"  Tamanho: {summary.get('total_size_mb', 0):.2f} MB")
    
    def _generate_collection_report(self):
        """Gera relatório da coleta"""
        report = []
        report.append("\n" + "="*60)
        report.append("RELATÓRIO DE COLETA DE DADOS")
        report.append("="*60)
        report.append(f"Última execução: {self.collection_status['last_run']}")
        report.append(f"Última bem-sucedida: {self.collection_status['last_success']}")
        
        report.append("\nSímbolos coletados:")
        for symbol, info in self.collection_status['symbols_collected'].items():
            report.append(f"  {symbol}: {info['status']} (até {info['last_date'][:10]})")
        
        if self.collection_status['errors']:
            report.append(f"\nErros encontrados: {len(self.collection_status['errors'])}")
            for error in self.collection_status['errors'][-5:]:  # Últimos 5 erros
                report.append(f"  - {error['symbol']}: {error['error'][:50]}...")
        
        # Resumo dos dados
        report.append("\nResumo dos dados coletados:")
        for symbol in self.config.get('symbols', []):
            summary = self.collector.get_data_summary(symbol)
            if 'dates' in summary:
                report.append(f"  {symbol}: {len(summary['dates'])} dias, {summary.get('total_size_mb', 0):.1f} MB")
        
        report.append("="*60)
        
        report_text = '\n'.join(report)
        self.logger.info(report_text)
        
        # Salvar relatório
        report_dir = Path(self.config.get('report_dir', 'reports'))
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / f"collection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
    
    def schedule_daily_collection(self, hour: int = 19):
        """
        Agenda coleta diária
        
        Args:
            hour: Hora do dia para executar (padrão: 19h)
        """
        schedule.every().day.at(f"{hour:02d}:00").do(self.update_recent_data)
        
        self.logger.info(f"Coleta diária agendada para {hour}:00")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Verificar a cada minuto


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Coletor automatizado de dados históricos')
    
    parser.add_argument('--config', type=str, default='config/collector_config.json',
                      help='Arquivo de configuração')
    
    parser.add_argument('--mode', type=str, choices=['full', 'update', 'gaps', 'schedule'],
                      default='update', help='Modo de execução')
    
    parser.add_argument('--start-date', type=str, 
                      help='Data inicial (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str,
                      help='Data final (YYYY-MM-DD)')
    
    parser.add_argument('--symbol', type=str,
                      help='Símbolo específico para coletar')
    
    parser.add_argument('--days-back', type=int, default=7,
                      help='Dias para atualizar (modo update)')
    
    args = parser.parse_args()
    
    # Criar configuração padrão se não existir
    config_path = Path(args.config)
    if not config_path.exists():
        default_config = {
            "symbols": ["WDOQ25"],
            "data_types": ["trades"],
            "data_dir": "data/historical",
            "csv_dir": "data/csv",
            "log_dir": "logs",
            "report_dir": "reports",
            "state_file": "data/collector_state.json",
            "dll_path": r"C:\\Users\\marth\\Downloads\\ProfitDLL\\DLLs\\Win64\\ProfitDLL.dll",
            "profitdll_server_port": 6790
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"Configuração padrão criada em {config_path}")
    
    # Inicializar coletor
    collector = AutomatedDataCollector(str(config_path))
    
    # Executar modo selecionado
    if args.mode == 'full':
        # Coleta completa
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
        
        collector.collect_all_symbols(start_date, end_date)
    
    elif args.mode == 'update':
        # Atualizar dados recentes
        collector.update_recent_data(args.days_back)
    
    elif args.mode == 'gaps':
        # Mostrar resumo dos dados
        collector.show_data_summary()
    
    elif args.mode == 'schedule':
        # Modo agendado
        collector.schedule_daily_collection()


if __name__ == "__main__":
    # Necessário para Windows
    multiprocessing.freeze_support()
    main()