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
import json
from pathlib import Path
import schedule
import time

from src.database.historical_data_collector import HistoricalDataCollector
from src.database.database_manager import DatabaseManager
from src.database.data_validator import DataValidator
from src.database.data_merger import DataMerger, DataSource


class AutomatedDataCollector:
    """Sistema automatizado de coleta de dados"""
    
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
        self.db_manager = DatabaseManager(self.config.get('db_path', 'data/trading_db'))
        self.validator = DataValidator(self.config)
        self.merger = DataMerger(self.config)
        
        # Estado
        self.collection_status = {
            'last_run': None,
            'last_success': None,
            'errors': [],
            'symbols_collected': {}
        }
        
        # Carregar estado anterior
        self._load_state()
    
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
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None):
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
                
                # Verificar última coleta
                last_collected = self._get_last_collection_date(symbol)
                if last_collected:
                    # Coletar apenas dados novos
                    collection_start = last_collected + timedelta(days=1)
                    if collection_start > end_date:
                        self.logger.info(f"{symbol} já está atualizado")
                        continue
                else:
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
        self._save_state()
        
        # Gerar relatório
        self._generate_collection_report()
    
    def _collect_symbol_data(self,
                           symbol: str,
                           start_date: datetime,
                           end_date: datetime,
                           data_types: List[str]) -> bool:
        """Coleta dados para um símbolo específico"""
        try:
            # 1. Coletar dados brutos
            raw_data = self.collector.collect_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_types=data_types
            )
            
            if not raw_data or all(df.empty for df in raw_data.values()):
                self.logger.warning(f"Nenhum dado coletado para {symbol}")
                return False
            
            # 2. Processar cada tipo de dado
            for data_type in data_types:
                if data_type not in raw_data or raw_data[data_type].empty:
                    continue
                
                self.logger.info(f"Processando {len(raw_data[data_type])} registros de {data_type}")
                
                # Dividir por dia para processamento
                daily_groups = raw_data[data_type].groupby(
                    pd.to_datetime(raw_data[data_type]['datetime']).dt.date
                )
                
                for date, day_data in daily_groups:
                    # 3. Validar dados
                    validation = self.validator.validate_data(
                        data=day_data,
                        data_type=data_type,
                        symbol=symbol,
                        date=date,
                        auto_fix=True
                    )
                    
                    if not validation.is_valid:
                        self.logger.error(f"Dados inválidos para {date}: {validation.errors}")
                        continue
                    
                    # 4. Armazenar no banco
                    stored = self.db_manager.store_data(
                        symbol=symbol,
                        data_type=data_type,
                        data=day_data,
                        date=datetime.combine(date, datetime.min.time()),
                        validate=False  # Já validado
                    )
                    
                    if stored:
                        self.logger.info(f"Armazenados dados de {data_type} para {date}")
                    else:
                        self.logger.error(f"Falha ao armazenar {data_type} para {date}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro no processamento: {e}")
            return False
    
    def update_recent_data(self, days_back: int = 7):
        """
        Atualiza dados recentes (última semana por padrão)
        
        Args:
            days_back: Número de dias para atualizar
        """
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=days_back)
        
        self.logger.info(f"Atualizando dados dos últimos {days_back} dias")
        self.collect_all_symbols(start_date, end_date)
    
    def fill_gaps(self):
        """Identifica e preenche gaps nos dados históricos"""
        self.logger.info("Verificando gaps nos dados...")
        
        symbols = self.config.get('symbols', [])
        
        for symbol in symbols:
            # Obter estatísticas do banco
            stats = self.db_manager.get_data_stats(symbol)
            
            if not stats:
                self.logger.warning(f"Sem dados para {symbol}")
                continue
            
            for stat in stats:
                # Verificar continuidade
                expected_days = (stat.end_date - stat.start_date).days
                actual_days = stat.total_records / 390  # ~390 minutos de pregão
                
                if actual_days < expected_days * 0.8:  # Menos de 80% dos dias esperados
                    self.logger.info(f"Gap detectado em {symbol} {stat.data_type}: "
                                   f"{actual_days:.0f}/{expected_days} dias")
                    
                    # Tentar preencher o gap
                    self._fill_data_gap(symbol, stat.data_type, 
                                      stat.start_date, stat.end_date)
    
    def _fill_data_gap(self, symbol: str, data_type: str, 
                      start_date: datetime, end_date: datetime):
        """Preenche gap específico nos dados"""
        # Identificar dias faltantes
        existing_dates = self.db_manager.get_available_dates(symbol, data_type)
        
        all_dates = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Dias úteis
                all_dates.append(current)
            current += timedelta(days=1)
        
        missing_dates = set(all_dates) - set(existing_dates)
        
        if missing_dates:
            self.logger.info(f"Preenchendo {len(missing_dates)} dias faltantes")
            
            # Agrupar em períodos contínuos
            missing_sorted = sorted(list(missing_dates))
            
            # Coletar períodos faltantes
            for date in missing_sorted:
                self._collect_symbol_data(
                    symbol=symbol,
                    start_date=date,
                    end_date=date,
                    data_types=[data_type]
                )
    
    def merge_sources(self, symbol: str, start_date: datetime, end_date: datetime):
        """
        Combina dados de múltiplas fontes
        
        Útil quando temos dados parciais de diferentes fontes
        """
        self.logger.info(f"Merging dados de múltiplas fontes para {symbol}")
        
        sources = []
        
        # Fonte 1: Banco de dados local
        db_data = self.db_manager.load_data(
            symbol=symbol,
            data_type='trades',
            start_date=start_date,
            end_date=end_date
        )
        
        if not db_data.empty:
            sources.append(DataSource(
                name="Database",
                data=db_data,
                quality_score=0.9,
                priority=2,
                metadata={'source': 'local_db'}
            ))
        
        # Fonte 2: CSV se disponível
        csv_path = Path(self.config.get('csv_dir', 'data/csv')) / f"{symbol}.csv"
        if csv_path.exists():
            csv_data = pd.read_csv(csv_path, parse_dates=['datetime'])
            sources.append(DataSource(
                name="CSV",
                data=csv_data,
                quality_score=0.8,
                priority=3,
                metadata={'source': 'csv_file'}
            ))
        
        # Executar merge se temos múltiplas fontes
        if len(sources) > 1:
            merged = self.merger.merge_sources(
                sources=sources,
                start_date=start_date,
                end_date=end_date,
                data_type='trades'
            )
            
            # Salvar resultado merged
            if not merged.empty:
                self.db_manager.store_data(
                    symbol=symbol,
                    data_type='trades_merged',
                    data=merged,
                    date=start_date,
                    validate=True
                )
                
                self.logger.info(f"Merge completo: {len(merged)} registros")
    
    def _get_last_collection_date(self, symbol: str) -> Optional[datetime]:
        """Obtém data da última coleta bem-sucedida"""
        stats = self.db_manager.get_data_stats(symbol)
        
        if stats:
            # Retornar a data mais recente entre todos os tipos de dados
            return max(stat.end_date for stat in stats)
        
        return None
    
    def _load_state(self):
        """Carrega estado da última execução"""
        state_file = Path(self.config.get('state_file', 'data/collector_state.json'))
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    self.collection_status = json.load(f)
            except Exception as e:
                self.logger.error(f"Erro carregando estado: {e}")
    
    def _save_state(self):
        """Salva estado atual"""
        state_file = Path(self.config.get('state_file', 'data/collector_state.json'))
        state_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(state_file, 'w') as f:
                json.dump(self.collection_status, f, indent=2)
        except Exception as e:
            self.logger.error(f"Erro salvando estado: {e}")
    
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
        
        # Estatísticas do banco
        report.append("\nEstatísticas do banco de dados:")
        for symbol in self.config.get('symbols', []):
            stats = self.db_manager.get_data_stats(symbol)
            if stats:
                total_records = sum(s.total_records for s in stats)
                total_size = sum(s.total_size_mb for s in stats)
                report.append(f"  {symbol}: {total_records:,} registros, {total_size:.1f} MB")
        
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
            "symbols": ["WDOU25", "WDOV25"],
            "data_types": ["trades", "candles"],
            "db_path": "data/trading_db",
            "csv_dir": "data/csv",
            "log_dir": "logs",
            "report_dir": "reports",
            "state_file": "data/collector_state.json",
            "connection": {
                "dll_path": "C:\\ProfitDLL\\profit.dll"
            }
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
        # Preencher gaps
        collector.fill_gaps()
    
    elif args.mode == 'schedule':
        # Modo agendado
        collector.schedule_daily_collection()


if __name__ == "__main__":
    import pandas as pd  # Import necessário para o script
    main()