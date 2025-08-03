"""
Historical Data Collector - Coleta dados históricos do ProfitDLL
==============================================================

Este módulo gerencia a coleta de dados históricos considerando as limitações:
- ProfitDLL: máximo 3 meses, 9 em 9 dias
- Combina com outras fontes (CSV, APIs) para histórico completo
- Usa processo isolado para evitar Segmentation Fault
"""

import os
import sys
import json
import time
import logging
import queue
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing.connection import Client
import requests

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class HistoricalDataCollector:
    """Coletor de dados históricos com gestão inteligente de fontes"""
    
    def __init__(self, config: Dict):
        """
        Inicializa coletor
        
        Args:
            config: Configurações do sistema
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Limites do ProfitDLL
        self.PROFIT_MAX_DAYS = 90  # 3 meses
        self.PROFIT_CHUNK_DAYS = 9  # 9 dias por vez
        
        # Diretórios
        self.data_dir = Path(config.get('data_dir', 'data/historical'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache de metadados
        self.metadata_file = self.data_dir / 'metadata.json'
        self.metadata = self._load_metadata()
        
        # Fontes de dados
        self.sources = {
            'profitdll': self._collect_from_profitdll,
            'csv': self._collect_from_csv,
            'api': self._collect_from_api
        }
        
    def collect_historical_data(self,
                              symbol: str,
                              start_date: datetime,
                              end_date: datetime,
                              data_types: List[str] = ['trades', 'candles', 'book']) -> Dict:
        """
        Coleta dados históricos usando estratégia otimizada
        
        Args:
            symbol: Símbolo para coletar
            start_date: Data inicial
            end_date: Data final
            data_types: Tipos de dados para coletar
            
        Returns:
            Dict com dados coletados
        """
        self.logger.info(f"Iniciando coleta histórica para {symbol}")
        self.logger.info(f"Período: {start_date} até {end_date}")
        
        # Verificar dados já existentes
        existing_data = self._check_existing_data(symbol, start_date, end_date)
        
        # Determinar períodos faltantes
        missing_periods = self._find_missing_periods(
            symbol, start_date, end_date, existing_data
        )
        
        if not missing_periods:
            self.logger.info("Todos os dados já estão disponíveis localmente")
            return self._load_existing_data(symbol, start_date, end_date, data_types)
        
        # Estratégia de coleta
        collection_plan = self._create_collection_plan(missing_periods)
        
        # Executar coleta
        collected_data = self._execute_collection_plan(
            symbol, collection_plan, data_types
        )
        
        # Salvar dados coletados
        self._save_collected_data(symbol, collected_data)
        
        # Combinar com dados existentes
        final_data = self._merge_all_data(
            symbol, start_date, end_date, data_types, 
            existing_data, collected_data
        )
        
        return final_data
    
    def _check_existing_data(self, 
                           symbol: str, 
                           start_date: datetime,
                           end_date: datetime) -> Dict:
        """Verifica dados já existentes no banco local"""
        existing = {
            'periods': [],
            'data_types': {}
        }
        
        symbol_dir = self.data_dir / symbol
        if not symbol_dir.exists():
            return existing
        
        # Verificar arquivos por data
        for date_dir in symbol_dir.iterdir():
            if date_dir.is_dir():
                try:
                    date = datetime.strptime(date_dir.name, '%Y%m%d')
                    if start_date <= date <= end_date:
                        existing['periods'].append(date)
                        
                        # Verificar tipos de dados disponíveis
                        for data_file in date_dir.iterdir():
                            data_type = data_file.stem
                            if data_type not in existing['data_types']:
                                existing['data_types'][data_type] = []
                            existing['data_types'][data_type].append(date)
                            
                except ValueError:
                    continue
        
        existing['periods'].sort()
        return existing
    
    def _find_missing_periods(self,
                            symbol: str,
                            start_date: datetime,
                            end_date: datetime,
                            existing_data: Dict) -> List[Tuple[datetime, datetime]]:
        """Identifica períodos sem dados"""
        missing_periods = []
        
        # Criar lista de todos os dias necessários
        current = start_date
        all_days = []
        while current <= end_date:
            # Pular fins de semana
            if current.weekday() < 5:  # Segunda a Sexta
                all_days.append(current)
            current += timedelta(days=1)
        
        # Encontrar gaps
        existing_days = set(existing_data['periods'])
        missing_days = [d for d in all_days if d not in existing_days]
        
        if not missing_days:
            return missing_periods
        
        # Agrupar em períodos contínuos
        missing_days.sort()
        period_start = missing_days[0]
        prev_day = missing_days[0]
        
        for day in missing_days[1:]:
            # Se há gap maior que 3 dias, novo período
            if (day - prev_day).days > 3:
                missing_periods.append((period_start, prev_day))
                period_start = day
            prev_day = day
        
        # Adicionar último período
        missing_periods.append((period_start, prev_day))
        
        return missing_periods
    
    def _create_collection_plan(self, 
                              missing_periods: List[Tuple[datetime, datetime]]) -> Dict:
        """Cria plano otimizado de coleta"""
        plan = {
            'profitdll': [],
            'csv': [],
            'api': []
        }
        
        now = datetime.now()
        three_months_ago = now - timedelta(days=90)
        
        for start, end in missing_periods:
            # Período recente (últimos 3 meses) - usar ProfitDLL
            if start >= three_months_ago:
                # Dividir em chunks de 9 dias
                current = start
                while current <= end:
                    chunk_end = min(current + timedelta(days=8), end)
                    plan['profitdll'].append((current, chunk_end))
                    current = chunk_end + timedelta(days=1)
            
            # Período antigo - usar CSV ou API
            else:
                # Verificar se temos CSV
                csv_available = self._check_csv_availability(start, end)
                if csv_available:
                    plan['csv'].append((start, end))
                else:
                    # Usar API externa como fallback
                    plan['api'].append((start, end))
        
        return plan
    
    def _execute_collection_plan(self,
                               symbol: str,
                               plan: Dict,
                               data_types: List[str]) -> Dict:
        """Executa plano de coleta"""
        collected_data = {}
        
        # Coletar de cada fonte
        for source, periods in plan.items():
            if periods and source in self.sources:
                self.logger.info(f"Coletando {len(periods)} períodos de {source}")
                
                for start, end in periods:
                    try:
                        data = self.sources[source](symbol, start, end, data_types)
                        
                        # Organizar por data
                        for date, day_data in data.items():
                            if date not in collected_data:
                                collected_data[date] = {}
                            collected_data[date].update(day_data)
                            
                    except Exception as e:
                        self.logger.error(f"Erro coletando {source} {start}-{end}: {e}")
                        continue
                    
                    # Delay entre requisições para ProfitDLL
                    if source == 'profitdll':
                        time.sleep(1)
        
        return collected_data
    
    def _collect_from_profitdll(self,
                               symbol: str,
                               start_date: datetime,
                               end_date: datetime,
                               data_types: List[str]) -> Dict:
        """Coleta dados do ProfitDLL usando servidor isolado"""
        collected = {}
        
        # Conectar ao servidor ProfitDLL isolado
        server_address = ('localhost', self.config.get('profitdll_server_port', 6789))
        
        try:
            self.logger.info(f"Conectando ao servidor ProfitDLL em {server_address}")
            client = Client(server_address, authkey=b'profit_dll_secret')
            
            # Receber mensagem de boas-vindas
            welcome = client.recv()
            if welcome.get('status') != 'connected':
                raise ConnectionError("Falha ao conectar ao servidor ProfitDLL")
            
            # Enviar comando para coletar dados históricos
            command = {
                'type': 'collect_historical',
                'symbol': symbol,
                'start_date': start_date.strftime('%d/%m/%Y'),
                'end_date': end_date.strftime('%d/%m/%Y'),
                'data_types': data_types
            }
            
            client.send(command)
            
            # Receber dados coletados
            self.logger.info(f"Aguardando dados históricos de {symbol}...")
            
            # Timeout de 60 segundos para coleta
            timeout = 60
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if client.poll(0.5):
                    response = client.recv()
                    
                    if response.get('type') == 'historical_data':
                        trades_data = response.get('data', [])
                        
                        # Organizar por data
                        for trade in trades_data:
                            date = pd.to_datetime(trade['timestamp']).date()
                            if date not in collected:
                                collected[date] = {}
                            if 'trades' not in collected[date]:
                                collected[date]['trades'] = []
                            collected[date]['trades'].append(trade)
                        
                        self.logger.info(f"Recebidos {len(trades_data)} trades")
                        break
                        
                    elif response.get('type') == 'error':
                        error_msg = response.get('message', 'Erro desconhecido')
                        self.logger.error(f"Erro do servidor: {error_msg}")
                        break
            
            # Fechar conexão
            client.send({'type': 'done'})
            client.close()
            
        except Exception as e:
            self.logger.error(f"Erro coletando do ProfitDLL isolado: {e}")
            # Fallback para método direto se servidor não estiver rodando
            self.logger.info("Tentando método direto (risco de Segfault)...")
            collected = self._collect_from_profitdll_direct(
                symbol, start_date, end_date, data_types
            )
        
        return collected
    
    def _collect_from_profitdll_direct(self,
                                     symbol: str,
                                     start_date: datetime,
                                     end_date: datetime,
                                     data_types: List[str]) -> Dict:
        """Coleta direta do ProfitDLL (método antigo - risco de Segfault)"""
        from src.connection_manager_v4 import ConnectionManagerV4
        
        dll_path = self.config.get('dll_path', r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll")
        conn = ConnectionManagerV4(dll_path)
        
        if not conn.initialize(
            key=self.config.get('key'),
            username=self.config.get('username'),
            password=self.config.get('password')
        ):
            raise ConnectionError("Falha ao conectar com ProfitDLL")
        
        collected = {}
        
        try:
            # Registrar callback para receber dados
            trades_buffer = []
            
            def on_history_trade(trade_data):
                trades_buffer.append(trade_data)
            
            conn.register_history_trade_callback(on_history_trade)
            
            # Solicitar dados históricos
            if conn.get_history_trades(
                symbol,
                "BMF",
                start_date.strftime('%d/%m/%Y'),
                end_date.strftime('%d/%m/%Y')
            ):
                # Aguardar dados
                time.sleep(5)
                
                # Organizar por data
                for trade in trades_buffer:
                    date = pd.to_datetime(trade['timestamp']).date()
                    if date not in collected:
                        collected[date] = {}
                    if 'trades' not in collected[date]:
                        collected[date]['trades'] = []
                    collected[date]['trades'].append(trade)
            
        finally:
            conn.disconnect()
        
        return collected
    
    def _collect_from_csv(self,
                         symbol: str,
                         start_date: datetime,
                         end_date: datetime,
                         data_types: List[str]) -> Dict:
        """Coleta dados de arquivos CSV"""
        collected = {}
        
        # Procurar CSVs no diretório
        csv_dir = Path(self.config.get('csv_dir', 'data/csv'))
        if not csv_dir.exists():
            return collected
        
        for csv_file in csv_dir.glob(f"{symbol}_*.csv"):
            try:
                # Ler CSV
                df = pd.read_csv(csv_file, parse_dates=['datetime'])
                
                # Filtrar período
                mask = (df['datetime'].dt.date >= start_date.date()) & \
                       (df['datetime'].dt.date <= end_date.date())
                df_filtered = df[mask]
                
                if df_filtered.empty:
                    continue
                
                # Organizar por data
                for date, group in df_filtered.groupby(df_filtered['datetime'].dt.date):
                    if date not in collected:
                        collected[date] = {}
                    
                    # Detectar tipo de dados
                    if 'side' in group.columns:  # Trades
                        collected[date]['trades'] = group.to_dict('records')
                    else:  # Candles
                        collected[date]['candles'] = group.to_dict('records')
                        
            except Exception as e:
                self.logger.error(f"Erro lendo CSV {csv_file}: {e}")
                continue
        
        return collected
    
    def _collect_from_api(self,
                         symbol: str,
                         start_date: datetime,
                         end_date: datetime,
                         data_types: List[str]) -> Dict:
        """Coleta dados de API externa (exemplo: Alpha Vantage, Yahoo Finance)"""
        collected = {}
        
        # Implementar integração com API específica
        # Este é um exemplo genérico
        api_key = self.config.get('api_key')
        api_url = self.config.get('api_url')
        
        if not api_key or not api_url:
            self.logger.warning("API não configurada")
            return collected
        
        try:
            # Fazer requisição
            params = {
                'symbol': symbol,
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'apikey': api_key
            }
            
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Processar resposta (formato depende da API)
            # ... implementar parsing específico ...
            
        except Exception as e:
            self.logger.error(f"Erro na API: {e}")
        
        return collected
    
    def _save_collected_data(self, symbol: str, collected_data: Dict):
        """Salva dados coletados no formato otimizado"""
        symbol_dir = self.data_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        
        for date, data in collected_data.items():
            # Criar diretório por data
            date_str = date.strftime('%Y%m%d') if isinstance(date, datetime) else str(date).replace('-', '')
            date_dir = symbol_dir / date_str
            date_dir.mkdir(exist_ok=True)
            
            # Salvar cada tipo de dado
            for data_type, records in data.items():
                if records:
                    # Converter para DataFrame
                    df = pd.DataFrame(records)
                    
                    # Salvar em formato Parquet (mais eficiente)
                    output_file = date_dir / f"{data_type}.parquet"
                    df.to_parquet(output_file, compression='snappy')
                    
                    self.logger.debug(f"Salvos {len(records)} registros em {output_file}")
        
        # Atualizar metadata
        self._update_metadata(symbol, collected_data)
    
    def _merge_all_data(self,
                       symbol: str,
                       start_date: datetime,
                       end_date: datetime,
                       data_types: List[str],
                       existing_data: Dict,
                       collected_data: Dict) -> Dict:
        """Combina dados existentes com novos coletados"""
        merged = {
            'trades': pd.DataFrame(),
            'candles': pd.DataFrame(),
            'book': pd.DataFrame()
        }
        
        # Carregar todos os dados do período
        symbol_dir = self.data_dir / symbol
        
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Dias úteis
                date_str = current.strftime('%Y%m%d')
                date_dir = symbol_dir / date_str
                
                if date_dir.exists():
                    for data_type in data_types:
                        file_path = date_dir / f"{data_type}.parquet"
                        if file_path.exists():
                            df = pd.read_parquet(file_path)
                            merged[data_type] = pd.concat([merged[data_type], df])
            
            current += timedelta(days=1)
        
        # Ordenar e remover duplicatas
        for data_type in data_types:
            if not merged[data_type].empty:
                if 'datetime' in merged[data_type].columns:
                    merged[data_type].sort_values('datetime', inplace=True)
                    merged[data_type].drop_duplicates(subset=['datetime'], inplace=True)
                merged[data_type].reset_index(drop=True, inplace=True)
        
        return merged
    
    def _load_metadata(self) -> Dict:
        """Carrega metadados do banco de dados"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _update_metadata(self, symbol: str, collected_data: Dict):
        """Atualiza metadados com informações da coleta"""
        if symbol not in self.metadata:
            self.metadata[symbol] = {
                'first_date': None,
                'last_date': None,
                'total_days': 0,
                'data_types': {},
                'last_update': None
            }
        
        # Atualizar datas
        dates = list(collected_data.keys())
        if dates:
            dates.sort()
            
            if not self.metadata[symbol]['first_date']:
                self.metadata[symbol]['first_date'] = str(dates[0])
            else:
                self.metadata[symbol]['first_date'] = str(min(
                    datetime.strptime(self.metadata[symbol]['first_date'], '%Y-%m-%d').date(),
                    dates[0]
                ))
            
            self.metadata[symbol]['last_date'] = str(max(
                datetime.strptime(self.metadata[symbol]['last_date'], '%Y-%m-%d').date() if self.metadata[symbol]['last_date'] else dates[0],
                dates[-1]
            ))
            
            self.metadata[symbol]['total_days'] = len(set(dates))
        
        # Atualizar tipos de dados
        for date, data in collected_data.items():
            for data_type in data.keys():
                if data_type not in self.metadata[symbol]['data_types']:
                    self.metadata[symbol]['data_types'][data_type] = 0
                self.metadata[symbol]['data_types'][data_type] += 1
        
        self.metadata[symbol]['last_update'] = datetime.now().isoformat()
        
        # Salvar
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _check_csv_availability(self, start_date: datetime, end_date: datetime) -> bool:
        """Verifica se temos CSVs cobrindo o período"""
        csv_dir = Path(self.config.get('csv_dir', 'data/csv'))
        if not csv_dir.exists():
            return False
        
        # Simplificado - verificar se existe algum CSV
        # Em produção, verificar cobertura completa do período
        return len(list(csv_dir.glob("*.csv"))) > 0
    
    def _load_existing_data(self,
                          symbol: str,
                          start_date: datetime,
                          end_date: datetime,
                          data_types: List[str]) -> Dict:
        """Carrega dados já existentes do período"""
        return self._merge_all_data(
            symbol, start_date, end_date, data_types, {}, {}
        )
    
    def get_available_data(self) -> Dict[str, List[str]]:
        """Retorna informações sobre dados disponíveis"""
        try:
            available = {}
            
            # Listar todos os arquivos parquet
            for file in self.data_dir.glob("*/*.parquet"):
                parts = file.parent.name.split('_')
                if len(parts) >= 1:
                    symbol = parts[0]
                    date = file.parent.name
                    
                    if symbol not in available:
                        available[symbol] = []
                    
                    if date not in available[symbol]:
                        available[symbol].append(date)
            
            # Ordenar datas
            for symbol in available:
                available[symbol] = sorted(available[symbol])
            
            return available
            
        except Exception as e:
            self.logger.error(f"Erro listando dados: {e}")
            return {}
    
    def get_data_summary(self, symbol: str) -> Dict:
        """Retorna resumo dos dados disponíveis"""
        if symbol in self.metadata:
            return self.metadata[symbol]
        
        # Analisar diretório
        symbol_dir = self.data_dir / symbol
        if not symbol_dir.exists():
            return {'error': 'Nenhum dado encontrado'}
        
        summary = {
            'dates': [],
            'data_types': set(),
            'total_size_mb': 0
        }
        
        for date_dir in symbol_dir.iterdir():
            if date_dir.is_dir():
                summary['dates'].append(date_dir.name)
                
                for file in date_dir.iterdir():
                    summary['data_types'].add(file.stem)
                    summary['total_size_mb'] += file.stat().st_size / 1024 / 1024
        
        summary['data_types'] = list(summary['data_types'])
        summary['dates'].sort()
        
        return summary


if __name__ == "__main__":
    # Teste do coletor
    config = {
        'data_dir': 'data/historical',
        'csv_dir': 'data/csv',
        'connection': {
            'dll_path': 'C:\\ProfitDLL\\profit.dll'
        }
    }
    
    collector = HistoricalDataCollector(config)
    
    # Coletar últimos 6 meses
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    data = collector.collect_historical_data(
        symbol='WDOU25',
        start_date=start_date,
        end_date=end_date,
        data_types=['trades', 'candles']
    )
    
    print(f"Trades coletados: {len(data['trades'])}")
    print(f"Candles coletados: {len(data['candles'])}")