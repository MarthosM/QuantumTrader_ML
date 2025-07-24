# src/training/data_loader.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import dask.dataframe as dd
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq

class TrainingDataLoader:
    """Carregador otimizado para grandes volumes de dados de trading"""
    
    def __init__(self, data_path: str, chunk_size: int = 100000):
        self.data_path = Path(data_path)
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
        
        # Validador de dados reais (reutilizar do sistema existente)
        try:
            from src.production_data_validator import ProductionDataValidator
            self.validator = ProductionDataValidator()
        except ImportError:
            try:
                from production_data_validator import ProductionDataValidator
                self.validator = ProductionDataValidator()
            except ImportError:
                # Mock validator se não encontrar
                self.logger.warning("ProductionDataValidator não encontrado, usando validação básica")
                
                class MockValidator:
                    def validate_data(self, data):
                        return {'is_valid': True, 'warnings': []}
                    
                    def validate_trading_data(self, data, context=None):
                        """Mock implementation for validate_trading_data"""
                        self.logger = logging.getLogger(__name__)
                        self.logger.info(f"Mock validator - validating {len(data)} records for context: {context}")
                        return {'is_valid': True, 'warnings': []}
                
                self.validator = MockValidator()
        
        # Cache para dados carregados
        self.data_cache = {}
        
    def load_historical_data(self, 
                           start_date: datetime, 
                           end_date: datetime,
                           symbols: List[str],
                           validate_realtime: bool = True) -> pd.DataFrame:
        """
        Carrega dados históricos do CSV com validação de produção
        
        Args:
            start_date: Data inicial
            end_date: Data final  
            symbols: Lista de símbolos para carregar
            validate_realtime: Se deve validar dados como reais
            
        Returns:
            DataFrame com dados validados e limpos
        """
        self.logger.info(f"Carregando dados de {start_date} até {end_date} para {symbols}")
        
        # Verificar cache primeiro
        cache_key = f"{start_date}_{end_date}_{'_'.join(symbols)}"
        if cache_key in self.data_cache:
            self.logger.info("Dados encontrados em cache")
            return self.data_cache[cache_key]
        
        # Colunas esperadas no CSV
        expected_columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'trades', 'buy_volume', 'sell_volume', 'vwap', 'symbol'
        ]
        
        # Carregar dados usando chunks para eficiência
        all_data = []
        
        for symbol in symbols:
            symbol_files = self._find_symbol_files(symbol, start_date, end_date)
            
            for file_path in symbol_files:
                if file_path.suffix == '.parquet':
                    # Otimizado para parquet
                    df = self._load_parquet_file(file_path, start_date, end_date)
                else:
                    # CSV com chunks
                    df = self._load_csv_chunked(file_path, start_date, end_date)
                
                if not df.empty:
                    all_data.append(df)
        
        if not all_data:
            raise ValueError(f"Nenhum dado encontrado para {symbols} no período especificado")
        
        # Concatenar todos os dados
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Validar dados reais se necessário
        if validate_realtime:
            self.logger.info("Validando dados como reais (produção)")
            # Em modo treinamento, apenas logar warnings sem bloquear
            training_mode = os.getenv('TRAINING_MODE', 'false').lower() == 'true'
            
            try:
                if training_mode:
                    # Modo treinamento: apenas validação leve
                    self.logger.info("🎓 MODO TREINAMENTO - Validação relaxada")
                    # Verificações básicas sem bloquear
                    if len(combined_data) > 0:
                        self.logger.info(f"✅ Dados carregados: {len(combined_data)} registros")
                        if 'close' in combined_data.columns:
                            self.logger.info(f"✅ Preços válidos: {combined_data['close'].min():.2f} - {combined_data['close'].max():.2f}")
                else:
                    # Modo produção rigoroso
                    self.validator.validate_trading_data(combined_data, 'historical_training')
            except Exception as e:
                if training_mode:
                    self.logger.warning(f"⚠️ Aviso de validação (modo treinamento): {e}")
                else:
                    raise
        
        # Limpar e processar dados
        clean_data = self._clean_data(combined_data)
        
        # Adicionar ao cache
        self.data_cache[cache_key] = clean_data
        
        self.logger.info(f"Carregados {len(clean_data)} registros para {len(symbols)} símbolos")
        return clean_data
    
    def _load_csv_chunked(self, file_path: Path, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Carrega CSV em chunks para eficiência de memória"""
        chunks = []
        
        try:
            # Primeiro, verificar as colunas disponíveis
            sample = pd.read_csv(file_path, nrows=1)
            
            # Mapear colunas comuns
            column_mapping = {
                'Date': 'timestamp',
                'date': 'timestamp', 
                'datetime': 'timestamp',
                'time': 'timestamp'
            }
            
            # Encontrar coluna de data
            date_col = None
            for col in sample.columns:
                if col.lower() in ['date', 'timestamp', 'datetime', 'time']:
                    date_col = col
                    break
            
            if not date_col:
                self.logger.warning(f"Nenhuma coluna de data encontrada em {file_path}")
                return pd.DataFrame()
            
            # Usar iterator para ler em chunks
            for chunk in pd.read_csv(file_path, 
                                    chunksize=self.chunk_size,
                                    parse_dates=[date_col]):
                
                # Renomear coluna de data para timestamp se necessário
                if date_col != 'timestamp':
                    chunk = chunk.rename(columns={date_col: 'timestamp'})
                
                # Definir timestamp como índice se não for
                if 'timestamp' not in chunk.index.names:
                    chunk = chunk.set_index('timestamp')
                
                # Filtrar por data
                mask = (chunk.index >= start_date) & (chunk.index <= end_date)
                filtered_chunk = chunk[mask]
                
                if not filtered_chunk.empty:
                    chunks.append(filtered_chunk)
                    
        except Exception as e:
            self.logger.error(f"Erro carregando {file_path}: {e}")
            import traceback
            traceback.print_exc()
            
        return pd.concat(chunks) if chunks else pd.DataFrame()
    
    def _load_parquet_file(self, file_path: Path, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Carrega arquivo parquet otimizado"""
        try:
            # Usar filtros para leitura eficiente
            filters = [
                ('timestamp', '>=', start_date),
                ('timestamp', '<=', end_date)
            ]
            
            table = pq.read_table(file_path, filters=filters)
            return table.to_pandas()
            
        except Exception as e:
            self.logger.error(f"Erro carregando parquet {file_path}: {e}")
            return pd.DataFrame()
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Limpa e valida dados"""
        
        # Verificar estrutura dos dados
        self.logger.info(f"Estrutura dos dados: {data.shape}")
        self.logger.info(f"Colunas disponíveis: {list(data.columns)}")
        self.logger.info(f"Índice: {data.index.name}")
        
        # Reset index se timestamp está como índice
        if data.index.name == 'timestamp' or 'timestamp' in str(data.index.name):
            data = data.reset_index()
        
        # MODO TREINAMENTO: Limpeza mais leve
        training_mode = os.getenv('TRAINING_MODE', 'false').lower() == 'true'
        
        if training_mode:
            self.logger.info("🎓 MODO TREINAMENTO - Limpeza relaxada de dados")
            
            # Apenas remover registros obviamente inválidos
            initial_count = len(data)
            
            # Remover apenas registros com preços zero ou NaN
            if 'close' in data.columns:
                invalid_price = (data['close'].isna()) | (data['close'] <= 0)
                data = data[~invalid_price]
                self.logger.info(f"Removidos {invalid_price.sum()} registros com preços inválidos")
            
            # Remover apenas volumes extremamente negativos (não apenas negativos)
            if 'volume' in data.columns:
                extreme_negative = data['volume'] < -1000000  # Apenas extremamente negativos
                data = data[~extreme_negative]
                self.logger.info(f"Removidos {extreme_negative.sum()} registros com volumes extremamente negativos")
                
            self.logger.info(f"Limpeza relaxada: {initial_count} → {len(data)} registros")
            
        else:
            # Modo produção: limpeza rigorosa (código original)
            # Remover duplicatas baseado nas colunas disponíveis
            duplicate_cols = []
            if 'timestamp' in data.columns:
                duplicate_cols.append('timestamp')
            if 'symbol' in data.columns:
                duplicate_cols.append('symbol')
            elif 'contract' in data.columns:
                duplicate_cols.append('contract')
                
            if duplicate_cols:
                data = data.drop_duplicates(subset=duplicate_cols)
            else:
                # Se não tem colunas de identificação, remover duplicatas de todas as colunas
                data = data.drop_duplicates()
            
            # Ordenar por timestamp se disponível
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp')
            elif data.index.name == 'timestamp':
                data = data.sort_index()
            
            # Validar OHLC se disponível
            ohlc_cols = ['open', 'high', 'low', 'close']
            if all(col in data.columns for col in ohlc_cols):
                invalid_ohlc = (
                    (data['high'] < data['low']) |
                    (data['high'] < data['open']) |
                    (data['high'] < data['close']) |
                    (data['low'] > data['open']) |
                    (data['low'] > data['close'])
                )
                
                if invalid_ohlc.any():
                    self.logger.warning(f"Removendo {invalid_ohlc.sum()} registros com OHLC inválidos")
                    data = data[~invalid_ohlc]
            
            # Remover valores negativos se existirem
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in ['volume', 'trades', 'quantidade']:
                    # Volume e trades não podem ser negativos
                    negative_mask = data[col] < 0
                    if negative_mask.any():
                        self.logger.warning(f"Removendo {negative_mask.sum()} registros com {col} negativo")
                        data = data[~negative_mask]
        
        self.logger.info(f"Dados limpos: {data.shape[0]} registros restantes")
        return data
        data = data[data['volume'] > 0]
        
        # Preencher valores ausentes de forma inteligente
        data['buy_volume'] = data['buy_volume'].fillna(data['volume'] * 0.5)
        data['sell_volume'] = data['sell_volume'].fillna(data['volume'] * 0.5)
        data['vwap'] = data['vwap'].fillna((data['high'] + data['low'] + data['close']) / 3)
        
        return data
    
    def create_training_splits(self, 
                             data: pd.DataFrame,
                             validation_method: str = 'walk_forward',
                             **kwargs) -> Dict[str, List[Tuple[pd.DataFrame, pd.DataFrame]]]:
        """
        Cria splits temporais para treinamento
        
        Args:
            data: DataFrame com dados
            validation_method: Método de validação ('walk_forward', 'purged_cv', 'time_series')
            **kwargs: Parâmetros específicos do método
            
        Returns:
            Dicionário com splits de treino/validação
        """
        if validation_method == 'walk_forward':
            return self._walk_forward_split(data, **kwargs)
        elif validation_method == 'purged_cv':
            return self._purged_cross_validation(data, **kwargs)
        elif validation_method == 'time_series':
            return self._time_series_split(data, **kwargs)
        else:
            raise ValueError(f"Método de validação desconhecido: {validation_method}")
    
    def _walk_forward_split(self, data: pd.DataFrame, 
                           initial_train_size: int = 5000,
                           step_size: int = 200,
                           test_size: int = 100) -> Dict[str, List]:
        """Walk-forward validation para séries temporais"""
        splits = []
        
        # Garantir que dados estão ordenados por tempo
        data = data.sort_index()
        total_size = len(data)
        
        current_pos = initial_train_size
        
        while current_pos + test_size <= total_size:
            # Janela de treino
            train_start = max(0, current_pos - initial_train_size)
            train_end = current_pos
            train_data = data.iloc[train_start:train_end]
            
            # Janela de teste
            test_start = current_pos
            test_end = min(current_pos + test_size, total_size)
            test_data = data.iloc[test_start:test_end]
            
            splits.append({
                'train': train_data,
                'test': test_data,
                'train_period': (train_data.index[0], train_data.index[-1]),
                'test_period': (test_data.index[0], test_data.index[-1])
            })
            
            # Avançar janela
            current_pos += step_size
        
        self.logger.info(f"Criados {len(splits)} splits walk-forward")
        return {'walk_forward': splits}
    
    def _purged_cross_validation(self, data: pd.DataFrame,
                                n_splits: int = 5,
                                purge_gap: int = 100,
                                embargo_gap: int = 50) -> Dict[str, List]:
        """Cross-validation com purge para evitar data leakage"""
        splits = []
        data = data.sort_index()
        data_length = len(data)
        fold_size = data_length // n_splits
        
        for i in range(n_splits):
            # Define test fold
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, data_length)
            
            # Train indices com purge e embargo
            train_indices = []
            
            # Dados antes do test fold (com purge)
            if test_start > purge_gap:
                train_indices.extend(range(0, test_start - purge_gap))
            
            # Dados depois do test fold (com embargo)
            if test_end + embargo_gap < data_length:
                train_indices.extend(range(test_end + embargo_gap, data_length))
            
            if len(train_indices) < 2000:  # Mínimo de dados
                continue
            
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_start:test_end]
            
            splits.append({
                'train': train_data,
                'test': test_data,
                'fold': i,
                'purge_gap': purge_gap,
                'embargo_gap': embargo_gap
            })
        
        self.logger.info(f"Criados {len(splits)} splits purged CV")
        return {'purged_cv': splits}
    
    def _time_series_split(self, data: pd.DataFrame,
                          n_splits: int = 5,
                          test_size_ratio: float = 0.2) -> Dict[str, List]:
        """Split sequencial para séries temporais mantendo ordem cronológica"""
        splits = []
        data = data.sort_index()
        data_length = len(data)
        
        # Calcular tamanho das janelas
        total_test_size = int(data_length * test_size_ratio)
        test_size_per_split = total_test_size // n_splits
        
        # Tamanho inicial de treino
        initial_train_size = data_length - total_test_size
        
        for i in range(n_splits):
            # Janela de treino - cresce progressivamente
            train_end = initial_train_size + (i * test_size_per_split)
            train_data = data.iloc[:train_end]
            
            # Janela de teste - sequencial após treino
            test_start = train_end
            test_end = min(test_start + test_size_per_split, data_length)
            test_data = data.iloc[test_start:test_end]
            
            if len(test_data) == 0:
                break
                
            splits.append({
                'train': train_data,
                'test': test_data,
                'split': i,
                'train_period': (train_data.index[0], train_data.index[-1]),
                'test_period': (test_data.index[0], test_data.index[-1])
            })
        
        self.logger.info(f"Criados {len(splits)} splits time series")
        return {'time_series': splits}
    
    def _find_symbol_files(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Path]:
        """Encontra arquivos de dados para um símbolo com busca flexível"""
        files = []
        
        # Extrair prefixo do símbolo (ex: WDOH25 -> WDO)
        symbol_prefix = symbol[:3].upper() if len(symbol) >= 3 else symbol
        
        # Padrões de busca ordenados por prioridade
        # 1. Busca exata do símbolo
        exact_patterns = [
            f"{symbol}_*.csv",
            f"{symbol}_*.parquet",
            f"*_{symbol}_*.csv", 
            f"*_{symbol}_*.parquet",
            f"{symbol.lower()}_*.csv",
            f"{symbol.lower()}_*.parquet"
        ]
        
        # 2. Busca por prefixo (ex: WDO*) - mais flexível
        prefix_patterns = [
            f"{symbol_prefix}*_*.csv",
            f"{symbol_prefix}*_*.parquet",
            f"*{symbol_prefix}*.csv",
            f"*{symbol_prefix}*.parquet",
            f"{symbol_prefix.lower()}*_*.csv",
            f"{symbol_prefix.lower()}*_*.parquet",
            f"*{symbol_prefix.lower()}*.csv",
            f"*{symbol_prefix.lower()}*.parquet"
        ]
        
        # 3. Padrões específicos para WDO (mini-dólar)
        wdo_patterns = []
        if 'WDO' in symbol.upper():
            wdo_patterns = [
                "WDO*.csv",
                "WDO*.parquet", 
                "wdo*.csv",
                "wdo*.parquet",
                "*WDO*.csv",
                "*WDO*.parquet",
                "*wdo*.csv", 
                "*wdo*.parquet"
            ]
        
        # Buscar arquivos seguindo ordem de prioridade
        all_patterns = exact_patterns + prefix_patterns + wdo_patterns
        
        for pattern in all_patterns:
            pattern_files = list(self.data_path.glob(pattern))
            files.extend(pattern_files)
            
            # Log dos arquivos encontrados para debugging
            if pattern_files:
                self.logger.info(f"Padrão '{pattern}' encontrou {len(pattern_files)} arquivo(s)")
                for f in pattern_files[:3]:  # Mostrar primeiros 3
                    self.logger.info(f"  - {f.name}")
        
        # Remover duplicatas mantendo ordem
        seen = set()
        unique_files = []
        for file in files:
            if file not in seen:
                seen.add(file)
                unique_files.append(file)
        
        files = unique_files
        self.logger.info(f"Total de arquivos únicos encontrados para {symbol}: {len(files)}")
        
        # Filtrar por data se necessário (implementação básica)
        filtered_files = []
        for file in files:
            try:
                if self._file_in_date_range(file, start_date, end_date):
                    filtered_files.append(file)
            except Exception as e:
                # Se não conseguir determinar data, incluir arquivo
                self.logger.warning(f"Não foi possível determinar data para {file.name}: {e}")
                filtered_files.append(file)
        
        result_files = sorted(filtered_files) if filtered_files else sorted(files)
        
        if result_files:
            self.logger.info(f"Arquivos selecionados para {symbol}:")
            for f in result_files:
                self.logger.info(f"  ✓ {f.name}")
        else:
            self.logger.warning(f"Nenhum arquivo encontrado para símbolo {symbol} (prefixo: {symbol_prefix})")
            self.logger.info(f"Verificando diretório: {self.data_path}")
            # Listar todos os arquivos para debug
            all_csv_files = list(self.data_path.glob("*.csv"))
            self.logger.info(f"Arquivos CSV disponíveis: {[f.name for f in all_csv_files[:10]]}")
        
        return result_files
    
    def _file_in_date_range(self, file_path: Path, start_date: datetime, end_date: datetime) -> bool:
        """Verifica se arquivo está no range de datas baseado no nome"""
        # Implementar lógica específica baseada no padrão de nomenclatura
        # Por enquanto, retornar True
        return True