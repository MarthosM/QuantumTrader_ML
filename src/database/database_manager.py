"""
Database Manager - Gerenciador de Banco de Dados Históricos
==========================================================

Este módulo gerencia o armazenamento eficiente de dados históricos
com suporte a múltiplos formatos e compressão inteligente.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import json
import pyarrow as pa
import pyarrow.parquet as pq
import gzip
import shutil
from contextlib import contextmanager
import threading
from dataclasses import dataclass, asdict


@dataclass
class DataStats:
    """Estatísticas dos dados armazenados"""
    symbol: str
    data_type: str
    start_date: datetime
    end_date: datetime
    total_records: int
    total_size_mb: float
    compression_ratio: float
    quality_score: float
    last_update: datetime


class DatabaseManager:
    """Gerenciador de banco de dados otimizado para séries temporais"""
    
    def __init__(self, db_path: str = "data/trading_db"):
        """
        Inicializa gerenciador de banco de dados
        
        Args:
            db_path: Caminho base para o banco de dados
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Paths específicos
        self.parquet_path = self.db_path / "parquet"
        self.sqlite_path = self.db_path / "metadata.db"
        self.backup_path = self.db_path / "backups"
        
        # Criar estrutura
        self.parquet_path.mkdir(exist_ok=True)
        self.backup_path.mkdir(exist_ok=True)
        
        # Lock para thread safety
        self._lock = threading.RLock()
        
        # Inicializar banco de metadados
        self._init_metadata_db()
        
        # Cache de estatísticas
        self._stats_cache = {}
        
    def _init_metadata_db(self):
        """Inicializa banco SQLite para metadados"""
        with self._get_db_connection() as conn:
            # Tabela de datasets
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    date DATE NOT NULL,
                    records INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    compressed_size_bytes INTEGER NOT NULL,
                    checksum TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, data_type, date)
                )
            """)
            
            # Tabela de estatísticas
            conn.execute("""
                CREATE TABLE IF NOT EXISTS statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    value REAL NOT NULL,
                    date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabela de qualidade
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    date DATE NOT NULL,
                    quality_score REAL NOT NULL,
                    missing_data_pct REAL,
                    outliers_pct REAL,
                    duplicates_pct REAL,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Índices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_symbol_date ON datasets(symbol, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_statistics_symbol_date ON statistics(symbol, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quality_symbol_date ON data_quality(symbol, date)")
            
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """Context manager para conexão SQLite thread-safe"""
        conn = sqlite3.connect(self.sqlite_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def store_data(self,
                  symbol: str,
                  data_type: str,
                  data: pd.DataFrame,
                  date: datetime,
                  validate: bool = True) -> bool:
        """
        Armazena dados no banco
        
        Args:
            symbol: Símbolo do ativo
            data_type: Tipo de dados (trades, candles, book)
            data: DataFrame com os dados
            date: Data dos dados
            validate: Se deve validar antes de armazenar
            
        Returns:
            bool: Sucesso da operação
        """
        with self._lock:
            try:
                # Validar dados
                if validate:
                    quality = self._validate_data(data, data_type)
                    if quality['score'] < 0.5:
                        self.logger.error(f"Dados rejeitados por baixa qualidade: {quality}")
                        return False
                
                # Preparar diretórios
                symbol_path = self.parquet_path / symbol / data_type
                symbol_path.mkdir(parents=True, exist_ok=True)
                
                # Nome do arquivo
                date_str = date.strftime('%Y%m%d')
                file_path = symbol_path / f"{date_str}.parquet.gz"
                
                # Comprimir e salvar
                self._save_compressed_parquet(data, file_path)
                
                # Calcular estatísticas
                stats = self._calculate_statistics(data, data_type)
                
                # Atualizar metadados
                self._update_metadata(
                    symbol=symbol,
                    data_type=data_type,
                    date=date,
                    records=len(data),
                    size_bytes=data.memory_usage(deep=True).sum(),
                    compressed_size_bytes=file_path.stat().st_size,
                    stats=stats,
                    quality=quality if validate else None
                )
                
                self.logger.info(f"Armazenados {len(data)} registros de {data_type} para {symbol} em {date_str}")
                return True
                
            except Exception as e:
                self.logger.error(f"Erro armazenando dados: {e}")
                return False
    
    def load_data(self,
                 symbol: str,
                 data_type: str,
                 start_date: datetime,
                 end_date: datetime,
                 columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Carrega dados do banco
        
        Args:
            symbol: Símbolo do ativo
            data_type: Tipo de dados
            start_date: Data inicial
            end_date: Data final
            columns: Colunas específicas (None = todas)
            
        Returns:
            DataFrame com os dados
        """
        with self._lock:
            all_data = []
            
            # Iterar por cada dia
            current = start_date
            while current <= end_date:
                if current.weekday() < 5:  # Dias úteis
                    date_str = current.strftime('%Y%m%d')
                    file_path = self.parquet_path / symbol / data_type / f"{date_str}.parquet.gz"
                    
                    if file_path.exists():
                        try:
                            df = self._load_compressed_parquet(file_path, columns)
                            all_data.append(df)
                        except Exception as e:
                            self.logger.error(f"Erro lendo {file_path}: {e}")
                
                current += timedelta(days=1)
            
            # Combinar dados
            if all_data:
                result = pd.concat(all_data, ignore_index=True)
                
                # Ordenar por tempo
                if 'datetime' in result.columns:
                    result.sort_values('datetime', inplace=True)
                elif 'timestamp' in result.columns:
                    result.sort_values('timestamp', inplace=True)
                
                return result
            else:
                return pd.DataFrame()
    
    def get_available_dates(self, 
                          symbol: str, 
                          data_type: str) -> List[datetime]:
        """Retorna lista de datas disponíveis"""
        with self._get_db_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT date 
                FROM datasets 
                WHERE symbol = ? AND data_type = ?
                ORDER BY date
            """, (symbol, data_type))
            
            dates = []
            for row in cursor:
                dates.append(datetime.strptime(row['date'], '%Y-%m-%d'))
            
            return dates
    
    def get_data_stats(self, 
                      symbol: str,
                      data_type: Optional[str] = None) -> List[DataStats]:
        """Retorna estatísticas dos dados armazenados"""
        with self._get_db_connection() as conn:
            query = """
                SELECT 
                    symbol, data_type,
                    MIN(date) as start_date,
                    MAX(date) as end_date,
                    SUM(records) as total_records,
                    SUM(size_bytes) / 1024.0 / 1024.0 as total_size_mb,
                    AVG(CAST(compressed_size_bytes AS REAL) / size_bytes) as compression_ratio,
                    AVG(q.quality_score) as avg_quality,
                    MAX(d.updated_at) as last_update
                FROM datasets d
                LEFT JOIN data_quality q ON 
                    d.symbol = q.symbol AND 
                    d.data_type = q.data_type AND 
                    d.date = q.date
                WHERE d.symbol = ?
            """
            
            params = [symbol]
            if data_type:
                query += " AND d.data_type = ?"
                params.append(data_type)
            
            query += " GROUP BY d.symbol, d.data_type"
            
            cursor = conn.execute(query, params)
            
            stats = []
            for row in cursor:
                stats.append(DataStats(
                    symbol=row['symbol'],
                    data_type=row['data_type'],
                    start_date=datetime.strptime(row['start_date'], '%Y-%m-%d'),
                    end_date=datetime.strptime(row['end_date'], '%Y-%m-%d'),
                    total_records=row['total_records'],
                    total_size_mb=row['total_size_mb'],
                    compression_ratio=row['compression_ratio'] or 0,
                    quality_score=row['avg_quality'] or 0,
                    last_update=datetime.strptime(row['last_update'], '%Y-%m-%d %H:%M:%S')
                ))
            
            return stats
    
    def optimize_storage(self, older_than_days: int = 30):
        """
        Otimiza armazenamento de dados antigos
        
        Args:
            older_than_days: Comprimir dados mais antigos que X dias
        """
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        with self._get_db_connection() as conn:
            # Buscar arquivos para otimizar
            cursor = conn.execute("""
                SELECT symbol, data_type, date, compressed_size_bytes
                FROM datasets
                WHERE date < ? AND compressed_size_bytes > 10485760  -- > 10MB
                ORDER BY size_bytes DESC
            """, (cutoff_date.strftime('%Y-%m-%d'),))
            
            for row in cursor:
                symbol = row['symbol']
                data_type = row['data_type']
                date = datetime.strptime(row['date'], '%Y-%m-%d')
                
                # Recomprimir com maior taxa
                self._recompress_data(symbol, data_type, date)
    
    def backup_database(self, backup_name: Optional[str] = None):
        """Cria backup do banco de dados"""
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_dir = self.backup_path / backup_name
        backup_dir.mkdir(exist_ok=True)
        
        # Copiar metadados
        shutil.copy2(self.sqlite_path, backup_dir / "metadata.db")
        
        # Comprimir dados parquet
        self.logger.info("Criando backup dos dados...")
        
        archive_path = self.backup_path / f"{backup_name}.tar.gz"
        
        import tarfile
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(self.parquet_path, arcname="parquet")
            tar.add(backup_dir / "metadata.db", arcname="metadata.db")
        
        # Remover diretório temporário
        shutil.rmtree(backup_dir)
        
        self.logger.info(f"Backup criado: {archive_path}")
        return archive_path
    
    def _validate_data(self, data: pd.DataFrame, data_type: str) -> Dict:
        """Valida qualidade dos dados"""
        quality = {
            'score': 1.0,
            'missing_data_pct': 0.0,
            'outliers_pct': 0.0,
            'duplicates_pct': 0.0,
            'issues': []
        }
        
        # Verificar dados faltantes
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        quality['missing_data_pct'] = missing_pct
        if missing_pct > 0.01:  # > 1%
            quality['score'] *= (1 - missing_pct)
            quality['issues'].append(f"Dados faltantes: {missing_pct:.2%}")
        
        # Verificar duplicatas
        if 'datetime' in data.columns:
            duplicates = data.duplicated(subset=['datetime']).sum()
            dup_pct = duplicates / len(data) if len(data) > 0 else 0
            quality['duplicates_pct'] = dup_pct
            if dup_pct > 0:
                quality['score'] *= (1 - dup_pct * 2)  # Penalizar mais
                quality['issues'].append(f"Duplicatas: {duplicates}")
        
        # Verificar outliers (simplificado)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outliers = 0
        for col in numeric_cols:
            if col in ['price', 'volume']:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers += ((data[col] < Q1 - 3*IQR) | (data[col] > Q3 + 3*IQR)).sum()
        
        outlier_pct = outliers / (len(data) * len(numeric_cols)) if len(data) > 0 else 0
        quality['outliers_pct'] = outlier_pct
        if outlier_pct > 0.001:  # > 0.1%
            quality['score'] *= 0.9
            quality['issues'].append(f"Outliers detectados: {outlier_pct:.2%}")
        
        # Verificações específicas por tipo
        if data_type == 'trades':
            # Verificar se há trades
            if len(data) < 100:
                quality['score'] *= 0.5
                quality['issues'].append(f"Poucos trades: {len(data)}")
            
            # Verificar intervalo de tempo
            if 'datetime' in data.columns:
                time_range = (data['datetime'].max() - data['datetime'].min()).total_seconds() / 3600
                if time_range < 6:  # Menos de 6 horas
                    quality['score'] *= 0.8
                    quality['issues'].append(f"Período curto: {time_range:.1f} horas")
        
        return quality
    
    def _calculate_statistics(self, data: pd.DataFrame, data_type: str) -> Dict:
        """Calcula estatísticas dos dados"""
        stats = {}
        
        if data_type == 'trades':
            stats['total_volume'] = data['volume'].sum()
            stats['avg_price'] = data['price'].mean()
            stats['price_std'] = data['price'].std()
            stats['trade_count'] = len(data)
            
            if 'side' in data.columns:
                stats['buy_ratio'] = (data['side'] == 'BUY').mean()
        
        elif data_type == 'candles':
            stats['avg_volume'] = data['volume'].mean()
            stats['avg_range'] = (data['high'] - data['low']).mean()
            stats['volatility'] = data['close'].pct_change().std()
        
        return stats
    
    def _save_compressed_parquet(self, data: pd.DataFrame, file_path: Path):
        """Salva DataFrame em Parquet comprimido"""
        # Otimizar tipos de dados
        for col in data.columns:
            if data[col].dtype == 'float64':
                data[col] = data[col].astype('float32')
            elif data[col].dtype == 'int64':
                if data[col].min() >= 0 and data[col].max() < 2**32:
                    data[col] = data[col].astype('uint32')
        
        # Salvar com compressão
        data.to_parquet(file_path, compression='gzip', engine='pyarrow')
    
    def _load_compressed_parquet(self, file_path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Carrega Parquet comprimido"""
        return pd.read_parquet(file_path, columns=columns, engine='pyarrow')
    
    def _update_metadata(self, **kwargs):
        """Atualiza metadados no SQLite"""
        with self._get_db_connection() as conn:
            # Inserir ou atualizar dataset
            conn.execute("""
                INSERT OR REPLACE INTO datasets 
                (symbol, data_type, date, records, size_bytes, compressed_size_bytes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                kwargs['symbol'],
                kwargs['data_type'],
                kwargs['date'].strftime('%Y-%m-%d'),
                kwargs['records'],
                kwargs['size_bytes'],
                kwargs['compressed_size_bytes']
            ))
            
            # Inserir estatísticas
            if 'stats' in kwargs and kwargs['stats']:
                for metric, value in kwargs['stats'].items():
                    conn.execute("""
                        INSERT INTO statistics (symbol, data_type, metric, value, date)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        kwargs['symbol'],
                        kwargs['data_type'],
                        metric,
                        value,
                        kwargs['date'].strftime('%Y-%m-%d')
                    ))
            
            # Inserir qualidade
            if 'quality' in kwargs and kwargs['quality']:
                conn.execute("""
                    INSERT INTO data_quality 
                    (symbol, data_type, date, quality_score, missing_data_pct, 
                     outliers_pct, duplicates_pct, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    kwargs['symbol'],
                    kwargs['data_type'],
                    kwargs['date'].strftime('%Y-%m-%d'),
                    kwargs['quality']['score'],
                    kwargs['quality']['missing_data_pct'],
                    kwargs['quality']['outliers_pct'],
                    kwargs['quality']['duplicates_pct'],
                    json.dumps(kwargs['quality']['issues'])
                ))
            
            conn.commit()
    
    def _recompress_data(self, symbol: str, data_type: str, date: datetime):
        """Recomprime dados para economizar espaço"""
        date_str = date.strftime('%Y%m%d')
        file_path = self.parquet_path / symbol / data_type / f"{date_str}.parquet.gz"
        
        if file_path.exists():
            # Carregar dados
            data = self._load_compressed_parquet(file_path)
            
            # Recomprimir com configurações mais agressivas
            temp_path = file_path.with_suffix('.tmp')
            data.to_parquet(
                temp_path,
                compression='gzip',
                compression_level=9,  # Máxima compressão
                engine='pyarrow'
            )
            
            # Substituir arquivo se menor
            if temp_path.stat().st_size < file_path.stat().st_size:
                temp_path.replace(file_path)
                self.logger.info(f"Recomprimido {file_path.name}: "
                               f"{file_path.stat().st_size / 1024:.1f}KB")
            else:
                temp_path.unlink()


if __name__ == "__main__":
    # Teste do gerenciador
    db = DatabaseManager("data/test_db")
    
    # Criar dados de exemplo
    trades = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01 09:00', periods=1000, freq='1min'),
        'price': np.random.normal(5900, 50, 1000),
        'volume': np.random.randint(100, 1000, 1000),
        'side': np.random.choice(['BUY', 'SELL'], 1000)
    })
    
    # Armazenar
    db.store_data('WDOU25', 'trades', trades, datetime(2024, 1, 1))
    
    # Carregar
    loaded = db.load_data('WDOU25', 'trades', 
                         datetime(2024, 1, 1), 
                         datetime(2024, 1, 31))
    
    print(f"Dados carregados: {len(loaded)} registros")
    
    # Ver estatísticas
    stats = db.get_data_stats('WDOU25')
    for stat in stats:
        print(f"\n{stat.data_type}:")
        print(f"  Período: {stat.start_date} a {stat.end_date}")
        print(f"  Registros: {stat.total_records:,}")
        print(f"  Tamanho: {stat.total_size_mb:.2f} MB")
        print(f"  Compressão: {stat.compression_ratio:.2%}")