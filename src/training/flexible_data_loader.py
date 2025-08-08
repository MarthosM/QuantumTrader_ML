"""
Flexible Data Loader para Book Data
Suporta leitura de arquivos únicos consolidados ou múltiplos arquivos
"""

import pandas as pd
from pathlib import Path
from typing import List, Union, Optional, Dict
import logging
from datetime import datetime, timedelta
import pyarrow.parquet as pq
import numpy as np

class FlexibleBookDataLoader:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger('FlexibleDataLoader')
        
    def load_data(self, 
                  source: Union[str, Path, List[Union[str, Path]]],
                  data_types: Optional[List[str]] = None,
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None,
                  sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Carrega dados de book de forma flexível
        
        Args:
            source: Pode ser:
                - String/Path para arquivo único
                - String/Path para diretório (carrega todos os parquets)
                - Lista de arquivos
            data_types: Filtrar por tipos específicos (tiny_book, offer_book, etc)
            start_date: Data inicial para filtrar
            end_date: Data final para filtrar
            sample_size: Número de registros para amostragem
            
        Returns:
            DataFrame consolidado
        """
        
        # Determinar arquivos para carregar
        files = self._resolve_files(source)
        
        if not files:
            self.logger.error("Nenhum arquivo encontrado")
            return pd.DataFrame()
            
        self.logger.info(f"Carregando {len(files)} arquivo(s)...")
        
        # Estratégia de carregamento baseada no número de arquivos
        if len(files) == 1:
            df = self._load_single_file(files[0])
        else:
            df = self._load_multiple_files(files)
            
        # Aplicar filtros
        df = self._apply_filters(df, data_types, start_date, end_date, sample_size)
        
        return df
        
    def _resolve_files(self, source: Union[str, Path, List]) -> List[Path]:
        """Resolve source para lista de arquivos"""
        files = []
        
        if isinstance(source, (str, Path)):
            path = Path(source)
            
            if path.is_file():
                files = [path]
            elif path.is_dir():
                # Buscar todos os parquets no diretório
                files = list(path.glob('*.parquet'))
                # Ordenar por nome (geralmente tem timestamp)
                files = sorted(files)
            else:
                self.logger.error(f"Caminho não encontrado: {path}")
                
        elif isinstance(source, list):
            for item in source:
                item_path = Path(item)
                if item_path.exists():
                    files.append(item_path)
                else:
                    self.logger.warning(f"Arquivo não encontrado: {item_path}")
                    
        return files
        
    def _load_single_file(self, file: Path) -> pd.DataFrame:
        """Carrega um único arquivo"""
        try:
            self.logger.info(f"Carregando: {file.name}")
            return pd.read_parquet(file)
        except Exception as e:
            self.logger.error(f"Erro ao carregar {file}: {e}")
            return pd.DataFrame()
            
    def _load_multiple_files(self, files: List[Path]) -> pd.DataFrame:
        """Carrega múltiplos arquivos de forma eficiente"""
        
        # Para muitos arquivos, usar carregamento em lote
        if len(files) > 50:
            return self._load_batch_streaming(files)
        else:
            return self._load_batch_memory(files)
            
    def _load_batch_memory(self, files: List[Path]) -> pd.DataFrame:
        """Carrega todos os arquivos na memória"""
        dfs = []
        
        for i, file in enumerate(files):
            if i % 10 == 0:
                self.logger.info(f"Progresso: {i}/{len(files)} arquivos")
                
            try:
                df = pd.read_parquet(file)
                dfs.append(df)
            except Exception as e:
                self.logger.error(f"Erro em {file}: {e}")
                
        if dfs:
            self.logger.info("Concatenando DataFrames...")
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()
            
    def _load_batch_streaming(self, files: List[Path]) -> pd.DataFrame:
        """Carrega arquivos em streaming para economizar memória"""
        self.logger.info(f"Usando carregamento em streaming para {len(files)} arquivos")
        
        # Usar PyArrow para streaming
        tables = []
        
        for i, file in enumerate(files):
            if i % 50 == 0:
                self.logger.info(f"Progresso: {i}/{len(files)} arquivos")
                
            try:
                table = pq.read_table(file)
                tables.append(table)
                
                # Concatenar a cada 100 arquivos para não usar muita memória
                if len(tables) >= 100:
                    combined_table = pa.concat_tables(tables)
                    tables = [combined_table]
                    
            except Exception as e:
                self.logger.error(f"Erro em {file}: {e}")
                
        if tables:
            final_table = pa.concat_tables(tables)
            return final_table.to_pandas()
        else:
            return pd.DataFrame()
            
    def _apply_filters(self, 
                      df: pd.DataFrame,
                      data_types: Optional[List[str]],
                      start_date: Optional[datetime],
                      end_date: Optional[datetime],
                      sample_size: Optional[int]) -> pd.DataFrame:
        """Aplica filtros ao DataFrame"""
        
        if df.empty:
            return df
            
        original_size = len(df)
        
        # Filtrar por tipo
        if data_types and 'type' in df.columns:
            df = df[df['type'].isin(data_types)]
            self.logger.info(f"Filtrado por tipos {data_types}: {len(df):,} registros")
            
        # Filtrar por data
        if (start_date or end_date) and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]
                
            self.logger.info(f"Filtrado por data: {len(df):,} registros")
            
        # Amostragem
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            self.logger.info(f"Amostragem: {sample_size:,} registros")
            
        self.logger.info(f"Total após filtros: {len(df):,} de {original_size:,} registros")
        
        return df
        
    def load_for_training(self,
                         date_range: int = 7,
                         data_types: List[str] = ['tiny_book', 'offer_book'],
                         consolidated: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Carrega dados prontos para treinamento
        
        Args:
            date_range: Número de dias para carregar
            data_types: Tipos de dados para incluir
            consolidated: Se True, busca arquivos consolidados primeiro
            
        Returns:
            Dict com DataFrames por tipo
        """
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=date_range)
        
        data_dict = {}
        
        # Buscar dados
        for day in range(date_range):
            current_date = start_date + timedelta(days=day)
            date_str = current_date.strftime('%Y%m%d')
            
            base_dir = Path('data/realtime/book') / date_str
            
            if not base_dir.exists():
                continue
                
            # Tentar carregar consolidado primeiro
            if consolidated:
                consolidated_dir = base_dir / 'consolidated'
                if consolidated_dir.exists():
                    for dtype in data_types:
                        consolidated_file = consolidated_dir / f'consolidated_{dtype}_{date_str}.parquet'
                        if consolidated_file.exists():
                            self.logger.info(f"Carregando consolidado: {consolidated_file.name}")
                            
                            if dtype not in data_dict:
                                data_dict[dtype] = []
                                
                            df = pd.read_parquet(consolidated_file)
                            data_dict[dtype].append(df)
                            continue
                            
            # Se não houver consolidado, carregar arquivos individuais
            self.logger.info(f"Carregando arquivos individuais para {date_str}")
            df = self.load_data(base_dir, data_types=data_types)
            
            if not df.empty and 'type' in df.columns:
                for dtype in data_types:
                    type_df = df[df['type'] == dtype]
                    if not type_df.empty:
                        if dtype not in data_dict:
                            data_dict[dtype] = []
                        data_dict[dtype].append(type_df)
                        
        # Consolidar por tipo
        result = {}
        for dtype, dfs in data_dict.items():
            if dfs:
                result[dtype] = pd.concat(dfs, ignore_index=True)
                self.logger.info(f"{dtype}: {len(result[dtype]):,} registros totais")
                
        return result
        
    def get_data_info(self, source: Union[str, Path]) -> Dict:
        """Retorna informações sobre os dados sem carregar tudo"""
        files = self._resolve_files(source)
        
        info = {
            'num_files': len(files),
            'total_size_mb': sum(f.stat().st_size for f in files) / 1024 / 1024,
            'file_list': [f.name for f in files[:10]],  # Primeiros 10
            'data_types': set(),
            'date_range': {'min': None, 'max': None}
        }
        
        # Analisar amostra
        if files:
            sample_df = pd.read_parquet(files[0])
            
            if 'type' in sample_df.columns:
                info['data_types'] = set(sample_df['type'].unique())
                
            if 'timestamp' in sample_df.columns:
                timestamps = pd.to_datetime(sample_df['timestamp'])
                info['date_range']['min'] = timestamps.min()
                info['date_range']['max'] = timestamps.max()
                
            info['columns'] = list(sample_df.columns)
            info['sample_rows'] = len(sample_df)
            
        return info


# Exemplo de uso no pipeline de treinamento
class BookDataTrainingPipeline:
    def __init__(self):
        self.loader = FlexibleBookDataLoader()
        
    def prepare_training_data(self, config: Dict) -> pd.DataFrame:
        """Prepara dados para treinamento com configuração flexível"""
        
        # Opção 1: Carregar arquivo consolidado único
        if config.get('use_consolidated'):
            consolidated_file = config.get('consolidated_file')
            if consolidated_file:
                return self.loader.load_data(consolidated_file)
                
        # Opção 2: Carregar múltiplos arquivos
        if config.get('data_dir'):
            return self.loader.load_data(
                config['data_dir'],
                data_types=config.get('data_types', ['tiny_book', 'offer_book']),
                start_date=config.get('start_date'),
                end_date=config.get('end_date')
            )
            
        # Opção 3: Carregar automaticamente últimos N dias
        if config.get('days_back', 7):
            data_dict = self.loader.load_for_training(
                date_range=config['days_back'],
                data_types=config.get('data_types', ['tiny_book', 'offer_book']),
                consolidated=config.get('prefer_consolidated', True)
            )
            
            # Combinar todos os tipos
            all_dfs = []
            for dtype, df in data_dict.items():
                all_dfs.append(df)
                
            if all_dfs:
                return pd.concat(all_dfs, ignore_index=True)
                
        return pd.DataFrame()


if __name__ == "__main__":
    # Teste do loader
    loader = FlexibleBookDataLoader()
    
    # Info sobre os dados
    info = loader.get_data_info('data/realtime/book/20250805')
    print(f"\nInformações dos dados:")
    print(f"Arquivos: {info['num_files']}")
    print(f"Tamanho: {info['total_size_mb']:.2f} MB")
    print(f"Tipos: {info['data_types']}")
    
    # Carregar dados
    df = loader.load_data(
        'data/realtime/book/20250805',
        data_types=['tiny_book'],
        sample_size=1000
    )
    
    print(f"\nDados carregados: {df.shape}")