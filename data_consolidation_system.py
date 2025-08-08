"""
Sistema de Consolidação Inteligente de Dados de Trading
Mantém continuidade temporal e otimiza armazenamento
"""

import os
import json
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DataConsolidation')


class TimeSeriesValidator:
    """Valida continuidade temporal dos dados"""
    
    @staticmethod
    def check_continuity(df1: pd.DataFrame, df2: pd.DataFrame, 
                         max_gap_seconds: int = 300) -> Tuple[bool, str]:
        """
        Verifica se dois DataFrames são contínuos no tempo
        
        Args:
            df1: DataFrame anterior
            df2: DataFrame posterior
            max_gap_seconds: Máximo gap permitido em segundos
            
        Returns:
            (is_continuous, message)
        """
        if df1.empty or df2.empty:
            return False, "DataFrame vazio"
        
        # Pegar último timestamp de df1 e primeiro de df2
        last_time = df1['timestamp'].max()
        first_time = df2['timestamp'].min()
        
        gap = first_time - last_time
        
        if gap < 0:
            return False, f"Sobreposição temporal: {gap:.2f}s"
        elif gap > max_gap_seconds:
            return False, f"Gap muito grande: {gap:.2f}s > {max_gap_seconds}s"
        else:
            return True, f"Contínuo (gap: {gap:.2f}s)"
    
    @staticmethod
    def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicatas mantendo o registro mais recente"""
        if 'timestamp' in df.columns:
            return df.drop_duplicates(subset=['timestamp'], keep='last')
        return df
    
    @staticmethod
    def fill_gaps(df: pd.DataFrame, method: str = 'forward') -> pd.DataFrame:
        """Preenche gaps nos dados"""
        if df.empty:
            return df
        
        # Criar índice temporal contínuo
        df = df.set_index('timestamp')
        
        # Detectar frequência dos dados
        freq = pd.infer_freq(df.index)
        if not freq:
            # Estimar frequência média
            diffs = df.index.to_series().diff().dropna()
            avg_diff = diffs.median()
            
            if avg_diff < pd.Timedelta(seconds=1):
                freq = '100ms'  # Alta frequência (ticks)
            elif avg_diff < pd.Timedelta(minutes=1):
                freq = '1s'     # Segundos
            else:
                freq = '1min'   # Minutos (candles)
        
        # Reindexar com frequência regular
        regular_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq
        )
        
        df = df.reindex(regular_index)
        
        # Preencher gaps
        if method == 'forward':
            df = df.fillna(method='ffill')
        elif method == 'interpolate':
            df = df.interpolate(method='time')
        
        return df.reset_index()


class DataConsolidator:
    """Consolida dados mantendo histórico contínuo"""
    
    def __init__(self, base_path: str = 'data/training'):
        self.base_path = Path(base_path)
        self.consolidated_path = self.base_path / 'consolidated'
        self.archive_path = self.base_path / 'archive'
        
        # Criar diretórios
        self.consolidated_path.mkdir(parents=True, exist_ok=True)
        self.archive_path.mkdir(parents=True, exist_ok=True)
        
        # Configurações
        self.config = {
            'retention_days': 30,        # Manter dados brutos por 30 dias
            'archive_days': 365,         # Manter arquivos por 1 ano
            'chunk_size': 1000000,       # Processar 1M registros por vez
            'compression': 'gzip',       # Tipo de compressão
            'merge_strategy': 'continuous',  # continuous, daily, weekly
            'validate_continuity': True,
            'remove_duplicates': True,
            'fill_gaps': False
        }
        
        self.validator = TimeSeriesValidator()
        
        # Estatísticas
        self.stats = {
            'files_processed': 0,
            'records_consolidated': 0,
            'duplicates_removed': 0,
            'gaps_filled': 0,
            'files_archived': 0,
            'bytes_saved': 0
        }
    
    def consolidate_continuous(self, data_type: str = 'tick',
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> str:
        """
        Consolida dados mantendo continuidade temporal
        
        Args:
            data_type: Tipo de dado (tick, book, candles, predictions)
            start_date: Data inicial (None = todos)
            end_date: Data final (None = hoje)
            
        Returns:
            Path do arquivo consolidado
        """
        logger.info(f"Iniciando consolidação de {data_type}")
        
        # Listar arquivos do tipo
        source_path = self.base_path / data_type
        if not source_path.exists():
            logger.warning(f"Diretório {source_path} não existe")
            return None
        
        # Filtrar arquivos por data
        files = self._get_files_in_range(source_path, start_date, end_date)
        if not files:
            logger.warning(f"Nenhum arquivo encontrado para consolidar")
            return None
        
        logger.info(f"Encontrados {len(files)} arquivos para consolidar")
        
        # Arquivo de destino consolidado
        if start_date and end_date:
            date_range = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        else:
            date_range = f"full_{datetime.now().strftime('%Y%m%d')}"
        
        output_file = self.consolidated_path / f"{data_type}_consolidated_{date_range}.parquet"
        
        # Se arquivo já existe, carregar para concatenar
        existing_df = None
        if output_file.exists():
            logger.info(f"Arquivo consolidado existe, carregando para merge...")
            existing_df = pd.read_parquet(output_file)
            logger.info(f"  Registros existentes: {len(existing_df):,}")
        
        # Processar arquivos em chunks
        all_chunks = []
        last_df = existing_df
        
        for file_path in sorted(files):
            logger.info(f"Processando {file_path.name}...")
            
            # Ler arquivo JSONL
            chunk_df = self._read_jsonl_file(file_path)
            if chunk_df.empty:
                continue
            
            # Validar continuidade se configurado
            if self.config['validate_continuity'] and last_df is not None:
                is_continuous, msg = self.validator.check_continuity(
                    last_df, chunk_df
                )
                
                if not is_continuous:
                    logger.warning(f"  Descontinuidade detectada: {msg}")
                    
                    if self.config['fill_gaps']:
                        logger.info("  Preenchendo gaps...")
                        chunk_df = self._merge_with_gap_filling(last_df, chunk_df)
                        self.stats['gaps_filled'] += 1
            
            # Remover duplicatas
            if self.config['remove_duplicates']:
                original_len = len(chunk_df)
                chunk_df = self.validator.detect_duplicates(chunk_df)
                removed = original_len - len(chunk_df)
                if removed > 0:
                    logger.info(f"  Removidas {removed} duplicatas")
                    self.stats['duplicates_removed'] += removed
            
            all_chunks.append(chunk_df)
            last_df = chunk_df
            self.stats['files_processed'] += 1
            self.stats['records_consolidated'] += len(chunk_df)
            
            # Salvar parcialmente se muitos chunks
            if len(all_chunks) >= 10:
                logger.info("  Salvando chunk parcial...")
                self._save_chunks(all_chunks, output_file, existing_df)
                all_chunks = []
                existing_df = pd.read_parquet(output_file)  # Recarregar com novos dados
        
        # Salvar chunks finais
        if all_chunks:
            self._save_chunks(all_chunks, output_file, existing_df)
        
        # Comprimir se configurado
        if self.config['compression'] == 'gzip':
            compressed_file = self._compress_file(output_file)
            logger.info(f"Arquivo comprimido: {compressed_file}")
            
        # Arquivar arquivos originais se configurado
        if self.config['retention_days'] > 0:
            self._archive_old_files(files)
        
        logger.info(f"Consolidação completa: {output_file}")
        logger.info(f"Estatísticas: {self.stats}")
        
        return str(output_file)
    
    def _get_files_in_range(self, path: Path, 
                           start_date: Optional[datetime],
                           end_date: Optional[datetime]) -> List[Path]:
        """Lista arquivos no range de datas"""
        all_files = list(path.glob("*.jsonl"))
        
        if not start_date and not end_date:
            return all_files
        
        filtered = []
        for file in all_files:
            # Extrair data do nome do arquivo (formato: type_YYYYMMDD_HH.jsonl)
            try:
                parts = file.stem.split('_')
                if len(parts) >= 2:
                    date_str = parts[1]
                    file_date = datetime.strptime(date_str, '%Y%m%d')
                    
                    if start_date and file_date < start_date:
                        continue
                    if end_date and file_date > end_date:
                        continue
                    
                    filtered.append(file)
            except:
                continue
        
        return filtered
    
    def _read_jsonl_file(self, file_path: Path) -> pd.DataFrame:
        """Lê arquivo JSONL e converte para DataFrame"""
        try:
            records = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            
            if not records:
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            
            # Converter timestamp para datetime se existir
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Ordenar por tempo
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Erro lendo {file_path}: {e}")
            return pd.DataFrame()
    
    def _merge_with_gap_filling(self, df1: pd.DataFrame, 
                               df2: pd.DataFrame) -> pd.DataFrame:
        """Merge dois DataFrames preenchendo gaps"""
        # Concatenar
        merged = pd.concat([df1, df2], ignore_index=True)
        
        # Preencher gaps
        merged = self.validator.fill_gaps(merged, method='forward')
        
        return merged
    
    def _save_chunks(self, chunks: List[pd.DataFrame], 
                    output_file: Path,
                    existing_df: Optional[pd.DataFrame] = None):
        """Salva chunks consolidados"""
        # Concatenar todos os chunks
        if chunks:
            new_df = pd.concat(chunks, ignore_index=True)
            
            # Merge com existente se houver
            if existing_df is not None:
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                # Remover duplicatas finais
                if self.config['remove_duplicates']:
                    final_df = self.validator.detect_duplicates(final_df)
            else:
                final_df = new_df
            
            # Ordenar por timestamp
            if 'timestamp' in final_df.columns:
                final_df = final_df.sort_values('timestamp')
            
            # Salvar como Parquet (mais eficiente que JSON)
            final_df.to_parquet(output_file, compression='snappy')
            logger.info(f"  Salvos {len(final_df):,} registros totais")
    
    def _compress_file(self, file_path: Path) -> Path:
        """Comprime arquivo com gzip"""
        compressed_path = file_path.with_suffix('.parquet.gz')
        
        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Calcular economia
        original_size = file_path.stat().st_size
        compressed_size = compressed_path.stat().st_size
        saved = original_size - compressed_size
        self.stats['bytes_saved'] += saved
        
        logger.info(f"  Compressão: {original_size/1024/1024:.2f}MB → "
                   f"{compressed_size/1024/1024:.2f}MB "
                   f"(economia: {saved/original_size*100:.1f}%)")
        
        # Remover arquivo original
        file_path.unlink()
        
        return compressed_path
    
    def _archive_old_files(self, files: List[Path]):
        """Arquiva arquivos processados"""
        for file in files:
            try:
                # Criar estrutura de diretórios no archive
                relative_path = file.relative_to(self.base_path)
                archive_file = self.archive_path / relative_path
                archive_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Mover arquivo
                shutil.move(str(file), str(archive_file))
                self.stats['files_archived'] += 1
                
            except Exception as e:
                logger.error(f"Erro arquivando {file}: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """
        Remove dados antigos mantendo apenas os últimos N dias
        
        Args:
            days_to_keep: Número de dias para manter
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        logger.info(f"Removendo dados anteriores a {cutoff_date.date()}")
        
        removed_count = 0
        removed_bytes = 0
        
        # Limpar arquivos brutos antigos
        for data_type in ['tick', 'book', 'candles', 'predictions']:
            type_path = self.base_path / data_type
            if not type_path.exists():
                continue
            
            for file in type_path.glob("*.jsonl"):
                try:
                    # Extrair data do arquivo
                    parts = file.stem.split('_')
                    if len(parts) >= 2:
                        date_str = parts[1]
                        file_date = datetime.strptime(date_str, '%Y%m%d')
                        
                        if file_date < cutoff_date:
                            size = file.stat().st_size
                            file.unlink()
                            removed_count += 1
                            removed_bytes += size
                            logger.info(f"  Removido: {file.name}")
                
                except Exception as e:
                    logger.error(f"Erro removendo {file}: {e}")
        
        # Limpar archives muito antigos
        archive_cutoff = datetime.now() - timedelta(days=self.config['archive_days'])
        
        for file in self.archive_path.rglob("*"):
            if file.is_file():
                try:
                    mtime = datetime.fromtimestamp(file.stat().st_mtime)
                    if mtime < archive_cutoff:
                        size = file.stat().st_size
                        file.unlink()
                        removed_count += 1
                        removed_bytes += size
                
                except Exception as e:
                    logger.error(f"Erro removendo archive {file}: {e}")
        
        logger.info(f"Limpeza completa: {removed_count} arquivos removidos "
                   f"({removed_bytes/1024/1024:.2f} MB liberados)")
    
    def get_data_summary(self) -> Dict:
        """Retorna resumo dos dados disponíveis"""
        summary = {
            'data_types': {},
            'total_size_mb': 0,
            'date_range': {},
            'consolidated_files': []
        }
        
        # Analisar cada tipo de dado
        for data_type in ['tick', 'book', 'candles', 'predictions']:
            type_path = self.base_path / data_type
            if not type_path.exists():
                continue
            
            files = list(type_path.glob("*.jsonl"))
            if not files:
                continue
            
            # Calcular tamanho total
            total_size = sum(f.stat().st_size for f in files)
            
            # Encontrar range de datas
            dates = []
            for f in files:
                try:
                    parts = f.stem.split('_')
                    if len(parts) >= 2:
                        date_str = parts[1]
                        dates.append(datetime.strptime(date_str, '%Y%m%d'))
                except:
                    continue
            
            if dates:
                summary['data_types'][data_type] = {
                    'files': len(files),
                    'size_mb': total_size / 1024 / 1024,
                    'oldest': min(dates).strftime('%Y-%m-%d'),
                    'newest': max(dates).strftime('%Y-%m-%d')
                }
                summary['total_size_mb'] += total_size / 1024 / 1024
        
        # Listar arquivos consolidados
        for f in self.consolidated_path.glob("*.parquet*"):
            summary['consolidated_files'].append({
                'name': f.name,
                'size_mb': f.stat().st_size / 1024 / 1024,
                'modified': datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            })
        
        return summary


def main():
    """Exemplo de uso do sistema de consolidação"""
    
    consolidator = DataConsolidator()
    
    print("\n" + "="*60)
    print("SISTEMA DE CONSOLIDAÇÃO DE DADOS")
    print("="*60)
    
    # Mostrar resumo atual
    print("\nResumo dos dados disponíveis:")
    summary = consolidator.get_data_summary()
    
    for data_type, info in summary['data_types'].items():
        print(f"\n{data_type.upper()}:")
        print(f"  Arquivos: {info['files']}")
        print(f"  Tamanho: {info['size_mb']:.2f} MB")
        print(f"  Período: {info['oldest']} a {info['newest']}")
    
    print(f"\nTotal: {summary['total_size_mb']:.2f} MB")
    
    if summary['consolidated_files']:
        print("\nArquivos consolidados:")
        for f in summary['consolidated_files']:
            print(f"  - {f['name']} ({f['size_mb']:.2f} MB)")
    
    # Menu de opções
    print("\n" + "-"*60)
    print("Opções:")
    print("1. Consolidar ticks (últimos 7 dias)")
    print("2. Consolidar todos os dados")
    print("3. Limpar dados antigos (> 30 dias)")
    print("4. Consolidar período específico")
    print("5. Sair")
    
    choice = input("\nEscolha uma opção: ").strip()
    
    if choice == '1':
        # Consolidar últimos 7 dias de ticks
        start_date = datetime.now() - timedelta(days=7)
        output = consolidator.consolidate_continuous(
            data_type='tick',
            start_date=start_date,
            end_date=datetime.now()
        )
        print(f"\nDados consolidados em: {output}")
        
    elif choice == '2':
        # Consolidar todos os tipos
        for data_type in ['tick', 'book', 'candles', 'predictions']:
            print(f"\nConsolidando {data_type}...")
            output = consolidator.consolidate_continuous(data_type=data_type)
            if output:
                print(f"  → {output}")
    
    elif choice == '3':
        # Limpar dados antigos
        days = input("Manter quantos dias? (padrão: 30): ").strip()
        days = int(days) if days else 30
        consolidator.cleanup_old_data(days_to_keep=days)
    
    elif choice == '4':
        # Período específico
        data_type = input("Tipo de dado (tick/book/candles/predictions): ").strip()
        start = input("Data inicial (YYYY-MM-DD): ").strip()
        end = input("Data final (YYYY-MM-DD): ").strip()
        
        start_date = datetime.strptime(start, '%Y-%m-%d') if start else None
        end_date = datetime.strptime(end, '%Y-%m-%d') if end else None
        
        output = consolidator.consolidate_continuous(
            data_type=data_type,
            start_date=start_date,
            end_date=end_date
        )
        print(f"\nDados consolidados em: {output}")
    
    print("\n" + "="*60)
    print("Consolidação finalizada!")
    print("="*60)


if __name__ == "__main__":
    main()