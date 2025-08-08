"""
Consolidação de dados do Book Collector
Combina múltiplos arquivos parquet em arquivos consolidados por tipo
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import List, Dict, Optional
import pyarrow.parquet as pq
import pyarrow as pa

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BookDataConsolidator:
    def __init__(self, date: Optional[str] = None):
        self.logger = logging.getLogger('BookConsolidator')
        self.date = date or datetime.now().strftime('%Y%m%d')
        self.data_dir = Path('data/realtime/book') / self.date
        
    def list_parquet_files(self) -> List[Path]:
        """Lista todos os arquivos parquet do dia"""
        if not self.data_dir.exists():
            self.logger.error(f"Diretório não encontrado: {self.data_dir}")
            return []
            
        files = list(self.data_dir.glob('*.parquet'))
        self.logger.info(f"Encontrados {len(files)} arquivos parquet")
        return sorted(files)
        
    def analyze_files(self) -> Dict:
        """Analisa os arquivos para entender a estrutura"""
        files = self.list_parquet_files()
        
        analysis = {
            'total_files': len(files),
            'total_size_mb': sum(f.stat().st_size for f in files) / 1024 / 1024,
            'file_patterns': {},
            'data_types': {},
            'total_records': 0
        }
        
        # Agrupar por padrão de nome
        for file in files:
            pattern = file.name.split('_')[0]  # wdo
            if pattern not in analysis['file_patterns']:
                analysis['file_patterns'][pattern] = []
            analysis['file_patterns'][pattern].append(file)
            
        # Analisar tipos de dados
        sample_files = files[:5] if len(files) > 5 else files
        for file in sample_files:
            try:
                df = pd.read_parquet(file)
                if 'type' in df.columns:
                    for data_type in df['type'].unique():
                        if data_type not in analysis['data_types']:
                            analysis['data_types'][data_type] = 0
                        analysis['data_types'][data_type] += len(df[df['type'] == data_type])
                analysis['total_records'] += len(df)
            except Exception as e:
                self.logger.error(f"Erro ao ler {file}: {e}")
                
        return analysis
        
    def consolidate_by_type(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Consolida arquivos separando por tipo de dado"""
        if output_dir is None:
            output_dir = self.data_dir / 'consolidated'
        output_dir.mkdir(exist_ok=True)
        
        files = self.list_parquet_files()
        if not files:
            return {}
            
        self.logger.info("Iniciando consolidação por tipo...")
        
        # Ler todos os arquivos
        all_data = []
        for i, file in enumerate(files):
            if i % 10 == 0:
                self.logger.info(f"Lendo arquivo {i+1}/{len(files)}...")
            try:
                df = pd.read_parquet(file)
                all_data.append(df)
            except Exception as e:
                self.logger.error(f"Erro ao ler {file}: {e}")
                
        # Combinar todos os dados
        self.logger.info("Combinando dados...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Estatísticas gerais
        self.logger.info(f"Total de registros: {len(combined_df):,}")
        
        output_files = {}
        
        # Separar por tipo se existir coluna 'type'
        if 'type' in combined_df.columns:
            for data_type in combined_df['type'].unique():
                type_df = combined_df[combined_df['type'] == data_type].copy()
                
                # Ordenar por timestamp
                if 'timestamp' in type_df.columns:
                    type_df['timestamp'] = pd.to_datetime(type_df['timestamp'])
                    type_df = type_df.sort_values('timestamp')
                    
                # Salvar arquivo consolidado
                output_file = output_dir / f'consolidated_{data_type}_{self.date}.parquet'
                type_df.to_parquet(output_file, compression='snappy', index=False)
                
                output_files[data_type] = output_file
                self.logger.info(f"✓ {data_type}: {len(type_df):,} registros → {output_file.name}")
                
        else:
            # Se não tiver tipo, salvar tudo junto
            output_file = output_dir / f'consolidated_all_{self.date}.parquet'
            combined_df.to_parquet(output_file, compression='snappy', index=False)
            output_files['all'] = output_file
            self.logger.info(f"✓ Todos os dados: {len(combined_df):,} registros → {output_file.name}")
            
        # Criar arquivo único com todos os dados também
        full_output = output_dir / f'consolidated_complete_{self.date}.parquet'
        combined_df.to_parquet(full_output, compression='snappy', index=False)
        output_files['complete'] = full_output
        
        # Salvar metadados
        self._save_metadata(output_dir, combined_df, output_files)
        
        return output_files
        
    def consolidate_by_hour(self) -> Dict[str, Path]:
        """Consolida arquivos agrupando por hora"""
        output_dir = self.data_dir / 'consolidated_hourly'
        output_dir.mkdir(exist_ok=True)
        
        files = self.list_parquet_files()
        self.logger.info("Consolidando por hora...")
        
        hourly_data = {}
        
        for file in files:
            try:
                # Extrair hora do nome do arquivo
                time_part = file.stem.split('_')[-1]  # HHMMSS
                hour = time_part[:2]
                
                if hour not in hourly_data:
                    hourly_data[hour] = []
                    
                df = pd.read_parquet(file)
                hourly_data[hour].append(df)
                
            except Exception as e:
                self.logger.error(f"Erro ao processar {file}: {e}")
                
        # Salvar arquivos por hora
        output_files = {}
        for hour, dfs in hourly_data.items():
            if dfs:
                hourly_df = pd.concat(dfs, ignore_index=True)
                output_file = output_dir / f'consolidated_hour_{hour}_{self.date}.parquet'
                hourly_df.to_parquet(output_file, compression='snappy', index=False)
                output_files[f'hour_{hour}'] = output_file
                self.logger.info(f"✓ Hora {hour}: {len(hourly_df):,} registros")
                
        return output_files
        
    def _save_metadata(self, output_dir: Path, df: pd.DataFrame, output_files: Dict):
        """Salva metadados da consolidação"""
        metadata = {
            'consolidation_date': datetime.now().isoformat(),
            'source_date': self.date,
            'total_records': len(df),
            'data_types': {},
            'files_created': {k: str(v) for k, v in output_files.items()},
            'time_range': {},
            'statistics': {}
        }
        
        # Estatísticas por tipo
        if 'type' in df.columns:
            for data_type in df['type'].unique():
                type_df = df[df['type'] == data_type]
                metadata['data_types'][data_type] = len(type_df)
                
                if 'price' in type_df.columns:
                    prices = type_df['price'][type_df['price'] > 0]
                    if not prices.empty:
                        metadata['statistics'][data_type] = {
                            'min_price': float(prices.min()),
                            'max_price': float(prices.max()),
                            'mean_price': float(prices.mean()),
                            'std_price': float(prices.std())
                        }
                        
        # Range temporal
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            metadata['time_range'] = {
                'start': timestamps.min().isoformat(),
                'end': timestamps.max().isoformat(),
                'duration_hours': (timestamps.max() - timestamps.min()).total_seconds() / 3600
            }
            
        # Salvar metadados
        metadata_file = output_dir / f'consolidation_metadata_{self.date}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Metadados salvos em: {metadata_file}")
        
    def create_training_ready_dataset(self) -> Path:
        """Cria dataset otimizado para treinamento"""
        output_dir = self.data_dir / 'training_ready'
        output_dir.mkdir(exist_ok=True)
        
        self.logger.info("Criando dataset para treinamento...")
        
        # Primeiro consolidar por tipo
        consolidated = self.consolidate_by_type()
        
        if 'complete' in consolidated:
            df = pd.read_parquet(consolidated['complete'])
            
            # Adicionar features temporais
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['minute'] = df['timestamp'].dt.minute
                df['second'] = df['timestamp'].dt.second
                
            # Remover duplicatas
            df = df.drop_duplicates()
            
            # Ordenar por timestamp
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
                
            # Salvar versão otimizada
            output_file = output_dir / f'training_data_{self.date}.parquet'
            df.to_parquet(output_file, compression='snappy', index=False)
            
            self.logger.info(f"✓ Dataset de treinamento criado: {output_file}")
            self.logger.info(f"  Shape: {df.shape}")
            self.logger.info(f"  Memória: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            return output_file
            
        return None

def consolidate_auto(date: Optional[str] = None, consolidate_all: bool = True):
    """Consolidação automática sem interação do usuário"""
    print("\n" + "="*70)
    print("CONSOLIDADOR AUTOMÁTICO DE DADOS DO BOOK COLLECTOR")
    print("="*70)
    
    consolidator = BookDataConsolidator(date)
    
    # Analisar primeiro
    print("\n=== ANÁLISE DOS ARQUIVOS ===")
    analysis = consolidator.analyze_files()
    
    if analysis['total_files'] == 0:
        print("Nenhum arquivo encontrado para consolidar")
        return
    
    print(f"\nTotal de arquivos: {analysis['total_files']}")
    print(f"Tamanho total: {analysis['total_size_mb']:.2f} MB")
    print(f"Total de registros (amostra): {analysis['total_records']:,}")
    
    if analysis['data_types']:
        print("\nTipos de dados encontrados:")
        for dtype, count in analysis['data_types'].items():
            print(f"  {dtype}: {count:,}")
    
    # Consolidar automaticamente
    print("\n=== INICIANDO CONSOLIDAÇÃO AUTOMÁTICA ===")
    
    # 1. Consolidar por tipo
    print("\n[1/3] Consolidando por tipo de dado...")
    type_files = consolidator.consolidate_by_type()
    
    if consolidate_all:
        # 2. Consolidar por hora
        print("\n[2/3] Consolidando por hora...")
        hourly_files = consolidator.consolidate_by_hour()
        
        # 3. Criar dataset de treinamento
        print("\n[3/3] Criando dataset otimizado para treinamento...")
        training_file = consolidator.create_training_ready_dataset()
    
    print("\n[OK] Consolidacao automatica concluida!")
    
    # Resumo final
    print("\n=== RESUMO DA CONSOLIDAÇÃO ===")
    print(f"Data processada: {consolidator.date}")
    print(f"Diretório: {consolidator.data_dir}")
    print(f"\nArquivos criados:")
    
    consolidated_dir = consolidator.data_dir / 'consolidated'
    if consolidated_dir.exists():
        for file in consolidated_dir.glob('*.parquet'):
            size_mb = file.stat().st_size / 1024 / 1024
            print(f"  - {file.name} ({size_mb:.2f} MB)")
    
    return type_files


def main():
    """Função principal com menu interativo"""
    import sys
    
    # Verificar se foi chamado com argumentos para modo automático
    if len(sys.argv) > 1:
        if sys.argv[1] == '--auto':
            date = sys.argv[2] if len(sys.argv) > 2 else None
            consolidate_auto(date)
            return
    
    # Modo interativo original
    print("\n" + "="*70)
    print("CONSOLIDADOR DE DADOS DO BOOK COLLECTOR")
    print("="*70)
    
    # Opções
    print("\nOpções:")
    print("1. Consolidar dados de hoje")
    print("2. Consolidar dados de uma data específica")
    print("3. Analisar arquivos apenas")
    print("4. Consolidação automática (sem interação)")
    
    choice = input("\nEscolha (1-4): ").strip()
    
    if choice == '4':
        # Modo automático
        date = None
        if input("Deseja especificar uma data? (s/n): ").lower() == 's':
            date = input("Digite a data (YYYYMMDD): ").strip()
        consolidate_auto(date)
        return
    
    if choice == '2':
        date = input("Digite a data (YYYYMMDD): ").strip()
        consolidator = BookDataConsolidator(date)
    else:
        consolidator = BookDataConsolidator()
        
    # Analisar primeiro
    print("\n=== ANÁLISE DOS ARQUIVOS ===")
    analysis = consolidator.analyze_files()
    
    print(f"\nTotal de arquivos: {analysis['total_files']}")
    print(f"Tamanho total: {analysis['total_size_mb']:.2f} MB")
    print(f"Total de registros (amostra): {analysis['total_records']:,}")
    
    if analysis['data_types']:
        print("\nTipos de dados encontrados:")
        for dtype, count in analysis['data_types'].items():
            print(f"  {dtype}: {count:,}")
            
    if choice == '3':
        return
        
    # Consolidar
    print("\n=== CONSOLIDAÇÃO ===")
    print("1. Por tipo de dado")
    print("2. Por hora")
    print("3. Dataset pronto para treinamento")
    print("4. Todas as opções")
    
    consolidation_choice = input("\nEscolha (1-4): ").strip()
    
    if consolidation_choice in ['1', '4']:
        print("\nConsolidando por tipo...")
        consolidator.consolidate_by_type()
        
    if consolidation_choice in ['2', '4']:
        print("\nConsolidando por hora...")
        consolidator.consolidate_by_hour()
        
    if consolidation_choice in ['3', '4']:
        print("\nCriando dataset de treinamento...")
        consolidator.create_training_ready_dataset()
        
    print("\n✅ Consolidação concluída!")

if __name__ == "__main__":
    main()