"""
Organiza a pasta data/realtime/book removendo arquivos desnecessários
e criando arquivo consolidado único para treinamento
"""

import os
import shutil
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

class DataOrganizer:
    def __init__(self, date='20250805'):
        self.date = date
        self.data_dir = Path(f'data/realtime/book/{date}')
        self.backup_dir = Path(f'data/realtime/book/{date}/backup_non_continuous')
        self.stats = {
            'files_analyzed': 0,
            'files_kept': 0,
            'files_moved': 0,
            'files_deleted': 0,
            'total_records': 0
        }
        
    def analyze_files(self):
        """Analisa todos os arquivos na pasta"""
        print("=" * 70)
        print("ANALISANDO ARQUIVOS NA PASTA DATA")
        print("=" * 70)
        
        if not self.data_dir.exists():
            print(f"Diretório não encontrado: {self.data_dir}")
            return
            
        # Listar todos os arquivos parquet
        parquet_files = list(self.data_dir.glob('*.parquet'))
        json_files = list(self.data_dir.glob('*.json'))
        
        print(f"\nTotal de arquivos encontrados:")
        print(f"  Parquet: {len(parquet_files)}")
        print(f"  JSON: {len(json_files)}")
        
        # Categorizar arquivos
        continuous_files = []
        other_files = []
        
        for file in parquet_files:
            self.stats['files_analyzed'] += 1
            
            if 'continuous' in file.name:
                continuous_files.append(file)
            else:
                other_files.append(file)
                
        print(f"\nCategorização:")
        print(f"  Arquivos continuous: {len(continuous_files)}")
        print(f"  Outros arquivos: {len(other_files)}")
        
        return continuous_files, other_files, json_files
        
    def backup_non_continuous_files(self, files_to_backup):
        """Move arquivos não-continuous para backup"""
        if not files_to_backup:
            return
            
        print(f"\n=== FAZENDO BACKUP DE ARQUIVOS NÃO-CONTINUOUS ===")
        
        # Criar diretório de backup
        self.backup_dir.mkdir(exist_ok=True)
        
        for file in files_to_backup:
            try:
                dest = self.backup_dir / file.name
                shutil.move(str(file), str(dest))
                self.stats['files_moved'] += 1
                print(f"  Movido: {file.name}")
                
                # Mover JSON correspondente se existir
                json_file = file.parent / f"summary_{file.stem}.json"
                if json_file.exists():
                    json_dest = self.backup_dir / json_file.name
                    shutil.move(str(json_file), str(json_dest))
                    
            except Exception as e:
                print(f"  Erro ao mover {file.name}: {e}")
                
    def create_consolidated_training_file(self, continuous_files):
        """Cria arquivo consolidado único para treinamento"""
        if not continuous_files:
            print("Nenhum arquivo continuous para consolidar")
            return None
            
        print(f"\n=== CRIANDO ARQUIVO CONSOLIDADO PARA TREINAMENTO ===")
        print(f"Consolidando {len(continuous_files)} arquivos...")
        
        all_data = []
        
        # Ler todos os arquivos continuous
        for i, file in enumerate(sorted(continuous_files)):
            try:
                print(f"  Lendo {i+1}/{len(continuous_files)}: {file.name}")
                df = pd.read_parquet(file)
                all_data.append(df)
                self.stats['total_records'] += len(df)
            except Exception as e:
                print(f"  Erro ao ler {file.name}: {e}")
                
        if not all_data:
            print("Nenhum dado para consolidar")
            return None
            
        # Concatenar todos os dados
        print("\nConcatenando dados...")
        consolidated_df = pd.concat(all_data, ignore_index=True)
        
        # Ordenar por timestamp
        if 'timestamp' in consolidated_df.columns:
            consolidated_df['timestamp'] = pd.to_datetime(consolidated_df['timestamp'])
            consolidated_df = consolidated_df.sort_values('timestamp')
            
        # Adicionar features temporais
        print("Adicionando features temporais...")
        if 'timestamp' in consolidated_df.columns:
            consolidated_df['hour'] = consolidated_df['timestamp'].dt.hour
            consolidated_df['minute'] = consolidated_df['timestamp'].dt.minute
            consolidated_df['second'] = consolidated_df['timestamp'].dt.second
            consolidated_df['day_of_week'] = consolidated_df['timestamp'].dt.dayofweek
            
        # Remover duplicatas
        original_len = len(consolidated_df)
        consolidated_df = consolidated_df.drop_duplicates()
        duplicates_removed = original_len - len(consolidated_df)
        
        if duplicates_removed > 0:
            print(f"Removidas {duplicates_removed:,} duplicatas")
            
        # Salvar arquivo consolidado
        output_file = self.data_dir / f'consolidated_training_{self.date}.parquet'
        consolidated_df.to_parquet(output_file, compression='snappy', index=False)
        
        print(f"\n✓ Arquivo consolidado criado: {output_file.name}")
        print(f"  Total de registros: {len(consolidated_df):,}")
        print(f"  Tamanho: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Criar metadados
        self._create_metadata(consolidated_df, output_file)
        
        return output_file
        
    def _create_metadata(self, df, output_file):
        """Cria arquivo de metadados do dataset consolidado"""
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'source_date': self.date,
            'total_records': len(df),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'file_size_mb': output_file.stat().st_size / 1024 / 1024,
            'organization_stats': self.stats
        }
        
        # Estatísticas por tipo
        if 'type' in df.columns:
            metadata['type_distribution'] = df['type'].value_counts().to_dict()
            
        # Range temporal
        if 'timestamp' in df.columns:
            metadata['time_range'] = {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat(),
                'duration_hours': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            }
            
        # Estatísticas de preço
        if 'price' in df.columns:
            valid_prices = df[df['price'] > 0]['price']
            if not valid_prices.empty:
                metadata['price_stats'] = {
                    'min': float(valid_prices.min()),
                    'max': float(valid_prices.max()),
                    'mean': float(valid_prices.mean()),
                    'std': float(valid_prices.std())
                }
                
        # Salvar metadados
        metadata_file = output_file.parent / f'metadata_training_{self.date}.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        print(f"  Metadados salvos: {metadata_file.name}")
        
    def clean_json_files(self):
        """Remove arquivos JSON órfãos"""
        print("\n=== LIMPANDO ARQUIVOS JSON ===")
        
        json_files = list(self.data_dir.glob('*.json'))
        removed = 0
        
        for json_file in json_files:
            # Pular metadados importantes
            if any(keep in json_file.name for keep in ['metadata', 'consolidation']):
                continue
                
            # Verificar se tem parquet correspondente
            if json_file.name.startswith('summary_'):
                base_name = json_file.stem.replace('summary_', '')
                parquet_exists = (self.data_dir / f"{base_name}.parquet").exists()
                
                if not parquet_exists:
                    # Verificar no backup também
                    backup_exists = (self.backup_dir / f"{base_name}.parquet").exists() if self.backup_dir.exists() else False
                    
                    if not backup_exists:
                        json_file.unlink()
                        removed += 1
                        print(f"  Removido JSON órfão: {json_file.name}")
                        
        print(f"Total de JSONs removidos: {removed}")
        
    def organize_final_structure(self):
        """Organiza estrutura final da pasta"""
        print("\n=== ORGANIZANDO ESTRUTURA FINAL ===")
        
        # Mover arquivos consolidados anteriores para consolidated/
        consolidated_files = list(self.data_dir.glob('consolidated_*.parquet'))
        
        if consolidated_files and not (self.data_dir / 'consolidated').exists():
            print("Mantendo diretório 'consolidated' existente")
            
        # Criar estrutura sugerida
        structure = {
            'continuous': "Arquivos originais do continuous collector",
            'training': "Datasets prontos para treinamento",
            'backup_non_continuous': "Backup de arquivos não-continuous"
        }
        
        for dir_name, description in structure.items():
            dir_path = self.data_dir / dir_name
            if not dir_path.exists() and dir_name != 'backup_non_continuous':  # backup já foi criado
                dir_path.mkdir(exist_ok=True)
                print(f"  Criado: {dir_name}/ - {description}")
                
        # Mover arquivo de treinamento consolidado
        training_file = self.data_dir / f'consolidated_training_{self.date}.parquet'
        if training_file.exists():
            training_dir = self.data_dir / 'training'
            training_dir.mkdir(exist_ok=True)
            
            dest = training_dir / training_file.name
            shutil.move(str(training_file), str(dest))
            
            # Mover metadados também
            metadata_file = self.data_dir / f'metadata_training_{self.date}.json'
            if metadata_file.exists():
                metadata_dest = training_dir / metadata_file.name
                shutil.move(str(metadata_file), str(metadata_dest))
                
            print(f"  Movido arquivo de treinamento para: training/")
            
        # Mover arquivos continuous para subpasta
        continuous_dir = self.data_dir / 'continuous'
        continuous_dir.mkdir(exist_ok=True)
        
        continuous_files = [f for f in self.data_dir.glob('wdo_continuous_*.parquet')]
        for file in continuous_files:
            dest = continuous_dir / file.name
            shutil.move(str(file), str(dest))
            
            # Mover JSON correspondente
            json_file = self.data_dir / f"summary_continuous_{file.stem.split('_')[-1]}.json"
            if json_file.exists():
                json_dest = continuous_dir / json_file.name
                shutil.move(str(json_file), str(json_dest))
                
        if continuous_files:
            print(f"  Movidos {len(continuous_files)} arquivos continuous")
            
    def generate_report(self):
        """Gera relatório final"""
        print("\n" + "=" * 70)
        print("RELATÓRIO FINAL DA ORGANIZAÇÃO")
        print("=" * 70)
        
        print(f"\nEstatísticas:")
        print(f"  Arquivos analisados: {self.stats['files_analyzed']}")
        print(f"  Arquivos mantidos: {self.stats['files_kept']}")
        print(f"  Arquivos movidos para backup: {self.stats['files_moved']}")
        print(f"  Total de registros consolidados: {self.stats['total_records']:,}")
        
        print(f"\nEstrutura final:")
        print(f"  {self.data_dir}/")
        print(f"    ├── consolidated/          # Consolidações por tipo (existente)")
        print(f"    ├── continuous/           # Arquivos originais do collector")
        print(f"    ├── training/             # Dataset único para ML")
        print(f"    ├── training_ready/       # Dataset otimizado (existente)")
        print(f"    └── backup_non_continuous/ # Backup de outros arquivos")
        
def main():
    print("ORGANIZADOR DE DADOS DO BOOK COLLECTOR")
    print("=" * 70)
    
    organizer = DataOrganizer()
    
    # 1. Analisar arquivos
    continuous_files, other_files, json_files = organizer.analyze_files()
    
    # 2. Fazer backup de arquivos não-continuous
    if other_files:
        response = input(f"\nMover {len(other_files)} arquivos não-continuous para backup? (s/n): ")
        if response.lower() == 's':
            organizer.backup_non_continuous_files(other_files)
            
    # 3. Criar arquivo consolidado único
    if continuous_files:
        organizer.create_consolidated_training_file(continuous_files)
        
    # 4. Limpar JSONs órfãos
    organizer.clean_json_files()
    
    # 5. Organizar estrutura final
    organizer.organize_final_structure()
    
    # 6. Gerar relatório
    organizer.generate_report()
    
    print("\n✅ Organização concluída!")

if __name__ == "__main__":
    main()