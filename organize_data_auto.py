"""
Organiza a pasta data/realtime/book automaticamente
"""

import os
import shutil
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

def organize_data_folder(date='20250805'):
    """Organiza pasta data automaticamente"""
    
    data_dir = Path(f'data/realtime/book/{date}')
    backup_dir = data_dir / 'backup_non_continuous'
    
    print("=" * 70)
    print("ORGANIZANDO PASTA DATA AUTOMATICAMENTE")
    print("=" * 70)
    
    if not data_dir.exists():
        print(f"Diretório não encontrado: {data_dir}")
        return
        
    # 1. Identificar arquivos
    print("\n=== IDENTIFICANDO ARQUIVOS ===")
    
    all_parquet = list(data_dir.glob('*.parquet'))
    continuous_files = [f for f in all_parquet if 'continuous' in f.name]
    other_files = [f for f in all_parquet if 'continuous' not in f.name]
    
    print(f"Total de arquivos parquet: {len(all_parquet)}")
    print(f"  - Continuous: {len(continuous_files)}")
    print(f"  - Outros: {len(other_files)}")
    
    # 2. Backup de arquivos não-continuous
    if other_files:
        print(f"\n=== MOVENDO {len(other_files)} ARQUIVOS NÃO-CONTINUOUS ===")
        backup_dir.mkdir(exist_ok=True)
        
        for file in other_files:
            try:
                dest = backup_dir / file.name
                shutil.move(str(file), str(dest))
                print(f"  Movido: {file.name}")
                
                # Mover JSON correspondente
                json_patterns = [
                    f"summary_{file.stem}.json",
                    f"{file.stem}.json"
                ]
                
                for pattern in json_patterns:
                    json_file = data_dir / pattern
                    if json_file.exists():
                        json_dest = backup_dir / json_file.name
                        shutil.move(str(json_file), str(json_dest))
                        
            except Exception as e:
                print(f"  Erro: {e}")
                
    # 3. Consolidar arquivos continuous
    if continuous_files:
        print(f"\n=== CONSOLIDANDO {len(continuous_files)} ARQUIVOS CONTINUOUS ===")
        
        all_data = []
        total_records = 0
        
        for i, file in enumerate(sorted(continuous_files)):
            if i % 10 == 0:
                print(f"  Lendo {i+1}/{len(continuous_files)}...")
                
            try:
                df = pd.read_parquet(file)
                all_data.append(df)
                total_records += len(df)
            except Exception as e:
                print(f"  Erro ao ler {file.name}: {e}")
                
        if all_data:
            print(f"\nConcatenando {total_records:,} registros...")
            consolidated = pd.concat(all_data, ignore_index=True)
            
            # Ordenar e limpar
            if 'timestamp' in consolidated.columns:
                consolidated['timestamp'] = pd.to_datetime(consolidated['timestamp'])
                consolidated = consolidated.sort_values('timestamp')
                
                # Adicionar features temporais
                consolidated['hour'] = consolidated['timestamp'].dt.hour
                consolidated['minute'] = consolidated['timestamp'].dt.minute
                consolidated['second'] = consolidated['timestamp'].dt.second
                
            # Remover duplicatas
            before = len(consolidated)
            consolidated = consolidated.drop_duplicates()
            print(f"Removidas {before - len(consolidated):,} duplicatas")
            
            # Criar diretório training
            training_dir = data_dir / 'training'
            training_dir.mkdir(exist_ok=True)
            
            # Salvar
            output_file = training_dir / f'consolidated_training_{date}.parquet'
            consolidated.to_parquet(output_file, compression='snappy', index=False)
            
            print(f"\n[OK] Arquivo consolidado salvo:")
            print(f"  {output_file}")
            print(f"  Registros: {len(consolidated):,}")
            print(f"  Tamanho: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
            
            # Metadados
            metadata = {
                'creation_date': datetime.now().isoformat(),
                'total_records': len(consolidated),
                'source_files': len(continuous_files),
                'columns': list(consolidated.columns),
                'file_size_mb': output_file.stat().st_size / 1024 / 1024
            }
            
            if 'type' in consolidated.columns:
                metadata['type_distribution'] = consolidated['type'].value_counts().to_dict()
                
            if 'timestamp' in consolidated.columns:
                metadata['time_range'] = {
                    'start': consolidated['timestamp'].min().isoformat(),
                    'end': consolidated['timestamp'].max().isoformat()
                }
                
            metadata_file = training_dir / f'metadata_{date}.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
    # 4. Mover continuous para subpasta
    continuous_dir = data_dir / 'continuous'
    continuous_dir.mkdir(exist_ok=True)
    
    remaining_continuous = [f for f in data_dir.glob('wdo_continuous_*.parquet')]
    if remaining_continuous:
        print(f"\n=== ORGANIZANDO {len(remaining_continuous)} ARQUIVOS CONTINUOUS ===")
        
        for file in remaining_continuous:
            dest = continuous_dir / file.name
            shutil.move(str(file), str(dest))
            
            # Mover JSONs relacionados
            json_file = data_dir / f"summary_continuous_{file.stem.split('_')[-1]}.json"
            if json_file.exists():
                json_dest = continuous_dir / json_file.name
                shutil.move(str(json_file), str(json_dest))
                
        print(f"Movidos para continuous/")
        
    # 5. Limpar JSONs órfãos
    orphan_jsons = []
    for json_file in data_dir.glob('*.json'):
        if any(keep in json_file.name for keep in ['metadata', 'consolidation']):
            continue
            
        # Verificar se é órfão
        is_orphan = True
        
        # Procurar parquet correspondente em todos os diretórios
        for subdir in [data_dir, continuous_dir, backup_dir]:
            if subdir.exists():
                base_name = json_file.stem.replace('summary_', '')
                if any(subdir.glob(f"*{base_name}*.parquet")):
                    is_orphan = False
                    break
                    
        if is_orphan:
            orphan_jsons.append(json_file)
            
    if orphan_jsons:
        print(f"\n=== REMOVENDO {len(orphan_jsons)} JSONs ÓRFÃOS ===")
        for json_file in orphan_jsons:
            json_file.unlink()
            print(f"  Removido: {json_file.name}")
            
    # 6. Relatório final
    print("\n" + "=" * 70)
    print("ORGANIZAÇÃO CONCLUÍDA")
    print("=" * 70)
    
    print("\nEstrutura final:")
    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            file_count = len(list(item.glob('*')))
            print(f"  {item.name}/ ({file_count} arquivos)")
            
    # Estatísticas do arquivo consolidado
    final_file = training_dir / f'consolidated_training_{date}.parquet'
    if final_file.exists():
        df_final = pd.read_parquet(final_file)
        
        print(f"\nDataset consolidado para treinamento:")
        print(f"  Arquivo: {final_file.name}")
        print(f"  Registros: {len(df_final):,}")
        print(f"  Colunas: {len(df_final.columns)}")
        
        if 'type' in df_final.columns:
            print(f"\n  Distribuição:")
            for dtype, count in df_final['type'].value_counts().items():
                pct = count / len(df_final) * 100
                print(f"    {dtype}: {count:,} ({pct:.1f}%)")

if __name__ == "__main__":
    organize_data_folder()