"""
Limpa dados coletados antes das 10:30 de hoje
Mantém apenas dados contínuos após reconexão
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, time
import pandas as pd
import json

def clean_data_before_cutoff():
    """Remove dados coletados antes das 10:30"""
    
    print("=" * 70)
    print("LIMPEZA DE DADOS COM GAP")
    print("=" * 70)
    
    # Configurações
    today = datetime.now().strftime('%Y%m%d')
    cutoff_time = time(10, 30)  # 10:30
    
    data_dir = Path(f'data/realtime/book/{today}')
    
    if not data_dir.exists():
        print(f"Diretório não encontrado: {data_dir}")
        return
    
    print(f"\nDiretório: {data_dir}")
    print(f"Cutoff: {cutoff_time}")
    
    # Criar diretório de backup
    backup_dir = data_dir / 'backup_before_1030'
    backup_dir.mkdir(exist_ok=True)
    
    # Estatísticas
    stats = {
        'files_analyzed': 0,
        'files_kept': 0,
        'files_moved': 0,
        'continuous_dir_files': 0
    }
    
    # 1. Limpar diretório continuous
    continuous_dir = data_dir / 'continuous'
    if continuous_dir.exists():
        print("\n=== LIMPANDO DIRETÓRIO CONTINUOUS ===")
        
        for file in continuous_dir.glob('*.parquet'):
            stats['files_analyzed'] += 1
            
            # Extrair hora do nome do arquivo
            # wdo_continuous_20250806_103035.parquet
            try:
                time_part = file.stem.split('_')[-1]  # 103035
                hour = int(time_part[:2])
                minute = int(time_part[2:4])
                file_time = time(hour, minute)
                
                if file_time < cutoff_time:
                    # Mover para backup
                    dest = backup_dir / file.name
                    shutil.move(str(file), str(dest))
                    stats['files_moved'] += 1
                    print(f"  Movido: {file.name} ({file_time})")
                    
                    # Mover JSON correspondente
                    json_file = continuous_dir / f"summary_continuous_{file.stem.split('_')[-1]}.json"
                    if json_file.exists():
                        json_dest = backup_dir / json_file.name
                        shutil.move(str(json_file), str(json_dest))
                else:
                    stats['files_kept'] += 1
                    print(f"  Mantido: {file.name} ({file_time})")
                    
            except Exception as e:
                print(f"  Erro ao processar {file.name}: {e}")
    
    # 2. Verificar arquivos na raiz
    print("\n=== VERIFICANDO ARQUIVOS NA RAIZ ===")
    
    for file in data_dir.glob('wdo_continuous_*.parquet'):
        stats['files_analyzed'] += 1
        
        try:
            time_part = file.stem.split('_')[-1]
            hour = int(time_part[:2])
            minute = int(time_part[2:4])
            file_time = time(hour, minute)
            
            if file_time < cutoff_time:
                dest = backup_dir / file.name
                shutil.move(str(file), str(dest))
                stats['files_moved'] += 1
                print(f"  Movido: {file.name}")
        except:
            pass
    
    # 3. Recriar arquivo consolidado de treinamento
    print("\n=== RECRIANDO ARQUIVO CONSOLIDADO ===")
    
    # Coletar apenas arquivos válidos
    valid_files = []
    
    if continuous_dir.exists():
        valid_files.extend(list(continuous_dir.glob('wdo_continuous_*.parquet')))
    
    valid_files.extend([f for f in data_dir.glob('wdo_continuous_*.parquet')])
    
    # Filtrar por horário novamente
    valid_files_filtered = []
    for file in valid_files:
        try:
            time_part = file.stem.split('_')[-1]
            hour = int(time_part[:2])
            minute = int(time_part[2:4])
            file_time = time(hour, minute)
            
            if file_time >= cutoff_time:
                valid_files_filtered.append(file)
        except:
            pass
    
    print(f"\nArquivos válidos para consolidação: {len(valid_files_filtered)}")
    
    if valid_files_filtered:
        # Consolidar dados
        all_data = []
        
        for file in sorted(valid_files_filtered):
            try:
                df = pd.read_parquet(file)
                all_data.append(df)
                print(f"  Lido: {file.name} ({len(df)} registros)")
            except Exception as e:
                print(f"  Erro ao ler {file.name}: {e}")
        
        if all_data:
            # Concatenar
            consolidated = pd.concat(all_data, ignore_index=True)
            
            # Ordenar por timestamp
            if 'timestamp' in consolidated.columns:
                consolidated['timestamp'] = pd.to_datetime(consolidated['timestamp'])
                consolidated = consolidated.sort_values('timestamp')
            
            # Remover duplicatas
            before = len(consolidated)
            consolidated = consolidated.drop_duplicates()
            print(f"\nRemovidas {before - len(consolidated):,} duplicatas")
            
            # Salvar
            training_dir = data_dir / 'training'
            training_dir.mkdir(exist_ok=True)
            
            output_file = training_dir / f'consolidated_continuous_{today}.parquet'
            consolidated.to_parquet(output_file, index=False)
            
            print(f"\n[OK] Novo arquivo consolidado salvo:")
            print(f"  {output_file}")
            print(f"  Registros: {len(consolidated):,}")
            print(f"  Período: {consolidated['timestamp'].min()} até {consolidated['timestamp'].max()}")
            
            # Metadados
            metadata = {
                'creation_date': datetime.now().isoformat(),
                'cutoff_time': str(cutoff_time),
                'source_files': len(valid_files_filtered),
                'total_records': len(consolidated),
                'period': {
                    'start': consolidated['timestamp'].min().isoformat(),
                    'end': consolidated['timestamp'].max().isoformat()
                },
                'cleaning_stats': stats
            }
            
            if 'type' in consolidated.columns:
                metadata['type_distribution'] = consolidated['type'].value_counts().to_dict()
            
            metadata_file = training_dir / f'metadata_continuous_{today}.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  Metadados: {metadata_file.name}")
    
    # 4. Limpar diretórios consolidados antigos
    dirs_to_check = ['consolidated', 'consolidated_hourly', 'training_ready']
    
    for dir_name in dirs_to_check:
        dir_path = data_dir / dir_name
        if dir_path.exists():
            # Mover todo o diretório para backup
            backup_subdir = backup_dir / dir_name
            if backup_subdir.exists():
                shutil.rmtree(backup_subdir)
            shutil.move(str(dir_path), str(backup_subdir))
            print(f"\nMovido diretório {dir_name} para backup")
            
    # Relatório final
    print("\n" + "=" * 70)
    print("RELATÓRIO DA LIMPEZA")
    print("=" * 70)
    print(f"Arquivos analisados: {stats['files_analyzed']}")
    print(f"Arquivos mantidos: {stats['files_kept']}")
    print(f"Arquivos movidos para backup: {stats['files_moved']}")
    print(f"\nBackup em: {backup_dir}")
    
    # Verificar integridade dos dados restantes
    if valid_files_filtered:
        print("\n=== VERIFICAÇÃO DE INTEGRIDADE ===")
        
        # Verificar continuidade temporal
        timestamps = []
        for file in sorted(valid_files_filtered):
            try:
                df = pd.read_parquet(file)
                if 'timestamp' in df.columns:
                    timestamps.extend(pd.to_datetime(df['timestamp']).tolist())
            except:
                pass
        
        if timestamps:
            timestamps = sorted(timestamps)
            time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            max_gap = max(time_diffs) if time_diffs else pd.Timedelta(0)
            
            print(f"Período contínuo: {min(timestamps)} até {max(timestamps)}")
            print(f"Maior gap entre registros: {max_gap}")
            
            if max_gap > pd.Timedelta(minutes=5):
                print("[AVISO] Ainda existem gaps maiores que 5 minutos")
            else:
                print("[OK] Dados contínuos sem gaps significativos")


if __name__ == "__main__":
    clean_data_before_cutoff()