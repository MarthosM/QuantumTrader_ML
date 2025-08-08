"""
Limpa dados corrompidos de price_book dos arquivos parquet
Mantém todos os outros tipos de dados válidos
"""

import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

def clean_parquet_file(file_path):
    """Limpa dados corrompidos de um arquivo parquet"""
    try:
        # Ler arquivo
        df = pd.read_parquet(file_path)
        original_size = len(df)
        
        # Identificar registros price_book corrompidos
        if 'type' in df.columns and 'price' in df.columns:
            # Encontrar price_book com valores inválidos
            price_book_mask = (df['type'] == 'price_book')
            invalid_price_mask = price_book_mask & (df['price'] < 1)
            
            if invalid_price_mask.any():
                # Remover apenas price_book com preços inválidos
                df_clean = df[~invalid_price_mask].copy()
                
                removed_count = original_size - len(df_clean)
                
                # Estatísticas dos dados mantidos
                stats = {
                    'original_size': original_size,
                    'cleaned_size': len(df_clean),
                    'removed_count': removed_count,
                    'types_remaining': df_clean['type'].value_counts().to_dict() if 'type' in df_clean.columns else {}
                }
                
                return df_clean, stats
            else:
                # Arquivo já está limpo
                return df, {'original_size': original_size, 'cleaned_size': original_size, 'removed_count': 0}
                
        return df, {'original_size': original_size, 'cleaned_size': original_size, 'removed_count': 0}
        
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None, None

def main():
    data_dir = Path('data/realtime/book/20250805')
    
    if not data_dir.exists():
        print(f"Diretório não encontrado: {data_dir}")
        return
        
    # Criar diretório para arquivos limpos
    clean_dir = data_dir / 'cleaned'
    clean_dir.mkdir(exist_ok=True)
    
    # Criar backup dos originais
    backup_dir = data_dir / 'backup_with_corrupted_price_book'
    backup_dir.mkdir(exist_ok=True)
    
    parquet_files = list(data_dir.glob('*.parquet'))
    print(f"\nProcessando {len(parquet_files)} arquivos...\n")
    
    total_removed = 0
    processed_files = 0
    
    for file in sorted(parquet_files):
        print(f"Processando: {file.name}")
        
        df_clean, stats = clean_parquet_file(file)
        
        if df_clean is not None and stats['removed_count'] > 0:
            # Fazer backup do original
            backup_path = backup_dir / file.name
            shutil.copy2(file, backup_path)
            
            # Salvar versão limpa no local original
            df_clean.to_parquet(file, compression='snappy', index=False)
            
            print(f"  [OK] Removidos {stats['removed_count']:,} registros price_book corrompidos")
            print(f"    Dados mantidos: {stats['cleaned_size']:,} registros")
            
            if stats['types_remaining']:
                print(f"    Tipos restantes: {stats['types_remaining']}")
                
            total_removed += stats['removed_count']
            processed_files += 1
            
        elif df_clean is not None:
            print(f"  [OK] Arquivo já está limpo")
            
    print(f"\n{'='*70}")
    print(f"RESUMO DA LIMPEZA:")
    print(f"{'='*70}")
    print(f"Arquivos processados: {processed_files}")
    print(f"Total de registros removidos: {total_removed:,}")
    print(f"\nBackup dos originais em: {backup_dir}")
    
    # Análise dos dados limpos
    print(f"\n{'='*70}")
    print(f"ANÁLISE DOS DADOS LIMPOS:")
    print(f"{'='*70}")
    
    all_types = {}
    total_records = 0
    
    for file in data_dir.glob('*.parquet'):
        try:
            df = pd.read_parquet(file)
            total_records += len(df)
            
            if 'type' in df.columns:
                for dtype, count in df['type'].value_counts().items():
                    all_types[dtype] = all_types.get(dtype, 0) + count
                    
        except:
            pass
            
    print(f"\nTotal de registros limpos: {total_records:,}")
    print(f"\nDistribuição por tipo:")
    for dtype, count in sorted(all_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_records * 100) if total_records > 0 else 0
        print(f"  {dtype:15}: {count:10,} ({percentage:5.1f}%)")

if __name__ == "__main__":
    main()