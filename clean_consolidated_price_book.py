"""
Limpa price_book corrompidos dos arquivos consolidados
"""

import pandas as pd
from pathlib import Path
import shutil

def clean_consolidated_files():
    """Limpa price_book dos arquivos consolidados"""
    
    consolidated_dir = Path('data/realtime/book/20250805/consolidated')
    
    # Arquivo price_book consolidado
    price_book_file = consolidated_dir / 'consolidated_price_book_20250805.parquet'
    
    if price_book_file.exists():
        print(f"Analisando: {price_book_file.name}")
        
        df = pd.read_parquet(price_book_file)
        print(f"Total de registros: {len(df):,}")
        
        # Filtrar apenas valores válidos (preços maiores que 1000)
        df_clean = df[df['price'] > 1000].copy()
        
        if len(df_clean) == 0:
            print(f"[AVISO] Nenhum price_book válido encontrado - removendo arquivo")
            price_book_file.unlink()
        else:
            print(f"Registros válidos: {len(df_clean):,}")
            df_clean.to_parquet(price_book_file, compression='snappy', index=False)
            
    # Limpar do arquivo completo
    complete_file = consolidated_dir / 'consolidated_complete_20250805.parquet'
    
    if complete_file.exists():
        print(f"\nLimpando price_book de: {complete_file.name}")
        
        df = pd.read_parquet(complete_file)
        original_size = len(df)
        
        # Remover price_book com valores inválidos
        if 'type' in df.columns and 'price' in df.columns:
            mask = (df['type'] == 'price_book') & (df['price'] < 1000)
            df_clean = df[~mask].copy()
            
            removed = original_size - len(df_clean)
            if removed > 0:
                print(f"Removidos {removed:,} price_book corrompidos")
                df_clean.to_parquet(complete_file, compression='snappy', index=False)
            else:
                print("Nenhum price_book corrompido encontrado")
                
    # Limpar do training_ready
    training_file = Path('data/realtime/book/20250805/training_ready/training_data_20250805.parquet')
    
    if training_file.exists():
        print(f"\nLimpando price_book de: {training_file.name}")
        
        df = pd.read_parquet(training_file)
        original_size = len(df)
        
        # Remover price_book com valores inválidos
        if 'type' in df.columns and 'price' in df.columns:
            mask = (df['type'] == 'price_book') & (df['price'] < 1000)
            df_clean = df[~mask].copy()
            
            removed = original_size - len(df_clean)
            if removed > 0:
                print(f"Removidos {removed:,} price_book corrompidos")
                df_clean.to_parquet(training_file, compression='snappy', index=False)
            else:
                print("Nenhum price_book corrompido encontrado")
                
    print("\n[OK] Limpeza concluída!")

if __name__ == "__main__":
    clean_consolidated_files()