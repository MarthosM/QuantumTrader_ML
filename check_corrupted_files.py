"""
Verificar arquivos corrompidos no diretório de book data
"""

import pandas as pd
from pathlib import Path
import json

def check_file_integrity(file_path):
    """Verifica integridade de um arquivo parquet"""
    try:
        df = pd.read_parquet(file_path)
        
        # Verificar se tem dados
        if df.empty:
            return False, "Arquivo vazio"
            
        # Verificar colunas essenciais
        if 'type' not in df.columns:
            return False, "Sem coluna 'type'"
            
        # Verificar valores de price_book
        if 'price' in df.columns:
            price_book_data = df[(df['type'] == 'price_book') & (df['price'] > 0)]
            if not price_book_data.empty:
                # Verificar se os preços são válidos (não são valores próximos de zero)
                min_price = price_book_data['price'].min()
                max_price = price_book_data['price'].max()
                
                if min_price < 1e-10 or max_price < 1:  # Valores muito pequenos
                    return False, f"price_book com valores inválidos: min={min_price:.2e}, max={max_price:.2e}"
                    
        # Verificar outros tipos
        valid_types = df[df['type'].isin(['tiny_book', 'offer_book', 'daily', 'trade'])]
        if len(valid_types) > 0:
            return True, f"OK - {len(df)} registros, tipos: {df['type'].unique()}"
        else:
            return False, "Sem dados válidos de book"
            
    except Exception as e:
        return False, f"Erro ao ler: {str(e)}"

def main():
    data_dir = Path('data/realtime/book/20250805')
    
    if not data_dir.exists():
        print(f"Diretório não encontrado: {data_dir}")
        return
        
    parquet_files = list(data_dir.glob('*.parquet'))
    print(f"\nVerificando {len(parquet_files)} arquivos parquet...\n")
    
    valid_files = []
    corrupted_files = []
    
    for file in sorted(parquet_files):
        is_valid, message = check_file_integrity(file)
        
        if is_valid:
            valid_files.append(file)
            print(f"[OK] {file.name}: {message}")
        else:
            corrupted_files.append((file, message))
            print(f"[ERRO] {file.name}: {message}")
            
    # Resumo
    print(f"\n{'='*70}")
    print(f"RESUMO:")
    print(f"{'='*70}")
    print(f"Arquivos válidos: {len(valid_files)}")
    print(f"Arquivos corrompidos: {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"\nARQUIVOS CORROMPIDOS ENCONTRADOS:")
        for file, reason in corrupted_files:
            print(f"  - {file.name}: {reason}")
            
        # Perguntar se deseja deletar
        response = input("\nDeseja deletar os arquivos corrompidos? (s/n): ")
        if response.lower() == 's':
            for file, _ in corrupted_files:
                try:
                    file.unlink()
                    print(f"  Deletado: {file.name}")
                except Exception as e:
                    print(f"  Erro ao deletar {file.name}: {e}")
                    
            print(f"\n[OK] {len(corrupted_files)} arquivos corrompidos foram deletados")
            
            # Deletar JSONs correspondentes
            json_files = list(data_dir.glob('*.json'))
            for json_file in json_files:
                # Verificar se existe parquet correspondente
                base_name = json_file.stem.replace('summary_', '')
                has_parquet = any(pf.stem.endswith(base_name.split('_')[-1]) for pf in valid_files)
                
                if not has_parquet:
                    try:
                        json_file.unlink()
                        print(f"  Deletado JSON órfão: {json_file.name}")
                    except:
                        pass

if __name__ == "__main__":
    main()