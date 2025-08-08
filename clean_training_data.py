"""
Limpa price_book corrompidos do arquivo de treinamento consolidado
"""

import pandas as pd
from pathlib import Path

def clean_training_data(date='20250805'):
    """Remove price_book corrompidos do dataset de treinamento"""
    
    training_file = Path(f'data/realtime/book/{date}/training/consolidated_training_{date}.parquet')
    
    if not training_file.exists():
        print(f"Arquivo não encontrado: {training_file}")
        return
        
    print("=" * 70)
    print("LIMPANDO DATASET DE TREINAMENTO")
    print("=" * 70)
    
    # Ler arquivo
    print(f"\nLendo: {training_file}")
    df = pd.read_parquet(training_file)
    
    print(f"Registros originais: {len(df):,}")
    
    # Analisar price_book
    if 'type' in df.columns and 'price' in df.columns:
        price_book_data = df[df['type'] == 'price_book']
        
        if not price_book_data.empty:
            print(f"\nAnalisando {len(price_book_data):,} registros price_book...")
            
            # Verificar valores
            valid_prices = price_book_data[price_book_data['price'] > 1000]
            invalid_prices = price_book_data[price_book_data['price'] <= 1000]
            
            print(f"  Válidos (price > 1000): {len(valid_prices):,}")
            print(f"  Inválidos (price <= 1000): {len(invalid_prices):,}")
            
            if len(invalid_prices) > 0:
                # Mostrar amostra dos valores inválidos
                sample_invalid = invalid_prices['price'].head(10)
                print(f"\n  Amostra de preços inválidos:")
                for price in sample_invalid:
                    print(f"    {price:.2e}")
                    
                # Remover price_book inválidos
                mask = (df['type'] == 'price_book') & (df['price'] <= 1000)
                df_clean = df[~mask].copy()
                
                removed = len(df) - len(df_clean)
                print(f"\nRemovendo {removed:,} price_book corrompidos...")
                
                # Salvar versão limpa
                df_clean.to_parquet(training_file, compression='snappy', index=False)
                
                print(f"\n[OK] Dataset limpo salvo:")
                print(f"  Registros finais: {len(df_clean):,}")
                print(f"  Redução: {removed:,} registros ({removed/len(df)*100:.1f}%)")
                
                # Nova distribuição
                print(f"\nNova distribuição:")
                for dtype, count in df_clean['type'].value_counts().items():
                    pct = count / len(df_clean) * 100
                    print(f"  {dtype}: {count:,} ({pct:.1f}%)")
                    
                # Atualizar metadados
                metadata_file = training_file.parent / f'metadata_{date}.json'
                if metadata_file.exists():
                    import json
                    
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        
                    metadata['total_records'] = len(df_clean)
                    metadata['type_distribution'] = df_clean['type'].value_counts().to_dict()
                    metadata['cleaning'] = {
                        'price_book_removed': removed,
                        'cleaning_date': pd.Timestamp.now().isoformat()
                    }
                    
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                        
                    print(f"\n[OK] Metadados atualizados")
                    
            else:
                print("\n[OK] Nenhum price_book corrompido encontrado!")
                
    # Análise final de qualidade
    print(f"\n{'='*50}")
    print("ANÁLISE DE QUALIDADE DO DATASET")
    print(f"{'='*50}")
    
    # Recarregar dados limpos
    df_final = pd.read_parquet(training_file)
    
    print(f"\nEstatísticas finais:")
    print(f"  Total de registros: {len(df_final):,}")
    print(f"  Colunas: {len(df_final.columns)}")
    print(f"  Tamanho: {training_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Verificar NaNs
    nan_counts = df_final.isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    
    if not cols_with_nans.empty:
        print(f"\nColunas com valores faltantes:")
        for col, count in cols_with_nans.items():
            pct = count / len(df_final) * 100
            print(f"  {col}: {count:,} ({pct:.1f}%)")
            
    # Range de preços por tipo
    if 'price' in df_final.columns and 'type' in df_final.columns:
        print(f"\nRange de preços por tipo:")
        
        for dtype in df_final['type'].unique():
            type_data = df_final[(df_final['type'] == dtype) & (df_final['price'] > 0)]
            
            if not type_data.empty:
                prices = type_data['price']
                print(f"\n  {dtype}:")
                print(f"    Min: {prices.min():.2f}")
                print(f"    Max: {prices.max():.2f}")
                print(f"    Média: {prices.mean():.2f}")

if __name__ == "__main__":
    clean_training_data()