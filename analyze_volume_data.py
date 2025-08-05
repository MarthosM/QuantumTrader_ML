"""
Análise de dados de volume coletados
Verifica se os deltas estão corretos
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

def analyze_volume_data(parquet_file):
    """Analisa dados de volume de um arquivo parquet"""
    print(f"\nAnalisando: {parquet_file}")
    
    # Carregar dados
    df = pd.read_parquet(parquet_file)
    
    # Filtrar apenas dados daily
    daily_df = df[df['type'] == 'daily'].copy()
    
    if daily_df.empty:
        print("Nenhum dado daily encontrado")
        return
        
    print(f"Total de registros daily: {len(daily_df)}")
    
    # Se tiver as novas colunas de delta
    if 'volume_delta' in daily_df.columns:
        print("\n=== ANÁLISE DE VOLUME COM DELTAS ===")
        
        # Estatísticas de volume delta
        vol_delta = daily_df['volume_delta'][daily_df['volume_delta'] > 0]
        if not vol_delta.empty:
            print(f"\nVolume Delta (incremental):")
            print(f"  Média: {vol_delta.mean():,.0f}")
            print(f"  Mediana: {vol_delta.median():,.0f}")
            print(f"  Min/Max: {vol_delta.min():,.0f} / {vol_delta.max():,.0f}")
            print(f"  Total incremental: {vol_delta.sum():,.0f}")
            
        # Trades delta
        if 'trades_delta' in daily_df.columns:
            trades_delta = daily_df['trades_delta'][daily_df['trades_delta'] > 0]
            if not trades_delta.empty:
                print(f"\nTrades Delta:")
                print(f"  Total trades incrementais: {trades_delta.sum():,}")
                print(f"  Média por callback: {trades_delta.mean():.1f}")
                
    else:
        print("\n=== ANÁLISE DE VOLUME CUMULATIVO ===")
        print("(Versão antiga - sem deltas)")
        
        # Calcular deltas manualmente
        if 'volume' in daily_df.columns:
            daily_df['volume_calc_delta'] = daily_df['volume'].diff().fillna(0)
            
            # Mostrar exemplo
            print("\nExemplo de cálculo de delta:")
            sample = daily_df[['timestamp', 'volume', 'volume_calc_delta']].head(10)
            print(sample)
            
            # Estatísticas
            vol_delta = daily_df['volume_calc_delta'][daily_df['volume_calc_delta'] > 0]
            if not vol_delta.empty:
                print(f"\nVolume Delta Calculado:")
                print(f"  Média: {vol_delta.mean():,.0f}")
                print(f"  Total: {vol_delta.sum():,.0f}")
                
    # Análise de frequência
    if 'timestamp' in daily_df.columns:
        daily_df['timestamp'] = pd.to_datetime(daily_df['timestamp'])
        daily_df['seconds_diff'] = daily_df['timestamp'].diff().dt.total_seconds()
        
        print(f"\nFrequência de callbacks:")
        print(f"  Média: {daily_df['seconds_diff'].mean():.2f} segundos")
        print(f"  Callbacks por minuto: {60/daily_df['seconds_diff'].mean():.1f}")

def analyze_all_volume_data():
    """Analisa todos os arquivos de hoje"""
    data_dir = Path('data/realtime/book') / datetime.now().strftime('%Y%m%d')
    
    if not data_dir.exists():
        print(f"Diretório não encontrado: {data_dir}")
        return
        
    # Buscar arquivos parquet
    parquet_files = list(data_dir.glob('*.parquet'))
    
    print(f"Encontrados {len(parquet_files)} arquivos")
    
    # Analisar cada arquivo
    for pf in sorted(parquet_files)[-3:]:  # Últimos 3 arquivos
        analyze_volume_data(pf)

if __name__ == "__main__":
    print("="*70)
    print("ANÁLISE DE DADOS DE VOLUME")
    print("="*70)
    
    analyze_all_volume_data()