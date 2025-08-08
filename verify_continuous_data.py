"""
Verifica continuidade dos dados após limpeza
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def verify_data_continuity():
    """Verifica gaps nos dados consolidados"""
    
    print("=" * 70)
    print("VERIFICAÇÃO DE CONTINUIDADE DOS DADOS")
    print("=" * 70)
    
    # Carregar arquivo consolidado
    file_path = Path('data/realtime/book/20250806/training/consolidated_continuous_20250806.parquet')
    
    if not file_path.exists():
        print(f"Arquivo não encontrado: {file_path}")
        return
        
    df = pd.read_parquet(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    print(f"\nDados carregados: {len(df):,} registros")
    print(f"Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
    
    # Analisar gaps por timestamp
    print("\n=== ANÁLISE DE GAPS ===")
    
    # Criar série temporal com todos os timestamps
    timestamps = df['timestamp'].drop_duplicates().sort_values()
    time_diffs = timestamps.diff()
    
    # Identificar gaps maiores que 1 minuto
    gaps = time_diffs[time_diffs > pd.Timedelta(minutes=1)]
    
    if not gaps.empty:
        print(f"\n{len(gaps)} gaps encontrados (> 1 minuto):")
        
        for idx, gap_duration in gaps.items():
            gap_idx = timestamps.index.get_loc(idx)
            gap_start = timestamps.iloc[gap_idx - 1]
            gap_end = timestamps.iloc[gap_idx]
            
            print(f"\nGAP {gap_idx}:")
            print(f"  Início: {gap_start}")
            print(f"  Fim: {gap_end}")
            print(f"  Duração: {gap_duration}")
            
            # Verificar se é antes ou depois das 10:30
            if gap_end.time() < pd.Timestamp('10:30').time():
                print("  [ANTES DAS 10:30 - DEVERIA TER SIDO REMOVIDO!]")
            else:
                print("  [DEPOIS DAS 10:30 - OK]")
    else:
        print("Nenhum gap significativo encontrado!")
        
    # Analisar distribuição temporal
    print("\n=== DISTRIBUIÇÃO TEMPORAL ===")
    
    # Agrupar por hora
    df['hour'] = df['timestamp'].dt.hour
    hourly_dist = df.groupby('hour').size()
    
    print("\nRegistros por hora:")
    for hour, count in hourly_dist.items():
        print(f"  {hour:02d}:00 - {count:,} registros")
        
    # Verificar se há dados antes das 10:30
    before_cutoff = df[df['timestamp'] < pd.Timestamp('2025-08-06 10:30:00')]
    if not before_cutoff.empty:
        print(f"\n[PROBLEMA] {len(before_cutoff):,} registros antes das 10:30!")
        print("Período problemático:")
        print(f"  De: {before_cutoff['timestamp'].min()}")
        print(f"  Até: {before_cutoff['timestamp'].max()}")
        
        # Sugerir limpeza adicional
        print("\n=== SUGESTÃO DE LIMPEZA ===")
        
        # Filtrar apenas dados após 10:30
        after_cutoff = df[df['timestamp'] >= pd.Timestamp('2025-08-06 10:30:00')]
        
        if not after_cutoff.empty:
            clean_file = Path('data/realtime/book/20250806/training/consolidated_clean_after_1030.parquet')
            after_cutoff.to_parquet(clean_file, index=False)
            
            print(f"\n[OK] Dados limpos salvos em:")
            print(f"  {clean_file}")
            print(f"  Registros: {len(after_cutoff):,}")
            print(f"  Período: {after_cutoff['timestamp'].min()} até {after_cutoff['timestamp'].max()}")
            
            # Verificar continuidade nos dados limpos
            clean_timestamps = after_cutoff['timestamp'].drop_duplicates().sort_values()
            clean_gaps = clean_timestamps.diff()
            max_clean_gap = clean_gaps.max()
            
            print(f"\nMaior gap nos dados limpos: {max_clean_gap}")
            
            if max_clean_gap < pd.Timedelta(minutes=5):
                print("[OK] Dados limpos são contínuos!")
            else:
                print("[AVISO] Ainda há gaps nos dados limpos")
    else:
        print("\n[OK] Nenhum dado antes das 10:30")
        
    # Análise por tipo de dado
    if 'type' in df.columns:
        print("\n=== DISTRIBUIÇÃO POR TIPO ===")
        for dtype, count in df['type'].value_counts().items():
            pct = count / len(df) * 100
            print(f"  {dtype}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    verify_data_continuity()