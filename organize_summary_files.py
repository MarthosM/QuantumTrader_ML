"""
Organiza arquivos summary JSON para ficarem junto dos parquets correspondentes
"""

import shutil
from pathlib import Path

def organize_summary_files(date='20250805'):
    """Move arquivos summary para as pastas corretas"""
    
    data_dir = Path(f'data/realtime/book/{date}')
    
    print("=" * 70)
    print("ORGANIZANDO ARQUIVOS SUMMARY")
    print("=" * 70)
    
    # Buscar todos os summary files
    summary_files = list(data_dir.rglob('summary_*.json'))
    
    print(f"\nEncontrados {len(summary_files)} arquivos summary")
    
    moved = 0
    errors = 0
    
    for summary_file in summary_files:
        try:
            # Extrair nome base do arquivo
            # summary_continuous_20250805_110825.json -> wdo_continuous_20250805_110825
            filename = summary_file.name
            
            if filename.startswith('summary_continuous_'):
                # Arquivo continuous
                base_name = filename.replace('summary_continuous_', 'wdo_continuous_')
                parquet_name = base_name.replace('.json', '.parquet')
                
                # Procurar o parquet correspondente
                parquet_file = None
                
                # Verificar na pasta continuous
                continuous_dir = data_dir / 'continuous'
                if continuous_dir.exists():
                    possible_parquet = continuous_dir / parquet_name
                    if possible_parquet.exists():
                        parquet_file = possible_parquet
                        
                if parquet_file and summary_file.parent != parquet_file.parent:
                    # Mover summary para mesma pasta do parquet
                    dest = parquet_file.parent / summary_file.name
                    shutil.move(str(summary_file), str(dest))
                    print(f"  Movido: {summary_file.name} -> {parquet_file.parent.name}/")
                    moved += 1
                    
            elif filename.startswith('summary_wdo_'):
                # Outros tipos (final, working, etc)
                # Já devem estar no backup junto com seus parquets
                pass
                
        except Exception as e:
            print(f"  Erro com {summary_file.name}: {e}")
            errors += 1
            
    print(f"\n{moved} arquivos movidos, {errors} erros")
    
    # Verificar consistência
    print("\n=== VERIFICANDO CONSISTÊNCIA ===")
    
    for subdir in ['continuous', 'backup_non_continuous']:
        full_path = data_dir / subdir
        if full_path.exists():
            parquets = list(full_path.glob('*.parquet'))
            summaries = list(full_path.glob('summary_*.json'))
            
            print(f"\n{subdir}:")
            print(f"  Parquets: {len(parquets)}")
            print(f"  Summaries: {len(summaries)}")
            
            # Verificar órfãos
            orphan_summaries = []
            for summary in summaries:
                # Encontrar parquet correspondente
                if 'continuous' in summary.name:
                    expected_parquet = summary.name.replace('summary_continuous_', 'wdo_continuous_').replace('.json', '.parquet')
                else:
                    # Para outros tipos, extrair padrão
                    base = summary.stem.replace('summary_', '')
                    expected_parquet = f"{base}.parquet"
                    
                if not (full_path / expected_parquet).exists():
                    orphan_summaries.append(summary)
                    
            if orphan_summaries:
                print(f"  [!] {len(orphan_summaries)} summaries órfãos encontrados")

if __name__ == "__main__":
    organize_summary_files()