#!/usr/bin/env python3
"""
Script de limpeza para gerenciar modelos antigos
Mantém apenas os N treinamentos mais recentes
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime

def cleanup_old_models(keep_count: int = 3, dry_run: bool = False):
    """
    Remove treinamentos antigos, mantendo apenas os mais recentes
    
    Args:
        keep_count: Número de treinamentos a manter
        dry_run: Se True, apenas mostra o que seria removido
    """
    models_dir = Path('src/training/models')
    
    if not models_dir.exists():
        print("❌ Diretório de modelos não encontrado")
        return False
    
    # Buscar diretórios de treinamento
    training_dirs = [d for d in models_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('training_')]
    
    if len(training_dirs) <= keep_count:
        print(f"✅ Apenas {len(training_dirs)} treinamentos encontrados, nada a limpar")
        return True
    
    # Ordenar por data de modificação (mais recente primeiro)
    training_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Separar os que manter dos que remover
    to_keep = training_dirs[:keep_count]
    to_remove = training_dirs[keep_count:]
    
    print(f"📊 Total de treinamentos: {len(training_dirs)}")
    print(f"🛡️  Manter: {keep_count}")
    print(f"🗑️  Remover: {len(to_remove)}")
    print()
    
    # Mostrar o que será mantido
    print("✅ Treinamentos que serão mantidos:")
    for i, dir_path in enumerate(to_keep, 1):
        size_mb = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file()) / (1024 * 1024)
        mod_time = datetime.fromtimestamp(dir_path.stat().st_mtime)
        print(f"  {i}. {dir_path.name} ({size_mb:.1f} MB) - {mod_time.strftime('%d/%m/%Y %H:%M')}")
    
    print()
    
    # Mostrar o que será removido
    if to_remove:
        print("❌ Treinamentos que serão removidos:")
        total_size = 0
        for i, dir_path in enumerate(to_remove, 1):
            size_mb = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file()) / (1024 * 1024)
            total_size += size_mb
            mod_time = datetime.fromtimestamp(dir_path.stat().st_mtime)
            print(f"  {i}. {dir_path.name} ({size_mb:.1f} MB) - {mod_time.strftime('%d/%m/%Y %H:%M')}")
        
        print(f"\n💾 Espaço que será liberado: {total_size:.1f} MB")
        
        if dry_run:
            print("\n🔍 MODO DRY-RUN: Nenhum arquivo foi removido")
        else:
            print("\n⚠️  Esta operação é irreversível!")
            response = input("Deseja continuar? (s/N): ").lower().strip()
            
            if response == 's':
                print("\n🗑️  Removendo treinamentos antigos...")
                removed_count = 0
                
                for dir_path in to_remove:
                    try:
                        shutil.rmtree(dir_path)
                        print(f"  ✅ Removido: {dir_path.name}")
                        removed_count += 1
                    except Exception as e:
                        print(f"  ❌ Erro removendo {dir_path.name}: {e}")
                
                print(f"\n✅ Limpeza concluída!")
                print(f"📊 Removidos: {removed_count}/{len(to_remove)} treinamentos")
                print(f"💾 Espaço liberado: {total_size:.1f} MB")
            else:
                print("❌ Operação cancelada")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Limpar modelos antigos de treinamento')
    parser.add_argument('--keep', type=int, default=3,
                       help='Número de treinamentos recentes a manter (padrão: 3)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Executar em modo dry-run (apenas mostrar o que seria removido)')
    
    args = parser.parse_args()
    
    print("🧹 Limpeza de Modelos ML Trading v2.0")
    print("=" * 40)
    
    if args.dry_run:
        print("🔍 MODO DRY-RUN ATIVO")
        print()
    
    cleanup_old_models(keep_count=args.keep, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
