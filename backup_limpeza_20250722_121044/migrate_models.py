#!/usr/bin/env python3
"""
Script para migrar modelos salvos de training_results/ para src/training/models/
Execute este script do diretório raiz do projeto
"""

import os
import shutil
from pathlib import Path
import json

def migrate_models():
    """Migra modelos de training_results para src/training/models"""
    
    # Diretórios
    old_results_dir = Path('training_results')
    new_models_dir = Path('src/training/models')
    
    print("🔄 Migrando modelos para nova estrutura...")
    print(f"📂 De: {old_results_dir}")
    print(f"📂 Para: {new_models_dir}")
    
    # Verificar se diretório antigo existe
    if not old_results_dir.exists():
        print("❌ Diretório training_results/ não encontrado")
        return False
    
    # Criar diretório novo se não existir
    new_models_dir.mkdir(parents=True, exist_ok=True)
    
    # Listar diretórios de treinamento
    training_dirs = [d for d in old_results_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('training_')]
    
    if not training_dirs:
        print("❌ Nenhum diretório de treinamento encontrado")
        return False
    
    migrated_count = 0
    
    for training_dir in training_dirs:
        print(f"\n📦 Processando: {training_dir.name}")
        
        # Destino
        dest_dir = new_models_dir / training_dir.name
        
        if dest_dir.exists():
            print(f"⚠️  Destino já existe: {dest_dir}")
            response = input("Deseja sobrescrever? (s/N): ").lower().strip()
            if response != 's':
                print("⏭️  Pulando...")
                continue
            else:
                shutil.rmtree(dest_dir)
        
        try:
            # Copiar diretório inteiro
            shutil.copytree(training_dir, dest_dir)
            print(f"✅ Copiado: {training_dir.name}")
            
            # Verificar se contém ensemble
            ensemble_dirs = list(dest_dir.glob('ensemble_*'))
            if ensemble_dirs:
                print(f"   📊 Ensemble encontrado: {len(ensemble_dirs)} versões")
                
                # Verificar modelos
                for ensemble_dir in ensemble_dirs:
                    model_files = list(ensemble_dir.glob('*.pkl'))
                    h5_files = list(ensemble_dir.glob('*.h5'))
                    print(f"   🤖 Modelos: {len(model_files)} .pkl, {len(h5_files)} .h5")
                    
                    # Verificar metadata
                    metadata_file = ensemble_dir / 'ensemble_metadata.json'
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            print(f"   📈 Métricas disponíveis: {list(metadata.get('ensemble_metrics', {}).keys())}")
                        except Exception as e:
                            print(f"   ⚠️  Erro lendo metadata: {e}")
            
            migrated_count += 1
            
        except Exception as e:
            print(f"❌ Erro copiando {training_dir.name}: {e}")
    
    print(f"\n✅ Migração concluída!")
    print(f"📊 Diretórios migrados: {migrated_count}/{len(training_dirs)}")
    
    # Perguntar se deve manter ou remover o diretório antigo
    if migrated_count > 0:
        print(f"\n🗑️  Diretório antigo: {old_results_dir}")
        response = input("Deseja manter o diretório antigo? (S/n): ").lower().strip()
        if response == 'n':
            try:
                shutil.rmtree(old_results_dir)
                print("✅ Diretório antigo removido")
            except Exception as e:
                print(f"❌ Erro removendo diretório antigo: {e}")
        else:
            print("📦 Diretório antigo mantido")
    
    return True

def show_current_models():
    """Mostra modelos disponíveis após migração"""
    models_dir = Path('src/training/models')
    
    if not models_dir.exists():
        print("❌ Nenhum modelo encontrado")
        return
    
    print(f"\n📊 Modelos disponíveis em {models_dir}:")
    
    training_dirs = [d for d in models_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('training_')]
    
    for training_dir in sorted(training_dirs):
        print(f"\n📅 {training_dir.name}")
        
        # Buscar ensemble
        ensemble_dirs = list(training_dir.glob('ensemble_*'))
        for ensemble_dir in ensemble_dirs:
            print(f"  📊 {ensemble_dir.name}")
            
            # Listar modelos
            model_files = list(ensemble_dir.glob('*.pkl'))
            h5_files = list(ensemble_dir.glob('*.h5'))
            
            for model_file in model_files:
                print(f"    🤖 {model_file.name}")
            for model_file in h5_files:
                print(f"    🧠 {model_file.name}")
            
            # Mostrar metadata se disponível
            metadata_file = ensemble_dir / 'ensemble_metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    if 'ensemble_metrics' in metadata:
                        metrics = metadata['ensemble_metrics']
                        print(f"    📈 Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                        print(f"    📈 F1-Score: {metrics.get('f1', 'N/A'):.4f}")
                        
                except Exception as e:
                    print(f"    ⚠️  Erro lendo metadata: {e}")

if __name__ == "__main__":
    print("🚀 Migrador de Modelos ML Trading v2.0")
    print("=" * 50)
    
    if migrate_models():
        show_current_models()
    else:
        print("❌ Migração falhou")
