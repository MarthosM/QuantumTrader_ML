#!/usr/bin/env python3
"""
Script de Limpeza Segura - ML Trading v2.0
Remove apenas itens seguros: caches, duplicatas óbvias
"""

import shutil
from pathlib import Path
from datetime import datetime

class SafeCleaner:
    """Limpeza segura focada nos itens mais óbvios"""
    
    def __init__(self):
        self.project_root = Path('.')
        self.cleaned_items = []
        self.total_saved_mb = 0
        
    def safe_cleanup(self):
        """Executa limpeza apenas dos itens seguros"""
        print("🧹 Limpeza Segura - ML Trading v2.0")
        print("=" * 50)
        print("Removendo apenas: caches Python, modelos duplicados e docs obsoletos")
        print()
        
        # 1. Limpar caches Python (__pycache__)
        self._clean_python_caches()
        
        # 2. Remover modelos duplicados (manter apenas os mais recentes)
        self._clean_duplicate_models()
        
        # 3. Remover documentos de análise temporários
        self._clean_analysis_docs()
        
        # 4. Limpar pytest cache
        self._clean_pytest_cache()
        
        return self._generate_summary()
    
    def _clean_python_caches(self):
        """Remove diretórios __pycache__"""
        print("🗂️  Removendo caches Python...")
        
        cache_dirs = list(self.project_root.rglob('__pycache__'))
        removed_count = 0
        
        for cache_dir in cache_dirs:
            if cache_dir.is_dir():
                try:
                    size_mb = self._get_directory_size(cache_dir) / (1024 * 1024)
                    shutil.rmtree(cache_dir)
                    self.cleaned_items.append(f"Cache: {cache_dir.relative_to(self.project_root)}")
                    self.total_saved_mb += size_mb
                    removed_count += 1
                except Exception as e:
                    print(f"  ⚠️  Erro removendo {cache_dir}: {e}")
        
        print(f"  ✅ {removed_count} caches Python removidos")
    
    def _clean_duplicate_models(self):
        """Remove modelos duplicados, mantendo os mais recentes"""
        print("🤖 Removendo modelos duplicados...")
        
        # Localizar todos os arquivos .pkl
        model_files = list(self.project_root.rglob('*.pkl'))
        
        # Agrupar por nome
        model_groups = {}
        for model in model_files:
            base_name = model.name
            if base_name not in model_groups:
                model_groups[base_name] = []
            model_groups[base_name].append(model)
        
        removed_count = 0
        for base_name, models in model_groups.items():
            if len(models) > 1:
                # Manter apenas o mais recente por diretório de treinamento
                training_groups = {}
                for model in models:
                    # Agrupar por diretório de treinamento
                    training_dir = None
                    for parent in model.parents:
                        if 'training_' in parent.name:
                            training_dir = parent.name
                            break
                    
                    if training_dir:
                        if training_dir not in training_groups:
                            training_groups[training_dir] = []
                        training_groups[training_dir].append(model)
                
                # Para cada grupo de treinamento, manter apenas o mais recente
                for training_dir, group_models in training_groups.items():
                    if len(group_models) > 1:
                        # Ordenar por data de modificação
                        group_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                        
                        # Remover todos exceto o primeiro (mais recente)
                        for model in group_models[1:]:
                            try:
                                size_mb = model.stat().st_size / (1024 * 1024)
                                model.unlink()
                                self.cleaned_items.append(f"Modelo duplicado: {model.relative_to(self.project_root)}")
                                self.total_saved_mb += size_mb
                                removed_count += 1
                            except Exception as e:
                                print(f"  ⚠️  Erro removendo {model}: {e}")
        
        print(f"  ✅ {removed_count} modelos duplicados removidos")
    
    def _clean_analysis_docs(self):
        """Remove documentos de análise temporários"""
        print("📚 Removendo documentação temporária...")
        
        # Documentos claramente de análise/temporários
        temp_docs = [
            'ANALISE_FINAL_CORRECOES.md',
            'ANALISE_TRADING_SYSTEM.md'
        ]
        
        removed_count = 0
        for doc_name in temp_docs:
            doc_path = self.project_root / doc_name
            if doc_path.exists():
                try:
                    size_mb = doc_path.stat().st_size / (1024 * 1024)
                    doc_path.unlink()
                    self.cleaned_items.append(f"Doc temporário: {doc_name}")
                    self.total_saved_mb += size_mb
                    removed_count += 1
                except Exception as e:
                    print(f"  ⚠️  Erro removendo {doc_path}: {e}")
        
        print(f"  ✅ {removed_count} documentos temporários removidos")
    
    def _clean_pytest_cache(self):
        """Remove cache do pytest"""
        print("🧪 Removendo cache do pytest...")
        
        pytest_cache = self.project_root / '.pytest_cache'
        if pytest_cache.exists() and pytest_cache.is_dir():
            try:
                size_mb = self._get_directory_size(pytest_cache) / (1024 * 1024)
                shutil.rmtree(pytest_cache)
                self.cleaned_items.append("Cache pytest")
                self.total_saved_mb += size_mb
                print("  ✅ Cache pytest removido")
            except Exception as e:
                print(f"  ⚠️  Erro removendo cache pytest: {e}")
        else:
            print("  ℹ️  Cache pytest não encontrado")
    
    def _get_directory_size(self, directory):
        """Calcula tamanho total de um diretório"""
        total = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total += file_path.stat().st_size
        except (PermissionError, OSError):
            pass
        return total
    
    def _generate_summary(self):
        """Gera resumo da limpeza"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'items_cleaned': len(self.cleaned_items),
            'space_saved_mb': round(self.total_saved_mb, 2),
            'cleaned_items': self.cleaned_items
        }
        
        print(f"\n{'='*50}")
        print("📊 RESUMO DA LIMPEZA SEGURA")
        print(f"Itens removidos: {summary['items_cleaned']}")
        print(f"Espaço liberado: {summary['space_saved_mb']:.1f} MB")
        print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.cleaned_items:
            print("\n📋 Itens removidos:")
            for item in self.cleaned_items[:10]:  # Mostrar primeiros 10
                print(f"  • {item}")
            if len(self.cleaned_items) > 10:
                print(f"  • ... e mais {len(self.cleaned_items) - 10} itens")
        
        return summary

def main():
    """Executa limpeza segura"""
    print("⚠️  LIMPEZA SEGURA")
    print("Esta limpeza remove apenas:")
    print("  • Caches Python (__pycache__)")
    print("  • Modelos .pkl duplicados (mantém mais recentes)")
    print("  • Documentos de análise temporários")
    print("  • Cache do pytest")
    print()
    
    response = input("Deseja continuar? (s/N): ").lower().strip()
    
    if response == 's':
        cleaner = SafeCleaner()
        result = cleaner.safe_cleanup()
        
        # Salvar relatório
        with open('safe_cleanup_report.txt', 'w', encoding='utf-8') as f:
            f.write(f"Relatório de Limpeza Segura - {result['timestamp']}\n")
            f.write(f"Itens removidos: {result['items_cleaned']}\n")
            f.write(f"Espaço liberado: {result['space_saved_mb']} MB\n\n")
            f.write("Itens removidos:\n")
            for item in result['cleaned_items']:
                f.write(f"  • {item}\n")
        
        print(f"\n📄 Relatório salvo em: safe_cleanup_report.txt")
    else:
        print("❌ Limpeza cancelada")

if __name__ == "__main__":
    main()
