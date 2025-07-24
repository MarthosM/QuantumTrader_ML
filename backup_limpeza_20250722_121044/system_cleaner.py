#!/usr/bin/env python3
"""
Script de Análise e Limpeza do Sistema ML Trading v2.0
Identifica arquivos obsoletos, caches, logs e duplicatas
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import json
import argparse
from typing import Dict, List, Tuple
import hashlib

class SystemCleaner:
    """Analisador e limpador de arquivos obsoletos"""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.project_root = Path('.')
        
        # Categorias de arquivos para análise
        self.analysis_results = {
            'test_files': [],
            'doc_files': [],
            'cache_dirs': [],
            'temp_files': [],
            'duplicate_models': [],
            'log_files': [],
            'large_files': [],
            'obsolete_scripts': []
        }
        
        # Padrões de arquivos conhecidamente obsoletos
        self.obsolete_patterns = [
            'test_*.py',
            '*_test.py', 
            '*.tmp',
            '*.bak',
            '*.old',
            '*~',
        ]
        
        # Diretórios de cache conhecidos
        self.cache_patterns = [
            '__pycache__',
            '.pytest_cache',
            'cache',
            '.cache',
            'tmp',
            'temp'
        ]
        
        # Extensões de arquivos temporários
        self.temp_extensions = ['.tmp', '.temp', '.bak', '.old', '.pyc', '.pyo']
        
    def analyze_system(self) -> Dict:
        """Executa análise completa do sistema"""
        print("🔍 Analisando sistema ML Trading v2.0...")
        print("=" * 50)
        
        self._analyze_test_files()
        self._analyze_documentation()
        self._analyze_cache_directories()
        self._analyze_temp_files()
        self._analyze_duplicate_models()
        self._analyze_log_files()
        self._analyze_large_files()
        self._analyze_obsolete_scripts()
        
        return self.analysis_results
    
    def _analyze_test_files(self):
        """Analisa arquivos de teste"""
        print("\n📝 Analisando arquivos de teste...")
        
        # Arquivos test_* no root
        test_files = list(self.project_root.glob('test_*.py'))
        
        for test_file in test_files:
            size_mb = test_file.stat().st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(test_file.stat().st_mtime)
            
            self.analysis_results['test_files'].append({
                'file': str(test_file),
                'size_mb': round(size_mb, 2),
                'modified': mod_time.strftime('%Y-%m-%d %H:%M'),
                'obsolete': self._is_test_obsolete(test_file)
            })
            
        print(f"   📊 Encontrados {len(test_files)} arquivos de teste")
    
    def _analyze_documentation(self):
        """Analisa documentação potencialmente obsoleta"""
        print("\n📚 Analisando documentação...")
        
        # Arquivos .md no root (exceto README.md)
        doc_files = [f for f in self.project_root.glob('*.md') 
                    if f.name != 'README.md']
        
        for doc_file in doc_files:
            size_kb = doc_file.stat().st_size / 1024
            mod_time = datetime.fromtimestamp(doc_file.stat().st_mtime)
            
            # Considerar obsoleto se > 30 dias sem modificação
            days_old = (datetime.now() - mod_time).days
            
            self.analysis_results['doc_files'].append({
                'file': str(doc_file),
                'size_kb': round(size_kb, 1),
                'modified': mod_time.strftime('%Y-%m-%d'),
                'days_old': days_old,
                'obsolete': days_old > 30 or 'ANALISE_' in doc_file.name
            })
            
        print(f"   📊 Encontrados {len(doc_files)} documentos")
    
    def _analyze_cache_directories(self):
        """Analisa diretórios de cache"""
        print("\n🗂️  Analisando caches...")
        
        for pattern in self.cache_patterns:
            cache_dirs = list(self.project_root.rglob(pattern))
            
            for cache_dir in cache_dirs:
                if cache_dir.is_dir():
                    size_mb = self._get_directory_size(cache_dir) / (1024 * 1024)
                    
                    self.analysis_results['cache_dirs'].append({
                        'directory': str(cache_dir),
                        'size_mb': round(size_mb, 2),
                        'removable': True
                    })
        
        total_cache = sum(c['size_mb'] for c in self.analysis_results['cache_dirs'])
        print(f"   📊 Caches: {len(self.analysis_results['cache_dirs'])} diretórios, {total_cache:.1f} MB")
    
    def _analyze_temp_files(self):
        """Analisa arquivos temporários"""
        print("\n🗑️  Analisando arquivos temporários...")
        
        for ext in self.temp_extensions:
            temp_files = list(self.project_root.rglob(f'*{ext}'))
            
            for temp_file in temp_files:
                if temp_file.is_file():
                    size_mb = temp_file.stat().st_size / (1024 * 1024)
                    
                    self.analysis_results['temp_files'].append({
                        'file': str(temp_file),
                        'size_mb': round(size_mb, 2),
                        'removable': True
                    })
        
        print(f"   📊 Temporários: {len(self.analysis_results['temp_files'])} arquivos")
    
    def _analyze_duplicate_models(self):
        """Analisa modelos duplicados"""
        print("\n🤖 Analisando modelos duplicados...")
        
        model_files = list(self.project_root.rglob('*.pkl')) + list(self.project_root.rglob('*.h5'))
        
        # Agrupar por nome base
        model_groups = {}
        for model in model_files:
            base_name = model.name
            if base_name not in model_groups:
                model_groups[base_name] = []
            model_groups[base_name].append(model)
        
        # Encontrar duplicatas
        for base_name, models in model_groups.items():
            if len(models) > 1:
                # Ordenar por data de modificação (mais recente primeiro)
                models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                for i, model in enumerate(models):
                    size_mb = model.stat().st_size / (1024 * 1024)
                    mod_time = datetime.fromtimestamp(model.stat().st_mtime)
                    
                    self.analysis_results['duplicate_models'].append({
                        'file': str(model),
                        'base_name': base_name,
                        'size_mb': round(size_mb, 2),
                        'modified': mod_time.strftime('%Y-%m-%d %H:%M'),
                        'keep': i == 0,  # Manter apenas o mais recente
                        'duplicate_group': len(models)
                    })
        
        duplicates = sum(1 for m in self.analysis_results['duplicate_models'] if not m['keep'])
        print(f"   📊 Duplicatas: {duplicates} modelos duplicados encontrados")
    
    def _analyze_log_files(self):
        """Analisa arquivos de log"""
        print("\n📋 Analisando logs...")
        
        log_files = list(self.project_root.rglob('*.log'))
        
        for log_file in log_files:
            size_mb = log_file.stat().st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(log_file.stat().st_mtime)
            days_old = (datetime.now() - mod_time).days
            
            self.analysis_results['log_files'].append({
                'file': str(log_file),
                'size_mb': round(size_mb, 2),
                'modified': mod_time.strftime('%Y-%m-%d'),
                'days_old': days_old,
                'removable': days_old > 7 or size_mb > 10  # Logs > 7 dias ou > 10MB
            })
        
        print(f"   📊 Logs: {len(log_files)} arquivos encontrados")
    
    def _analyze_large_files(self):
        """Analisa arquivos grandes que podem ser obsoletos"""
        print("\n📦 Analisando arquivos grandes...")
        
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                
                if size_mb > 5:  # Arquivos > 5MB
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    days_old = (datetime.now() - mod_time).days
                    
                    self.analysis_results['large_files'].append({
                        'file': str(file_path),
                        'size_mb': round(size_mb, 2),
                        'modified': mod_time.strftime('%Y-%m-%d'),
                        'days_old': days_old,
                        'extension': file_path.suffix,
                        'review_needed': size_mb > 10 or days_old > 30
                    })
        
        # Ordenar por tamanho
        self.analysis_results['large_files'].sort(key=lambda x: x['size_mb'], reverse=True)
        large_count = len([f for f in self.analysis_results['large_files'] if f['review_needed']])
        print(f"   📊 Arquivos grandes: {large_count} precisam revisão")
    
    def _analyze_obsolete_scripts(self):
        """Analisa scripts possivelmente obsoletos"""
        print("\n⚙️  Analisando scripts obsoletos...")
        
        # Scripts de teste e desenvolvimento no root
        root_scripts = [f for f in self.project_root.glob('*.py') 
                       if not f.name in ['run_training.py', 'migrate_models.py', 'cleanup_old_models.py']]
        
        for script in root_scripts:
            size_kb = script.stat().st_size / 1024
            mod_time = datetime.fromtimestamp(script.stat().st_mtime)
            days_old = (datetime.now() - mod_time).days
            
            # Considerar obsoleto se nome sugere teste/desenvolvimento
            is_obsolete = any(keyword in script.name.lower() 
                            for keyword in ['test', 'debug', 'temp', 'old', 'backup'])
            
            self.analysis_results['obsolete_scripts'].append({
                'file': str(script),
                'size_kb': round(size_kb, 1),
                'modified': mod_time.strftime('%Y-%m-%d'),
                'days_old': days_old,
                'obsolete': is_obsolete or days_old > 60
            })
        
        obsolete_count = sum(1 for s in self.analysis_results['obsolete_scripts'] if s['obsolete'])
        print(f"   📊 Scripts: {obsolete_count} possivelmente obsoletos")
    
    def _is_test_obsolete(self, test_file: Path) -> bool:
        """Verifica se arquivo de teste é obsoleto"""
        mod_time = datetime.fromtimestamp(test_file.stat().st_mtime)
        days_old = (datetime.now() - mod_time).days
        
        # Considerações: idade, nome, tamanho
        obsolete_keywords = ['old', 'backup', 'temp', 'debug']
        has_obsolete_name = any(keyword in test_file.name.lower() for keyword in obsolete_keywords)
        
        return days_old > 30 or has_obsolete_name
    
    def _get_directory_size(self, directory: Path) -> int:
        """Calcula tamanho total de um diretório"""
        total = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total += file_path.stat().st_size
        except (PermissionError, OSError):
            pass
        return total
    
    def generate_report(self) -> str:
        """Gera relatório detalhado"""
        report = []
        report.append("🧹 RELATÓRIO DE LIMPEZA - ML Trading v2.0")
        report.append("=" * 50)
        report.append(f"📅 Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Resumo por categoria
        categories = [
            ('📝 Arquivos de Teste', 'test_files'),
            ('📚 Documentação', 'doc_files'),
            ('🗂️  Caches', 'cache_dirs'),
            ('🗑️  Temporários', 'temp_files'),
            ('🤖 Modelos Duplicados', 'duplicate_models'),
            ('📋 Logs', 'log_files'),
            ('📦 Arquivos Grandes', 'large_files'),
            ('⚙️  Scripts Obsoletos', 'obsolete_scripts')
        ]
        
        total_files = 0
        total_size = 0
        removable_files = 0
        removable_size = 0
        
        for category_name, category_key in categories:
            items = self.analysis_results[category_key]
            if not items:
                continue
                
            report.append(f"\n{category_name}")
            report.append("-" * 30)
            
            category_size = 0
            category_removable = 0
            category_removable_size = 0
            
            for item in items:
                total_files += 1
                
                # Calcular tamanho
                if 'size_mb' in item:
                    size = item['size_mb']
                    unit = "MB"
                    category_size += size
                    total_size += size
                elif 'size_kb' in item:
                    size = item['size_kb'] / 1024
                    unit = "MB"
                    category_size += size
                    total_size += size
                else:
                    size = 0
                    unit = ""
                
                # Verificar se removível
                is_removable = item.get('removable', False) or \
                              item.get('obsolete', False) or \
                              (not item.get('keep', True))
                
                if is_removable:
                    removable_files += 1
                    removable_size += size
                    category_removable += 1
                    category_removable_size += size
                    status = "🗑️  REMOVER"
                elif item.get('review_needed', False):
                    status = "⚠️  REVISAR"
                else:
                    status = "✅ MANTER"
                
                # Formatear linha do relatório
                file_name = Path(item.get('file', item.get('directory', ''))).name
                if len(file_name) > 40:
                    file_name = file_name[:37] + "..."
                
                report.append(f"  {status} {file_name:<40} {size:>8.2f} {unit}")
            
            # Resumo da categoria
            report.append(f"  Total: {len(items)} itens, {category_size:.1f} MB")
            if category_removable > 0:
                report.append(f"  Removíveis: {category_removable} itens, {category_removable_size:.1f} MB")
        
        # Resumo geral
        report.append(f"\n{'='*50}")
        report.append("📊 RESUMO GERAL")
        report.append(f"Total analisado: {total_files} itens, {total_size:.1f} MB")
        report.append(f"Pode ser removido: {removable_files} itens, {removable_size:.1f} MB")
        report.append(f"Economia potencial: {removable_size/total_size*100:.1f}% do espaço")
        
        return "\n".join(report)
    
    def execute_cleanup(self) -> Dict:
        """Executa limpeza baseada na análise"""
        if self.dry_run:
            print("\n🔍 MODO DRY-RUN - Nenhum arquivo será removido")
            return {'status': 'dry_run', 'would_remove': self._count_removable()}
        
        print("\n🧹 Executando limpeza...")
        
        removed_count = 0
        removed_size = 0
        errors = []
        
        # Remover caches
        for cache_info in self.analysis_results['cache_dirs']:
            if cache_info['removable']:
                try:
                    cache_path = Path(cache_info['directory'])
                    if cache_path.exists():
                        shutil.rmtree(cache_path)
                        removed_count += 1
                        removed_size += cache_info['size_mb']
                        print(f"  ✅ Removido cache: {cache_path.name}")
                except Exception as e:
                    errors.append(f"Erro removendo {cache_path}: {e}")
        
        # Remover arquivos temporários
        for temp_info in self.analysis_results['temp_files']:
            if temp_info['removable']:
                try:
                    temp_path = Path(temp_info['file'])
                    if temp_path.exists():
                        temp_path.unlink()
                        removed_count += 1
                        removed_size += temp_info['size_mb']
                        print(f"  ✅ Removido temporário: {temp_path.name}")
                except Exception as e:
                    errors.append(f"Erro removendo {temp_path}: {e}")
        
        # Remover duplicatas de modelos
        for model_info in self.analysis_results['duplicate_models']:
            if not model_info['keep']:
                try:
                    model_path = Path(model_info['file'])
                    if model_path.exists():
                        model_path.unlink()
                        removed_count += 1
                        removed_size += model_info['size_mb']
                        print(f"  ✅ Removido duplicata: {model_path.name}")
                except Exception as e:
                    errors.append(f"Erro removendo {model_path}: {e}")
        
        # Remover logs antigos
        for log_info in self.analysis_results['log_files']:
            if log_info['removable']:
                try:
                    log_path = Path(log_info['file'])
                    if log_path.exists():
                        log_path.unlink()
                        removed_count += 1
                        removed_size += log_info['size_mb']
                        print(f"  ✅ Removido log: {log_path.name}")
                except Exception as e:
                    errors.append(f"Erro removendo {log_path}: {e}")
        
        return {
            'status': 'completed',
            'removed_count': removed_count,
            'removed_size_mb': round(removed_size, 2),
            'errors': errors
        }
    
    def _count_removable(self) -> Dict:
        """Conta itens removíveis para dry-run"""
        count = 0
        size = 0
        
        for category in self.analysis_results.values():
            for item in category:
                if item.get('removable', False) or item.get('obsolete', False) or not item.get('keep', True):
                    count += 1
                    size += item.get('size_mb', item.get('size_kb', 0) / 1024)
        
        return {'count': count, 'size_mb': round(size, 2)}

def main():
    parser = argparse.ArgumentParser(description='Análise e limpeza do sistema ML Trading')
    parser.add_argument('--execute', action='store_true', 
                       help='Executar limpeza (padrão: apenas análise)')
    parser.add_argument('--report', type=str, default='cleanup_report.txt',
                       help='Arquivo para salvar relatório')
    
    args = parser.parse_args()
    
    # Executar análise
    cleaner = SystemCleaner(dry_run=not args.execute)
    cleaner.analyze_system()
    
    # Gerar relatório
    report = cleaner.generate_report()
    print(report)
    
    # Salvar relatório
    with open(args.report, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Relatório salvo em: {args.report}")
    
    # Executar limpeza se solicitado
    if args.execute:
        print("\n⚠️  ATENÇÃO: Esta operação removerá arquivos permanentemente!")
        response = input("Deseja continuar? (s/N): ").lower().strip()
        if response == 's':
            result = cleaner.execute_cleanup()
            print(f"\n✅ Limpeza concluída!")
            print(f"📊 Removidos: {result['removed_count']} itens")
            print(f"💾 Espaço liberado: {result['removed_size_mb']} MB")
            if result['errors']:
                print(f"⚠️  Erros: {len(result['errors'])}")
        else:
            print("❌ Limpeza cancelada")

if __name__ == "__main__":
    main()
