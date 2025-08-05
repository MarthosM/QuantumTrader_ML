"""
Script de limpeza do sistema v2
Remove arquivos não utilizados e organiza estrutura
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json

class SystemCleaner:
    def __init__(self):
        self.cleanup_report = {
            'timestamp': datetime.now().isoformat(),
            'files_to_delete': [],
            'files_to_keep': [],
            'directories_to_remove': [],
            'git_deletions': [],
            'size_saved': 0
        }
        
    def analyze_git_deletions(self):
        """Analisa arquivos já deletados no git"""
        import subprocess
        
        print("\n=== ANALISANDO ARQUIVOS DELETADOS NO GIT ===")
        
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        
        for line in result.stdout.strip().split('\n'):
            if line.startswith(' D '):
                file_path = line[3:]
                self.cleanup_report['git_deletions'].append(file_path)
                
        print(f"Encontrados {len(self.cleanup_report['git_deletions'])} arquivos deletados")
        
    def identify_obsolete_files(self):
        """Identifica arquivos obsoletos para deletar"""
        
        print("\n=== IDENTIFICANDO ARQUIVOS OBSOLETOS ===")
        
        # Padrões de arquivos para deletar
        patterns_to_delete = [
            # Scripts de teste antigos
            'test_*.py',
            '*_test.py',
            'test_old_*.py',
            
            # Scripts de análise temporários
            'analyze_*.py',
            'debug_*.py',
            'diagnose_*.py',
            'check_*.py',
            
            # Backups antigos
            'backup_*.py',
            '*_backup.py',
            '*_old.py',
            
            # Scripts de migração/setup únicos
            'create_*.py',
            'prepare_*.py',
            'migrate_*.py',
            'setup_*.py',
            
            # Scripts temporários
            'quick_*.py',
            'temp_*.py',
            'tmp_*.py',
        ]
        
        # Arquivos específicos para manter
        files_to_keep = [
            'book_collector.py',
            'book_collector_continuous.py',
            'start_continuous_collection.py',
            'start_book_collector_simple.py',
            'analyze_volume_data.py',
            'cleanup_system.py',
            'cleanup_system_v2.py',
            'main.py',
            'run_backtest.py'
        ]
        
        # Buscar arquivos
        for pattern in patterns_to_delete:
            for file in Path('.').glob(pattern):
                if file.is_file() and file.name not in files_to_keep:
                    size = file.stat().st_size
                    self.cleanup_report['files_to_delete'].append({
                        'path': str(file),
                        'size': size,
                        'pattern': pattern
                    })
                    self.cleanup_report['size_saved'] += size
                    
    def identify_empty_directories(self):
        """Identifica diretórios vazios"""
        
        print("\n=== IDENTIFICANDO DIRETÓRIOS VAZIOS ===")
        
        for root, dirs, files in os.walk('.'):
            # Pular diretórios git e venv
            if '.git' in root or '.venv' in root or '__pycache__' in root:
                continue
                
            if not dirs and not files:
                self.cleanup_report['directories_to_remove'].append(root)
                
    def show_summary(self):
        """Mostra resumo do que será limpo"""
        
        print("\n" + "="*70)
        print("RESUMO DA LIMPEZA")
        print("="*70)
        
        print(f"\nArquivos já deletados (git): {len(self.cleanup_report['git_deletions'])}")
        print(f"Arquivos para deletar: {len(self.cleanup_report['files_to_delete'])}")
        print(f"Diretórios vazios: {len(self.cleanup_report['directories_to_remove'])}")
        print(f"Espaço a ser liberado: {self.cleanup_report['size_saved']/1024/1024:.2f} MB")
        
        if self.cleanup_report['files_to_delete']:
            print("\n=== ARQUIVOS PARA DELETAR ===")
            for item in sorted(self.cleanup_report['files_to_delete'][:20], key=lambda x: x['path']):
                print(f"  {item['path']} ({item['size']/1024:.1f} KB)")
            if len(self.cleanup_report['files_to_delete']) > 20:
                print(f"  ... e mais {len(self.cleanup_report['files_to_delete'])-20} arquivos")
                
        if self.cleanup_report['git_deletions']:
            print("\n=== PRINCIPAIS DELEÇÕES GIT ===")
            # Agrupar por tipo
            backups = [f for f in self.cleanup_report['git_deletions'] if 'backup' in f]
            tests = [f for f in self.cleanup_report['git_deletions'] if 'test' in f]
            scripts = [f for f in self.cleanup_report['git_deletions'] if f.endswith('.py') and f not in backups and f not in tests]
            
            print(f"  Backups: {len(backups)} arquivos")
            print(f"  Testes: {len(tests)} arquivos")
            print(f"  Scripts: {len(scripts)} arquivos")
            
    def save_report(self):
        """Salva relatório da limpeza"""
        
        report_file = f'cleanup_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(self.cleanup_report, f, indent=2)
            
        print(f"\nRelatório salvo em: {report_file}")
        
    def execute_cleanup(self):
        """Executa a limpeza (com confirmação)"""
        
        response = input("\nDeseja prosseguir com a limpeza? (s/n): ")
        if response.lower() != 's':
            print("Limpeza cancelada")
            return False
            
        print("\n=== EXECUTANDO LIMPEZA ===")
        
        # Deletar arquivos
        deleted_count = 0
        for item in self.cleanup_report['files_to_delete']:
            try:
                os.remove(item['path'])
                deleted_count += 1
            except Exception as e:
                print(f"Erro ao deletar {item['path']}: {e}")
                
        print(f"Deletados {deleted_count} arquivos")
        
        # Remover diretórios vazios
        removed_dirs = 0
        for dir_path in self.cleanup_report['directories_to_remove']:
            try:
                os.rmdir(dir_path)
                removed_dirs += 1
            except Exception as e:
                print(f"Erro ao remover {dir_path}: {e}")
                
        print(f"Removidos {removed_dirs} diretórios vazios")
        
        return True

def main():
    print("="*70)
    print("SISTEMA DE LIMPEZA v2")
    print("="*70)
    
    cleaner = SystemCleaner()
    
    # Analisar
    cleaner.analyze_git_deletions()
    cleaner.identify_obsolete_files()
    cleaner.identify_empty_directories()
    
    # Mostrar resumo
    cleaner.show_summary()
    
    # Salvar relatório
    cleaner.save_report()
    
    # Executar limpeza
    if cleaner.execute_cleanup():
        print("\n✅ Limpeza concluída com sucesso!")
        print("\nPróximos passos:")
        print("1. git add -A")
        print("2. git commit -m '🧹 Limpeza do sistema - Remoção de arquivos obsoletos'")
        print("3. git push")
    
if __name__ == "__main__":
    main()