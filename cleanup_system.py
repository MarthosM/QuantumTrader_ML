"""
Script de Limpeza do Sistema QuantumTrader_ML
Data: 03/08/2025

Este script remove arquivos obsoletos e organiza o sistema
"""

import os
import shutil
import glob
from datetime import datetime
from pathlib import Path
import json

class SystemCleaner:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.removed_files = []
        self.archived_files = []
        self.kept_files = []
        self.errors = []
        self.space_saved = 0
        
    def log(self, message):
        """Log com timestamp"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def get_file_size(self, file_path):
        """Retorna tamanho do arquivo em MB"""
        try:
            return os.path.getsize(file_path) / (1024 * 1024)
        except:
            return 0
    
    def remove_file(self, file_path, reason=""):
        """Remove arquivo (ou simula se dry_run)"""
        try:
            size = self.get_file_size(file_path)
            if self.dry_run:
                self.log(f"[DRY RUN] Removeria: {file_path} ({size:.2f}MB) - {reason}")
            else:
                os.remove(file_path)
                self.log(f"✓ Removido: {file_path} ({size:.2f}MB)")
            
            self.removed_files.append(file_path)
            self.space_saved += size
            
        except Exception as e:
            self.errors.append(f"Erro ao remover {file_path}: {e}")
    
    def remove_directory(self, dir_path, reason=""):
        """Remove diretório (ou simula se dry_run)"""
        try:
            # Calcular tamanho total
            total_size = 0
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    total_size += self.get_file_size(os.path.join(root, file))
            
            if self.dry_run:
                self.log(f"[DRY RUN] Removeria diretório: {dir_path} ({total_size:.2f}MB) - {reason}")
            else:
                shutil.rmtree(dir_path)
                self.log(f"✓ Removido diretório: {dir_path} ({total_size:.2f}MB)")
            
            self.removed_files.append(dir_path)
            self.space_saved += total_size
            
        except Exception as e:
            self.errors.append(f"Erro ao remover diretório {dir_path}: {e}")
    
    def archive_file(self, file_path, archive_subdir=""):
        """Move arquivo para pasta archive"""
        try:
            # Criar diretório archive se não existir
            archive_dir = Path("archive") / archive_subdir
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Destino
            dest_path = archive_dir / Path(file_path).name
            
            size = self.get_file_size(file_path)
            
            if self.dry_run:
                self.log(f"[DRY RUN] Arquivaria: {file_path} → {dest_path} ({size:.2f}MB)")
            else:
                shutil.move(file_path, dest_path)
                self.log(f"✓ Arquivado: {file_path} → {dest_path}")
            
            self.archived_files.append(file_path)
            
        except Exception as e:
            self.errors.append(f"Erro ao arquivar {file_path}: {e}")
    
    def clean_pycache(self):
        """Remove todos os diretórios __pycache__ e arquivos .pyc"""
        self.log("\n=== Limpando Cache Python ===")
        
        # Encontrar todos __pycache__
        for root, dirs, files in os.walk("."):
            if "__pycache__" in dirs:
                cache_path = os.path.join(root, "__pycache__")
                self.remove_directory(cache_path, "Cache Python")
        
        # Remover arquivos .pyc soltos
        for pyc_file in glob.glob("**/*.pyc", recursive=True):
            self.remove_file(pyc_file, "Arquivo Python compilado")
    
    def clean_backups(self):
        """Move backups para arquivo"""
        self.log("\n=== Organizando Backups ===")
        
        # Backups estruturados
        if os.path.exists("backup_pre_v4_20250731_191210"):
            self.archive_file("backup_pre_v4_20250731_191210", "backups")
        
        if os.path.exists("backups"):
            self.archive_file("backups", "backups")
        
        # Backups individuais
        for backup in glob.glob("src/*_backup_*.py"):
            self.archive_file(backup, "backups/src")
    
    def clean_test_scripts(self):
        """Remove scripts de teste temporários do diretório raiz"""
        self.log("\n=== Limpando Scripts de Teste ===")
        
        # Padrões de arquivos temporários no raiz
        patterns = [
            "test_*.py",
            "debug_*.py", 
            "diagnose_*.py",
            "create_*.py",
            "prepare_*.py",
            "analyze_*.py",
            "backtest_*.py",
            "ml_backtest*.py",
            "quick_*.py"
        ]
        
        # Scripts importantes para MANTER
        keep_scripts = [
            "test_connection.py",  # Mantém teste básico de conexão
            "create_models.py",    # Mantém criação de modelos
        ]
        
        for pattern in patterns:
            for file in glob.glob(pattern):
                if file not in keep_scripts and os.path.isfile(file):
                    self.remove_file(file, "Script de teste temporário")
    
    def clean_logs(self):
        """Remove logs antigos"""
        self.log("\n=== Limpando Logs Antigos ===")
        
        # Remover logs específicos antigos
        old_logs = [
            "logs/data_collection_20250802.log",
            "logs/profit_dll_server.log",
            "stress_test_report_20250728_071919.json",
            "test_complete_system_report.json"
        ]
        
        for log_file in old_logs:
            if os.path.exists(log_file):
                self.remove_file(log_file, "Log antigo")
        
        # Remover relatórios antigos
        for report in glob.glob("reports/*_2025072*.txt"):
            self.remove_file(report, "Relatório antigo")
    
    def clean_temp_files(self):
        """Remove arquivos temporários diversos"""
        self.log("\n=== Limpando Arquivos Temporários ===")
        
        # Arquivo nul
        if os.path.exists("nul"):
            self.remove_file("nul", "Arquivo temporário vazio")
        
        # Scripts corrompidos/duplicados
        if os.path.exists("scriptstest_simple_collection.py"):
            self.remove_file("scriptstest_simple_collection.py", "Script corrompido")
        
        # Arquivos de resultado antigos
        for result in glob.glob("results/*_v3_*.json"):
            self.archive_file(result, "results_v3")
    
    def clean_v3_files(self):
        """Arquiva arquivos V3 (com cuidado)"""
        self.log("\n=== Arquivando Arquivos V3 ===")
        
        # Modelos V3
        for model in glob.glob("models/*_v3_*.pkl") + glob.glob("models/*_v3.pkl"):
            self.archive_file(model, "models_v3")
        
        # Features V3 duplicadas
        if os.path.exists("src/ml/models/feature_scaler_v3.pkl"):
            self.remove_file("src/ml/models/feature_scaler_v3.pkl", "Scaler V3 duplicado")
    
    def clean_empty_dirs(self):
        """Remove diretórios vazios"""
        self.log("\n=== Removendo Diretórios Vazios ===")
        
        for root, dirs, files in os.walk(".", topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path) and "__pycache__" not in dir_path:
                        if self.dry_run:
                            self.log(f"[DRY RUN] Removeria diretório vazio: {dir_path}")
                        else:
                            os.rmdir(dir_path)
                            self.log(f"✓ Removido diretório vazio: {dir_path}")
                except:
                    pass
    
    def generate_report(self):
        """Gera relatório da limpeza"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "removed_files": len(self.removed_files),
            "archived_files": len(self.archived_files),
            "space_saved_mb": round(self.space_saved, 2),
            "errors": len(self.errors),
            "details": {
                "removed": self.removed_files[:20],  # Primeiros 20
                "archived": self.archived_files,
                "errors": self.errors
            }
        }
        
        # Salvar relatório
        report_path = f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Mostrar resumo
        self.log("\n" + "="*60)
        self.log("RESUMO DA LIMPEZA")
        self.log("="*60)
        self.log(f"Modo: {'SIMULAÇÃO' if self.dry_run else 'EXECUÇÃO REAL'}")
        self.log(f"Arquivos removidos: {len(self.removed_files)}")
        self.log(f"Arquivos arquivados: {len(self.archived_files)}")
        self.log(f"Espaço liberado: {self.space_saved:.2f} MB")
        self.log(f"Erros: {len(self.errors)}")
        self.log(f"Relatório salvo em: {report_path}")
        
        if self.errors:
            self.log("\nERROS ENCONTRADOS:")
            for error in self.errors[:5]:
                self.log(f"  - {error}")
    
    def run(self):
        """Executa limpeza completa"""
        self.log("🧹 INICIANDO LIMPEZA DO SISTEMA")
        self.log(f"Modo: {'SIMULAÇÃO (dry run)' if self.dry_run else 'EXECUÇÃO REAL'}")
        
        # Executar limpezas
        self.clean_pycache()
        self.clean_backups()
        self.clean_test_scripts()
        self.clean_logs()
        self.clean_temp_files()
        self.clean_v3_files()
        self.clean_empty_dirs()
        
        # Gerar relatório
        self.generate_report()


def main():
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║         LIMPEZA DO SISTEMA QUANTUMTRADER_ML          ║
    ╚═══════════════════════════════════════════════════════╝
    
    Este script irá:
    - Remover cache Python (__pycache__, *.pyc)
    - Arquivar backups antigos
    - Remover scripts de teste temporários
    - Limpar logs antigos
    - Organizar arquivos V3
    - Remover diretórios vazios
    
    """)
    
    # Perguntar modo
    response = input("Executar em modo SIMULAÇÃO primeiro? (s/n) [s]: ").strip().lower()
    dry_run = response != 'n'
    
    if not dry_run:
        confirm = input("\n⚠️  ATENÇÃO: Modo REAL irá REMOVER arquivos! Confirmar? (sim/não): ")
        if confirm.lower() != 'sim':
            print("Operação cancelada.")
            return
    
    # Criar e executar limpeza
    cleaner = SystemCleaner(dry_run=dry_run)
    cleaner.run()
    
    if dry_run:
        print("\n✅ Simulação concluída! Execute novamente com modo REAL para aplicar as mudanças.")


if __name__ == "__main__":
    main()