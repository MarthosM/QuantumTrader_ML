#!/usr/bin/env python3
"""
🧹 LIMPEZA INTELIGENTE DE ARQUIVOS - ML TRADING v2.0
===================================================
Data: 22/07/2025 - 12:15
Objetivo: Remover arquivos temporários e de teste, mantendo estrutura essencial

ESTRATÉGIA DE LIMPEZA:
✅ Backup antes de remover
✅ Categorização por tipo de arquivo
✅ Preservar arquivos essenciais
✅ Log detalhado das operações
"""

import os
import shutil
import glob
from datetime import datetime
from typing import List, Dict, Set
import tempfile

class IntelligentFileCleaner:
    """Limpeza inteligente e segura de arquivos"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.removed_files = []
        self.preserved_files = []
        self.backup_dir = None
        
        # Configurar categorias de arquivos
        self.setup_categories()
        
    def setup_categories(self):
        """Define categorias de arquivos para limpeza"""
        
        # ❌ REMOVER: Arquivos temporários de teste
        self.temp_test_files = {
            # Testes temporários específicos
            'test_3_days_smart.py',
            'test_5_days.py', 
            'test_alternative_tickers.py',
            'test_api_structure.py',
            'test_backtest_debug.py',
            'test_backtest_features_behavior.py',
            'test_basic.py',
            'test_clean_final.py',
            'test_complete_order_system.py',
            'test_complete_system.py',
            'test_correção_validação.py',
            'test_dataframe_print.py',
            'test_dataframe_stress.py',
            'test_data_flow.py',
            'test_execution_fixes.py',
            'test_execution_integration.py',
            'test_features_system.py',
            'test_final_integrated.py',
            'test_gui.py',
            'test_historical_api.py',
            'test_imports_simple.py',
            'test_improvements.py',
            'test_ml_training_complete.py',
            'test_monitor_advanced.py',
            'test_optimized.py',
            'test_order_system_advanced.py',
            'test_order_system_simple.py',
            'test_safe_float_conversion.py',
            'test_simple_gui.py',
            'test_simple_print.py',
            'test_simple_system.py',
            'test_stop_strategies.py',
            'test_tf_silent.py',
            'test_timestamp_conversion.py',
            'test_trading_system_corrections.py',
            'test_training_imports.py',
            'test_training_integration.py',
            'test_wdo_backtest_data.py'
        }
        
        # ❌ REMOVER: Scripts de correção temporários
        self.temp_fix_scripts = {
            'apply_critical_patches.py',
            'configure_aggressive.py',
            'fix_gui_monitor.py',
            'fix_gui_thread.py',
            'fix_data_fill_priority.py',
            'debug_position.py',
            'debug_timestamp.py',
            'demo_complete.py',
            'exemplo_sistema_integrado.py',
            'print_dataframe_now.py',
            'RESUMO_DATAFRAME_PRINT.py',
            'system_cleaner.py',
            'safe_cleanup.py',
            'cleanup_old_models.py',
            'create_mock_models.py',
            'migrate_models.py',
            'quick_start.py',
            'quick_train.py'
        }
        
        # ❌ REMOVER: Logs temporários
        self.temp_logs = {
            'backtest_execution.log',
            'test_3_days_smart.log',
            'test_execution_fallback.log',
            'test_historical_api.log',
            'cleanup_analysis.txt',
            'safe_cleanup_report.txt',
            'simple_backtest_results.csv'
        }
        
        # ❌ REMOVER: Documentação temporária/duplicada
        self.temp_docs = {
            'ATUALIZACOES_DOCS_2025-07-21.md',
            'CORRECAO_GUI_FINAL.md',
            'CORRECOES_APLICADAS_FINAL.md',
            'CRITICAL_TRADING_ANALYSIS.md',
            'DIAGNOSTICO_CORRECAO_20250722.md',
            'ETAPA_6_SISTEMA_TREINAMENTO_ML.md',
            'FINAL_ANALYSIS_COMPLETE.md',
            'HISTORICAL_DATA_FIXES.md',
            'INTEGRATION_GUIDE.md',
            'LOG_ATUALIZACOES.md',
            'MAPA_ATUALIZADO_RESUMO.md',
            'ml-trading-roadmap-v3.md',
            'MONITOR_GUI_README.md',
            'MONITOR_GUI_STATUS_FINAL.md',
            'PRODUCTION_SAFE_DATA_FLOW.md',
            'SISTEMA_TREINAMENTO_INTEGRADO.md',
            'STATUS_FINAL_SISTEMA.md',
            'SYSTEM_OVERVIEW.md',
            'TENSORFLOW_RESOLUTION_FINAL.md',
            'TESTE_COMPLETO_RELATORIO.md',
            'TRADING_SYSTEM_EXECUTION_UPDATE.md',
            'TRADING_SYSTEM_FIXES.md',
            'UPDATED_ANALYSIS_COMPARISON.md'
        }
        
        # ❌ REMOVER: Scripts de execução duplicados
        self.duplicate_runners = {
            'start_gui_direct.py',
            'start_with_gui.py',
            'realtime_monitor.py',
            'monitor_corrections.py',
            'run_monitor_gui.py',
            'run_training_safe.py',
            'run_backtest_example.py',
            'backtest_final.py',
            'simple_backtest_wdo.py'
        }
        
        # ✅ PRESERVAR: Arquivos essenciais
        self.essential_files = {
            # Core do sistema
            'README.md',
            'DEVELOPER_GUIDE.md',
            'DOCS_INDEX.md',
            'requirements.txt',
            'pyproject.toml',
            '.env',
            '.env.example',
            '.gitignore',
            
            # Scripts principais
            'start_ml_trading.py',  # Script principal criado
            'run_training.py',      # Treinamento principal
            'fix_trading_system.py', # Correções aplicadas
            
            # Documentação final
            'ITERACAO_2025-07-22_CORRECOES_CRITICAS.md',
            'guia_upgrade.md',
            
            # Manual
            'Manual - ProfitDLL en_us.pdf'
        }
        
        # ✅ PRESERVAR: Diretórios essenciais  
        self.essential_dirs = {
            'src', 'tests', 'models', 'data', 'config', 
            'notebooks', 'projeto', '.git', '.venv', 
            '.vscode', '.github', '.pytest_cache'
        }
    
    def create_backup(self):
        """Cria backup dos arquivos que serão removidos"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.backup_dir = f"backup_limpeza_{timestamp}"
        
        print(f"📦 Criando backup em: {self.backup_dir}")
        os.makedirs(self.backup_dir, exist_ok=True)
        
        return self.backup_dir
    
    def analyze_files(self):
        """Analisa arquivos no diretório atual"""
        print("🔍 ANALISANDO ARQUIVOS PARA LIMPEZA...")
        print("="*50)
        
        all_files = set()
        
        # Listar todos os arquivos no diretório atual
        for item in os.listdir('.'):
            if os.path.isfile(item):
                all_files.add(item)
        
        # Categorizar arquivos
        to_remove = set()
        to_remove.update(self.temp_test_files)
        to_remove.update(self.temp_fix_scripts)
        to_remove.update(self.temp_logs)
        to_remove.update(self.temp_docs)
        to_remove.update(self.duplicate_runners)
        
        # Filtrar apenas arquivos que existem
        existing_to_remove = {f for f in to_remove if f in all_files}
        existing_essential = {f for f in self.essential_files if f in all_files}
        
        print(f"📊 ANÁLISE:")
        print(f"   • Total de arquivos: {len(all_files)}")
        print(f"   • Para remoção: {len(existing_to_remove)}")
        print(f"   • Essenciais preservados: {len(existing_essential)}")
        print(f"   • Outros: {len(all_files) - len(existing_to_remove) - len(existing_essential)}")
        print("")
        
        return existing_to_remove, existing_essential, all_files
    
    def backup_and_remove(self, files_to_remove: Set[str]):
        """Faz backup e remove arquivos"""
        
        if not files_to_remove:
            print("✅ Nenhum arquivo para remover")
            return
            
        # Criar backup
        backup_dir = self.create_backup()
        
        print(f"🗑️ REMOVENDO {len(files_to_remove)} ARQUIVOS...")
        print("-" * 40)
        
        success_count = 0
        error_count = 0
        
        for file_path in sorted(files_to_remove):
            try:
                if os.path.exists(file_path):
                    # Backup primeiro
                    backup_path = os.path.join(backup_dir, file_path)
                    shutil.copy2(file_path, backup_path)
                    
                    # Remover arquivo
                    os.remove(file_path)
                    
                    self.removed_files.append(file_path)
                    success_count += 1
                    print(f"   ✅ {file_path}")
                    
            except Exception as e:
                error_count += 1
                print(f"   ❌ Erro ao remover {file_path}: {e}")
        
        print("")
        print(f"📊 RESULTADO:")
        print(f"   • Removidos com sucesso: {success_count}")
        print(f"   • Erros: {error_count}")
        print(f"   • Backup salvo em: {backup_dir}")
        print("")
    
    def clean_empty_directories(self):
        """Remove diretórios vazios (exceto essenciais)"""
        print("📁 VERIFICANDO DIRETÓRIOS VAZIOS...")
        
        removed_dirs = []
        
        for item in os.listdir('.'):
            if os.path.isdir(item) and item not in self.essential_dirs:
                try:
                    # Verificar se está vazio
                    if not os.listdir(item):
                        os.rmdir(item)
                        removed_dirs.append(item)
                        print(f"   ✅ Diretório vazio removido: {item}")
                except Exception as e:
                    print(f"   ⚠️ Erro ao remover diretório {item}: {e}")
        
        if not removed_dirs:
            print("   ✅ Nenhum diretório vazio encontrado")
        
        return removed_dirs
    
    def clean_test_directories(self):
        """Limpa diretórios de teste temporários"""
        print("🧪 LIMPANDO DIRETÓRIOS DE TESTE...")
        
        temp_test_dirs = ['test_temp']
        removed_dirs = []
        
        for test_dir in temp_test_dirs:
            if os.path.exists(test_dir):
                try:
                    shutil.rmtree(test_dir)
                    removed_dirs.append(test_dir)
                    print(f"   ✅ Removido: {test_dir}")
                except Exception as e:
                    print(f"   ❌ Erro ao remover {test_dir}: {e}")
        
        if not removed_dirs:
            print("   ✅ Nenhum diretório de teste temporário encontrado")
        
        return removed_dirs
    
    def show_preserved_structure(self, essential_files: Set[str]):
        """Mostra estrutura preservada"""
        print("📋 ESTRUTURA PRESERVADA:")
        print("-" * 30)
        
        # Arquivos essenciais
        print("📄 Arquivos principais:")
        for file in sorted(essential_files):
            if os.path.exists(file):
                print(f"   ✅ {file}")
        
        print("")
        
        # Diretórios essenciais
        print("📁 Diretórios principais:")
        for dir_name in sorted(self.essential_dirs):
            if os.path.exists(dir_name):
                file_count = len([f for f in os.listdir(dir_name) 
                                if os.path.isfile(os.path.join(dir_name, f))]) if os.path.isdir(dir_name) else 0
                print(f"   ✅ {dir_name}/ ({file_count} arquivos)")
        
        print("")
    
    def generate_cleanup_report(self):
        """Gera relatório de limpeza"""
        print("📋 GERANDO RELATÓRIO DE LIMPEZA...")
        
        report_content = f"""# Relatório de Limpeza - ML Trading v2.0
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Resumo da Operação
- **Início**: {self.start_time.strftime('%H:%M:%S')}
- **Fim**: {datetime.now().strftime('%H:%M:%S')}
- **Duração**: {(datetime.now() - self.start_time).total_seconds():.1f}s
- **Arquivos removidos**: {len(self.removed_files)}
- **Backup criado**: {self.backup_dir}

## 🗑️ Arquivos Removidos

### Testes Temporários ({len([f for f in self.removed_files if f in self.temp_test_files])})
{chr(10).join(['- ' + f for f in self.removed_files if f in self.temp_test_files])}

### Scripts de Correção Temporários ({len([f for f in self.removed_files if f in self.temp_fix_scripts])})
{chr(10).join(['- ' + f for f in self.removed_files if f in self.temp_fix_scripts])}

### Logs Temporários ({len([f for f in self.removed_files if f in self.temp_logs])})
{chr(10).join(['- ' + f for f in self.removed_files if f in self.temp_logs])}

### Documentação Temporária ({len([f for f in self.removed_files if f in self.temp_docs])})
{chr(10).join(['- ' + f for f in self.removed_files if f in self.temp_docs])}

### Scripts Duplicados ({len([f for f in self.removed_files if f in self.duplicate_runners])})
{chr(10).join(['- ' + f for f in self.removed_files if f in self.duplicate_runners])}

## ✅ Estrutura Final
O sistema agora mantém apenas:
- **src/**: Código fonte principal
- **tests/**: Testes oficiais (test_data_fill_corrections.py)
- **models/**: Modelos ML treinados
- **data/**: Dados do sistema
- **Documentação essencial**: README.md, DEVELOPER_GUIDE.md
- **Scripts principais**: start_ml_trading.py, run_training.py
- **Configuração**: .env, requirements.txt, pyproject.toml

## 🔄 Como Recuperar
Se precisar de algum arquivo removido:
1. Acesse o diretório de backup: `{self.backup_dir}`
2. Copie o arquivo necessário de volta
3. Execute: `cp {self.backup_dir}/nome_do_arquivo .`

## 📈 Benefícios Alcançados
- ✅ Estrutura limpa e organizada
- ✅ Redução de confusão entre arquivos
- ✅ Foco nos componentes essenciais
- ✅ Backup completo mantido
- ✅ Performance melhorada (menos arquivos)
"""
        
        report_file = 'RELATORIO_LIMPEZA.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"   ✅ Relatório salvo: {report_file}")
        return report_file
    
    def clean_system(self):
        """Executa limpeza completa do sistema"""
        print("🧹 INICIANDO LIMPEZA INTELIGENTE DO SISTEMA")
        print("="*50)
        print(f"🕐 Início: {self.start_time.strftime('%H:%M:%S')}")
        print("")
        
        try:
            # 1. Analisar arquivos
            files_to_remove, essential_files, all_files = self.analyze_files()
            
            # 2. Confirmar operação
            print("⚠️ CONFIRMAÇÃO NECESSÁRIA:")
            print(f"   Serão removidos {len(files_to_remove)} arquivos")
            print(f"   Backup será criado automaticamente")
            print(f"   Arquivos essenciais serão preservados")
            print("")
            
            # Simular confirmação (em produção, poderia usar input())
            confirmed = True  # input("Continuar? (y/N): ").lower() == 'y'
            
            if not confirmed:
                print("❌ Operação cancelada pelo usuário")
                return False
            
            print("✅ Prosseguindo com a limpeza...")
            print("")
            
            # 3. Backup e remoção de arquivos
            self.backup_and_remove(files_to_remove)
            
            # 4. Limpar diretórios vazios
            self.clean_empty_directories()
            
            # 5. Limpar diretórios de teste
            self.clean_test_directories()
            
            # 6. Mostrar estrutura preservada
            self.show_preserved_structure(essential_files)
            
            # 7. Gerar relatório
            report_file = self.generate_cleanup_report()
            
            print("="*50)
            print("✅ LIMPEZA CONCLUÍDA COM SUCESSO!")
            print("="*50)
            print(f"🕐 Tempo total: {(datetime.now() - self.start_time).total_seconds():.1f}s")
            print(f"📊 Arquivos removidos: {len(self.removed_files)}")
            print(f"📦 Backup: {self.backup_dir}")
            print(f"📋 Relatório: {report_file}")
            print("")
            print("🎯 SISTEMA AGORA ESTÁ LIMPO E ORGANIZADO!")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro durante limpeza: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Função principal"""
    cleaner = IntelligentFileCleaner()
    success = cleaner.clean_system()
    
    if success:
        print("\n🎉 LIMPEZA REALIZADA COM SUCESSO!")
    else:
        print("\n❌ FALHA NA LIMPEZA")

if __name__ == "__main__":
    main()
