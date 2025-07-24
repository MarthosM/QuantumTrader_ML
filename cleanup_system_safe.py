"""
Script de Limpeza Segura do Sistema ML Trading v2.0
Data: 2025-07-24
Objetivo: Remover arquivos obsoletos, temporários e desnecessários

IMPORTANTE: Este script lista arquivos a serem removidos e pede confirmação
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

def create_cleanup_report():
    """Cria relatório de limpeza"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"cleanup_report_{timestamp}.txt"

def analyze_files_to_remove():
    """Analisa e lista arquivos para remoção"""
    
    base_path = Path(".")
    files_to_remove = {
        'backup_directories': [],
        'test_files_root': [],
        'fixed_files': [],
        'temp_scripts': [],
        'old_logs': [],
        'csv_files': [],
        'simple_versions': [],
        'debug_scripts': []
    }
    
    # 1. Diretório de backup inteiro
    backup_dir = base_path / "backup_limpeza_20250722_121044"
    if backup_dir.exists():
        files_to_remove['backup_directories'].append(backup_dir)
    
    # 2. Arquivos de teste no diretório raiz
    test_patterns = [
        'test_*.py',
        'debug_*.py',
        'diagnose_*.py'
    ]
    
    for pattern in test_patterns:
        for file in base_path.glob(pattern):
            if file.is_file():
                # Excluir arquivos essenciais de debug recentes
                if file.name not in ['diagnose_data_flow.py', 'debug_system_predictions.py']:
                    if 'test_' in file.name:
                        files_to_remove['test_files_root'].append(file)
                    elif 'debug_' in file.name or 'diagnose_' in file.name:
                        files_to_remove['debug_scripts'].append(file)
    
    # 3. Arquivos _fixed e backups
    fixed_patterns = [
        'src/*_fixed.py',
        'src/*.backup*',
        '*.backup*'
    ]
    
    for pattern in fixed_patterns:
        for file in base_path.glob(pattern):
            if file.is_file():
                files_to_remove['fixed_files'].append(file)
    
    # 4. Scripts temporários de correção
    temp_scripts = [
        'apply_ml_flow_integration.py',
        'apply_ml_flow_integration_fixed.py',
        'fix_gui_threading.py',
        'fix_integration_tests.py',
        'fix_trading_system.py',
        'final_fixes.py',
        'intelligent_cleanup.py',
        'ml_data_flow_integrator.py',
        'data_flow_monitor.py',
        'monitor_system_status.py'
    ]
    
    for script in temp_scripts:
        file_path = base_path / script
        if file_path.exists():
            files_to_remove['temp_scripts'].append(file_path)
    
    # 5. Logs e relatórios antigos
    log_patterns = [
        'ml_trading_*.log',
        'src/backtest_report_*.html',
        'production_test_detailed_*.txt',
        'data_flow_diagnosis_*.txt',
        'ml_system_bypass_test_*.txt',
        'ml_predictions_test_*.txt'
    ]
    
    for pattern in log_patterns:
        for file in base_path.glob(pattern):
            if file.is_file():
                files_to_remove['old_logs'].append(file)
    
    # 6. Arquivos CSV temporários
    csv_files = [
        'features_output.csv',
        'features_basic_7200.csv',
        'features_full_7200.csv',
        'features_optimized_7200.csv'
    ]
    
    for csv in csv_files:
        file_path = base_path / csv
        if file_path.exists():
            files_to_remove['csv_files'].append(file_path)
    
    # 7. Versões _simple (verificar se ainda são usadas)
    simple_files = [
        'src/model_monitor_simple.py',
        'src/performance_analyzer_simple.py',
        'src/execution_integration_simple.py',
        'src/diagnostics_simple.py',
        'src/dashboard_simple.py',
        'src/alerting_system_simple.py'
    ]
    
    # Verificar se arquivos _simple são importados
    imports_found = False
    for py_file in base_path.glob('src/*.py'):
        if py_file.is_file():
            try:
                content = py_file.read_text(encoding='utf-8')
                for simple in simple_files:
                    simple_name = Path(simple).stem
                    if f'import {simple_name}' in content or f'from {simple_name}' in content:
                        imports_found = True
                        break
            except:
                pass
    
    if not imports_found:
        for simple in simple_files:
            file_path = base_path / simple
            if file_path.exists():
                files_to_remove['simple_versions'].append(file_path)
    
    return files_to_remove

def calculate_size(files_dict):
    """Calcula tamanho total dos arquivos"""
    total_size = 0
    total_files = 0
    
    for category, files in files_dict.items():
        for file_path in files:
            try:
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    total_files += 1
                elif file_path.is_dir():
                    # Calcular tamanho do diretório
                    for root, dirs, files in os.walk(file_path):
                        for f in files:
                            fp = Path(root) / f
                            if fp.exists():
                                total_size += fp.stat().st_size
                                total_files += 1
            except:
                pass
    
    return total_size, total_files

def print_files_to_remove(files_dict):
    """Imprime lista organizada de arquivos"""
    print("\n" + "="*80)
    print("ARQUIVOS IDENTIFICADOS PARA REMOÇÃO")
    print("="*80)
    
    categories = {
        'backup_directories': '📁 Diretórios de Backup',
        'test_files_root': '🧪 Arquivos de Teste (raiz)',
        'fixed_files': '🔧 Arquivos _fixed e Backups',
        'temp_scripts': '⚡ Scripts Temporários',
        'old_logs': '📝 Logs e Relatórios Antigos',
        'csv_files': '📊 Arquivos CSV Temporários',
        'simple_versions': '📄 Versões _simple',
        'debug_scripts': '🐛 Scripts de Debug'
    }
    
    total_count = 0
    for category, title in categories.items():
        files = files_dict.get(category, [])
        if files:
            print(f"\n{title}:")
            print("-" * 40)
            for file in sorted(files):
                print(f"  • {file}")
                total_count += 1
    
    return total_count

def perform_cleanup(files_dict, report_file):
    """Executa a limpeza com confirmação"""
    
    # Calcular estatísticas
    total_size, total_files = calculate_size(files_dict)
    size_mb = total_size / (1024 * 1024)
    
    # Mostrar resumo
    print("\n" + "="*80)
    print("RESUMO DA LIMPEZA")
    print("="*80)
    print(f"Total de arquivos/diretórios: {total_files}")
    print(f"Tamanho total: {size_mb:.2f} MB")
    print(f"Relatório será salvo em: {report_file}")
    
    # Confirmar
    print("\n⚠️  ATENÇÃO: Esta operação é IRREVERSÍVEL!")
    response = input("\nDeseja prosseguir com a limpeza? (sim/não): ").lower().strip()
    
    if response not in ['sim', 's', 'yes', 'y']:
        print("\n❌ Limpeza cancelada pelo usuário.")
        return False
    
    # Criar relatório
    removed_count = 0
    errors = []
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"RELATÓRIO DE LIMPEZA - ML Trading v2.0\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Executar remoção
        for category, files in files_dict.items():
            if files:
                f.write(f"\n{category.upper()}:\n")
                f.write("-"*40 + "\n")
                
                for file_path in files:
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                            f.write(f"✓ Removido: {file_path}\n")
                            removed_count += 1
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                            f.write(f"✓ Removido diretório: {file_path}\n")
                            removed_count += 1
                    except Exception as e:
                        error_msg = f"✗ Erro ao remover {file_path}: {e}"
                        f.write(error_msg + "\n")
                        errors.append(error_msg)
        
        # Resumo final
        f.write(f"\n\nRESUMO FINAL:\n")
        f.write(f"Arquivos/diretórios removidos: {removed_count}\n")
        f.write(f"Erros encontrados: {len(errors)}\n")
        f.write(f"Espaço liberado: ~{size_mb:.2f} MB\n")
    
    return removed_count, errors

def main():
    """Função principal"""
    print("🧹 SISTEMA DE LIMPEZA SEGURA - ML Trading v2.0")
    print("="*80)
    
    # Analisar arquivos
    print("\n📋 Analisando sistema de arquivos...")
    files_to_remove = analyze_files_to_remove()
    
    # Mostrar arquivos
    total_count = print_files_to_remove(files_to_remove)
    
    if total_count == 0:
        print("\n✅ Nenhum arquivo desnecessário encontrado!")
        return
    
    # Executar limpeza
    report_file = create_cleanup_report()
    removed_count, errors = perform_cleanup(files_to_remove, report_file)
    
    # Resultado final
    print("\n" + "="*80)
    print("LIMPEZA CONCLUÍDA")
    print("="*80)
    
    if removed_count > 0:
        print(f"✅ {removed_count} arquivos/diretórios removidos com sucesso!")
        print(f"📄 Relatório salvo em: {report_file}")
    
    if errors:
        print(f"\n⚠️  {len(errors)} erros encontrados:")
        for error in errors[:5]:  # Mostrar até 5 erros
            print(f"  • {error}")
    
    print("\n🎯 Recomendações pós-limpeza:")
    print("  1. Execute 'pytest' para verificar que testes ainda funcionam")
    print("  2. Execute 'python src/main.py' para testar o sistema")
    print("  3. Verifique o relatório de limpeza para detalhes")
    print("  4. Considere fazer commit das mudanças")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Limpeza interrompida pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro crítico: {e}")
        import traceback
        traceback.print_exc()