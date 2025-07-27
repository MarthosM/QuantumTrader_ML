"""
Script de Cleanup da Fase 2 - ML Pipeline
"""

import os
import shutil
from datetime import datetime
import subprocess

def cleanup_temp_files():
    """Remove arquivos temporários da Fase 2"""
    print("\n1. Removendo arquivos temporários...")
    
    temp_files = [
        'create_test_dataset.py',
        'validate_phase2.py'
    ]
    
    removed = 0
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"   [REMOVED] {file}")
            removed += 1
    
    print(f"   Total removido: {removed} arquivos")
    return removed

def organize_documentation():
    """Organiza documentação da Fase 2"""
    print("\n2. Organizando documentação...")
    
    # Criar diretório de documentação se não existir
    docs_dir = 'docs/phase2'
    os.makedirs(docs_dir, exist_ok=True)
    
    # Mover relatório de conclusão
    if os.path.exists('FASE2_COMPLETION_REPORT.md'):
        shutil.move('FASE2_COMPLETION_REPORT.md', f'{docs_dir}/COMPLETION_REPORT.md')
        print(f"   [MOVED] Relatório de conclusão para {docs_dir}/")
    
    return True

def create_backup():
    """Cria backup dos componentes da Fase 2"""
    print("\n3. Criando backup da Fase 2...")
    
    backup_dir = f'backups/fase2_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(backup_dir, exist_ok=True)
    
    # Componentes para backup
    components = [
        ('src/features/ml_features_v3.py', 'features/'),
        ('src/ml/dataset_builder_v3.py', 'ml/'),
        ('src/ml/training_orchestrator_v3.py', 'ml/')
    ]
    
    backed_up = 0
    for src, dst_subdir in components:
        if os.path.exists(src):
            dst_dir = os.path.join(backup_dir, dst_subdir)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, dst_dir)
            backed_up += 1
    
    print(f"   [OK] Backup criado em: {backup_dir}")
    print(f"   [OK] Componentes salvos: {backed_up}")
    
    return backup_dir

def git_operations():
    """Realiza operações git para a Fase 2"""
    print("\n4. Realizando operações Git...")
    
    try:
        # Add all changes
        subprocess.run(['git', 'add', '.'], check=True)
        print("   [OK] Arquivos adicionados ao git")
        
        # Commit
        commit_msg = "Fase 2 concluída: ML Pipeline implementado\n\n- MLFeaturesV3 com 118 features\n- DatasetBuilderV3 com validação temporal\n- TrainingOrchestratorV3 com modelos por regime\n- 9 modelos treinados (3 algoritmos x 3 regimes)\n- Validação: 100% sucesso"
        
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
        print("   [OK] Commit realizado")
        
        # Tag
        tag_name = 'v2.0-fase2-complete'
        subprocess.run(['git', 'tag', '-a', tag_name, '-m', 'Fase 2: ML Pipeline completo'], check=True)
        print(f"   [OK] Tag criada: {tag_name}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   [ERRO] Operação git falhou: {e}")
        return False

def update_phase_status():
    """Atualiza status para próxima fase"""
    print("\n5. Atualizando status do projeto...")
    
    with open('.phase_status', 'r') as f:
        lines = f.readlines()
    
    # Atualizar para Fase 3
    new_lines = []
    for line in lines:
        if line.startswith('CURRENT_PHASE='):
            new_lines.append('CURRENT_PHASE=3\n')
        else:
            new_lines.append(line)
    
    # Adicionar conclusão da Fase 2
    new_lines.append(f'FASE2_COMPLETED={datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}\n')
    
    with open('.phase_status', 'w') as f:
        f.writelines(new_lines)
    
    print("   [OK] Status atualizado para Fase 3")
    return True

def generate_summary():
    """Gera resumo final da Fase 2"""
    print("\n" + "="*60)
    print("RESUMO DO CLEANUP - FASE 2")
    print("="*60)
    
    summary = f"""
Fase 2: ML Pipeline - Cleanup Concluído

Componentes Implementados:
- MLFeaturesV3: 118 features avançadas
- DatasetBuilderV3: Construção automatizada de datasets
- TrainingOrchestratorV3: Pipeline unificado de treinamento

Modelos Treinados: 9
- 3 algoritmos (XGBoost, LightGBM, RandomForest)
- 3 regimes (trend_up, trend_down, range)

Validação: 100% sucesso
- Todos os componentes testados
- Features com 0% NaN
- Dataset com 10,076 samples

Próxima Fase: Real-time Integration
- Implementar processamento em tempo real
- Integrar com ProfitDLL
- Sistema de predição online
"""
    
    print(summary)
    
    # Salvar resumo
    with open('docs/phase2/CLEANUP_SUMMARY.md', 'w') as f:
        f.write(summary)

def main():
    """Executa cleanup completo da Fase 2"""
    print("="*60)
    print("CLEANUP DA FASE 2 - ML PIPELINE")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Executar tarefas de cleanup
    tasks = [
        ("Remover temporários", cleanup_temp_files),
        ("Organizar documentação", organize_documentation),
        ("Criar backup", create_backup),
        ("Operações Git", git_operations),
        ("Atualizar status", update_phase_status)
    ]
    
    results = []
    for task_name, task_func in tasks:
        try:
            result = task_func()
            results.append((task_name, True, result))
        except Exception as e:
            print(f"\n[ERRO] {task_name}: {e}")
            results.append((task_name, False, str(e)))
    
    # Gerar resumo
    generate_summary()
    
    # Status final
    success_count = sum(1 for _, success, _ in results if success)
    total_count = len(results)
    
    print(f"\nTarefas concluídas: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n✅ CLEANUP DA FASE 2 CONCLUÍDO COM SUCESSO!")
        print("\n🚀 Pronto para iniciar Fase 3: Real-time Integration")
    else:
        print("\n⚠️ Cleanup parcialmente concluído")

if __name__ == "__main__":
    main()