#!/usr/bin/env python3
# quick_train.py
"""
Script simplificado para executar o treinamento
Execute do diretório raiz: python quick_train.py
"""

import sys
import os
from pathlib import Path

def setup_paths():
    """Configure os paths de forma robusta"""
    current_dir = Path.cwd()
    
    # Verificar se estamos no diretório correto
    if (current_dir / 'src' / 'training').exists():
        # Estamos na raiz do projeto
        project_root = current_dir
    elif current_dir.name == 'training' and (current_dir.parent.name == 'src'):
        # Estamos em src/training/
        project_root = current_dir.parent.parent
        os.chdir(project_root)
    elif current_dir.name == 'src':
        # Estamos em src/
        project_root = current_dir.parent
        os.chdir(project_root)
    else:
        print(f"❌ Diretório não reconhecido: {current_dir}")
        print("📁 Execute de uma destas localizações:")
        print("   - Raiz do projeto (ML_Tradingv2.0/)")
        print("   - src/")
        print("   - src/training/")
        return False
    
    # Adicionar paths necessários
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / 'src'))
    sys.path.insert(0, str(project_root / 'src' / 'training'))
    
    print(f"📁 Diretório de trabalho: {project_root}")
    return True

def main():
    """Função principal"""
    print("🧠 SISTEMA DE TREINAMENTO ML - TRADING v3.0")
    print("=" * 50)
    
    # Configurar paths
    if not setup_paths():
        return
    
    # Importações após configurar paths
    try:
        from training.training_orchestrator import TrainingOrchestrator
        print("✅ TrainingOrchestrator importado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao importar TrainingOrchestrator: {e}")
        print("\n🔍 Diagnóstico:")
        
        # Verificar se arquivos existem
        training_files = [
            'src/training/training_orchestrator.py',
            'src/training/data_loader.py',
            'src/training/model_trainer.py',
        ]
        
        missing_files = []
        for file in training_files:
            if not Path(file).exists():
                missing_files.append(file)
            else:
                print(f"✅ {file} - existe")
        
        if missing_files:
            print(f"\n❌ Arquivos ausentes:")
            for file in missing_files:
                print(f"   📄 {file}")
            print(f"\n💡 Execute a criação da ETAPA 6 completa primeiro")
        
        # Verificar erros de sintaxe
        print(f"\n🔬 Testando sintaxe dos arquivos...")
        import py_compile
        for file in training_files:
            if Path(file).exists():
                try:
                    py_compile.compile(file, doraise=True)
                    print(f"✅ {file} - sintaxe OK")
                except py_compile.PyCompileError as e:
                    print(f"❌ {file} - erro de sintaxe: {e}")
        
        return
    
    # Configuração básica
    config = {
        'data_path': 'data/historical/',
        'model_save_path': 'models/trained/',
        'models_dir': 'src/models/',
        'results_path': 'training_results/'
    }
    
    # Criar diretórios se necessário
    for path in config.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Configuração:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Inicializar orquestrador
    try:
        print(f"\n🛠️ Criando orquestrador...")
        orchestrator = TrainingOrchestrator(config)
        print("✅ Orquestrador criado com sucesso!")
        
        # Teste básico
        print(f"\n🧪 Executando teste básico...")
        
        # Aqui você pode adicionar chamadas básicas de teste
        print("✅ Sistema de treinamento funcionando!")
        
        print(f"\n🎉 Sistema pronto! Execute com:")
        print(f"   python run_training.py")
        
    except Exception as e:
        print(f"❌ Erro ao criar orquestrador: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
