# test_training_imports.py
"""
Script para testar as importações do sistema de treinamento
"""

import sys
import os

print("🔍 Testando importações do sistema de treinamento...\n")

# Adicionar paths necessários
current_dir = os.getcwd()
print(f"📁 Diretório atual: {current_dir}")

# Adicionar src ao path se necessário
src_path = os.path.join(current_dir, 'src')
if os.path.exists(src_path):
    sys.path.insert(0, src_path)
    print(f"✅ Adicionado ao path: {src_path}")
else:
    # Se não existe src/ no diretório atual, talvez estejamos em src/
    if 'src' in current_dir:
        parent_dir = os.path.dirname(current_dir)
        sys.path.insert(0, parent_dir)
        print(f"✅ Adicionado ao path: {parent_dir}")

print(f"📋 Python path atual:")
for i, path in enumerate(sys.path[:5]):  # Mostrar apenas primeiros 5
    print(f"   {i+1}. {path}")

# Testar importações uma por uma
test_modules = [
    ('training.data_loader', 'TrainingDataLoader'),
    ('training.preprocessor', 'DataPreprocessor'), 
    ('training.feature_pipeline', 'FeatureEngineeringPipeline'),
    ('training.model_trainer', 'ModelTrainer'),
    ('training.ensemble_trainer', 'EnsembleTrainer'),
    ('training.validation_engine', 'ValidationEngine'),
    ('training.hyperopt_engine', 'HyperparameterOptimizer'),
    ('training.trading_metrics', 'TradingMetricsAnalyzer'),
    ('training.performance_analyzer', 'PerformanceAnalyzer'),
    ('training.training_orchestrator', 'TrainingOrchestrator'),
]

print(f"\n🧪 Testando importações dos módulos de treinamento:")

success_count = 0
failed_imports = []

for module_name, class_name in test_modules:
    try:
        module = __import__(module_name, fromlist=[class_name])
        class_obj = getattr(module, class_name)
        print(f"✅ {module_name}.{class_name} - OK")
        success_count += 1
    except ImportError as e:
        print(f"❌ {module_name}.{class_name} - FALHOU: {e}")
        failed_imports.append((module_name, class_name, str(e)))
    except AttributeError as e:
        print(f"⚠️  {module_name}.{class_name} - CLASSE NÃO ENCONTRADA: {e}")
        failed_imports.append((module_name, class_name, str(e)))
    except Exception as e:
        print(f"💥 {module_name}.{class_name} - ERRO: {e}")
        failed_imports.append((module_name, class_name, str(e)))

print(f"\n📊 Resultado do teste:")
print(f"   Sucessos: {success_count}/{len(test_modules)}")
print(f"   Falhas: {len(failed_imports)}/{len(test_modules)}")

if failed_imports:
    print(f"\n❌ Módulos com problemas:")
    for module, class_name, error in failed_imports:
        print(f"   📄 {module}.{class_name}: {error}")
        
    print(f"\n💡 Possíveis soluções:")
    print(f"   1. Verifique se todos os arquivos .py existem no diretório src/training/")
    print(f"   2. Confirme que não há erros de sintaxe nos arquivos")
    print(f"   3. Execute do diretório raiz do projeto: python test_training_imports.py")
else:
    print(f"\n🎉 Todos os módulos importaram com sucesso!")
    print(f"✅ O sistema de treinamento está pronto para uso!")
