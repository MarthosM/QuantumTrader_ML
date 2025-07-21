# src/training/load_trained_models.py
"""
Carrega modelos treinados no sistema de trading
"""

import sys
import os
from pathlib import Path
import json

# Adicionar paths necessários
current_dir = Path(__file__).parent
src_dir = current_dir.parent
project_root = src_dir.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

# Importações opcionais para quando integrado ao sistema
try:
    from src.models.model_manager import ModelManager
    from src.integration.training_integration import TrainingIntegration
    INTEGRATION_AVAILABLE = True
except ImportError:
    print("⚠️  Integração com sistema principal não disponível")
    INTEGRATION_AVAILABLE = False

def find_latest_training_results():
    """Encontra os resultados de treinamento mais recentes"""
    # Tentar tanto caminho relativo quanto absoluto
    possible_paths = [
        Path('models'),  # Quando executado de src/training/
        Path('src/training/models'),  # Quando executado do root
        current_dir / 'models'  # Caminho absoluto
    ]
    
    models_dir = None
    for path in possible_paths:
        if path.exists():
            models_dir = path
            break
    
    if not models_dir:
        print(f"❌ Diretório de modelos não encontrado. Tentativas: {possible_paths}")
        return None
    
    print(f"📂 Usando diretório: {models_dir.absolute()}")
    
    # Buscar por diretórios de treinamento
    training_dirs = [d for d in models_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('training_')]
    
    if not training_dirs:
        print("❌ Nenhum resultado de treinamento encontrado")
        return None
    
    # Pegar o mais recente
    latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📅 Treinamento mais recente: {latest_dir.name}")
    
    # Procurar metadata do ensemble
    ensemble_dirs = [d for d in latest_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('ensemble')]
    
    if not ensemble_dirs:
        print(f"❌ Nenhum ensemble encontrado em {latest_dir}")
        return None
    
    latest_ensemble = max(ensemble_dirs, key=lambda x: x.stat().st_mtime)
    
    # Buscar diretório ensemble_* dentro de ensemble/
    if latest_ensemble.name == 'ensemble':
        ensemble_subdirs = [d for d in latest_ensemble.iterdir() 
                           if d.is_dir() and d.name.startswith('ensemble_')]
        if ensemble_subdirs:
            latest_ensemble = max(ensemble_subdirs, key=lambda x: x.stat().st_mtime)
    
    metadata_file = latest_ensemble / 'ensemble_metadata.json'
    
    if not metadata_file.exists():
        print(f"❌ Metadata não encontrado: {metadata_file}")
        return None
        
    return metadata_file

def load_latest_models():
    """Carrega os modelos mais recentes"""
    metadata_file = find_latest_training_results()
    
    if not metadata_file:
        return False
    
    try:
        # Carregar resultados do treinamento
        with open(metadata_file, 'r') as f:
            training_results = json.load(f)
        
        print(f"✅ Carregando modelos de: {metadata_file}")
        print(f"📊 Modelos disponíveis: {list(training_results['model_paths'].keys())}")
        
        # Integração com sistema (quando disponível)
        if INTEGRATION_AVAILABLE:
            # trading_system = get_trading_system_instance()  # Sua instância do sistema
            # integration = TrainingIntegration(trading_system)
            # success = integration.update_models_from_training(training_results)
            print("🔗 Integração com sistema disponível (desabilitada para demonstração)")
        
        # Por enquanto, apenas confirmar carregamento
        print("✅ Metadata carregada com sucesso!")
        return training_results
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")
        return False

if __name__ == "__main__":
    result = load_latest_models()
    if result:
        print("\n📈 Métricas do Ensemble:")
        if 'ensemble_metrics' in result:
            for metric, value in result['ensemble_metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
    else:
        print("❌ Falha ao carregar modelos")