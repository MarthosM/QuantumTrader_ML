#!/usr/bin/env python3
"""
Teste de carregamento de modelos
"""
import os
import sys
from pathlib import Path

# Adicionar o diretório src ao path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

from model_manager import ModelManager

def test_model_loading():
    """Testa o carregamento dos modelos"""
    
    # Carregar do .env
    from dotenv import load_dotenv
    load_dotenv()
    
    models_dir = os.getenv('MODELS_DIR')
    print(f"📁 Diretório dos modelos: {models_dir}")
    
    if not os.path.exists(models_dir):
        print(f"❌ Diretório não existe: {models_dir}")
        return False
        
    # Listar arquivos
    files = os.listdir(models_dir)
    print(f"📋 Arquivos encontrados: {files}")
    
    # Tentar carregar modelos
    print("\n🤖 Carregando modelos...")
    model_manager = ModelManager(models_dir)
    success = model_manager.load_models()
    
    if success:
        print(f"✅ Modelos carregados com sucesso: {len(model_manager.models)}")
        for name in model_manager.models.keys():
            features = model_manager.model_features.get(name, [])
            print(f"  - {name}: {len(features)} features")
    else:
        print("❌ Falha ao carregar modelos")
        
    return success

if __name__ == "__main__":
    test_model_loading()
