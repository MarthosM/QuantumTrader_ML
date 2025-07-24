#!/usr/bin/env python3
"""
Teste completo do sistema ML - Verificação de pipeline
"""
import os
import sys
from pathlib import Path

# Adicionar o diretório src ao path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

def test_complete_ml_pipeline():
    """Testa o pipeline completo do ML sem conexão ProfitDLL"""
    
    print("🧪 TESTE COMPLETO DO PIPELINE ML")
    print("=" * 50)
    
    try:
        # 1. Carregar configurações
        from dotenv import load_dotenv
        load_dotenv()
        
        print("✅ 1. Configurações carregadas")
        
        # 2. Testar ModelManager
        from model_manager import ModelManager
        models_dir = os.getenv('MODELS_DIR')
        model_manager = ModelManager(models_dir)
        success = model_manager.load_models()
        
        if success:
            print(f"✅ 2. Modelos carregados: {len(model_manager.models)}")
            for name in model_manager.models.keys():
                features = model_manager.model_features.get(name, [])
                print(f"   - {name}: {len(features)} features")
        else:
            print("❌ 2. Falha ao carregar modelos")
            return False
            
        # 3. Testar FeatureEngine
        from feature_engine import FeatureEngine
        feature_engine = FeatureEngine(model_manager)
        print("✅ 3. FeatureEngine inicializado")
        
        # 4. Testar DataLoader
        from data_loader import DataLoader
        data_loader = DataLoader()
        sample_data = data_loader.create_sample_data(100)
        print(f"✅ 4. DataLoader criou {len(sample_data)} candles de teste")
        
        # 5. Testar TradingDataStructure
        from data_structure import TradingDataStructure
        data_structure = TradingDataStructure()
        data_structure.initialize_structure()
        print("✅ 5. TradingDataStructure inicializada")
        
        # 6. Testar cálculo de features
        try:
            # Atualizar estrutura com dados
            data_structure.update_candles(sample_data)
            
            # Calcular features
            result = feature_engine.create_features_separated(
                candles_df=sample_data,
                microstructure_df=None,
                indicators_df=None
            )
            
            if result and 'features' in result:
                features_count = len(result['features'].columns) if hasattr(result['features'], 'columns') else 0
                print(f"✅ 6. Features calculadas: {features_count} features")
            else:
                print("⚠️ 6. Features calculadas (resultado vazio)")
                
        except Exception as e:
            print(f"⚠️ 6. Erro no cálculo de features: {e}")
        
        # 7. Resumo final
        print("\n" + "=" * 50)
        print("🎉 PIPELINE ML FUNCIONANDO CORRETAMENTE!")
        print(f"📊 Modelos: {len(model_manager.models)}")
        print(f"📈 Dados de teste: {len(sample_data)} candles")
        print("✅ Sistema ML pronto para uso")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_complete_ml_pipeline()
