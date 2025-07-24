"""
Script de Diagnóstico de Performance Simplificado
"""

import os
import sys
import time
import logging
from datetime import datetime

# Adicionar o diretório src ao path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_structure import TradingDataStructure
from src.data_loader import DataLoader
from src.feature_engine import FeatureEngine

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PerformanceTest')

def test_feature_calculation():
    """Testa cálculo de features e salva resultado"""
    print("=" * 80)
    print("TESTE DE CALCULO DE FEATURES")
    print("=" * 80)
    
    # Definir ambiente como desenvolvimento
    os.environ['TRADING_ENV'] = 'development'
    
    # Preparar dados
    print("1. Preparando dados...")
    data_structure = TradingDataStructure()
    data_structure.initialize_structure()
    
    data_loader = DataLoader()
    
    print("2. Carregando 1000 candles...")
    start_time = time.time()
    df = data_loader.create_sample_data(1000)
    load_time = time.time() - start_time
    print(f"   Tempo de carregamento: {load_time:.3f}s")
    
    print("3. Atualizando TradingDataStructure...")
    start_time = time.time()
    data_structure.update_candles(df)
    update_time = time.time() - start_time
    print(f"   Tempo de atualizacao: {update_time:.3f}s")
    print(f"   Candles na estrutura: {len(data_structure.candles)}")
    
    print("4. Criando FeatureEngine...")
    feature_engine = FeatureEngine()
    
    print("5. Calculando features...")
    start_time = time.time()
    
    try:
        result = feature_engine.calculate(
            data=data_structure,
            force_recalculate=True,
            use_advanced=False
        )
        
        calc_time = time.time() - start_time
        print(f"   Tempo de calculo: {calc_time:.3f}s")
        
        # Verificar resultados
        print("\n6. Analisando resultados...")
        
        if 'features' in result:
            features_df = result['features']
            print(f"   Features calculadas: {len(features_df.columns)} colunas")
            print(f"   Shape: {features_df.shape}")
            print(f"   Primeiras colunas: {list(features_df.columns[:5])}")
            
            # Verificar se foi salvo na data_structure
            ds_features = data_structure.get_features()
            print(f"   Features na DataStructure: {len(ds_features.columns) if not ds_features.empty else 0}")
            
            if not ds_features.empty:
                print("   SUCESSO: Features salvas na DataStructure!")
                
                # Salvar em arquivo para inspeção
                output_file = "features_output.csv"
                ds_features.to_csv(output_file)
                print(f"   Features salvas em: {output_file}")
                
                # Mostrar estatísticas
                print(f"   Valores NaN: {ds_features.isna().sum().sum()}")
                print(f"   Linhas completas: {ds_features.dropna().shape[0]}")
                
            else:
                print("   PROBLEMA: Features NAO foram salvas na DataStructure!")
                
        else:
            print("   ERRO: Nenhuma feature foi calculada!")
            
        # Verificar outros resultados
        for key, value in result.items():
            if hasattr(value, 'shape'):
                print(f"   {key}: {value.shape}")
            else:
                print(f"   {key}: {type(value)}")
                
    except Exception as e:
        print(f"   ERRO no calculo: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "=" * 80)
    print("TESTE CONCLUIDO")
    print("=" * 80)

if __name__ == "__main__":
    test_feature_calculation()