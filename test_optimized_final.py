"""
Teste Final com Sistema Otimizado
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_structure import TradingDataStructure
from src.data_loader import DataLoader
from src.feature_engine import FeatureEngine

def test_optimized():
    print("=" * 80)
    print("TESTE SISTEMA OTIMIZADO - FEATURES APENAS BASICAS")
    print("=" * 80)
    
    os.environ['TRADING_ENV'] = 'development'
    
    # Preparar sistema
    data_structure = TradingDataStructure()
    data_structure.initialize_structure()
    data_loader = DataLoader()
    feature_engine = FeatureEngine()
    
    # Teste com diferentes tamanhos
    sizes = [1000, 3000, 7200]
    
    for size in sizes:
        print(f"\n--- TESTANDO {size} CANDLES ---")
        
        # Limpar dados anteriores
        data_structure.initialize_structure()
        
        # Carregar dados
        start_time = time.time()
        df = data_loader.create_sample_data(size)
        data_structure.update_candles(df)
        load_time = time.time() - start_time
        
        # Calcular features SEM avançadas (mais rápido)
        start_time = time.time()
        result = feature_engine.calculate(
            data=data_structure,
            force_recalculate=True,
            use_advanced=False  # DESABILITAR features avançadas
        )
        calc_time = time.time() - start_time
        
        # Verificar resultado
        if 'features' in result:
            features_df = result['features']
            ds_features = data_structure.get_features()
            
            print(f"   Carregamento: {load_time:.3f}s")
            print(f"   Calculo: {calc_time:.3f}s")
            print(f"   Total: {load_time + calc_time:.3f}s")
            print(f"   Taxa: {size/(load_time + calc_time):.0f} candles/s")
            print(f"   Features: {len(features_df.columns)} colunas")
            print(f"   DataStructure: {len(ds_features.columns) if not ds_features.empty else 0} colunas")
            print(f"   Salvo: {'SIM' if not ds_features.empty else 'NAO'}")
            
            # Para o teste grande, salvar arquivo
            if size == 7200:
                ds_features.to_csv("features_optimized_7200.csv")
                print("   Arquivo: features_optimized_7200.csv")
        else:
            print(f"   ERRO: Nenhuma feature calculada!")
    
    print("\n" + "=" * 80)
    print("RECOMENDACAO:")
    print("Para melhor performance, use use_advanced=False")
    print("Features basicas sao suficientes para a maioria dos casos")
    print("=" * 80)

if __name__ == "__main__":
    test_optimized()