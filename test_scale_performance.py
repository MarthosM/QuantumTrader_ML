"""
Teste de Performance em Escala - 7200 candles (5 dias)
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_structure import TradingDataStructure
from src.data_loader import DataLoader
from src.feature_engine import FeatureEngine

def test_full_scale():
    print("=" * 80)
    print("TESTE DE ESCALA COMPLETA - 7200 CANDLES (5 DIAS)")
    print("=" * 80)
    
    os.environ['TRADING_ENV'] = 'development'
    
    # Etapa 1: Carregar dados
    print("1. Carregando 7200 candles...")
    start_time = time.time()
    
    data_structure = TradingDataStructure()
    data_structure.initialize_structure()
    data_loader = DataLoader()
    
    df = data_loader.create_sample_data(7200)
    data_structure.update_candles(df)
    
    load_time = time.time() - start_time
    print(f"   Dados carregados: {load_time:.3f}s")
    print(f"   Candles na estrutura: {len(data_structure.candles)}")
    
    # Etapa 2: Calcular features SEM avançadas (mais rápido)
    print("2. Calculando features básicas...")
    start_time = time.time()
    
    feature_engine = FeatureEngine()
    result_basic = feature_engine.calculate(
        data=data_structure,
        force_recalculate=True,
        use_advanced=False  # SEM features avançadas
    )
    
    basic_time = time.time() - start_time
    print(f"   Features básicas: {basic_time:.3f}s")
    
    if 'features' in result_basic:
        features_df = result_basic['features']
        print(f"   Features calculadas: {len(features_df.columns)} colunas")
        print(f"   Shape: {features_df.shape}")
        
        # Salvar resultado básico
        features_df.to_csv("features_basic_7200.csv")
        print("   Salvo em: features_basic_7200.csv")
    
    # Etapa 3: Calcular COM features avançadas (mais lento)
    print("3. Calculando features completas...")
    start_time = time.time()
    
    result_full = feature_engine.calculate(
        data=data_structure,
        force_recalculate=True,
        use_advanced=True  # COM features avançadas
    )
    
    full_time = time.time() - start_time
    print(f"   Features completas: {full_time:.3f}s")
    
    if 'features' in result_full:
        features_df = result_full['features']
        print(f"   Features calculadas: {len(features_df.columns)} colunas")
        print(f"   Shape: {features_df.shape}")
        
        # Salvar resultado completo
        features_df.to_csv("features_full_7200.csv")
        print("   Salvo em: features_full_7200.csv")
    
    # Resumo
    total_time = load_time + basic_time + full_time
    print(f"\n4. Resumo de Performance:")
    print(f"   Carregamento: {load_time:.3f}s ({7200/load_time:.0f} candles/s)")
    print(f"   Features básicas: {basic_time:.3f}s ({7200/basic_time:.0f} candles/s)")
    print(f"   Features completas: {full_time:.3f}s ({7200/full_time:.0f} candles/s)")
    print(f"   TOTAL: {total_time:.3f}s")
    
    if total_time > 60:
        print("   ALERTA: Tempo total > 1 minuto - otimização necessária!")
    elif total_time > 30:
        print("   AVISO: Tempo total > 30 segundos - pode melhorar")
    else:
        print("   OK: Performance aceitável")

if __name__ == "__main__":
    test_full_scale()