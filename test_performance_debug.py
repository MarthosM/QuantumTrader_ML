"""
Script de Diagnóstico de Performance
Identifica gargalos no carregamento de dados e cálculo de features
"""

import os
import sys
import time
import logging
from datetime import datetime
import pandas as pd

# Adicionar o diretório src ao path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_structure import TradingDataStructure
from src.data_loader import DataLoader
from src.feature_engine import FeatureEngine
from src.model_manager import ModelManager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PerformanceDebug')

class PerformanceDebugger:
    def __init__(self):
        self.logger = logger
        # Definir ambiente como desenvolvimento
        os.environ['TRADING_ENV'] = 'development'
        
    def test_data_loading_speed(self):
        """Testa velocidade de carregamento de dados"""
        print("=" * 80)
        print("TESTE 1: VELOCIDADE DE CARREGAMENTO DE DADOS")
        print("=" * 80)
        
        data_loader = DataLoader()
        
        # Teste com diferentes tamanhos
        sizes = [100, 1000, 7200]  # 7200 = 5 dias
        
        for size in sizes:
            print(f"\nTestando carregamento de {size} candles...")
            
            start_time = time.time()
            df = data_loader.create_sample_data(size)
            load_time = time.time() - start_time
            
            print(f"   Tempo de carregamento: {load_time:.3f}s")
            print(f"   Taxa: {size/load_time:.0f} candles/s")
            print(f"   Tamanho DataFrame: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            print(f"   Shape: {df.shape}")
            
    def test_data_structure_update_speed(self):
        """Testa velocidade de atualização do TradingDataStructure"""
        print("\n" + "=" * 80)
        print("🔍 TESTE 2: VELOCIDADE DE ATUALIZAÇÃO DA DATA STRUCTURE")
        print("=" * 80)
        
        data_structure = TradingDataStructure()
        data_structure.initialize_structure()
        
        data_loader = DataLoader()
        
        # Carregar dados de teste
        print("📊 Carregando 1000 candles...")
        start_time = time.time()
        df = data_loader.create_sample_data(1000)
        load_time = time.time() - start_time
        print(f"   ⏱️ Carregamento: {load_time:.3f}s")
        
        # Atualizar TradingDataStructure
        print("🔄 Atualizando TradingDataStructure...")
        start_time = time.time()
        data_structure.update_candles(df)
        update_time = time.time() - start_time
        print(f"   ⏱️ Atualização: {update_time:.3f}s")
        
        # Verificar dados
        print(f"   ✅ Candles na estrutura: {len(data_structure.candles)}")
        
    def test_feature_calculation_speed(self):
        """Testa velocidade de cálculo de features"""
        print("\n" + "=" * 80)
        print("🔍 TESTE 3: VELOCIDADE DE CÁLCULO DE FEATURES")
        print("=" * 80)
        
        # Preparar dados
        data_structure = TradingDataStructure()
        data_structure.initialize_structure()
        
        data_loader = DataLoader()
        df = data_loader.create_sample_data(1000)
        data_structure.update_candles(df)
        
        print(f"📊 Preparados {len(df)} candles para teste")
        
        # Criar FeatureEngine
        feature_engine = FeatureEngine()
        
        print("⚙️ Calculando features...")
        start_time = time.time()
        
        try:
            result = feature_engine.calculate(
                data=data_structure,
                force_recalculate=True,
                use_advanced=False  # Desabilitar features avançadas para teste
            )
            
            calc_time = time.time() - start_time
            
            print(f"   ⏱️ Tempo total: {calc_time:.3f}s")
            print(f"   📈 Taxa: {len(df)/calc_time:.0f} candles/s")
            
            # Verificar resultados
            if 'features' in result:
                features_df = result['features']
                print(f"   🎯 Features calculadas: {len(features_df.columns)} colunas")
                print(f"   📏 Shape: {features_df.shape}")
                print(f"   💾 Tamanho: {features_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
                # Verificar se foi salvo na data_structure
                ds_features = data_structure.get_features()
                print(f"   📊 Features na DataStructure: {len(ds_features.columns) if not ds_features.empty else 0}")
                
                if not ds_features.empty:
                    print("   ✅ Features SALVAS na DataStructure com sucesso!")
                else:
                    print("   ❌ Features NÃO foram salvas na DataStructure!")
                    
            else:
                print("   ❌ Nenhuma feature foi calculada!")
                
        except Exception as e:
            print(f"   ❌ ERRO no cálculo: {e}")
            import traceback
            traceback.print_exc()
            
    def test_incremental_performance(self):
        """Testa performance incremental com diferentes tamanhos"""
        print("\n" + "=" * 80)
        print("🔍 TESTE 4: PERFORMANCE INCREMENTAL")
        print("=" * 80)
        
        data_structure = TradingDataStructure()
        data_structure.initialize_structure()
        data_loader = DataLoader()
        feature_engine = FeatureEngine()
        
        sizes = [100, 500, 1000, 2000, 5000]
        
        for size in sizes:
            print(f"\n📊 Testando com {size} candles...")
            
            # Carregar dados
            start_time = time.time()
            df = data_loader.create_sample_data(size)
            data_structure.update_candles(df)
            data_prep_time = time.time() - start_time
            
            # Calcular features
            start_time = time.time()
            try:
                result = feature_engine.calculate(
                    data=data_structure,
                    force_recalculate=True,
                    use_advanced=False
                )
                calc_time = time.time() - start_time
                
                print(f"   📈 Prep dados: {data_prep_time:.3f}s")
                print(f"   ⚙️ Calc features: {calc_time:.3f}s")
                print(f"   🏃 Total: {data_prep_time + calc_time:.3f}s")
                print(f"   📊 Taxa: {size/(data_prep_time + calc_time):.0f} candles/s")
                
                if 'features' in result:
                    features_shape = result['features'].shape
                    print(f"   🎯 Features: {features_shape[1]} cols x {features_shape[0]} rows")
                
            except Exception as e:
                print(f"   ❌ ERRO: {e}")
    
    def run_full_diagnosis(self):
        """Executa diagnóstico completo"""
        print("INICIANDO DIAGNOSTICO DE PERFORMANCE")
        print("Hora de inicio:", datetime.now().strftime("%H:%M:%S"))
        
        try:
            self.test_data_loading_speed()
            self.test_data_structure_update_speed()
            self.test_feature_calculation_speed()
            self.test_incremental_performance()
            
            print("\n" + "=" * 80)
            print("DIAGNOSTICO CONCLUIDO COM SUCESSO")
            print("Hora de termino:", datetime.now().strftime("%H:%M:%S"))
            print("=" * 80)
            
        except Exception as e:
            print(f"\nERRO no diagnostico: {e}")
            import traceback
            traceback.print_exc()

def main():
    debugger = PerformanceDebugger()
    debugger.run_full_diagnosis()

if __name__ == "__main__":
    main()