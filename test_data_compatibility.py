"""
Script para testar compatibilidade entre sistema de coleta e TradingDataStructure
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from src.data.profitdll_real_collector import ProfitDLLRealCollector
from src.data_structure import TradingDataStructure

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_data_flow():
    """Testa o fluxo de dados do coletor para a estrutura"""
    
    print("="*60)
    print("TESTE DE COMPATIBILIDADE DE DADOS")
    print("="*60)
    
    # 1. Criar instâncias
    collector = ProfitDLLRealCollector()
    data_structure = TradingDataStructure()
    
    # 2. Inicializar
    print("\n1. Inicializando componentes...")
    
    if not collector.initialize():
        print("[ERRO] Falha ao inicializar coletor")
        return
    
    data_structure.initialize_structure()
    print("[OK] Componentes inicializados")
    
    # 3. Conectar
    print("\n2. Conectando ao ProfitChart...")
    
    if not collector.connect_and_login():
        print("[ERRO] Falha na conexão")
        
        # Testar com dados simulados
        print("\n3. Testando com dados simulados...")
        test_with_simulated_data(data_structure)
        return
    
    print("[OK] Conectado com sucesso")
    
    # 4. Coletar dados reais
    print("\n3. Coletando dados reais...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(minutes=15)  # Últimos 15 minutos
    ticker = "WDOQ25"
    
    try:
        data = collector.collect_historical_data(ticker, start_date, end_date)
        
        if data:
            print(f"[OK] Dados coletados:")
            for key, df in data.items():
                if not df.empty:
                    print(f"  - {key}: {df.shape}")
            
            # 5. Testar compatibilidade
            print("\n4. Testando compatibilidade...")
            test_compatibility(data, data_structure)
            
        else:
            print("[AVISO] Nenhum dado coletado")
            test_with_simulated_data(data_structure)
            
    except Exception as e:
        print(f"[ERRO] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        collector.disconnect()


def test_compatibility(collected_data: dict, data_structure: TradingDataStructure):
    """Testa se os dados coletados são compatíveis com a estrutura"""
    
    print("\n--- TESTE DE COMPATIBILIDADE ---")
    
    # 1. Verificar candles
    if 'candles' in collected_data:
        candles = collected_data['candles']
        print(f"\nCandles coletados: {len(candles)} registros")
        print(f"Colunas: {list(candles.columns)}")
        
        # Verificar colunas esperadas
        expected_candle_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in expected_candle_cols if col not in candles.columns]
        
        if missing_cols:
            print(f"[AVISO] Colunas faltando em candles: {missing_cols}")
        else:
            print("[OK] Todas colunas de candles presentes")
            
        # Tentar atualizar
        if data_structure.update_candles(candles):
            print("[OK] Candles atualizados na estrutura")
        else:
            print("[ERRO] Falha ao atualizar candles")
    
    # 2. Verificar microestrutura
    if 'microstructure' in collected_data:
        micro = collected_data['microstructure']
        print(f"\nMicroestrutura coletada: {len(micro)} registros")
        print(f"Colunas: {list(micro.columns)}")
        
        # Verificar colunas esperadas
        expected_micro_cols = ['buy_volume', 'sell_volume', 'buy_pressure', 'sell_pressure']
        missing_cols = [col for col in expected_micro_cols if col not in micro.columns]
        
        if missing_cols:
            print(f"[AVISO] Colunas faltando em microestrutura: {missing_cols}")
        else:
            print("[OK] Colunas principais de microestrutura presentes")
            
        # Tentar atualizar
        if data_structure.update_microstructure(micro):
            print("[OK] Microestrutura atualizada na estrutura")
        else:
            print("[ERRO] Falha ao atualizar microestrutura")
    
    # 3. Verificar índices temporais
    print("\n--- VERIFICAÇÃO DE ÍNDICES ---")
    
    all_aligned = True
    indices = []
    
    for key, df in collected_data.items():
        if not df.empty and hasattr(df, 'index'):
            indices.append((key, df.index[0], df.index[-1]))
            
            # Verificar se é datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"[AVISO] {key} não tem DatetimeIndex")
                all_aligned = False
    
    if indices:
        print("\nPeríodos dos dados:")
        for key, start, end in indices:
            print(f"  {key}: {start} até {end}")
    
    if all_aligned:
        print("[OK] Todos os dados têm índices temporais")
    
    # 4. Testar recuperação de dados
    print("\n--- TESTE DE RECUPERAÇÃO ---")
    
    candles_retrieved = data_structure.get_candles()
    if not candles_retrieved.empty:
        print(f"[OK] Candles recuperados: {len(candles_retrieved)} registros")
        
        # Verificar janela
        window = data_structure.get_candles_window(5)
        print(f"[OK] Janela de 5 candles: {len(window)} registros")
        
        # Verificar último candle
        latest = data_structure.get_latest_candle()
        if latest is not None:
            print(f"[OK] Último candle: Close={latest.get('close', 'N/A')}")
    
    # 5. Verificar metadados
    print("\n--- METADADOS ---")
    print(f"Último update: {data_structure.last_update}")
    print(f"Último preço: {data_structure.last_price}")
    print(f"Último volume: {data_structure.last_volume}")
    print(f"Total candles: {data_structure.data_quality['total_candles']}")


def test_with_simulated_data(data_structure: TradingDataStructure):
    """Testa com dados simulados para verificar estrutura"""
    
    print("\n--- TESTE COM DADOS SIMULADOS ---")
    
    # Criar dados simulados
    timestamps = pd.date_range(
        end=datetime.now(), 
        periods=10, 
        freq='1min'
    )
    
    # Candles simulados
    candles = pd.DataFrame(index=timestamps)
    candles['open'] = 5000 + np.random.randn(10) * 10
    candles['high'] = candles['open'] + np.random.uniform(0, 10, 10)
    candles['low'] = candles['open'] - np.random.uniform(0, 10, 10)
    candles['close'] = candles['open'] + np.random.randn(10) * 5
    candles['volume'] = np.random.uniform(100000, 500000, 10)
    candles['quantidade'] = candles['volume'] / 100
    
    # Microestrutura simulada
    micro = pd.DataFrame(index=timestamps)
    micro['buy_volume'] = candles['volume'] * 0.55
    micro['sell_volume'] = candles['volume'] * 0.45
    micro['buy_trades'] = 100
    micro['sell_trades'] = 90
    micro['buy_pressure'] = 0.55
    micro['sell_pressure'] = 0.45
    micro['volume_imbalance'] = micro['buy_volume'] - micro['sell_volume']
    micro['trade_imbalance'] = 10
    micro['buy_ratio'] = 0.55
    
    # Orderbook simulado
    orderbook = pd.DataFrame(index=timestamps)
    orderbook['bid'] = candles['close'] - 0.5
    orderbook['ask'] = candles['close'] + 0.5
    orderbook['spread'] = 1.0
    orderbook['bid_volume'] = 50000
    orderbook['ask_volume'] = 50000
    orderbook['bid_count'] = 10
    orderbook['ask_count'] = 10
    orderbook['depth_imbalance'] = 0.0
    
    # Testar atualizações
    print("\nAtualizando estrutura com dados simulados...")
    
    if data_structure.update_candles(candles):
        print("[OK] Candles atualizados")
    
    if data_structure.update_microstructure(micro):
        print("[OK] Microestrutura atualizada")
    
    if data_structure.update_orderbook(orderbook):
        print("[OK] Orderbook atualizado")
    
    # Verificar estrutura
    print(f"\nEstrutura após atualização:")
    print(f"  Candles: {len(data_structure.candles)} registros")
    print(f"  Microestrutura: {len(data_structure.microstructure)} registros")
    print(f"  Orderbook: {len(data_structure.orderbook)} registros")
    
    # Verificar qualidade
    quality = data_structure.check_data_quality()
    print(f"\nQualidade dos dados:")
    print(f"  Score: {quality.get('overall_score', 'N/A')}")


if __name__ == "__main__":
    test_data_flow()