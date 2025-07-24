#!/usr/bin/env python3
"""
Teste específico do método create_or_update_candle
"""
import sys
from pathlib import Path

# Adicionar o diretório src ao path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

from data_loader import DataLoader
from datetime import datetime
import pandas as pd

def test_create_or_update_candle():
    """Testa o método create_or_update_candle"""
    
    print("🧪 TESTE DO MÉTODO create_or_update_candle")
    print("=" * 50)
    
    data_loader = DataLoader()
    
    # Simular trades em momentos diferentes
    trades = [
        {'timestamp': '2025-07-22 13:50:00', 'price': 5600.0, 'volume': 1000, 'side': 'buy'},
        {'timestamp': '2025-07-22 13:50:15', 'price': 5601.0, 'volume': 500, 'side': 'buy'},
        {'timestamp': '2025-07-22 13:50:30', 'price': 5599.0, 'volume': 800, 'side': 'sell'},
        {'timestamp': '2025-07-22 13:51:00', 'price': 5602.0, 'volume': 1200, 'side': 'buy'},  # Novo minuto
        {'timestamp': '2025-07-22 13:51:30', 'price': 5603.0, 'volume': 900, 'side': 'buy'},
    ]
    
    completed_candles = []
    
    print("📊 Processando trades...")
    for i, trade in enumerate(trades):
        print(f"  Trade {i+1}: {trade['timestamp']} - Preço: {trade['price']} - Volume: {trade['volume']}")
        
        result = data_loader.create_or_update_candle(trade)
        
        if result is not None:
            completed_candles.append(result)
            print(f"    ✅ Candle completo: {len(result)} registros")
        else:
            print(f"    ⏳ Atualizando candle atual...")
    
    print(f"\n📈 Resultado: {len(completed_candles)} candles completos")
    
    for i, candle in enumerate(completed_candles):
        print(f"\nCandle {i+1}:")
        print(candle.to_string())
        
    print("\n✅ Teste do create_or_update_candle concluído!")
    return True

if __name__ == "__main__":
    test_create_or_update_candle()
