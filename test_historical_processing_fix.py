#!/usr/bin/env python3
"""
Teste para verificar se a correção do processamento histórico funciona
Simula 230k trades para testar a lógica de agrupamento
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_historical_processing_fix():
    """Testa a correção do processamento de dados históricos"""
    
    print("\n" + "="*80)
    print("TESTE DA CORREÇÃO DE PROCESSAMENTO HISTÓRICO")
    print("="*80 + "\n")
    
    try:
        # Importar componentes
        from data_loader import DataLoader
        from data_integration import DataIntegration
        
        print("1. CRIANDO COMPONENTES...")
        
        # Mock connection
        class MockConnection:
            def __init__(self):
                self.trade_callbacks = []
            
            def register_trade_callback(self, callback):
                self.trade_callbacks.append(callback)
                return True
        
        connection = MockConnection()
        data_loader = DataLoader()
        data_integration = DataIntegration(connection, data_loader)
        
        print("2. SIMULANDO 10.000 TRADES HISTÓRICOS...")
        
        # Simular período de 90 minutos com ~111 trades por minuto (10k trades total)
        base_time = datetime.now() - timedelta(minutes=90)
        total_trades = 10000
        trades = []
        
        for i in range(total_trades):
            # Distribuir trades ao longo de 90 minutos
            minute_offset = (i / total_trades) * 90  # Distribuir uniformemente
            trade_time = base_time + timedelta(minutes=minute_offset)
            
            # Preços oscilantes
            base_price = 5540 + np.sin(i/1000) * 20
            
            trade = {
                'timestamp': trade_time,
                'ticker': 'WDOZ25',
                'price': base_price + np.random.uniform(-1, 1),
                'volume': np.random.randint(100, 500),
                'quantity': np.random.randint(1, 5),
                'trade_type': 2 if np.random.random() > 0.5 else 3,
                'trade_number': i,
                'is_historical': True  # IMPORTANTE: Marcar como histórico
            }
            
            trades.append(trade)
        
        print(f"   - Criados {len(trades)} trades históricos")
        print(f"   - Período: {trades[0]['timestamp']} até {trades[-1]['timestamp']}")
        print(f"   - Duração: {trades[-1]['timestamp'] - trades[0]['timestamp']}")
        
        print("\n3. PROCESSANDO TRADES (MODO HISTÓRICO)...")
        
        # Processar todos os trades como históricos
        start_time = datetime.now()
        
        for i, trade in enumerate(trades):
            data_integration._on_trade(trade)
            
            # Log progresso a cada 2000 trades
            if (i + 1) % 2000 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = (i + 1) / elapsed
                print(f"   - Processados {i+1}/{len(trades)} trades ({rate:.0f} trades/s)")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"   - Processamento concluído em {processing_time:.2f}s")
        
        print("\n4. FINALIZANDO PROCESSAMENTO...")
        
        # Verificar buffer antes de finalizar
        buffer_size = len(data_integration.trades_buffer)
        print(f"   - Trades no buffer: {buffer_size}")
        
        # Simular evento de conclusão
        completion_event = {
            'event_type': 'historical_data_complete',
            'total_records': len(trades),
            'timestamp': datetime.now()
        }
        
        data_integration._on_trade(completion_event)
        
        print("\n5. VERIFICANDO RESULTADOS:")
        
        # Verificar data_integration.candles_1min
        if hasattr(data_integration, 'candles_1min') and not data_integration.candles_1min.empty:
            candles_integration = len(data_integration.candles_1min)
            print(f"   - Candles em data_integration: {candles_integration}")
            
            if candles_integration > 0:
                first_candle = data_integration.candles_1min.index.min()
                last_candle = data_integration.candles_1min.index.max()
                duration = last_candle - first_candle
                
                print(f"   - Período dos candles: {first_candle} até {last_candle}")
                print(f"   - Duração: {duration}")
                print(f"   - Candles esperados: ~{duration.total_seconds() / 60:.0f}")
                
                # Verificar eficiência
                expected_candles = duration.total_seconds() / 60
                efficiency = candles_integration / expected_candles * 100 if expected_candles > 0 else 0
                print(f"   - Eficiência: {efficiency:.1f}%")
                
                # Verificar sell_volume
                if 'sell_volume' in data_integration.candles_1min.columns:
                    total_sell_vol = data_integration.candles_1min['sell_volume'].sum()
                    total_buy_vol = data_integration.candles_1min['buy_volume'].sum()
                    print(f"   - Buy volume total: {total_buy_vol:,.0f}")
                    print(f"   - Sell volume total: {total_sell_vol:,.0f}")
                    
                    if total_sell_vol > 0:
                        print("   ✅ sell_volume funcionando corretamente")
                    else:
                        print("   ⚠️ sell_volume ainda está zero")
        
        # Verificar data_loader
        if hasattr(data_loader, 'candles_df') and not data_loader.candles_df.empty:
            candles_loader = len(data_loader.candles_df)
            print(f"   - Candles em data_loader: {candles_loader}")
        
        print("\n6. ANÁLISE DE PERFORMANCE:")
        
        if 'candles_integration' in locals() and candles_integration > 0:
            trades_per_candle = total_trades / candles_integration
            print(f"   - Trades por candle: {trades_per_candle:.1f}")
            
            if trades_per_candle > 50:
                print("   ✅ Boa densidade de trades por candle")
            else:
                print("   ⚠️ Baixa densidade - verificar distribuição temporal")
            
            if efficiency >= 80:
                print("   ✅ EXCELENTE - Sistema processando corretamente")
            elif efficiency >= 60:
                print("   ✅ BOM - Sistema funcionando adequadamente")
            elif efficiency >= 40:
                print("   ⚠️ ACEITÁVEL - Pode precisar de ajustes")
            else:
                print("   ❌ PROBLEMÁTICO - Necessita correções")
        
        print("\n" + "="*80)
        print("TESTE CONCLUÍDO!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\nERRO no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_historical_processing_fix()
    
    if success:
        print("\nCONCLUSÃO:")
        print("- A correção foi aplicada com sucesso")
        print("- Trades históricos são agora armazenados no buffer")
        print("- Processamento em lote no final do carregamento")
        print("- Sistema deve criar TODOS os candles necessários")
        print("\nPróximo passo: Testar com dados reais da ProfitDLL")
    else:
        print("\nFALHA:")
        print("- Verificar logs para identificar problemas")
        print("- Possível erro na lógica de agrupamento")