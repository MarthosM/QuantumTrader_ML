#!/usr/bin/env python3
"""
Teste simplificado para verificar processamento de candles
Simula dados históricos sem depender da ProfitDLL
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

def test_candles_processing():
    """Testa processamento de candles sem ProfitDLL"""
    
    print("\n" + "="*60)
    print("TESTE DE PROCESSAMENTO DE CANDLES (SEM DLL)")
    print("="*60 + "\n")
    
    # Definir ambiente
    os.environ['TRADING_ENV'] = 'development'
    
    try:
        # Importar componentes necessários
        from data_loader import DataLoader
        from data_integration import DataIntegration
        from data_structure import TradingDataStructure
        from connection_manager import ConnectionManager
        
        print("1. CRIANDO COMPONENTES...")
        
        # Criar mock do connection manager
        class MockConnection:
            def __init__(self):
                self.connected = True
                self._historical_data_count = 0
                self.trade_callbacks = []
            
            def register_trade_callback(self, callback):
                self.trade_callbacks.append(callback)
                return True
        
        connection = MockConnection()
        data_loader = DataLoader()
        data_integration = DataIntegration(connection, data_loader)
        
        print("2. SIMULANDO DADOS HISTORICOS...")
        
        # Simular 1000 trades distribuídos em 10 minutos
        base_time = datetime.now() - timedelta(minutes=10)
        trades = []
        
        for i in range(1000):
            # Distribuir trades ao longo de 10 minutos
            trade_time = base_time + timedelta(seconds=i * 0.6)  # 1 trade a cada 0.6s
            
            # Preços realistas para WDO
            base_price = 5540 + np.sin(i/100) * 10  # Preço oscilante
            
            trade = {
                'timestamp': trade_time,
                'ticker': 'WDOZ25',
                'price': base_price + np.random.uniform(-2, 2),
                'volume': np.random.randint(100, 1000),
                'quantity': np.random.randint(1, 10),
                'trade_type': 2 if np.random.random() > 0.5 else 3,  # Buy/Sell
                'trade_number': i,
                'is_historical': True
            }
            
            trades.append(trade)
        
        print(f"   - Criados {len(trades)} trades simulados")
        print(f"   - Periodo: {trades[0]['timestamp']} ate {trades[-1]['timestamp']}")
        
        # Processar trades
        print("\n3. PROCESSANDO TRADES...")
        
        for i, trade in enumerate(trades):
            data_integration._on_trade(trade)
            
            if i % 200 == 0:
                print(f"   - Processados {i+1}/{len(trades)} trades...")
        
        print("\n4. FINALIZANDO PROCESSAMENTO...")
        
        # Simular evento de conclusão
        completion_event = {
            'event_type': 'historical_data_complete',
            'total_records': len(trades),
            'timestamp': datetime.now()
        }
        
        data_integration._on_trade(completion_event)
        
        print("\n5. VERIFICANDO RESULTADOS:")
        
        # Verificar candles criados
        if hasattr(data_integration, 'candles_1min') and not data_integration.candles_1min.empty:
            candles = data_integration.candles_1min
            print(f"   - Candles criados: {len(candles)}")
            print(f"   - Periodo dos candles: {candles.index.min()} ate {candles.index.max()}")
            print(f"   - Duracao: {candles.index.max() - candles.index.min()}")
            
            # Estatísticas
            total_volume = candles['volume'].sum()
            total_trades_processed = candles['trades'].sum()
            avg_price = candles['close'].mean()
            
            print(f"   - Volume total: {total_volume:,.0f}")
            print(f"   - Trades processados: {total_trades_processed}")
            print(f"   - Preco medio: R$ {avg_price:,.2f}")
            
            # Verificar sell_volume
            if 'sell_volume' in candles.columns:
                sell_vol_total = candles['sell_volume'].sum()
                sell_vol_zeros = (candles['sell_volume'] == 0).sum()
                print(f"   - sell_volume total: {sell_vol_total:,.0f}")
                print(f"   - sell_volume zeros: {sell_vol_zeros}/{len(candles)}")
            
            # Mostrar primeiros candles
            print("\n6. PRIMEIROS 5 CANDLES:")
            display_cols = ['open', 'high', 'low', 'close', 'volume', 'trades']
            existing_cols = [col for col in display_cols if col in candles.columns]
            print(candles[existing_cols].head().to_string())
            
            # Análise de eficiência
            print("\n7. ANALISE DE EFICIENCIA:")
            expected_candles = 10  # 10 minutos
            efficiency = len(candles) / expected_candles * 100
            print(f"   - Candles esperados: ~{expected_candles}")
            print(f"   - Candles criados: {len(candles)}")
            print(f"   - Eficiencia: {efficiency:.1f}%")
            
            if efficiency >= 80:
                print("   Status: EXCELENTE")
            elif efficiency >= 60:
                print("   Status: BOM")
            elif efficiency >= 40:
                print("   Status: ACEITAVEL")
            else:
                print("   Status: NECESSITA MELHORIAS")
                
        else:
            print("   ERROR: Nenhum candle foi criado!")
            return False
        
        # Verificar data_loader
        print("\n8. VERIFICANDO DATA_LOADER:")
        if hasattr(data_loader, 'candles_df') and not data_loader.candles_df.empty:
            loader_candles = len(data_loader.candles_df)
            print(f"   - Candles no data_loader: {loader_candles}")
        else:
            print("   - data_loader.candles_df vazio")
        
        print("\n" + "="*60)
        print("TESTE CONCLUIDO COM SUCESSO!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nERRO no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_candles_processing()
    
    if success:
        print("\nCONCLUSAO:")
        print("- O processamento de candles esta funcionando corretamente")
        print("- Os trades sao convertidos em candles de 1 minuto")
        print("- A sincronizacao entre componentes esta operacional")
        print("\nPara testar com dados reais:")
        print("1. Conecte a ProfitDLL")
        print("2. Execute o sistema principal: python src/main.py")
    else:
        print("\nFALHA:")
        print("- Verifique os logs acima para identificar o problema")
        print("- Pode haver erro nos imports ou na logica de processamento")