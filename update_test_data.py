"""
Atualizar dados de teste para período atual
"""

import pandas as pd
from datetime import datetime, timedelta
import os

def update_test_data():
    print("=" * 80)
    print("ATUALIZANDO DADOS DE TESTE PARA PERÍODO ATUAL")
    print("=" * 80)
    
    # Arquivo de teste
    test_file = "tests/data/WDOQ25_test_data.csv"
    
    # Ler dados existentes
    if os.path.exists(test_file):
        df = pd.read_csv(test_file, parse_dates=['timestamp'], index_col='timestamp')
        print(f"1. Dados existentes: {len(df)} candles")
        print(f"   Período: {df.index[0]} até {df.index[-1]}")
        
        # Calcular novo período (últimos 2 dias)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        print(f"\n2. Novo período desejado:")
        print(f"   De: {start_date}")
        print(f"   Até: {end_date}")
        
        # Criar novos dados baseados nos existentes
        # Usar padrões dos dados originais
        sample_data = df.head(100)  # Usar primeiros 100 candles como base
        
        # Gerar timestamps para novo período
        new_timestamps = pd.date_range(
            start=start_date,
            end=end_date,
            freq='1min'
        )
        
        print(f"   Timestamps gerados: {len(new_timestamps)}")
        
        # Criar novos dados repetindo padrões
        new_data = []
        base_price = 130000
        
        for i, timestamp in enumerate(new_timestamps):
            # Usar variação baseada nos dados originais
            row_idx = i % len(sample_data)
            base_row = sample_data.iloc[row_idx]
            
            # Calcular variação baseada no timestamp
            variation = (timestamp.hour * 10) + (timestamp.minute * 2)
            
            new_data.append({
                'timestamp': timestamp,
                'open': base_price + variation,
                'high': base_price + variation + 50,
                'low': base_price + variation - 30, 
                'close': base_price + variation + 20,
                'volume': base_row['volume']
            })
        
        # Criar novo DataFrame
        new_df = pd.DataFrame(new_data)
        new_df.set_index('timestamp', inplace=True)
        
        print(f"\n3. Novos dados gerados: {len(new_df)} candles")
        print(f"   Primeiro: {new_df.index[0]}")
        print(f"   Último: {new_df.index[-1]}")
        
        # Salvar arquivo atualizado
        new_df.to_csv(test_file)
        print(f"\n4. Arquivo atualizado: {test_file}")
        
        # Verificar se dados estão no período de hoje
        today = datetime.now().date()
        todays_data = new_df[new_df.index.date == today]
        print(f"   Dados de hoje: {len(todays_data)} candles")
        
        if len(todays_data) > 0:
            print("   ✓ SUCESSO: Dados atualizados para período atual!")
            return True
        else:
            print("   ⚠ AVISO: Poucos dados para hoje")
            return False
    else:
        print(f"ERRO: Arquivo não encontrado: {test_file}")
        return False

if __name__ == "__main__":
    success = update_test_data()
    print("\n" + "=" * 80)
    if success:
        print("SUCESSO: Dados de teste atualizados!")
    else:
        print("FALHA: Problema na atualização")
    print("=" * 80)