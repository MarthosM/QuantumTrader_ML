#!/usr/bin/env python3
"""
Script simplificado para teste rápido de backtest
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.ml_backtester import AdvancedMLBacktester, BacktestConfig, BacktestMode

def create_sample_data(days=30):
    """Cria dados de exemplo para teste"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Gerar timestamps minuto a minuto
    timestamps = pd.date_range(start=start_date, end=end_date, freq='1min')
    
    # Filtrar apenas horário de pregão (9h às 18h)
    timestamps = timestamps[(timestamps.hour >= 9) & (timestamps.hour < 18)]
    
    # Preço base WDO
    base_price = 5600
    n_points = len(timestamps)
    
    # Gerar movimento de preço realista
    returns = np.random.normal(0, 0.0002, n_points)  # 0.02% volatilidade por minuto
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Criar OHLCV
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        open_price = prices[i-1] if i > 0 else base_price
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.0001)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.0001)))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low, 
            'close': close,
            'volume': volume,
            'trades': np.random.randint(10, 100)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    print(f"📊 Dados criados: {len(df)} candles")
    print(f"📅 Período: {df.index.min()} até {df.index.max()}")
    print(f"💰 Preço médio: R$ {df['close'].mean():,.2f}")
    
    return df

def run_simple_backtest():
    """Executa um backtest simplificado"""
    print("=" * 60)
    print("🚀 BACKTEST RÁPIDO - ML TRADING SYSTEM")
    print("=" * 60)
    
    # Configuração do backtest
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now(),
        initial_capital=100000.0,
        commission_per_contract=0.50,
        slippage_ticks=1,
        mode=BacktestMode.REALISTIC
    )
    
    print(f"💰 Capital inicial: R$ {config.initial_capital:,.2f}")
    print(f"📅 Período: {config.start_date.date()} até {config.end_date.date()}")
    print(f"⚙️  Modo: {config.mode.value}")
    print("-" * 60)
    
    # Criar backtester
    backtester = AdvancedMLBacktester(config)
    
    # Para teste, não usar modelos ML reais (simular sinais)
    # Inicializar sem componentes externos
    backtester.ml_models = {}
    backtester.feature_engine = None
    backtester.market_simulator = None
    
    # Criar cost_model mock
    class MockCostModel:
        def __init__(self, config):
            self.config = config
            
        def calculate_commission(self, quantity):
            return quantity * self.config.commission_per_contract
            
        def calculate_slippage(self, base_price, side, market_data):
            return self.config.slippage_ticks * 0.5  # 0.5 pontos por tick no WDO
    
    backtester.cost_model = MockCostModel(config)
    
    # Criar dados de teste
    print("\n📊 Gerando dados de teste...")
    historical_data = create_sample_data(days=7)
    
    # Simular alguns sinais de trading
    print("\n🎯 Executando backtest com sinais simulados...")
    
    # Método simplificado: processar dados manualmente
    backtester._reset_state()
    
    # Processar cada candle
    signals_generated = 0
    for i, (timestamp, candle) in enumerate(historical_data.iterrows()):
        try:
            # Converter timestamp para datetime se necessário
            timestamp_dt = backtester._ensure_datetime(timestamp)
            
            # Atualizar equity
            backtester._update_equity(candle, timestamp_dt)
            
            # Gerar sinal aleatório a cada 100 candles
            if i % 100 == 0 and i > 0:
                # Sinal simulado
                action = np.random.choice(['buy', 'sell', 'none'], p=[0.3, 0.3, 0.4])
                
                if action != 'none':
                    signal = {
                        'action': action,
                        'confidence': np.random.uniform(0.6, 0.9),
                        'symbol': 'WDO',
                        'price': candle['close']
                    }
                    
                    backtester._process_signal(signal, candle, timestamp_dt)
                    signals_generated += 1
                    print(f"  Sinal {signals_generated}: {action.upper()} @ R$ {candle['close']:,.2f}")
            
            # Verificar stops
            backtester._check_stops(candle)
            
        except Exception as e:
            print(f"Erro processando candle: {e}")
    
    # Fechar posições abertas
    if backtester.positions:
        print(f"\n🔒 Fechando {len(backtester.positions)} posições abertas...")
        backtester._close_all_positions(historical_data.iloc[-1], "end_of_backtest")
    
    # Calcular métricas
    print("\n📊 Calculando métricas finais...")
    metrics = backtester._calculate_final_metrics()
    
    # Exibir resultados
    print("\n" + "=" * 60)
    print("📊 RESULTADOS DO BACKTEST")
    print("=" * 60)
    
    print(f"\n💰 RESUMO FINANCEIRO:")
    print(f"Capital Inicial: R$ {config.initial_capital:,.2f}")
    print(f"Capital Final: R$ {metrics['final_equity']:,.2f}")
    print(f"Lucro/Prejuízo: R$ {metrics['total_pnl']:,.2f}")
    if 'total_return' in metrics:
        print(f"Retorno: {metrics['total_return']*100:.2f}%")
    else:
        # Calcular retorno manualmente se não existir
        retorno = ((metrics['final_equity'] / config.initial_capital) - 1) * 100
        print(f"Retorno: {retorno:.2f}%")
    
    print(f"\n📈 ESTATÍSTICAS:")
    print(f"Total de Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"Trades Vencedoras: {metrics['winning_trades']}")
    print(f"Trades Perdedoras: {metrics['losing_trades']}")
    
    if metrics['total_trades'] > 0:
        print(f"\n💎 MÉTRICAS AVANÇADAS:")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"Expectativa: R$ {metrics.get('expectancy', 0):,.2f}")
        print(f"Ganho Médio: R$ {metrics.get('avg_win', 0):,.2f}")
        print(f"Perda Média: R$ {metrics.get('avg_loss', 0):,.2f}")
    
    print("\n✅ Backtest concluído!")
    
    return metrics

if __name__ == "__main__":
    try:
        metrics = run_simple_backtest()
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()