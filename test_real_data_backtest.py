#!/usr/bin/env python3
"""
Teste do backtest com dados históricos reais do CSV
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.ml_backtester import AdvancedMLBacktester, BacktestConfig, BacktestMode

def load_real_wdo_data():
    """Carrega dados reais do CSV"""
    csv_path = r"C:\Users\marth\OneDrive\Programacao\Python\Projetos\ML_Tradingv2.0\src\training\data\historical\wdo_data_20_06_2025.csv"
    
    print(f"🔍 Verificando arquivo: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ Arquivo não encontrado: {csv_path}")
        return None
    
    try:
        print("📊 Carregando dados reais...")
        df = pd.read_csv(csv_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        
        print(f"✅ Dados carregados: {len(df)} registros")
        print(f"📅 Período: {df.index.min()} até {df.index.max()}")
        print(f"📋 Colunas: {list(df.columns)}")
        
        # Limpar dados
        columns_to_remove = ['contract', 'preco']
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
                print(f"🗑️  Removida coluna: {col}")
        
        # Manter apenas colunas numéricas
        essential_cols = ['open', 'high', 'low', 'close', 'volume']
        optional_cols = ['buy_volume', 'sell_volume', 'quantidade', 'trades']
        
        final_cols = []
        for col in essential_cols:
            if col in df.columns:
                final_cols.append(col)
            else:
                print(f"⚠️  Coluna essencial faltando: {col}")
        
        for col in optional_cols:
            if col in df.columns:
                final_cols.append(col)
        
        df = df[final_cols]
        
        # Converter para numérico
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remover NaN
        initial_len = len(df)
        df = df.dropna()
        final_len = len(df)
        
        if initial_len != final_len:
            print(f"🧹 Removidos {initial_len - final_len} registros com NaN")
        
        print(f"💰 Preço médio: R$ {df['close'].mean():,.2f}")
        print(f"📊 Volume médio: {df['volume'].mean():,.0f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Erro carregando dados: {e}")
        return None

def run_backtest_with_real_data():
    """Executa backtest com dados reais"""
    print("=" * 60)
    print("🚀 BACKTEST COM DADOS REAIS - WDO")
    print("=" * 60)
    
    # Carregar dados reais
    historical_data = load_real_wdo_data()
    
    if historical_data is None:
        print("❌ Não foi possível carregar dados reais")
        return None
    
    # Selecionar período para backtest (últimos 30 dias dos dados)
    end_date = historical_data.index.max()
    start_date = end_date - timedelta(days=30)
    
    # Filtrar dados pelo período
    mask = (historical_data.index >= start_date) & (historical_data.index <= end_date)
    backtest_data = historical_data[mask]
    
    if len(backtest_data) == 0:
        print("❌ Nenhum dado no período selecionado")
        return None
    
    print(f"\n📅 Período do backtest: {start_date.date()} até {end_date.date()}")
    print(f"📊 Dados para backtest: {len(backtest_data)} registros")
    
    # Configuração do backtest
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000.0,
        commission_per_contract=0.50,
        slippage_ticks=1,
        mode=BacktestMode.REALISTIC
    )
    
    print(f"💰 Capital inicial: R$ {config.initial_capital:,.2f}")
    print(f"⚙️  Modo: {config.mode.value}")
    print("-" * 60)
    
    # Criar backtester
    backtester = AdvancedMLBacktester(config)
    
    # Inicializar sem modelos ML (usar sinais simulados)
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
            return self.config.slippage_ticks * 0.5
    
    backtester.cost_model = MockCostModel(config)
    
    print("\n🎯 Executando backtest com dados reais...")
    
    # Resetar estado
    backtester._reset_state()
    
    # Processar dados reais
    signals_generated = 0
    for i, (timestamp, candle) in enumerate(backtest_data.iterrows()):
        try:
            # Converter timestamp
            timestamp_dt = backtester._ensure_datetime(timestamp)
            
            # Atualizar equity
            backtester._update_equity(candle, timestamp_dt)
            
            # Gerar sinal com base em lógica simples (RSI ou similar)
            if i > 50 and i % 200 == 0:  # A cada 200 candles depois de 50 iniciais
                # Lógica simples: comparar preço com média móvel
                recent_prices = backtest_data['close'].iloc[max(0, i-20):i]
                current_price = candle['close']
                avg_price = recent_prices.mean()
                
                # Sinal baseado em reversão à média
                if current_price > avg_price * 1.002:  # 0.2% acima da média
                    action = 'sell'
                    confidence = 0.65
                elif current_price < avg_price * 0.998:  # 0.2% abaixo da média
                    action = 'buy'
                    confidence = 0.65
                else:
                    action = 'none'
                    confidence = 0.5
                
                if action != 'none':
                    signal = {
                        'action': action,
                        'confidence': confidence,
                        'symbol': 'WDO',
                        'price': current_price
                    }
                    
                    backtester._process_signal(signal, candle, timestamp_dt)
                    signals_generated += 1
                    print(f"  Sinal {signals_generated}: {action.upper()} @ R$ {current_price:,.2f} (avg: R$ {avg_price:,.2f})")
            
            # Verificar stops
            backtester._check_stops(candle)
            
        except Exception as e:
            if i < 5:  # Mostrar apenas os primeiros erros
                print(f"Erro processando candle {i}: {e}")
    
    # Fechar posições abertas
    if backtester.positions:
        print(f"\n🔒 Fechando {len(backtester.positions)} posições abertas...")
        backtester._close_all_positions(backtest_data.iloc[-1], "end_of_backtest")
    
    # Calcular métricas
    print("\n📊 Calculando métricas finais...")
    metrics = backtester._calculate_final_metrics()
    
    # Exibir resultados
    print("\n" + "=" * 60)
    print("📊 RESULTADOS DO BACKTEST COM DADOS REAIS")
    print("=" * 60)
    
    print(f"\n💰 RESUMO FINANCEIRO:")
    print(f"Capital Inicial: R$ {config.initial_capital:,.2f}")
    print(f"Capital Final: R$ {metrics['final_equity']:,.2f}")
    print(f"Lucro/Prejuízo: R$ {metrics['total_pnl']:,.2f}")
    
    if 'total_return' in metrics:
        print(f"Retorno: {metrics['total_return']*100:.2f}%")
    else:
        retorno = ((metrics['final_equity'] / config.initial_capital) - 1) * 100
        print(f"Retorno: {retorno:.2f}%")
    
    print(f"\n📈 ESTATÍSTICAS:")
    print(f"Total de Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"Trades Vencedoras: {metrics['winning_trades']}")
    print(f"Trades Perdedoras: {metrics['losing_trades']}")
    print(f"Sinais Gerados: {signals_generated}")
    
    if metrics['total_trades'] > 0:
        print(f"\n💎 MÉTRICAS AVANÇADAS:")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"Expectativa: R$ {metrics.get('expectancy', 0):,.2f}")
        print(f"Ganho Médio: R$ {metrics.get('avg_win', 0):,.2f}")
        print(f"Perda Média: R$ {metrics.get('avg_loss', 0):,.2f}")
        
        if 'max_drawdown' in metrics:
            print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    
    print("\n✅ Backtest com dados reais concluído!")
    
    return metrics

if __name__ == "__main__":
    try:
        metrics = run_backtest_with_real_data()
        if metrics:
            print(f"\n🎯 Resultado Final: {((metrics['final_equity']/100000) - 1)*100:.2f}% de retorno")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()