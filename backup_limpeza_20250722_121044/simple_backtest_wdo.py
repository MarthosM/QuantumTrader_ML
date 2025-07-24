#!/usr/bin/env python3
"""
Backtest simples sem features complexas para testar os dados reais do WDO
"""
import pandas as pd
import numpy as np
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_wdo_data():
    """Carrega dados reais do WDO"""
    try:
        csv_path = r"C:\Users\marth\OneDrive\Programacao\Python\Projetos\ML_Tradingv2.0\src\training\data\historical\wdo_data_20_06_2025.csv"
        
        df = pd.read_csv(csv_path)
        
        # Renomear colunas se necessário
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'timestamp'})
            
        # Converter timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Manter apenas colunas numéricas necessárias
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[numeric_cols]
        
        # Converter para numérico
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remover linhas com NaN
        df = df.dropna()
        
        logger.info(f"✅ Dados carregados: {len(df)} registros")
        logger.info(f"Período: {df.index.min()} até {df.index.max()}")
        logger.info(f"Preço médio: R$ {df['close'].mean():.2f}")
        
        return df
        
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return None

def simple_moving_average_strategy(df, short_window=20, long_window=50):
    """Estratégia simples de médias móveis"""
    
    # Calcular médias móveis
    df['sma_short'] = df['close'].rolling(window=short_window).mean()
    df['sma_long'] = df['close'].rolling(window=long_window).mean()
    
    # Gerar sinais
    df['signal'] = 0
    df['signal'][short_window:] = np.where(
        df['sma_short'][short_window:] > df['sma_long'][short_window:], 1, 0
    )
    
    # Posições (1 = comprado, 0 = vendido)
    df['position'] = df['signal'].diff()
    
    return df

def run_simple_backtest(df, initial_capital=100000):
    """Executa backtest simples"""
    
    # Aplicar estratégia
    df = simple_moving_average_strategy(df)
    
    # Calcular returns
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
    
    # Calcular equity curve
    df['equity'] = initial_capital * (1 + df['strategy_returns']).cumprod()
    
    # Métricas básicas
    total_returns = (df['equity'].iloc[-1] / initial_capital - 1) * 100
    buy_hold_returns = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    
    # Contar trades
    trades = df[df['position'] != 0]
    total_trades = len(trades)
    
    # Sharpe ratio simples
    strategy_vol = df['strategy_returns'].std() * np.sqrt(252 * 24 * 60)  # Assumindo dados de 1min
    sharpe = df['strategy_returns'].mean() / strategy_vol if strategy_vol > 0 else 0
    
    results = {
        'initial_capital': initial_capital,
        'final_equity': df['equity'].iloc[-1],
        'total_return_pct': total_returns,
        'buy_hold_return_pct': buy_hold_returns,
        'total_trades': total_trades,
        'sharpe_ratio': sharpe,
        'data_points': len(df)
    }
    
    return results, df

def main():
    """Função principal"""
    print("=== BACKTEST SIMPLES WDO ===")
    
    # Carregar dados
    data = load_wdo_data()
    if data is None:
        print("❌ Falha ao carregar dados")
        return
    
    # Filtrar período específico para teste
    start_date = '2025-06-13'
    end_date = '2025-06-20'
    
    mask = (data.index >= start_date) & (data.index <= end_date)
    test_data = data.loc[mask].copy()
    
    print(f"Período do backtest: {start_date} até {end_date}")
    print(f"Dados no período: {len(test_data)} registros")
    
    if len(test_data) < 100:
        print("❌ Dados insuficientes para backtest")
        return
    
    # Executar backtest
    print("Executando backtest...")
    results, backtest_data = run_simple_backtest(test_data)
    
    # Mostrar resultados
    print("\n=== RESULTADOS ===")
    print(f"Capital inicial: R$ {results['initial_capital']:,.2f}")
    print(f"Capital final: R$ {results['final_equity']:,.2f}")
    print(f"Retorno estratégia: {results['total_return_pct']:.2f}%")
    print(f"Retorno buy & hold: {results['buy_hold_return_pct']:.2f}%")
    print(f"Total de trades: {results['total_trades']}")
    print(f"Sharpe ratio: {results['sharpe_ratio']:.3f}")
    print(f"Pontos de dados: {results['data_points']}")
    
    # Salvar resultados detalhados
    output_file = "simple_backtest_results.csv"
    backtest_data.to_csv(output_file)
    print(f"Resultados salvos em: {output_file}")
    
    print("\n✅ Backtest simples executado com sucesso!")

if __name__ == "__main__":
    main()
