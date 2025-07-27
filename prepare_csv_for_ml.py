"""
Prepara CSV do WDO para uso completo com sistema ML
Adiciona dados faltantes de forma inteligente
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging


def prepare_wdo_data_for_ml(input_file: str = "wdo_data_20_06_2025.csv", 
                           output_file: str = "wdo_ml_ready.csv"):
    """
    Prepara dados WDO adicionando colunas necessárias para ML
    
    Estratégia:
    1. Usar dados existentes
    2. Estimar bid/ask baseado em volatilidade e volume
    3. Calcular métricas derivadas
    4. Adicionar indicadores base
    """
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("="*80)
    print("PREPARAÇÃO DE DADOS WDO PARA ML")
    print("="*80)
    
    # Carregar dados
    logger.info(f"Carregando {input_file}...")
    df = pd.read_csv(input_file)
    
    # Converter Date para datetime index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    logger.info(f"Dados carregados: {len(df)} registros")
    logger.info(f"Período: {df.index[0]} até {df.index[-1]}")
    
    # 1. ESTIMAR BID/ASK
    logger.info("\n1. Estimando bid/ask spread...")
    
    # Calcular volatilidade intracandle
    intrabar_vol = (df['high'] - df['low']) / df['close']
    
    # Calcular pressão compradora/vendedora
    close_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    buy_pressure = df['buy_volume'] / (df['volume'] + 1e-10)
    
    # Spread adaptativo baseado em volatilidade e pressão
    base_spread = 0.5  # 0.5 ponto base no WDO
    vol_factor = intrabar_vol / intrabar_vol.rolling(20).mean().fillna(intrabar_vol.mean())
    
    # Ajustar spread por pressão (maior pressão = spread maior)
    pressure_imbalance = abs(buy_pressure - 0.5) * 2  # 0 a 1
    spread_adjustment = 1 + (pressure_imbalance * 0.5)  # 1 a 1.5x
    
    # Spread final
    spread = base_spread * vol_factor * spread_adjustment
    spread = spread.clip(lower=0.5, upper=5.0)  # Limitar entre 0.5 e 5 pontos
    
    # Posicionar bid/ask baseado na pressão
    mid_adjustment = (buy_pressure - 0.5) * spread * 0.3  # Desloca o mid
    
    df['mid_price'] = df['close'] + mid_adjustment
    df['bid'] = df['mid_price'] - spread/2
    df['ask'] = df['mid_price'] + spread/2
    df['spread'] = spread
    df['spread_pct'] = (spread / df['close']) * 100
    
    logger.info(f"Spread médio estimado: {spread.mean():.2f} pontos ({df['spread_pct'].mean():.3f}%)")
    
    # 2. ESTIMAR TRADES POR LADO
    logger.info("\n2. Estimando número de trades...")
    
    # Assumir distribuição proporcional ao volume
    total_trades = df['quantidade']  # Total de negócios
    df['buy_trades'] = (total_trades * buy_pressure).round().astype(int)
    df['sell_trades'] = total_trades - df['buy_trades']
    
    # 3. CALCULAR MICROESTRUTURA
    logger.info("\n3. Calculando métricas de microestrutura...")
    
    # Imbalance
    df['imbalance'] = df['buy_volume'] - df['sell_volume']
    df['imbalance_ratio'] = df['imbalance'] / (df['volume'] + 1e-10)
    
    # Trade sizes
    df['avg_trade_size'] = df['volume'] / (df['quantidade'] + 1e-10)
    df['avg_buy_size'] = df['buy_volume'] / (df['buy_trades'] + 1e-10)
    df['avg_sell_size'] = df['sell_volume'] / (df['sell_trades'] + 1e-10)
    
    # Trade intensity
    df['trade_intensity'] = df['quantidade'] / (df['volume'] / 1000000)  # Trades por milhão
    
    # 4. ADICIONAR MÉTRICAS DE PREÇO
    logger.info("\n4. Calculando métricas de preço...")
    
    # VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    
    # Price position
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Ranges
    df['hl_range'] = df['high'] - df['low']
    df['hl_pct'] = (df['hl_range'] / df['close']) * 100
    df['true_range'] = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    
    # 5. ADICIONAR INDICADORES BÁSICOS
    logger.info("\n5. Calculando indicadores técnicos básicos...")
    
    # EMAs principais
    for period in [5, 9, 20, 50]:
        if len(df) >= period:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # ATR
    df['atr'] = df['true_range'].rolling(14).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume metrics
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
    
    # Buy/Sell pressure
    df['buy_pressure'] = df['buy_volume'] / (df['volume'] + 1e-10)
    df['sell_pressure'] = df['sell_volume'] / (df['volume'] + 1e-10)
    
    # 6. ADICIONAR FEATURES TEMPORAIS
    logger.info("\n6. Adicionando features temporais...")
    
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    df['timestamp'] = df.index.astype(np.int64) // 10**9
    
    # 7. VALIDAÇÃO E LIMPEZA
    logger.info("\n7. Validando e limpando dados...")
    
    # Preencher NaN
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Remover infinitos
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Verificar integridade
    essential_cols = ['open', 'high', 'low', 'close', 'volume', 'bid', 'ask', 
                     'buy_volume', 'sell_volume', 'imbalance', 'quantity']
    
    missing_cols = [col for col in essential_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Colunas ainda faltando: {missing_cols}")
    
    # 8. ESTATÍSTICAS FINAIS
    logger.info("\n" + "="*60)
    logger.info("ESTATÍSTICAS DO DATASET PREPARADO:")
    logger.info("="*60)
    logger.info(f"Total de registros: {len(df)}")
    logger.info(f"Total de colunas: {len(df.columns)}")
    logger.info(f"Período: {df.index[0]} até {df.index[-1]}")
    logger.info(f"\nMétricas principais:")
    logger.info(f"  Volume médio: {df['volume'].mean():,.0f}")
    logger.info(f"  Trades médios: {df['quantidade'].mean():.0f}")
    logger.info(f"  Spread médio: {df['spread'].mean():.2f} pontos")
    logger.info(f"  Buy pressure média: {df['buy_pressure'].mean():.2%}")
    logger.info(f"  ATR médio: {df['atr'].mean():.1f} pontos")
    
    # Verificar NaN
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"\nAinda existem {nan_counts.sum()} valores NaN")
        logger.warning(nan_counts[nan_counts > 0].head())
    
    # 9. SALVAR
    df.to_csv(output_file)
    logger.info(f"\n✅ Dados salvos em: {output_file}")
    logger.info(f"Shape final: {df.shape}")
    
    # Mostrar colunas disponíveis
    logger.info("\nColunas disponíveis para features:")
    for i, col in enumerate(sorted(df.columns), 1):
        print(f"{i:3d}. {col}")
    
    return df


def validate_ml_readiness(df: pd.DataFrame) -> bool:
    """Valida se o DataFrame está pronto para ML"""
    
    print("\n" + "="*60)
    print("VALIDAÇÃO DE PRONTIDÃO PARA ML")
    print("="*60)
    
    required_for_ml = {
        'OHLCV': ['open', 'high', 'low', 'close', 'volume'],
        'Microestrutura': ['bid', 'ask', 'buy_volume', 'sell_volume', 'imbalance'],
        'Trades': ['quantity', 'buy_trades', 'sell_trades'],
        'Indicadores Base': ['ema_20', 'rsi', 'atr'],
        'Métricas': ['returns', 'vwap', 'buy_pressure']
    }
    
    all_ready = True
    
    for category, cols in required_for_ml.items():
        missing = [col for col in cols if col not in df.columns]
        if missing:
            print(f"❌ {category}: Faltando {missing}")
            all_ready = False
        else:
            print(f"✅ {category}: OK")
    
    if all_ready:
        print("\n✅ DATASET PRONTO PARA ML!")
    else:
        print("\n❌ Dataset precisa de ajustes")
    
    return all_ready


if __name__ == "__main__":
    # Preparar dados
    df = prepare_wdo_data_for_ml()
    
    # Validar
    is_ready = validate_ml_readiness(df)
    
    if is_ready:
        print("\n🚀 Dados prontos para uso!")
        print("Execute: python backtest_with_real_data.py")
    
    # Mostrar amostra
    print("\nAmostra dos dados preparados:")
    print(df[['close', 'bid', 'ask', 'spread', 'buy_pressure', 'atr']].tail())