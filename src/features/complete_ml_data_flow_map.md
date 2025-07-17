# Mapeamento Completo do Fluxo de Dados ML - Sistema de Trading

## 📋 Índice

1. [Visão Geral do Fluxo](#visão-geral-do-fluxo)
2. [Diagrama Geral do Sistema](#diagrama-geral-do-sistema)
3. [Etapa 1: Carregamento dos Modelos](#etapa-1-carregamento-dos-modelos-e-identificação-de-features)
4. [Etapa 2: Carregamento de Dados](#etapa-2-carregamento-e-concatenação-de-dados)
5. [Etapa 3: Cálculo de Indicadores](#etapa-3-cálculo-de-indicadores-técnicos)
6. [Etapa 4: Cálculo de Features ML](#etapa-4-cálculo-de-features-ml)
7. [Etapa 5: Predição](#etapa-5-preparação-e-execução-da-predição)
8. [Etapa 6: Sinal de Trading](#etapa-6-geração-de-sinal-de-trading)
9. [Diagrama de Features](#diagrama-detalhado-de-features)
10. [Exemplo Prático Completo](#exemplo-prático-completo)
11. [Resumo das Features](#resumo-das-features-utilizadas)

---

## Visão Geral do Fluxo

O fluxo de dados segue a seguinte sequência principal:

```
1. Carregamento dos Modelos ML → Identificação das features necessárias
2. Carregamento de Dados → Históricos + Tempo Real
3. Cálculo de Indicadores → Processamento técnico
4. Cálculo de Features ML → Preparação para predição
5. Predição → Execução dos modelos
6. Resultado → Sinal de trading
```

---

## Diagrama Geral do Sistema

```mermaid
graph TB
    %% Etapa 1: Carregamento dos Modelos
    A[Início] --> B[MLIntegration.__init__]
    B --> C[model_trainer.load_saved_models]
    C --> D[_discover_model_features]
    D --> E[Lista de Features Requeridas]
    
    %% Etapa 2: Carregamento de Dados
    E --> F[EnhancedHistoricalLoader]
    F --> G[load_historical_data]
    G --> H[df_candles<br/>df_microstructure<br/>df_orderbook]
    
    E --> I[MarketDataProcessor]
    I --> J[add_trade em tempo real]
    J --> H
    
    %% Etapa 3: Alinhamento
    H --> K[get_aligned_data_for_ml]
    K --> L[DataFrame Unificado<br/>e Alinhado]
    
    %% Etapa 4: Cálculo de Indicadores
    L --> M[request_indicator_calculation]
    M --> N[FeatureGeneratorAdapter]
    N --> O[_calculate_technical_indicators]
    O --> P[df_indicators<br/>EMAs, RSI, MACD, BB, etc]
    
    %% Etapa 5: Cálculo de Features
    P --> Q[create_features_separated]
    Q --> R[_calculate_momentum_features]
    Q --> S[_calculate_volatility_features]
    Q --> T[_calculate_microstructure_features]
    Q --> U[_calculate_composite_features]
    
    R --> V[df_features]
    S --> V
    T --> V
    U --> V
    
    %% Etapa 6: Predição
    V --> W[MLIntegration.request_prediction]
    W --> X[_prepare_features_for_prediction]
    X --> Y[Validação de Features]
    Y --> Z[batch_predict_next_candles]
    
    %% Etapa 7: Resultado
    Z --> AA[Prediction Result<br/>direction, magnitude, confidence]
    AA --> AB[TradingStrategy.generate_signal]
    AB --> AC[Sinal de Trading<br/>buy/sell/none]
    
    %% Estilos
    classDef modelClass fill:#f9f,stroke:#333,stroke-width:2px
    classDef dataClass fill:#bbf,stroke:#333,stroke-width:2px
    classDef featureClass fill:#bfb,stroke:#333,stroke-width:2px
    classDef predClass fill:#fbf,stroke:#333,stroke-width:2px
    classDef signalClass fill:#fbb,stroke:#333,stroke-width:2px
    
    class B,C,D,E modelClass
    class F,G,H,I,J,K,L dataClass
    class M,N,O,P,Q,R,S,T,U,V featureClass
    class W,X,Y,Z,AA predClass
    class AB,AC signalClass
```

---

## Etapa 1: Carregamento dos Modelos e Identificação de Features

### 1.1 Inicialização do MLIntegration

```python
MLIntegration.__init__(self, config: Dict)
    ├── ModelLoader.__init__(self, model_dir: str = 'saved_models')
    ├── model_trainer.load_saved_models() -> bool
    │   └── Carrega modelos salvos do diretório
    └── self.discovered_features = self._discover_model_features()
```

**Argumentos:**
- `config`: Dict contendo:
  - `models_dir`: Diretório dos modelos (default: 'saved_models')
  - `prediction_interval`: Intervalo entre predições em segundos
  - `backfill_candles`: Número de candles necessárias para predição

### 1.2 Descoberta de Features do Modelo

```python
MLIntegration._discover_model_features() -> Dict[str, List[str]]
    ├── model_trainer.models -> Dict com modelos carregados
    ├── Para cada modelo:
    │   ├── model.get_booster().feature_names -> List[str]
    │   └── Adiciona ao conjunto de features
    └── Retorna: {
            'all_features': List[str],      # Todas as features únicas
            'by_model': Dict[str, List[str]] # Features por modelo
        }
```

### 1.3 Sincronização do Feature Generator

```python
FeatureGeneratorAdapter.sync_with_model(model_features: List[str])
    ├── self.model_features = model_features
    ├── self.required_features = set(model_features)
    └── Categoriza features em:
        ├── 'basic': ['open', 'high', 'low', 'close', 'volume']
        ├── 'indicators': ['ema_*', 'rsi', 'macd', etc.]
        └── 'features': ['momentum_*', 'volatility_*', etc.]
```

---

## Etapa 2: Carregamento e Concatenação de Dados

### 2.1 Dados Históricos

```python
EnhancedHistoricalLoader.load_historical_data(
    ticker: str,
    start_date: datetime = None,
    end_date: datetime = None,
    days_back: int = 10
) -> Dict[str, pd.DataFrame]
    ├── _process_trades_to_separated_dataframes(trades: List)
    │   ├── Agrupa trades por minuto
    │   ├── Calcula OHLCV → df_candles
    │   ├── Calcula microestrutura → df_microstructure
    │   └── Calcula orderbook → df_orderbook
    └── Retorna: {
            'candles': pd.DataFrame,
            'microstructure': pd.DataFrame,
            'orderbook': pd.DataFrame,
            'indicators': pd.DataFrame  # Vazio inicialmente
        }
```

**Argumentos:**
- `ticker`: Símbolo do ativo (ex: "WDOQ25")
- `start_date`: Data inicial (opcional)
- `end_date`: Data final (opcional)
- `days_back`: Dias retroativos se datas não especificadas

### 2.2 Dados em Tempo Real

```python
MarketDataProcessor.add_trade(trade_data: Dict[str, Any]) -> bool
    ├── _update_candles_dataframe(candle_time, trade_data)
    │   └── Atualiza OHLCV no df_candles
    ├── _update_microstructure_dataframe(candle_time, trade_data)
    │   └── Atualiza buy/sell volume no df_microstructure
    └── _on_new_candle_formed(candle_time) 
        └── Marca flag para recálculo
```

**Argumentos de trade_data:**
- `timestamp`: datetime do trade
- `price`: Preço do trade
- `volume`: Volume negociado
- `trade_type`: 2 (buy) ou 3 (sell)
- `trade_id`: ID único do trade

### 2.3 Alinhamento de Dados para ML

```python
MarketDataProcessor.get_aligned_data_for_ml(
    lookback_candles: int
) -> pd.DataFrame
    ├── Seleciona período: últimas N candles
    ├── Combina dataframes:
    │   ├── df_candles (base)
    │   ├── df_microstructure (alinhado)
    │   ├── df_indicators (forward fill)
    │   └── df_features (forward fill)
    └── Retorna: DataFrame unificado e alinhado
```

---

## Etapa 3: Cálculo de Indicadores Técnicos

### 3.1 Requisição de Cálculo

```python
MarketDataProcessor.request_indicator_calculation(
    recalculate: bool = False,
    callback: callable = None
) -> bool
    ├── Verifica debouncing (min 500ms entre cálculos)
    ├── Adiciona à fila: indicator_queue.put({
    │       'candles_df': df_candles.copy(),
    │       'microstructure_df': df_microstructure.copy(),
    │       'indicators_df': df_indicators.copy(),
    │       'recalculate': recalculate,
    │       'callback': callback
    │   })
    └── Executa em thread separada
```

### 3.2 Processamento de Indicadores

```python
FeatureGeneratorAdapter._calculate_technical_indicators(
    candles_df: pd.DataFrame,
    indicators_df: pd.DataFrame
)
    ├── _calculate_moving_averages(candles_df, indicators_df)
    │   └── EMAs: [9, 20, 50, 200] períodos
    ├── _calculate_rsi(candles_df, indicators_df)
    │   └── RSI com período 14
    ├── _calculate_macd(candles_df, indicators_df)
    │   └── MACD (12, 26, 9)
    ├── _calculate_bollinger_bands(candles_df, indicators_df)
    │   └── BB com períodos [20, 50]
    ├── _calculate_stochastic(candles_df, indicators_df)
    │   └── Stoch K/D e Slow K/D
    └── _calculate_atr(candles_df, indicators_df)
        └── ATR com período 14
```

---

## Etapa 4: Cálculo de Features ML

### 4.1 Método Principal de Features

```python
FeatureGeneratorAdapter.create_features_separated(
    candles_df: pd.DataFrame,
    microstructure_df: Optional[pd.DataFrame] = None,
    indicators_df: Optional[pd.DataFrame] = None,
    recalculate: bool = False
) -> Dict[str, pd.DataFrame]
    ├── Se sincronizado com modelo:
    │   └── _create_model_specific_features()  # Otimizado
    └── Senão:
        ├── _calculate_momentum_features(candles_df, features_df)
        ├── _calculate_volatility_features(candles_df, features_df)
        ├── _calculate_microstructure_features(microstructure_df, features_df)
        └── _calculate_composite_features(...)
```

**Retorno:**
```python
{
    'indicators': pd.DataFrame,  # Indicadores técnicos
    'features': pd.DataFrame,    # Features ML calculadas
    'all': pd.DataFrame         # DataFrame unificado
}
```

### 4.2 Features de Momentum

```python
_calculate_momentum_features(candles_df, features_df)
    ├── momentum_[1,3,5,10,15,20]: price - price.shift(period)
    ├── momentum_pct_[1,3,5,10,15,20]: price.pct_change(period) * 100
    ├── return_[5,10,20,50]: price.pct_change(period) * 100
    └── volume_momentum_[5,10,20,50]: volume.pct_change(period) * 100
```

### 4.3 Features de Volatilidade

```python
_calculate_volatility_features(candles_df, features_df)
    ├── volatility_[5,10,20,50]: price.rolling(window).std()
    ├── range_percent: (high - low) / close * 100
    ├── high_low_range_[5,10,20,50]: rolling high-low range
    └── volatility_ratio_[5,10,20]: volatility ratios
```

### 4.4 Features de Microestrutura

```python
_calculate_microstructure_features(microstructure_df, features_df)
    ├── buy_pressure: buy_volume / total_volume
    ├── volume_ratio_[5,10]: volume ratios
    ├── buy_sell_imbalance: (buy - sell) / total
    └── flow_imbalance: abs(buy_volume - sell_volume)
```

---

## Etapa 5: Preparação e Execução da Predição

### 5.1 Requisição de Predição

```python
MLIntegration.request_prediction(
    market_data: pd.DataFrame,
    force_calculation: bool = False
) -> bool
    ├── Validações:
    │   ├── Verifica se há dados suficientes
    │   └── Verifica se modelos estão carregados
    ├── prediction_queue.put({
    │       'market_data': market_data,
    │       'timestamp': datetime.now(),
    │       'force': force_calculation
    │   })
    └── Processado por prediction_worker_thread
```

### 5.2 Preparação de Features para Predição

```python
MLIntegration._prepare_features_for_prediction(
    market_data: pd.DataFrame
) -> pd.DataFrame
    ├── feature_creator.create_features_separated(
    │       candles_df=market_data[candle_cols],
    │       microstructure_df=market_data[micro_cols],
    │       recalculate=True
    │   )
    ├── Resultado: Dict com 'all' → DataFrame unificado
    ├── Validação de features:
    │   └── Verifica se todas as features do modelo estão presentes
    └── Retorna: DataFrame com features ordenadas conforme modelo
```

### 5.3 Execução da Predição

```python
ModelLoader.batch_predict_next_candles(
    features_df: pd.DataFrame,
    num_candles: int = 1,
    use_ensemble: bool = True
) -> Dict[str, Any]
    ├── Para cada modelo em self.models:
    │   ├── model.predict(features_df) → array
    │   └── Armazena predições
    ├── Se use_ensemble:
    │   └── Calcula média ponderada das predições
    └── Retorna: {
            'predictions': np.array,      # [num_candles, 3] (dir, mag, conf)
            'direction': float,           # -1 a 1
            'magnitude': float,           # Movimento esperado
            'confidence': float,          # 0 a 1
            'model_predictions': Dict,    # Por modelo
            'timestamp': datetime
        }
```

---

## Etapa 6: Geração de Sinal de Trading

### 6.1 Processamento da Predição

```python
MLIntegration._process_prediction(prediction_result: Dict)
    ├── Análise de regime de mercado
    ├── trading_strategy.generate_signal(
    │       prediction=prediction_result,
    │       features_df=features_df,
    │       market_data=market_data,
    │       current_time=datetime.now(),
    │       regime_info=regime_info
    │   )
    └── Armazena resultado em latest_signal
```

### 6.2 Geração do Sinal

```python
TradingStrategy.generate_signal(...) -> Dict[str, Any]
    ├── Validações:
    │   ├── direction_threshold: 0.45
    │   ├── magnitude_threshold: 0.00015
    │   └── confidence_threshold: 0.15
    ├── Análise de tendência
    ├── Cálculo de stop/target baseado em:
    │   ├── ATR (volatilidade)
    │   ├── Magnitude prevista
    │   └── Regime de mercado
    └── Retorna: {
            'action': 'buy'|'sell'|'none',
            'reason': str,
            'stop_loss': float,
            'take_profit': float,
            'confidence': float,
            'timestamp': datetime
        }
```

---

## Diagrama Detalhado de Features

```mermaid
graph LR
    %% Dados de Entrada
    subgraph "Dados de Entrada"
        A1[df_candles<br/>OHLCV]
        A2[df_microstructure<br/>Buy/Sell Volume]
        A3[df_indicators<br/>Indicadores Técnicos]
    end
    
    %% Cálculo de Indicadores
    subgraph "Indicadores Técnicos"
        B1[EMAs<br/>9, 20, 50, 200]
        B2[RSI<br/>período 14]
        B3[MACD<br/>12, 26, 9]
        B4[Bollinger<br/>20, 50]
        B5[Stochastic<br/>K, D, Slow]
        B6[ATR<br/>período 14]
    end
    
    %% Features de ML
    subgraph "Features ML"
        C1[Momentum<br/>1-20 períodos]
        C2[Volatilidade<br/>5-50 períodos]
        C3[Microestrutura<br/>Pressão, Imbalance]
        C4[Compostas<br/>Regime, Pattern]
    end
    
    %% Modelo Específico
    subgraph "Features do Modelo"
        D1[Feature Selection<br/>Baseado no modelo]
        D2[Feature Ordering<br/>Ordem específica]
        D3[Default Values<br/>Para missing]
    end
    
    %% Predição
    subgraph "Predição ML"
        E1[DataFrame Final<br/>Features ordenadas]
        E2[model.predict<br/>XGBoost/LightGBM]
        E3[Ensemble<br/>Média ponderada]
    end
    
    %% Fluxo
    A1 --> B1
    A1 --> B2
    A1 --> B3
    A1 --> B4
    A1 --> B5
    A1 --> B6
    
    A1 --> C1
    A1 --> C2
    A2 --> C3
    B1 --> C4
    B2 --> C4
    B3 --> C4
    
    C1 --> D1
    C2 --> D1
    C3 --> D1
    C4 --> D1
    
    D1 --> D2
    D2 --> D3
    D3 --> E1
    E1 --> E2
    E2 --> E3
    
    %% Resultado
    E3 --> F[direction: -1 a 1<br/>magnitude: movimento<br/>confidence: 0 a 1]
    
    %% Estilos
    classDef inputClass fill:#e6f3ff,stroke:#0066cc,stroke-width:2px
    classDef indicatorClass fill:#e6ffe6,stroke:#009900,stroke-width:2px
    classDef featureClass fill:#fff0e6,stroke:#ff6600,stroke-width:2px
    classDef modelClass fill:#ffe6f0,stroke:#cc0066,stroke-width:2px
    classDef predictionClass fill:#f0e6ff,stroke:#6600cc,stroke-width:2px
    
    class A1,A2,A3 inputClass
    class B1,B2,B3,B4,B5,B6 indicatorClass
    class C1,C2,C3,C4 featureClass
    class D1,D2,D3 modelClass
    class E1,E2,E3,F predictionClass
```

---

## Exemplo Prático Completo

```python
# ========================================
# 1. INICIALIZAÇÃO E DESCOBERTA DE FEATURES
# ========================================

# Inicializar MLIntegration
ml_integration = MLIntegration(config={
    'models_dir': 'saved_models',
    'prediction_interval': 60,
    'backfill_candles': 500
})

# Descobrir features dos modelos carregados
discovered_features = ml_integration._discover_model_features()
# Retorna: {
#     'all_features': ['close', 'volume', 'ema_9', 'ema_20', 'rsi', ...],
#     'by_model': {
#         'xgb_trend_model': ['close', 'ema_9', 'ema_20', ...],
#         'lgb_regime_model': ['rsi', 'volatility_10', ...]
#     }
# }

# Sincronizar feature generator
feature_generator.sync_with_model(
    model_features=discovered_features['all_features']
)

# ========================
# 2. CARREGAMENTO DE DADOS
# ========================

# Carregar dados históricos
historical_data = loader.load_historical_data(
    ticker="WDOQ25",
    start_date=None,  # Usa days_back
    end_date=None,    # Até agora
    days_back=10
)
# Retorna: {
#     'candles': pd.DataFrame,      # Colunas: open, high, low, close, volume
#     'microstructure': pd.DataFrame, # Colunas: buy_volume, sell_volume, buy_trades, sell_trades
#     'orderbook': pd.DataFrame,     # Colunas: bid, ask, spread, bid_volume, ask_volume
#     'indicators': pd.DataFrame     # Vazio inicialmente
# }

# Processar trade em tempo real
success = data_processor.add_trade(trade_data={
    'timestamp': datetime.now(),
    'price': 5432.5,
    'volume': 10,
    'trade_type': 2,  # 2=buy, 3=sell
    'trade_id': 123456
})

# ================================
# 3. ALINHAMENTO DE DADOS PARA ML
# ================================

aligned_data = data_processor.get_aligned_data_for_ml(
    lookback_candles=500  # Últimas 500 candles
)
# Retorna: pd.DataFrame com todas as colunas alinhadas

# =========================
# 4. CÁLCULO DE INDICADORES
# =========================

# Requisitar cálculo (assíncrono)
data_processor.request_indicator_calculation(
    recalculate=True,
    callback=lambda result: print("Indicadores calculados!")
)

# Internamente executa:
result = feature_generator.create_features_separated(
    candles_df=df_candles,
    microstructure_df=df_microstructure,
    indicators_df=df_indicators,  # Pode ser None
    recalculate=True
)
# Retorna: {
#     'indicators': pd.DataFrame,  # EMAs, RSI, MACD, BB, etc
#     'features': pd.DataFrame,    # Momentum, volatilidade, etc
#     'all': pd.DataFrame         # Tudo unificado
# }

# =========================
# 5. REQUISIÇÃO DE PREDIÇÃO
# =========================

# Requisitar predição ML
ml_integration.request_prediction(
    market_data=aligned_data,
    force_calculation=True
)

# Internamente prepara features:
prepared_features = ml_integration._prepare_features_for_prediction(
    market_data=aligned_data
)

# =======================
# 6. EXECUÇÃO DA PREDIÇÃO
# =======================

# Executar predição com modelos
prediction_result = model_loader.batch_predict_next_candles(
    features_df=prepared_features,
    num_candles=1,      # Prever próxima candle
    use_ensemble=True   # Usar ensemble de modelos
)
# Retorna: {
#     'predictions': np.array([[0.7, 0.0003, 0.85]]),  # [direction, magnitude, confidence]
#     'direction': 0.7,        # -1 (sell) a +1 (buy)
#     'magnitude': 0.0003,     # Movimento esperado
#     'confidence': 0.85,      # 0 a 1
#     'model_predictions': {
#         'xgb_trend': {'direction': 0.8, ...},
#         'lgb_regime': {'direction': 0.6, ...}
#     },
#     'timestamp': datetime.now()
# }

# ===================
# 7. GERAÇÃO DE SINAL
# ===================

# Processar predição e gerar sinal
signal = trading_strategy.generate_signal(
    prediction=prediction_result,
    features_df=prepared_features,
    market_data=aligned_data,
    current_time=datetime.now(),
    regime_info={
        'market_regime': 'trending',
        'volatility_regime': 'normal',
        'volume_regime': 'high'
    }
)
# Retorna: {
#     'action': 'buy',              # 'buy', 'sell' ou 'none'
#     'reason': 'strong_trend',     # Razão da decisão
#     'stop_loss': 5420.0,         # Preço de stop
#     'take_profit': 5445.0,       # Preço de alvo
#     'confidence': 0.85,          # Confiança do sinal
#     'risk_reward': 2.5,          # Relação risco/retorno
#     'position_size': 1,          # Tamanho da posição
#     'timestamp': datetime.now()
# }
```

---

## Resumo das Features Utilizadas

### Features Básicas (OHLCV)
- `open`, `high`, `low`, `close`, `volume`

### Indicadores Técnicos
- **EMAs**: `ema_9`, `ema_20`, `ema_50`, `ema_200`
- **RSI**: `rsi`
- **MACD**: `macd`, `macd_signal`, `macd_hist`
- **Bollinger Bands**: `bb_upper_20`, `bb_lower_20`, `bb_width_20`
- **Stochastic**: `stoch_k`, `stoch_d`, `slow_k`, `slow_d`
- **ATR**: `atr`, `true_range`

### Features de Momentum
- **Momentum**: `momentum_1` até `momentum_20`
- **Momentum %**: `momentum_pct_1` até `momentum_pct_20`
- **Returns**: `return_5`, `return_10`, `return_20`, `return_50`

### Features de Volatilidade
- **Volatility**: `volatility_5`, `volatility_10`, `volatility_20`, `volatility_50`
- **Ranges**: `range_percent`, `high_low_range_*`

### Features de Microestrutura
- **Volume**: `buy_volume`, `sell_volume`, `buy_trades`, `sell_trades`
- **Pressão**: `buy_pressure`, `buy_sell_ratio`, `buy_sell_imbalance`
- **Flow**: `flow_imbalance`

### Features Compostas
- **Regime**: `regime_strength`, `regime_trend`, `regime_volatility`
- **Pattern**: `price_position_bb`, `rsi_divergence`
- **Volume**: `volume_trend`, `volume_surge`

### Exemplo de DataFrame Final

```python
# DataFrame final contém ~80-100 features, incluindo:
features_example = {
    # Básicas
    'open': 5430.0,
    'high': 5435.0,
    'low': 5428.0,
    'close': 5432.5,
    'volume': 1250,
    
    # Indicadores
    'ema_9': 5431.2,
    'ema_20': 5429.8,
    'rsi': 58.5,
    'macd': 0.8,
    'macd_signal': 0.6,
    'bb_upper_20': 5440.0,
    'bb_lower_20': 5420.0,
    'atr': 12.5,
    
    # Momentum
    'momentum_5': 2.5,
    'momentum_10': 4.0,
    'momentum_pct_5': 0.046,
    'return_10': 0.074,
    
    # Volatilidade
    'volatility_10': 8.5,
    'volatility_20': 10.2,
    'range_percent': 0.26,
    
    # Microestrutura
    'buy_volume': 750,
    'sell_volume': 500,
    'buy_pressure': 0.6,
    'buy_sell_imbalance': 0.2,
    'flow_imbalance': 250,
    
    # Compostas
    'regime_strength': 0.75,
    'trend_strength': 0.8,
    'price_position_bb': 0.6
}
```

---

## Observações Importantes

### 1. Sincronização
O sistema mantém dataframes separados mas sincronizados por timestamp, garantindo consistência temporal entre todos os dados.

### 2. Cache e Performance
- Features são recalculadas sempre que `recalculate=True` para garantir frescor dos dados
- Cálculos pesados são executados em threads separadas para não bloquear o fluxo principal
- Sistema de debouncing evita cálculos desnecessários

### 3. Validação
- Cada etapa valida os dados antes de prosseguir
- Features faltantes recebem valores padrão apropriados
- Logs detalhados em cada etapa para debugging

### 4. Modularidade
- Cada componente pode ser testado independentemente
- DataFrames separados facilitam manutenção e otimização
- Sistema preparado para adicionar novos tipos de features

### 5. Threading e Filas
- `indicator_queue`: Fila prioritária para cálculo de indicadores
- `prediction_queue`: Fila para requisições de predição ML
- Threads separadas para cálculo e predição

---

## Conclusão

Este mapeamento completo mostra como o sistema de trading ML processa dados desde a entrada (trades) até a saída (sinais de trading), passando por todas as etapas de transformação e cálculo de features. O sistema é otimizado para:

1. **Eficiência**: Calcula apenas features necessárias quando sincronizado com modelo
2. **Flexibilidade**: Suporta diferentes tipos de modelos e features
3. **Robustez**: Validações e tratamento de erros em cada etapa
4. **Performance**: Uso de threads e dataframes separados
5. **Manutenibilidade**: Código modular e bem documentado

Total de features processadas: **~80-100** dependendo do modelo e configuração.