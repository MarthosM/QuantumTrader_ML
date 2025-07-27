# NOVO MAPA DE FLUXO DE DADOS - SISTEMA ML TRADING V2.0

## VISÃO GERAL
```
DADOS REAIS → PROCESSAMENTO → FEATURES → ML → SINAIS → EXECUÇÃO
```

## 1. FONTE DE DADOS (ProfitDLL)

### 1.1 HISTÓRICOS (Para Treinamento)
```
┌─────────────────────────────────────────────────────────────┐
│                    DADOS HISTÓRICOS                        │
├─────────────────────────────────────────────────────────────┤
│ DLLGetHistoricalData()                                      │
│ ├─ Candles OHLCV + bid/ask                                 │
│ ├─ Volume e quantity reais                                 │
│ └─ Trades count por período                                │
│                                                             │
│ DLLGetHistoricalTrades()                                    │
│ ├─ Cada negócio individual                                 │
│ ├─ Side real (BUY/SELL)                                    │
│ ├─ Price, volume, quantity                                 │
│ └─ Timestamp preciso                                       │
│                                                             │
│ DLLGetHistoricalBook()                                      │
│ ├─ Top 5 níveis bid/ask                                    │
│ ├─ Quantities por nível                                    │
│ └─ Timestamps de mudanças                                  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 TEMPO REAL (Para Trading)
```
┌─────────────────────────────────────────────────────────────┐
│                    TEMPO REAL                              │
├─────────────────────────────────────────────────────────────┤
│ DLLSetNewTradeCallback()                                    │
│ ├─ Cada negócio em tempo real                              │
│ ├─ Agregação para formar candles                           │
│ └─ Cálculo de microestrutura                               │
│                                                             │
│ DLLSetNewBookCallback()                                     │
│ ├─ Mudanças no book em tempo real                          │
│ ├─ Bid/ask atualizados                                     │
│ └─ Spread tracking                                         │
│                                                             │
│ DLLSetNewQuoteCallback()                                    │
│ ├─ Cotações atualizadas                                    │
│ └─ Sincronização de preços                                 │
└─────────────────────────────────────────────────────────────┘
```

## 2. CAMADA DE PROCESSAMENTO

### 2.1 AGREGAÇÃO DE DADOS
```
┌─────────────────────────────────────────────────────────────┐
│                  DATA AGGREGATOR                           │
├─────────────────────────────────────────────────────────────┤
│ Entrada: Trades individuais                                │
│ Saída: Candles + Microestrutura                           │
│                                                             │
│ Processo:                                                   │
│ 1. Agrupar trades por timeframe (1min)                     │
│ 2. Calcular OHLCV real                                     │
│ 3. Separar buy_volume/sell_volume por side                 │
│ 4. Contar buy_trades/sell_trades                           │
│ 5. Calcular métricas de microestrutura                     │
│ 6. Sincronizar com book data                               │
│                                                             │
│ Thread Dedicada: RealTimeAggregator                        │
│ Buffer: Circular buffer para trades                        │
│ Sync: Event-driven com timestamps                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 ESTRUTURA DE DADOS UNIFICADA
```
┌─────────────────────────────────────────────────────────────┐
│                UNIFIED DATA STRUCTURE                      │
├─────────────────────────────────────────────────────────────┤
│ TradingDataStructure:                                       │
│                                                             │
│ ├─ candles: DataFrame                                       │
│ │  ├─ datetime (index)                                      │
│ │  ├─ open, high, low, close                               │
│ │  ├─ volume, quantity, trades                             │
│ │  ├─ bid, ask, spread, mid_price                          │
│ │  └─ contract, ticker                                      │
│ │                                                           │
│ ├─ microstructure: DataFrame                               │
│ │  ├─ datetime (index)                                      │
│ │  ├─ buy_volume, sell_volume                              │
│ │  ├─ buy_trades, sell_trades                              │
│ │  ├─ buy_qty, sell_qty                                    │
│ │  ├─ volume_imbalance, trade_imbalance                    │
│ │  ├─ avg_buy_size, avg_sell_size                          │
│ │  ├─ vwap_period, price_volatility                        │
│ │  └─ buy_pressure, sell_pressure                          │
│ │                                                           │
│ ├─ book: DataFrame                                          │
│ │  ├─ datetime (index)                                      │
│ │  ├─ bid_price_1-5, bid_qty_1-5                          │
│ │  ├─ ask_price_1-5, ask_qty_1-5                          │
│ │  ├─ total_bid_qty, total_ask_qty                         │
│ │  └─ book_imbalance, weighted_mid                         │
│ │                                                           │
│ ├─ indicators: DataFrame (calculado)                       │
│ │  ├─ ema_5, ema_9, ema_20, ema_50                        │
│ │  ├─ rsi, macd, bollinger                                │
│ │  ├─ atr, adx, stochastic                                │
│ │  └─ custom indicators                                    │
│ │                                                           │
│ └─ features: DataFrame (calculado)                         │
│    ├─ momentum_1, momentum_5, momentum_10                  │
│    ├─ volatility_5, volatility_20                         │
│    ├─ volume_ratio_20, buy_pressure                       │
│    ├─ microstructure features                             │
│    └─ composite features                                   │
└─────────────────────────────────────────────────────────────┘
```

## 3. PIPELINE DE FEATURES

### 3.1 FLUXO DE CÁLCULO
```
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│ Input: TradingDataStructure                                 │
│                                                             │
│ Step 1: Technical Indicators                               │
│ ├─ TechnicalIndicators.calculate_all(candles)             │
│ ├─ Usa apenas: OHLCV + volume                             │
│ └─ Output: indicators DataFrame                            │
│                                                             │
│ Step 2: ML Features                                        │
│ ├─ MLFeatures.calculate_all(candles, microstructure, indicators) │
│ ├─ Momentum features (usa OHLC + microestrutura)          │
│ ├─ Volatility features (usa OHLC + trades)                │
│ ├─ Volume features (usa volume real + microestrutura)     │
│ ├─ Microstructure features (usa dados agregados)          │
│ ├─ Composite features (combina tudo)                      │
│ └─ Output: features DataFrame                              │
│                                                             │
│ Step 3: Feature Validation                                 │
│ ├─ NaN handling com RobustNaNHandler                      │
│ ├─ Quality score calculation                              │
│ ├─ Feature selection baseada em importância               │
│ └─ Output: validated features                             │
│                                                             │
│ Threading: Parallel calculation onde possível             │
│ Caching: Results cached por timestamp                     │
│ Monitoring: Performance e quality metrics                 │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 DEPENDÊNCIAS DE FEATURES
```
┌─────────────────────────────────────────────────────────────┐
│               FEATURE DEPENDENCIES                         │
├─────────────────────────────────────────────────────────────┤
│ BÁSICAS (só precisam OHLCV):                              │
│ ├─ momentum_1, momentum_5, momentum_10                     │
│ ├─ volatility_5, volatility_20                            │
│ ├─ ema_20, rsi, atr, macd                                 │
│ └─ returns, log_returns                                    │
│                                                             │
│ VOLUME (precisam volume real):                            │
│ ├─ volume_ratio_20, volume_ma                             │
│ ├─ vwap, price_to_vwap                                    │
│ └─ obv, pvt_momentum                                       │
│                                                             │
│ MICROESTRUTURA (precisam trades agregados):               │
│ ├─ buy_pressure, sell_pressure                            │
│ ├─ volume_imbalance, trade_imbalance                      │
│ ├─ avg_trade_size, trade_intensity                        │
│ └─ microstructure_momentum                                 │
│                                                             │
│ BOOK (precisam bid/ask real):                             │
│ ├─ spread, spread_pct                                     │
│ ├─ book_imbalance, weighted_mid                           │
│ └─ bid_ask_pressure                                       │
│                                                             │
│ COMPOSTAS (combinam múltiplas fontes):                    │
│ ├─ regime_features (OHLC + volume + microestrutura)      │
│ ├─ momentum_volume (momentum + volume real)               │
│ └─ volatility_microstructure                             │
└─────────────────────────────────────────────────────────────┘
```

## 4. SISTEMA ML

### 4.1 FLUXO DE PREDIÇÃO
```
┌─────────────────────────────────────────────────────────────┐
│                  ML PREDICTION FLOW                        │
├─────────────────────────────────────────────────────────────┤
│ Input: Validated features                                   │
│                                                             │
│ Step 1: Regime Detection                                   │
│ ├─ RegimeAnalyzer.analyze_market(unified_data)            │
│ ├─ Usa: OHLC + EMAs + ADX + volume                        │
│ ├─ Output: regime_info (trend_up/trend_down/range)        │
│ └─ Confidence score                                        │
│                                                             │
│ Step 2: Model Selection                                    │
│ ├─ ModelManager.get_models_for_regime(regime)             │
│ ├─ Carrega modelos específicos para o regime              │
│ └─ Verifica features requeridas                           │
│                                                             │
│ Step 3: Feature Preparation                               │
│ ├─ Seleciona features requeridas pelos modelos            │
│ ├─ Normalização/scaling se necessário                     │
│ ├─ Handle missing features com fallback                   │
│ └─ Quality check final                                     │
│                                                             │
│ Step 4: Prediction                                         │
│ ├─ PredictionEngine.predict_by_regime()                   │
│ ├─ Ensemble prediction se múltiplos modelos               │
│ ├─ Confidence calculation                                  │
│ └─ Output: direction, magnitude, confidence                │
│                                                             │
│ Step 5: Validation                                         │
│ ├─ MLCoordinator.validate_prediction()                    │
│ ├─ Threshold checking por regime                          │
│ ├─ Risk checks                                            │
│ └─ Output: can_trade decision                             │
└─────────────────────────────────────────────────────────────┘
```

## 5. GERAÇÃO DE SINAIS

### 5.1 SIGNAL PIPELINE
```
┌─────────────────────────────────────────────────────────────┐
│                  SIGNAL GENERATION                         │
├─────────────────────────────────────────────────────────────┤
│ Input: ML prediction + market data                         │
│                                                             │
│ Step 1: Signal Validation                                  │
│ ├─ SignalGenerator.validate_prediction()                  │
│ ├─ Check direction, magnitude, confidence thresholds      │
│ ├─ Regime-specific validation                             │
│ └─ Market condition checks                                │
│                                                             │
│ Step 2: Entry Logic                                        │
│ ├─ Determine entry price (current bid/ask)                │
│ ├─ Calculate position size based on risk                  │
│ ├─ Set stop loss using ATR real                           │
│ ├─ Set take profit based on regime                        │
│ └─ Add slippage estimates                                  │
│                                                             │
│ Step 3: Risk Management                                    │
│ ├─ RiskManager.validate_signal()                          │
│ ├─ Position size limits                                   │
│ ├─ Daily loss limits                                      │
│ ├─ Market hours validation                                │
│ └─ Correlation checks                                     │
│                                                             │
│ Step 4: Signal Output                                      │
│ ├─ action: BUY/SELL/HOLD                                  │
│ ├─ entry_price: Real bid/ask price                       │
│ ├─ position_size: Contracts to trade                     │
│ ├─ stop_loss: Risk-based exit                            │
│ ├─ take_profit: Target exit                              │
│ ├─ confidence: ML confidence score                       │
│ └─ metadata: Full prediction context                     │
└─────────────────────────────────────────────────────────────┘
```

## 6. EXECUÇÃO

### 6.1 ORDER MANAGEMENT
```
┌─────────────────────────────────────────────────────────────┐
│                  ORDER EXECUTION                           │
├─────────────────────────────────────────────────────────────┤
│ Input: Trading signal                                       │
│                                                             │
│ Step 1: Order Preparation                                  │
│ ├─ OrderManager.prepare_order()                           │
│ ├─ Convert signal to ProfitDLL order format               │
│ ├─ Add order metadata and tracking                        │
│ └─ Final validation                                        │
│                                                             │
│ Step 2: Market Execution                                   │
│ ├─ ConnectionManager.send_order()                         │
│ ├─ DLLSendOrder() para execução                           │
│ ├─ Order status tracking                                  │
│ └─ Fill confirmation                                       │
│                                                             │
│ Step 3: Position Tracking                                 │
│ ├─ Update position in PositionManager                     │
│ ├─ Set stop loss orders                                   │
│ ├─ Set take profit orders                                 │
│ └─ Monitor position P&L                                    │
│                                                             │
│ Step 4: Performance Tracking                              │
│ ├─ Log trade details                                      │
│ ├─ Update performance metrics                             │
│ ├─ Feed back to adaptive systems                          │
│ └─ Risk monitoring                                         │
└─────────────────────────────────────────────────────────────┘
```

## 7. MONITORAMENTO E FEEDBACK

### 7.1 SYSTEM MONITORING
```
┌─────────────────────────────────────────────────────────────┐
│               SYSTEM MONITORING                            │
├─────────────────────────────────────────────────────────────┤
│ Data Quality:                                              │
│ ├─ FeatureDebugger monitoring                             │
│ ├─ NaN percentages tracking                               │
│ ├─ Data latency monitoring                                │
│ └─ Callback health checks                                 │
│                                                             │
│ ML Performance:                                            │
│ ├─ Prediction accuracy tracking                           │
│ ├─ Feature importance drift                               │
│ ├─ Model performance by regime                            │
│ └─ Confidence calibration                                 │
│                                                             │
│ Trading Performance:                                       │
│ ├─ Signal generation rate                                 │
│ ├─ Execution quality                                      │
│ ├─ Slippage tracking                                      │
│ └─ P&L attribution                                        │
│                                                             │
│ System Health:                                             │
│ ├─ Memory usage                                           │
│ ├─ Processing latency                                     │
│ ├─ Connection stability                                   │
│ └─ Error rates                                            │
└─────────────────────────────────────────────────────────────┘
```

## 8. VALIDAÇÃO E TESTES

### 8.1 TESTING PIPELINE
```
┌─────────────────────────────────────────────────────────────┐
│                    TESTING FLOW                           │
├─────────────────────────────────────────────────────────────┤
│ Historical Validation:                                     │
│ ├─ Load real historical data via DLL                      │
│ ├─ Calculate all features using real data                 │
│ ├─ Train models with consistent data                      │
│ ├─ Validate with walk-forward analysis                    │
│ └─ Performance attribution analysis                       │
│                                                             │
│ Real-time Testing:                                        │
│ ├─ Paper trading with live data                          │
│ ├─ Feature consistency validation                         │
│ ├─ Latency and performance testing                        │
│ ├─ Edge case handling                                     │
│ └─ System stability testing                               │
│                                                             │
│ Integration Testing:                                       │
│ ├─ End-to-end data flow testing                           │
│ ├─ ProfitDLL integration testing                          │
│ ├─ Error handling and recovery                            │
│ ├─ Memory and resource testing                            │
│ └─ Concurrent access testing                              │
└─────────────────────────────────────────────────────────────┘
```

## CRÍTICO: PONTOS DE VALIDAÇÃO

1. **Dados Históricos = Dados Tempo Real**: Features calculadas devem ser idênticas
2. **Qualidade de Dados**: Monitoramento contínuo de NaN e outliers  
3. **Consistência Temporal**: Timestamps sincronizados entre todas as fontes
4. **Performance**: Latência < 100ms para cálculo de features
5. **Robustez**: Sistema deve funcionar mesmo com falhas parciais de dados

---

**PRÓXIMOS PASSOS:**
1. Implementar `load_complete_historical_data.py` 
2. Criar novo `TradingDataStructure` 
3. Refazer pipeline de features
4. Retreinar modelos ML
5. Validar consistência histórico vs. tempo real