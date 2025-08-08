# 📊 Mapeamento Completo das 65 Features

## Resumo
Este documento mapeia todas as 65 features necessárias para os modelos ML treinados, baseado no arquivo `models/csv_5m_realistic/features_20250807_061838.csv`.

## 📈 Features por Categoria

### 1. Volatilidade (10 features)

| # | Nome | Descrição | Fórmula/Cálculo | Dados Necessários |
|---|------|-----------|-----------------|-------------------|
| 1 | `volatility_10` | Desvio padrão dos retornos (10 períodos) | `std(returns[-10:])` | Últimos 10 candles |
| 2 | `volatility_20` | Desvio padrão dos retornos (20 períodos) | `std(returns[-20:])` | Últimos 20 candles |
| 3 | `volatility_50` | Desvio padrão dos retornos (50 períodos) | `std(returns[-50:])` | Últimos 50 candles |
| 4 | `volatility_100` | Desvio padrão dos retornos (100 períodos) | `std(returns[-100:])` | Últimos 100 candles |
| 5 | `volatility_ratio_10` | Razão volatilidade 10/20 | `volatility_10 / volatility_20` | volatility_10, volatility_20 |
| 6 | `volatility_ratio_20` | Razão volatilidade 20/50 | `volatility_20 / volatility_50` | volatility_20, volatility_50 |
| 7 | `volatility_ratio_50` | Razão volatilidade 50/100 | `volatility_50 / volatility_100` | volatility_50, volatility_100 |
| 8 | `volatility_ratio_100` | Razão volatilidade 100/200 | `volatility_100 / volatility_200` | volatility_100, volatility_200 |
| 9 | `volatility_gk` | Garman-Klass volatility | `sqrt(sum((log(H/L))^2)/n)` | High, Low de cada candle |
| 10 | `bb_position` | Posição nas Bollinger Bands | `(price - BB_lower) / (BB_upper - BB_lower)` | Preço, BB superior e inferior |

### 2. Retornos (10 features)

| # | Nome | Descrição | Fórmula/Cálculo | Dados Necessários |
|---|------|-----------|-----------------|-------------------|
| 11 | `returns_1` | Retorno 1 período | `(price[t] - price[t-1]) / price[t-1]` | Últimos 2 preços |
| 12 | `returns_2` | Retorno 2 períodos | `(price[t] - price[t-2]) / price[t-2]` | Últimos 3 preços |
| 13 | `returns_5` | Retorno 5 períodos | `(price[t] - price[t-5]) / price[t-5]` | Últimos 6 preços |
| 14 | `returns_10` | Retorno 10 períodos | `(price[t] - price[t-10]) / price[t-10]` | Últimos 11 preços |
| 15 | `returns_20` | Retorno 20 períodos | `(price[t] - price[t-20]) / price[t-20]` | Últimos 21 preços |
| 16 | `returns_50` | Retorno 50 períodos | `(price[t] - price[t-50]) / price[t-50]` | Últimos 51 preços |
| 17 | `returns_100` | Retorno 100 períodos | `(price[t] - price[t-100]) / price[t-100]` | Últimos 101 preços |
| 18 | `log_returns_1` | Log retorno 1 período | `log(price[t] / price[t-1])` | Últimos 2 preços |
| 19 | `log_returns_5` | Log retorno 5 períodos | `log(price[t] / price[t-5])` | Últimos 6 preços |
| 20 | `log_returns_20` | Log retorno 20 períodos | `log(price[t] / price[t-20])` | Últimos 21 preços |

### 3. Order Flow (8 features)

| # | Nome | Descrição | Fórmula/Cálculo | Dados Necessários |
|---|------|-----------|-----------------|-------------------|
| 21 | `order_flow_imbalance_10` | Desequilíbrio de fluxo (10 per) | `(buy_vol - sell_vol) / (buy_vol + sell_vol)` | Volume compra/venda 10 per |
| 22 | `order_flow_imbalance_20` | Desequilíbrio de fluxo (20 per) | `(buy_vol - sell_vol) / (buy_vol + sell_vol)` | Volume compra/venda 20 per |
| 23 | `order_flow_imbalance_50` | Desequilíbrio de fluxo (50 per) | `(buy_vol - sell_vol) / (buy_vol + sell_vol)` | Volume compra/venda 50 per |
| 24 | `order_flow_imbalance_100` | Desequilíbrio de fluxo (100 per) | `(buy_vol - sell_vol) / (buy_vol + sell_vol)` | Volume compra/venda 100 per |
| 25 | `cumulative_signed_volume` | Volume assinado acumulado | `sum(signed_volume)` | Todos os volumes assinados |
| 26 | `signed_volume` | Volume com direção | `volume * sign(price_change)` | Volume e direção do preço |
| 27 | `volume_weighted_return` | Retorno ponderado por volume | `sum(return * volume) / sum(volume)` | Retornos e volumes |
| 28 | `agent_turnover` | Rotatividade de agentes | `unique_traders / total_trades` | IDs de traders, total trades |

### 4. Volume (8 features)

| # | Nome | Descrição | Fórmula/Cálculo | Dados Necessários |
|---|------|-----------|-----------------|-------------------|
| 29 | `volume_ratio_20` | Razão volume atual/média 20 | `volume / mean(volume[-20:])` | Últimos 20 volumes |
| 30 | `volume_ratio_50` | Razão volume atual/média 50 | `volume / mean(volume[-50:])` | Últimos 50 volumes |
| 31 | `volume_ratio_100` | Razão volume atual/média 100 | `volume / mean(volume[-100:])` | Últimos 100 volumes |
| 32 | `volume_zscore_20` | Z-score do volume (20 per) | `(volume - mean) / std` | Média e desvio 20 per |
| 33 | `volume_zscore_50` | Z-score do volume (50 per) | `(volume - mean) / std` | Média e desvio 50 per |
| 34 | `volume_zscore_100` | Z-score do volume (100 per) | `(volume - mean) / std` | Média e desvio 100 per |
| 35 | `trade_intensity` | Intensidade de negociação | `num_trades / time_interval` | Número de trades, tempo |
| 36 | `trade_intensity_ratio` | Razão intensidade atual/média | `intensity / mean(intensity)` | Intensidades históricas |

### 5. Indicadores Técnicos (8 features)

| # | Nome | Descrição | Fórmula/Cálculo | Dados Necessários |
|---|------|-----------|-----------------|-------------------|
| 37 | `ma_5_20_ratio` | Razão MA5/MA20 | `MA(5) / MA(20)` | Médias móveis 5 e 20 |
| 38 | `ma_20_50_ratio` | Razão MA20/MA50 | `MA(20) / MA(50)` | Médias móveis 20 e 50 |
| 39 | `momentum_5_20` | Momentum 5 vs 20 períodos | `MA(5) - MA(20)` | Médias móveis 5 e 20 |
| 40 | `momentum_20_50` | Momentum 20 vs 50 períodos | `MA(20) - MA(50)` | Médias móveis 20 e 50 |
| 41 | `sharpe_5` | Sharpe ratio (5 períodos) | `mean(returns) / std(returns)` | Retornos 5 períodos |
| 42 | `sharpe_20` | Sharpe ratio (20 períodos) | `mean(returns) / std(returns)` | Retornos 20 períodos |
| 43 | `time_normalized` | Tempo normalizado do dia | `seconds_since_open / total_seconds` | Horário atual |
| 44 | `bb_position` | (Duplicada - já listada) | - | - |

### 6. Microestrutura de Mercado (15 features)

| # | Nome | Descrição | Fórmula/Cálculo | Dados Necessários |
|---|------|-----------|-----------------|-------------------|
| 45 | `top_buyer_0_active` | Top comprador 0 ativo | `1 if active else 0` | Book de ofertas, ID traders |
| 46 | `top_buyer_1_active` | Top comprador 1 ativo | `1 if active else 0` | Book de ofertas, ID traders |
| 47 | `top_buyer_2_active` | Top comprador 2 ativo | `1 if active else 0` | Book de ofertas, ID traders |
| 48 | `top_buyer_3_active` | Top comprador 3 ativo | `1 if active else 0` | Book de ofertas, ID traders |
| 49 | `top_buyer_4_active` | Top comprador 4 ativo | `1 if active else 0` | Book de ofertas, ID traders |
| 50 | `top_seller_0_active` | Top vendedor 0 ativo | `1 if active else 0` | Book de ofertas, ID traders |
| 51 | `top_seller_1_active` | Top vendedor 1 ativo | `1 if active else 0` | Book de ofertas, ID traders |
| 52 | `top_seller_2_active` | Top vendedor 2 ativo | `1 if active else 0` | Book de ofertas, ID traders |
| 53 | `top_seller_3_active` | Top vendedor 3 ativo | `1 if active else 0` | Book de ofertas, ID traders |
| 54 | `top_seller_4_active` | Top vendedor 4 ativo | `1 if active else 0` | Book de ofertas, ID traders |
| 55 | `top_buyers_count` | Número de top compradores | `count(top_buyers)` | Lista de top compradores |
| 56 | `top_sellers_count` | Número de top vendedores | `count(top_sellers)` | Lista de top vendedores |
| 57 | `buyer_changed` | Mudança no top comprador | `1 if changed else 0` | Histórico de top buyers |
| 58 | `seller_changed` | Mudança no top vendedor | `1 if changed else 0` | Histórico de top sellers |
| 59 | `is_buyer_aggressor` | Comprador é agressor | `1 if buyer_initiated else 0` | Flags de iniciador do trade |
| 60 | `is_seller_aggressor` | Vendedor é agressor | `1 if seller_initiated else 0` | Flags de iniciador do trade |

### 7. Features Temporais (6 features)

| # | Nome | Descrição | Fórmula/Cálculo | Dados Necessários |
|---|------|-----------|-----------------|-------------------|
| 61 | `minute` | Minuto do dia | `timestamp.minute` | Timestamp atual |
| 62 | `hour` | Hora do dia | `timestamp.hour` | Timestamp atual |
| 63 | `day_of_week` | Dia da semana | `timestamp.weekday()` | Timestamp atual |
| 64 | `is_opening_30min` | Primeiros 30 min | `1 if minutes_since_open < 30 else 0` | Horário de abertura |
| 65 | `is_closing_30min` | Últimos 30 min | `1 if minutes_to_close < 30 else 0` | Horário de fechamento |
| 66 | `is_lunch_hour` | Horário de almoço | `1 if 12 <= hour < 13 else 0` | Hora atual |

## 📊 Resumo de Dados Necessários

### Callbacks Essenciais
1. **Daily Callback**: Candles OHLCV
2. **PriceBook Callback**: Book de preços (bid/ask com volumes)
3. **OfferBook Callback**: Book de ofertas detalhado
4. **TinyBook Callback**: Book resumido para top levels
5. **Trade Callback**: Detalhes de cada trade executado

### Buffers Necessários
- **Candles**: Mínimo 200 períodos (para volatility_100 e retornos)
- **Book Data**: Mínimo 100 snapshots
- **Trades**: Mínimo 1000 trades recentes
- **Order Flow**: Agregação por período (10, 20, 50, 100)

### Dados de Microestrutura
- **Identificação de traders**: Para rastrear top buyers/sellers
- **Flags de agressão**: Quem iniciou cada trade
- **Timestamps precisos**: Para features temporais
- **Volume por lado**: Separar volume de compra e venda

## 🔄 Prioridade de Implementação

### Alta Prioridade (Core - 20 features)
- Todas as volatilidades (4)
- Retornos principais (returns_1, 5, 20, 50, 100)
- Order flow imbalance (4 períodos)
- Volume ratios (3)
- Features temporais (6)

### Média Prioridade (Enhancement - 25 features)
- Volatility ratios (4)
- Log returns (3)
- Volume z-scores (3)
- Indicadores técnicos (8)
- Signed volumes (3)

### Baixa Prioridade (Advanced - 20 features)
- Microestrutura detalhada (15)
- Trade intensity (2)
- Agent turnover (1)
- Volatility GK (1)
- BB position (1)

## 📝 Notas de Implementação

1. **Cálculo Incremental**: Muitas features podem ser calculadas incrementalmente para melhor performance
2. **Cache**: Features que não mudam frequentemente (ex: day_of_week) devem ser cacheadas
3. **Validação**: Cada feature deve ter validação de range e tratamento de NaN
4. **Fallback**: Ter valores default para quando dados estão faltando

---

*Documento criado: 08/08/2025*
*Fonte: models/csv_5m_realistic/features_20250807_061838.csv*