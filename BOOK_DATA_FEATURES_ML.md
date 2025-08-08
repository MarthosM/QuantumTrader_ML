# Informações do Book Collector para Treinamento de ML

## Dados Consolidados Disponíveis

### 1. **Tiny Book** (Melhores Ofertas - Top of Book)
Contém as melhores ofertas de compra e venda a cada momento.

**Campos principais:**
- `price`: Preço da melhor oferta
- `quantity`: Quantidade disponível
- `side`: Lado da oferta ('bid' ou 'ask')
- `timestamp`: Momento exato (precisão de microsegundos)

**Uso em ML:**
- Calcular spread instantâneo
- Detectar pressão compradora/vendedora
- Identificar momentum de preço
- Prever direção de curto prazo

### 2. **Offer Book** (Livro de Ofertas Completo)
Snapshot completo do livro de ofertas com todas as ordens.

**Campos principais:**
- `price`: Preço de cada ordem
- `quantity`: Quantidade de cada ordem
- `side`: Compra (0) ou Venda (1)
- `position`: Posição/nível no livro (profundidade)
- `action`: Tipo de mudança (New=0, Edit=1, Delete=2)
- `agent`: Código do participante
- `offer_id`: ID único da ordem

**Uso em ML:**
- Analisar profundidade e liquidez
- Detectar grandes ordens (icebergs)
- Identificar suporte/resistência
- Prever impacto de mercado

### 3. **Daily** (Dados Agregados)
Resumo agregado com OHLC e volume.

**Campos principais:**
- `open`, `high`, `low`, `close`: Preços OHLC
- `volume_delta`: Volume incremental negociado
- `qty_delta`: Quantidade incremental de contratos
- `trades_delta`: Número incremental de negócios
- `volume_buyer`/`volume_seller`: Volume por lado
- `contracts_open`: Contratos em aberto

**Uso em ML:**
- Análise de tendência
- Detecção de breakouts
- Volume profile analysis
- Sentimento de mercado

## Features Prontas para ML

### Microestrutura de Mercado
```python
# Spread
spread = ask_price - bid_price

# Mid price
mid_price = (ask_price + bid_price) / 2

# Imbalance de quantidade
qty_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)

# Pressão de preço
price_pressure = bid_qty / (bid_qty + ask_qty)
```

### Dinâmica do Livro
```python
# Profundidade ponderada
weighted_bid_depth = sum(price * qty for price, qty in bid_levels)
weighted_ask_depth = sum(price * qty for price, qty in ask_levels)

# Inclinação do livro
book_slope = (ask_prices - bid_prices).mean()

# Velocidade de mudanças
order_flow_rate = num_new_orders / time_window
```

### Indicadores de Fluxo
```python
# VWAP (Volume Weighted Average Price)
vwap = sum(price * volume) / sum(volume)

# Trade size médio
avg_trade_size = volume_delta / trades_delta

# Buy/Sell ratio
buy_sell_ratio = volume_buyer / volume_seller

# Rate de volume
volume_rate = volume_delta / time_delta
```

### Features Temporais
```python
# Hora do dia (ciclical encoding)
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

# Minutos desde abertura
minutes_since_open = (current_time - market_open).seconds / 60

# Proximidade do fechamento
minutes_to_close = (market_close - current_time).seconds / 60
```

## Exemplo de Pipeline de Features

```python
import pandas as pd
import numpy as np

def create_ml_features(tiny_book_df, offer_book_df, daily_df):
    """Cria features para ML a partir dos dados do book"""
    
    features = pd.DataFrame()
    
    # 1. Features de Spread (Tiny Book)
    bid_data = tiny_book_df[tiny_book_df['side'] == 'bid']
    ask_data = tiny_book_df[tiny_book_df['side'] == 'ask']
    
    # Merge por timestamp mais próximo
    merged = pd.merge_asof(
        bid_data.sort_values('timestamp'),
        ask_data.sort_values('timestamp'),
        on='timestamp',
        suffixes=('_bid', '_ask'),
        direction='nearest'
    )
    
    features['spread'] = merged['price_ask'] - merged['price_bid']
    features['mid_price'] = (merged['price_ask'] + merged['price_bid']) / 2
    features['qty_imbalance'] = (merged['quantity_bid'] - merged['quantity_ask']) / \
                                (merged['quantity_bid'] + merged['quantity_ask'])
    
    # 2. Features de Profundidade (Offer Book)
    # Agregar por timestamp e lado
    depth_features = offer_book_df.groupby(['timestamp', 'side']).agg({
        'quantity': ['sum', 'mean', 'std'],
        'price': ['min', 'max', 'std'],
        'position': 'max'  # profundidade máxima
    })
    
    # 3. Features de Volume (Daily)
    daily_features = daily_df.copy()
    daily_features['volume_rate'] = daily_features['volume_delta'] / \
                                    daily_features['timestamp'].diff().dt.total_seconds()
    daily_features['avg_trade_size'] = daily_features['volume_delta'] / \
                                       daily_features['trades_delta']
    daily_features['buy_pressure'] = daily_features['volume_buyer'] / \
                                     (daily_features['volume_buyer'] + daily_features['volume_seller'])
    
    # 4. Features Temporais
    features['hour'] = features.index.hour
    features['minute'] = features.index.minute
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    
    # 5. Features de Janela Móvel
    features['spread_ma_5min'] = features['spread'].rolling('5T').mean()
    features['volume_std_10min'] = daily_features['volume_delta'].rolling('10T').std()
    features['price_momentum'] = features['mid_price'].pct_change(periods=10)
    
    return features
```

## Vantagens dos Dados de Book vs CSV Tradicional

| Aspecto | Book Data | CSV Tradicional |
|---------|-----------|-----------------|
| **Granularidade** | Microsegundos | Minutos/Diário |
| **Profundidade** | Livro completo | Apenas OHLCV |
| **Microestrutura** | Spread, imbalance, fluxo | Não disponível |
| **Participantes** | IDs dos agentes | Não disponível |
| **Dinâmica** | Mudanças em tempo real | Snapshots agregados |

## Aplicações em HMARL

Os dados do Book são ideais para modelos HMARL porque capturam:

1. **Hierarquia Natural**:
   - Nível 1: Tiny Book (decisões rápidas)
   - Nível 2: Offer Book (táticas de mercado)
   - Nível 3: Daily aggregates (estratégia geral)

2. **Multi-Agente**:
   - Identificar comportamento de diferentes participantes
   - Modelar interações entre agentes
   - Detectar padrões de grandes players

3. **Reinforcement Learning**:
   - Estados ricos em informação
   - Rewards baseados em execução real
   - Simulação realista de impacto de mercado

## Integração com Sistema Existente

```python
# Carregar dados consolidados
from src.training.flexible_data_loader import FlexibleBookDataLoader

loader = FlexibleBookDataLoader()

# Opção 1: Carregar tudo
all_data = loader.load_data('data/realtime/book/20250805/consolidated/consolidated_complete_20250805.parquet')

# Opção 2: Carregar por tipo
book_data = loader.load_data(
    'data/realtime/book/20250805/consolidated/',
    data_types=['tiny_book', 'offer_book']
)

# Criar features
features = create_ml_features(
    book_data[book_data['type'] == 'tiny_book'],
    book_data[book_data['type'] == 'offer_book'],
    book_data[book_data['type'] == 'daily']
)
```