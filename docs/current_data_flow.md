# 📈 Fluxo de Dados Atual - Sistema QuantumTrader ML

## Visão Geral
Documentação do fluxo de dados atual no sistema, desde a recepção via ProfitDLL até a geração de sinais.

## 🔄 Fluxo Atual (production_fixed.py)

### 1. Inicialização
```
start_hmarl_production_enhanced.py
    └── EnhancedHMARLProductionSystem (herda de ProductionFixedSystem)
        └── ProductionFixedSystem
            ├── _load_ml_models() - Carrega modelos .pkl
            ├── initialize() - Conecta ProfitDLL
            └── _create_all_callbacks() - Registra 7 callbacks
```

### 2. Callbacks Registrados

```python
# production_fixed.py - Linha 240-260
self.callback_refs = {
    'state': StateCallbackFunc(self._on_state_callback),      # Status conexão
    'history': HistoryCallbackFunc(self._on_history_callback), # Histórico
    'daily': DailyCallbackFunc(self._on_daily_callback),       # Candles OHLCV
    'price_book': PriceBookCallbackFunc(self._on_price_book),  # Book preços
    'offer_book': OfferBookCallbackFunc(self._on_offer_book),  # Book ofertas
    'progress': ProgressCallbackFunc(self._on_progress),       # Progresso
    'tiny_book': TinyBookCallbackFunc(self._on_tiny_book)      # Book resumido
}
```

### 3. Processamento de Callbacks

#### Daily Callback (Candles)
```python
def _on_daily_callback(self, asset, daily_data):
    # Recebe: OHLCV data
    # Armazena em: self.candles[]
    # Usado para: Cálculo de features básicas
    
    candle = {
        'time': daily_data.TradeTime,
        'open': daily_data.sOpen,
        'high': daily_data.sHigh,
        'low': daily_data.sLow,
        'close': daily_data.sClose,
        'volume': daily_data.sVol
    }
    self.candles.append(candle)
```

#### PriceBook Callback
```python
def _on_price_book(self, asset, book_type, position, old_data, new_data):
    # Recebe: Mudanças no book de preços
    # Armazena em: Não está armazenando!
    # Problema: Dados perdidos para order flow
    self.callbacks['price_book'] += 1  # Apenas conta
```

#### TinyBook Callback
```python
def _on_tiny_book(self, asset, book_data):
    # Recebe: Top 5 níveis bid/ask
    # Armazena em: Não está armazenando!
    # Problema: Dados perdidos para microestrutura
    
    # Apenas loga primeiro nível
    if book_data[0].Offer[0].Price > 0:
        self.logger.info(f"BID: {book_data[0].Offer[0].Price}")
```

### 4. Cálculo de Features (LIMITADO)

```python
def _calculate_features(self):
    # Calcula apenas 11 features básicas:
    features = {
        'price_current': self.current_price,
        'price_mean_5': mean(prices[-5:]),
        'price_mean_20': mean(prices[-20:]),
        'price_std_20': std(prices[-20:]),
        'return_1': (price[t] - price[t-1]) / price[t-1],
        'return_mean_5': mean(returns[-5:]),
        'return_std_5': std(returns[-5:]),
        'volume_mean_5': mean(volumes[-5:]),
        'volume_ratio': volume / mean(volumes[-5:]),
        'rsi_14': RSI calculation,
        'momentum_10': (price[t] / price[t-10]) - 1
    }
    return features
```

### 5. Predição ML

```python
def _make_prediction(self):
    features = self._calculate_features()  # 11 features
    
    # PROBLEMA: Modelos esperam 65 features!
    for model in self.models:
        X = create_feature_vector(features)  # 11 valores
        prediction = model.predict(X)  # ERRO: Shape mismatch
```

## 🚨 Problemas Identificados

### 1. Dados de Book Não Armazenados
- **PriceBook** e **TinyBook** callbacks apenas contam chamadas
- Sem buffer para dados históricos de book
- Impossível calcular order flow imbalance

### 2. Trades Não Capturados
- Sem callback para trades individuais
- Não identifica agressores (buyer/seller)
- Sem informação de traders únicos

### 3. Features Incompletas
- Apenas 11 de 65 features calculadas
- Sem features de microestrutura
- Sem order flow metrics
- Sem volatility ratios

### 4. Falta de Sincronização
- Callbacks assíncronos sem sincronização
- Possível defasagem entre diferentes dados
- Sem timestamp unificado

## 🔧 Mudanças Necessárias

### 1. Adicionar Buffers
```python
class EnhancedProductionSystem:
    def __init__(self):
        self.candle_buffer = CircularBuffer(200)
        self.book_buffer = CircularBuffer(100)
        self.trade_buffer = CircularBuffer(1000)
```

### 2. Armazenar Dados de Book
```python
def _on_price_book(self, asset, book_type, position, old_data, new_data):
    book_snapshot = {
        'timestamp': time.time(),
        'bid_price': [new_data.Price],
        'bid_volume': [new_data.Qtd],
        # ... mais dados
    }
    self.book_buffer.add(book_snapshot)
```

### 3. Implementar Feature Calculator
```python
def _calculate_all_features(self):
    # Base features (30)
    volatility_features = self.calculate_volatility_features()
    return_features = self.calculate_return_features()
    
    # Book features (20)
    order_flow_features = self.calculate_order_flow_features()
    microstructure_features = self.calculate_microstructure_features()
    
    # ... todas as 65 features
    return all_features
```

## 📊 Comparação: Atual vs Necessário

| Aspecto | Sistema Atual | Sistema Necessário |
|---------|--------------|-------------------|
| **Callbacks ativos** | 7 (mas só usa 2) | 7 (todos utilizados) |
| **Dados armazenados** | Apenas candles | Candles + Book + Trades |
| **Features calculadas** | 11 básicas | 65 completas |
| **Buffers** | Lista simples | CircularBuffer otimizado |
| **Sincronização** | Nenhuma | Timestamp unificado |
| **Performance** | ~10ms | Target < 200ms |

## 🎯 Próximos Passos

1. **Implementar buffers circulares** para todos os tipos de dados
2. **Modificar callbacks** para armazenar dados completos
3. **Criar BookFeatureEngineer** para calcular 65 features
4. **Sincronizar dados** com timestamp unificado
5. **Validar features** com dados históricos

---

*Documento criado: 08/08/2025*
*Baseado em: production_fixed.py e start_hmarl_production_enhanced.py*