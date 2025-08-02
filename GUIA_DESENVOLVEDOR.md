# 📖 GUIA DO DESENVOLVEDOR - QuantumTrader_ML v2.0

## 🎯 Visão Geral

O QuantumTrader_ML é um sistema de trading algorítmico baseado em Machine Learning que integra análise de mercado em tempo real, detecção de regime e execução automatizada através da ProfitDLL v4.0.

### Arquitetura Principal
- **Linguagem**: Python 3.8+
- **Framework ML**: TensorFlow/Keras, XGBoost, LightGBM
- **Dados**: ProfitDLL para market data real-time
- **Threading**: Processamento assíncrono multi-thread
- **Padrão**: Orientado a objetos com injeção de dependência

## 🏗️ Classes Principais e Métodos

### 1. TradingSystem (`trading_system.py`)
**Orquestrador principal do sistema**

```python
class TradingSystem:
    def __init__(self, config: Dict)
    def initialize(self) -> bool
    def start(self, ticker: Optional[str] = None) -> bool
    def stop(self)
    def _load_historical_data_safe(self, ticker: str, days_back: int) -> bool
    def _calculate_initial_features(self)
    def _start_processing_threads(self)
    def _main_loop(self)
    def _get_current_contract(self, date: datetime) -> str
```

**Responsabilidades:**
- Inicializar todos os componentes
- Gerenciar ciclo de vida do sistema
- Coordenar threads de processamento
- Validar dados de produção

**Configuração Mínima:**
```python
config = {
    'dll_path': './ProfitDLL64.dll',
    'username': 'user',
    'password': 'pass',
    'models_dir': './models/',
    'historical_days': 1,
    'ml_interval': 60,  # segundos
    'use_gui': True
}
```

### 2. ConnectionManagerV4 (`connection_manager_v4.py`)
**Interface com ProfitDLL**

```python
class ConnectionManagerV4:
    def __init__(self, dll_path: str)
    def initialize(self, key: str, username: str, password: str, ...) -> bool
    def connect(self) -> bool
    def disconnect(self)
    def subscribe_ticker(self, ticker: str) -> bool
    def request_historical_data(self, ticker: str, start_date: datetime, end_date: datetime) -> int
    def wait_for_historical_data(self, timeout_seconds: int = 30) -> bool
    def send_order(self, order: SendOrder) -> int
```

**Callbacks Importantes:**
- `on_trade_callback`: Processa trades em tempo real
- `on_agg_trade_callback`: Processa trades agregados
- `on_ticker_callback`: Atualiza preços bid/ask
- `on_order_book_callback`: Atualiza livro de ofertas
- `on_historical_data_callback`: Recebe dados históricos

### 3. TradingDataStructure (`data_structure.py`)
**Estrutura centralizada de dados**

```python
class TradingDataStructure:
    # DataFrames principais
    candles: pd.DataFrame      # OHLCV
    microstructure: pd.DataFrame # Buy/sell volume, imbalance
    orderbook: pd.DataFrame     # Bid/ask, spread
    indicators: pd.DataFrame    # Technical indicators
    features: pd.DataFrame      # ML features
    
    def initialize_structure(self) -> None
    def update_candles(self, new_candles: pd.DataFrame) -> bool
    def update_microstructure(self, new_micro: pd.DataFrame) -> bool
    def update_indicators(self, new_indicators: pd.DataFrame) -> bool
    def update_features(self, new_features: pd.DataFrame) -> bool
    def check_data_quality(self) -> Dict[str, Any]
    def get_unified_dataframe(self) -> pd.DataFrame
```

**Estrutura de Dados:**
```python
# Candles (1 minuto)
{
    'open': float,
    'high': float, 
    'low': float,
    'close': float,
    'volume': float,
    'quantidade': int
}

# Microestrutura
{
    'buy_volume': float,
    'sell_volume': float,
    'buy_trades': int,
    'sell_trades': int,
    'volume_imbalance': float
}
```

### 4. FeatureEngine (`feature_engine.py`)
**Motor de cálculo de features**

```python
class FeatureEngine:
    def __init__(self, required_features: List[str], allow_historical_data: bool = True)
    def calculate(self, data: TradingDataStructure) -> Dict[str, pd.DataFrame]
    def _calculate_technical_indicators(self, candles: pd.DataFrame) -> pd.DataFrame
    def _calculate_ml_features(self, data: TradingDataStructure) -> pd.DataFrame
    def _validate_features(self, features: pd.DataFrame) -> bool
    def get_feature_stats(self) -> Dict[str, Any]
```

**Features Calculadas:**
- **Indicadores Técnicos** (~45): EMAs, RSI, MACD, Bollinger Bands, ATR, Stochastic
- **Features ML** (~80-100): Momentum, volatilidade, retornos, volume ratios, regime strength

### 5. MLCoordinator (`ml_coordinator.py`)
**Coordenador do processo ML**

```python
class MLCoordinator:
    def __init__(self, model_manager, feature_engine, prediction_engine, regime_trainer)
    def process_prediction_request(self, data) -> Optional[Dict[str, Any]]
    def force_prediction(self, data) -> Optional[Dict[str, Any]]
    def _detect_market_regime(self, data) -> Optional[Dict[str, Any]]
    def _prepare_unified_data(self, data) -> pd.DataFrame
    def _analyze_support_resistance(self, data) -> Dict[str, Any]
    def _confirm_trend_direction(self, data) -> Dict[str, Any]
    def get_coordinator_stats(self) -> Dict[str, Any]
```

**Fluxo de Predição:**
1. Detectar regime de mercado
2. Calcular features
3. Executar predição baseada no regime
4. Validar se pode operar

### 6. RegimeAnalyzer (`training/regime_analyzer.py`)
**Detecta regime de mercado**

```python
class RegimeAnalyzer:
    def __init__(self, logger=None)
    def analyze_market(self, data: pd.DataFrame) -> Dict[str, Any]
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series
    def _check_ema_alignment(self, data: pd.DataFrame) -> Tuple[bool, int]
    def _analyze_support_resistance(self, data: pd.DataFrame) -> Dict[str, float]
    def _calculate_regime_confidence(self, adx: float, regime: str) -> float
```

**Regimes Detectados:**
- `trend_up`: ADX > 25, EMAs alinhadas para cima
- `trend_down`: ADX > 25, EMAs alinhadas para baixo
- `range`: ADX < 25 (lateralização)
- `undefined`: ADX > 25 mas EMAs não alinhadas

### 7. PredictionEngine (`prediction_engine.py`)
**Motor de predições ML**

```python
class PredictionEngine:
    def __init__(self, model_manager, logger=None)
    def predict(self, features: pd.DataFrame) -> Optional[Dict[str, Any]]
    def predict_by_regime(self, features: pd.DataFrame, regime_info: Dict[str, Any]) -> Optional[Dict[str, Any]]
    def batch_predict(self, features_list: List[pd.DataFrame]) -> List[Dict[str, Any]]
    def _convert_model_result(self, model_result: Any) -> Dict[str, Any]
    def _convert_model_result_with_regime(self, model_result: Any, regime_info: Dict[str, Any]) -> Dict[str, Any]
```

**Formato de Predição:**
```python
{
    'direction': float,      # -1 (sell) a 1 (buy)
    'magnitude': float,      # Movimento esperado
    'confidence': float,     # 0 a 1
    'regime': str,          # Regime detectado
    'timestamp': str,       # ISO format
    'model_used': str       # Modelo utilizado
}
```

### 8. SignalGenerator (`signal_generator.py`)
**Gera sinais de trading**

```python
class SignalGenerator:
    def __init__(self, config: Dict)
    def generate_signal(self, prediction: Dict, market_data) -> Dict
    def _validate_thresholds(self, direction: float, magnitude: float, confidence: float) -> Optional[Dict]
    def _calculate_levels(self, action: str, current_price: float, atr: float, magnitude: float, regime: str) -> Tuple[float, float]
    def _calculate_position_size(self, current_price: float, stop_loss: float, market_data) -> int
```

**Thresholds Padrão:**
```python
{
    'direction_threshold': 0.3,
    'magnitude_threshold': 0.0001,
    'confidence_threshold': 0.6,
    'min_stop_points': 5,
    'default_risk_reward': 2.0
}
```

### 9. RiskManager (`risk_manager.py`)
**Gerencia risco das operações**

```python
class RiskManager:
    def __init__(self, config: Dict)
    def validate_signal(self, signal: Dict, account_balance: float, current_positions: Optional[List[Dict]] = None) -> Tuple[bool, str]
    def validate_signal_ml(self, signal: Dict, market_data: pd.DataFrame, portfolio_state: Dict, account_state: Dict) -> Dict
    def register_position(self, position: Dict)
    def close_position(self, position_id: str, exit_price: float, exit_reason: str) -> Dict
    def update_position_stop(self, position_id: str, new_stop: float) -> bool
    def get_risk_metrics(self) -> Dict
```

**Validações de Risco:**
1. Limite de posições abertas
2. Perda diária máxima
3. Risco por trade
4. Risk/reward mínimo
5. Correlação entre posições
6. Horário de trading
7. Volatilidade do mercado

### 10. ModelManager (`model_manager.py`)
**Gerenciador de modelos ML**

```python
class ModelManager:
    def __init__(self, models_dir: str)
    def load_models(self) -> bool
    def predict(self, features: pd.DataFrame) -> Optional[Any]
    def predict_with_ensemble(self, features, market_regime: str = 'undefined') -> Optional[Dict]
    def get_all_required_features(self) -> Set[str]
    def update_model_performance(self, model_name: str, metrics: Dict)
    def setup_auto_retraining(self, config: Dict)
```

**Modelos Suportados:**
- XGBoost (fast)
- LightGBM (balanced)
- Random Forest (stable)
- LSTM (intraday patterns)
- Transformer (attention-based)

### 11. OrderManager (`order_manager_v4.py`)
**Gerencia execução de ordens**

```python
class OrderExecutionManagerV4:
    def __init__(self, connection_manager: ConnectionManagerV4)
    def initialize(self) -> bool
    def create_order(self, signal: Dict, account_info: Dict) -> Optional[SendOrder]
    def send_order(self, order: SendOrder) -> Dict[str, Any]
    def cancel_order(self, order_id: int) -> bool
    def get_order_status(self, order_id: int) -> Optional[Dict]
    def get_open_orders(self) -> List[Dict]
```

### 12. DataPipeline (`data_pipeline.py`)
**Pipeline de processamento de dados**

```python
class DataPipeline:
    def __init__(self, data_structure: TradingDataStructure)
    def process_historical_trades(self, trades: List[Dict]) -> Dict[str, pd.DataFrame]
    def get_aligned_data(self, limit: Optional[int] = None) -> pd.DataFrame
    def update_indicators(self, indicators: pd.DataFrame)
    def clear_old_data(self, keep_days: int = 7)
```

### 13. RealTimeProcessor (`real_time_processor.py`)
**Processa dados em tempo real**

```python
class RealTimeProcessor:
    def __init__(self, data_structure: TradingDataStructure, candle_interval: int = 60)
    def process_trade(self, trade: Dict) -> bool
    def force_close_candle(self)
    def get_current_state(self) -> Dict
    def get_latest_data(self, n_candles: int = 100) -> Dict[str, pd.DataFrame]
```

## 🔄 Fluxo de Dados Completo

### 1. Inicialização
```python
# 1. Criar sistema
system = TradingSystem(config)

# 2. Inicializar componentes
system.initialize()

# 3. Iniciar operação
system.start('WDOU25')  # Contrato específico
```

### 2. Recepção de Dados
```python
# Callback de trade (automático via ProfitDLL)
def on_trade_callback(trade_data):
    # Processa trade
    real_time_processor.process_trade(trade_data)
    
    # Atualiza data structure
    data_structure.update_candles(new_candles)
```

### 3. Ciclo de ML (a cada 60s)
```python
# No ml_worker thread
def _ml_worker():
    while is_running:
        # 1. Calcular features
        features = feature_engine.calculate(data_structure)
        
        # 2. Processar predição
        prediction = ml_coordinator.process_prediction_request(data_structure)
        
        # 3. Enviar para fila de sinais
        if prediction and prediction['can_trade']:
            signal_queue.put(prediction)
```

### 4. Geração de Sinais
```python
# No signal_worker thread
def _signal_worker():
    while is_running:
        prediction = signal_queue.get()
        
        # 1. Gerar sinal
        signal = signal_generator.generate_signal(prediction, data_structure)
        
        # 2. Validar risco
        is_valid, reason = risk_manager.validate_signal(signal, account_balance)
        
        # 3. Executar se válido
        if is_valid:
            order_manager.send_order(signal)
```

## 📊 Estruturas de Dados Importantes

### Trade (ProfitDLL)
```python
{
    'timestamp': datetime,
    'price': float,
    'volume': int,
    'trade_type': int,  # 2=buy, 3=sell
    'ticker': str
}
```

### Candle (OHLCV)
```python
{
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': float,
    'timestamp': pd.Timestamp  # index
}
```

### Signal
```python
{
    'timestamp': datetime,
    'action': str,          # 'buy' ou 'sell'
    'entry_price': float,
    'stop_loss': float,
    'take_profit': float,
    'confidence': float,
    'position_size': int,
    'reason': str,
    'metadata': {
        'prediction': dict,
        'atr': float,
        'regime': str
    }
}
```

### Position
```python
{
    'id': str,
    'action': str,
    'entry_price': float,
    'stop_loss': float,
    'take_profit': float,
    'position_size': int,
    'open_time': datetime,
    'status': str  # 'open', 'closed'
}
```

## 🛠️ Desenvolvimento e Extensão

### Adicionar Nova Feature

1. **Editar `ml_features.py`**:
```python
def calculate_new_feature(self, data: pd.DataFrame) -> pd.Series:
    """Calcula nova feature"""
    # Implementar cálculo
    return feature_series
```

2. **Registrar em `FeatureEngine`**:
```python
# Em calculate_ml_features()
features['new_feature'] = ml_features.calculate_new_feature(data)
```

3. **Atualizar `all_required_features.json`**:
```json
{
    "features": [
        "existing_features...",
        "new_feature"
    ]
}
```

### Adicionar Novo Modelo

1. **Treinar modelo** usando `TrainingOrchestrator`
2. **Salvar com metadata**:
```python
# Salvar modelo
joblib.dump(model, 'models/new_model.pkl')

# Salvar features
with open('models/new_model_features.json', 'w') as f:
    json.dump({'features': feature_list}, f)
```

3. **ModelManager carregará automaticamente**

### Implementar Nova Estratégia

1. **Criar classe de estratégia**:
```python
class NewStrategy:
    def evaluate(self, prediction: Dict, market_data) -> Dict:
        # Lógica da estratégia
        return signal
```

2. **Integrar em `StrategyEngine`**:
```python
self.strategies['new_strategy'] = NewStrategy(config)
```

### Adicionar Novo Indicador

1. **Editar `technical_indicators.py`**:
```python
@staticmethod
def calculate_new_indicator(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcula novo indicador"""
    # Implementação
    return indicator_series
```

2. **Registrar em `_calculate_all_indicators()`**:
```python
indicators['new_indicator'] = self.calculate_new_indicator(candles)
```

## 🔧 Configuração e Ambiente

### Variáveis de Ambiente
```bash
# .env file
TRADING_ENV=development  # ou 'production'
LOG_LEVEL=INFO
USE_GPU=false
```

### Estrutura de Diretórios
```
QuantumTrader_ML/
├── src/                    # Código fonte
│   ├── training/          # Sistema de treinamento
│   ├── features/          # Documentação de features
│   └── tests/             # Testes unitários
├── models/                # Modelos treinados (.pkl)
├── data/                  # Dados históricos
├── logs/                  # Logs do sistema
└── config/                # Configurações
```

### Logging
```python
# Configurar logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        logging.StreamHandler()
    ]
)
```

## 🚨 Pontos Críticos de Atenção

### 1. Validação de Dados de Produção
```python
# SEMPRE validar dados antes de usar
self._validate_production_data(data, source, data_type)
```

### 2. Threading e Sincronização
```python
# Usar Event para sincronização
self.historical_data_ready = threading.Event()

# Aguardar dados
if not self.historical_data_ready.wait(timeout=180):
    raise TimeoutError("Dados não carregados")
```

### 3. Contratos WDO
```python
# Sistema determina contrato automaticamente
contract = self._get_current_contract(datetime.now())
# Ex: Em agosto/2025 → WDOU25
```

### 4. Gestão de Memória
```python
# Limpar dados antigos periodicamente
data_pipeline.clear_old_data(keep_days=7)

# Limitar tamanho de DataFrames
MAX_CANDLES = 10000
```

### 5. Tratamento de Erros
```python
try:
    # Operação crítica
    result = critical_operation()
except SpecificError as e:
    logger.error(f"Erro específico: {e}")
    # Ação de recuperação
except Exception as e:
    logger.error(f"Erro não esperado: {e}", exc_info=True)
    # Falhar de forma segura
```

## 📝 Checklist para Novos Desenvolvedores

- [ ] Ler `CLAUDE.md` para entender filosofia do sistema
- [ ] Estudar `MAPA_FLUXO_DADOS_SISTEMA_01082025.md`
- [ ] Configurar ambiente de desenvolvimento
- [ ] Executar testes unitários: `pytest`
- [ ] Rodar sistema em modo simulação primeiro
- [ ] Entender regime detection antes de ML
- [ ] Sempre validar dados de produção
- [ ] Usar logging extensivamente
- [ ] Seguir padrões de código existentes
- [ ] Documentar mudanças significativas

## 🔗 Referências Importantes

- **Documentação ProfitDLL**: `profitdll-order-guide.md`
- **ML Strategy**: `ml-prediction-strategy-doc.md`
- **Sistema de Treinamento**: `SISTEMA_TREINAMENTO_INTEGRADO.md`
- **Developer Guide**: `DEVELOPER_GUIDE.md`

---

*Última atualização: 01/08/2025*
*Versão do Sistema: 2.0*