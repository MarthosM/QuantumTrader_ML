# 🔧 ML Trading v2.0 - Guia Técnico do Desenvolvedor

> **Documento Técnico Detalhado**  
> Para desenvolvedores implementando upgrades no sistema  

## 🎯 Pontos de Entrada do Sistema

### 1. Arquivo Principal: `src/main.py`
```python
def main():
    # 1. Carrega configurações do .env
    config = load_config()
    
    # 2. Cria e inicializa sistema de trading
    trading_system = TradingSystem(config)
    
    # 3. Executa sistema
    trading_system.run()
```

### 2. Inicialização do Sistema: `TradingSystem.__init__()`
```python
# Ordem de inicialização (IMPORTANTE!)
1. self.connection = ConnectionManager()          # Interface Profit
2. self.model_manager = ModelManager()           # Modelos ML
3. self.data_structure = TradingDataStructure()  # Dados centralizados
4. self.data_pipeline = DataPipeline()           # Pipeline de dados
5. self.feature_engine = FeatureEngine()         # Motor de features
6. self.ml_coordinator = MLCoordinator()         # Coordenador ML
7. self.signal_generator = SignalGenerator()     # Geração de sinais
8. self.risk_manager = RiskManager()             # Gestão de risco
```

---

## 🔄 Fluxo de Dados Críticos

### A. Carregamento de Modelos
```python
# model_manager.py - Sequência crítica
def load_models():
    1. Lista arquivos .pkl no diretório
    2. Para cada modelo:
       - Carrega com joblib.load()
       - Extrai features com _extract_features()
       - Carrega metadados se existirem
    3. Consolida todas as features em get_all_required_features()
```

**⚠️ IMPORTANTE**: O sistema usa `all_required_features.json` como fonte de verdade para features necessárias.

### B. Cálculo de Features
```python
# feature_engine.py - Pipeline principal
def calculate():
    1. technical_indicators.calculate_all()  # 45 indicadores
    2. ml_features.calculate_all()          # 80+ features ML
    3. _prepare_model_data()                # Seleciona 32 features finais
    4. Valida e preenche features ausentes
```

### C. Predição ML
```python
# ml_coordinator.py - Fluxo de predição
def process_prediction_request():
    1. Detecta regime de mercado
    2. Seleciona estratégia (trend/range)
    3. Valida condições de entrada
    4. Executa predição específica
    5. Retorna decisão de trading
```

---

## 🏗️ Arquitetura de Dados

### Estrutura Central: `TradingDataStructure`
```python
class TradingDataStructure:
    def __init__(self):
        # DataFrames principais (Thread-Safe)
        self.candles = pd.DataFrame()        # OHLCV
        self.microstructure = pd.DataFrame() # Buy/Sell pressure
        self.orderbook = pd.DataFrame()      # Book de ofertas
        self.indicators = pd.DataFrame()     # 45 indicadores técnicos
        self.features = pd.DataFrame()       # 80+ features ML
        
        # Locks para thread safety
        self.candles_lock = threading.Lock()
        self.indicators_lock = threading.Lock()
        # ... outros locks
```

### Indexação Temporal
- **CRÍTICO**: Todos os DataFrames usam `datetime` como índice
- **Alinhamento**: `pd.concat(..., axis=1)` para sincronização
- **Forward Fill**: `.ffill()` para preencher gaps temporais

---

## 🧠 Sistema de ML Detalhado

### A. Features Requeridas (32 features finais)

#### **Lista Completa do `all_required_features.json`**:
```json
[
  "ema_diff", "volume_ratio_10", "high_low_range_20", "momentum",
  "range_percent", "momentum_1", "volume_ratio_50", "momentum_10",
  "momentum_5", "volume_ratio_5", "high_low_range_50", "return_20",
  "bb_width_10", "high_low_range_5", "adx_substitute", "return_50",
  "return_5", "rsi", "momentum_3", "high_low_range_10", "adx",
  "volume_ratio_20", "ema_diff_fast", "bb_width_20", 
  "ichimoku_conversion_line", "momentum_15", "return_10", "bb_width",
  "volume_ratio", "momentum_20", "volatility_ratio"
]
```

### B. Mapeamento Features → Módulos

| Feature | Módulo Responsável | Método |
|---------|-------------------|--------|
| `ema_diff` | `technical_indicators.py` | `_calculate_composite_features()` |
| `return_*` | `ml_features.py` | `_calculate_momentum_features()` |
| `momentum_*` | `ml_features.py` | `_calculate_momentum_features()` |
| `volume_ratio_*` | `ml_features.py` | `_calculate_volume_features()` |
| `high_low_range_*` | `ml_features.py` | `_calculate_volume_features()` |
| `bb_width_*` | `technical_indicators.py` | `_calculate_bollinger_bands()` |
| `rsi` | `technical_indicators.py` | `_calculate_rsi()` |
| `adx` | `technical_indicators.py` | `_calculate_adx()` |

### C. Validação de Features
```python
# feature_engine.py - _prepare_model_data()
def _prepare_model_data():
    missing_features = []
    for feature in self.model_features:
        if feature not in available_data:
            missing_features.append(feature)
            # Criar com valor 0
            model_data[feature] = 0
    
    if missing_features:
        self.logger.warning(f"Features não encontradas: {missing_features[:5]}...")
```

---

## ⚡ Processamento em Tempo Real

### Threading Architecture
```python
# trading_system.py - Threads principais
class TradingSystem:
    def __init__(self):
        # Queues para comunicação entre threads
        self.ml_queue = queue.Queue(maxsize=10)
        self.signal_queue = queue.Queue(maxsize=10)
        
        # Threads de processamento
        self.ml_thread = threading.Thread(target=self._ml_worker)
        self.signal_thread = threading.Thread(target=self._signal_worker)
```

### Callbacks de Dados
```python
# connection_manager.py - Callbacks da DLL
@WINFUNCTYPE(None, c_wchar_p, c_double, c_int64, c_int, c_int)
def trade_callback(symbol, price, volume, trade_type, trade_id):
    """Callback chamado a cada trade recebido"""
    trade_data = {
        'symbol': symbol,
        'price': float(price),
        'volume': int(volume),
        'trade_type': int(trade_type),  # 2=buy, 3=sell
        'trade_id': int(trade_id),
        'timestamp': datetime.now()
    }
    
    # Processar em thread separada
    real_time_processor.process_trade(trade_data)
```

### Triggers de Recálculo
```python
# Condições que disparam recálculo de features:
1. Novo candle formado (mudança de minuto)
2. Volume significativo acumulado (> threshold)
3. Intervalo de tempo configurável (ML_INTERVAL=60s)
4. Request manual via força (force_calculation=True)
```

---

## 🎛️ Configurações Críticas

### A. Thresholds por Estratégia
```python
# ml_coordinator.py - CONFIGURAÇÕES CRÍTICAS
TREND_THRESHOLDS = {
    'confidence': 0.60,      # Confiança do regime
    'probability': 0.60,     # Probabilidade do modelo
    'direction': 0.70,       # Força da direção
    'magnitude': 0.003       # Magnitude mínima
}

RANGE_THRESHOLDS = {
    'confidence': 0.60,
    'probability': 0.55,
    'direction': 0.50,
    'magnitude': 0.0015
}
```

### B. Parâmetros de Risco
```python
# risk_manager.py - LIMITES DE SEGURANÇA
MAX_DAILY_LOSS = 0.05       # 5% perda máxima diária
MAX_POSITIONS = 1           # 1 posição simultânea
MAX_TRADES_PER_DAY = 10     # 10 trades máximo/dia
RISK_PER_TRADE = 0.02       # 2% risco por trade

# Horários de operação
TRADING_START = "09:00"
TRADING_END = "17:55"
```

### C. Intervalos de Processamento
```python
# trading_system.py - TIMINGS
ML_INTERVAL = 60            # Predição ML a cada 60s
FEATURE_INTERVAL = 5        # Recálculo features a cada 5s
METRICS_INTERVAL = 60       # Log métricas a cada 60s
CONTRACT_CHECK = 3600       # Verificar contrato a cada 1h
```

---

## 🔧 Pontos de Extensão

### 1. **Adicionando Novas Features**
```python
# ml_features.py - Template para nova feature
def _calculate_new_feature_category(self, candles, features):
    try:
        # Sua lógica aqui
        features['new_feature'] = calculation_result
        
        # Logging para debug
        self.logger.info(f"Nova feature calculada: new_feature")
        
    except Exception as e:
        self.logger.error(f"Erro calculando nova feature: {e}")
```

### 2. **Adicionando Novos Modelos**
```python
# model_manager.py - Suporte para novos tipos
def _extract_features(self, model, model_name):
    # Adicionar suporte para novo tipo de modelo
    elif hasattr(model, 'new_model_attribute'):
        features = model.get_feature_names()
    
    # Resto do código...
```

### 3. **Novos Indicadores Técnicos**
```python
# technical_indicators.py - Adicionar método
def _calculate_new_indicator(self, candles, indicators):
    try:
        # Cálculo do indicador
        indicators['new_indicator'] = result
        
    except Exception as e:
        self.logger.error(f"Erro calculando novo indicador: {e}")

# Adicionar ao calculate_all()
def calculate_all(self, candles):
    # ... indicadores existentes ...
    self._calculate_new_indicator(candles, indicators)
```

### 4. **Novas Estratégias de Trading**
```python
# ml_coordinator.py - Nova estratégia
def _apply_new_strategy(self, prediction_result, features_df):
    """Nova estratégia personalizada"""
    
    # Validações específicas
    if not self._validate_new_strategy_conditions(features_df):
        return self._create_hold_decision()
    
    # Lógica da estratégia
    decision = self._calculate_new_strategy_decision(prediction_result)
    
    return decision
```

---

## 🧪 Testes e Debugging

### A. Executar Testes
```bash
# Testes unitários principais
pytest tests/test_*.py -v

# Testes específicos de componente
pytest src/test_etapa*.py -v

# Teste com cobertura
pytest --cov=src tests/
```

### B. Debugging de Features
```python
# feature_engine.py - Debug helper
def debug_features(self, data):
    """Função para debug de features"""
    print(f"Candles shape: {data.candles.shape}")
    print(f"Indicators shape: {data.indicators.shape}")
    print(f"Features shape: {data.features.shape}")
    
    # Verificar features faltantes
    required = self.model_features
    available = data.features.columns.tolist()
    missing = [f for f in required if f not in available]
    print(f"Missing features: {missing}")
```

### C. Logs de Diagnóstico
```python
# Configurar nível de log para debug
logging.basicConfig(level=logging.DEBUG)

# Logs importantes para monitorar:
- "Features não encontradas": Indica features faltantes
- "Modelo X: Y features": Confirma carregamento correto
- "DataFrame do modelo preparado": Confirma preparação final
- "Erro calculando": Indica problemas de cálculo
```

---

## 🚨 Problemas Comuns e Soluções

### 1. **Features Faltantes**
```python
# PROBLEMA: "Features não encontradas: ['return_5', 'ema_5']"
# SOLUÇÃO: Verificar se feature está sendo calculada

# 1. Verificar se está em ml_features.py ou technical_indicators.py
# 2. Verificar se método está sendo chamado
# 3. Verificar se há erro na lógica de cálculo
```

### 2. **Modelos Não Carregando**
```python
# PROBLEMA: "Nenhuma feature extraída para modelo_X"
# SOLUÇÃO: Verificar extract_features()

# 1. Verificar tipo do modelo (XGBoost/LightGBM/Sklearn)
# 2. Verificar se arquivo _features.json existe
# 3. Adicionar suporte para novo tipo se necessário
```

### 3. **Dados Desalinhados**
```python
# PROBLEMA: IndexError ou shapes inconsistentes
# SOLUÇÃO: Verificar alinhamento temporal

# 1. Verificar se todos DataFrames usam mesmo índice datetime
# 2. Usar pd.concat(..., axis=1, join='outer')
# 3. Aplicar .ffill() para preencher gaps
```

### 4. **Performance Lenta**
```python
# PROBLEMA: Sistema lento para calcular features
# SOLUÇÃO: Otimizar cálculos

# 1. Verificar se cache está funcionando
# 2. Ativar processamento paralelo
# 3. Reduzir lookback period se possível
# 4. Usar vectorização pandas em vez de loops
```

---

## 📊 Monitoramento de Produção

### Métricas de Saúde do Sistema
```python
# metrics_collector.py - Métricas críticas
{
    'system_health': {
        'connection_status': 'connected',
        'models_loaded': 5,
        'features_calculated': 32,
        'last_prediction_time': datetime,
        'memory_usage_mb': 150.5
    },
    'trading_metrics': {
        'trades_today': 3,
        'win_rate': 0.65,
        'pnl_today': 250.0,
        'max_drawdown': 0.02
    }
}
```

### Alertas Críticos
- **Connection Lost**: Conexão com Profit perdida
- **Model Error**: Erro na predição ML
- **Risk Limit**: Limite de risco atingido
- **Data Quality**: Dados inconsistentes detectados

---

## 🔮 Roadmap de Melhorias

### Prioridade Alta
1. **Online Learning**: Retreinamento contínuo de modelos
2. **Multi-Asset**: Suporte para múltiplos ativos
3. **Advanced Risk**: Gestão de risco baseada em portfolio

### Prioridade Média
1. **Deep Learning**: Modelos LSTM/Transformer
2. **Alternative Data**: Integração com dados alternativos
3. **Cloud Deployment**: Deploy em AWS/Azure

### Prioridade Baixa
1. **Web Dashboard**: Interface web para monitoramento
2. **Mobile Alerts**: Notificações via app móvel
3. **Backtesting Engine**: Framework de backtesting avançado

---

**Happy Coding! 🚀**

> Este guia deve ser atualizado sempre que modificações significativas forem feitas no sistema.
