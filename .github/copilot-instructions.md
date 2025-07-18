# Instruções Personalizadas para GitHub Copilot
# ML Trading v2.0 - Sistema de Trading Algorítmico

## 📋 Contexto do Projeto

Este é um sistema avançado de trading algorítmico que utiliza Machine Learning para análise de mercado financeiro. O projeto segue uma arquitetura modular bem definida.

## 🗺️ Referências de Arquitetura

### Documentação Principal
- **Fluxo de Dados ML**: `src/features/complete_ml_data_flow_map.md` - Referência obrigatória para arquitetura
- **Estratégia ML Trading**: `src/features/ml-prediction-strategy-doc.md` - Documento ESSENCIAL com estratégias de predição por regime
- **README**: `README.md` - Estrutura geral do projeto
- **Testes Etapa 1**: `tests/test_etapa1.py` - Padrões de teste estabelecidos
- **Testes Etapa 2**: `src/test_etapa2.py` - Testes de pipeline e processamento

### Estrutura Atual do Projeto
```
ML_Tradingv2.0/
├── src/                         # Código fonte principal
│   ├── connection_manager.py    # Gerencia conexão com ProfitDLL
│   ├── model_manager.py        # Carregamento e gestão de modelos ML
│   ├── data_structure.py       # Estrutura centralizada de dados
│   ├── data_pipeline.py        # Pipeline de processamento histórico
│   ├── real_time_processor.py  # Processamento em tempo real
│   ├── data_loader.py          # Carregamento de dados
│   ├── feature_engine.py       # Motor principal de features
│   ├── technical_indicators.py # Indicadores técnicos
│   ├── ml_features.py          # Features de ML
│   ├── test_etapa2.py          # Testes específicos da etapa 2
│   ├── data/                   # Processamento de dados
│   ├── features/               # Engenharia de features
│   ├── models/                 # Modelos de ML
│   └── utils/                  # Utilitários
├── projeto/                     # Estrutura de projeto organizada
│   └── tests/                  # Testes organizados
│       └── __init__.py         # Inicialização dos testes
├── tests/                      # Testes principais
├── .venv/                      # Ambiente virtual Python
├── .pytest_cache/              # Cache do pytest
└── requirements.txt            # Dependências do projeto
```

## 🎯 Diretrizes de Desenvolvimento

### 1. Arquitetura e Fluxo de Dados
- **SEMPRE** consulte o mapeamento completo em `src/features/complete_ml_data_flow_map.md`
- **ESSENCIAL** consulte as estratégias de trading em `src/features/ml-prediction-strategy-doc.md`
- Siga o fluxo: Modelos → Dados → Indicadores → Features → Detecção Regime → Predição → Sinal
- Mantenha dataframes separados: candles, microstructure, orderbook, indicators, features
- Use threading para cálculos pesados (queues: indicator_queue, prediction_queue)

### 2. Sistema de Regimes de Mercado (OBRIGATÓRIO)
- **Tendência (Trend)**: EMA alinhadas, ADX > 25, movimento direcional claro
  - `trend_up`: EMA9 > EMA20 > EMA50
  - `trend_down`: EMA9 < EMA20 < EMA50
- **Lateralização (Range)**: ADX < 25, preço entre suporte/resistência
- **Undefined**: Condições mistas, regime indefinido
- **Confiança mínima**: 60% para qualquer operação

### 3. Estratégias por Regime (IMPLEMENTAÇÃO OBRIGATÓRIA)

#### Estratégia de Tendência
- **Risk/Reward**: 1:2 (Stop: 5 pontos, Target: 10 pontos)
- **Thresholds**: 
  - Confiança regime: >60%
  - Probabilidade modelo: >60%
  - Direção: >0.7
  - Magnitude: >0.003
- **Lógica**: Operar a favor da tendência estabelecida

#### Estratégia de Range
- **Risk/Reward**: 1:1.5 (Stop: ATR-based, mín. 3 pontos)
- **Thresholds**:
  - Confiança regime: >60%
  - Probabilidade modelo: >55%
  - Direção: >0.5
  - Magnitude: >0.0015
- **Lógica**: Reversões em suporte/resistência
- **Posições**: near_support → BUY, near_resistance → SELL

### 4. Validações de Trading (RIGOROSAMENTE SEGUIR)
- **HOLD obrigatório quando**:
  - Regime undefined ou confiança <60%
  - Thresholds não atingidos
  - Range em posição neutra
  - Trend contra-tendência
- **Horário operação**: 09:00-17:55 (WDO)
- **Limites sessão**: Max 1 posição, 10 trades/dia, 5% perda máxima

### 2. Padrões de Código
- **ModelManager**: Gerencia modelos ML e extração de features
- **ConnectionManager**: Gerencia DLL do Profit com callbacks
- **TradingDataStructure**: Centraliza todos os dados do sistema
- **DataPipeline**: Processamento histórico de dados
- **RealTimeProcessor**: Processamento em tempo real
- **DataLoader**: Carregamento e geração de dados de teste
- **FeatureEngine**: Motor principal de cálculo de features
- **TechnicalIndicators**: Cálculo de indicadores técnicos
- **MLFeatures**: Cálculo de features de ML
- **MLCoordinator**: Coordena detecção de regime e predição (ESSENCIAL)
- **PredictionEngine**: Predições específicas por regime
- **SignalGenerator**: Converte predições em sinais de trading
- **RiskManager**: Gestão de risco com parâmetros por regime
- **Features**: ~80-100 features incluindo OHLCV, indicadores, momentum, volatilidade

### 3. Features Essenciais
```python
# Básicas: open, high, low, close, volume
# Indicadores: ema_*, rsi, macd, bb_*, atr, adx
# Momentum: momentum_*, momentum_pct_*, return_*
# Volatilidade: volatility_*, range_*
# Microestrutura: buy_pressure, flow_imbalance
```

### 4. Padrões de Teste
- Use **pytest** para todos os testes (framework oficial do projeto)
- Estrutura dual: `tests/` para testes principais, `src/test_*.py` para testes específicos
- Use **fixtures** para setup/teardown consistente
- Crie diretórios temporários para testes isolados
- Teste com DLL real quando possível (use `pytest.skip()` se não disponível)
- Valide tanto cenários de sucesso quanto falha
- Use `@pytest.mark.parametrize` para testar múltiplos cenários

### 5. Estrutura de Testes com Pytest
```python
# Localização dos testes:
# - tests/test_etapa1.py: Testes principais da etapa 1
# - src/test_etapa2.py: Testes de pipeline e processamento
# - src/test_etapa3.py: Testes de features e indicadores
# - projeto/tests/: Estrutura organizacional futura

# Padrão de fixtures
@pytest.fixture
def data_structure():
    """Fixture para criar estrutura de dados"""
    ds = TradingDataStructure()
    ds.initialize_structure()
    return ds

@pytest.fixture
def sample_trades():
    """Fixture para gerar trades de exemplo"""
    # Implementação de dados de teste
    pass

# Uso de parametrize para múltiplos casos
@pytest.mark.parametrize("invalid_trade", [
    {'price': -100, 'volume': 10},
    {'price': 0, 'volume': 10},
    {'price': 5000, 'volume': -5}
])
def test_invalid_trades(self, rt_processor, invalid_trade):
    success = rt_processor.process_trade(invalid_trade)
    assert success is False
```

### 6. Comandos de Teste
```bash
# Executar todos os testes
pytest

# Executar testes específicos
pytest src/test_etapa2.py
pytest src/test_etapa3.py

# Executar com cobertura
pytest --cov=src

# Executar com output detalhado
pytest -v

# Executar testes específicos por nome
pytest -k "test_pipeline"
```

### 5. Logging e Debugging
- Use logging.getLogger('ModuleName') em cada classe
- Log resumos de modelos carregados
- Log qualidade de dados e validações
- Mantenha logs detalhados para debugging

## 📊 Exemplo de Implementação

### Detecção de Regime e Predição
```python
# Sempre seguir o fluxo: Regime → Estratégia → Predição
ml_coordinator = MLCoordinator(model_manager, feature_engine, prediction_engine, regime_trainer)
prediction = ml_coordinator.process_prediction_request(data)

# Resultado inclui regime detectado e estratégia aplicada
# {
#   'regime': 'trend_up',
#   'confidence': 0.85,
#   'trade_decision': 'BUY',
#   'can_trade': True,
#   'risk_reward_target': 2.0
# }
```

### Carregamento de Modelos
```python
# Sempre extrair features dos modelos carregados
model_manager = ModelManager(models_dir)
model_manager.load_models()
all_features = model_manager.get_all_required_features()
```

### Processamento de Dados
```python
# Seguir o padrão de dataframes separados
data_structure = TradingDataStructure()
data_structure.initialize_structure()
data_structure.update_candles(new_candles)
```

### Cálculo de Features
```python
# Sempre sincronizar com features dos modelos
feature_generator.sync_with_model(model_features)
result = feature_generator.create_features_separated(
    candles_df, microstructure_df, indicators_df
)
```

### Geração de Sinais com Regime
```python
# Sinal baseado em regime e thresholds específicos
signal = signal_generator.generate_regime_based_signal(prediction, market_data)
# Inclui stop/target baseados na estratégia do regime
```

## 🚨 Regras Importantes

### ❌ NÃO FAÇA
- Não ignore o mapeamento de fluxo de dados
- Não ignore as estratégias por regime em `ml-prediction-strategy-doc.md`
- Não opere sem detectar regime com confiança >60%
- Não use thresholds diferentes dos definidos por regime
- Não misture dados de diferentes timeframes sem alinhamento
- Não use caminhos hardcoded (exceto para testes)
- Não bloqueie a thread principal com cálculos pesados

### ✅ SEMPRE FAÇA
- Consulte `complete_ml_data_flow_map.md` antes de implementar
- Consulte `ml-prediction-strategy-doc.md` para validações de trading
- Implemente detecção de regime ANTES de qualquer predição
- Use thresholds específicos por regime (trend vs range)
- Valide proximidade de suporte/resistência em range
- Confirme alinhamento de EMAs em tendência
- Use tipos específicos (Dict[str, Any], List[str], etc.)
- Implemente validação de dados em cada etapa
- Use ambiente virtual (.venv) para dependências
- Mantenha compatibilidade com ProfitDLL.dll

## 🧪 Padrão de Testes

```python
# Estrutura padrão de teste
class TestModuleName(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_functionality(self):
        # Teste com dados reais quando possível
        # Use skipTest para dependências externas
```

## 📚 Bibliotecas Principais

- **ML**: scikit-learn, lightgbm, xgboost, optuna
- **Dados**: pandas, numpy, ta (technical analysis)
- **Visualização**: matplotlib, seaborn
- **DB**: SQLAlchemy, alembic
- **Testes**: pytest, unittest.mock

## 🎭 Persona de Desenvolvimento

Atue como um **desenvolvedor sênior especializado em trading algorítmico** que:
- Conhece profundamente os mercados financeiros
- Domina Machine Learning aplicado a trading
- Segue padrões de código enterprise
- Prioriza performance e confiabilidade
- Sempre valida dados antes de processar
- Mantém código modular e testável

## 📈 Métricas de Qualidade

- **Coverage**: Manter >90% cobertura de testes
- **Performance**: Cálculos em <1s para predições
- **Confiabilidade**: Validação rigorosa de dados
- **Manutenibilidade**: Código autodocumentado

## 📊 KPIs de Trading (OBRIGATÓRIO MONITORAR)

- **Win Rate**: Taxa de acerto (alvo: >55%)
- **Profit Factor**: Lucro total / Perda total (alvo: >1.5)
- **Sharpe Ratio**: Retorno ajustado ao risco (alvo: >1.0)
- **Max Drawdown**: Perda máxima da carteira (limite: 10%)
- **Taxa de sinais**: 3-5 por dia em condições normais
- **Tendência**: Win rate esperado 60-65%
- **Range**: Win rate esperado 55-60%

---

**Lembre-se**: Este é um sistema de trading real que pode envolver riscos financeiros. Sempre priorize precisão, validação e confiabilidade sobre velocidade de desenvolvimento.
