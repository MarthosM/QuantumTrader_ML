# Instruções Personalizadas para Gi### Documentação principal
- **Fluxo de Dados ML**: `src/features/complete_ml_data_flow_map.md` - Referência obrigatória para arquitetura
- **DEVELOPER_GUIDE.md** - **ESSENCIAL**: Atualizações recentes, política de dados, guio técnica detailado  
- **Estratégia ML Trading**: `src/features/ml-prediction-strategy-doc.md` - Document ESSENCIAL com estratégias**⚠️ LEMBRETE CRÍTICO:** Este sistema opera com dinheiro real - NUNCA comprometa a segurança com dados mock em situações importantes!

**📚 SEMPRE CONSULTAR:**
- `DEVELOPER_GUIDE.md` - Politique CRÍTICA de dados e atualizações técnicas
- `src/features/complete_ml_data_flow_map.md` - Mapeamento completo do sistema
- `ml_backtester.py` - Sistema de backtest funcional e integradode predição por região
- **README**: `README.md` - Estrutura geral do projeto
- **ml_backtester.py** - Sistema de backtest ML funcional e integrado
- **Testes Etapa 1**: `tests/test_etapa1.py` - Padrões de teste estabelecidos  
- **Testes Etapa 2**: `src/test_etapa2.py` - Testes de pipeline e processamentoopilot  
# ML Trading v2.0 - Sistema de Trading Algorítmico

## 📋 Contexto do Projeto

Este é um sistem REAL de trading algorítmico que utiliza Machine Learning para análise de marcão financeiro e **pode envolver riscos financeiros significativos**. O projeto segue uma arquitetura modular bem definida.

## 🛡️ **POLÍTICA DE DADOS CRÍTICA**

### ⚠️ **DADOS REAIS OBRIGATÓRIOS**
- **Sistema real** opera com dinheiro real e pode gerar prejuízos
- **Todos os testes finais** DEVEM usar dados reais da ProfiDLL
- **Todos os backtests** OBRIGATORIAMENTE com dados históricos reais  
- **Produção** bloqueia automaticamente dados sintéticos

### ⚠️ **DADOS MOCK - USO EXTREMAMENTE RESTRITO**
- **APENAS durante desenvolvimento** de componentes isolados
- **NUNCA em testes de integração final ou backtests**
- **APAGAR IMEDIATAMENTE** após uso em testes intermediários
- **Sistema bloqueia mock em produção** automaticamente

```python
# ❌ NUNCA FAZER EM TESTES FINAIS:
def test_backtest_system():
    mock_data = generate_fake_data()  # ❌ PROIBIDO!

# ✅ SEMPRE FAZER:
def test_backtest_system():
    real_data = load_real_market_data()  # ✅ CORRETO
    if real_data.empty:
        pytest.skip("Dados reais indisponíveis")
```

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

### 4. **Política de Dados Crítica (FUNDAMENTAL)**
#### ⚠️ **DADOS REAIS OBRIGATÓRIOS**
- **SEMPRE priorizar dados reais da ProfitDLL quando disponível**
- **MOCK PROIBIDO em testes finais**, backtests e sistemas importantes
- **TESTES FINAIS SEMPRE com dados reais** ou skip automaticamente
- **Sistema bloqueia automaticamente mock em produção**

#### 🧪 **PADRÕES DE TESTE CRÍTICOS**
```python
# ✅ PADRÃO CORRETO para testes:
def test_component_functionality():
    """Teste intermediário - mock permitido apenas aqui"""
    temp_mock = create_simple_test_data()
    result = component.process(temp_mock)
    assert result.is_valid()
    del temp_mock  # ⚠️ OBRIGATÓRIO: Apagar imediatamente

def test_integration_final():
    """Teste final DEVE usar dados reais"""
    real_data = load_real_historical_data()
    if real_data.empty:
        pytest.skip("Dados reais indisponíveis - teste adiado")
    
    # ✅ Execução de teste com dados reais
    result_real = system.process(real_data)
    assert result_real.confidence > 0.80
```

#### 🚨 **O QUE NUNCA FAZER**
- ❌ Mock em backtests
- ❌ Mock em testes de integração final  
- ❌ Manter dados sintéticos após teste
- ❌ Usar dados fake em ambiente produtivo

### 5. **Framework de Testes**
- Use **pytest** para todos os testes (framework oficial do projeto)
- Estrutura dual: `tests/` para testes principais, `src/test_*.py` para testes específicos
- **SEMPRE validar origem dos dados** antes de processar
- Use `@pytest.skip()` quando dados reais não estão disponíveis

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

## 🔄 **ATUALIZAÇÕES RECENTES (2025-07-21)**

### ✅ **Sistema de Backtest ML Funcional**
- **ml_backtester.py**: Sistema completo de backtest integrado com ML
- **30 Features Principais**: EMA, ATR, ADX, Bollinger, volatilidades
- **Modelos Reais**: LightGBM + Random Forest + XGBoost treinados com 83% de confiança
- **Manual Feature Calculation**: Fallback robusto para features indisponíveis  
- **Conservative Trading**: Sistema inteligente de rejeição de sinais baixa confiança

### ✅ **Política de Dados Implementada**
- **`_load_test_data_isolated()`**: Verificação dupla de ambiente
- **Bloqueio automático**: Dados sintéticos proibidos em produção
- **Validação de produção**: `_validate_production_data()` obrigatória
- **Prioridade absoluta**: Sistema sempre prefere dados reais da ProfitDLL

### ✅ **Sistema Integrado e Testado**  
- **Treinamento robusto**: Models treinados com 30 features e validação temporal
- **FeatureEngine integrada**: Cálculo manual quando componente não está disponível
- **Backtests validados**: Sistema executa backtests reais sem mock
- **Performance conservativa**: Zero trades por design com alta confiança em HOLD

---

**⚠️ LEMBRETE CRÍTICO:** Este sistema opera com dinheiro real. NUNCA comprometa a segurança com dados mock em situações importantes!

---

## 📝 **PROCESSO DE DOCUMENTAÇÃO E MANUTENÇÃO (OBRIGATÓRIO)**

### 🔄 **Ao Final de Cada Iteração Importante**

**SEMPRE que terminar uma iteração significativa:**

1. **📋 Relatório em Markdown**:
   - Após **confirmação do usuário**, gerar arquivo markdown detalhado
   - **Nome padrão**: `ITERACAO_YYYY-MM-DD_HH-MM.md`
   - **Conteúdo obrigatório**:
     - ✅ **O que foi implementado/corrigido**
     - ⚠️ **Problemas identificados e soluções**
     - 🔧 **Configurações alteradas** (`.env`, arquivos de config)
     - 📊 **Resultados de testes/performance**
     - 🚨 **Impactos no sistema** (se houver)

2. **📚 Atualização da Documentação Principal**:
   - **SEMPRE sugerir atualizações para**:
     - `DEVELOPER_GUIDE.md` - Para mudanças técnicas importantes
     - `src/features/complete_ml_data_flow_map.md` - Para alterações no fluxo de dados
     - `src/features/ml-prediction-strategy-doc.md` - Para mudanças em estratégias
   
### 🚨 **Critérios para Atualização Obrigatória dos Guias**

#### ⚡ **DEVELOPER_GUIDE.md** - Atualizar quando:
- Novas configurações no `.env`
- Mudanças na arquitetura de componentes
- Novos padrões de teste implementados
- Alterações na política de dados
- Novos requisitos de sistema
- Mudanças em dependências críticas

#### 🗺️ **complete_ml_data_flow_map.md** - Atualizar quando:
- Mudanças no fluxo de processamento de dados
- Novos componentes de dados adicionados
- Alterações nas estruturas de DataFrames
- Modificações na pipeline ML
- Mudanças na integração entre componentes
- Novos pontos de validação de dados

#### 📈 **ml-prediction-strategy-doc.md** - Atualizar quando:
- Novos regimes de mercado implementados
- Alterações em thresholds de confiança
- Mudanças nas estratégias de trading
- Novos indicadores técnicos adicionados
- Modificações no sistema de risco
- Alterações nos critérios de sinais

### 📋 **Template do Relatório de Iteração**

```markdown
# Relatório de Iteração - YYYY-MM-DD

## 🎯 Objetivo da Iteração
[Descrever o que foi planejado fazer]

## ✅ Implementações Realizadas
- [ ] Item 1
- [ ] Item 2
- [ ] Item 3

## 🔧 Configurações Alteradas
### Arquivo `.env`
- `VARIAVEL_X`: valor_antigo → valor_novo (motivo)
- Nova variável: `NOVA_VAR=valor` (propósito)

### Outros Arquivos de Configuração
[Se houver alterações em outros arquivos]

## 🧪 Testes e Validações
### Resultados dos Testes
- Performance: X segundos
- Cobertura: X%
- Testes passaram: X/Y

### Validações de Sistema
- Conexão ProfitDLL: ✅/❌
- Carregamento de modelos: ✅/❌
- Processamento de dados: ✅/❌

## ⚠️ Problemas Identificados
1. **Problema X**: Descrição
   - **Solução aplicada**: Como foi resolvido
   - **Status**: Resolvido/Pendente

## 📊 Performance e Métricas
- Dados processados: X registros
- Candles formados: X
- Tempo de processamento: X segundos
- Uso de memória: X MB

## 🚨 Impactos no Sistema
### Mudanças de Arquitetura
[Se houver mudanças significativas na arquitetura]

### Compatibilidade
- Versões anteriores: Compatível/Incompatível
- Dependências: Alteradas/Inalteradas

## 📝 Sugestões de Atualização da Documentação
- [ ] DEVELOPER_GUIDE.md - Motivo: [explicar]
- [ ] complete_ml_data_flow_map.md - Motivo: [explicar]
- [ ] ml-prediction-strategy-doc.md - Motivo: [explicar]

## 🔜 Próximos Passos
1. Item pendente 1
2. Item pendente 2
3. Item pendente 3

---
**Gerado em**: YYYY-MM-DD HH:MM:SS
**Por**: GitHub Copilot
**Versão do Sistema**: ML Trading v2.0
```

### 🎯 **Processo de Aprovação**

1. **Confirmar com usuário**: "Deseja gerar relatório de iteração?"
2. **Criar arquivo markdown** com nome único
3. **Listar sugestões específicas** para cada guia de documentação
4. **Aguardar aprovação** antes de atualizar documentação principal
5. **Implementar atualizações aprovadas** nos guias relevantes

---

**📌 LEMBRE-SE**: Este processo garante que o sistema sempre tenha documentação atualizada e que mudanças importantes não sejam perdidas!
