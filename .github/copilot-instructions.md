# Instruções Personalizadas para GitHub Copilot
# ML Trading v2.0 - Sistema de Trading Algorítmico

## 📋 Contexto do Projeto

Este é um sistema avançado de trading algorítmico que utiliza Machine Learning para análise de mercado financeiro. O projeto segue uma arquitetura modular bem definida.

## 🗺️ Referências de Arquitetura

### Documentação Principal
- **Fluxo de Dados ML**: `src/features/complete_ml_data_flow_map.md` - Referência obrigatória para arquitetura
- **README**: `README.md` - Estrutura geral do projeto
- **Testes**: `tests/test_etapa1.py` - Padrões de teste estabelecidos

### Estrutura de Classes Principais
```
src/
├── connection_manager.py    # Gerencia conexão com ProfitDLL
├── model_manager.py        # Carregamento e gestão de modelos ML
├── data_structure.py       # Estrutura centralizada de dados
├── data/                   # Processamento de dados
├── features/               # Engenharia de features
├── models/                 # Modelos de ML
└── utils/                  # Utilitários
```

## 🎯 Diretrizes de Desenvolvimento

### 1. Arquitetura e Fluxo de Dados
- **SEMPRE** consulte o mapeamento completo em `src/features/complete_ml_data_flow_map.md`
- Siga o fluxo: Modelos → Dados → Indicadores → Features → Predição → Sinal
- Mantenha dataframes separados: candles, microstructure, orderbook, indicators, features
- Use threading para cálculos pesados (queues: indicator_queue, prediction_queue)

### 2. Padrões de Código
- **ModelManager**: Gerencia modelos ML e extração de features
- **ConnectionManager**: Gerencia DLL do Profit com callbacks
- **TradingDataStructure**: Centraliza todos os dados do sistema
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
- Use pytest para todos os testes
- Crie diretórios temporários para testes isolados
- Teste com DLL real quando possível (use skipTest se não disponível)
- Valide tanto cenários de sucesso quanto falha

### 5. Logging e Debugging
- Use logging.getLogger('ModuleName') em cada classe
- Log resumos de modelos carregados
- Log qualidade de dados e validações
- Mantenha logs detalhados para debugging

## 📊 Exemplo de Implementação

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

## 🚨 Regras Importantes

### ❌ NÃO FAÇA
- Não ignore o mapeamento de fluxo de dados
- Não misture dados de diferentes timeframes sem alinhamento
- Não use caminhos hardcoded (exceto para testes)
- Não bloqueie a thread principal com cálculos pesados

### ✅ SEMPRE FAÇA
- Consulte `complete_ml_data_flow_map.md` antes de implementar
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

---

**Lembre-se**: Este é um sistema de trading real que pode envolver riscos financeiros. Sempre priorize precisão, validação e confiabilidade sobre velocidade de desenvolvimento.
