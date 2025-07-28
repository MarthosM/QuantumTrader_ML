# 🔧 ML Trading v2.0 - Guia do Desenvolvedor

> **Atualizado: 2025-01-28**  
> Sistema de trading algorítmico com Machine Learning

## 📋 Visão Geral

O ML Trading v2.0 é um sistema completo de trading algorítmico que utiliza modelos de Machine Learning para análise e execução automática de operações no mercado futuro brasileiro (B3).

### 🎯 Status Atual
- **Dados Históricos**: Via arquivos CSV (formato Nelogica)
- **Dados em Tempo Real**: ProfitDLL integrado
- **Modelos ML**: Ensemble por regime de mercado
- **Execução**: Conta simulação/real da corretora

## 🏗️ Arquitetura do Sistema

### Fluxo de Dados Principal
```
CSV Data → Dataset Builder → ML Training → Models
                                              ↓
Live Data (ProfitDLL) → Features → ML Prediction → Trading Signal → Order Execution
```

### Componentes Principais

#### 1. **Processamento de Dados Históricos**
```python
# create_dataset_from_csv.py
- Carrega dados CSV no formato: ticker,date,time,price,qty,trade_type...
- Cria candles OHLCV
- Calcula microestrutura (buy/sell pressure, imbalances)
- Gera dataset ML completo
```

#### 2. **Sistema de Trading em Tempo Real**
```python
# src/trading_system.py
- ConnectionManager: Interface com ProfitDLL
- TradingDataStructure: Armazenamento centralizado
- FeatureEngine: Cálculo de features em tempo real
- MLCoordinator: Coordenação de predições por regime
- SignalGenerator: Geração de sinais de trading
- OrderManager: Execução de ordens
```

#### 3. **Machine Learning Pipeline**
```python
# src/ml/
- dataset_builder_v3.py: Construção de datasets
- training_orchestrator_v3.py: Pipeline de treinamento
- prediction_engine_v3.py: Motor de predições

# Features V3 (118 features):
- Price/Momentum (35)
- Volume/Microstructure (24)
- Technical Indicators (11)
- Volatility (13)
- Advanced/Composite (35)
```

## 🚀 Quick Start

### 1. Preparar Ambiente
```bash
# Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Configurar credenciais
copy .env.example .env
# Editar .env com suas credenciais
```

### 2. Criar Dataset a partir de CSV
```bash
# Analisar dados primeiro
python create_dataset_from_csv.py dados_wdo.csv --analyze-only

# Criar dataset
python create_dataset_from_csv.py dados_wdo.csv

# Verificar compatibilidade
python analyze_dataset_compatibility.py
```

### 3. Treinar Modelos
```bash
# Treinar ensemble completo
python src/ml/training_orchestrator_v3.py

# Verificar modelos treinados
ls models/
```

### 4. Executar Sistema
```bash
# Modo simulação (recomendado para testes)
python src/main.py

# Sistema detecta automaticamente conta simulação/real
```

## 📊 Formato de Dados CSV

### Estrutura Esperada
```csv
ticker,date,time,trade_number,price,qty,vol,buy_agent,sell_agent,trade_type,aft
WDOFUT,20240108,090038,10,5840.00,25,146000.0,Capital,XP,Compra Agressão,N
```

### Tipos de Trade Suportados
- **Compra Agressão**: Comprador cruza spread (BUY)
- **Venda Agressão**: Vendedor cruza spread (SELL)
- **Leilão**: Trades de leilão
- **Cross Trade**: Negociação direta
- **Others**: Surveillance, OTC, Options Exercise, etc.

## 🔧 Configuração

### Variáveis de Ambiente (.env)
```env
# ProfitDLL
PROFIT_DLL_PATH=C:\Path\To\ProfitDLL.dll
PROFIT_KEY=sua_chave
PROFIT_USER=seu_usuario
PROFIT_PASSWORD=sua_senha

# Trading
TICKER=WDOFUT
INITIAL_BALANCE=100000
RISK_PER_TRADE=0.02

# ML
ML_INTERVAL=10
DIRECTION_THRESHOLD=0.60
MAGNITUDE_THRESHOLD=0.003
```

### Parâmetros do Dataset
```python
# Em create_dataset_from_csv.py
lookback_periods = 100    # Janela histórica para features
target_periods = 5        # Períodos futuros para predição
target_threshold = 0.001  # 0.1% para classificação
train_ratio = 0.7        # 70% treino
valid_ratio = 0.15       # 15% validação
test_ratio = 0.15        # 15% teste
```

## 🧪 Testes e Validação

### 1. Validar Dataset
```bash
python analyze_dataset_compatibility.py
# Verifica:
# - Features V3 presentes
# - Labels corretas
# - Regime detection
# - Qualidade dos dados
```

### 2. Backtest
```bash
python src/backtesting/backtester_v3.py
# Métricas esperadas:
# - Win Rate > 55%
# - Sharpe Ratio > 1.0
# - Max Drawdown < 10%
```

### 3. Conta Simulação
- Configure conta simulação na corretora
- Sistema enviará ordens reais pela infraestrutura de simulação
- Permite validação completa sem risco

## 📈 Regime de Mercado

O sistema detecta automaticamente 3 regimes:

### 1. **Trend (ADX > 25)**
- `trend_up`: EMA9 > EMA20 > EMA50
- `trend_down`: EMA9 < EMA20 < EMA50
- Estratégia: Follow trend, R:R 1:2

### 2. **Range (ADX < 25)**
- Estratégia: Mean reversion
- R:R 1:1.5

### 3. **Undefined**
- ADX > 25 mas EMAs desalinhadas
- Ação: HOLD (não opera)

## 🛡️ Segurança e Boas Práticas

### Validação de Dados
- Sistema bloqueia dados sintéticos em produção
- Validação automática de integridade
- Logs detalhados de todas operações

### Risk Management
- Stop loss obrigatório
- Limite de exposição por operação
- Circuit breakers automáticos

### Monitoramento
```python
# Logs importantes para monitorar
"Features calculadas"     # Cálculo de features
"Predição gerada"        # ML predictions
"Sinal de trading"       # Trading signals
"Ordem enviada"          # Order execution
```

## 🔍 Troubleshooting

### Dataset não compatível
```bash
# Verificar formato CSV
head -5 seu_arquivo.csv

# Analisar tipos de trade
python create_dataset_from_csv.py seu_arquivo.csv --analyze-only
```

### Modelos não carregando
```bash
# Verificar arquivos
ls models/*.pkl

# Verificar features
cat models/*_features.json
```

### Sem dados em tempo real
- Verificar se ProfitChart está aberto
- Confirmar credenciais no .env
- Testar conexão isoladamente

## 📚 Estrutura de Diretórios

```
ML_Tradingv2.0/
├── src/
│   ├── main.py                 # Entry point
│   ├── trading_system.py       # Sistema principal
│   ├── connection_manager.py   # Interface ProfitDLL
│   ├── ml/                     # ML components
│   ├── features/               # Feature engineering
│   └── data/                   # Data handling
├── models/                     # Modelos treinados
├── datasets/                   # Datasets processados
├── create_dataset_from_csv.py  # Processar CSV
├── analyze_dataset_compatibility.py # Validar dataset
├── requirements.txt            # Dependências
└── .env.example               # Template configuração
```

## 🚀 Próximos Passos

1. **Preparar dados**: Obter CSV com histórico de trades
2. **Criar dataset**: Processar com `create_dataset_from_csv.py`
3. **Treinar modelos**: Executar training orchestrator
4. **Validar**: Backtest e conta simulação
5. **Produção**: Deploy com monitoramento

## 📞 Suporte

- **Issues**: GitHub Issues
- **Docs**: Este arquivo + CLAUDE.md
- **Logs**: Verificar logs detalhados em tempo real

---

*Sistema em constante evolução. Mantenha este guia atualizado com mudanças significativas.*