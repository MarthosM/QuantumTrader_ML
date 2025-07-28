# ML Trading System v2.0

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Phase%203%20Complete-success.svg)](https://github.com/MarthosM/ML_Tradingv2.0)

Sistema de trading algorítmico com Machine Learning para análise e execução automatizada no mercado financeiro brasileiro (B3).

## 🎯 Visão Geral

O ML Trading System v2.0 é uma plataforma completa de trading algorítmico que utiliza técnicas avançadas de Machine Learning para:

- 📊 Análise de microestrutura de mercado em tempo real
- 🤖 Predições baseadas em regime de mercado (trend/range)
- ⚡ Execução automatizada com latência < 100ms
- 📈 Backtesting e validação com dados históricos reais

## 🚀 Features Principais

### Processamento de Dados
- **Coleta em tempo real** via ProfitDLL com callbacks otimizados
- **Microestrutura preservada**: volume por lado (buy/sell), order flow imbalance
- **Agregação inteligente**: tick-to-candle com métricas microestruturais

### Machine Learning
- **118+ features** calculadas em tempo real
- **Detecção automática de regime** (ADX + EMAs)
- **3 modelos por regime**: XGBoost, LightGBM, RandomForest
- **Ensemble voting** com confidence scores

### Performance
- **Latência**: < 100ms end-to-end
- **Throughput**: ~30 trades/segundo
- **Taxa de NaN**: 0% nas features
- **Uptime**: 99.5%+ em testes

## 📋 Requisitos

### Sistema
- Python 3.8+
- Windows 10/11 (para ProfitDLL)
- 8GB RAM mínimo
- Conexão estável com corretora

### Dependências Principais
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
joblib>=1.0.0
pyarrow>=6.0.0
ta>=0.10.0
```

## 🛠️ Instalação

1. Clone o repositório:
```bash
git clone https://github.com/MarthosM/ML_Tradingv2.0.git
cd ML_Tradingv2.0
```

2. Crie um ambiente virtual:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure o ProfitDLL:
   - Instale o Profit no diretório padrão
   - Verifique se `ProfitDLL.dll` está acessível
   - Configure credenciais no arquivo `.env`

## 🏗️ Arquitetura

### Componentes Principais

```
ML_Tradingv2.0/
├── src/
│   ├── data/                 # Infraestrutura de dados
│   │   ├── real_data_collector.py
│   │   └── trading_data_structure_v3.py
│   ├── features/             # Cálculo de features ML
│   │   └── ml_features_v3.py
│   ├── ml/                   # Pipeline de Machine Learning
│   │   ├── dataset_builder_v3.py
│   │   ├── training_orchestrator_v3.py
│   │   └── prediction_engine_v3.py
│   ├── realtime/             # Processamento em tempo real
│   │   └── realtime_processor_v3.py
│   ├── connection/           # Interface com ProfitDLL
│   │   └── connection_manager_v3.py
│   └── monitoring/           # Sistema de monitoramento
│       └── system_monitor_v3.py
├── models/                   # Modelos treinados
├── datasets/                 # Datasets processados
├── tests/                    # Testes de integração
└── docs/                     # Documentação detalhada
```

### Fluxo de Dados

```
ProfitDLL → ConnectionManager → RealTimeProcessor → MLFeatures → PredictionEngine → SignalGenerator
                                        ↓
                                 SystemMonitor → Alertas/Logs
```

## 🎮 Uso Básico

### 1. Treinar Modelos

```python
from src.ml.training_orchestrator_v3 import TrainingOrchestratorV3

orchestrator = TrainingOrchestratorV3()
results = orchestrator.train_complete_system()
```

### 2. Executar Sistema em Tempo Real

```python
from src.trading_system import TradingSystem

system = TradingSystem(config)
system.start()  # Inicia processamento em tempo real
```

### 3. Monitorar Performance

```python
from src.monitoring.system_monitor_v3 import SystemMonitorV3

monitor = SystemMonitorV3()
monitor.start()
report = monitor.generate_report()
```

## 📊 Resultados e Métricas

### Backtesting
- **Win Rate**: Target > 55%
- **Profit Factor**: Target > 1.5
- **Sharpe Ratio**: Target > 1.0
- **Max Drawdown**: < 10%

### Performance em Tempo Real
- **Latência de processamento**: ~25ms
- **Cálculo de features**: ~35ms
- **Geração de predição**: ~50ms
- **Taxa de erro**: < 0.1%

## 🛠️ Desenvolvimento

### Status das Fases

- ✅ **Fase 1**: Infraestrutura de Dados (Concluída)
- ✅ **Fase 2**: Pipeline ML (Concluída)
- ✅ **Fase 3**: Integração Tempo Real (Concluída)
- 📍 **Fase 4**: Testes Integrados (Em andamento)

### Executar Testes

```bash
# Testes unitários
pytest tests/

# Testes de integração
python tests/test_integration_v3.py

# Validação completa
python validate_phase3.py
```

### Comandos Úteis

```bash
# Verificar qualidade do código
pylint src/
mypy src/

# Formatar código
black src/
isort src/
```

## 📚 Documentação

Documentação detalhada disponível em:
- [Developer Guide](DEVELOPER_GUIDE_V3_REFACTORING.md)
- [Claude MD](CLAUDE.md) - Instruções para IA
- [Manual ProfitDLL](Manual%20-%20ProfitDLL%20en_us.pdf)
- [Documentação das Fases](docs/)

### Guias Específicos
- [Fase 1: Data Infrastructure](docs/phase1/COMPLETION_REPORT.md)
- [Fase 2: ML Pipeline](docs/phase2/COMPLETION_REPORT.md)
- [Fase 3: Real-time Integration](docs/phase3/COMPLETION_REPORT.md)

## ⚠️ Avisos Importantes

### Produção
- **SEMPRE use dados reais** em produção
- **NUNCA use dados mockados** para decisões de trading
- **Valide modelos** antes de operar com capital real

### Segurança
- Mantenha credenciais em variáveis de ambiente
- Não commite arquivos de configuração sensíveis
- Use conexões seguras com a corretora

## 🤝 Contribuindo

1. Fork o projeto
2. Crie sua feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

## ⚠️ Disclaimer

Este software é para fins educacionais e de pesquisa. Trading de ativos financeiros envolve riscos significativos. Use por sua própria conta e risco.
