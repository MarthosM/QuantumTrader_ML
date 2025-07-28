# Guia de Implantação em Produção - ML Trading V3

## Resumo Executivo

Este guia documenta todos os procedimentos necessários para implantar o sistema ML Trading V3 em ambiente de produção. O sistema foi completamente refatorado para usar dados reais tick-by-tick do ProfitDLL.

**Status do Sistema**: ✅ Pronto para deploy após treinamento de modelos ML

## Índice

1. [Checklist Pré-Produção](#checklist-pré-produção)
2. [Requisitos de Sistema](#requisitos-de-sistema)
3. [Instalação e Configuração](#instalação-e-configuração)
4. [Treinamento de Modelos](#treinamento-de-modelos)
5. [Configuração de Trading](#configuração-de-trading)
6. [Monitoramento e Alertas](#monitoramento-e-alertas)
7. [Procedimentos Operacionais](#procedimentos-operacionais)
8. [Troubleshooting](#troubleshooting)
9. [Rollback e Recuperação](#rollback-e-recuperação)

## Checklist Pré-Produção

### ✅ Fase 1 - Coleta de Dados Real
- [x] RealDataCollector implementado
- [x] TradingDataStructureV3 com thread-safety
- [x] Validação de dados tick-by-tick
- [x] Testes de integração com CSV

### ✅ Fase 2 - Pipeline ML
- [x] MLFeaturesV3 com 118 features
- [x] DatasetBuilderV3 para preparação
- [x] TrainingOrchestratorV3 unificado
- [x] Validação temporal implementada

### ✅ Fase 3 - Sistema Real-Time
- [x] RealTimeProcessorV3 multi-thread
- [x] PredictionEngineV3 com modelos reais
- [x] ConnectionManagerV3 para ProfitDLL
- [x] SystemMonitorV3 com alertas

### ✅ Fase 4 - Testes de Integração
- [x] End-to-end test: 100% sucesso
- [x] Backtesting: Sistema funcional (estratégia precisa otimização)
- [x] Paper trading: Implementado
- [x] Risk metrics: Sistema completo
- [x] Stress testing: 8 cenários implementados

### ⚠️ Pendente
- [ ] Treinar modelos ML com dados históricos reais
- [ ] Otimizar estratégias por regime de mercado
- [ ] Configurar conexão real com ProfitDLL
- [ ] Definir limites de risco para produção

## Requisitos de Sistema

### Hardware Mínimo
```yaml
CPU: 4 cores @ 2.4GHz
RAM: 8GB (16GB recomendado)
Disco: 50GB SSD
Rede: Latência < 50ms para broker
```

### Software
```yaml
OS: Windows 10/11 64-bit
Python: 3.8-3.10
ProfitDLL: Versão mais recente
Dependências: requirements.txt
```

### Limites Operacionais (do Stress Test)
```yaml
Trades/segundo: 100-200 (máximo sustentável)
Threads paralelas: 20-50 (ideal)
Memória máxima: 2GB para operação estável
Latência aceitável: < 100ms
```

## Instalação e Configuração

### 1. Preparação do Ambiente
```bash
# Clonar repositório
git clone https://github.com/MarthosM/ML_Tradingv2.0.git
cd ML_Tradingv2.0

# Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Configuração Inicial
```python
# config/production_config.json
{
    "environment": "production",
    "data_source": "profitdll",
    "connection": {
        "dll_path": "C:\\ProfitDLL\\profit.dll",
        "server": "prod.broker.com",
        "port": 80,
        "username": "YOUR_USER",
        "password": "YOUR_PASS"
    },
    "trading": {
        "symbols": ["WDOU25"],
        "initial_capital": 100000,
        "max_position_size": 5,
        "risk_per_trade": 0.01
    },
    "risk_limits": {
        "max_drawdown": 0.10,
        "daily_loss_limit": 0.02,
        "max_positions": 1,
        "min_sharpe": 1.0
    },
    "monitoring": {
        "alert_email": "trader@company.com",
        "health_check_interval": 60,
        "metrics_export_interval": 300
    }
}
```

### 3. Estrutura de Diretórios
```
ML_Tradingv2.0/
├── data/              # Dados históricos
├── models/            # Modelos treinados (.pkl)
├── logs/              # Logs do sistema
├── config/            # Configurações
├── backups/           # Backups automáticos
└── monitoring/        # Dashboards e relatórios
```

## Treinamento de Modelos

### 1. Coletar Dados Históricos
```python
from src.data.real_data_collector import RealDataCollector
from datetime import datetime, timedelta

# Coletar últimos 6 meses
collector = RealDataCollector()
historical_data = collector.collect_from_csv(
    csv_path="data/wdo_6months.csv",
    start_date=datetime.now() - timedelta(days=180),
    end_date=datetime.now()
)
```

### 2. Treinar Modelos
```python
from src.ml.training_orchestrator_v3 import TrainingOrchestratorV3

# Configurar e treinar
orchestrator = TrainingOrchestratorV3(config)
results = orchestrator.train_complete_system(
    historical_data=historical_data,
    target_metrics={
        'accuracy': 0.55,
        'sharpe_ratio': 1.5
    }
)

# Modelos salvos em models/
# - xgboost_trend_up.pkl
# - lightgbm_trend_down.pkl
# - rf_range.pkl
# etc.
```

### 3. Validar Modelos
```python
# Backtest com dados out-of-sample
from src.backtesting.backtester_v3 import BacktesterV3

backtester = BacktesterV3(config)
validation_results = backtester.run_backtest(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
    data_path="data/validation_data.csv"
)

# Verificar métricas mínimas
assert validation_results['sharpe_ratio'] > 1.0
assert validation_results['max_drawdown'] < 0.10
```

## Configuração de Trading

### 1. Inicializar Sistema
```python
from src.main_v3 import TradingSystemV3

# Criar sistema
system = TradingSystemV3(config_path="config/production_config.json")

# Verificar componentes
system.run_health_check()
```

### 2. Conectar ao Broker
```python
# Conexão com ProfitDLL
system.connect_to_broker()

# Verificar conexão
if system.connection_manager.is_connected():
    print("Conectado ao broker")
else:
    raise ConnectionError("Falha na conexão")
```

### 3. Iniciar Trading
```python
# Modo paper trading primeiro
system.start_paper_trading(duration_days=5)

# Após validação, modo real
system.start_real_trading()
```

## Monitoramento e Alertas

### 1. Dashboard de Monitoramento
```python
# Sistema monitora automaticamente:
- Latência de processamento
- Taxa de erros
- P&L em tempo real
- Métricas de risco
- Saúde do sistema

# Acessar via:
http://localhost:8080/dashboard
```

### 2. Configurar Alertas
```python
# config/alerts.yaml
alerts:
  - name: "Max Drawdown"
    metric: "drawdown"
    threshold: 0.08
    action: "email,slack"
    
  - name: "High Latency"
    metric: "processing_latency"
    threshold: 100  # ms
    action: "log,email"
    
  - name: "System Error"
    metric: "error_rate"
    threshold: 0.01
    action: "email,stop_trading"
```

### 3. Logs e Auditoria
```python
# Logs estruturados em:
logs/
├── trading/      # Ordens e execuções
├── system/       # Eventos do sistema
├── errors/       # Erros e exceções
└── performance/  # Métricas de performance

# Rotação automática diária
# Retenção: 30 dias
```

## Procedimentos Operacionais

### Startup Diário
```bash
# 1. Verificar mercado
python scripts/check_market_status.py

# 2. Atualizar dados
python scripts/update_historical_data.py

# 3. Verificar modelos
python scripts/validate_models.py

# 4. Iniciar sistema
python src/main_v3.py --mode=production

# 5. Verificar dashboard
# http://localhost:8080
```

### Shutdown Diário
```bash
# 1. Fechar posições abertas
python scripts/close_all_positions.py

# 2. Gerar relatório diário
python scripts/generate_daily_report.py

# 3. Backup
python scripts/backup_system.py

# 4. Parar sistema
python src/main_v3.py --stop
```

### Manutenção Semanal
```bash
# 1. Análise de performance
python scripts/weekly_performance_analysis.py

# 2. Re-treinar modelos (se necessário)
python scripts/retrain_models.py --check-drift

# 3. Otimizar parâmetros
python scripts/optimize_parameters.py

# 4. Limpar logs antigos
python scripts/cleanup_logs.py --days=30
```

## Troubleshooting

### Problemas Comuns

#### 1. Conexão ProfitDLL
```python
# Erro: "DLL not found"
Solução: Verificar caminho da DLL e permissões

# Erro: "Connection timeout"
Solução: Verificar firewall e latência de rede
```

#### 2. Modelos não carregando
```python
# Erro: "Model file not found"
Solução: Verificar diretório models/ e arquivos .pkl

# Erro: "Feature mismatch"
Solução: Re-treinar modelos com features atuais
```

#### 3. Performance degradada
```python
# Sintoma: Latência > 100ms
Solução: 
- Verificar CPU/memória
- Reduzir threads paralelas
- Habilitar cache de features
```

### Logs de Diagnóstico
```bash
# Habilitar debug mode
python src/main_v3.py --debug --log-level=DEBUG

# Analisar logs específicos
python scripts/analyze_logs.py --component=prediction_engine
```

## Rollback e Recuperação

### Backup Automático
```python
# Configurado em config/backup.yaml
backup:
  frequency: "daily"
  retention: 7
  components:
    - models
    - config
    - data/processed
  destination: "backups/"
```

### Procedimento de Rollback
```bash
# 1. Parar sistema
python src/main_v3.py --emergency-stop

# 2. Identificar versão estável
python scripts/list_backups.py

# 3. Restaurar backup
python scripts/restore_backup.py --date=2025-01-27

# 4. Verificar integridade
python scripts/verify_system.py

# 5. Reiniciar
python src/main_v3.py --safe-mode
```

### Recuperação de Desastres
```bash
# 1. Ativar modo emergência
python scripts/emergency_mode.py

# 2. Fechar todas posições
python scripts/emergency_close_all.py

# 3. Gerar relatório de incidente
python scripts/incident_report.py

# 4. Notificar equipe
python scripts/notify_team.py --priority=high
```

## Checklist Final de Deploy

### Antes do Go-Live
- [ ] Todos testes passando (pytest)
- [ ] Modelos treinados e validados
- [ ] Backtest com Sharpe > 1.0
- [ ] Paper trading por 5 dias
- [ ] Conexão ProfitDLL testada
- [ ] Alertas configurados
- [ ] Backup testado
- [ ] Equipe treinada

### Dia 1 - Soft Launch
- [ ] Capital limitado (10% do total)
- [ ] Monitoramento intensivo
- [ ] Trades manuais em paralelo
- [ ] Registro de todas anomalias

### Semana 1 - Estabilização
- [ ] Aumentar capital gradualmente
- [ ] Ajustar parâmetros conforme necessário
- [ ] Documentar todos ajustes
- [ ] Revisar métricas diariamente

### Mês 1 - Otimização
- [ ] Análise completa de performance
- [ ] Re-treinar modelos com dados recentes
- [ ] Otimizar estratégias
- [ ] Planejar melhorias

## Contatos de Emergência

```yaml
Suporte Técnico:
  Email: support@tradingsystem.com
  Telefone: +55 11 9999-9999
  
Administrador do Sistema:
  Email: admin@company.com
  Celular: +55 11 8888-8888
  
Broker/Corretora:
  Mesa: +55 11 7777-7777
  Suporte: support@broker.com
```

## Conclusão

O sistema ML Trading V3 está tecnicamente pronto para produção, mas requer:

1. **Treinamento de modelos ML** com dados históricos reais
2. **Validação em paper trading** antes de usar capital real
3. **Monitoramento constante** nas primeiras semanas
4. **Ajustes contínuos** baseados em performance real

Seguindo este guia e mantendo disciplina operacional, o sistema pode operar de forma segura e eficiente em produção.

---

**Última atualização**: 2025-07-28
**Versão**: 3.0.0
**Status**: Aguardando treinamento de modelos para deploy