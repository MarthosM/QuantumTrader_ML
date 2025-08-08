# 📖 Manual de Operação - QuantumTrader ML

## Sistema de Trading com 65 Features + HMARL

### Versão 2.0.0 | Produção Local

---

## 🚀 Início Rápido

### 1. Iniciar o Sistema

```bash
# Ativar ambiente virtual
.venv\Scripts\activate

# Iniciar sistema de produção
python start_production_65features.py
```

### 2. Parar o Sistema

```bash
# Parada segura
python stop_production.py
```

### 3. Verificar Status

```bash
# Verificar se está rodando
python check_system_status.py
```

---

## 🏗️ Arquitetura do Sistema

### Componentes Principais

1. **Enhanced Production System** - Core com 65 features
2. **HMARL Agents** - 4 agentes especializados
3. **Consensus Engine** - Combina ML (40%) + Agentes (60%)
4. **Metrics & Alerts** - Monitoramento em tempo real
5. **Structured Logger** - Logs em JSON para análise
6. **Enhanced Monitor V2** - Interface visual (opcional)

### Fluxo de Dados

```
ProfitDLL → Callbacks → Buffers → Features (65) → ML + Agents → Consenso → Trading
                                        ↓
                                   Broadcasting ZMQ
                                        ↓
                                Monitor & Logs & Metrics
```

---

## ⚙️ Configuração

### Arquivos de Configuração

1. **`.env.production`** - Variáveis de ambiente
2. **`config_production.json`** - Configuração detalhada do sistema

### Parâmetros Importantes

| Parâmetro | Valor Padrão | Descrição |
|-----------|--------------|-----------|
| MAX_POSITION | 1 | Posição máxima |
| STOP_LOSS | 0.005 | Stop loss (0.5%) |
| MIN_CONFIDENCE | 0.60 | Confiança mínima para trade |
| MAX_DAILY_TRADES | 10 | Máximo de trades por dia |
| ML_WEIGHT | 0.40 | Peso do modelo ML |
| AGENT_WEIGHT | 0.60 | Peso dos agentes |

### Ajustar Configurações

```bash
# Editar configuração de produção
notepad .env.production

# Editar configuração detalhada
notepad config_production.json
```

---

## 📊 Monitoramento

### Monitor Visual

O sistema inclui um monitor visual que mostra:
- **65 Features** em tempo real
- **Sinais dos 4 Agentes HMARL**
- **Gráficos de Microestrutura**
- **Métricas de Performance**
- **Logs do Sistema**

### Logs Estruturados

Logs são salvos em formato JSON em `logs/`:

```json
{
  "timestamp": "2025-08-08T10:30:00",
  "level": "INFO",
  "component": "TradingSystem",
  "message": "Trade signal: BUY",
  "data": {
    "signal": "BUY",
    "confidence": 0.75,
    "features_used": 65
  }
}
```

### Métricas

Métricas em tempo real disponíveis em `metrics/current_metrics.json`:
- Features/segundo
- Latência média
- Win rate
- Drawdown
- PnL

---

## 🛠️ Troubleshooting

### Sistema não inicia

1. **Verificar dependências**:
```bash
pip install -r requirements.txt
```

2. **Verificar ProfitDLL**:
- Profit Chart deve estar aberto
- Verificar se a chave está correta em `.env.production`

3. **Verificar portas**:
```bash
# Verificar se portas estão livres
netstat -an | findstr "5559"  # ZMQ
netstat -an | findstr "8080"  # Monitor Web (se habilitado)
```

### Features retornando zero

1. **Verificar buffers**:
- Aguardar pelo menos 200 candles
- Verificar callbacks do ProfitDLL

2. **Verificar modelos**:
```bash
# Listar modelos disponíveis
dir models\*.pkl
```

### Agentes não respondendo

1. **Verificar ZMQ**:
- Porta 5559 deve estar livre
- Verificar logs de broadcasting

2. **Verificar features necessárias**:
- Cada agente requer features específicas
- Verificar se todas estão sendo calculadas

### Memória alta

1. **Ajustar buffers** em `.env.production`:
```
FEATURE_BUFFER_SIZE=100  # Reduzir se necessário
BOOK_BUFFER_SIZE=50
```

2. **Habilitar garbage collection**:
```
GARBAGE_COLLECTION_INTERVAL_MIN=15
```

---

## 📈 Performance

### Requisitos do Sistema

- **CPU**: 4+ cores recomendado
- **RAM**: 4GB mínimo, 8GB recomendado
- **Disco**: 10GB livre para logs e backups
- **OS**: Windows 10/11

### Otimização

1. **Desabilitar monitor visual** se não necessário:
```json
"enhanced_monitor": {
  "enabled": false
}
```

2. **Ajustar intervalo de cálculo**:
```bash
# Em .env.production
FEATURE_CALCULATION_INTERVAL_MS=1000  # Aumentar se necessário
```

3. **Limitar logs**:
```bash
LOG_LEVEL=WARNING  # Reduzir verbosidade
```

---

## 🔐 Segurança

### Backup Automático

O sistema faz backup automático a cada 60 minutos em `backups/`

### Recovery

Em caso de crash, o sistema tenta recuperar automaticamente:
- Máximo 3 tentativas
- Intervalo de 5 segundos entre tentativas

### Parada de Emergência

```bash
# Parada forçada (usar apenas em emergência)
taskkill /F /IM python.exe
```

---

## 📋 Checklist Diário

### Antes de Abrir o Mercado

- [ ] Verificar conexão com Profit Chart
- [ ] Verificar espaço em disco (logs)
- [ ] Limpar logs antigos se necessário
- [ ] Verificar configuração de risco
- [ ] Iniciar sistema com 15 min de antecedência

### Durante o Pregão

- [ ] Monitorar latência (deve ser < 10ms)
- [ ] Verificar alertas no log
- [ ] Acompanhar drawdown
- [ ] Verificar posições abertas

### Após Fechamento

- [ ] Verificar PnL do dia
- [ ] Fazer backup dos logs
- [ ] Analisar trades executados
- [ ] Parar sistema de forma segura

---

## 📞 Comandos Úteis

### Status e Monitoramento

```bash
# Ver logs em tempo real
tail -f logs\production_*.log

# Contar trades do dia
findstr "Trade signal" logs\production_*.log | find /c "Trade"

# Ver últimos erros
findstr "ERROR" logs\production_*.log

# Verificar uso de memória
wmic process where name="python.exe" get WorkingSetSize
```

### Manutenção

```bash
# Limpar logs antigos (> 30 dias)
forfiles /p "logs" /s /m *.log /d -30 /c "cmd /c del @file"

# Limpar cache
del /Q cache\*.tmp

# Compactar backups
tar -czf backups_archive.tar.gz backups/
```

### Análise

```bash
# Analisar features mais importantes
python analyze_feature_importance.py

# Verificar performance dos agentes
python analyze_agent_performance.py

# Gerar relatório diário
python generate_daily_report.py
```

---

## 🚨 Alertas e Notificações

### Níveis de Alerta

| Nível | Descrição | Ação Recomendada |
|-------|-----------|------------------|
| INFO | Operação normal | Nenhuma |
| WARNING | Atenção necessária | Monitorar |
| ERROR | Problema detectado | Investigar |
| CRITICAL | Sistema em risco | Intervir imediatamente |

### Alertas Comuns

1. **High Latency** (> 50ms)
   - Verificar CPU/Memória
   - Reduzir frequência de cálculo

2. **Low Win Rate** (< 45%)
   - Revisar parâmetros
   - Verificar condições de mercado

3. **High Drawdown** (> 10%)
   - Reduzir tamanho de posição
   - Considerar parar trading

4. **Connection Lost**
   - Verificar Profit Chart
   - Verificar rede

---

## 📊 Relatórios

### Relatório Diário

Gerado automaticamente em `reports/daily/`:
- Total de trades
- Win rate
- PnL
- Drawdown máximo
- Features mais utilizadas
- Performance dos agentes

### Análise Semanal

```bash
# Gerar análise semanal
python generate_weekly_analysis.py
```

### Export para Excel

```bash
# Exportar métricas para Excel
python export_metrics_excel.py
```

---

## 🔄 Atualizações

### Atualizar Modelos

```bash
# Treinar novos modelos com dados recentes
python train_models_production.py

# Validar novos modelos
python validate_models.py

# Deploy (substituir modelos)
python deploy_models.py
```

### Atualizar Sistema

```bash
# Backup atual
python backup_system.py

# Atualizar código
git pull origin main

# Reinstalar dependências
pip install -r requirements.txt --upgrade

# Testar
python test_system_integrity.py
```

---

## 💡 Dicas e Boas Práticas

1. **Sempre iniciar com paper trading** para validar configurações
2. **Monitorar as primeiras horas** após mudanças
3. **Manter logs por pelo menos 30 dias** para análise
4. **Fazer backup antes de mudanças** importantes
5. **Documentar todas as alterações** de configuração
6. **Revisar alertas diariamente**
7. **Analisar trades perdedores** para melhorias
8. **Atualizar modelos mensalmente** com novos dados
9. **Testar em horários de menor volume** primeiro
10. **Ter plano de contingência** para falhas

---

## 📝 Registro de Mudanças

### Template para Registro

```markdown
Data: YYYY-MM-DD HH:MM
Responsável: Nome
Mudança: Descrição da alteração
Motivo: Por que foi feita
Resultado: O que aconteceu
```

### Exemplo

```markdown
Data: 2025-08-08 14:30
Responsável: Sistema
Mudança: Ajustado MIN_CONFIDENCE de 0.55 para 0.60
Motivo: Reduzir número de falsos positivos
Resultado: Win rate aumentou de 52% para 58%
```

---

## 🆘 Suporte

### Logs de Debug

Para debug detalhado:
```bash
# Ativar modo debug
set LOG_LEVEL=DEBUG
python start_production_65features.py
```

### Verificar Integridade

```bash
# Testar todos os componentes
python test_all_components.py
```

### Reset Completo

```bash
# Parar tudo
python stop_production.py

# Limpar tudo
python clean_all.py

# Reiniciar
python start_production_65features.py
```

---

## 📚 Referências

- [CLAUDE.md](CLAUDE.md) - Guia para desenvolvimento
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Documentação técnica
- [config_production.json](config_production.json) - Configuração completa
- [.env.production](.env.production) - Variáveis de ambiente

---

**Última atualização**: 08/08/2025  
**Versão do Sistema**: 2.0.0  
**Status**: Produção Local