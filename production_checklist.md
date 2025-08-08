# 📋 Checklist para Iniciar em Produção (Conta Simulador)

## ✅ Pré-requisitos

### 1. **Ambiente Python**
- [ ] Python 3.8+ instalado
- [ ] Ambiente virtual criado e ativado
- [ ] Todas as dependências instaladas (`pip install -r requirements.txt`)

### 2. **ProfitChart**
- [ ] ProfitChart aberto e logado
- [ ] Conta SIMULADOR selecionada
- [ ] Gráfico do WDO aberto
- [ ] Conexão com servidor estável

### 3. **Credenciais**
- [ ] Criar arquivo `.env` baseado no `.env.example`
- [ ] Configurar PROFIT_USER e PROFIT_PASS
- [ ] Verificar se as credenciais estão corretas

### 4. **Modelos ML**
- [ ] Modelos treinados presentes em:
  - `models/csv_5m/lightgbm_tick.txt`
  - `models/book_only/lightgbm_book_only_optimized.txt`
- [ ] Arquivos de features e scalers correspondentes

### 5. **Estrutura de Diretórios**
```
QuantumTrader_ML/
├── logs/
│   └── production/
├── data/
│   └── checkpoints/
├── reports/
├── alerts/
└── models/
    ├── csv_5m/
    └── book_only/
```

## 🚀 Processo de Inicialização

### 1. **Verificar Sistema**
```bash
# Testar conexão com ProfitDLL
python test_profitdll_connection.py

# Verificar modelos
python -c "from src.strategies.hybrid_strategy import HybridStrategy; s = HybridStrategy({}); s.load_models()"
```

### 2. **Iniciar com Parâmetros Conservadores**
```bash
# Configurar variáveis de ambiente
set PROFIT_USER=seu_usuario
set PROFIT_PASS=sua_senha

# Iniciar sistema
python start_production.py
```

### 3. **Monitorar Sistema**
- Dashboard: http://localhost:5000
- Logs: `logs/production/trading_YYYYMMDD_HHMMSS.log`
- Kill switch: Criar arquivo `STOP_TRADING.txt` para parar

## 📊 Métricas para Acompanhar

### Primeira Hora
- [ ] Sistema conectou corretamente
- [ ] Dados sendo recebidos (tick + book)
- [ ] Features sendo calculadas
- [ ] Modelos gerando predições
- [ ] Sem erros críticos nos logs

### Primeiro Dia
- [ ] Número de sinais gerados
- [ ] Número de trades executados
- [ ] P&L do dia
- [ ] Drawdown máximo
- [ ] Taxa de acerto
- [ ] Latência média de execução

### Primeira Semana
- [ ] Consistência dos resultados
- [ ] Comportamento em diferentes regimes de mercado
- [ ] Necessidade de ajustes nos parâmetros
- [ ] Performance do sistema (CPU/memória)

## ⚠️ Sinais de Alerta

**Parar imediatamente se:**
- Drawdown > 2% do capital
- Mais de 5 trades consecutivos com perda
- Latência de execução > 100ms consistentemente
- Erros repetidos nos logs
- Comportamento anormal nas predições

## 📈 Plano de Escalonamento

### Semana 1: Validação
- 1 contrato máximo
- Stop loss 0.5%
- Operar apenas 4 horas por dia

### Semana 2-4: Ajustes
- Analisar resultados
- Ajustar thresholds de confiança
- Otimizar parâmetros de risco

### Mês 2: Expansão Gradual
- Aumentar para 2 contratos (se lucrativo)
- Expandir horário de operação
- Habilitar mais features avançadas

### Mês 3+: Otimização
- Retreino com dados de produção
- A/B testing de novos modelos
- Otimização de latência

## 🛠️ Troubleshooting

### Problema: Não conecta com ProfitDLL
```bash
# Verificar se ProfitChart está aberto
# Verificar credenciais no .env
# Testar conexão isoladamente
python test_profitdll_connection.py
```

### Problema: Modelos não carregam
```bash
# Verificar caminhos
python -c "import os; print(os.listdir('models'))"
# Recriar modelos se necessário
python train_models.py
```

### Problema: Sistema muito lento
- Reduzir buffer_size no config
- Desabilitar features não essenciais
- Verificar uso de CPU/memória
- Considerar otimizações de código

## 📞 Suporte

Em caso de problemas:
1. Verificar logs em `logs/production/`
2. Criar snapshot do estado: `python create_diagnostic_snapshot.py`
3. Documentar:
   - Hora do problema
   - Mensagens de erro
   - Estado do mercado
   - Ações tomadas

---

**LEMBRE-SE**: Começar conservador e escalar gradualmente!