# 🧪 Testes de Integração End-to-End

Este diretório contém testes de integração completos que validam o funcionamento do sistema QuantumTrader ML como um todo.

## 📋 Visão Geral

Os testes de integração verificam a interação entre múltiplos componentes do sistema, simulando cenários reais de trading.

## 🔧 Estrutura dos Testes

### 1. **test_complete_flow.py** - Fluxo Completo
Testa o pipeline completo desde a recepção de dados até a execução de ordens.

#### Cenários Testados:
- ✅ Dados → Predição → Sinal → Execução
- ✅ Integração com gestão de risco
- ✅ Sistema de aprendizado online
- ✅ Monitoramento adaptativo
- ✅ Ciclo completo de trade (abertura → monitoramento → fechamento)
- ✅ Recuperação após falhas de componentes

### 2. **test_risk_integration.py** - Gestão de Risco
Testa cenários complexos de risco com múltiplos componentes.

#### Cenários Testados:
- ✅ Cascata de stop loss em múltiplas posições
- ✅ Enforcement de limites de exposição
- ✅ Proteção contra drawdown excessivo
- ✅ Trailing stops integrados
- ✅ Position sizing dinâmico baseado em risco
- ✅ Agregação de risco entre símbolos correlacionados
- ✅ Fechamento de emergência
- ✅ Persistência de métricas de risco

### 3. **test_performance.py** - Performance do Sistema
Avalia latência, throughput e uso de recursos.

#### Métricas Testadas:
- ✅ Throughput de processamento de dados (>1000 msgs/seg)
- ✅ Latência de execução de ordens (<10ms média)
- ✅ Performance de cálculos de risco (<1ms)
- ✅ Operações concorrentes (10 threads)
- ✅ Uso de memória (<500MB aumento)
- ✅ Performance do rastreador de P&L
- ✅ Stress test completo (10 segundos, 100Hz dados)

### 4. **test_failure_recovery.py** - Recuperação de Falhas
Testa resiliência e recuperação após falhas.

#### Cenários Testados:
- ✅ Restart de componentes individuais
- ✅ Recuperação após perda de dados
- ✅ Tratamento de falhas de conexão
- ✅ Falha parcial do sistema
- ✅ Persistência de estado entre sessões
- ✅ Prevenção de falhas em cascata
- ✅ Degradação graciosa
- ✅ Sistema de checkpoints

## 🚀 Como Executar

### Executar Todos os Testes
```bash
python tests/integration/run_all_integration_tests.py
```

### Executar Categoria Específica
```bash
# Testes rápidos
python tests/integration/run_all_integration_tests.py quick

# Testes de performance
python tests/integration/run_all_integration_tests.py performance

# Testes de risco
python tests/integration/run_all_integration_tests.py risk

# Todos os testes
python tests/integration/run_all_integration_tests.py full
```

### Executar Teste Individual
```bash
# Teste específico
pytest tests/integration/test_complete_flow.py -v

# Com output detalhado
pytest tests/integration/test_performance.py -v -s

# Teste específico dentro do módulo
pytest tests/integration/test_risk_integration.py::TestRiskIntegration::test_stop_loss_cascade -v
```

## 📊 Interpretação dos Resultados

### Relatório Gerado
Após executar todos os testes, um relatório JSON é gerado:
```
integration_test_report_YYYYMMDD_HHMMSS.json
```

### Estrutura do Relatório
```json
{
  "start_time": "2024-01-01T10:00:00",
  "modules": {
    "test_complete_flow.py": {
      "exit_code": 0,
      "duration": 45.2,
      "status": "PASSED"
    }
  },
  "summary": {
    "total_passed": 4,
    "total_failed": 0,
    "success_rate": 100.0
  }
}
```

## 🎯 Critérios de Sucesso

### Performance
- **Throughput**: >1000 mensagens/segundo
- **Latência de ordem**: <10ms média, <50ms P99
- **Uso de memória**: <500MB para 100 posições + 50k data points

### Confiabilidade
- **Recuperação**: Sistema volta ao normal após falhas
- **Persistência**: Estado mantido entre sessões
- **Circuit breakers**: Ativam corretamente sob stress

### Funcionalidade
- **Fluxo completo**: Dados → Execução funciona end-to-end
- **Gestão de risco**: Todos os limites são respeitados
- **Integração**: Componentes se comunicam corretamente

## 🔍 Debugging

### Logs Detalhados
```bash
# Aumentar verbosidade
pytest tests/integration/test_complete_flow.py -vvv

# Mostrar prints
pytest tests/integration/test_performance.py -s

# Parar no primeiro erro
pytest tests/integration/test_risk_integration.py -x
```

### Executar com Coverage
```bash
pytest tests/integration/ --cov=src --cov-report=html
```

## 🛠️ Configuração de Ambiente

### Requisitos
- Python 3.8+
- Todos os requirements instalados
- Mínimo 4GB RAM disponível
- SSD recomendado para testes de performance

### Variáveis de Ambiente (Opcional)
```bash
# Reduzir carga para máquinas mais lentas
export TEST_PERFORMANCE_SCALE=0.5

# Aumentar timeouts
export TEST_TIMEOUT_MULTIPLIER=2.0
```

## 📈 Métricas de Cobertura

Os testes de integração cobrem:
- **Componentes**: 100% dos componentes principais
- **Fluxos**: Principais fluxos de negócio
- **Cenários de erro**: Falhas comuns e recuperação
- **Performance**: Carga normal e stress

## 🚨 Problemas Conhecidos

1. **Testes de performance** podem falhar em máquinas com baixo desempenho
2. **Testes de concorrência** podem ter resultados variáveis dependendo do CPU
3. **Uso de memória** pode variar entre sistemas operacionais

## 🔄 Manutenção

### Adicionar Novo Teste
1. Criar arquivo `test_novo_cenario.py`
2. Seguir padrão dos testes existentes
3. Adicionar ao `run_all_integration_tests.py`
4. Documentar neste README

### Atualizar Limites
Os limites de performance estão definidos nos próprios testes e podem ser ajustados conforme necessário.

---

**Última atualização**: Janeiro 2024
**Mantido por**: Equipe QuantumTrader ML