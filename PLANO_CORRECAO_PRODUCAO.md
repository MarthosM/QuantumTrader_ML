# Plano de Correção - Sistema de Produção

## 🎯 Objetivo
Criar um sistema de trading em produção que receba dados reais, calcule features e execute trades com ML.

## 📋 Fases de Correção

### Fase 1: Diagnóstico (Imediato) ✓
1. **test_credentials.py** - Validar credenciais corretas
2. **compare_callbacks.py** - Identificar diferenças na estrutura

### Fase 2: Desenvolvimento do Script Corrigido

#### 2.1 Criar production_fixed.py
```python
# Estrutura base copiada do book_collector_continuous.py
class ProductionFixed:
    def __init__(self):
        # FLAGS IDÊNTICAS ao book_collector
        self.bAtivo = False
        self.bMarketConnected = False
        self.bConnectado = False
        self.bBrokerConnected = False
        
    def _create_all_callbacks(self):
        # TODOS os callbacks ANTES do login
        # state, history, daily, price_book, offer_book, progress, tiny_book
```

#### 2.2 Validação de Dados
- Log detalhado de cada callback
- Contador para cada tipo de dado
- Validação de preços recebidos

#### 2.3 Integração ML
- Carregar modelos disponíveis
- Calcular features com dados reais
- Fazer predições quando houver dados suficientes

### Fase 3: Testes Progressivos

#### 3.1 Teste de Conexão
```bash
python test_credentials.py
# Validar qual credencial funciona
```

#### 3.2 Teste de Callbacks
```bash
python compare_callbacks.py
# Confirmar estrutura correta
```

#### 3.3 Teste de Recepção de Dados
```bash
python production_fixed.py --test-data
# Apenas conectar e mostrar dados recebidos
```

#### 3.4 Teste de Features
```bash
python production_fixed.py --test-features
# Calcular e validar features
```

#### 3.5 Teste de Predições
```bash
python production_fixed.py --test-ml
# Fazer predições sem executar trades
```

#### 3.6 Teste Completo
```bash
python production_fixed.py
# Sistema completo em produção
```

### Fase 4: Monitoramento

#### 4.1 Logs Estruturados
```
logs/production/
├── connection_YYYYMMDD.log     # Logs de conexão
├── data_YYYYMMDD.log          # Dados recebidos
├── features_YYYYMMDD.log      # Features calculadas
├── predictions_YYYYMMDD.log   # Predições ML
├── trades_YYYYMMDD.log        # Execuções
└── errors_YYYYMMDD.log        # Erros
```

#### 4.2 Métricas em Tempo Real
- Callbacks por segundo
- Latência de dados
- Features calculadas
- Predições por minuto
- P&L em tempo real

### Fase 5: Correções Específicas

#### 5.1 Problema: Login Error 200
**Solução**:
1. Executar test_credentials.py
2. Identificar credencial correta
3. Atualizar .env

#### 5.2 Problema: Sem Callbacks de Dados
**Solução**:
1. Usar estrutura EXATA do book_collector
2. Criar callbacks ANTES do login
3. Passar todos os 14 parâmetros para DLLInitializeLogin

#### 5.3 Problema: Features com NaN
**Solução**:
1. Aguardar mínimo de 20 candles
2. Validar dados antes de calcular
3. Usar valores default seguros

## 📊 Checklist de Implementação

### Pré-requisitos
- [ ] Credenciais validadas
- [ ] DLL carregando corretamente
- [ ] Estrutura de callbacks idêntica ao funcional

### Desenvolvimento
- [ ] production_fixed.py criado
- [ ] Callbacks implementados corretamente
- [ ] Recepção de dados validada
- [ ] Features calculadas com dados reais
- [ ] ML integrado e testado

### Testes
- [ ] Conexão bem sucedida
- [ ] Dados sendo recebidos
- [ ] Features sem NaN
- [ ] Predições coerentes
- [ ] Gestão de risco ativa

### Produção
- [ ] Monitor iniciando automaticamente
- [ ] Logs estruturados funcionando
- [ ] Sistema estável por 1 hora
- [ ] Primeiros trades simulados
- [ ] Métricas sendo coletadas

## 🚀 Comandos de Execução

```bash
# 1. Validar ambiente
python test_credentials.py
python compare_callbacks.py

# 2. Testar conexão
python production_fixed.py --test-connection

# 3. Testar dados
python production_fixed.py --test-data

# 4. Testar ML
python production_fixed.py --test-ml

# 5. Produção
python production_fixed.py

# 6. Monitor
# Inicia automaticamente ou:
python monitor_gui.py
```

## ⚠️ Pontos Críticos

1. **NUNCA** pular a criação de callbacks
2. **SEMPRE** usar a ordem exata de parâmetros
3. **JAMAIS** usar dados mockados em produção
4. **VERIFICAR** mercado aberto antes de esperar dados
5. **VALIDAR** ticker correto do mês

## 📈 Métricas de Sucesso

- Callbacks recebidos > 1000/minuto
- Features calculadas sem NaN
- Predições com confidence > 0.6
- Zero erros críticos em 1 hora
- P&L tracking funcionando

---

**Próximo Passo**: Executar `test_credentials.py` para validar credenciais