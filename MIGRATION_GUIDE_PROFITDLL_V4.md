# Guia de Migração para ProfitDLL v4.0.0.30

## 📋 Resumo das Mudanças

Este guia documenta o processo de migração do sistema atual para compatibilidade total com ProfitDLL v4.0.0.30.

### Status da Migração

- ✅ **Estruturas de Dados**: Implementadas em `profit_dll_structures.py`
- ✅ **Order Manager v4**: Criado `order_manager_v4.py` com suporte completo
- ✅ **Connection Manager v4**: Criado `connection_manager_v4.py` com callbacks atualizados
- ✅ **Testes**: Implementados em `test_profitdll_v4_compatibility.py`

## 🚀 Passos para Migração

### 1. Backup do Sistema Atual

Antes de iniciar a migração, faça backup dos arquivos existentes:

```bash
# Criar diretório de backup
mkdir backup_pre_v4
cp src/connection_manager.py backup_pre_v4/
cp src/order_manager.py backup_pre_v4/
```

### 2. Instalar Novos Arquivos

Os seguintes arquivos foram criados e devem ser adicionados ao projeto:

1. **`src/profit_dll_structures.py`**
   - Contém todas as estruturas ctypes para v4.0.0.30
   - Enums para tipos de ordem, status, etc.
   - Funções auxiliares para criar estruturas

2. **`src/order_manager_v4.py`**
   - Gerenciador de ordens compatível com v4.0.0.30
   - Usa `SendOrder` unificado em vez de funções depreciadas
   - Implementa `SendCancelOrderV2` para cancelamentos

3. **`src/connection_manager_v4.py`**
   - Callbacks com assinaturas corretas para v4.0.0.30
   - Implementa `SetOrderCallback` e outros callbacks V2
   - Validação aprimorada de dados de mercado

4. **`tests/test_profitdll_v4_compatibility.py`**
   - Suite completa de testes para validar a implementação

### 3. Atualizar Imports no Sistema

#### No `trading_system.py`:

```python
# Antes:
from connection_manager import ConnectionManager
from order_manager import OrderExecutionManager

# Depois:
from connection_manager_v4 import ConnectionManagerV4 as ConnectionManager
from order_manager_v4 import OrderExecutionManagerV4 as OrderExecutionManager
```

#### Em outros arquivos que usam estruturas:

```python
# Adicionar import das estruturas v4
from profit_dll_structures import (
    OrderSide, OrderType, OrderStatus,
    TConnectorAccountIdentifier, NResult
)
```

### 4. Mudanças Específicas no Código

#### A. Envio de Ordens

**Antes (Depreciado):**
```python
# Order Manager antigo
def _send_market_order(self, order, exchange):
    if order.side == OrderSide.BUY:
        return dll.SendMarketBuyOrder(...)  # DEPRECIADO
    else:
        return dll.SendMarketSellOrder(...)  # DEPRECIADO
```

**Depois (v4.0.0.30):**
```python
# Order Manager v4
def _send_order_v4(self, order):
    send_order = create_send_order(
        account=self.account_identifier,
        symbol=order.symbol,
        side=order.side,
        order_type=order.order_type,
        quantity=order.quantity,
        price=order.price
    )
    return self.dll.SendOrder(byref(send_order))
```

#### B. Cancelamento de Ordens

**Antes (Depreciado):**
```python
result = dll.SendCancelOrder(...)  # DEPRECIADO
```

**Depois (v4.0.0.30):**
```python
cancel_order = create_cancel_order(
    account=self.account_identifier,
    client_order_id=order.cl_ord_id
)
result = dll.SendCancelOrderV2(byref(cancel_order))
```

#### C. Gestão de Contas

**Antes:**
```python
# Informações básicas de conta
self.account_info = {
    'account_id': '12345',
    'broker_id': '1'
}
```

**Depois (v4.0.0.30):**
```python
# Usar TConnectorAccountIdentifier
self.account_identifier = create_account_identifier(
    broker_id=1,  # Numérico
    account_id='12345',
    sub_account_id='001'  # Novo campo
)
```

### 5. Executar Testes de Validação

```bash
# Executar suite de testes
pytest tests/test_profitdll_v4_compatibility.py -v

# Executar com coverage
pytest tests/test_profitdll_v4_compatibility.py --cov=src --cov-report=html
```

### 6. Rollback (Se Necessário)

Se houver problemas, volte aos arquivos originais:

```bash
# Restaurar arquivos originais
cp backup_pre_v4/connection_manager.py src/
cp backup_pre_v4/order_manager.py src/
```

## ⚠️ Pontos de Atenção

### 1. Compatibilidade de Callbacks

Os callbacks mudaram significativamente. Certifique-se de que todos os callbacks estejam usando as novas assinaturas:

- ❌ `TOrderChangeCallback` → ✅ `TConnectorOrderCallback`
- ❌ `THistoryCallback` → ✅ `TConnectorAccountCallback`
- ❌ `SetTradeCallback` → ✅ `SetTradeCallbackV2`

### 2. Tipos de Dados

- IDs de corretora agora são **numéricos** (int), não strings
- Client Order IDs devem ser únicos e gerados pelo sistema
- Novos campos em estruturas (ex: SubAccountID)

### 3. Códigos de Retorno

Use os enums `NResult` para verificar retornos:

```python
if result == NResult.NL_OK:
    # Sucesso
elif result == NResult.NL_INVALID_ARGS:
    # Argumentos inválidos
```

### 4. Gestão de Posições

```python
# Usar GetPositionV2 com estrutura
position = TConnectorTradingAccountPosition()
result = dll.GetPositionV2(byref(position))
```

## 📊 Benefícios da Migração

1. **Performance**: Funções otimizadas na v4.0.0.30
2. **Confiabilidade**: Menos bugs e melhor tratamento de erros
3. **Funcionalidades**: Suporte a sub-contas e novos tipos de ordem
4. **Futuro**: Funções antigas serão removidas em versões futuras

## 🔍 Checklist de Migração

- [ ] Backup dos arquivos atuais
- [ ] Instalar novos arquivos (structures, managers v4)
- [ ] Atualizar imports no trading_system.py
- [ ] Atualizar imports em outros módulos
- [ ] Executar testes de validação
- [ ] Testar em ambiente de desenvolvimento
- [ ] Monitorar logs para erros
- [ ] Validar envio de ordens
- [ ] Validar cancelamento de ordens
- [ ] Validar callbacks de atualização
- [ ] Deploy em produção

## 📞 Suporte

Em caso de dúvidas ou problemas:

1. Consulte o manual oficial: `Manual - ProfitDLL en_us.pdf`
2. Verifique os logs detalhados do sistema
3. Execute os testes de compatibilidade
4. Revise este guia de migração

## 🎯 Próximos Passos

Após migração bem-sucedida:

1. Remover arquivos antigos (após período de estabilização)
2. Atualizar documentação do projeto
3. Treinar equipe nas novas APIs
4. Implementar funcionalidades avançadas (sub-contas, etc.)

---

**Data da Migração**: Janeiro 2025  
**Versão ProfitDLL**: 4.0.0.30  
**Responsável**: Sistema de Trading ML v2.0