# Guia ProfitDLL - Conexão e Envio de Ordens

## 1. Inicialização e Conexão

### 1.1 Inicialização com Routing (permite envio de ordens)

```python
# Função principal para inicializar com routing
DLLInitializeLogin(
    pwcActivationKey,      # Chave de ativação fornecida
    pwcUser,              # Usuário da conta
    pwcPassword,          # Senha de login
    StateCallback,        # Callback de estado da conexão
    HistoryCallback,      # Callback de histórico de ordens
    OrderChangeCallback,  # Callback de mudança de ordens
    AccountCallback,      # Callback de informações da conta
    NewTradeCallback,     # Callback de trades em tempo real
    NewDailyCallback,     # Callback de dados diários
    PriceBookCallback,    # Callback de book de preços
    OfferBookCallback,    # Callback de book de ofertas
    HistoryTradeCallback, # Callback de histórico de trades
    ProgressCallback,     # Callback de progresso
    TinyBookCallback      # Callback de topo do book
)
```

### 1.2 Estados de Conexão

```python
# Estados principais para monitorar via StateCallback
CONNECTION_STATE_LOGIN = 0        # Conexão ao servidor de login
CONNECTION_STATE_ROTEAMENTO = 1   # Conexão ao servidor de roteamento
CONNECTION_STATE_MARKET_DATA = 2  # Conexão ao servidor de market data

# Status de conexão bem-sucedida
LOGIN_CONNECTED = 0               # Login conectado
ROTEAMENTO_CONNECTED = 2          # Roteamento conectado
MARKET_CONNECTED = 4              # Market data conectado
```

### 1.3 Obter Informações da Conta

```python
# Após conexão, obter dados das contas disponíveis
GetAccount()  # Retorna informações via AccountCallback
```

## 2. Envio de Ordens

### 2.1 Ordens Limitadas

```python
# Ordem de COMPRA limitada
order_id = SendBuyOrder(
    pwcIDAccount,      # ID da conta (obtido via GetAccount)
    pwcIDCorretora,    # ID da corretora (obtido via GetAccount)
    pwcSenha,          # Senha de roteamento
    pwcTicker,         # Ticker do ativo (ex: "WDOQ25")
    pwcBolsa,          # Bolsa (ex: "F" para futuros)
    dPrice,            # Preço alvo
    nAmount            # Quantidade
)

# Ordem de VENDA limitada
order_id = SendSellOrder(
    pwcIDAccount,      # ID da conta
    pwcIDCorretora,    # ID da corretora
    pwcSenha,          # Senha de roteamento
    pwcTicker,         # Ticker do ativo
    pwcBolsa,          # Bolsa
    dPrice,            # Preço alvo
    nAmount            # Quantidade
)
```

### 2.2 Ordens a Mercado

```python
# Ordem de COMPRA a mercado
order_id = SendMarketBuyOrder(
    pwcIDAccount,      # ID da conta
    pwcIDCorretora,    # ID da corretora
    pwcSenha,          # Senha de roteamento
    pwcTicker,         # Ticker do ativo
    pwcBolsa,          # Bolsa
    nAmount            # Quantidade
)

# Ordem de VENDA a mercado
order_id = SendMarketSellOrder(
    pwcIDAccount,      # ID da conta
    pwcIDCorretora,    # ID da corretora
    pwcSenha,          # Senha de roteamento
    pwcTicker,         # Ticker do ativo
    pwcBolsa,          # Bolsa
    nAmount            # Quantidade
)
```

### 2.3 Ordens Stop

```python
# Stop de COMPRA
order_id = SendStopBuyOrder(
    pwcIDAccount,      # ID da conta
    pwcIDCorretora,    # ID da corretora
    pwcSenha,          # Senha de roteamento
    pwcTicker,         # Ticker do ativo
    pwcBolsa,          # Bolsa
    dPrice,            # Preço alvo de compra
    dStopPrice,        # Preço de disparo do stop
    nAmount            # Quantidade
)

# Stop de VENDA
order_id = SendStopSellOrder(
    pwcIDAccount,      # ID da conta
    pwcIDCorretora,    # ID da corretora
    pwcSenha,          # Senha de roteamento
    pwcTicker,         # Ticker do ativo
    pwcBolsa,          # Bolsa
    dPrice,            # Preço alvo de venda
    dStopPrice,        # Preço de disparo do stop
    nAmount            # Quantidade
)
```

### 2.4 Gerenciamento de Ordens

```python
# Modificar ordem existente
SendChangeOrder(
    pwcIDAccount,      # ID da conta
    pwcIDCorretora,    # ID da corretora
    pwcSenha,          # Senha de roteamento
    pwcstrClOrdID,     # ClOrdID da ordem (obtido via OrderChangeCallback)
    dPrice,            # Novo preço
    nAmount            # Nova quantidade
)

# Cancelar ordem específica
SendCancelOrder(
    pwcIDAccount,      # ID da conta
    pwcIDCorretora,    # ID da corretora
    pwcClOrdId,        # ClOrdID da ordem a cancelar
    pwcSenha           # Senha de roteamento
)

# Cancelar todas as ordens de um ativo
SendCancelOrders(
    pwcIDAccount,      # ID da conta
    pwcIDCorretora,    # ID da corretora
    pwcSenha,          # Senha de roteamento
    pwcTicker,         # Ticker do ativo
    pwcBolsa           # Bolsa
)

# Cancelar TODAS as ordens abertas
SendCancelAllOrders(
    pwcIDAccount,      # ID da conta
    pwcIDCorretora,    # ID da corretora
    pwcSenha           # Senha de roteamento
)
```

### 2.5 Zerar Posição

```python
# Zerar posição com preço limite
order_id = SendZeroPosition(
    pwcIDAccount,      # ID da conta
    pwcIDCorretora,    # ID da corretora
    pwcTicker,         # Ticker do ativo
    pwcBolsa,          # Bolsa
    pwcSenha,          # Senha de roteamento
    dPrice             # Preço da ordem
)

# Zerar posição a mercado
order_id = SendZeroPositionAtMarket(
    pwcIDAccount,      # ID da conta
    pwcIDCorretora,    # ID da corretora
    pwcTicker,         # Ticker do ativo
    pwcBolsa,          # Bolsa
    pwcSenha           # Senha de roteamento
)
```

## 3. Consulta de Ordens e Posições

```python
# Obter histórico de ordens
GetOrders(
    pwcIDAccount,      # ID da conta
    pwcIDCorretora,    # ID da corretora
    dtStart,           # Data inicial (formato: "DD/MM/YYYY")
    dtEnd              # Data final (formato: "DD/MM/YYYY")
)

# Obter ordem específica por ClOrdID
GetOrder(pwcClOrdId)

# Obter ordem por ProfitID (ID interno da sessão)
GetOrderProfitID(nProfitID)

# Obter posição atual
position_data = GetPosition(
    pwcIDAccount,      # ID da conta
    pwcIDCorretora,    # ID da corretora
    pwcTicker,         # Ticker do ativo
    pwcBolsa           # Bolsa
)
```

## 4. Callbacks Importantes para Trading

### 4.1 OrderChangeCallback
Monitora mudanças no status das ordens:
- Status da ordem (nova, executada, cancelada, etc.)
- Quantidade executada vs pendente
- Preço médio de execução
- Mensagens de erro

```python
def order_change_callback(
    rAssetID,          # Ativo
    nCorretora,        # ID da corretora
    nQtd,              # Quantidade total
    nTradedQtd,        # Quantidade executada
    nLeavesQtd,        # Quantidade pendente
    nSide,             # Lado (1=Compra, 2=Venda)
    dPrice,            # Preço da ordem
    dStopPrice,        # Preço stop (se aplicável)
    dAvgPrice,         # Preço médio executado
    nProfitID,         # ID interno da sessão
    TipoOrdem,         # Tipo da ordem
    Conta,             # ID da conta
    Titular,           # Nome do titular
    ClOrdID,           # ID único da ordem
    Status,            # Status atual
    Date,              # Data da ordem
    TextMessage        # Mensagem adicional
):
    # Processar mudança de status da ordem
    pass
```

### 4.2 HistoryCallback
Recebe histórico de ordens do dia

### 4.3 AccountCallback
Fornece informações das contas disponíveis

```python
def account_callback(
    nCorretora,              # ID da corretora
    CorretoraNomeCompleto,   # Nome completo da corretora
    AccountID,               # ID da conta
    NomeTitular              # Nome do titular da conta
):
    # Armazenar informações da conta
    pass
```

## 5. Exemplo de Fluxo Completo

```python
import ctypes
from ctypes import c_int, c_double, c_wchar_p, WINFUNCTYPE

# 1. Definir callbacks
@WINFUNCTYPE(None, c_int, c_int)
def state_callback(conn_type, result):
    if conn_type == 0 and result == 0:  # LOGIN_CONNECTED
        print("Login conectado")
    elif conn_type == 1 and result == 2:  # ROTEAMENTO_CONNECTED
        print("Roteamento conectado")
    elif conn_type == 2 and result == 4:  # MARKET_CONNECTED
        print("Market data conectado")

# 2. Carregar DLL
dll = ctypes.windll.LoadLibrary("ProfitDLL64.dll")

# 3. Inicializar com routing
dll.DLLInitializeLogin(
    "ACTIVATION_KEY",
    "USERNAME",
    "PASSWORD",
    state_callback,
    # ... outros callbacks
)

# 4. Aguardar conexão completa
# (verificar via state_callback)

# 5. Obter informações da conta
dll.GetAccount()  # Dados recebidos via AccountCallback

# 6. Enviar ordem de compra
order_id = dll.SendBuyOrder(
    "12345",      # Account ID
    "308",        # Broker ID
    "senha123",   # Senha de roteamento
    "WDOQ25",     # Ticker
    "F",          # Bolsa (Futuros)
    5000.0,       # Preço
    1             # Quantidade
)

# 7. Monitorar ordem via OrderChangeCallback
# Status possíveis: "New", "PartiallyFilled", "Filled", "Canceled", etc.

# 8. Finalizar quando terminar
dll.DLLFinalize()
```

## 6. Configurações Adicionais

### 6.1 Day Trade
```python
# Ativar modo day trade (ordens são enviadas com tag DayTrade)
SetDayTrade(1)  # 1 = ativar, 0 = desativar
```

### 6.2 Debug
```python
# Ativar logs de debug
SetEnabledLogToDebug(1)  # 1 = salvar logs, 0 = não salvar
```

### 6.3 Histórico de Ordens
```python
# Desabilitar carregamento automático de histórico na inicialização
SetEnabledHistOrder(0)  # 0 = desabilitar, 1 = habilitar
```

## 7. Observações Importantes

1. **Threading**: Todos os callbacks são executados na thread ConnectorThread
2. **Não chamar funções da DLL dentro de callbacks**: Pode causar travamentos
3. **Processamento rápido nos callbacks**: Evitar operações pesadas (DB, I/O)
4. **Formato de data/hora**: "DD/MM/YYYY HH:mm:SS"
5. **Identificadores de Bolsa**:
   - "B" = Bovespa (ações)
   - "F" = BMF (futuros)
   - "D" = Câmbio
   - "M" = CME
   - "N" = Nasdaq
   - "Y" = NYSE
6. **Lados da Ordem**:
   - 1 = Compra (Buy)
   - 2 = Venda (Sell)
7. **Retorno das funções de ordem**: Retornam um ID interno (ProfitID) válido apenas durante a sessão

## 8. Tratamento de Erros

```python
# Códigos de erro principais
NL_OK = 0x00000000                  # Sucesso
NL_INTERNAL_ERROR = 0x80000001      # Erro interno
NL_NOT_INITIALIZED = 0x80000002     # Não inicializado
NL_INVALID_ARGS = 0x80000003        # Argumentos inválidos
NL_WAITING_SERVER = 0x80000004      # Aguardando servidor

# Verificar retorno das funções
result = dll.SendBuyOrder(...)
if result <= 0:
    print("Erro ao enviar ordem")
else:
    print(f"Ordem enviada com ID: {result}")
```

## 9. Estrutura de Posição

Ao chamar `GetPosition()`, a estrutura retornada contém:
- Quantidade intraday
- Preço médio intraday
- Vendas e compras do dia
- Quantidades em custódia (D+1, D+2, D+3)
- Quantidade disponível
- Lado da posição (1=Comprado, 2=Vendido, 0=Zerado)