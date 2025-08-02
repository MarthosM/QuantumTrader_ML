# Guia Prático ProfitDLL - Envio e Gerenciamento de Ordens

## 1. Inicialização do Sistema

### 1.1 Inicialização com Routing (Obrigatório para Ordens)

```python
# Python
def initialize_trading_system(activation_key, user, password):
    """Inicializa o sistema com capacidade de routing"""
    
    # Definir callbacks obrigatórios
    state_callback = create_state_callback()
    history_callback = create_history_callback()
    order_change_callback = create_order_change_callback()
    account_callback = create_account_callback()
    # ... outros callbacks necessários
    
    result = dll.DLLInitializeLogin(
        activation_key,
        user,
        password,
        state_callback,
        history_callback,
        order_change_callback,
        account_callback,
        # ... outros callbacks
    )
    
    if result == NL_OK:
        print("Sistema inicializado com sucesso")
        # Aguardar conexões
        wait_for_connections()
        # Solicitar informações de contas
        dll.GetAccount()
    
    return result
```

### 1.2 Verificação de Status de Conexão

```python
def create_state_callback():
    @WINFUNCTYPE(None, c_int, c_int)
    def state_callback(conn_state_type, result):
        states = {
            0: "LOGIN",
            1: "ROTEAMENTO", 
            2: "MARKET_DATA",
            3: "MARKET_LOGIN"
        }
        
        if conn_state_type == 1:  # ROTEAMENTO
            if result == 2:  # CONNECTED
                print("✓ Routing conectado - Pronto para enviar ordens")
            elif result == 0:  # DISCONNECTED
                print("✗ Routing desconectado - Ordens bloqueadas")
                
    return state_callback
```

## 2. Estruturas de Dados para Ordens

### 2.1 Identificadores

```python
# Estrutura para identificar conta
class TConnectorAccountIdentifier:
    Version = 0  # Byte
    BrokerID = 0  # Integer
    AccountID = ""  # String
    SubAccountID = ""  # String (opcional)
    Reserved = 0  # Int64

# Estrutura para identificar ativo
class TConnectorAssetIdentifier:
    Version = 0  # Byte
    Ticker = ""  # String (ex: "WDOQ25")
    Exchange = ""  # String (ex: "F" para futuros)
    FeedType = 0  # Byte

# Estrutura para identificar ordem
class TConnectorOrderIdentifier:
    Version = 0  # Byte
    LocalOrderID = 0  # Int64 (ID da sessão)
    ClOrderID = ""  # String (ID permanente)
```

### 2.2 Tipos e Status de Ordens

```python
# Tipos de Ordem
class TConnectorOrderType:
    MARKET = 1
    LIMIT = 2
    STOP_LIMIT = 4

# Lado da Ordem
class TConnectorOrderSide:
    BUY = 1
    SELL = 2

# Status da Ordem
class TConnectorOrderStatus:
    NEW = 0
    PARTIALLY_FILLED = 1
    FILLED = 2
    DONE_FOR_DAY = 3
    CANCELED = 4
    REPLACED = 5
    PENDING_CANCEL = 6
    STOPPED = 7
    REJECTED = 8
    SUSPENDED = 9
    PENDING_NEW = 10
    # ... outros status
```

## 3. Envio de Ordens

### 3.1 Estrutura Principal para Envio

```python
class TConnectorSendOrder:
    """Estrutura unificada para todos os tipos de ordens"""
    Version = 0  # ou 1 para versão mais recente
    AccountID = TConnectorAccountIdentifier()
    AssetID = TConnectorAssetIdentifier()
    Password = ""  # Senha de roteamento
    OrderType = 0  # TConnectorOrderType
    OrderSide = 0  # TConnectorOrderSide
    Price = 0.0  # -1 para ordens a mercado
    StopPrice = 0.0  # -1 para ordens não-stop
    Quantity = 0  # Int64
```

### 3.2 Exemplos de Envio de Ordens

#### Ordem Limite de Compra
```python
def send_limit_buy_order(account, ticker, price, quantity, password):
    """Envia ordem limite de compra"""
    
    order = TConnectorSendOrder()
    order.Version = 1
    
    # Configurar conta
    order.AccountID.BrokerID = account['broker_id']
    order.AccountID.AccountID = account['account_id']
    order.AccountID.SubAccountID = account.get('sub_account_id', '')
    
    # Configurar ativo
    order.AssetID.Ticker = ticker  # Ex: "WDOQ25"
    order.AssetID.Exchange = "F"   # F para futuros
    order.AssetID.FeedType = 0
    
    # Configurar ordem
    order.Password = password
    order.OrderType = TConnectorOrderType.LIMIT
    order.OrderSide = TConnectorOrderSide.BUY
    order.Price = price
    order.StopPrice = -1  # Não é stop
    order.Quantity = quantity
    
    # Enviar
    local_order_id = dll.SendOrder(byref(order))
    
    if local_order_id > 0:
        print(f"Ordem enviada com sucesso. ID: {local_order_id}")
    else:
        print(f"Erro ao enviar ordem: {local_order_id}")
        
    return local_order_id
```

#### Ordem a Mercado de Venda
```python
def send_market_sell_order(account, ticker, quantity, password):
    """Envia ordem a mercado de venda"""
    
    order = TConnectorSendOrder()
    order.Version = 1
    
    # Configurar conta e ativo (similar ao exemplo anterior)
    # ...
    
    # Configurar ordem a mercado
    order.OrderType = TConnectorOrderType.MARKET
    order.OrderSide = TConnectorOrderSide.SELL
    order.Price = -1  # Mercado
    order.StopPrice = -1
    order.Quantity = quantity
    
    return dll.SendOrder(byref(order))
```

#### Ordem Stop Limit
```python
def send_stop_limit_order(account, ticker, price, stop_price, quantity, side, password):
    """Envia ordem stop limit"""
    
    order = TConnectorSendOrder()
    order.Version = 1
    
    # Configurar conta e ativo
    # ...
    
    # Configurar stop limit
    order.OrderType = TConnectorOrderType.STOP_LIMIT
    order.OrderSide = side  # BUY ou SELL
    order.Price = price  # Preço limite após trigger
    order.StopPrice = stop_price  # Preço de trigger
    order.Quantity = quantity
    
    return dll.SendOrder(byref(order))
```

## 4. Modificação de Ordens

### 4.1 Estrutura para Modificação

```python
class TConnectorChangeOrder:
    Version = 0
    AccountID = TConnectorAccountIdentifier()
    OrderID = TConnectorOrderIdentifier()
    Password = ""
    Price = 0.0  # Novo preço
    StopPrice = 0.0  # Novo stop (se aplicável)
    Quantity = 0  # Nova quantidade
```

### 4.2 Exemplo de Modificação

```python
def modify_order(account, cl_order_id, new_price, new_quantity, password):
    """Modifica uma ordem existente"""
    
    change_order = TConnectorChangeOrder()
    change_order.Version = 0
    
    # Configurar conta
    change_order.AccountID = account
    
    # Identificar ordem
    change_order.OrderID.ClOrderID = cl_order_id
    # Ou usar LocalOrderID se disponível
    
    # Novos valores
    change_order.Price = new_price
    change_order.StopPrice = -1  # Manter inalterado
    change_order.Quantity = new_quantity
    change_order.Password = password
    
    result = dll.SendChangeOrderV2(byref(change_order))
    
    if result == NL_OK:
        print("Ordem modificada com sucesso")
    else:
        print(f"Erro ao modificar ordem: {result}")
        
    return result
```

## 5. Cancelamento de Ordens

### 5.1 Cancelar Ordem Específica

```python
def cancel_order(account, cl_order_id, password):
    """Cancela uma ordem específica"""
    
    cancel_order = TConnectorCancelOrder()
    cancel_order.Version = 0
    cancel_order.AccountID = account
    cancel_order.OrderID.ClOrderID = cl_order_id
    cancel_order.Password = password
    
    result = dll.SendCancelOrderV2(byref(cancel_order))
    
    if result == NL_OK:
        print(f"Ordem {cl_order_id} cancelada")
    
    return result
```

### 5.2 Cancelar Todas as Ordens de um Ativo

```python
def cancel_all_orders_for_asset(account, ticker, exchange, password):
    """Cancela todas as ordens de um ativo"""
    
    cancel_orders = TConnectorCancelOrders()
    cancel_orders.Version = 0
    cancel_orders.AccountID = account
    cancel_orders.AssetID.Ticker = ticker
    cancel_orders.AssetID.Exchange = exchange
    cancel_orders.Password = password
    
    return dll.SendCancelOrdersV2(byref(cancel_orders))
```

### 5.3 Cancelar Todas as Ordens

```python
def cancel_all_orders(account, password):
    """Cancela todas as ordens abertas"""
    
    cancel_all = TConnectorCancelAllOrders()
    cancel_all.Version = 0
    cancel_all.AccountID = account
    cancel_all.Password = password
    
    return dll.SendCancelAllOrdersV2(byref(cancel_all))
```

## 6. Zerar Posição

```python
def zero_position(account, ticker, exchange, price, password, position_type=None):
    """Zera posição de um ativo"""
    
    zero_pos = TConnectorZeroPosition()
    zero_pos.Version = 1 if position_type else 0
    zero_pos.AccountID = account
    zero_pos.AssetID.Ticker = ticker
    zero_pos.AssetID.Exchange = exchange
    zero_pos.Password = password
    zero_pos.Price = price  # -1 para mercado
    
    if position_type:
        zero_pos.PositionType = position_type  # DAY_TRADE=1, CONSOLIDATED=2
    
    local_order_id = dll.SendZeroPositionV2(byref(zero_pos))
    
    if local_order_id > 0:
        print(f"Ordem de zeragem enviada. ID: {local_order_id}")
    
    return local_order_id
```

## 7. Callbacks para Acompanhamento de Ordens

### 7.1 Callback de Mudança de Status

```python
def create_order_callback():
    """Cria callback para mudanças em ordens"""
    
    @WINFUNCTYPE(None, POINTER(TConnectorOrderIdentifier))
    def order_callback(order_id_ptr):
        # Obter detalhes da ordem
        order_details = TConnectorOrderOut()
        order_details.OrderID = order_id_ptr.contents
        
        result = dll.GetOrderDetails(byref(order_details))
        
        if result == NL_OK:
            print(f"Ordem atualizada:")
            print(f"  ClOrderID: {order_details.OrderID.ClOrderID}")
            print(f"  Status: {order_details.OrderStatus}")
            print(f"  Qtd Total: {order_details.Quantity}")
            print(f"  Qtd Executada: {order_details.TradedQuantity}")
            print(f"  Qtd Restante: {order_details.LeavesQuantity}")
            print(f"  Preço Médio: {order_details.AveragePrice}")
            
            # Processar conforme status
            if order_details.OrderStatus == TConnectorOrderStatus.FILLED:
                print("✓ Ordem totalmente executada!")
            elif order_details.OrderStatus == TConnectorOrderStatus.CANCELED:
                print("✗ Ordem cancelada")
            elif order_details.OrderStatus == TConnectorOrderStatus.REJECTED:
                print("✗ Ordem rejeitada")
                print(f"  Motivo: {order_details.TextMessage}")
    
    return order_callback
```

### 7.2 Histórico de Ordens

```python
def get_order_history(account, start_date, end_date):
    """Obtém histórico de ordens"""
    
    # Verificar se há ordens no período
    has_orders = dll.HasOrdersInInterval(
        byref(account),
        start_date,
        end_date
    )
    
    if has_orders == NL_OK:
        # Enumerar ordens
        orders = []
        
        @WINFUNCTYPE(BOOL, POINTER(TConnectorOrder), LPARAM)
        def enum_callback(order_ptr, param):
            order = order_ptr.contents
            orders.append({
                'cl_order_id': order.OrderID.ClOrderID,
                'ticker': order.AssetID.Ticker,
                'side': order.OrderSide,
                'type': order.OrderType,
                'status': order.OrderStatus,
                'quantity': order.Quantity,
                'price': order.Price,
                'avg_price': order.AveragePrice,
                'traded_qty': order.TradedQuantity
            })
            return True  # Continuar enumeração
        
        dll.EnumerateOrdersByInterval(
            byref(account),
            1,  # Versão da estrutura
            start_date,
            end_date,
            0,  # Param
            enum_callback
        )
        
        return orders
    
    elif has_orders == NL_WAITING_SERVER:
        print("Aguardando histórico do servidor...")
        return None
```

## 8. Boas Práticas e Considerações

### 8.1 Gestão de Conexão

```python
class OrderManager:
    """Gerenciador de ordens com verificações de segurança"""
    
    def __init__(self):
        self.connected = False
        self.routing_connected = False
        self.accounts = {}
        
    def is_ready_for_trading(self):
        """Verifica se sistema está pronto para trading"""
        return self.connected and self.routing_connected and len(self.accounts) > 0
    
    def send_order_safe(self, order_params):
        """Envia ordem com verificações"""
        
        if not self.is_ready_for_trading():
            print("Sistema não está pronto para trading")
            return -1
            
        # Validar parâmetros
        if order_params['quantity'] <= 0:
            print("Quantidade inválida")
            return -1
            
        if order_params['order_type'] == TConnectorOrderType.LIMIT:
            if order_params['price'] <= 0:
                print("Preço inválido para ordem limite")
                return -1
        
        # Enviar ordem
        return self.send_order(order_params)
```

### 8.2 Tratamento de Erros

```python
# Códigos de erro importantes
ERROR_CODES = {
    0: "Sucesso",
    -2147483647: "Erro interno",
    -2147483646: "Não inicializado",
    -2147483645: "Argumentos inválidos",
    -2147483644: "Aguardando servidor",
    -2147483643: "Sem login",
    -2147483620: "Senha não fornecida"
}

def handle_order_error(error_code):
    """Trata erros de ordens"""
    
    if error_code in ERROR_CODES:
        print(f"Erro: {ERROR_CODES[error_code]}")
    else:
        print(f"Erro desconhecido: {error_code}")
    
    # Ações específicas por erro
    if error_code == -2147483643:  # Sem login
        # Tentar reconectar
        reconnect()
```

### 8.3 Monitoramento de Posição

```python
def monitor_position(account, ticker, exchange):
    """Monitora posição em tempo real"""
    
    position = TConnectorTradingAccountPosition()
    position.Version = 2
    position.AccountID = account
    position.AssetID.Ticker = ticker
    position.AssetID.Exchange = exchange
    position.PositionType = 1  # Day trade
    
    result = dll.GetPositionV2(byref(position))
    
    if result == NL_OK:
        print(f"Posição {ticker}:")
        print(f"  Quantidade: {position.OpenQuantity}")
        print(f"  Preço Médio: {position.OpenAveragePrice}")
        print(f"  Lado: {'Compra' if position.OpenSide == 1 else 'Venda'}")
        print(f"  Qtd Comprada Hoje: {position.DailyBuyQuantity}")
        print(f"  Qtd Vendida Hoje: {position.DailySellQuantity}")
        print(f"  Qtd Disponível: {position.DailyQuantityAvailable}")
        
        return position
    
    return None
```

## 9. Exemplo Completo de Trading

```python
class TradingBot:
    """Bot de trading completo"""
    
    def __init__(self, config):
        self.config = config
        self.dll = None
        self.account = None
        self.active_orders = {}
        
    def initialize(self):
        """Inicializa sistema de trading"""
        # Configurar DLL e callbacks
        self.setup_dll()
        self.setup_callbacks()
        
        # Inicializar
        result = self.dll.DLLInitializeLogin(
            self.config['activation_key'],
            self.config['user'],
            self.config['password'],
            # ... callbacks
        )
        
        if result != NL_OK:
            raise Exception(f"Falha na inicialização: {result}")
            
        # Aguardar conexões
        self.wait_for_connections()
        
        # Obter conta
        self.dll.GetAccount()
        self.wait_for_account()
        
    def place_bracket_order(self, ticker, side, quantity, entry_price, stop_price, target_price):
        """Coloca ordem com stop e alvo"""
        
        # 1. Ordem de entrada
        entry_order = self.send_limit_order(
            ticker, side, entry_price, quantity
        )
        
        if entry_order <= 0:
            return False
            
        # Aguardar execução
        self.wait_for_fill(entry_order)
        
        # 2. Ordem de stop loss
        stop_side = TConnectorOrderSide.SELL if side == TConnectorOrderSide.BUY else TConnectorOrderSide.BUY
        stop_order = self.send_stop_order(
            ticker, stop_side, stop_price, quantity
        )
        
        # 3. Ordem de take profit
        target_order = self.send_limit_order(
            ticker, stop_side, target_price, quantity
        )
        
        # Monitorar ordens
        self.monitor_bracket_orders(stop_order, target_order)
        
        return True
    
    def monitor_bracket_orders(self, stop_id, target_id):
        """Monitora ordens bracket e cancela a que sobrar"""
        
        while True:
            stop_status = self.get_order_status(stop_id)
            target_status = self.get_order_status(target_id)
            
            # Se uma foi executada, cancelar a outra
            if stop_status == TConnectorOrderStatus.FILLED:
                self.cancel_order(target_id)
                break
            elif target_status == TConnectorOrderStatus.FILLED:
                self.cancel_order(stop_id)
                break
                
            time.sleep(0.1)
```

## 10. Checklist de Implementação

### Antes de Enviar Ordens:
- [ ] DLL inicializado com DLLInitializeLogin (não DLLInitializeMarketLogin)
- [ ] Conexão de routing estabelecida (ROTEAMENTO_CONNECTED)
- [ ] Conta carregada via GetAccount()
- [ ] Senha de roteamento disponível
- [ ] Callbacks configurados para receber atualizações

### Para Cada Ordem:
- [ ] Validar parâmetros (quantidade > 0, preço válido)
- [ ] Verificar margem/saldo disponível
- [ ] Confirmar ticker e exchange corretos
- [ ] Usar estrutura correta (TConnectorSendOrder)
- [ ] Tratar retorno (ID > 0 = sucesso)
- [ ] Monitorar status via callbacks

### Melhores Práticas:
- [ ] Implementar reconexão automática
- [ ] Log de todas as ordens enviadas
- [ ] Controle de risco (max loss, max orders)
- [ ] Validação dupla antes de enviar
- [ ] Timeout para ordens não executadas
- [ ] Sistema de fallback em caso de erro

## Observações Importantes

1. **Thread Safety**: Callbacks são executados em thread separada (ConnectorThread)
2. **Não chamar funções DLL dentro de callbacks**: Pode causar deadlock
3. **IDs de Ordem**: LocalOrderID é válido apenas na sessão, ClOrderID é permanente
4. **Horário**: Verificar se mercado está aberto antes de enviar ordens
5. **Latência**: Processar callbacks rapidamente para não atrasar fila de mensagens