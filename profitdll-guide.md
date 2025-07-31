# Guia Completo ProfitDLL - Conexão e Extração de Dados

## Índice
1. [Introdução](#introdução)
2. [Inicialização e Conexão](#inicialização-e-conexão)
3. [Callbacks Essenciais](#callbacks-essenciais-para-banco-de-dados)
4. [Comandos de Subscrição](#comandos-para-subscrição-de-dados)
5. [Dados Históricos](#comandos-para-dados-históricos)
6. [Estrutura do Banco de Dados](#estrutura-de-dados-para-banco)
7. [Implementação Completa](#fluxo-completo-de-implementação)
8. [Referências Rápidas](#referências-rápidas)
9. [Troubleshooting](#troubleshooting)

## Introdução

A ProfitDLL é uma biblioteca que permite conexão com servidores de Market Data e Routing para desenvolvimento de aplicações de trading. Este guia foca na extração de dados para construção de banco de dados.

### Versões Disponíveis
- **32 bits**: ProfitDLL.dll (limitado a 4GB de memória)
- **64 bits**: ProfitDLL64.dll (recomendado para grandes volumes)

### Convenção de Chamada
Todas as funções usam `stdcall` tanto em 32 quanto 64 bits.

## Inicialização e Conexão

### Inicialização Completa (Market Data + Routing)

```python
# Função principal para inicialização completa
result = DLLInitializeLogin(
    pwcActivationKey,      # PWideChar - Chave de ativação fornecida
    pwcUser,               # PWideChar - Usuário da conta
    pwcPassword,           # PWideChar - Senha da conta
    StateCallback,         # TStateCallback - Callback de estado da conexão
    HistoryCallback,       # THistoryCallback - Callback histórico de ordens
    OrderChangeCallback,   # TOrderChangeCallback - Callback mudanças de ordem
    AccountCallback,       # TAccountCallback - Callback informações da conta
    NewTradeCallback,      # TNewTradeCallback - Callback trades em tempo real
    NewDailyCallback,      # TNewDailyCallback - Callback dados agregados diários
    PriceBookCallback,     # TPriceBookCallback - Callback book de preços
    OfferBookCallback,     # TOfferBookCallback - Callback book de ofertas
    HistoryTradeCallback,  # THistoryTradeCallback - Callback histórico de trades
    ProgressCallback,      # TProgressCallback - Callback progresso download
    TinyBookCallback       # TTinyBookCallback - Callback topo do book
)

# Retorno:
# NL_OK (0x00000000) = Sucesso
# NL_INTERNAL_ERROR (0x80000001) = Erro interno
# NL_INVALID_ARGS (0x80000003) = Argumentos inválidos
```

### Inicialização Apenas Market Data

```python
# Para apenas receber dados de mercado (sem routing)
result = DLLInitializeMarketLogin(
    pwcActivationKey,      # PWideChar - Chave de ativação
    pwcUser,               # PWideChar - Usuário
    pwcPassword,           # PWideChar - Senha
    StateCallback,         # TStateCallback - Estado da conexão
    NewTradeCallback,      # TNewTradeCallback - Trades tempo real
    NewDailyCallback,      # TNewDailyCallback - Dados diários
    PriceBookCallback,     # TPriceBookCallback - Book de preços
    OfferBookCallback,     # TOfferBookCallback - Book de ofertas
    HistoryTradeCallback,  # THistoryTradeCallback - Histórico
    ProgressCallback,      # TProgressCallback - Progresso
    TinyBookCallback       # TTinyBookCallback - Topo do book
)
```

### Finalização

```python
# Sempre finalizar ao encerrar a aplicação
DLLFinalize()
```

## Callbacks Essenciais para Banco de Dados

### TStateCallback - Monitorar Estado da Conexão

```python
# Assinatura: procedure(nConnStateType: Integer; nResult: Integer) stdcall;

def state_callback(nConnStateType, nResult):
    """
    Monitora o estado das conexões
    
    nConnStateType:
    - 0 = CONNECTION_STATE_LOGIN
    - 1 = CONNECTION_STATE_ROTEAMENTO  
    - 2 = CONNECTION_STATE_MARKET_DATA
    - 3 = CONNECTION_STATE_MARKET_LOGIN
    
    Estados de sucesso:
    - LOGIN: nResult = 0 (LOGIN_CONNECTED)
    - ROTEAMENTO: nResult = 2 (ROTEAMENTO_CONNECTED)
    - MARKET_DATA: nResult = 4 (MARKET_CONNECTED)
    - MARKET_LOGIN: nResult = 0 (CONNECTION_ACTIVATE_VALID)
    """
    
    if nConnStateType == 2 and nResult == 4:
        print("Market Data conectado com sucesso!")
    elif nConnStateType == 0 and nResult == 0:
        print("Login realizado com sucesso!")
```

### TNewTradeCallback - Trades em Tempo Real

```python
# Assinatura completa do callback
def new_trade_callback(
    rAssetID,          # TAssetIDRec - Identificação do ativo
    pwcDate,           # PWideChar - Data/hora formato: DD/MM/YYYY HH:mm:SS.ZZZ
    nTradeNumber,      # Cardinal - Número sequencial único do trade
    dPrice,            # Double - Preço de execução
    dVol,              # Double - Volume financeiro
    nQtd,              # Integer - Quantidade negociada
    nBuyAgent,         # Integer - ID do agente comprador
    nSellAgent,        # Integer - ID do agente vendedor
    nTradeType,        # Integer - Tipo do trade (ver tabela abaixo)
    bEdit              # Char - 'T' se é edição, 'F' se é novo
):
    """
    Processa cada trade executado no mercado
    
    Tipos de Trade (nTradeType):
    1 = Cross trade
    2 = Buy agressivo
    3 = Sell agressivo
    4 = Leilão
    5 = Vigilância
    6 = Ex-pit
    7 = Exercício de opções
    8 = Mercado de balcão
    9 = Termo
    10 = Índice
    11 = BTC
    12 = On Behalf
    13 = RLP
    32 = Desconhecido
    """
    
    # Extrair símbolo e bolsa
    symbol = rAssetID.contents.pwcTicker
    exchange = rAssetID.contents.pwcBolsa
    
    # Converter timestamp
    timestamp = datetime.strptime(pwcDate, '%d/%m/%Y %H:%M:%S.%f')
    
    # Armazenar no banco
    save_trade_to_db(symbol, exchange, timestamp, dPrice, nQtd, dVol, nTradeType)
```

### THistoryTradeCallback - Dados Históricos

```python
def history_trade_callback(
    rAssetID,          # TAssetIDRec - Ativo
    pwcDate,           # PWideChar - Data/hora
    nTradeNumber,      # Cardinal - Número do trade
    dPrice,            # Double - Preço
    dVol,              # Double - Volume financeiro
    nQtd,              # Integer - Quantidade
    nBuyAgent,         # Integer - Agente comprador
    nSellAgent,        # Integer - Agente vendedor
    nTradeType         # Integer - Tipo do trade
):
    """
    Recebe trades históricos solicitados via GetHistoryTrades
    Formato idêntico ao NewTradeCallback mas para dados passados
    """
    # Processar igual ao tempo real mas em batch
    pass
```

### TProgressCallback - Progresso de Download

```python
def progress_callback(rAssetID, nProgress):
    """
    Monitora progresso de download de dados históricos
    
    nProgress:
    - 0-100: Percentual do download
    - 1000: Download completo, todos os trades enviados
    """
    symbol = rAssetID.contents.pwcTicker
    
    if nProgress == 1000:
        print(f"Download completo para {symbol}")
    else:
        print(f"Progresso {symbol}: {nProgress}%")
```

### TPriceBookCallback - Book Agregado por Preço

```python
def price_book_callback(
    rAssetID,          # TAssetIDRec - Ativo
    nAction,           # Integer - Ação (0=Add, 1=Edit, 2=Delete, 3=DeleteFrom, 4=FullBook)
    nPosition,         # Integer - Posição no book
    nSide,             # Integer - Lado (0=Buy, 1=Sell)
    nQtds,             # Integer - Quantidade total
    nCount,            # Integer - Número de ofertas
    dPrice,            # Double - Preço
    pArraySell,        # Pointer - Array completo lado venda
    pArrayBuy          # Pointer - Array completo lado compra
):
    """
    Atualiza book agregado por nível de preço
    
    Ações (nAction):
    0 = Adicionar nova entrada
    1 = Editar entrada existente
    2 = Deletar entrada
    3 = Deletar a partir da posição
    4 = Book completo (snapshot)
    """
    pass
```

## Comandos para Subscrição de Dados

### Subscrever Trades em Tempo Real

```python
# Começar a receber trades
result = SubscribeTicker(
    pwcTicker="WDOQ25",   # Símbolo do ativo
    pwcBolsa="F"          # Bolsa (ver tabela de códigos)
)

# Parar de receber
result = UnsubscribeTicker(
    pwcTicker="WDOQ25",
    pwcBolsa="F"
)
```

### Subscrever Book de Ofertas

```python
# Book completo com todas as ofertas individuais
result = SubscribeOfferBook(pwcTicker="WDOQ25", pwcBolsa="F")
result = UnsubscribeOfferBook(pwcTicker="WDOQ25", pwcBolsa="F")

# Book agregado por nível de preço
result = SubscribePriceBook(pwcTicker="WDOQ25", pwcBolsa="F")
result = UnsubscribePriceBook(pwcTicker="WDOQ25", pwcBolsa="F")
```

### Subscrever Ajustes

```python
# Receber histórico de ajustes (dividendos, splits, etc)
result = SubscribeAdjustHistory(pwcTicker="PETR4", pwcBolsa="B")
result = UnsubscribeAdjustHistory(pwcTicker="PETR4", pwcBolsa="B")
```

## Comandos para Dados Históricos

### Solicitar Histórico de Trades

```python
# Solicitar dados históricos de trades
result = GetHistoryTrades(
    pwcTicker="WDOQ25",                    # Símbolo
    pwcBolsa="F",                          # Bolsa
    dtDateStart="20/12/2024 09:00:00",     # Data inicial DD/MM/YYYY HH:mm:SS
    dtDateEnd="20/12/2024 18:00:00"        # Data final DD/MM/YYYY HH:mm:SS
)

# Os dados chegam via:
# 1. THistoryTradeCallback - cada trade
# 2. TProgressCallback - progresso (0-100, 1000=completo)
```

### Obter Fechamento Anterior

```python
# Útil para calcular variações
dClose = ctypes.c_double()
result = GetLastDailyClose(
    pwcTicker="WDOQ25",
    pwcBolsa="F",
    dClose=ctypes.byref(dClose),  # Valor retornado aqui
    bAdjusted=0                    # 0=não ajustado, 1=ajustado
)

if result == NL_OK:
    print(f"Fechamento anterior: {dClose.value}")
```

### Informações do Ativo

```python
# Solicitar informações detalhadas
result = RequestTickerInfo("WDOQ25", "F")
# Dados chegam via TAssetListInfoCallback
```

## Estrutura de Dados para Banco

### Schema MySQL/PostgreSQL

```sql
-- Tabela principal de trades
CREATE TABLE trades (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    symbol VARCHAR(20) NOT NULL,
    exchange CHAR(1) NOT NULL,
    trade_number BIGINT NOT NULL,
    timestamp DATETIME(3) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    volume DECIMAL(15,2) NOT NULL,
    quantity INT NOT NULL,
    buy_agent INT,
    sell_agent INT,
    trade_type INT NOT NULL,
    is_edit BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_symbol_timestamp (symbol, timestamp),
    INDEX idx_trade_number (trade_number),
    INDEX idx_timestamp (timestamp),
    INDEX idx_trade_type (trade_type)
) ENGINE=InnoDB;

-- Tabela de candles OHLCV
CREATE TABLE candles (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    symbol VARCHAR(20) NOT NULL,
    exchange CHAR(1) NOT NULL,
    timeframe ENUM('1m', '5m', '15m', '30m', '1h', '1d') NOT NULL,
    timestamp DATETIME NOT NULL,
    open DECIMAL(10,2) NOT NULL,
    high DECIMAL(10,2) NOT NULL,
    low DECIMAL(10,2) NOT NULL,
    close DECIMAL(10,2) NOT NULL,
    volume DECIMAL(15,2) NOT NULL,
    trades INT NOT NULL,
    buy_volume DECIMAL(15,2),
    sell_volume DECIMAL(15,2),
    vwap DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    UNIQUE KEY unique_candle (symbol, timeframe, timestamp),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB;

-- Tabela de book (snapshot)
CREATE TABLE book_snapshots (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    symbol VARCHAR(20) NOT NULL,
    exchange CHAR(1) NOT NULL,
    timestamp DATETIME(3) NOT NULL,
    side ENUM('BUY', 'SELL') NOT NULL,
    position INT NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    quantity INT NOT NULL,
    orders_count INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_symbol_timestamp (symbol, timestamp),
    INDEX idx_timestamp_side (timestamp, side)
) ENGINE=InnoDB;

-- Tabela de informações dos ativos
CREATE TABLE asset_info (
    id INT PRIMARY KEY AUTO_INCREMENT,
    symbol VARCHAR(20) NOT NULL,
    exchange CHAR(1) NOT NULL,
    name VARCHAR(100),
    description VARCHAR(255),
    min_order_qty INT,
    max_order_qty INT,
    lot_size INT,
    security_type INT,
    security_subtype INT,
    min_price_increment DECIMAL(10,6),
    contract_multiplier DECIMAL(10,4),
    expiration_date DATE,
    isin VARCHAR(20),
    sector VARCHAR(50),
    subsector VARCHAR(50),
    segment VARCHAR(50),
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    UNIQUE KEY unique_asset (symbol, exchange)
) ENGINE=InnoDB;
```

## Fluxo Completo de Implementação

### Exemplo Python Completo

```python
import ctypes
from ctypes import wintypes
from datetime import datetime
import mysql.connector
import pandas as pd
from collections import deque
import threading
import time

# Configuração da DLL
dll = ctypes.CDLL('ProfitDLL64.dll')  # ou ProfitDLL.dll para 32 bits

# Estrutura AssetIDRec
class AssetIDRec(ctypes.Structure):
    _fields_ = [
        ("pwcTicker", ctypes.c_wchar_p),
        ("pwcBolsa", ctypes.c_wchar_p),
        ("nFeed", ctypes.c_int)
    ]

# Definir assinaturas das funções
dll.DLLInitializeMarketLogin.argtypes = [
    ctypes.c_wchar_p,  # pwcActivationKey
    ctypes.c_wchar_p,  # pwcUser
    ctypes.c_wchar_p,  # pwcPassword
    ctypes.c_void_p,   # StateCallback
    ctypes.c_void_p,   # NewTradeCallback
    ctypes.c_void_p,   # NewDailyCallback
    ctypes.c_void_p,   # PriceBookCallback
    ctypes.c_void_p,   # OfferBookCallback
    ctypes.c_void_p,   # HistoryTradeCallback
    ctypes.c_void_p,   # ProgressCallback
    ctypes.c_void_p    # TinyBookCallback
]
dll.DLLInitializeMarketLogin.restype = ctypes.c_int

# Tipos de callback
STATE_CALLBACK = ctypes.WINFUNCTYPE(None, ctypes.c_int, ctypes.c_int)
TRADE_CALLBACK = ctypes.WINFUNCTYPE(
    None,
    ctypes.POINTER(AssetIDRec),
    ctypes.c_wchar_p,
    ctypes.c_uint32,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_char
)
PROGRESS_CALLBACK = ctypes.WINFUNCTYPE(
    None,
    ctypes.POINTER(AssetIDRec),
    ctypes.c_int
)

class ProfitDLLClient:
    def __init__(self, db_config):
        self.dll = dll
        self.db_config = db_config
        self.trades_buffer = deque(maxlen=1000)
        self.is_connected = False
        self.buffer_lock = threading.Lock()
        
        # Callbacks
        self.state_cb = STATE_CALLBACK(self.on_state_change)
        self.trade_cb = TRADE_CALLBACK(self.on_new_trade)
        self.history_cb = TRADE_CALLBACK(self.on_history_trade)
        self.progress_cb = PROGRESS_CALLBACK(self.on_progress)
        
        # Conexão DB
        self.db_conn = None
        self.connect_db()
        
    def connect_db(self):
        """Conecta ao banco de dados"""
        self.db_conn = mysql.connector.connect(**self.db_config)
        self.cursor = self.db_conn.cursor()
        
    def initialize(self, activation_key, user, password):
        """Inicializa conexão com ProfitDLL"""
        result = self.dll.DLLInitializeMarketLogin(
            activation_key,
            user,
            password,
            self.state_cb,
            self.trade_cb,
            None,  # NewDailyCallback
            None,  # PriceBookCallback
            None,  # OfferBookCallback
            self.history_cb,
            self.progress_cb,
            None   # TinyBookCallback
        )
        
        if result != 0:
            raise Exception(f"Erro ao inicializar: {result}")
            
        # Aguardar conexão
        timeout = 30
        start = time.time()
        while not self.is_connected and (time.time() - start) < timeout:
            time.sleep(0.1)
            
        if not self.is_connected:
            raise Exception("Timeout ao conectar")
            
    @STATE_CALLBACK
    def on_state_change(self, conn_type, result):
        """Callback de mudança de estado"""
        print(f"Estado: tipo={conn_type}, resultado={result}")
        
        if conn_type == 2 and result == 4:  # Market Data Connected
            self.is_connected = True
            print("Market Data conectado!")
            
    @TRADE_CALLBACK
    def on_new_trade(self, asset_id, date_str, trade_num, price, vol, 
                     qty, buy_agent, sell_agent, trade_type, is_edit):
        """Callback de novo trade"""
        try:
            symbol = asset_id.contents.pwcTicker
            exchange = asset_id.contents.pwcBolsa
            
            # Parse timestamp
            timestamp = datetime.strptime(
                date_str.replace(',', '.'), 
                '%d/%m/%Y %H:%M:%S.%f'
            )
            
            trade = {
                'symbol': symbol,
                'exchange': exchange,
                'trade_number': trade_num,
                'timestamp': timestamp,
                'price': price,
                'volume': vol,
                'quantity': qty,
                'buy_agent': buy_agent,
                'sell_agent': sell_agent,
                'trade_type': trade_type,
                'is_edit': is_edit == ord('T')
            }
            
            with self.buffer_lock:
                self.trades_buffer.append(trade)
                
                # Salvar em batch
                if len(self.trades_buffer) >= 100:
                    self.save_trades_batch()
                    
        except Exception as e:
            print(f"Erro processando trade: {e}")
            
    @TRADE_CALLBACK
    def on_history_trade(self, asset_id, date_str, trade_num, price, vol,
                        qty, buy_agent, sell_agent, trade_type):
        """Callback de trade histórico"""
        # Reutiliza lógica do trade em tempo real
        self.on_new_trade(
            asset_id, date_str, trade_num, price, vol,
            qty, buy_agent, sell_agent, trade_type, ord('F')
        )
        
    @PROGRESS_CALLBACK
    def on_progress(self, asset_id, progress):
        """Callback de progresso"""
        symbol = asset_id.contents.pwcTicker
        
        if progress == 1000:
            print(f"{symbol}: Download completo!")
            with self.buffer_lock:
                self.save_trades_batch()  # Salvar trades restantes
        else:
            print(f"{symbol}: {progress}%")
            
    def save_trades_batch(self):
        """Salva trades em batch no banco"""
        if not self.trades_buffer:
            return
            
        trades = list(self.trades_buffer)
        self.trades_buffer.clear()
        
        query = """
        INSERT INTO trades 
        (symbol, exchange, trade_number, timestamp, price, volume, 
         quantity, buy_agent, sell_agent, trade_type, is_edit)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        price = VALUES(price),
        is_edit = VALUES(is_edit)
        """
        
        data = [
            (t['symbol'], t['exchange'], t['trade_number'],
             t['timestamp'], t['price'], t['volume'],
             t['quantity'], t['buy_agent'], t['sell_agent'],
             t['trade_type'], t['is_edit'])
            for t in trades
        ]
        
        try:
            self.cursor.executemany(query, data)
            self.db_conn.commit()
            print(f"Salvos {len(trades)} trades")
        except Exception as e:
            print(f"Erro salvando trades: {e}")
            self.db_conn.rollback()
            
    def subscribe_ticker(self, symbol, exchange='F'):
        """Subscreve para receber trades"""
        self.dll.SubscribeTicker.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        result = self.dll.SubscribeTicker(symbol, exchange)
        print(f"Subscrito em {symbol}: {result}")
        
    def get_historical_trades(self, symbol, start_date, end_date, exchange='F'):
        """Solicita trades históricos"""
        self.dll.GetHistoryTrades.argtypes = [
            ctypes.c_wchar_p, ctypes.c_wchar_p,
            ctypes.c_wchar_p, ctypes.c_wchar_p
        ]
        
        result = self.dll.GetHistoryTrades(
            symbol, exchange,
            start_date.strftime('%d/%m/%Y %H:%M:%S'),
            end_date.strftime('%d/%m/%Y %H:%M:%S')
        )
        print(f"Histórico solicitado: {result}")
        
    def run(self):
        """Loop principal"""
        try:
            while True:
                time.sleep(1)
                
                # Salvar buffer periodicamente
                with self.buffer_lock:
                    if self.trades_buffer:
                        self.save_trades_batch()
                        
        except KeyboardInterrupt:
            print("Encerrando...")
            self.cleanup()
            
    def cleanup(self):
        """Finaliza conexões"""
        # Salvar trades restantes
        with self.buffer_lock:
            self.save_trades_batch()
            
        # Finalizar DLL
        self.dll.DLLFinalize()
        
        # Fechar DB
        if self.db_conn:
            self.cursor.close()
            self.db_conn.close()

# Uso do cliente
if __name__ == "__main__":
    db_config = {
        'host': 'localhost',
        'user': 'trading_user',
        'password': 'senha_segura',
        'database': 'trading_db'
    }
    
    client = ProfitDLLClient(db_config)
    
    # Inicializar
    client.initialize(
        activation_key="SUA_CHAVE_AQUI",
        user="SEU_USUARIO",
        password="SUA_SENHA"
    )
    
    # Subscrever ativos
    client.subscribe_ticker("WDOQ25", "F")
    client.subscribe_ticker("WINQ25", "F")
    
    # Solicitar histórico
    from datetime import timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    client.get_historical_trades("WDOQ25", start_date, end_date)
    
    # Rodar cliente
    client.run()
```

### Construção de Candles a partir de Trades

```python
def build_candles_from_trades(trades_df, timeframe='1m'):
    """
    Constrói candles OHLCV a partir de trades
    
    timeframe: '1m', '5m', '15m', '30m', '1h', '1d'
    """
    # Configurar resample rule
    rule_map = {
        '1m': '1T', '5m': '5T', '15m': '15T',
        '30m': '30T', '1h': '1H', '1d': '1D'
    }
    
    rule = rule_map.get(timeframe, '1T')
    
    # Separar por tipo de trade
    buy_trades = trades_df[trades_df['trade_type'] == 2].copy()
    sell_trades = trades_df[trades_df['trade_type'] == 3].copy()
    
    # Agrupar por tempo
    ohlc = trades_df.groupby(pd.Grouper(freq=rule)).agg({
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum',
        'quantity': 'sum',
        'trade_number': 'count'
    })
    
    # Renomear colunas
    ohlc.columns = ['open', 'high', 'low', 'close', 'volume', 'quantity', 'trades']
    
    # Volume de compra/venda
    buy_vol = buy_trades.groupby(pd.Grouper(freq=rule))['volume'].sum()
    sell_vol = sell_trades.groupby(pd.Grouper(freq=rule))['volume'].sum()
    
    ohlc['buy_volume'] = buy_vol
    ohlc['sell_volume'] = sell_vol
    
    # VWAP
    ohlc['vwap'] = ohlc['volume'] / ohlc['quantity']
    
    # Forward fill para períodos sem trades
    ohlc['open'] = ohlc['open'].fillna(method='ffill')
    ohlc['high'] = ohlc['high'].fillna(ohlc['open'])
    ohlc['low'] = ohlc['low'].fillna(ohlc['open'])
    ohlc['close'] = ohlc['close'].fillna(ohlc['open'])
    
    return ohlc
```

## Referências Rápidas

### Códigos de Bolsas

| Código | Bolsa | Descrição |
|--------|-------|-----------|
| B | Bovespa | Ações à vista |
| F | BMF | Futuros e derivativos |
| M | CME | Chicago Mercantile Exchange |
| N | Nasdaq | Nasdaq |
| Y | NYSE | New York Stock Exchange |
| D | Câmbio | Mercado de câmbio |
| K | Metrics | Métricas |

### Estados de Conexão

| Estado | Valor | Significado |
|--------|-------|-------------|
| CONNECTION_STATE_LOGIN | 0 | Conexão com servidor de login |
| CONNECTION_STATE_ROTEAMENTO | 1 | Conexão com servidor de roteamento |
| CONNECTION_STATE_MARKET_DATA | 2 | Conexão com servidor de dados |
| CONNECTION_STATE_MARKET_LOGIN | 3 | Login no servidor de dados |

### Resultados de Conexão

| Resultado | Valor | Significado |
|-----------|-------|-------------|
| LOGIN_CONNECTED | 0 | Login bem-sucedido |
| LOGIN_INVALID | 1 | Login inválido |
| LOGIN_INVALID_PASS | 2 | Senha inválida |
| LOGIN_BLOCKED_PASS | 3 | Senha bloqueada |
| MARKET_CONNECTED | 4 | Conectado ao market data |

### Tipos de Trade

| Tipo | Valor | Descrição |
|------|-------|-----------|
| Cross Trade | 1 | Negócio direto |
| Buy Agressivo | 2 | Compra agressiva |
| Sell Agressivo | 3 | Venda agressiva |
| Leilão | 4 | Negócio em leilão |
| Vigilância | 5 | Sob vigilância |
| Ex-pit | 6 | Fora do pregão |
| Exercício | 7 | Exercício de opções |
| Balcão | 8 | Mercado de balcão |
| Termo | 9 | Operação a termo |
| Índice | 10 | Operação com índice |

### Horário do Servidor

```python
# Obter horário do servidor
dtDate = ctypes.c_double()
year = ctypes.c_int()
month = ctypes.c_int()
day = ctypes.c_int()
hour = ctypes.c_int()
minute = ctypes.c_int()
sec = ctypes.c_int()
milisec = ctypes.c_int()

result = dll.GetServerClock(
    ctypes.byref(dtDate),
    ctypes.byref(year),
    ctypes.byref(month),
    ctypes.byref(day),
    ctypes.byref(hour),
    ctypes.byref(minute),
    ctypes.byref(sec),
    ctypes.byref(milisec)
)

if result == 0:
    print(f"Horário do servidor: {day.value}/{month.value}/{year.value} "
          f"{hour.value}:{minute.value}:{sec.value}.{milisec.value}")
```

## Troubleshooting

### Problemas Comuns

1. **DLL não encontrada**
   - Verificar se a DLL está no PATH ou no diretório da aplicação
   - Usar caminho absoluto: `ctypes.CDLL(r'C:\path\to\ProfitDLL64.dll')`

2. **Erro de inicialização (NL_INVALID_ARGS)**
   - Verificar se todos os callbacks foram passados
   - Confirmar que a chave de ativação está correta
   - Verificar credenciais de login

3. **Callbacks não sendo chamados**
   - Confirmar que subscrição foi feita após conexão estabelecida
   - Verificar se símbolo e bolsa estão corretos
   - Aguardar callback de estado confirmar conexão

4. **Overflow de memória (32 bits)**
   - Usar versão 64 bits para grandes volumes
   - Processar dados em batches menores
   - Implementar paginação para histórico

5. **Timestamps incorretos**
   - Formato esperado: DD/MM/YYYY HH:mm:SS.ZZZ
   - Atenção para vírgula vs ponto nos milissegundos
   - Considerar timezone do servidor

### Performance e Otimizações

1. **Processamento de Callbacks**
   - Manter processamento mínimo dentro do callback
   - Usar buffer/fila para processar em thread separada
   - Evitar I/O dentro de callbacks

2. **Batch Insert no Banco**
   - Acumular trades em buffer antes de salvar
   - Usar prepared statements
   - Considerar particionamento de tabelas

3. **Gerenciamento de Conexão**
   - Implementar reconexão automática
   - Monitorar estado constantemente
   - Log de todos os eventos de conexão

### Exemplo de Monitoramento

```python
class ConnectionMonitor:
    def __init__(self, client):
        self.client = client
        self.last_heartbeat = time.time()
        self.reconnect_attempts = 0
        
    def monitor_loop(self):
        while True:
            if not self.client.is_connected:
                self.attempt_reconnect()
            
            # Verificar atividade
            if time.time() - self.last_heartbeat > 60:
                print("Aviso: Sem atividade há 60 segundos")
                
            time.sleep(5)
            
    def attempt_reconnect(self):
        self.reconnect_attempts += 1
        print(f"Tentativa de reconexão {self.reconnect_attempts}")
        
        try:
            self.client.cleanup()
            time.sleep(5)
            self.client.initialize()
            self.reconnect_attempts = 0
        except Exception as e:
            print(f"Falha na reconexão: {e}")
            
            if self.reconnect_attempts > 10:
                print("Muitas tentativas falhadas. Encerrando.")
                sys.exit(1)
```

---

## Notas Finais

- **Threading**: Todos os callbacks executam na ConnectorThread
- **Segurança**: Nunca chamar funções da DLL dentro de callbacks
- **Versão**: Usar 64 bits para produção com grandes volumes
- **Latência**: Processar callbacks rapidamente para não bloquear fila interna
- **Persistência**: Implementar salvamento em batch para eficiência

Este guia fornece a base completa para construir um sistema robusto de coleta e armazenamento de dados de mercado usando a ProfitDLL.