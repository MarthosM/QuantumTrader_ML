# Referência Completa de Funções - ProfitDLL v4.0.0.30

## 1. Funções de Inicialização e Finalização

### DLLInitializeLogin
```c
function DLLInitializeLogin(
    const pwcActivationKey: PWideChar;
    const pwcUser: PWideChar;
    const pwcPassword: PWideChar;
    StateCallback: TStateCallback;
    HistoryCallback: THistoryCallback;
    OrderChangeCallback: TOrderChangeCallback;
    AccountCallback: TAccountCallback;
    NewTradeCallback: TNewTradeCallback;
    NewDailyCallback: TNewDailyCallback;
    PriceBookCallback: TPriceBookCallback;
    OfferBookCallback: TOfferBookCallback;
    HistoryTradeCallback: THistoryTradeCallback;
    ProgressCallback: TProgressCallback;
    TinyBookCallback: TTinyBookCallback
): Integer;
```
**Descrição**: Inicializa serviços completos (Market Data + Routing)  
**Uso**: Obrigatório para enviar ordens

### DLLInitializeMarketLogin
```c
function DLLInitializeMarketLogin(
    const pwcActivationKey: PWideChar;
    const pwcUser: PWideChar;
    const pwcPassword: PWideChar;
    StateCallback: TStateCallback;
    NewTradeCallback: TNewTradeCallback;
    NewDailyCallback: TNewDailyCallback;
    PriceBookCallback: TPriceBookCallback;
    OfferBookCallback: TOfferBookCallback;
    HistoryTradeCallback: THistoryTradeCallback;
    ProgressCallback: TProgressCallback;
    TinyBookCallback: TTinyBookCallback
): Integer;
```
**Descrição**: Inicializa apenas Market Data (sem routing)  
**Uso**: Para receber dados de mercado sem operar

### DLLFinalize
```c
function DLLFinalize: Integer;
```
**Descrição**: Finaliza todos os serviços da DLL

## 2. Funções de Configuração

### SetServerAndPort
```c
function SetServerAndPort(const strServer, strPort: PWideChar): Integer;
```
**Descrição**: Define servidor específico de Market Data  
**Nota**: Usar apenas com orientação da equipe de desenvolvimento

### SetDayTrade
```c
function SetDayTrade(bUseDayTrade: Integer): Integer;
```
**Descrição**: Ativa/desativa flag de day trade  
**Valores**: 1 = ativar, 0 = desativar

### SetEnabledHistOrder
```c
function SetEnabledHistOrder(bEnabled: Integer): Integer;
```
**Descrição**: Habilita/desabilita carregamento automático de histórico de ordens  
**Importante**: Desabilitar compromete cálculo de posição

### SetEnabledLogToDebug
```c
function SetEnabledLogToDebug(bEnabled: Integer): Integer;
```
**Descrição**: Ativa/desativa logs de debug

## 3. Funções de Market Data

### SubscribeTicker
```c
function SubscribeTicker(pwcTicker: PWideChar; pwcBolsa: PWideChar): Integer;
```
**Descrição**: Assina cotações em tempo real  
**Exemplo**: `SubscribeTicker("WDOQ25", "F")`

### UnsubscribeTicker
```c
function UnsubscribeTicker(pwcTicker: PWideChar; pwcBolsa: PWideChar): Integer;
```
**Descrição**: Cancela assinatura de cotações

### SubscribePriceBook
```c
function SubscribePriceBook(pwcTicker: PWideChar; pwcBolsa: PWideChar): Integer;
```
**Descrição**: Assina book de preços agregado

### UnsubscribePriceBook
```c
function UnsubscribePriceBook(pwcTicker: PWideChar; pwcBolsa: PWideChar): Integer;
```
**Descrição**: Cancela assinatura de book de preços

### SubscribeOfferBook
```c
function SubscribeOfferBook(pwcTicker: PWideChar; pwcBolsa: PWideChar): Integer;
```
**Descrição**: Assina book de ofertas detalhado

### UnsubscribeOfferBook
```c
function UnsubscribeOfferBook(pwcTicker: PWideChar; pwcBolsa: PWideChar): Integer;
```
**Descrição**: Cancela assinatura de book de ofertas

### SubscribeAdjustHistory
```c
function SubscribeAdjustHistory(pwcTicker: PWideChar; pwcBolsa: PWideChar): Integer;
```
**Descrição**: Assina histórico de ajustes do ativo

### UnsubscribeAdjustHistory
```c
function UnsubscribeAdjustHistory(pwcTicker: PWideChar; pwcBolsa: PWideChar): Integer;
```
**Descrição**: Cancela assinatura de ajustes

### GetHistoryTrades
```c
function GetHistoryTrades(
    const pwcTicker: PWideChar;
    const pwcBolsa: PWideChar;
    dtDateStart: PWideChar;
    dtDateEnd: PWideChar
): Integer;
```
**Descrição**: Solicita histórico de negócios  
**Formato data**: "DD/MM/YYYY HH:mm:SS"

### RequestTickerInfo
```c
function RequestTickerInfo(const pwcTicker: PWideChar; const pwcBolsa: PWideChar): Integer;
```
**Descrição**: Solicita informações detalhadas do ativo

## 4. Funções de Dados

### GetServerClock
```c
function GetServerClock(
    var dtDate: Double;
    var nYear, nMonth, nDay, nHour, nMin, nSec, nMilisec: Integer
): Integer;
```
**Descrição**: Obtém horário do servidor

### GetLastDailyClose
```c
function GetLastDailyClose(
    const pwcTicker, pwcBolsa: PWideChar;
    var dClose: Double;
    bAdjusted: Integer
): Integer;
```
**Descrição**: Obtém fechamento do dia anterior  
**bAdjusted**: 0 = sem ajuste, 1 = ajustado

### GetAgentNameLength
```c
function GetAgentNameLength(nAgentID: Integer; nShortName: Cardinal): Integer;
```
**Descrição**: Obtém tamanho do nome do agente

### GetAgentName
```c
function GetAgentName(
    nCount: Integer;
    nAgentID: Integer;
    pwcAgent: PWideChar;
    nShortName: Cardinal
): Integer;
```
**Descrição**: Obtém nome completo ou abreviado do agente

### GetAgentNameByID (Deprecated)
```c
function GetAgentNameByID(nID: Integer): PWideChar;
```
**Descrição**: Retorna nome completo do agente  
**Status**: Depreciado - usar GetAgentName

### GetAgentShortNameByID (Deprecated)
```c
function GetAgentShortNameByID(nID: Integer): PWideChar;
```
**Descrição**: Retorna nome abreviado do agente  
**Status**: Depreciado - usar GetAgentName

## 5. Funções de Contas

### GetAccount
```c
function GetAccount: Integer;
```
**Descrição**: Solicita informações de contas disponíveis

### GetAccountCount
```c
function GetAccountCount: Integer;
```
**Descrição**: Retorna número total de contas (sem sub-contas)

### GetAccounts
```c
function GetAccounts(
    const a_nStartSource: Integer;
    const a_nStartDest: Integer;
    const a_nCount: Integer;
    const a_arAccounts: PConnectorAccountIdentifierArrayOut
): Integer;
```
**Descrição**: Obtém lista de identificadores de contas

### GetAccountDetails
```c
function GetAccountDetails(var a_Account: TConnectorTradingAccountOut): Integer;
```
**Descrição**: Obtém detalhes de uma conta específica

### GetAccountCountByBroker
```c
function GetAccountCountByBroker(a_nBrokerID: Integer): Integer;
```
**Descrição**: Retorna número de contas filtradas por corretora

### GetAccountsByBroker
```c
function GetAccountsByBroker(
    a_nBrokerID: Integer;
    a_nStartSource: Integer;
    a_nStartDest: Integer;
    a_nCount: Integer;
    a_arAccounts: PConnectorAccountIdentifierArrayOut
): Integer;
```
**Descrição**: Obtém contas filtradas por corretora

### GetSubAccountCount
```c
function GetSubAccountCount(const a_MasterAccountID: PConnectorAccountIdentifier): Integer;
```
**Descrição**: Retorna número de sub-contas de uma conta master

### GetSubAccounts
```c
function GetSubAccounts(
    const a_MasterAccountID: PConnectorAccountIdentifier;
    const a_nStartSource: Integer;
    const a_nStartDest: Integer;
    const a_nCount: Integer;
    const a_arAccounts: PConnectorAccountIdentifierArrayOut
): Integer;
```
**Descrição**: Obtém lista de sub-contas

## 6. Funções de Ordens - Envio

### SendOrder
```c
function SendOrder(const a_SendOrder: PConnectorSendOrder): Int64;
```
**Descrição**: Função unificada para enviar todos os tipos de ordens  
**Retorno**: LocalOrderID (> 0 = sucesso)

### SendBuyOrder (Deprecated)
```c
function SendBuyOrder(
    pwcIDAccount: PWideChar;
    pwcIDCorretora: PWideChar;
    pwcSenha: PWideChar;
    pwcTicker: PWideChar;
    pwcBolsa: PWideChar;
    dPrice: Double;
    nAmount: Integer
): Int64;
```
**Status**: Depreciado - usar SendOrder

### SendSellOrder (Deprecated)
```c
function SendSellOrder(...): Int64;
```
**Status**: Depreciado - usar SendOrder

### SendMarketBuyOrder (Deprecated)
```c
function SendMarketBuyOrder(...): Int64;
```
**Status**: Depreciado - usar SendOrder

### SendMarketSellOrder (Deprecated)
```c
function SendMarketSellOrder(...): Int64;
```
**Status**: Depreciado - usar SendOrder

### SendStopBuyOrder (Deprecated)
```c
function SendStopBuyOrder(...): Int64;
```
**Status**: Depreciado - usar SendOrder

### SendStopSellOrder (Deprecated)
```c
function SendStopSellOrder(...): Int64;
```
**Status**: Depreciado - usar SendOrder

## 7. Funções de Ordens - Modificação e Cancelamento

### SendChangeOrderV2
```c
function SendChangeOrderV2(const a_ChangeOrder: PConnectorChangeOrder): Integer;
```
**Descrição**: Modifica ordem existente

### SendChangeOrder (Deprecated)
```c
function SendChangeOrder(...): Integer;
```
**Status**: Depreciado - usar SendChangeOrderV2

### SendCancelOrderV2
```c
function SendCancelOrderV2(const a_CancelOrder: PConnectorCancelOrder): Integer;
```
**Descrição**: Cancela ordem específica

### SendCancelOrder (Deprecated)
```c
function SendCancelOrder(...): Integer;
```
**Status**: Depreciado - usar SendCancelOrderV2

### SendCancelOrdersV2
```c
function SendCancelOrdersV2(const a_CancelOrder: PConnectorCancelOrders): Integer;
```
**Descrição**: Cancela todas as ordens de um ativo

### SendCancelOrders (Deprecated)
```c
function SendCancelOrders(...): Integer;
```
**Status**: Depreciado - usar SendCancelOrdersV2

### SendCancelAllOrdersV2
```c
function SendCancelAllOrdersV2(const a_CancelOrder: PConnectorCancelAllOrders): Integer;
```
**Descrição**: Cancela todas as ordens abertas

### SendCancelAllOrders (Deprecated)
```c
function SendCancelAllOrders(...): Integer;
```
**Status**: Depreciado - usar SendCancelAllOrdersV2

## 8. Funções de Ordens - Consulta

### GetOrderDetails
```c
function GetOrderDetails(var a_Order: TConnectorOrderOut): Integer;
```
**Descrição**: Obtém detalhes de uma ordem específica

### GetOrder (Deprecated)
```c
function GetOrder(pwcClOrdId: PWideChar): Integer;
```
**Status**: Depreciado - usar GetOrderDetails

### GetOrderProfitID (Deprecated)
```c
function GetOrderProfitID(nProfitID: Int64): Integer;
```
**Status**: Depreciado - usar GetOrderDetails

### GetOrders (Deprecated)
```c
function GetOrders(
    pwcIDAccount: PWideChar;
    pwcIDCorretora: PWideChar;
    dtStart: PWideChar;
    dtEnd: PWideChar
): Integer;
```
**Status**: Depreciado - usar HasOrdersInInterval e EnumerateOrdersByInterval

### HasOrdersInInterval
```c
function HasOrdersInInterval(
    const a_AccountID: PConnectorAccountIdentifier;
    const a_dtStart: TSystemTime;
    const a_dtEnd: TSystemTime
): NResult;
```
**Descrição**: Verifica se há ordens no período  
**Retorno**: NL_OK = tem ordens, NL_WAITING_SERVER = solicitando

### EnumerateOrdersByInterval
```c
function EnumerateOrdersByInterval(
    const a_AccountID: PConnectorAccountIdentifier;
    const a_OrderVersion: Byte;
    const a_dtStart: TSystemTime;
    const a_dtEnd: TSystemTime;
    const a_Param: LPARAM;
    const a_Callback: TConnectorEnumerateOrdersProc
): NResult;
```
**Descrição**: Enumera ordens em um período

### EnumerateAllOrders
```c
function EnumerateAllOrders(
    const a_AccountID: PConnectorAccountIdentifier;
    const a_OrderVersion: Byte;
    const a_Param: LPARAM;
    const a_Callback: TConnectorEnumerateOrdersProc
): NResult;
```
**Descrição**: Enumera todas as ordens da conta

## 9. Funções de Posição

### GetPositionV2
```c
function GetPositionV2(var a_Position: TConnectorTradingAccountPosition): Integer;
```
**Descrição**: Obtém posição detalhada de um ativo

### GetPosition (Deprecated)
```c
function GetPosition(
    pwcIDAccount: PWideChar;
    pwcIDCorretora: PWideChar;
    pwcTicker: PWideChar;
    pwcBolsa: PWideChar
): Pointer;
```
**Status**: Depreciado - usar GetPositionV2

### SendZeroPositionV2
```c
function SendZeroPositionV2(const a_ZeroPosition: PConnectorZeroPosition): Int64;
```
**Descrição**: Zera posição de um ativo  
**Price**: -1 para ordem a mercado

### SendZeroPosition (Deprecated)
```c
function SendZeroPosition(...): Int64;
```
**Status**: Depreciado - usar SendZeroPositionV2

### SendZeroPositionAtMarket (Deprecated)
```c
function SendZeroPositionAtMarket(...): Int64;
```
**Status**: Depreciado - usar SendZeroPositionV2 com price = -1

### EnumerateAllPositionAssets
```c
function EnumerateAllPositionAssets(
    const a_AccountID: PConnectorAccountIdentifier;
    const a_AssetVersion: Byte;
    const a_Param: LPARAM;
    const a_Callback: TConnectorEnumerateAssetProc
): NResult;
```
**Descrição**: Enumera todos os ativos com posição aberta

## 10. Funções de Callbacks

### SetStateCallback
```c
function SetStateCallback(const a_StateCallback: TStateCallback): Integer;
```
**Descrição**: Define callback de estado de conexão

### SetAssetListCallback
```c
function SetAssetListCallback(const a_AssetListCallback: TAssetListCallback): Integer;
```
**Descrição**: Define callback de lista de ativos

### SetAssetListInfoCallback
```c
function SetAssetListInfoCallback(const a_AssetListInfoCallback: TAssetListInfoCallback): Integer;
```
**Descrição**: Define callback de informações detalhadas de ativos

### SetAssetListInfoCallbackV2
```c
function SetAssetListInfoCallbackV2(const a_AssetListInfoCallbackV2: TAssetListInfoCallbackV2): Integer;
```
**Descrição**: Versão estendida com setor, subsetor e segmento

### SetInvalidTickerCallback
```c
function SetInvalidTickerCallback(const a_InvalidTickerCallback: TInvalidTickerCallback): Integer;
```
**Descrição**: Define callback para tickers inválidos

### SetTradeCallback (Deprecated)
```c
function SetTradeCallback(const a_TradeCallback: TTradeCallback): Integer;
```
**Status**: Depreciado - usar SetTradeCallbackV2

### SetTradeCallbackV2
```c
function SetTradeCallbackV2(const a_TradeCallbackV2: TConnectorTradeCallback): NResult;
```
**Descrição**: Define callback para negócios em tempo real

### SetHistoryTradeCallback (Deprecated)
```c
function SetHistoryTradeCallback(const a_HistoryTradeCallback: THistoryTradeCallback): Integer;
```
**Status**: Depreciado - usar SetHistoryTradeCallbackV2

### SetHistoryTradeCallbackV2
```c
function SetHistoryTradeCallbackV2(const a_HistoryTradeCallbackV2: TConnectorTradeCallback): NResult;
```
**Descrição**: Define callback para histórico de negócios

### SetDailyCallback
```c
function SetDailyCallback(const a_DailyCallback: TDailyCallback): Integer;
```
**Descrição**: Define callback para dados diários agregados

### SetTheoreticalPriceCallback
```c
function SetTheoreticalPriceCallback(const a_TheoreticalPriceCallback: TTheoreticalPriceCallback): Integer;
```
**Descrição**: Define callback para preços teóricos em leilão

### SetTinyBookCallback
```c
function SetTinyBookCallback(const a_TinyBookCallback: TTinyBookCallback): Integer;
```
**Descrição**: Define callback para topo do book

### SetChangeCotationCallback
```c
function SetChangeCotationCallback(const a_ChangeCotation: TChangeCotation): Integer;
```
**Descrição**: Define callback para mudanças de cotação

### SetChangeStateTickerCallback
```c
function SetChangeStateTickerCallback(const a_ChangeStateTicker: TChangeStateTicker): Integer;
```
**Descrição**: Define callback para mudanças de estado do ativo

### SetSerieProgressCallback
```c
function SetSerieProgressCallback(const a_SerieProgressCallback: TProgressCallback): Integer;
```
**Descrição**: Define callback de progresso para downloads

### SetOfferBookCallback
```c
function SetOfferBookCallback(const a_OfferBookCallback: TOfferBookCallback): Integer;
```
**Descrição**: Define callback para book de ofertas

### SetOfferBookCallbackV2
```c
function SetOfferBookCallbackV2(const a_OfferBookCallbackV2: TOfferBookCallbackV2): Integer;
```
**Descrição**: Versão atualizada do callback de book de ofertas

### SetPriceBookCallback
```c
function SetPriceBookCallback(const a_PriceBookCallback: TPriceBookCallback): Integer;
```
**Descrição**: Define callback para book de preços

### SetPriceBookCallbackV2
```c
function SetPriceBookCallbackV2(const a_PriceBookCallbackV2: TPriceBookCallbackV2): Integer;
```
**Descrição**: Versão atualizada do callback de book de preços

### SetAdjustHistoryCallback
```c
function SetAdjustHistoryCallback(const a_AdjustHistoryCallback: TAdjustHistoryCallback): Integer;
```
**Descrição**: Define callback para histórico de ajustes

### SetAdjustHistoryCallbackV2
```c
function SetAdjustHistoryCallbackV2(const a_AdjustHistoryCallbackV2: TAdjustHistoryCallbackV2): Integer;
```
**Descrição**: Versão estendida com flags e multiplicador

### SetAssetPositionListCallback
```c
function SetAssetPositionListCallback(const a_AssetPositionListCallback: TConnectorAssetPositionListCallback): Integer;
```
**Descrição**: Define callback para mudanças nas posições

### SetAccountCallback
```c
function SetAccountCallback(const a_AccountCallback: TAccountCallback): Integer;
```
**Descrição**: Define callback para informações de contas

### SetHistoryCallback (Deprecated)
```c
function SetHistoryCallback(const a_HistoryCallback: THistoryCallback): Integer;
```
**Status**: Depreciado - usar SetOrderHistoryCallback

### SetHistoryCallbackV2 (Deprecated)
```c
function SetHistoryCallbackV2(const a_HistoryCallbackV2: THistoryCallbackV2): Integer;
```
**Status**: Depreciado - usar SetOrderHistoryCallback

### SetOrderChangeCallback (Deprecated)
```c
function SetOrderChangeCallback(const a_OrderChangeCallback: TOrderChangeCallback): Integer;
```
**Status**: Depreciado - usar SetOrderCallback

### SetOrderChangeCallbackV2 (Deprecated)
```c
function SetOrderChangeCallbackV2(const a_OrderChangeCallbackV2: TOrderChangeCallbackV2): Integer;
```
**Status**: Depreciado - usar SetOrderCallback

### SetOrderCallback
```c
function SetOrderCallback(const a_OrderCallback: TConnectorOrderCallback): Integer;
```
**Descrição**: Define callback unificado para mudanças em ordens

### SetOrderHistoryCallback
```c
function SetOrderHistoryCallback(const a_OrderHistoryCallback: TConnectorAccountCallback): NResult;
```
**Descrição**: Define callback para carregamento do histórico de ordens

### SetBrokerAccountListChangedCallback
```c
function SetBrokerAccountListChangedCallback(const a_Callback: TBrokerAccountListCallback): Integer;
```
**Descrição**: Define callback para mudanças na lista de contas

### SetBrokerSubAccountListChangedCallback
```c
function SetBrokerSubAccountListChangedCallback(const a_Callback: TBrokerSubAccountListCallback): Integer;
```
**Descrição**: Define callback para mudanças na lista de sub-contas

## 11. Função Auxiliar

### TranslateTrade
```c
function TranslateTrade(const a_pTrade: Pointer; var a_Trade: TConnectorTrade): NResult;
```
**Descrição**: Traduz dados de trade recebidos em callbacks

## Códigos de Retorno

### Sucesso
- **NL_OK** (0): Operação bem-sucedida

### Erros Comuns
- **NL_INTERNAL_ERROR** (-2147483647): Erro interno
- **NL_NOT_INITIALIZED** (-2147483646): DLL não inicializada
- **NL_INVALID_ARGS** (-2147483645): Argumentos inválidos
- **NL_WAITING_SERVER** (-2147483644): Aguardando dados do servidor
- **NL_NO_LOGIN** (-2147483643): Sem login
- **NL_NO_LICENSE** (-2147483642): Sem licença
- **NL_NO_POSITION** (-2147483637): Posição não encontrada
- **NL_NOT_FOUND** (-2147483636): Recurso não encontrado
- **NL_NO_PASSWORD** (-2147483620): Senha não fornecida

## Exchanges (Bolsas)

- **B** (66): Bovespa
- **F** (70): BMF (Futuros)
- **M** (77): CME
- **N** (78): Nasdaq
- **Y** (89): NYSE

## Estados de Conexão

### CONNECTION_STATE_LOGIN (0)
- LOGIN_CONNECTED (0): Conectado
- LOGIN_INVALID (1): Login inválido
- LOGIN_INVALID_PASS (2): Senha inválida
- LOGIN_BLOCKED_PASS (3): Senha bloqueada

### CONNECTION_STATE_ROTEAMENTO (1)
- ROTEAMENTO_DISCONNECTED (0): Desconectado
- ROTEAMENTO_CONNECTED (2): Conectado

### CONNECTION_STATE_MARKET_DATA (2)
- MARKET_DISCONNECTED (0): Desconectado
- MARKET_CONNECTED (4): Conectado

### CONNECTION_STATE_MARKET_LOGIN (3)
- CONNECTION_ACTIVATE_VALID (0): Ativação válida
- CONNECTION_ACTIVATE_INVALID (1): Ativação inválida