# Solução: Callbacks ProfitDLL v4.0.0.30

## Resumo das Descobertas

### 1. Convenção de Chamada Correta
- **Solução**: Usar `WinDLL` (stdcall) ao invés de `CDLL` (cdecl)
- **Manual**: Confirma que callbacks devem usar stdcall
- **Exemplo oficial**: Usa `WinDLL` e `WINFUNCTYPE`

### 2. Callbacks Funcionando
✅ **CONFIRMADO**: Os callbacks estão funcionando corretamente!
- State callbacks: Recebendo estados de conexão
- Login: Estado 0 = conectado
- Market: Estado 4 = conectado
- Routing: Estado 5 = conectado

### 3. Problema com SubscribeTicker
**Erro -2147483645** = `NL_INVALID_ARGS` (argumentos inválidos)

Possíveis causas:
1. Formato incorreto do ticker/bolsa
2. Ticker não existe ou está inativo
3. Conta sem permissão para o ativo

### 4. Código Corrigido

```python
from ctypes import WinDLL, WINFUNCTYPE, c_int32, c_wchar_p

# 1. Carregar DLL com WinDLL
dll = WinDLL("./ProfitDLL64.dll")
dll.argtypes = None  # Importante!

# 2. Definir callback com WINFUNCTYPE
@WINFUNCTYPE(None, c_int32, c_int32)
def state_callback(nType, nResult):
    print(f"[STATE] Type={nType}, Result={nResult}")
    return None  # Retornar None explicitamente

# 3. Registrar callback
dll.SetStateCallback(state_callback)

# 4. Login com todos os parâmetros
result = dll.DLLInitializeLogin(
    c_wchar_p("HMARL"),
    c_wchar_p("username"),
    c_wchar_p("password"),
    state_callback,  # state
    None,           # history
    None,           # order change
    None,           # account
    None,           # asset list
    None,           # daily
    None,           # price book
    None,           # offer book
    None,           # new history
    None,           # progress
    None            # tiny book
)
```

## Estados de Conexão

### Login (Type=0)
- Result=0: Conectado ✅
- Result=1: Erro de conexão

### Routing/Broker (Type=1)
- Result=1: Conectando
- Result=2: Aguardando
- Result=4: Estabelecendo conexão
- Result=5: Conectado ✅

### Market Data (Type=2)
- Result=1: Conectando
- Result=2: Aguardando
- Result=4: Conectado ✅

### Ativação (Type=3)
- Result=0: Ativado ✅
- Result=1: Erro de ativação

## Tickers que Funcionam

Com base nos testes:
- ❌ `WDOQ25` sem bolsa: Erro -2147483645
- ✅ `WDOQ25` com bolsa `F`: Funcionou
- ❌ `WDOQ25` com bolsa `BMF`: Erro
- ✅ `PETR4` com bolsa `B`: Funcionou
- ✅ `VALE3` com bolsa `B`: Funcionou

## Próximos Passos

1. **Para receber dados de mercado**:
   - Aguardar mercado aberto (dias úteis)
   - Usar tickers com bolsa correta (`F` para futuros, `B` para ações)
   - Verificar permissões da conta

2. **Para callback de trades**:
   ```python
   @WINFUNCTYPE(None, TConnectorAssetIdentifier, c_size_t, c_uint)
   def trade_callback_v2(assetId, pTrade, flags):
       # Processar trade
       pass
   
   dll.SetTradeCallbackV2(trade_callback_v2)
   ```

3. **Para book de ofertas**:
   ```python
   # Subscrever ao book
   dll.SubscribeOfferBook(c_wchar_p("WDOQ25"), c_wchar_p("F"))
   ```

## Conclusão

✅ **O sistema está funcionando corretamente!**

Os callbacks estão sendo chamados, a conexão está estabelecida. O único problema é o formato dos tickers para subscrição, que precisa incluir a bolsa correta.

### Checklist Final
- [x] DLL carregada com WinDLL
- [x] Callbacks usando WINFUNCTYPE
- [x] State callbacks funcionando
- [x] Login bem sucedido
- [x] Market conectado
- [x] Routing conectado
- [ ] Dados de mercado (aguardar mercado aberto)
- [ ] Formato correto de ticker/bolsa