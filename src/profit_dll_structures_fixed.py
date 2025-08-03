"""
Estruturas corrigidas do ProfitDLL v4.0.0.30
Com tipos de retorno corretos para os callbacks
"""

from ctypes import (
    Structure, Union, c_int, c_int64, c_uint64, c_double, c_wchar_p,
    c_char, c_uint, c_void_p, c_bool, c_byte, POINTER, WINFUNCTYPE
)

# Importar estruturas originais
from src.profit_dll_structures import (
    TAssetID, TConnectorOrderIdentifier, TConnectorOrder,
    TConnectorAccountIdentifier, TConnectorAccountInfo,
    TConnectorMarketDepthItem, ConnectionState
)

# =============================================================================
# CALLBACKS CORRIGIDOS - TODOS DEVEM RETORNAR c_int
# =============================================================================

# Callback de estado de conexão
TStateCallbackFixed = WINFUNCTYPE(c_int, c_int, c_int)

# Callback de trades em tempo real
TNewTradeCallbackFixed = WINFUNCTYPE(
    c_int,         # RETORNO int
    TAssetID,      # Asset
    c_wchar_p,     # Date
    c_uint,        # TradeNumber
    c_double,      # Price
    c_double,      # Volume
    c_int,         # Quantity
    c_int,         # BuyAgent
    c_int,         # SellAgent
    c_int,         # TradeType
    c_char         # IsEdit
)

# Callback de histórico de trades
THistoryTradeCallbackFixed = WINFUNCTYPE(
    c_int,         # RETORNO int
    TAssetID,      # Asset
    c_wchar_p,     # Date
    c_uint,        # TradeNumber
    c_double,      # Price
    c_double,      # Volume
    c_int,         # Quantity
    c_int,         # BuyAgent
    c_int,         # SellAgent
    c_int          # TradeType
)

# Callback de progresso
TProgressCallbackFixed = WINFUNCTYPE(c_int, TAssetID, c_int)

# Callback de conta
TAccountCallbackFixed = WINFUNCTYPE(
    c_int,         # RETORNO int
    c_int,         # AccountID
    c_wchar_p,     # BrokerID
    c_wchar_p,     # AccountNumber
    c_wchar_p      # AccountInfo (pode ser NULL)
)

# Callback de histórico (ordem)
THistoryCallbackFixed = WINFUNCTYPE(c_int, c_void_p)

# Callback de mudança de ordem
TOrderChangeCallbackFixed = WINFUNCTYPE(c_int, c_void_p)

# Callback de dados diários
TNewDailyCallbackFixed = WINFUNCTYPE(c_int, c_void_p)

# Callback de book de preços
TPriceBookCallbackFixed = WINFUNCTYPE(
    c_int,         # RETORNO int
    TAssetID,      # Asset
    c_int,         # Side (0=Buy, 1=Sell)
    c_int,         # Position
    c_int,         # Quantity
    c_int,         # Count
    c_double,      # Price
    c_void_p       # Reserved
)

# Callback de book de ofertas
TOfferBookCallbackFixed = WINFUNCTYPE(
    c_int,         # RETORNO int
    TAssetID,      # Asset
    c_int,         # Side
    c_int,         # Position
    c_int,         # Quantity
    c_int,         # Count
    c_double,      # Price
    c_int,         # Action
    c_uint,        # DateTime
    c_int,         # ID
    POINTER(TConnectorOrderIdentifier),  # OrderID
    c_void_p       # Reserved
)

# Callback de tiny book
TTinyBookCallbackFixed = WINFUNCTYPE(c_int, c_void_p)


# =============================================================================
# FUNÇÃO AUXILIAR PARA CRIAR CALLBACKS SEGUROS
# =============================================================================

def create_safe_callback(callback_type, func):
    """
    Cria um callback seguro que sempre retorna 0
    """
    def safe_wrapper(*args):
        try:
            # Chamar função original
            result = func(*args)
            # Sempre retornar 0 para sucesso
            return 0
        except Exception as e:
            print(f"Erro em callback: {e}")
            # Retornar 0 mesmo em erro para não crashar
            return 0
    
    return callback_type(safe_wrapper)