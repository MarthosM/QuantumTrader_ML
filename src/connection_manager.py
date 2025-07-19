"""
Connection Manager - Gerencia conexão com ProfitDLL e callbacks
Baseado em real_ml.py e enhanced_historical.py
"""

import logging
import time
from typing import Dict, Optional, Callable, Any
from ctypes import WINFUNCTYPE, WinDLL, c_int, c_wchar_p, c_double, c_uint, c_char, c_longlong, c_void_p
from datetime import datetime
import os

# Import da estrutura TAssetID do enhanced_historical
from ctypes import Structure

class TAssetID(Structure):
    _fields_ = [
        ("pwcTicker", c_wchar_p),
        ("pwcBolsa", c_wchar_p), 
        ("nFeed", c_int)
    ]

class ConnectionManager:
    """Gerencia conexão com Profit e callbacks essenciais"""
    
    def __init__(self, dll_path: str):
        self.dll_path = dll_path if dll_path else r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
        self.dll = None
        self.connected = False
        self.callbacks = {}
        self.logger = logging.getLogger('ConnectionManager')
        
        # Estados de conexão (baseado em enhanced_historical.py)
        self.market_connected = False
        self.broker_connected = False
        self.routing_connected = False
        self.login_state = -1
        self.routing_state = -1
        self.market_state = -1

        # Constantes de estado do manual
        self.CONNECTION_STATE_LOGIN = 0
        self.CONNECTION_STATE_ROTEAMENTO = 1
        self.CONNECTION_STATE_MARKET_DATA = 2
        self.CONNECTION_STATE_MARKET_LOGIN = 3

        self.LOGIN_CONNECTED = 0
        self.MARKET_CONNECTED = 4
        self.ROTEAMENTO_BROKER_CONNECTED = 5
                
        # Configurações do servidor
        self.server_address = os.getenv("SERVER_ADDRESS", "producao.nelogica.com.br")
        self.server_port = os.getenv("SERVER_PORT", "8184")
        
        # Callbacks registrados
        self.trade_callbacks = []
        self.state_callbacks = []
        
    def initialize(self, key: str, username: str, password: str, 
                  account_id: Optional[str] = None, broker_id: Optional[str] = None, 
                  trading_password: Optional[str] = None) -> bool:
        """
        Inicializa conexão com Profit
        Baseado em real_ml.py _initialize_dll()
        
        Args:
            key: Chave de acesso do Profit
            username: Nome de usuário
            password: Senha de login
            account_id: ID da conta (para simulador)
            broker_id: ID da corretora (para simulador)
            trading_password: Senha de trading (se necessária)
        """
        try:
            # Log dos parâmetros (sem senhas por segurança)
            self.logger.info(f"Inicializando conexão com usuário: {username}")
            if account_id:
                self.logger.info(f"Conta: {account_id}")
            if broker_id:
                self.logger.info(f"Corretora: {broker_id}")
            
            # Carregar DLL
            self.dll = self._load_dll()
            if not self.dll:
                return False
            
            # Configurar servidor
            self.logger.info(f"Configurando servidor: {self.server_address}:{self.server_port}")
            server_result = self.dll.SetServerAndPort(
                c_wchar_p(self.server_address),
                c_wchar_p(self.server_port)
            )
            self.logger.info(f"Resultado da configuração do servidor: {server_result}")
            
            # Configurar callbacks básicos
            self._setup_callbacks()
            
            # Conectar usando DLLInitializeLogin (inclui market data e routing)
            self.logger.info("Inicializando conexão completa com Profit...")
            init_result = self.dll.DLLInitializeLogin(
                c_wchar_p(key),
                c_wchar_p(username),
                c_wchar_p(password),
                self.callbacks['state'],           # StateCallback
                self.callbacks['order_history'],   # HistoryCallback
                self.callbacks['order_change'],    # OrderChangeCallback
                self.callbacks['account'],         # AccountCallback
                self.callbacks['trade'],           # NewTradeCallback
                None,                             # NewDailyCallback
                self.callbacks['price_book'],      # PriceBookCallback
                self.callbacks['offer_book'],      # OfferBookCallback
                self.callbacks['history'],         # HistoryTradeCallback
                self.callbacks['progress'],        # ProgressCallback
                None                              # TinyBookCallback
            )
            
            if init_result == 0:  # NL_OK
                self.logger.info("DLL inicializada com sucesso")
                
                # Aguardar conexões
                if self._wait_for_connections(timeout=30):
                    self.connected = True
                    return True
                else:
                    self.logger.error("Timeout aguardando conexões")
                    return False
            else:
                self.logger.error(f"Falha na inicialização da DLL: código {init_result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {e}", exc_info=True)
            return False
    
    def _load_dll(self) -> Optional[WinDLL]:
        """Carrega a DLL do Profit"""
        try:
            if not os.path.exists(self.dll_path):
                self.logger.error(f"DLL não encontrada em: {self.dll_path}")
                return None
                
            dll = WinDLL(self.dll_path)
            self.logger.info("DLL carregada com sucesso")
            return dll
            
        except Exception as e:
            self.logger.error(f"Erro carregando DLL: {e}")
            return None
    
    def _setup_callbacks(self):
        """
        Configura callbacks básicos
        Baseado em enhanced_historical.py _initialize_callbacks()
        """
        
        # State callback
        @WINFUNCTYPE(None, c_int, c_int)
        def state_callback(nConnStateType, nResult):
            try:
                if nConnStateType == self.CONNECTION_STATE_LOGIN:
                    self.login_state = nResult
                    self.logger.info(f"Estado de login: {nResult}")
                    if nResult == self.LOGIN_CONNECTED:
                        self.logger.info("Login conectado com sucesso")
                        
                elif nConnStateType == self.CONNECTION_STATE_ROTEAMENTO:
                    self.routing_state = nResult
                    self.logger.info(f"Estado de roteamento: {nResult}")
                    if nResult == self.ROTEAMENTO_BROKER_CONNECTED:
                        self.routing_connected = True
                        self.broker_connected = True
                        
                elif nConnStateType == self.CONNECTION_STATE_MARKET_DATA:
                    self.market_state = nResult
                    self.logger.info(f"Estado de market data: {nResult}")
                    if nResult == self.MARKET_CONNECTED:
                        self.market_connected = True
                            
                # Notificar callbacks registrados
                for callback in self.state_callbacks:
                    callback(nConnStateType, nResult)
                    
            except Exception as e:
                self.logger.error(f"Erro no state callback: {e}")
        
        # Trade callback (tempo real)
        @WINFUNCTYPE(None, TAssetID, c_wchar_p, c_uint, c_double, c_double, 
                     c_int, c_int, c_int, c_int, c_char)
        def trade_callback(asset_id, date, trade_number, price, vol, qtd, 
                          buy_agent, sell_agent, trade_type, b_edit):
            try:
                timestamp = datetime.strptime(str(date), '%d/%m/%Y %H:%M:%S.%f')
                
                trade_data = {
                    'timestamp': timestamp,
                    'ticker': asset_id.pwcTicker,
                    'price': float(price),
                    'volume': float(vol),
                    'quantity': int(qtd),
                    'trade_type': int(trade_type),
                    'trade_number': int(trade_number)
                }
                
                # Notificar callbacks registrados
                for callback in self.trade_callbacks:
                    callback(trade_data)
                    
            except Exception as e:
                self.logger.error(f"Erro no trade callback: {e}")
        
        # History trade callback
        @WINFUNCTYPE(None, TAssetID, c_wchar_p, c_uint, c_double, c_double,
                     c_int, c_int, c_int, c_int)
        def history_callback(asset_id, date, trade_number, price, vol, qtd,
                           buy_agent, sell_agent, trade_type):
            # Por enquanto, apenas log
            pass
        
        # Progress callback
        @WINFUNCTYPE(None, TAssetID, c_int)
        def progress_callback(asset_id, progress):
            self.logger.debug(f"Progresso: {progress}%")
        
        # Callbacks vazios para completar interface
        @WINFUNCTYPE(None, TAssetID, c_int, c_int, c_int, c_longlong, c_int,
                     c_longlong, c_double, c_char, c_char, c_char, c_char, 
                     c_char, c_wchar_p, c_void_p, c_void_p)
        def offer_book_callback(*args):
            pass
        
        @WINFUNCTYPE(None, TAssetID, c_int, c_int, c_int, c_longlong, c_int,
                     c_double, c_void_p, c_void_p)
        def price_book_callback(*args):
            pass
        
        # TOrderChangeCallback tem 17 parâmetros, não 18
        @WINFUNCTYPE(None, TAssetID, c_int, c_int, c_int, c_int, c_int, c_double,
                    c_double, c_double, c_longlong, c_wchar_p, c_wchar_p, c_wchar_p,
                    c_wchar_p, c_wchar_p, c_wchar_p, c_wchar_p)
        def order_change_callback(asset_id, corretora, qtd, traded_qtd, leaves_qtd,
                                side, price, stop_price, avg_price, profit_id,
                                tipo_ordem, conta, titular, cl_ord_id, status,
                                date, text_message):
            pass

        # THistoryCallback tem 16 parâmetros, não 17  
        @WINFUNCTYPE(None, TAssetID, c_int, c_int, c_int, c_int, c_int, c_double,
                    c_double, c_double, c_longlong, c_wchar_p, c_wchar_p, c_wchar_p,
                    c_wchar_p, c_wchar_p, c_wchar_p)
        def order_history_callback(asset_id, corretora, qtd, traded_qtd, leaves_qtd,
                                side, price, stop_price, avg_price, profit_id,
                                tipo_ordem, conta, titular, cl_ord_id, status, date):
            pass
        
        @WINFUNCTYPE(None, c_int, c_wchar_p, c_wchar_p, c_wchar_p)
        def account_callback(*args):
            pass
        
        # Armazenar callbacks
        self.callbacks = {
            'state': state_callback,
            'trade': trade_callback,
            'history': history_callback,
            'progress': progress_callback,
            'offer_book': offer_book_callback,
            'price_book': price_book_callback,
            'order_change': order_change_callback,
            'order_history': order_history_callback,
            'account': account_callback
        }
        
        self.logger.info("Callbacks configurados")
    
    def _wait_for_connections(self, timeout: int = 30) -> bool:
        """Aguarda conexões serem estabelecidas"""
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            # Verificar conexão de market data
            if self.market_connected:
                self.logger.info("Conectado aos dados de mercado")
                return True
                
            # Log periódico
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                self.logger.info(f"Aguardando conexões ({int(elapsed)}s)...")
                self._log_connection_states()
                
            time.sleep(1)
            
        return False
    
    def _log_connection_states(self):
        """Log dos estados de conexão"""
        self.logger.info("=== Estados de Conexão ===")
        self.logger.info(f"Login: {self.login_state}")
        self.logger.info(f"Roteamento: {self.routing_state} (conectado: {self.routing_connected})")
        self.logger.info(f"Market Data: {self.market_state} (conectado: {self.market_connected})")
        self.logger.info("========================")
    
    def register_trade_callback(self, callback: Callable):
        """Registra callback para trades em tempo real"""
        self.trade_callbacks.append(callback)
        
    def register_state_callback(self, callback: Callable):
        """Registra callback para mudanças de estado"""
        self.state_callbacks.append(callback)
    
    def subscribe_ticker(self, ticker: str, exchange: str = "F") -> bool:
        """Subscreve para receber dados de um ticker"""
        try:
            if self.dll is None:
                self.logger.error("DLL não está carregada. Inicialize antes de subscrever ticker.")
                return False
            if not hasattr(self.dll, "SubscribeTicker"):
                self.logger.error("Método SubscribeTicker não encontrado na DLL.")
                return False
            result = self.dll.SubscribeTicker(c_wchar_p(ticker), c_wchar_p(exchange))
            if result == 0:
                self.logger.info(f"Subscrito para {ticker} em {exchange}")
                return True
            else:
                self.logger.error(f"Falha ao subscrever {ticker}: código {result}")
                return False
        except Exception as e:
            self.logger.error(f"Erro ao subscrever ticker: {e}")
            return False
    
    def request_historical_data(self, ticker: str, start_date: datetime, 
                               end_date: datetime) -> int:
        """Solicita dados históricos"""
        try:
            if self.dll is None:
                self.logger.error("DLL não está carregada. Inicialize antes de solicitar dados históricos.")
                return -1

            start_str = start_date.strftime('%d/%m/%Y %H:%M:00')
            end_str = end_date.strftime('%d/%m/%Y %H:%M:00')
            
            if not hasattr(self.dll, "GetHistoryTrades"):
                self.logger.error("Método GetHistoryTrades não encontrado na DLL.")
                return -1

            result = self.dll.GetHistoryTrades(
                c_wchar_p(ticker),
                c_wchar_p("F"),
                c_wchar_p(start_str),
                c_wchar_p(end_str)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro solicitando dados históricos: {e}")
            return -1
    
    def unsubscribe_ticker(self, ticker: str) -> bool:
        """
        Cancela subscrição de um ticker
        
        Args:
            ticker: Código do ativo
            
        Returns:
            bool: Sucesso da operação
        """
        try:
            # Implementar chamada específica da DLL
            # Por enquanto, apenas log
            self.logger.info(f"Cancelando subscrição de {ticker}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao cancelar subscrição: {e}")
            return False
        
    def disconnect(self):
        """Desconecta e limpa recursos"""
        try:
            if self.dll:
                self.dll.DLLFinalize()
                self.logger.info("DLL finalizada")
            self.connected = False
        except Exception as e:
            self.logger.error(f"Erro ao desconectar: {e}")

    def get_account_info(self) -> bool:
        """
        Solicita informações das contas disponíveis
        
        Returns:
            bool: Sucesso da operação
        """
        try:
            if not self.dll or not self.connected:
                self.logger.error("DLL não está conectada")
                return False
                
            result = self.dll.GetAccount()
            if result == 0:  # NL_OK
                self.logger.info("Solicitação de contas enviada")
                return True
            else:
                self.logger.error(f"Erro ao solicitar contas: código {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro ao obter informações de conta: {e}")
            return False