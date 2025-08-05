"""
Connection Manager CORRIGIDO baseado no exemplo oficial da ProfitDLL
Usa WinDLL e WINFUNCTYPE conforme documentação
"""

import os
import sys
import time
import threading
import logging
from pathlib import Path
from datetime import datetime
from ctypes import *
import pandas as pd
from collections import deque
import queue

# Importar tipos do exemplo oficial
class TAssetID(Structure):
    _fields_ = [
        ("ticker", c_wchar_p),
        ("bolsa", c_wchar_p),
        ("feed", c_int)
    ]

class TConnectorAssetIdentifier(Structure):
    _fields_ = [
        ("Version", c_ubyte),
        ("Ticker", c_wchar_p),
        ("Exchange", c_wchar_p),
        ("FeedType", c_ubyte)
    ]

class TConnectorTrade(Structure):
    _fields_ = [
        ("Version", c_ubyte),
        ("Price", c_double),
        ("Quantity", c_int),
        ("DateTime", c_wchar_p),
        ("Side", c_ubyte),
        ("TradeID", c_int64)
    ]

class ConnectionManagerFixed:
    """Connection Manager corrigido baseado no exemplo oficial"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('ConnectionManagerFixed')
        
        # DLL
        self.dll = None
        
        # Estado
        self.is_connected = False
        self.market_connected = False
        self.broker_connected = False
        self.is_active = False
        self.states_received = []
        
        # Filas
        self.trade_queue = queue.Queue(maxsize=10000)
        self.book_queue = queue.Queue(maxsize=10000)
        
        # Referências dos callbacks (IMPORTANTE!)
        self.callback_refs = {}
        
        # Estatísticas
        self.stats = {
            'state_callbacks': 0,
            'trade_callbacks': 0,
            'book_callbacks': 0,
            'cotation_callbacks': 0
        }
        
    def initialize(self):
        """Inicializa DLL com WinDLL conforme exemplo oficial"""
        try:
            dll_path = Path(self.config['dll_path']).absolute()
            self.logger.info(f"Carregando DLL: {dll_path}")
            
            # USAR WinDLL CONFORME EXEMPLO OFICIAL
            self.dll = WinDLL(str(dll_path))
            self.logger.info("✓ DLL carregada com WinDLL (stdcall)")
            
            # Configurar tipos de retorno
            self._setup_dll_types()
            
            # Configurar callbacks ANTES do login
            self._setup_callbacks()
            
            # Login
            return self._do_login()
            
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _setup_dll_types(self):
        """Configura tipos das funções conforme exemplo oficial"""
        # DLLInitializeLogin
        self.dll.DLLInitializeLogin.argtypes = [
            c_wchar_p,  # key
            c_wchar_p,  # username  
            c_wchar_p,  # password
            c_void_p,   # stateCallback
            c_void_p,   # historyCallback
            c_void_p,   # orderChangeCallback
            c_void_p,   # accountCallback
            c_void_p,   # assetListCallback
            c_void_p,   # newDailyCallback
            c_void_p,   # priceBookCallback
            c_void_p,   # offerBookCallback
            c_void_p,   # newHistoryCallback
            c_void_p,   # progressCallback
            c_void_p    # tinyBookCallback
        ]
        self.dll.DLLInitializeLogin.restype = c_int
        
        # SetTradeCallbackV2
        if hasattr(self.dll, 'SetTradeCallbackV2'):
            self.dll.SetTradeCallbackV2.argtypes = [c_void_p]
            self.dll.SetTradeCallbackV2.restype = None
            
        # SetOfferBookCallbackV2
        if hasattr(self.dll, 'SetOfferBookCallbackV2'):
            self.dll.SetOfferBookCallbackV2.argtypes = [c_void_p]
            self.dll.SetOfferBookCallbackV2.restype = c_int
            
        # SubscribeTicker
        self.dll.SubscribeTicker.argtypes = [c_wchar_p, c_wchar_p]
        self.dll.SubscribeTicker.restype = c_int
        
        # SetAssetListCallback
        if hasattr(self.dll, 'SetAssetListCallback'):
            self.dll.SetAssetListCallback.argtypes = [c_void_p]
            self.dll.SetAssetListCallback.restype = None
            
    def _setup_callbacks(self):
        """Configura callbacks usando WINFUNCTYPE conforme exemplo oficial"""
        self.logger.info("Configurando callbacks...")
        
        # 1. State Callback - EXATAMENTE como no exemplo
        @WINFUNCTYPE(None, c_int32, c_int32)
        def state_callback(nType, nResult):
            self.stats['state_callbacks'] += 1
            self.states_received.append((nType, nResult))
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.logger.info(f"[{timestamp}] STATE: Type={nType}, Result={nResult}")
            
            if nType == 0:  # Login
                if nResult == 0:
                    self.is_connected = True
                    self.logger.info("✓ Login: conectado")
                else:
                    self.is_connected = False
                    self.logger.error(f"Login error: {nResult}")
                    
            elif nType == 1:  # Broker
                if nResult == 5:
                    self.broker_connected = True
                    self.logger.info("✓ Broker: Conectado")
                else:
                    self.broker_connected = False
                    self.logger.warning(f"Broker status: {nResult}")
                    
            elif nType == 2:  # Market
                if nResult == 4:
                    self.market_connected = True
                    self.logger.info("✓ Market: Conectado")
                else:
                    self.market_connected = False
                    self.logger.warning(f"Market status: {nResult}")
                    
            elif nType == 3:  # Activation
                if nResult == 0:
                    self.is_active = True
                    self.logger.info("✓ Ativação: OK")
                else:
                    self.is_active = False
                    self.logger.error(f"Ativação error: {nResult}")
                    
            if self.market_connected and self.is_active and self.is_connected:
                self.logger.info("✓ TODOS OS SERVIÇOS CONECTADOS!")
                
        # GUARDAR REFERÊNCIA!
        self.callback_refs['state'] = state_callback
        
        # 2. Trade Callback V2
        if hasattr(self.dll, 'SetTradeCallbackV2'):
            @WINFUNCTYPE(None, TConnectorAssetIdentifier, c_size_t, c_uint)
            def trade_callback_v2(assetId, pTrade, flags):
                self.stats['trade_callbacks'] += 1
                
                try:
                    is_edit = bool(flags & 1)
                    trade = TConnectorTrade(Version=0)
                    
                    if hasattr(self.dll, 'TranslateTrade'):
                        if self.dll.TranslateTrade(pTrade, byref(trade)):
                            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            ticker = assetId.Ticker if assetId.Ticker else "N/A"
                            
                            self.logger.info(f"[{timestamp}] TRADE: {ticker} | "
                                           f"R$ {trade.Price:.2f} | Qtd: {trade.Quantity} | "
                                           f"Edit={is_edit}")
                            
                            # Adicionar à fila
                            data = {
                                'type': 'trade',
                                'ticker': ticker,
                                'price': trade.Price,
                                'quantity': trade.Quantity,
                                'side': trade.Side,
                                'timestamp': datetime.now(),
                                'is_edit': is_edit
                            }
                            
                            try:
                                self.trade_queue.put_nowait(data)
                            except queue.Full:
                                pass
                                
                except Exception as e:
                    if self.stats.get('trade_errors', 0) < 5:
                        self.stats['trade_errors'] = self.stats.get('trade_errors', 0) + 1
                        self.logger.error(f"Erro no trade callback: {e}")
                        
            self.callback_refs['trade_v2'] = trade_callback_v2
            self.dll.SetTradeCallbackV2(self.callback_refs['trade_v2'])
            self.logger.info("✓ Trade callback V2 registrado")
            
        # 3. Asset List Callback
        if hasattr(self.dll, 'SetAssetListCallback'):
            @WINFUNCTYPE(None, TAssetID, c_wchar_p)
            def asset_list_callback(assetId, strName):
                timestamp = datetime.now().strftime('%H:%M:%S')
                self.logger.info(f"[{timestamp}] ASSET: {assetId.ticker} - {strName}")
                
            self.callback_refs['asset_list'] = asset_list_callback
            self.dll.SetAssetListCallback(self.callback_refs['asset_list'])
            self.logger.info("✓ Asset list callback registrado")
            
        # 4. Offer Book Callback V2 (complexo como no exemplo)
        if hasattr(self.dll, 'SetOfferBookCallbackV2'):
            @WINFUNCTYPE(None, TAssetID, c_int, c_int, c_int, c_int, c_int, 
                        c_longlong, c_double, c_int, c_int, c_int, c_int, c_int,
                        c_wchar_p, POINTER(c_ubyte), POINTER(c_ubyte))
            def offer_book_callback_v2(assetId, nAction, nPosition, Side, nQtd, nAgent, 
                                     nOfferID, sPrice, bHasPrice, bHasQtd, bHasDate, 
                                     bHasOfferID, bHasAgent, date, pArraySell, pArrayBuy):
                self.stats['book_callbacks'] += 1
                
                try:
                    ticker = assetId.ticker if assetId.ticker else "N/A"
                    
                    if self.stats['book_callbacks'] <= 10:
                        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        side_str = "BID" if Side == 0 else "ASK"
                        self.logger.info(f"[{timestamp}] BOOK: {ticker} | {side_str} | "
                                       f"Pos: {nPosition} | R$ {sPrice:.2f} | Qtd: {nQtd}")
                        
                    # Adicionar à fila
                    data = {
                        'type': 'book',
                        'ticker': ticker,
                        'action': nAction,
                        'position': nPosition,
                        'side': 'bid' if Side == 0 else 'ask',
                        'price': sPrice,
                        'quantity': nQtd,
                        'agent': nAgent,
                        'timestamp': datetime.now()
                    }
                    
                    try:
                        self.book_queue.put_nowait(data)
                    except queue.Full:
                        pass
                        
                except Exception as e:
                    if self.stats.get('book_errors', 0) < 5:
                        self.stats['book_errors'] = self.stats.get('book_errors', 0) + 1
                        self.logger.error(f"Erro no book callback: {e}")
                        
            self.callback_refs['offer_book_v2'] = offer_book_callback_v2
            result = self.dll.SetOfferBookCallbackV2(self.callback_refs['offer_book_v2'])
            self.logger.info(f"✓ Offer Book V2 callback registrado (result: {result})")
            
    def _do_login(self):
        """Faz login usando a função completa como no exemplo"""
        try:
            # Callbacks vazios para parâmetros não usados
            empty_callback = None
            
            # Preparar parâmetros
            key = self.config.get('key', 'HMARL')
            username = self.config['username']
            password = self.config['password']
            
            # Converter para wide strings
            key_w = c_wchar_p(key)
            user_w = c_wchar_p(username)
            pass_w = c_wchar_p(password)
            
            self.logger.info(f"Fazendo login: key={key}, user={username}")
            
            # Chamar DLLInitializeLogin com TODOS os parâmetros
            result = self.dll.DLLInitializeLogin(
                key_w,                              # key
                user_w,                             # username
                pass_w,                             # password
                self.callback_refs['state'],        # stateCallback
                empty_callback,                     # historyCallback
                empty_callback,                     # orderChangeCallback
                empty_callback,                     # accountCallback
                empty_callback,                     # assetListCallback
                empty_callback,                     # newDailyCallback
                empty_callback,                     # priceBookCallback
                empty_callback,                     # offerBookCallback
                empty_callback,                     # newHistoryCallback
                empty_callback,                     # progressCallback
                empty_callback                      # tinyBookCallback
            )
            
            self.logger.info(f"DLLInitializeLogin result: {result}")
            
            if result == 0:
                self.logger.info("✓ Login iniciado com sucesso!")
                
                # Aguardar callbacks
                timeout = 10
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    if self.market_connected:
                        self.logger.info("✓ Conectado ao market!")
                        return True
                        
                    time.sleep(0.1)
                    
                    # Log periódico
                    if int(time.time() - start_time) % 2 == 0:
                        self.logger.info(f"Aguardando conexão... Estados: {self.states_received}")
                        
                self.logger.warning("Timeout aguardando conexão")
                return True  # Retorna True mesmo assim, pois login foi bem sucedido
                
            else:
                self.logger.error(f"Falha no login: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro no login: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def subscribe_ticker(self, ticker, exchange=''):
        """Subscreve a um ticker"""
        try:
            ticker_w = c_wchar_p(ticker)
            exchange_w = c_wchar_p(exchange)
            
            result = self.dll.SubscribeTicker(ticker_w, exchange_w)
            
            if result == 0:
                self.logger.info(f"✓ Subscrito a {ticker} (exchange: {exchange})")
                return True
            else:
                self.logger.error(f"Erro ao subscrever {ticker}: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro na subscrição: {e}")
            return False
            
    def get_stats(self):
        """Retorna estatísticas"""
        return {
            'connected': self.is_connected,
            'market_connected': self.market_connected,
            'broker_connected': self.broker_connected,
            'active': self.is_active,
            'states': self.states_received,
            'total_callbacks': sum([
                self.stats.get(k, 0) for k in self.stats
                if k.endswith('_callbacks')
            ]),
            **self.stats
        }
        
    def stop(self):
        """Para o connection manager"""
        if hasattr(self.dll, 'DLLFinalize'):
            result = self.dll.DLLFinalize()
            self.logger.info(f"DLLFinalize: {result}")
            

def main():
    """Teste do connection manager corrigido"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("TESTE CONNECTION MANAGER CORRIGIDO")
    print("Baseado no exemplo oficial da ProfitDLL")
    print("="*60)
    
    config = {
        'dll_path': './ProfitDLL64.dll',
        'key': 'HMARL',
        'username': os.getenv('PROFIT_USERNAME', '29936354842'),
        'password': os.getenv('PROFIT_PASSWORD', 'Ultrajiu33!')
    }
    
    manager = ConnectionManagerFixed(config)
    
    try:
        if not manager.initialize():
            print("[ERRO] Falha na inicialização")
            return 1
            
        # Subscrever a tickers
        tickers = ['WDOQ25', 'WINQ25', 'PETR4', 'VALE3']
        
        print("\nSubscrevendo tickers...")
        for ticker in tickers:
            manager.subscribe_ticker(ticker)
            
        print("\n[OK] Sistema inicializado!")
        print("\nAguardando dados...")
        print("Pressione Ctrl+C para parar\n")
        
        last_stats_time = time.time()
        
        while True:
            time.sleep(5)
            
            if time.time() - last_stats_time > 10:
                stats = manager.get_stats()
                print(
                    f"[STATS] Conectado: {stats['connected']} | "
                    f"Market: {stats['market_connected']} | "
                    f"Total callbacks: {stats['total_callbacks']}"
                )
                
                # Detalhamento
                for key, value in stats.items():
                    if key.endswith('_callbacks') and value > 0:
                        print(f"        {key}: {value}")
                        
                last_stats_time = time.time()
                
    except KeyboardInterrupt:
        print("\n\nParando...")
    finally:
        manager.stop()
        
        stats = manager.get_stats()
        print("\n" + "="*60)
        print("ESTATÍSTICAS FINAIS:")
        print(f"Total de callbacks: {stats['total_callbacks']}")
        print(f"Estados recebidos: {stats['states']}")
        print("="*60)
        
    return 0


if __name__ == "__main__":
    sys.exit(main())