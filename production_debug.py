"""
Script de Produção com Debug Completo - QuantumTrader ML
Sistema completo com diagnóstico detalhado
"""

import os
import sys
import time
import ctypes
from ctypes import *
from datetime import datetime
from pathlib import Path
import logging
import threading
import signal
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import joblib
import json
import subprocess
import webbrowser
import traceback

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging com mais detalhes
log_file = f'logs/production/debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
Path("logs/production").mkdir(parents=True, exist_ok=True)

# Criar logger com múltiplos handlers
logger = logging.getLogger('DebugTrading')
logger.setLevel(logging.DEBUG)

# Handler para arquivo - DEBUG
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
file_handler.setFormatter(file_formatter)

# Handler para console - INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Estruturas ProfitDLL
class TAssetIDRec(Structure):
    _fields_ = [
        ("ticker", c_wchar * 35),
        ("bolsa", c_wchar * 15),
    ]

class TAssetListInfoRec(Structure):
    _fields_ = [
        ("ticker", c_wchar * 35),
        ("bolsa", c_wchar * 15),
        ("descricao", c_wchar * 255),
        ("tipo", c_int),
        ("lote_padrao", c_int),
        ("decimais", c_int),
    ]

class TAccountInfoRec(Structure):
    _fields_ = [
        ("corretora", c_int32),
        ("corretoraDigito", c_int32),
        ("titular", c_wchar * 100),
        ("titularDigito", c_int32),
        ("tipo_conta", c_int32),
        ("digito_tipo_conta", c_int32),
        ("moeda", c_wchar * 3),
        ("status", c_int32),
    ]

class TNewBookInfo(Structure):
    _fields_ = [
        ("price", c_double),
        ("qtd", c_int32),
        ("sinalPr", c_int32)
    ]

class DebugTradingSystem:
    def __init__(self):
        self.dll = None
        self.logger = logger
        
        # Debug counters
        self.debug_counters = {
            'state_callbacks': 0,
            'price_callbacks': 0,
            'book_callbacks': 0,
            'tiny_book_callbacks': 0,
            'daily_callbacks': 0,
            'trade_callbacks': 0,
            'account_callbacks': 0,
            'errors': 0
        }
        
        # Flags
        self.bMarketConnected = False
        self.bConnectado = False
        self.is_running = False
        
        # Callbacks
        self.callback_refs = {}
        
        # Market data
        self.current_price = 0
        self.bid_price = 0
        self.ask_price = 0
        self.last_price_update = 0
        self.price_updates_count = 0
        self.candles = []
        
        # Book data
        self.book_bids = []
        self.book_asks = []
        
        # Trading
        self.position = 0
        self.entry_price = 0
        self.daily_pnl = 0
        self.max_daily_loss = -500  # R$ 500
        
        # ML
        self.models = {}
        self.features_lists = {}
        
        # Risk parameters
        self.max_position = 1
        self.stop_loss_pct = 0.005  # 0.5%
        self.take_profit_pct = 0.01  # 1%
        
        # Stats
        self.stats = {
            'start_time': time.time(),
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'predictions': 0,
            'total_pnl': 0,
            'price_updates': 0
        }
        
        # Monitor process
        self.monitor_process = None
        
        # Debug mode
        self.debug_mode = True
        
    def start_monitor(self):
        """Inicia o monitor GUI em processo separado"""
        try:
            self.logger.info("Iniciando monitor GUI...")
            
            # Verificar qual monitor usar
            use_web_monitor = os.getenv('USE_WEB_MONITOR', 'false').lower() == 'true'
            
            if use_web_monitor:
                # Iniciar monitor web
                self.monitor_process = subprocess.Popen(
                    [sys.executable, "monitor_web.py"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
                )
                
                # Aguardar servidor iniciar
                time.sleep(3)
                
                # Abrir navegador
                self.logger.info("Abrindo monitor web em http://localhost:5000")
                webbrowser.open("http://localhost:5000")
            else:
                # Iniciar monitor GUI desktop
                self.monitor_process = subprocess.Popen(
                    [sys.executable, "monitor_gui.py"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
                )
                
            self.logger.info("[OK] Monitor iniciado")
            
        except Exception as e:
            self.logger.warning(f"Não foi possível iniciar o monitor: {e}")
            self.logger.debug(traceback.format_exc())
            self.logger.info("Sistema continuará sem interface gráfica")
            
    def stop_monitor(self):
        """Para o monitor se estiver rodando"""
        if self.monitor_process:
            try:
                self.monitor_process.terminate()
                self.logger.info("Monitor finalizado")
            except:
                pass
        
    def initialize(self):
        """Inicializa sistema com debug detalhado"""
        try:
            self.logger.info("=== INICIALIZANDO SISTEMA ===")
            
            # Iniciar monitor primeiro
            self.start_monitor()
            
            # Aguardar monitor carregar
            time.sleep(2)
            
            # Carregar modelos ML
            self._load_ml_models()
            
            # Carregar DLL
            dll_path = os.getenv('PROFIT_DLL_PATH', './ProfitDLL64.dll')
            if not os.path.exists(dll_path):
                dll_path = './ProfitDLL64.dll'
                
            self.logger.info(f"Carregando DLL: {dll_path}")
            self.logger.debug(f"DLL existe: {os.path.exists(dll_path)}")
            self.logger.debug(f"DLL tamanho: {os.path.getsize(dll_path) if os.path.exists(dll_path) else 0}")
            
            self.dll = WinDLL(dll_path)
            self.logger.info("[OK] DLL carregada")
            
            # Listar funções disponíveis na DLL (debug)
            self._debug_dll_functions()
            
            # Criar callbacks
            self._create_callbacks()
            
            # Login
            key = c_wchar_p("HMARL")
            user = c_wchar_p(os.getenv('PROFIT_USERNAME', ''))
            pwd = c_wchar_p(os.getenv('PROFIT_PASSWORD', ''))
            
            self.logger.debug(f"Tentando login com user: {os.getenv('PROFIT_USERNAME', '')[:5]}...")
            
            # Tentar login com diferentes combinações de callbacks
            result = self._try_login_combinations(key, user, pwd)
            
            if result != 0:
                self.logger.error(f"Erro no login: {result}")
                return False
                
            self.logger.info("[OK] Login bem sucedido")
            
            # Aguardar conexão
            timeout = 15
            start = time.time()
            while (time.time() - start) < timeout:
                if self.bMarketConnected:
                    self.logger.info("[OK] Conectado ao mercado")
                    self._log_debug_status()
                    return True
                time.sleep(0.1)
                
            self.logger.warning("Timeout aguardando conexão ao mercado")
            self._log_debug_status()
            return False
            
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {e}")
            self.logger.debug(traceback.format_exc())
            self.debug_counters['errors'] += 1
            return False
            
    def _debug_dll_functions(self):
        """Lista funções disponíveis na DLL para debug"""
        try:
            # Funções esperadas
            expected_functions = [
                'DLLInitializeLogin',
                'DLLFinalize',
                'SubscribeTicker',
                'UnsubscribeTicker',
                'GetHistory',
                'SendOrder',
                'CancelOrder',
                'SubscribeMarketBook',
                'UnsubscribeMarketBook',
                'GetAccount',
                'GetPosition'
            ]
            
            for func_name in expected_functions:
                try:
                    func = getattr(self.dll, func_name)
                    self.logger.debug(f"[DLL] Função '{func_name}' disponível")
                except AttributeError:
                    self.logger.warning(f"[DLL] Função '{func_name}' NÃO encontrada")
                    
        except Exception as e:
            self.logger.debug(f"Erro ao listar funções DLL: {e}")
            
    def _try_login_combinations(self, key, user, pwd):
        """Tenta diferentes combinações de callbacks para login"""
        # Primeiro tentar com callbacks mínimos (que funcionou antes)
        self.logger.debug("Tentando login com callbacks mínimos...")
        
        try:
            result = self.dll.DLLInitializeLogin(
                key, user, pwd,
                self.callback_refs['state'],
                None, None, None, None,
                self.callback_refs['daily'],
                None, None, None, None,
                self.callback_refs['tiny_book']
            )
            
            if result == 0:
                self.logger.debug("Login bem sucedido com callbacks mínimos")
                return result
                
        except Exception as e:
            self.logger.debug(f"Erro no login mínimo: {e}")
            
        # Tentar com mais callbacks
        self.logger.debug("Tentando login com callbacks expandidos...")
        
        try:
            result = self.dll.DLLInitializeLogin(
                key, user, pwd,
                self.callback_refs['state'],
                self.callback_refs['account'],
                None, None, None,
                self.callback_refs['daily'],
                self.callback_refs['price'],
                self.callback_refs['trade'],
                self.callback_refs['book'],
                None,
                self.callback_refs['tiny_book']
            )
            
            if result == 0:
                self.logger.debug("Login bem sucedido com callbacks expandidos")
                return result
                
        except Exception as e:
            self.logger.debug(f"Erro no login expandido: {e}")
            
        return -1
            
    def _load_ml_models(self):
        """Carrega todos os modelos ML disponíveis"""
        try:
            self.logger.info("=== CARREGANDO MODELOS ML ===")
            
            models_dir = Path('models')
            if not models_dir.exists():
                self.logger.warning("Diretório de modelos não encontrado")
                return
                
            # Carregar cada modelo .pkl
            for model_file in models_dir.glob('*.pkl'):
                try:
                    model_name = model_file.stem
                    
                    # Pular scalers
                    if 'scaler' in model_name.lower():
                        continue
                        
                    self.logger.info(f"Carregando modelo: {model_name}")
                    
                    # Carregar modelo
                    model = joblib.load(model_file)
                    self.models[model_name] = model
                    
                    # Carregar features
                    features_file = model_file.with_suffix('.json')
                    if features_file.exists():
                        with open(features_file) as f:
                            data = json.load(f)
                            self.features_lists[model_name] = data.get('features', [])
                            
                    self.logger.info(f"[OK] {model_name}: {len(self.features_lists.get(model_name, []))} features")
                    self.logger.debug(f"Features: {self.features_lists.get(model_name, [])[:5]}...")
                    
                except Exception as e:
                    self.logger.error(f"Erro ao carregar {model_file}: {e}")
                    self.logger.debug(traceback.format_exc())
                    
            self.logger.info(f"Total de modelos carregados: {len(self.models)}")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelos: {e}")
            self.logger.debug(traceback.format_exc())
            
    def _create_callbacks(self):
        """Cria callbacks com debug detalhado"""
        self.logger.info("=== CRIANDO CALLBACKS ===")
        
        # State callback
        @WINFUNCTYPE(None, c_int32, c_int32)
        def stateCallback(nType, nResult):
            try:
                self.debug_counters['state_callbacks'] += 1
                
                if nType == 0:  # Login
                    self.bConnectado = (nResult == 0)
                    self.logger.info(f"[STATE] Login - Result: {nResult} - Conectado: {self.bConnectado}")
                elif nType == 2:  # Market
                    self.bMarketConnected = (nResult in [2, 3, 4])
                    self.logger.info(f"[STATE] Market - Result: {nResult} - Connected: {self.bMarketConnected}")
                else:
                    self.logger.debug(f"[STATE] Type={nType} Result={nResult}")
                    
            except Exception as e:
                self.logger.error(f"Erro no stateCallback: {e}")
                self.debug_counters['errors'] += 1
                
        self.callback_refs['state'] = stateCallback
        
        # Account callback
        @WINFUNCTYPE(None, POINTER(TAccountInfoRec))
        def accountCallback(accountInfo):
            try:
                self.debug_counters['account_callbacks'] += 1
                
                if accountInfo:
                    acc = accountInfo.contents
                    self.logger.info(f"[ACCOUNT] Corretora: {acc.corretora} | Titular: {acc.titular} | Status: {acc.status}")
                    
            except Exception as e:
                self.logger.error(f"Erro no accountCallback: {e}")
                self.debug_counters['errors'] += 1
                
        self.callback_refs['account'] = accountCallback
        
        # Price callback
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_int32, c_double, c_int32, c_int32, c_wchar_p)
        def priceCallback(assetId, date, price, volume, qtd, side, hora):
            try:
                self.debug_counters['price_callbacks'] += 1
                
                if price > 0 and price < 10000:
                    self.current_price = float(price)
                    self.last_price_update = time.time()
                    self.price_updates_count += 1
                    self.stats['price_updates'] += 1
                    
                    # Log detalhado a cada 10 atualizações
                    if self.debug_counters['price_callbacks'] % 10 == 0:
                        self.logger.debug(f"[PRICE] #{self.debug_counters['price_callbacks']} - Price: {price:.2f} vol={volume} qtd={qtd} side={side}")
                        
            except Exception as e:
                self.logger.error(f"Erro no priceCallback: {e}")
                self.debug_counters['errors'] += 1
                
        self.callback_refs['price'] = priceCallback
        
        # Trade callback
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_int32, c_double, c_int32, c_int32, c_wchar_p)
        def tradeCallback(assetId, date, price, volume, qtd, side, hora):
            try:
                self.debug_counters['trade_callbacks'] += 1
                
                if price > 0 and price < 10000:
                    # Atualizar último preço negociado
                    self.current_price = float(price)
                    
                    # Log trades grandes
                    if volume > 50:
                        side_str = 'BUY' if side == 0 else 'SELL'
                        self.logger.info(f"[TRADE] Big: {price:.2f} vol={volume} side={side_str}")
                        
            except Exception as e:
                self.logger.error(f"Erro no tradeCallback: {e}")
                self.debug_counters['errors'] += 1
                
        self.callback_refs['trade'] = tradeCallback
        
        # Book callback
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_int32, POINTER(TNewBookInfo), POINTER(TNewBookInfo))
        def bookCallback(assetId, lado, buyArray, sellArray):
            try:
                self.debug_counters['book_callbacks'] += 1
                
                # Debug primeira vez
                if self.debug_counters['book_callbacks'] == 1:
                    self.logger.info("[BOOK] Primeiro callback de book recebido!")
                
                # Atualizar book de ofertas
                self.book_bids = []
                self.book_asks = []
                
                # Processar bids
                for i in range(5):  # Top 5 níveis
                    if buyArray and buyArray[i].price > 0:
                        self.book_bids.append({
                            'price': float(buyArray[i].price),
                            'qtd': int(buyArray[i].qtd)
                        })
                        
                # Processar asks
                for i in range(5):  # Top 5 níveis
                    if sellArray and sellArray[i].price > 0:
                        self.book_asks.append({
                            'price': float(sellArray[i].price),
                            'qtd': int(sellArray[i].qtd)
                        })
                        
                # Atualizar bid/ask
                if self.book_bids:
                    self.bid_price = self.book_bids[0]['price']
                if self.book_asks:
                    self.ask_price = self.book_asks[0]['price']
                    
                # Log a cada 100 callbacks
                if self.debug_counters['book_callbacks'] % 100 == 0:
                    self.logger.debug(f"[BOOK] #{self.debug_counters['book_callbacks']} - Bid: {self.bid_price:.2f} Ask: {self.ask_price:.2f}")
                    
            except Exception as e:
                self.logger.error(f"Erro no bookCallback: {e}")
                self.logger.debug(traceback.format_exc())
                self.debug_counters['errors'] += 1
                
        self.callback_refs['book'] = bookCallback
        
        # TinyBook callback
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_double, c_int, c_int)
        def tinyBookCallBack(assetId, price, qtd, side):
            try:
                self.debug_counters['tiny_book_callbacks'] += 1
                
                if 0 < price < 10000:
                    self.current_price = float(price)
                    self.last_price_update = time.time()
                    
                    # Debug primeira vez
                    if self.debug_counters['tiny_book_callbacks'] == 1:
                        self.logger.info(f"[TINY_BOOK] Primeiro callback recebido! Price: {price:.2f}")
                        
                    # Log a cada 100
                    if self.debug_counters['tiny_book_callbacks'] % 100 == 0:
                        self.logger.debug(f"[TINY_BOOK] #{self.debug_counters['tiny_book_callbacks']} - Price: {price:.2f}")
                        
            except Exception as e:
                self.logger.error(f"Erro no tinyBookCallBack: {e}")
                self.debug_counters['errors'] += 1
                
        self.callback_refs['tiny_book'] = tinyBookCallBack
        
        # Daily callback
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_wchar_p, c_double, c_double, 
                    c_double, c_double, c_double, c_double, c_double, c_double, 
                    c_double, c_double, c_int, c_int, c_int, c_int, c_int, c_int, c_int)
        def dailyCallback(assetId, date, sOpen, sHigh, sLow, sClose, sVol, sAjuste, 
                         sMaxLimit, sMinLimit, sVolBuyer, sVolSeller, nQtd, nNegocios, 
                         nContratosOpen, nQtdBuyer, nQtdSeller, nNegBuyer, nNegSeller):
            try:
                self.debug_counters['daily_callbacks'] += 1
                
                # Adicionar candle
                candle = {
                    'timestamp': datetime.now(),
                    'open': float(sOpen),
                    'high': float(sHigh),
                    'low': float(sLow),
                    'close': float(sClose),
                    'volume': float(sVol),
                    'trades': int(nNegocios)
                }
                
                self.candles.append(candle)
                
                # Log
                self.logger.info(f"[DAILY] #{self.debug_counters['daily_callbacks']} - OHLC: {sOpen:.2f}/{sHigh:.2f}/{sLow:.2f}/{sClose:.2f} Vol: {sVol}")
                
                # Manter apenas últimos 100
                if len(self.candles) > 100:
                    self.candles.pop(0)
                    
            except Exception as e:
                self.logger.error(f"Erro no dailyCallback: {e}")
                self.debug_counters['errors'] += 1
                
        self.callback_refs['daily'] = dailyCallback
        
        self.logger.info(f"[OK] {len(self.callback_refs)} callbacks criados")
        
    def subscribe_ticker(self, ticker="WDOU25"):
        """Subscreve ticker com debug detalhado"""
        try:
            self.logger.info(f"=== SUBSCRIBING TO {ticker} ===")
            
            # Subscrever ticker principal
            result = self.dll.SubscribeTicker(
                c_wchar_p(ticker), 
                c_wchar_p("F")
            )
            
            self.logger.info(f"SubscribeTicker result: {result}")
            
            if result != 0:
                self.logger.error(f"Erro ao subscrever {ticker}: {result}")
                return False
                
            self.logger.info(f"[OK] Subscrito a {ticker}")
            
            # Tentar ativar recursos adicionais
            self._try_advanced_subscriptions(ticker)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro em subscribe_ticker: {e}")
            self.logger.debug(traceback.format_exc())
            return False
            
    def _try_advanced_subscriptions(self, ticker):
        """Tenta ativar recursos avançados de subscrição"""
        # Tentar ativar book em tempo real
        try:
            if hasattr(self.dll, 'SubscribeMarketBook'):
                result = self.dll.SubscribeMarketBook(c_wchar_p(ticker))
                self.logger.info(f"SubscribeMarketBook result: {result}")
            else:
                self.logger.debug("SubscribeMarketBook não disponível")
        except Exception as e:
            self.logger.debug(f"Erro em SubscribeMarketBook: {e}")
            
        # Tentar solicitar histórico
        try:
            if hasattr(self.dll, 'GetHistory'):
                result = self.dll.GetHistory(
                    c_wchar_p(ticker),
                    c_int(0),  # Periodicidade
                    c_int(100),  # Qtd barras
                    c_int(0),  # Ajuste
                    c_wchar_p("")  # Data inicial
                )
                self.logger.info(f"GetHistory result: {result}")
            else:
                self.logger.debug("GetHistory não disponível")
        except Exception as e:
            self.logger.debug(f"Erro em GetHistory: {e}")
            
        # Tentar ativar trades
        try:
            if hasattr(self.dll, 'SubscribeTrades'):
                result = self.dll.SubscribeTrades(c_wchar_p(ticker))
                self.logger.info(f"SubscribeTrades result: {result}")
            else:
                self.logger.debug("SubscribeTrades não disponível")
        except Exception as e:
            self.logger.debug(f"Erro em SubscribeTrades: {e}")
            
    def _log_debug_status(self):
        """Log detalhado do status de debug"""
        self.logger.info("\n=== DEBUG STATUS ===")
        self.logger.info(f"Callbacks recebidos:")
        for callback_type, count in self.debug_counters.items():
            if count > 0:
                self.logger.info(f"  {callback_type}: {count}")
        self.logger.info(f"Preço atual: {self.current_price:.2f}")
        self.logger.info(f"Bid/Ask: {self.bid_price:.2f}/{self.ask_price:.2f}")
        self.logger.info(f"Candles: {len(self.candles)}")
        self.logger.info(f"Última atualização: {time.time() - self.last_price_update:.1f}s atrás")
        self.logger.info("==================\n")
        
    def _calculate_features(self):
        """Calcula features para ML"""
        if len(self.candles) < 20:
            return None
            
        try:
            df = pd.DataFrame(self.candles)
            features = {}
            
            # Preços
            closes = df['close'].values
            features['price_current'] = self.current_price if self.current_price > 0 else closes[-1]
            features['price_mean_5'] = np.mean(closes[-5:])
            features['price_mean_20'] = np.mean(closes[-20:])
            features['price_std_20'] = np.std(closes[-20:])
            
            # Retornos
            returns = np.diff(closes) / closes[:-1]
            features['return_1'] = returns[-1] if len(returns) > 0 else 0
            features['return_mean_5'] = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            features['return_std_5'] = np.std(returns[-5:]) if len(returns) >= 5 else 0
            
            # Volume
            volumes = df['volume'].values
            features['volume_mean_5'] = np.mean(volumes[-5:])
            features['volume_ratio'] = volumes[-1] / features['volume_mean_5'] if features['volume_mean_5'] > 0 else 1
            
            # RSI
            gains = [r if r > 0 else 0 for r in returns[-14:]]
            losses = [-r if r < 0 else 0 for r in returns[-14:]]
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            features['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Momentum
            if len(closes) >= 10:
                features['momentum_10'] = (features['price_current'] / closes[-10]) - 1
            else:
                features['momentum_10'] = 0
                
            # Features de microestrutura (se disponível)
            if self.bid_price > 0 and self.ask_price > 0:
                features['spread'] = self.ask_price - self.bid_price
                features['mid_price'] = (self.bid_price + self.ask_price) / 2
                features['price_position'] = (self.current_price - self.bid_price) / (self.ask_price - self.bid_price) if self.ask_price > self.bid_price else 0.5
            else:
                # Valores padrão se não tiver book
                features['spread'] = 0.5
                features['mid_price'] = features['price_current']
                features['price_position'] = 0.5
                
            self.logger.debug(f"Features calculadas: {list(features.keys())}")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular features: {e}")
            self.logger.debug(traceback.format_exc())
            return None
            
    def _make_ensemble_prediction(self):
        """Faz predição usando ensemble de modelos"""
        if not self.models:
            return None
            
        try:
            features = self._calculate_features()
            if not features:
                self.logger.debug("Sem features disponíveis para predição")
                return None
                
            predictions = []
            confidences = []
            
            # Fazer predição com cada modelo
            for model_name, model in self.models.items():
                try:
                    # Criar vetor de features
                    feature_list = self.features_lists.get(model_name, [])
                    if not feature_list:
                        self.logger.debug(f"Modelo {model_name} sem lista de features")
                        continue
                        
                    feature_vector = []
                    missing_features = []
                    
                    for feat_name in feature_list:
                        if feat_name in features:
                            feature_vector.append(features[feat_name])
                        else:
                            feature_vector.append(0)
                            missing_features.append(feat_name)
                            
                    if missing_features and len(missing_features) < 5:
                        self.logger.debug(f"Features faltando para {model_name}: {missing_features}")
                        
                    X = np.array([feature_vector])
                    
                    # Predição
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)[0]
                        pred = proba[1]  # Probabilidade de alta
                        conf = max(proba)
                    else:
                        pred = model.predict(X)[0]
                        conf = abs(pred)
                        
                    predictions.append(pred)
                    confidences.append(conf)
                    
                    self.logger.debug(f"Predição {model_name}: dir={pred:.3f} conf={conf:.3f}")
                    
                except Exception as e:
                    self.logger.debug(f"Erro na predição {model_name}: {e}")
                    
            if not predictions:
                return None
                
            # Ensemble - média ponderada pela confiança
            total_conf = sum(confidences)
            if total_conf > 0:
                weighted_pred = sum(p * c for p, c in zip(predictions, confidences)) / total_conf
                avg_conf = np.mean(confidences)
            else:
                weighted_pred = np.mean(predictions)
                avg_conf = 0.5
                
            self.stats['predictions'] += 1
            
            result = {
                'direction': weighted_pred,
                'confidence': avg_conf,
                'models_used': len(predictions)
            }
            
            self.logger.debug(f"Ensemble result: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro na predição ensemble: {e}")
            self.logger.debug(traceback.format_exc())
            return None
            
    def run_strategy(self):
        """Estratégia com ML e gestão de risco"""
        self.logger.info("[STRATEGY] Iniciando estratégia ML com debug completo")
        self.logger.info("[MONITOR] Dashboard disponível para acompanhamento visual")
        
        last_prediction_time = 0
        last_status_time = 0
        last_debug_time = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Debug status a cada 30 segundos
                if (current_time - last_debug_time) > 30:
                    self._log_debug_status()
                    last_debug_time = current_time
                
                # Verificar se está recebendo dados
                if self.last_price_update > 0:
                    data_age = current_time - self.last_price_update
                    if data_age > 10:  # Sem dados há 10 segundos
                        self.logger.warning(f"[DATA] Sem atualizações há {data_age:.1f}s")
                        self.logger.debug("Verificar se mercado está aberto e ticker está correto")
                
                # Verificar limite diário
                if self.daily_pnl <= self.max_daily_loss:
                    self.logger.warning(f"[RISK] Limite diário atingido: R$ {self.daily_pnl:.2f}")
                    time.sleep(60)
                    continue
                    
                # Predição a cada 30 segundos
                if (current_time - last_prediction_time) > 30 and len(self.candles) >= 20:
                    self.logger.debug("Tentando fazer predição...")
                    prediction = self._make_ensemble_prediction()
                    
                    if prediction:
                        direction = prediction['direction']
                        confidence = prediction['confidence']
                        models = prediction['models_used']
                        
                        self.logger.info(f"\n[ML] Dir: {direction:.3f} | Conf: {confidence:.3f} | Models: {models}")
                        
                        # Gestão de posição
                        self._manage_position(direction, confidence)
                        
                    last_prediction_time = current_time
                    
                # Verificar stops
                if self.position != 0 and self.current_price > 0:
                    self._check_stops()
                    
                # Status a cada 60 segundos
                if (current_time - last_status_time) > 60:
                    self._log_status()
                    last_status_time = current_time
                    
                time.sleep(0.1)  # Loop rápido
                
            except Exception as e:
                self.logger.error(f"Erro no loop de estratégia: {e}")
                self.logger.debug(traceback.format_exc())
                self.debug_counters['errors'] += 1
                time.sleep(5)
                
    def _manage_position(self, direction, confidence):
        """Gerencia posição baseado em predição"""
        # Thresholds
        entry_confidence = 0.65
        entry_direction_long = 0.65
        entry_direction_short = 0.35
        exit_confidence = 0.55
        
        self.logger.debug(f"Gerenciando posição: pos={self.position} dir={direction:.3f} conf={confidence:.3f}")
        
        if self.position == 0:
            # Entrada
            if confidence > entry_confidence:
                if direction > entry_direction_long:
                    self.logger.debug("Sinal de COMPRA detectado")
                    self._send_order(1)  # Compra
                elif direction < entry_direction_short:
                    self.logger.debug("Sinal de VENDA detectado")
                    self._send_order(-1)  # Venda
                else:
                    self.logger.debug("Direção neutra, sem ação")
            else:
                self.logger.debug("Confiança insuficiente para entrada")
                
        else:
            # Saída
            if confidence > exit_confidence:
                if (self.position > 0 and direction < 0.4) or \
                   (self.position < 0 and direction > 0.6):
                    self.logger.debug("Sinal de SAÍDA detectado")
                    self._send_order(-self.position)  # Fecha posição
                else:
                    self.logger.debug("Mantendo posição")
            else:
                self.logger.debug("Confiança insuficiente para decisão")
                    
    def _check_stops(self):
        """Verifica stop loss e take profit"""
        if self.position == 0 or self.entry_price == 0:
            return
            
        pnl_pct = (self.current_price - self.entry_price) / self.entry_price * self.position
        
        # Stop loss
        if pnl_pct <= -self.stop_loss_pct:
            self.logger.warning(f"[STOP LOSS] Fechando posição com perda de {pnl_pct:.2%}")
            self._send_order(-self.position)
            
        # Take profit
        elif pnl_pct >= self.take_profit_pct:
            self.logger.info(f"[TAKE PROFIT] Fechando posição com lucro de {pnl_pct:.2%}")
            self._send_order(-self.position)
            
    def _send_order(self, side):
        """Envia ordem (simulada por enquanto)"""
        if side == 0 or abs(self.position + side) > self.max_position:
            return
            
        side_str = "COMPRA" if side > 0 else "VENDA"
        self.logger.info(f"\n[ORDER] {side_str} 1 @ {self.current_price:.2f}")
        
        # TODO: Implementar envio real via DLL quando pronto
        # Por enquanto apenas simula
        
        # Calcular P&L se fechando
        if self.position != 0 and (self.position + side) == 0:
            pnl = (self.current_price - self.entry_price) * self.position * 5  # Multiplicador do contrato
            self.daily_pnl += pnl
            self.stats['total_pnl'] += pnl
            
            if pnl > 0:
                self.stats['wins'] += 1
            else:
                self.stats['losses'] += 1
                
            self.logger.info(f"[P&L] Trade: R$ {pnl:.2f} | Diário: R$ {self.daily_pnl:.2f}")
            
        # Atualizar posição
        self.position += side
        if self.position != 0:
            self.entry_price = self.current_price
        else:
            self.entry_price = 0
            
        self.stats['trades'] += 1
        self.logger.info(f"[POSITION] {self.position} contratos")
        
    def _log_status(self):
        """Log de status"""
        elapsed = (time.time() - self.stats['start_time']) / 60
        win_rate = self.stats['wins'] / (self.stats['wins'] + self.stats['losses']) if (self.stats['wins'] + self.stats['losses']) > 0 else 0
        
        self.logger.info(f"\n[STATUS] {elapsed:.1f}min | Price: {self.current_price:.2f} | Pos: {self.position}")
        self.logger.info(f"Trades: {self.stats['trades']} | Win Rate: {win_rate:.1%} | P&L: R$ {self.stats['total_pnl']:.2f}")
        self.logger.info(f"Updates: {self.stats['price_updates']} | Bid: {self.bid_price:.2f} | Ask: {self.ask_price:.2f}")
        
    def start(self):
        """Inicia sistema"""
        self.is_running = True
        
        # Thread para estratégia
        strategy_thread = threading.Thread(target=self.run_strategy)
        strategy_thread.daemon = True
        strategy_thread.start()
        
        return True
        
    def stop(self):
        """Para sistema"""
        self.is_running = False
        
        # Log final de debug
        self._log_debug_status()
        
        # Fechar posições
        if self.position != 0:
            self.logger.info("Fechando posição...")
            self._send_order(-self.position)
            
        # Parar monitor
        self.stop_monitor()
            
    def cleanup(self):
        """Finaliza DLL"""
        if self.dll and hasattr(self.dll, 'DLLFinalize'):
            try:
                self.dll.DLLFinalize()
                self.logger.info("DLL finalizada")
            except Exception as e:
                self.logger.error(f"Erro ao finalizar DLL: {e}")

# Variável global
system = None

def signal_handler(signum, frame):
    """Handler para Ctrl+C"""
    global system
    print("\n\nFinalizando sistema...")
    if system:
        system.stop()
    sys.exit(0)

def main():
    global system
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "="*60)
    print("QUANTUM TRADER ML - PRODUÇÃO COM DEBUG")
    print("="*60)
    print(f"Data: {datetime.now()}")
    print("Conta: SIMULADOR")
    print("Modo: DEBUG COMPLETO")
    print("="*60)
    print("\nParâmetros de Risco:")
    print("- Máximo 1 contrato por vez")
    print("- Stop Loss: 0.5%")
    print("- Take Profit: 1%")
    print("- Limite diário: R$ 500")
    print("="*60)
    
    # Usar configuração do ambiente ou padrão
    monitor_choice = os.getenv('MONITOR_TYPE', '1')
    if monitor_choice == '2':
        os.environ['USE_WEB_MONITOR'] = 'true'
        print("\nUsando Monitor Web")
    else:
        os.environ['USE_WEB_MONITOR'] = 'false'
        print("\nUsando Monitor GUI Desktop")
    
    try:
        # Criar sistema
        system = DebugTradingSystem()
        
        # Inicializar (incluindo monitor)
        if not system.initialize():
            print("\nERRO: Falha na inicialização")
            print("Verifique os logs para detalhes")
            return 1
            
        # Aguardar
        print("\nSistema inicializado. Aguardando estabilização...")
        time.sleep(3)
        
        # Subscrever
        ticker = "WDOU25"  # TODO: Ajustar conforme mês
        print(f"\nSubscrevendo ao ticker {ticker}...")
        
        if not system.subscribe_ticker(ticker):
            print(f"\nERRO: Falha ao subscrever {ticker}")
            return 1
            
        # Aguardar dados
        print("\nAguardando dados do mercado...")
        time.sleep(5)
        
        # Verificar se está recebendo dados
        if system.debug_counters['tiny_book_callbacks'] == 0 and \
           system.debug_counters['price_callbacks'] == 0:
            print("\nAVISO: Não está recebendo dados de preço")
            print("Possíveis causas:")
            print("- Mercado fechado")
            print("- Ticker incorreto")
            print("- Problemas de conexão")
            print("\nContinuando mesmo assim...")
        
        # Iniciar
        if not system.start():
            return 1
            
        models_info = "COM ML" if system.models else "SEM ML"
        print("\n" + "="*60)
        print(f"SISTEMA OPERACIONAL - {models_info}")
        print(f"Modelos carregados: {len(system.models)}")
        print("Monitor: ATIVO")
        print("Modo: DEBUG")
        print("Para parar: CTRL+C")
        print("="*60)
        
        # Loop principal
        while system.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário")
        
    except Exception as e:
        print(f"\nERRO FATAL: {e}")
        logger.error(f"Erro fatal: {e}", exc_info=True)
        
    finally:
        if system:
            system.stop()
            system.cleanup()
            
        # Stats finais
        if system:
            runtime = (time.time() - system.stats['start_time']) / 60
            total_trades = system.stats['wins'] + system.stats['losses']
            win_rate = system.stats['wins'] / total_trades if total_trades > 0 else 0
            
            print("\n" + "="*60)
            print("ESTATÍSTICAS FINAIS")
            print("="*60)
            print(f"Tempo: {runtime:.1f} min")
            print(f"Trades: {system.stats['trades']}")
            print(f"Vitórias: {system.stats['wins']}")
            print(f"Derrotas: {system.stats['losses']}")
            print(f"Win Rate: {win_rate:.1%}")
            print(f"P&L Total: R$ {system.stats['total_pnl']:.2f}")
            print(f"Predições ML: {system.stats['predictions']}")
            print(f"Atualizações de preço: {system.stats['price_updates']}")
            print("\nCallbacks recebidos:")
            for cb_type, count in system.debug_counters.items():
                if count > 0:
                    print(f"  {cb_type}: {count}")
            print("="*60)
            
        print(f"\nLogs detalhados: {log_file}")
        print("\nVerifique os logs para análise completa")

if __name__ == "__main__":
    sys.exit(main())