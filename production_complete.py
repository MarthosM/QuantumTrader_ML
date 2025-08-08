"""
Sistema de Produção Completo - QuantumTrader ML
Baseado no book_collector_continuous.py funcional
Com ML, gestão de risco e monitor integrado
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

# Configurar logging
log_file = f'logs/production/complete_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
Path("logs/production").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('CompleteTradingSystem')

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

class TOfferBookInfo(Structure):
    _fields_ = [
        ("price", c_double),
        ("qtd", c_int32),
        ("nOrders", c_int32),
        ("side", c_int32),
        ("datetime", c_wchar * 25),
        ("position", c_int32),
        ("sinalPr", c_int32)
    ]

class TDailyCallback(Structure):
    _fields_ = [
        ("assetId", TAssetIDRec),
        ("date", c_wchar * 20),
        ("sOpen", c_double),
        ("sHigh", c_double), 
        ("sLow", c_double),
        ("sClose", c_double),
        ("sVol", c_double),
        ("sAjuste", c_double),
        ("sMaxLimit", c_double),
        ("sMinLimit", c_double),
        ("sVolBuyer", c_double),
        ("sVolSeller", c_double),
        ("nQtd", c_int),
        ("nNegocios", c_int),
        ("nContratosOpen", c_int),
        ("nQtdBuyer", c_int),
        ("nQtdSeller", c_int),
        ("nNegBuyer", c_int),
        ("nNegSeller", c_int)
    ]

class CompleteProductionSystem:
    def __init__(self):
        self.dll = None
        self.logger = logger
        
        # Flags de controle (como no book_collector)
        self.bAtivo = False
        self.bMarketConnected = False
        self.bConnectado = False
        self.bBrokerConnected = False
        self.is_running = False
        
        # Callbacks
        self.callback_refs = {}
        self.callback_counters = {
            'state': 0,
            'history': 0,
            'daily': 0,
            'price_book': 0,
            'offer_book': 0,
            'progress': 0,
            'tiny_book': 0
        }
        
        # Market data
        self.current_price = 0
        self.bid_price = 0
        self.ask_price = 0
        self.last_price_update = 0
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
            'price_updates': 0,
            'last_price': 0
        }
        
        # Monitor process
        self.monitor_process = None
        
        # Ticker alvo
        self.target_ticker = "WDOU25"
        
    def start_monitor(self):
        """Inicia o monitor GUI em processo separado"""
        try:
            self.logger.info("Iniciando monitor GUI...")
            
            # Verificar qual monitor usar
            monitor_type = os.getenv('MONITOR_TYPE', '1')
            
            if monitor_type == '2':
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
        """Inicializa sistema como no book_collector"""
        try:
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
                
            self.logger.info(f"Carregando DLL: {os.path.abspath(dll_path)}")
            
            self.dll = WinDLL(dll_path)
            self.logger.info("[OK] DLL carregada")
            
            # Criar TODOS os callbacks ANTES do login
            self._create_all_callbacks()
            
            # Login com callbacks como no book_collector
            key = c_wchar_p("HMARL")
            user = c_wchar_p(os.getenv('PROFIT_USERNAME', ''))
            pwd = c_wchar_p(os.getenv('PROFIT_PASSWORD', ''))
            
            self.logger.info("Fazendo login com callbacks completos...")
            
            # DLLInitializeLogin exatamente como no book_collector
            result = self.dll.DLLInitializeLogin(
                key, user, pwd,
                self.callback_refs['state'],         # stateCallback
                self.callback_refs['history'],       # historyCallback  
                None,                                # orderChangeCallback
                None,                                # accountCallback
                None,                                # accountInfoCallback
                self.callback_refs['daily'],         # newDailyCallback
                self.callback_refs['price_book'],    # priceBookCallback
                self.callback_refs['offer_book'],    # offerBookCallback
                None,                                # historyTradeCallback
                self.callback_refs['progress'],      # progressCallBack
                self.callback_refs['tiny_book']      # tinyBookCallBack
            )
            
            if result != 0:
                self.logger.error(f"Erro no login: {result}")
                return False
                
            self.logger.info(f"[OK] Login bem sucedido: {result}")
            
            # Aguardar conexão completa
            if not self._wait_login():
                self.logger.error("Timeout aguardando conexão")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Erro: {e}")
            self.logger.error(traceback.format_exc())
            return False
            
    def _wait_login(self):
        """Aguarda conexão completa como no book_collector"""
        timeout = 30
        start = time.time()
        
        while (time.time() - start) < timeout:
            if self.bMarketConnected and self.bAtivo and self.bConnectado:
                self.logger.info(">>> SISTEMA TOTALMENTE CONECTADO <<<")
                return True
                
            # Log progresso
            if int(time.time() - start) % 5 == 0:
                self.logger.info(f"Aguardando... Market: {self.bMarketConnected}, Ativo: {self.bAtivo}, Login: {self.bConnectado}")
                
            time.sleep(0.1)
            
        return False
            
    def _load_ml_models(self):
        """Carrega todos os modelos ML disponíveis"""
        try:
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
                    
                except Exception as e:
                    self.logger.error(f"Erro ao carregar {model_file}: {e}")
                    
            self.logger.info(f"Total de modelos carregados: {len(self.models)}")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelos: {e}")
            
    def _create_all_callbacks(self):
        """Cria TODOS os callbacks como no book_collector"""
        
        # State callback - CRÍTICO
        @WINFUNCTYPE(None, c_int32, c_int32)
        def stateCallback(nType, nResult):
            self.callback_counters['state'] += 1
            
            states = {0: "Login", 1: "Broker", 2: "Market", 3: "Ativacao"}
            self.logger.info(f"[STATE] {states.get(nType, f'Type{nType}')}: {nResult}")
            
            if nType == 0:  # Login
                self.bConnectado = (nResult == 0)
            elif nType == 1:  # Broker
                self.bBrokerConnected = (nResult == 5)
            elif nType == 2:  # Market
                self.bMarketConnected = (nResult == 4 or nResult == 3 or nResult == 2)
            elif nType == 3:  # Ativacao
                self.bAtivo = (nResult == 0)
                
            if self.bMarketConnected and self.bAtivo and self.bConnectado:
                self.logger.info(">>> SISTEMA TOTALMENTE CONECTADO <<<")
                
            return None
            
        self.callback_refs['state'] = stateCallback
        
        # TinyBook callback
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_double, c_int, c_int)
        def tinyBookCallBack(assetId, price, qtd, side):
            self.callback_counters['tiny_book'] += 1
            
            # Validar preço
            if price > 0 and price < 10000:
                self.current_price = float(price)
                self.last_price_update = time.time()
                self.stats['price_updates'] += 1
                
                # Log a cada 100 ou mudança significativa
                if self.callback_counters['tiny_book'] % 100 == 0 or abs(price - self.stats['last_price']) > 1:
                    side_str = "BID" if side == 0 else "ASK"
                    self.logger.debug(f'[TINY #{self.callback_counters["tiny_book"]:,}] {side_str}: R$ {price:.2f} x {qtd}')
                    self.stats['last_price'] = price
                        
            return None
            
        self.callback_refs['tiny_book'] = tinyBookCallBack
        
        # PriceBook callback - Para book detalhado
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_int32, c_int32, c_int32, POINTER(TOfferBookInfo))
        def priceBookCallback(assetId, side, nAction, nPosition, pBook):
            self.callback_counters['price_book'] += 1
            
            try:
                if pBook and pBook.contents.price > 0:
                    price = float(pBook.contents.price)
                    qtd = int(pBook.contents.qtd)
                    
                    if side == 0:  # Compra
                        if self.bid_price != price:
                            self.bid_price = price
                            self.logger.debug(f"[BOOK] Novo BID: {price:.2f} x {qtd}")
                    else:  # Venda
                        if self.ask_price != price:
                            self.ask_price = price
                            self.logger.debug(f"[BOOK] Novo ASK: {price:.2f} x {qtd}")
                            
            except Exception as e:
                self.logger.error(f"Erro no priceBookCallback: {e}")
                
            return None
            
        self.callback_refs['price_book'] = priceBookCallback
        
        # OfferBook callback - Para ofertas agregadas
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_int32, c_wchar_p)
        def offerBookCallback(assetId, side, strJson):
            self.callback_counters['offer_book'] += 1
            
            try:
                if strJson:
                    data = json.loads(strJson)
                    
                    if side == 0:  # Compra
                        self.book_bids = data.get('levels', [])[:5]
                    else:  # Venda
                        self.book_asks = data.get('levels', [])[:5]
                        
                    # Log primeira vez
                    if self.callback_counters['offer_book'] == 1:
                        self.logger.info(f"[OFFER_BOOK] Recebendo dados de book")
                        
            except Exception as e:
                self.logger.error(f"Erro no offerBookCallback: {e}")
                
            return None
            
        self.callback_refs['offer_book'] = offerBookCallback
        
        # Daily callback - Dados agregados
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_wchar_p, c_double, c_double, 
                    c_double, c_double, c_double, c_double, c_double, c_double, 
                    c_double, c_double, c_int, c_int, c_int, c_int, c_int, c_int, c_int)
        def dailyCallback(assetId, date, sOpen, sHigh, sLow, sClose, sVol, sAjuste, 
                         sMaxLimit, sMinLimit, sVolBuyer, sVolSeller, nQtd, nNegocios, 
                         nContratosOpen, nQtdBuyer, nQtdSeller, nNegBuyer, nNegSeller):
            
            self.callback_counters['daily'] += 1
            
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
            self.logger.info(f"[DAILY #{self.callback_counters['daily']}] OHLC: {sOpen:.2f}/{sHigh:.2f}/{sLow:.2f}/{sClose:.2f} Vol: {sVol}")
            
            # Manter apenas últimos 100
            if len(self.candles) > 100:
                self.candles.pop(0)
                
            return None
                
        self.callback_refs['daily'] = dailyCallback
        
        # History callback
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_wchar_p, c_double, c_double, 
                    c_double, c_double, c_double, c_double, c_double, c_double, 
                    c_double, c_double, c_int, c_int, c_int, c_int, c_int, c_int, c_int)
        def historyCallback(assetId, date, sOpen, sHigh, sLow, sClose, sVol, sAjuste, 
                           sMaxLimit, sMinLimit, sVolBuyer, sVolSeller, nQtd, nNegocios, 
                           nContratosOpen, nQtdBuyer, nQtdSeller, nNegBuyer, nNegSeller):
            self.callback_counters['history'] += 1
            
            # Processar histórico como candle
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
            
            if self.callback_counters['history'] % 10 == 0:
                self.logger.debug(f"[HISTORY] Recebidos {self.callback_counters['history']} candles históricos")
                
            return None
            
        self.callback_refs['history'] = historyCallback
        
        # Progress callback
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_int32)
        def progressCallBack(assetId, nProgress):
            self.callback_counters['progress'] += 1
            
            if nProgress == 100:
                self.logger.info(f"[PROGRESS] Download histórico completo")
            elif nProgress % 25 == 0:
                self.logger.debug(f"[PROGRESS] {nProgress}%")
                
            return None
            
        self.callback_refs['progress'] = progressCallBack
        
        self.logger.info(f"[OK] {len(self.callback_refs)} callbacks criados")
        
    def subscribe_ticker(self, ticker):
        """Subscreve ticker"""
        try:
            result = self.dll.SubscribeTicker(
                c_wchar_p(ticker), 
                c_wchar_p("F")
            )
            
            if result == 0:
                self.logger.info(f"[OK] Subscrito a {ticker}")
                self.target_ticker = ticker
                return True
            else:
                self.logger.error(f"Erro ao subscrever {ticker}: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro: {e}")
            return False
            
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
                
            # Features de microestrutura
            if self.bid_price > 0 and self.ask_price > 0:
                features['spread'] = self.ask_price - self.bid_price
                features['mid_price'] = (self.bid_price + self.ask_price) / 2
                features['price_position'] = (self.current_price - self.bid_price) / (self.ask_price - self.bid_price) if self.ask_price > self.bid_price else 0.5
            else:
                features['spread'] = 0.5
                features['mid_price'] = features['price_current']
                features['price_position'] = 0.5
                
            return features
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular features: {e}")
            return None
            
    def _make_ensemble_prediction(self):
        """Faz predição usando ensemble de modelos"""
        if not self.models:
            return None
            
        try:
            features = self._calculate_features()
            if not features:
                return None
                
            predictions = []
            confidences = []
            
            # Fazer predição com cada modelo
            for model_name, model in self.models.items():
                try:
                    # Criar vetor de features
                    feature_list = self.features_lists.get(model_name, [])
                    if not feature_list:
                        continue
                        
                    feature_vector = []
                    for feat_name in feature_list:
                        value = features.get(feat_name, 0)
                        feature_vector.append(value)
                        
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
            
            return {
                'direction': weighted_pred,
                'confidence': avg_conf,
                'models_used': len(predictions)
            }
            
        except Exception as e:
            self.logger.error(f"Erro na predição ensemble: {e}")
            return None
            
    def run_strategy(self):
        """Estratégia com ML e gestão de risco"""
        self.logger.info("[STRATEGY] Iniciando estratégia ML com gestão de risco")
        self.logger.info("[MONITOR] Dashboard disponível para acompanhamento visual")
        
        last_prediction_time = 0
        last_status_time = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Verificar se está recebendo dados
                if self.last_price_update > 0:
                    data_age = current_time - self.last_price_update
                    if data_age > 10:  # Sem dados há 10 segundos
                        self.logger.warning(f"[DATA] Sem atualizações há {data_age:.1f}s")
                
                # Verificar limite diário
                if self.daily_pnl <= self.max_daily_loss:
                    self.logger.warning(f"[RISK] Limite diário atingido: R$ {self.daily_pnl:.2f}")
                    time.sleep(60)
                    continue
                    
                # Predição a cada 30 segundos
                if (current_time - last_prediction_time) > 30 and len(self.candles) >= 20:
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
                    
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Erro: {e}")
                time.sleep(5)
                
    def _manage_position(self, direction, confidence):
        """Gerencia posição baseado em predição"""
        # Thresholds
        entry_confidence = 0.65
        entry_direction_long = 0.65
        entry_direction_short = 0.35
        exit_confidence = 0.55
        
        if self.position == 0:
            # Entrada
            if confidence > entry_confidence:
                if direction > entry_direction_long:
                    self._send_order(1)  # Compra
                elif direction < entry_direction_short:
                    self._send_order(-1)  # Venda
                    
        else:
            # Saída
            if confidence > exit_confidence:
                if (self.position > 0 and direction < 0.4) or \
                   (self.position < 0 and direction > 0.6):
                    self._send_order(-self.position)  # Fecha posição
                    
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
        
        # TODO: Implementar envio real via DLL
        
        # Calcular P&L se fechando
        if self.position != 0 and (self.position + side) == 0:
            pnl = (self.current_price - self.entry_price) * self.position * 5  # Multiplicador
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
        """Log de status com informações completas"""
        elapsed = (time.time() - self.stats['start_time']) / 60
        win_rate = self.stats['wins'] / (self.stats['wins'] + self.stats['losses']) if (self.stats['wins'] + self.stats['losses']) > 0 else 0
        
        self.logger.info(f"\n[STATUS] {elapsed:.1f}min | Price: {self.current_price:.2f} | Pos: {self.position}")
        self.logger.info(f"Trades: {self.stats['trades']} | Win Rate: {win_rate:.1%} | P&L: R$ {self.stats['total_pnl']:.2f}")
        self.logger.info(f"Updates: {self.stats['price_updates']} | Bid: {self.bid_price:.2f} | Ask: {self.ask_price:.2f}")
        
        # Log callbacks
        callback_summary = ", ".join([f"{k}: {v}" for k, v in self.callback_counters.items() if v > 0])
        self.logger.info(f"Callbacks: {callback_summary}")
        
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
            except:
                pass

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
    print("QUANTUM TRADER ML - SISTEMA COMPLETO DE PRODUÇÃO")
    print("="*60)
    print(f"Data: {datetime.now()}")
    print("Conta: SIMULADOR")
    print("="*60)
    print("\nParâmetros de Risco:")
    print("- Máximo 1 contrato por vez")
    print("- Stop Loss: 0.5%")
    print("- Take Profit: 1%")
    print("- Limite diário: R$ 500")
    print("="*60)
    
    # Monitor configurado via ambiente
    monitor_type = os.getenv('MONITOR_TYPE', '1')
    print(f"\nMonitor: {'Web' if monitor_type == '2' else 'GUI Desktop'}")
    
    try:
        # Criar sistema
        system = CompleteProductionSystem()
        
        # Inicializar
        if not system.initialize():
            print("\nERRO: Falha na inicialização")
            return 1
            
        # Aguardar estabilização
        print("\nSistema inicializado. Aguardando dados...")
        time.sleep(3)
        
        # Subscrever
        ticker = "WDOU25"
        if not system.subscribe_ticker(ticker):
            print(f"\nERRO: Falha ao subscrever {ticker}")
            return 1
            
        # Aguardar dados
        time.sleep(5)
        
        # Verificar recepção de dados
        if system.callback_counters['tiny_book'] == 0:
            print("\nAVISO: Não está recebendo dados de preço")
            print("Verifique se o mercado está aberto")
        else:
            print(f"\nRecebendo dados: {system.callback_counters['tiny_book']} atualizações")
        
        # Iniciar estratégia
        if not system.start():
            return 1
            
        models_info = f"{len(system.models)} modelos" if system.models else "SEM MODELOS"
        print("\n" + "="*60)
        print(f"SISTEMA OPERACIONAL - {models_info}")
        print("Monitor: ATIVO")
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
            for cb_type, count in system.callback_counters.items():
                if count > 0:
                    print(f"  {cb_type}: {count:,}")
            print("="*60)
            
        print(f"\nLogs: {log_file}")

if __name__ == "__main__":
    sys.exit(main())