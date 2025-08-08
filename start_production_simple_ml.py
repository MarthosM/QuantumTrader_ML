"""
Script de Produção Simples com ML
Versão minimalista mas funcional
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

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
log_file = f'logs/production/simple_ml_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
Path("logs/production").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('SimpleMLTrading')

# Estrutura TAssetIDRec
class TAssetIDRec(Structure):
    _fields_ = [
        ("ticker", c_wchar * 35),
        ("bolsa", c_wchar * 15),
    ]

class SimpleMLSystem:
    def __init__(self):
        self.dll = None
        self.logger = logger
        
        # Flags
        self.bMarketConnected = False
        self.bConnectado = False
        self.is_running = False
        
        # Callbacks
        self.callback_refs = {}
        
        # Market data
        self.current_price = 0
        self.candles = []
        
        # Trading
        self.position = 0
        self.entry_price = 0
        
        # ML
        self.model = None
        self.features_list = []
        
        # Stats
        self.stats = {
            'start_time': time.time(),
            'trades': 0,
            'predictions': 0,
            'pnl': 0
        }
        
    def initialize(self):
        """Inicializa sistema"""
        try:
            # Carregar modelo ML
            if not self._load_ml_model():
                self.logger.warning("Operando sem ML")
                
            # Carregar DLL
            dll_path = "./ProfitDLL64.dll"
            self.logger.info(f"Carregando DLL: {dll_path}")
            
            self.dll = WinDLL(dll_path)
            self.logger.info("[OK] DLL carregada")
            
            # Criar callbacks
            self._create_callbacks()
            
            # Login
            key = c_wchar_p("HMARL")
            user = c_wchar_p(os.getenv('PROFIT_USERNAME', ''))
            pwd = c_wchar_p(os.getenv('PROFIT_PASSWORD', ''))
            
            result = self.dll.DLLInitializeLogin(
                key, user, pwd,
                self.callback_refs['state'],
                None, None, None, None,
                self.callback_refs['daily'],
                None, None, None, None,
                self.callback_refs['tiny_book']
            )
            
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
                    return True
                time.sleep(0.1)
                
            return False
            
        except Exception as e:
            self.logger.error(f"Erro: {e}")
            return False
            
    def _load_ml_model(self):
        """Carrega modelo ML mais simples"""
        try:
            # Tentar carregar o primeiro modelo disponível
            model_files = list(Path('models').glob('*.pkl'))
            if not model_files:
                return False
                
            model_file = model_files[0]
            self.logger.info(f"Carregando modelo: {model_file}")
            
            self.model = joblib.load(model_file)
            
            # Carregar features
            features_file = model_file.with_suffix('.json')
            if features_file.exists():
                import json
                with open(features_file) as f:
                    data = json.load(f)
                    self.features_list = data.get('features', [])
                    
            self.logger.info(f"[OK] Modelo carregado: {len(self.features_list)} features")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
            return False
            
    def _create_callbacks(self):
        """Cria callbacks mínimos"""
        
        @WINFUNCTYPE(None, c_int32, c_int32)
        def stateCallback(nType, nResult):
            if nType == 0:  # Login
                self.bConnectado = (nResult == 0)
            elif nType == 2:  # Market
                self.bMarketConnected = (nResult in [2, 3, 4])
            self.logger.info(f"[STATE] Type={nType} Result={nResult}")
            
        self.callback_refs['state'] = stateCallback
        
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_double, c_int, c_int)
        def tinyBookCallBack(assetId, price, qtd, side):
            if 0 < price < 10000:
                self.current_price = float(price)
                
        self.callback_refs['tiny_book'] = tinyBookCallBack
        
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_wchar_p, c_double, c_double, 
                    c_double, c_double, c_double, c_double, c_double, c_double, 
                    c_double, c_double, c_int, c_int, c_int, c_int, c_int, c_int, c_int)
        def dailyCallback(assetId, date, sOpen, sHigh, sLow, sClose, sVol, sAjuste, 
                         sMaxLimit, sMinLimit, sVolBuyer, sVolSeller, nQtd, nNegocios, 
                         nContratosOpen, nQtdBuyer, nQtdSeller, nNegBuyer, nNegSeller):
            
            # Adicionar candle
            self.candles.append({
                'timestamp': datetime.now(),
                'close': float(sClose),
                'volume': float(sVol),
                'trades': int(nNegocios)
            })
            
            # Manter apenas últimos 100
            if len(self.candles) > 100:
                self.candles.pop(0)
                
        self.callback_refs['daily'] = dailyCallback
        
    def subscribe_ticker(self, ticker="WDOU25"):
        """Subscreve ticker"""
        try:
            result = self.dll.SubscribeTicker(
                c_wchar_p(ticker), 
                c_wchar_p("F")
            )
            if result == 0:
                self.logger.info(f"[OK] Subscrito a {ticker}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Erro: {e}")
            return False
            
    def _calculate_simple_features(self):
        """Calcula features simples"""
        if len(self.candles) < 20:
            return None
            
        try:
            # Criar DataFrame
            df = pd.DataFrame(self.candles)
            
            # Features básicas
            features = {}
            
            # Preços
            closes = df['close'].values
            features['price_current'] = closes[-1]
            features['price_mean_5'] = np.mean(closes[-5:])
            features['price_mean_20'] = np.mean(closes[-20:])
            features['price_std_20'] = np.std(closes[-20:])
            
            # Retornos
            returns = np.diff(closes) / closes[:-1]
            features['return_1'] = returns[-1] if len(returns) > 0 else 0
            features['return_mean_5'] = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            
            # Volume
            volumes = df['volume'].values
            features['volume_mean_5'] = np.mean(volumes[-5:])
            features['volume_ratio'] = volumes[-1] / features['volume_mean_5'] if features['volume_mean_5'] > 0 else 1
            
            # RSI simples
            gains = [r if r > 0 else 0 for r in returns[-14:]]
            losses = [-r if r < 0 else 0 for r in returns[-14:]]
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            features['rsi_14'] = 100 - (100 / (1 + rs))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular features: {e}")
            return None
            
    def _make_prediction(self):
        """Faz predição ML"""
        if not self.model:
            return None
            
        try:
            # Calcular features
            features = self._calculate_simple_features()
            if not features:
                return None
                
            # Criar vetor de features na ordem correta
            feature_vector = []
            for feat_name in self.features_list:
                # Usar 0 como default para features não calculadas
                value = features.get(feat_name, 0)
                feature_vector.append(value)
                
            # Fazer predição
            X = np.array([feature_vector])
            
            # Probabilidades
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)[0]
                prediction = {
                    'direction': proba[1],  # Probabilidade de alta
                    'confidence': max(proba)
                }
            else:
                # Regressão
                pred = self.model.predict(X)[0]
                prediction = {
                    'direction': 0.5 + pred,  # Converter para 0-1
                    'confidence': abs(pred)
                }
                
            self.stats['predictions'] += 1
            return prediction
            
        except Exception as e:
            self.logger.error(f"Erro na predição: {e}")
            return None
            
    def run_strategy(self):
        """Estratégia simples com ML"""
        self.logger.info("[STRATEGY] Iniciando estratégia ML simples")
        
        last_prediction_time = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Predição a cada 30 segundos
                if (current_time - last_prediction_time) > 30:
                    prediction = self._make_prediction()
                    
                    if prediction:
                        direction = prediction['direction']
                        confidence = prediction['confidence']
                        
                        self.logger.info(f"\n[ML] Direction: {direction:.3f} | Confidence: {confidence:.3f}")
                        
                        # Sinais simples
                        if self.position == 0:
                            if direction > 0.7 and confidence > 0.65:
                                self._send_order(1)  # Compra
                            elif direction < 0.3 and confidence > 0.65:
                                self._send_order(-1)  # Venda
                                
                        elif self.position != 0:
                            # Fechar posição se sinal contrário
                            if (self.position > 0 and direction < 0.4) or \
                               (self.position < 0 and direction > 0.6):
                                self._send_order(-self.position)
                                
                    last_prediction_time = current_time
                    
                # Status
                if int(current_time) % 60 == 0:
                    elapsed = (current_time - self.stats['start_time']) / 60
                    self.logger.info(f"\n[STATUS] {elapsed:.1f}min | Price: {self.current_price:.2f} | Pos: {self.position} | Trades: {self.stats['trades']}")
                    
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Erro: {e}")
                time.sleep(5)
                
    def _send_order(self, side):
        """Envia ordem (simulada)"""
        if side == 0:
            return
            
        side_str = "COMPRA" if side > 0 else "VENDA"
        self.logger.info(f"\n[ORDER] {side_str} 1 @ {self.current_price:.2f}")
        
        # Atualizar posição
        old_position = self.position
        self.position += side
        
        # Calcular P&L se fechando posição
        if old_position != 0 and self.position == 0:
            pnl = (self.current_price - self.entry_price) * old_position
            self.stats['pnl'] += pnl
            self.logger.info(f"[P&L] R$ {pnl:.2f} | Total: R$ {self.stats['pnl']:.2f}")
        elif self.position != 0:
            self.entry_price = self.current_price
            
        self.stats['trades'] += 1
        self.logger.info(f"[POSITION] {self.position} contratos")
        
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
            
    def cleanup(self):
        """Finaliza DLL"""
        if self.dll and hasattr(self.dll, 'DLLFinalize'):
            self.dll.DLLFinalize()

# Variável global
system = None

def signal_handler(signum, frame):
    """Handler para Ctrl+C"""
    global system
    print("\n\nFinalizando...")
    if system:
        system.stop()
    sys.exit(0)

def main():
    global system
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "="*60)
    print("QUANTUM TRADER - ML SIMPLES")
    print("="*60)
    print(f"Data: {datetime.now()}")
    print("="*60)
    
    try:
        # Criar sistema
        system = SimpleMLSystem()
        
        # Inicializar
        if not system.initialize():
            return 1
            
        # Aguardar
        time.sleep(3)
        
        # Subscrever
        if not system.subscribe_ticker("WDOU25"):
            return 1
            
        # Aguardar dados
        time.sleep(3)
        
        # Iniciar
        if not system.start():
            return 1
            
        logger.info("\n" + "="*60)
        logger.info("SISTEMA OPERACIONAL")
        logger.info("ML: " + ("Ativo" if system.model else "Inativo"))
        logger.info("Para parar: CTRL+C")
        logger.info("="*60)
        
        # Loop principal
        while system.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("\nInterrompido")
        
    except Exception as e:
        logger.error(f"Erro: {e}", exc_info=True)
        
    finally:
        if system:
            system.stop()
            system.cleanup()
            
        # Stats finais
        if system:
            runtime = (time.time() - system.stats['start_time']) / 60
            logger.info(f"\nTempo: {runtime:.1f} min")
            logger.info(f"Trades: {system.stats['trades']}")
            logger.info(f"Predições ML: {system.stats['predictions']}")
            logger.info(f"P&L Total: R$ {system.stats['pnl']:.2f}")
            
        logger.info(f"\nLogs: {log_file}")

if __name__ == "__main__":
    sys.exit(main())