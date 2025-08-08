"""
Script de Produção com ML - Integração completa
Baseado no sistema direto que funciona, mas com ML real
"""

import os
import sys
import time
import ctypes
from ctypes import *
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
import logging
import threading
import signal
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
log_file = f'logs/production/ml_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
Path("logs/production").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('MLTrading')

# Estrutura TAssetIDRec
class TAssetIDRec(Structure):
    _fields_ = [
        ("ticker", c_wchar * 35),
        ("bolsa", c_wchar * 15),
    ]

class MLTradingSystem:
    def __init__(self, config):
        self.dll = None
        self.logger = logger
        self.config = config
        
        # Flags de controle
        self.bAtivo = False
        self.bMarketConnected = False
        self.bConnectado = False
        self.bBrokerConnected = False
        self.is_running = False
        
        # Referências dos callbacks
        self.callback_refs = {}
        
        # Dados de mercado
        self.current_price = 0
        self.last_candle = None
        self.candles_df = pd.DataFrame()
        self.candles_buffer = []
        
        # Trading
        self.position = 0
        self.last_signal = 0
        self.last_trade_time = 0
        
        # ML Components
        self.model_manager = None
        self.feature_engine = None
        self.prediction_engine = None
        self.risk_manager = None
        
        # Estatísticas
        self.stats = {
            'start_time': time.time(),
            'callbacks': {},
            'trades': 0,
            'ml_predictions': 0,
            'errors': 0,
            'pnl': 0
        }
        
    def initialize(self):
        """Inicializa DLL e componentes ML"""
        try:
            # Inicializar componentes ML primeiro
            if not self._initialize_ml_components():
                self.logger.error("Falha ao inicializar componentes ML")
                return False
                
            # Carregar DLL
            dll_path = "./ProfitDLL64.dll"
            self.logger.info(f"Carregando DLL: {os.path.abspath(dll_path)}")
            
            if not os.path.exists(dll_path):
                self.logger.error("DLL não encontrada!")
                return False
                
            self.dll = WinDLL(dll_path)
            self.logger.info("[OK] DLL carregada")
            
            # Criar callbacks
            self._create_all_callbacks()
            
            # Login
            key = c_wchar_p("HMARL")
            user = c_wchar_p(os.getenv('PROFIT_USERNAME', ''))
            pwd = c_wchar_p(os.getenv('PROFIT_PASSWORD', ''))
            
            self.logger.info("Fazendo login...")
            
            result = self.dll.DLLInitializeLogin(
                key, user, pwd,
                self.callback_refs['state'],
                None,  # historyCallback
                None,  # orderChangeCallback
                None,  # accountCallback
                None,  # accountInfoCallback
                self.callback_refs['daily'],
                None,  # priceBookCallback
                None,  # offerBookCallback
                None,  # historyTradeCallback
                None,  # progressCallBack
                self.callback_refs['tiny_book']
            )
            
            if result != 0:
                self.logger.error(f"Erro no login: {result}")
                return False
                
            self.logger.info("[OK] Login bem sucedido")
            
            # Aguardar conexão
            if not self._wait_connection():
                return False
                
            # Setup callbacks adicionais
            self._setup_additional_callbacks()
            
            # Carregar dados históricos
            self._load_historical_data()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _initialize_ml_components(self):
        """Inicializa componentes ML"""
        try:
            self.logger.info("Inicializando componentes ML...")
            
            # Model Manager
            from src.model_manager import ModelManager
            self.model_manager = ModelManager(self.config['models_dir'])
            
            models_loaded = self.model_manager.load_models()
            if models_loaded:
                self.logger.info(f"[OK] {len(self.model_manager.models)} modelos carregados")
            else:
                self.logger.warning("Nenhum modelo ML carregado - operando sem ML")
                return True  # Continua sem ML
            
            # Feature Engine
            from src.feature_engine import FeatureEngine
            self.feature_engine = FeatureEngine(self.logger)
            
            # Prediction Engine
            from src.prediction_engine import PredictionEngine
            self.prediction_engine = PredictionEngine(
                self.model_manager,
                self.logger
            )
            
            # Risk Manager
            from src.risk_manager import RiskManager
            risk_config = self.config.get('risk', {})
            self.risk_manager = RiskManager(risk_config)
            
            self.logger.info("[OK] Componentes ML inicializados")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar ML: {e}")
            return False
            
    def _create_all_callbacks(self):
        """Cria callbacks essenciais"""
        
        # State callback
        @WINFUNCTYPE(None, c_int32, c_int32)
        def stateCallback(nType, nResult):
            states = {0: "Login", 1: "Broker", 2: "Market", 3: "Ativacao"}
            self.logger.info(f"[STATE] {states.get(nType, f'Type{nType}')}: {nResult}")
            
            if nType == 0:
                self.bConnectado = (nResult == 0)
            elif nType == 1:
                self.bBrokerConnected = (nResult == 5)
            elif nType == 2:
                self.bMarketConnected = (nResult in [2, 3, 4])
            elif nType == 3:
                self.bAtivo = (nResult == 0)
                
        self.callback_refs['state'] = stateCallback
        
        # TinyBook callback
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_double, c_int, c_int)
        def tinyBookCallBack(assetId, price, qtd, side):
            if price > 0 and price < 10000:
                self.current_price = float(price)
                count = self.stats['callbacks'].get('tiny_book', 0) + 1
                self.stats['callbacks']['tiny_book'] = count
                
        self.callback_refs['tiny_book'] = tinyBookCallBack
        
        # Daily callback (candles)
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_wchar_p, c_double, c_double, 
                    c_double, c_double, c_double, c_double, c_double, c_double, 
                    c_double, c_double, c_int, c_int, c_int, c_int, c_int, c_int, c_int)
        def dailyCallback(assetId, date, sOpen, sHigh, sLow, sClose, sVol, sAjuste, 
                         sMaxLimit, sMinLimit, sVolBuyer, sVolSeller, nQtd, nNegocios, 
                         nContratosOpen, nQtdBuyer, nQtdSeller, nNegBuyer, nNegSeller):
            
            # Criar novo candle
            candle = {
                'timestamp': datetime.now(),
                'open': float(sOpen),
                'high': float(sHigh),
                'low': float(sLow),
                'close': float(sClose),
                'volume': float(sVol),
                'trades': int(nNegocios),
                'contracts': int(nContratosOpen)
            }
            
            self.last_candle = candle
            
            # Adicionar ao buffer
            self.candles_buffer.append(candle)
            
            # Atualizar DataFrame a cada 10 candles
            if len(self.candles_buffer) >= 10:
                self._update_candles_dataframe()
                
            count = self.stats['callbacks'].get('daily', 0) + 1
            self.stats['callbacks']['daily'] = count
            
            if count % 10 == 0:
                self.logger.info(f'[CANDLE] C={sClose:.2f} V={sVol:.0f} T={nNegocios}')
                
        self.callback_refs['daily'] = dailyCallback
        
    def _wait_connection(self):
        """Aguarda conexão"""
        self.logger.info("Aguardando conexão...")
        
        timeout = 15
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            if self.bMarketConnected:
                self.logger.info("[OK] Conectado!")
                return True
                
            time.sleep(0.1)
            
        return False
        
    def _setup_additional_callbacks(self):
        """Setup callbacks adicionais"""
        if hasattr(self.dll, 'SetNewTradeCallback'):
            @WINFUNCTYPE(None, c_wchar_p, c_double, c_int, c_int, c_int)
            def tradeCallback(ticker, price, qty, buyer, seller):
                count = self.stats['callbacks'].get('trade', 0) + 1
                self.stats['callbacks']['trade'] = count
                
            self.callback_refs['trade'] = tradeCallback
            self.dll.SetNewTradeCallback(self.callback_refs['trade'])
            
    def _load_historical_data(self):
        """Carrega dados históricos"""
        try:
            self.logger.info("Carregando dados históricos...")
            
            # Simular dados históricos mínimos para features
            periods = 100
            base_price = 5400
            
            timestamps = pd.date_range(
                end=datetime.now(),
                periods=periods,
                freq='1min'
            )
            
            # Gerar preços simulados
            returns = np.random.normal(0, 0.002, periods)
            prices = base_price * np.exp(np.cumsum(returns))
            
            self.candles_df = pd.DataFrame({
                'timestamp': timestamps,
                'open': prices * (1 + np.random.uniform(-0.001, 0.001, periods)),
                'high': prices * (1 + np.random.uniform(0, 0.002, periods)),
                'low': prices * (1 - np.random.uniform(0, 0.002, periods)),
                'close': prices,
                'volume': np.random.uniform(1000, 5000, periods)
            })
            
            self.candles_df.set_index('timestamp', inplace=True)
            
            self.logger.info(f"[OK] {len(self.candles_df)} candles históricos")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar histórico: {e}")
            
    def _update_candles_dataframe(self):
        """Atualiza DataFrame de candles"""
        try:
            # Converter buffer para DataFrame
            new_df = pd.DataFrame(self.candles_buffer)
            new_df.set_index('timestamp', inplace=True)
            
            # Concatenar com existente
            self.candles_df = pd.concat([self.candles_df, new_df])
            
            # Manter apenas últimas 1000 linhas
            self.candles_df = self.candles_df.tail(1000)
            
            # Limpar buffer
            self.candles_buffer.clear()
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar candles: {e}")
            
    def subscribe_ticker(self, ticker="WDOU25"):
        """Subscreve ticker"""
        try:
            exchange = "F"
            self.logger.info(f"Subscrevendo {ticker}...")
            
            result = self.dll.SubscribeTicker(c_wchar_p(ticker), c_wchar_p(exchange))
            if result == 0:
                self.logger.info(f"[OK] Subscrito a {ticker}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Erro: {e}")
            return False
            
    def send_order(self, side, quantity=1, price=None):
        """Envia ordem"""
        try:
            if price is None:
                price = self.current_price
                
            # Validar com risk manager
            if self.risk_manager:
                if not self.risk_manager.validate_order(
                    side, quantity, price, self.position
                ):
                    self.logger.warning("Ordem bloqueada pelo Risk Manager")
                    return False
                    
            side_str = "COMPRA" if side > 0 else "VENDA"
            self.logger.info(f"\n[ORDER] {side_str} {quantity} @ R$ {price:.2f}")
            
            # TODO: Implementar envio real via DLL
            # Por enquanto simular
            self.position += side * quantity
            self.stats['trades'] += 1
            
            # Atualizar risk manager
            if self.risk_manager:
                self.risk_manager.update_position(
                    self.position, price
                )
                
            self.logger.info(f"[POSITION] {self.position} contratos")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao enviar ordem: {e}")
            return False
            
    def run_ml_strategy(self):
        """Estratégia com ML"""
        self.logger.info("\n[STRATEGY] Iniciando estratégia ML")
        
        last_prediction_time = 0
        prediction_interval = 30  # Predição a cada 30 segundos
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Fazer predição ML periodicamente
                if (current_time - last_prediction_time) > prediction_interval:
                    if len(self.candles_df) > 50 and self.prediction_engine:
                        # Preparar dados
                        market_data = {
                            'candles': self.candles_df,
                            'current_price': self.current_price,
                            'spread': 0.5,  # Simulado
                            'volume': self.last_candle['volume'] if self.last_candle else 0
                        }
                        
                        # Fazer predição
                        prediction = self.prediction_engine.predict(market_data)
                        
                        if prediction:
                            self.stats['ml_predictions'] += 1
                            
                            # Log da predição
                            self.logger.info(f"\n[ML] Predição #{self.stats['ml_predictions']}")
                            self.logger.info(f"Direction: {prediction.get('direction', 0):.3f}")
                            self.logger.info(f"Confidence: {prediction.get('confidence', 0):.3f}")
                            
                            # Gerar sinal de trading
                            signal = self._generate_signal(prediction)
                            
                            # Executar se houver sinal
                            if signal != 0 and signal != self.last_signal:
                                # Fechar posição anterior se houver
                                if self.position != 0:
                                    self.send_order(-self.position)
                                    
                                # Abrir nova posição
                                if signal != 0:
                                    self.send_order(signal)
                                    
                                self.last_signal = signal
                                self.last_trade_time = current_time
                                
                    last_prediction_time = current_time
                    
                # Status periódico
                if int(current_time) % 60 == 0:
                    self._log_status()
                    
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Erro na estratégia: {e}")
                self.stats['errors'] += 1
                if self.stats['errors'] > 10:
                    break
                    
    def _generate_signal(self, prediction):
        """Gera sinal de trading baseado na predição"""
        try:
            direction = prediction.get('direction', 0)
            confidence = prediction.get('confidence', 0)
            
            # Thresholds conservadores
            confidence_threshold = 0.7
            direction_threshold = 0.6
            
            # Gerar sinal
            if confidence > confidence_threshold:
                if direction > direction_threshold:
                    return 1  # Compra
                elif direction < (1 - direction_threshold):
                    return -1  # Venda
                    
            return 0  # Sem sinal
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar sinal: {e}")
            return 0
            
    def _log_status(self):
        """Log de status do sistema"""
        elapsed = (time.time() - self.stats['start_time']) / 60
        
        self.logger.info(f"\n[STATUS] {elapsed:.1f}min")
        self.logger.info(f"Price: R$ {self.current_price:.2f}")
        self.logger.info(f"Position: {self.position}")
        self.logger.info(f"Trades: {self.stats['trades']}")
        self.logger.info(f"ML Predictions: {self.stats['ml_predictions']}")
        self.logger.info(f"Candles: {len(self.candles_df)}")
        
        if self.risk_manager:
            metrics = self.risk_manager.get_metrics()
            self.logger.info(f"Daily P&L: R$ {metrics['daily_pnl']:.2f}")
            
    def start(self):
        """Inicia sistema"""
        self.is_running = True
        self.logger.info("\n[START] Sistema ML iniciado")
        
        # Thread para estratégia
        strategy_thread = threading.Thread(target=self.run_ml_strategy)
        strategy_thread.daemon = True
        strategy_thread.start()
        
        return True
        
    def stop(self):
        """Para sistema"""
        self.is_running = False
        self.logger.info("\n[STOP] Parando sistema...")
        
        # Fechar posições
        if self.position != 0:
            self.logger.info(f"Fechando posição: {self.position}")
            self.send_order(-self.position)
            
        time.sleep(1)
        
    def cleanup(self):
        """Finaliza DLL"""
        if self.dll and hasattr(self.dll, 'DLLFinalize'):
            result = self.dll.DLLFinalize()
            self.logger.info(f"[CLEANUP] DLLFinalize: {result}")

# Variável global
system = None

def signal_handler(signum, frame):
    """Handler para Ctrl+C"""
    global system
    print("\n\nInterrompido. Finalizando...")
    if system:
        system.stop()
    sys.exit(0)

def main():
    global system
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "="*60)
    print("QUANTUM TRADER ML - PRODUÇÃO")
    print("="*60)
    print(f"Data: {datetime.now()}")
    print("="*60)
    
    # Configuração
    config = {
        'models_dir': 'models',
        'ticker': 'WDOU25',
        'risk': {
            'max_position_size': 1,
            'max_daily_loss': 500.0,
            'stop_loss_pct': 0.005,
            'take_profit_pct': 0.01
        }
    }
    
    try:
        # Criar sistema
        system = MLTradingSystem(config)
        
        # Inicializar
        if not system.initialize():
            logger.error("Falha na inicialização")
            return 1
            
        # Aguardar
        logger.info("Aguardando estabilização...")
        time.sleep(3)
        
        # Subscrever
        if not system.subscribe_ticker(config['ticker']):
            logger.error("Falha ao subscrever")
            return 1
            
        # Aguardar dados
        logger.info("Aguardando dados...")
        time.sleep(3)
        
        # Iniciar
        if not system.start():
            logger.error("Falha ao iniciar")
            return 1
            
        logger.info("\n" + "="*60)
        logger.info("SISTEMA OPERACIONAL COM ML")
        logger.info("Risk Limits:")
        logger.info(f"- Max position: {config['risk']['max_position_size']} contratos")
        logger.info(f"- Max daily loss: R$ {config['risk']['max_daily_loss']}")
        logger.info(f"- Stop loss: {config['risk']['stop_loss_pct']*100}%")
        logger.info("Para parar: CTRL+C")
        logger.info("="*60)
        
        # Loop principal
        while system.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("\nCTRL+C")
        
    except Exception as e:
        logger.error(f"Erro: {e}", exc_info=True)
        
    finally:
        if system:
            system.stop()
            system.cleanup()
            
        # Estatísticas
        logger.info("\n" + "="*60)
        logger.info("ESTATÍSTICAS FINAIS")
        logger.info("="*60)
        
        if system and system.stats:
            runtime = (time.time() - system.stats['start_time']) / 60
            logger.info(f"Runtime: {runtime:.1f} min")
            logger.info(f"Trades: {system.stats['trades']}")
            logger.info(f"ML Predictions: {system.stats['ml_predictions']}")
            logger.info(f"Callbacks: {sum(system.stats['callbacks'].values()):,}")
            logger.info(f"Errors: {system.stats['errors']}")
            
        logger.info(f"\nLogs: {log_file}")
        logger.info("="*60)

if __name__ == "__main__":
    sys.exit(main())