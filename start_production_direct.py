"""
Script de Produção Direto - Baseado no book_collector que funciona
Conecta diretamente à DLL sem usar o TradingSystem complexo
"""

import os
import sys
import time
import ctypes
from ctypes import *
from datetime import datetime, time as dtime
from pathlib import Path
import logging
import threading
import signal
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
log_file = f'logs/production/direct_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
Path("logs/production").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('DirectTrading')

# Estrutura TAssetIDRec
class TAssetIDRec(Structure):
    _fields_ = [
        ("ticker", c_wchar * 35),
        ("bolsa", c_wchar * 15),
    ]

class DirectTradingSystem:
    def __init__(self):
        self.dll = None
        self.logger = logger
        
        # Flags de controle
        self.bAtivo = False
        self.bMarketConnected = False
        self.bConnectado = False
        self.bBrokerConnected = False
        self.is_running = False
        
        # Referências dos callbacks (IMPORTANTE: manter referências)
        self.callback_refs = {}
        
        # Dados de mercado
        self.current_price = 0
        self.last_candle = None
        self.position = 0
        
        # Estatísticas
        self.stats = {
            'start_time': time.time(),
            'callbacks': {},
            'trades': 0,
            'errors': 0
        }
        
    def initialize(self):
        """Inicializa DLL e callbacks"""
        try:
            # Carregar DLL
            dll_path = "./ProfitDLL64.dll"
            self.logger.info(f"Carregando DLL: {os.path.abspath(dll_path)}")
            
            if not os.path.exists(dll_path):
                self.logger.error("DLL não encontrada!")
                return False
                
            self.dll = WinDLL(dll_path)
            self.logger.info("[OK] DLL carregada")
            
            # Criar TODOS os callbacks ANTES do login
            self._create_all_callbacks()
            
            # Login com callbacks
            key = c_wchar_p("HMARL")
            user = c_wchar_p(os.getenv('PROFIT_USERNAME', ''))
            pwd = c_wchar_p(os.getenv('PROFIT_PASSWORD', ''))
            
            self.logger.info("Fazendo login...")
            
            # DLLInitializeLogin com callbacks
            result = self.dll.DLLInitializeLogin(
                key, user, pwd,
                self.callback_refs['state'],      # stateCallback
                None,                             # historyCallback
                None,                             # orderChangeCallback
                None,                             # accountCallback (evitar segfault)
                None,                             # accountInfoCallback
                self.callback_refs['daily'],      # newDailyCallback
                None,                             # priceBookCallback
                None,                             # offerBookCallback
                None,                             # historyTradeCallback
                None,                             # progressCallBack
                self.callback_refs['tiny_book']   # tinyBookCallBack
            )
            
            if result != 0:
                self.logger.error(f"Erro no login: {result}")
                return False
                
            self.logger.info(f"[OK] Login bem sucedido")
            
            # Aguardar conexão
            if not self._wait_connection():
                return False
                
            # Setup callbacks adicionais
            self._setup_additional_callbacks()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _create_all_callbacks(self):
        """Cria callbacks essenciais"""
        
        # State callback
        @WINFUNCTYPE(None, c_int32, c_int32)
        def stateCallback(nType, nResult):
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
                
            if self.bMarketConnected and self.bConnectado:
                self.logger.info(">>> SISTEMA CONECTADO <<<")
                
        self.callback_refs['state'] = stateCallback
        
        # TinyBook callback (preços)
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_double, c_int, c_int)
        def tinyBookCallBack(assetId, price, qtd, side):
            if price > 0 and price < 10000:
                self.current_price = float(price)
                # Log a cada 1000 callbacks
                count = self.stats['callbacks'].get('tiny_book', 0) + 1
                self.stats['callbacks']['tiny_book'] = count
                if count % 1000 == 0:
                    side_str = "BID" if side == 0 else "ASK"
                    self.logger.info(f'[PRICE] {side_str}: R$ {price:.2f} x {qtd}')
                    
        self.callback_refs['tiny_book'] = tinyBookCallBack
        
        # Daily callback (candles)
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_wchar_p, c_double, c_double, 
                    c_double, c_double, c_double, c_double, c_double, c_double, 
                    c_double, c_double, c_int, c_int, c_int, c_int, c_int, c_int, c_int)
        def dailyCallback(assetId, date, sOpen, sHigh, sLow, sClose, sVol, sAjuste, 
                         sMaxLimit, sMinLimit, sVolBuyer, sVolSeller, nQtd, nNegocios, 
                         nContratosOpen, nQtdBuyer, nQtdSeller, nNegBuyer, nNegSeller):
            
            self.last_candle = {
                'open': float(sOpen),
                'high': float(sHigh),
                'low': float(sLow),
                'close': float(sClose),
                'volume': float(sVol),
                'trades': int(nNegocios)
            }
            
            count = self.stats['callbacks'].get('daily', 0) + 1
            self.stats['callbacks']['daily'] = count
            if count % 10 == 0:
                self.logger.info(f'[CANDLE] C={sClose:.2f} V={sVol:.0f} Trades={nNegocios}')
                
        self.callback_refs['daily'] = dailyCallback
        
    def _wait_connection(self):
        """Aguarda conexão completa"""
        self.logger.info("Aguardando conexão...")
        
        timeout = 15
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            if self.bMarketConnected:
                self.logger.info("[OK] Conectado ao mercado!")
                return True
                
            if int(time.time() - start_time) % 2 == 0:
                self.logger.info(f"Status: Market={self.bMarketConnected}, Login={self.bConnectado}")
                
            time.sleep(0.1)
            
        self.logger.error("Timeout aguardando conexão")
        return False
        
    def _setup_additional_callbacks(self):
        """Configura callbacks adicionais"""
        
        # Trade callback
        if hasattr(self.dll, 'SetNewTradeCallback'):
            @WINFUNCTYPE(None, c_wchar_p, c_double, c_int, c_int, c_int)
            def tradeCallback(ticker, price, qty, buyer, seller):
                count = self.stats['callbacks'].get('trade', 0) + 1
                self.stats['callbacks']['trade'] = count
                if count % 100 == 0:
                    self.logger.info(f'[TRADE] @ R$ {price:.2f} x {qty}')
                    
            self.callback_refs['trade'] = tradeCallback
            self.dll.SetNewTradeCallback(self.callback_refs['trade'])
            
        # Re-registrar TinyBook
        if hasattr(self.dll, 'SetTinyBookCallback'):
            self.dll.SetTinyBookCallback(self.callback_refs['tiny_book'])
            
    def subscribe_ticker(self, ticker="WDOU25"):
        """Subscreve ticker"""
        try:
            exchange = "F"
            self.logger.info(f"Subscrevendo {ticker}...")
            
            result = self.dll.SubscribeTicker(c_wchar_p(ticker), c_wchar_p(exchange))
            if result == 0:
                self.logger.info(f"[OK] Subscrito a {ticker}")
                return True
            else:
                self.logger.error(f"Erro ao subscrever: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro: {e}")
            return False
            
    def send_order(self, side, quantity=1, price=None):
        """Envia ordem (simulada por enquanto)"""
        try:
            if price is None:
                price = self.current_price
                
            side_str = "COMPRA" if side > 0 else "VENDA"
            self.logger.info(f"\n[ORDER] {side_str} {quantity} @ R$ {price:.2f}")
            
            # Simular execução
            self.position += side * quantity
            self.stats['trades'] += 1
            
            self.logger.info(f"[POSITION] {self.position} contratos")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao enviar ordem: {e}")
            return False
            
    def run_simple_strategy(self):
        """Estratégia simples para teste"""
        self.logger.info("\n[STRATEGY] Iniciando estratégia simples")
        self.logger.info("Regras: Compra se preço < 5370, Vende se > 5390")
        
        last_action_time = 0
        action_interval = 60  # Ação no máximo a cada 60 segundos
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Verificar se temos preço
                if self.current_price > 0:
                    # Estratégia simples de range
                    if self.position == 0:  # Sem posição
                        if self.current_price < 5370:
                            if current_time - last_action_time > action_interval:
                                self.send_order(1)  # Compra
                                last_action_time = current_time
                                
                        elif self.current_price > 5390:
                            if current_time - last_action_time > action_interval:
                                self.send_order(-1)  # Venda
                                last_action_time = current_time
                                
                    elif self.position > 0:  # Comprado
                        if self.current_price > 5380:  # Target
                            self.send_order(-1)  # Fecha compra
                            last_action_time = current_time
                            
                        elif self.current_price < 5365:  # Stop
                            self.send_order(-1)  # Stop loss
                            last_action_time = current_time
                            
                    elif self.position < 0:  # Vendido
                        if self.current_price < 5380:  # Target
                            self.send_order(1)  # Fecha venda
                            last_action_time = current_time
                            
                        elif self.current_price > 5395:  # Stop
                            self.send_order(1)  # Stop loss
                            last_action_time = current_time
                            
                # Status periódico
                if int(current_time) % 30 == 0:
                    elapsed = (current_time - self.stats['start_time']) / 60
                    self.logger.info(f"\n[STATUS] {elapsed:.1f}min | Price: R$ {self.current_price:.2f} | Pos: {self.position} | Trades: {self.stats['trades']}")
                    
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Erro na estratégia: {e}")
                self.stats['errors'] += 1
                if self.stats['errors'] > 10:
                    break
                    
    def start(self):
        """Inicia sistema"""
        self.is_running = True
        self.logger.info("\n[START] Sistema iniciado")
        
        # Thread para estratégia
        strategy_thread = threading.Thread(target=self.run_simple_strategy)
        strategy_thread.daemon = True
        strategy_thread.start()
        
        return True
        
    def stop(self):
        """Para sistema"""
        self.is_running = False
        self.logger.info("\n[STOP] Parando sistema...")
        
        # Fechar posições
        if self.position != 0:
            self.logger.info(f"Fechando posição: {self.position} contratos")
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
    
    # Handler de sinal
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "="*60)
    print("QUANTUM TRADER ML - VERSÃO DIRETA")
    print("="*60)
    print(f"Data: {datetime.now()}")
    print("="*60)
    print("\nConectando diretamente à DLL...")
    print("Estratégia simples de teste")
    print("="*60)
    
    try:
        # Criar sistema
        system = DirectTradingSystem()
        
        # Inicializar
        if not system.initialize():
            logger.error("Falha na inicialização")
            return 1
            
        # Aguardar estabilização
        logger.info("Aguardando estabilização...")
        time.sleep(3)
        
        # Subscrever
        if not system.subscribe_ticker("WDOU25"):
            logger.error("Falha ao subscrever ticker")
            return 1
            
        # Aguardar dados
        logger.info("Aguardando dados...")
        time.sleep(3)
        
        # Iniciar sistema
        if not system.start():
            logger.error("Falha ao iniciar")
            return 1
            
        logger.info("\n" + "="*60)
        logger.info("SISTEMA OPERACIONAL")
        logger.info("Para parar: CTRL+C")
        logger.info("="*60)
        
        # Loop principal
        while system.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("\nCTRL+C pressionado")
        
    except Exception as e:
        logger.error(f"Erro crítico: {e}", exc_info=True)
        
    finally:
        if system:
            system.stop()
            system.cleanup()
            
        # Estatísticas finais
        logger.info("\n" + "="*60)
        logger.info("ESTATÍSTICAS FINAIS")
        logger.info("="*60)
        
        if system and system.stats:
            runtime = (time.time() - system.stats['start_time']) / 60
            logger.info(f"Tempo total: {runtime:.1f} minutos")
            logger.info(f"Trades executados: {system.stats['trades']}")
            logger.info(f"Callbacks recebidos: {sum(system.stats['callbacks'].values()):,}")
            logger.info(f"Erros: {system.stats['errors']}")
            
        logger.info(f"\nLogs: {log_file}")
        logger.info("="*60)

if __name__ == "__main__":
    sys.exit(main())