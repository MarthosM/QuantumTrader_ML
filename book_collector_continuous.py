"""
Book Collector Contínuo - Coleta até o fim do pregão
Baseado no book_collector.py funcional
"""

import os
import sys
import time
import ctypes
from ctypes import *
from datetime import datetime, time as dtime
from pathlib import Path
import pandas as pd
import json
import logging
import threading
import signal

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Estrutura TAssetIDRec
class TAssetIDRec(Structure):
    _fields_ = [
        ("ticker", c_wchar * 35),
        ("bolsa", c_wchar * 15),
    ]

class ContinuousBookCollector:
    def __init__(self):
        self.logger = logging.getLogger('ContinuousCollector')
        self.dll = None
        
        # Flags de controle
        self.bAtivo = False
        self.bMarketConnected = False
        self.bConnectado = False
        self.bBrokerConnected = False
        self.is_running = False
        self.force_stop = False
        
        # Contadores
        self.callbacks = {
            'state': 0,
            'trade': 0,
            'tiny_book': 0,
            'offer_book': 0,
            'price_book': 0,
            'daily': 0,
            'history': 0,
            'progress': 0
        }
        
        # Dados coletados
        self.data = []
        self.data_lock = threading.Lock()
        
        # Referências dos callbacks
        self.callback_refs = {}
        
        # Ticker que estamos monitorando
        self.target_ticker = "WDOU25"
        
        # Dados anteriores para cálculo de delta
        self.last_daily = {
            'volume': 0,
            'trades': 0,
            'qty': 0,
            'timestamp': None
        }
        
        # Configurações de mercado
        self.market_open = dtime(9, 0)     # 09:00
        self.market_close = dtime(18, 0)   # 18:00
        self.save_interval = 300            # Salvar a cada 5 minutos
        self.rotate_size = 100000           # Rotacionar arquivo a cada 100k registros
        self.last_consolidation_hour = -1   # Controle de consolidacao
        
        # Estatísticas
        self.stats = {
            'start_time': None,
            'last_save': None,
            'total_saved': 0,
            'files_created': 0,
            'last_price': 0,
            'errors': 0
        }
        
    def initialize(self):
        """Inicializa DLL e callbacks"""
        try:
            # Carregar DLL
            dll_path = "./ProfitDLL64.dll"
            self.logger.info(f"Carregando DLL: {os.path.abspath(dll_path)}")
            
            print(f"[DEBUG] Verificando se DLL existe: {os.path.exists(dll_path)}")
            
            self.dll = WinDLL(dll_path)
            self.logger.info("[OK] DLL carregada")
            print("[DEBUG] DLL carregada com sucesso")
            
            # Criar TODOS os callbacks ANTES do login
            self._create_all_callbacks()
            
            # Login com callbacks
            key = c_wchar_p("HMARL")
            user = c_wchar_p(os.getenv('PROFIT_USERNAME', '29936354842'))
            pwd = c_wchar_p(os.getenv('PROFIT_PASSWORD', 'Ultrajiu33!'))
            
            self.logger.info("Fazendo login com callbacks...")
            
            # DLLInitializeLogin com TODOS os callbacks possíveis
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
            
            print(f"[DEBUG] DLLInitializeLogin retornou: {result}")
            
            if result != 0:
                self.logger.error(f"Erro no login: {result}")
                return False
                
            self.logger.info(f"[OK] Login bem sucedido: {result}")
            
            # Aguardar conexão completa
            if not self._wait_login():
                self.logger.error("Timeout aguardando conexão")
                return False
            
            # Configurar callbacks adicionais após login
            self._setup_additional_callbacks()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _create_all_callbacks(self):
        """Cria TODOS os callbacks possíveis"""
        
        # State callback - CRÍTICO
        @WINFUNCTYPE(None, c_int32, c_int32)
        def stateCallback(nType, nResult):
            self.callbacks['state'] += 1
            
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
            self.callbacks['tiny_book'] += 1
            
            ticker = self.target_ticker
            
            # Validar preço
            if price > 0 and price < 10000:
                # Log a cada 1000 ou mudança significativa de preço
                if self.callbacks['tiny_book'] % 1000 == 0 or abs(price - self.stats['last_price']) > 1:
                    side_str = "BID" if side == 0 else "ASK"
                    self.logger.info(f'[TINY #{self.callbacks["tiny_book"]:,}] {ticker} {side_str}: R$ {price:.2f} x {qtd}')
                    self.stats['last_price'] = price
                        
                # Salvar dados
                with self.data_lock:
                    self.data.append({
                        'type': 'tiny_book',
                        'ticker': ticker,
                        'side': 'bid' if side == 0 else 'ask',
                        'price': float(price),
                        'quantity': int(qtd),
                        'timestamp': datetime.now().isoformat()
                    })
            
            return None
            
        self.callback_refs['tiny_book'] = tinyBookCallBack
        
        # Price Book callback V2
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_int, c_int, c_int, c_double, c_int, c_double, POINTER(c_int), POINTER(c_int))
        def priceBookCallback(assetId, nAction, nPosition, Side, sPrice, nQtd, nCount, pArraySell, pArrayBuy):
            self.callbacks['price_book'] += 1
            
            ticker = self.target_ticker
            
            # Validar dados antes de processar
            if sPrice > 0 and sPrice < 10000 and nQtd > 0 and nQtd < 10000:
                if self.callbacks['price_book'] % 1000 == 0:
                    self.logger.info(f'[PRICE #{self.callbacks["price_book"]:,}] {ticker} Price={sPrice:.2f} Qty={nQtd}')
                    
                with self.data_lock:
                    self.data.append({
                        'type': 'price_book',
                        'ticker': ticker,
                        'action': nAction,
                        'position': nPosition,
                        'side': 'bid' if Side == 0 else 'ask',
                        'price': float(sPrice),
                        'quantity': int(nQtd),
                        'timestamp': datetime.now().isoformat()
                    })
            
            return None
            
        self.callback_refs['price_book'] = priceBookCallback
        
        # Offer Book callback V2
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_int, c_int, c_int, c_int, c_int, c_longlong, c_double, c_int, c_int, c_int, c_int, c_int,
                   c_wchar_p, POINTER(c_ubyte), POINTER(c_ubyte))
        def offerBookCallback(assetId, nAction, nPosition, Side, nQtd, nAgent, nOfferID, sPrice, bHasPrice,
                             bHasQtd, bHasDate, bHasOfferID, bHasAgent, date, pArraySell, pArrayBuy):
            self.callbacks['offer_book'] += 1
            
            ticker = self.target_ticker
            
            # Validar dados antes de processar
            if bHasPrice and bHasQtd and sPrice > 0 and sPrice < 10000 and nQtd > 0 and nQtd < 10000:
                if self.callbacks['offer_book'] % 5000 == 0:
                    side_str = "BID" if Side == 0 else "ASK"
                    self.logger.info(f'[OFFER #{self.callbacks["offer_book"]:,}] {ticker} {side_str} @ R$ {sPrice:.2f} x {nQtd}')
                    
                with self.data_lock:
                    self.data.append({
                        'type': 'offer_book',
                        'ticker': ticker,
                        'side': 'bid' if Side == 0 else 'ask',
                        'price': float(sPrice),
                        'quantity': int(nQtd),
                        'agent': int(nAgent),
                        'offer_id': int(nOfferID),
                        'action': int(nAction),
                        'position': int(nPosition),
                        'timestamp': datetime.now().isoformat()
                    })
            
            return None
            
        self.callback_refs['offer_book'] = offerBookCallback
        
        # Daily callback
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_wchar_p, c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double,
                   c_double, c_int, c_int, c_int, c_int, c_int, c_int, c_int)
        def dailyCallback(assetId, date, sOpen, sHigh, sLow, sClose, sVol, sAjuste, sMaxLimit, sMinLimit, sVolBuyer,
                         sVolSeller, nQtd, nNegocios, nContratosOpen, nQtdBuyer, nQtdSeller, nNegBuyer, nNegSeller):
            self.callbacks['daily'] += 1
            
            ticker = self.target_ticker
            
            # Calcular deltas (volume incremental)
            volume_delta = float(sVol) - self.last_daily['volume'] if self.last_daily['volume'] > 0 else 0
            trades_delta = int(nNegocios) - self.last_daily['trades'] if self.last_daily['trades'] > 0 else 0
            qty_delta = int(nQtd) - self.last_daily['qty'] if self.last_daily['qty'] > 0 else 0
            
            # Log apenas mudanças significativas ou a cada 100 callbacks
            if volume_delta > 0 or self.callbacks['daily'] % 100 == 1:
                self.logger.info(f'[DAILY] {ticker}: C={sClose:.2f} VolΔ={volume_delta:.0f} TradesΔ={trades_delta} QtyΔ={qty_delta}')
                
            # Salvar dados com deltas e valores absolutos
            with self.data_lock:
                self.data.append({
                    'type': 'daily',
                    'ticker': ticker,
                    'open': float(sOpen),
                    'high': float(sHigh),
                    'low': float(sLow),
                    'close': float(sClose),
                    'volume_total': float(sVol),           # Volume total do dia
                    'volume_delta': volume_delta,          # Volume desde último update
                    'qty_total': int(nQtd),               # Quantidade total
                    'qty_delta': qty_delta,               # Quantidade incremental
                    'trades_total': int(nNegocios),       # Total de negócios
                    'trades_delta': trades_delta,         # Negócios incrementais
                    'contracts_open': int(nContratosOpen),
                    'volume_buyer': float(sVolBuyer),
                    'volume_seller': float(sVolSeller),
                    'qty_buyer': int(nQtdBuyer),
                    'qty_seller': int(nQtdSeller),
                    'timestamp': datetime.now().isoformat()
                })
                
            # Atualizar últimos valores
            self.last_daily['volume'] = float(sVol)
            self.last_daily['trades'] = int(nNegocios)
            self.last_daily['qty'] = int(nQtd)
            self.last_daily['timestamp'] = datetime.now()
            
            return None
            
        self.callback_refs['daily'] = dailyCallback
        
        # Progress callback
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_int)
        def progressCallback(assetId, progress):
            self.callbacks['progress'] += 1
            
            ticker = self.target_ticker
            self.logger.info(f'[PROGRESS] {ticker}: {progress}%')
            
            return None
            
        self.callback_refs['progress'] = progressCallback
        
        # History callback
        @WINFUNCTYPE(None, POINTER(TAssetIDRec))
        def historyCallback(assetId):
            self.callbacks['history'] += 1
            
            ticker = self.target_ticker
            self.logger.info(f'[HISTORY] {ticker}')
            
            return None
            
        self.callback_refs['history'] = historyCallback
        
    def _wait_login(self):
        """Aguarda login completo"""
        self.logger.info("Aguardando conexão completa...")
        
        timeout = 15  # 15 segundos
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            if self.bMarketConnected:
                self.logger.info("[OK] Market conectado!")
                return True
                
            # Log periódico do status
            if int(time.time() - start_time) % 2 == 0:
                self.logger.info(f"Status: Market={self.bMarketConnected}, Broker={self.bBrokerConnected}, Login={self.bConnectado}, Ativo={self.bAtivo}")
                
            time.sleep(0.1)
            
        return False
        
    def _setup_additional_callbacks(self):
        """Configura callbacks adicionais após login"""
        
        # SetNewTradeCallback
        if hasattr(self.dll, 'SetNewTradeCallback'):
            @WINFUNCTYPE(None, c_wchar_p, c_double, c_int, c_int, c_int)
            def tradeCallback(ticker, price, qty, buyer, seller):
                self.callbacks['trade'] += 1
                
                ticker_str = self.target_ticker
                
                if price > 0 and price < 10000:
                    if self.callbacks['trade'] % 500 == 0:
                        self.logger.info(f'[TRADE #{self.callbacks["trade"]:,}] {ticker_str} @ R$ {price:.2f} x {qty}')
                        
                    with self.data_lock:
                        self.data.append({
                            'type': 'trade',
                            'ticker': ticker_str,
                            'price': float(price),
                            'quantity': int(qty),
                            'buyer': int(buyer),
                            'seller': int(seller),
                            'timestamp': datetime.now().isoformat()
                        })
                
                return None
                
            self.callback_refs['trade'] = tradeCallback
            self.dll.SetNewTradeCallback(self.callback_refs['trade'])
            self.logger.info("[OK] Trade callback registrado")
            
        # SetTinyBookCallback (redundante mas garante)
        if hasattr(self.dll, 'SetTinyBookCallback'):
            self.dll.SetTinyBookCallback(self.callback_refs['tiny_book'])
            self.logger.info("[OK] TinyBook callback re-registrado")
            
        # SetOfferBookCallbackV2
        if hasattr(self.dll, 'SetOfferBookCallbackV2'):
            self.dll.SetOfferBookCallbackV2(self.callback_refs['offer_book'])
            self.logger.info("[OK] OfferBook V2 callback registrado")
            
        # SetPriceBookCallback
        if hasattr(self.dll, 'SetPriceBookCallback'):
            self.dll.SetPriceBookCallback(self.callback_refs['price_book'])
            self.logger.info("[OK] PriceBook callback registrado")
            
    def subscribe_wdo(self):
        """Subscreve apenas WDOU25"""
        try:
            ticker = self.target_ticker
            exchange = "F"
            
            self.logger.info(f"\nSubscrevendo {ticker} na bolsa {exchange}...")
            
            # SubscribeTicker
            result = self.dll.SubscribeTicker(c_wchar_p(ticker), c_wchar_p(exchange))
            self.logger.info(f"SubscribeTicker({ticker}, {exchange}) = {result}")
            
            if result == 0:
                self.logger.info(f"[OK] Subscrito a {ticker}/{exchange}")
                
            # SubscribeOfferBook
            if hasattr(self.dll, 'SubscribeOfferBook'):
                result = self.dll.SubscribeOfferBook(c_wchar_p(ticker), c_wchar_p(exchange))
                self.logger.info(f"SubscribeOfferBook({ticker}, {exchange}) = {result}")
                
            # SubscribePriceBook
            if hasattr(self.dll, 'SubscribePriceBook'):
                result = self.dll.SubscribePriceBook(c_wchar_p(ticker), c_wchar_p(exchange))
                self.logger.info(f"SubscribePriceBook({ticker}, {exchange}) = {result}")
                
            return True
                
        except Exception as e:
            self.logger.error(f"Erro na subscrição: {e}")
            return False
            
    def is_market_open(self):
        """Verifica se o mercado está aberto"""
        now = datetime.now()
        current_time = now.time()
        weekday = now.weekday()
        
        # Fim de semana
        if weekday >= 5:
            return False
            
        # Horário de mercado
        return self.market_open <= current_time <= self.market_close
        
    def monitor_status(self):
        """Monitora status do sistema"""
        total_callbacks = sum(self.callbacks.values())
        elapsed = time.time() - self.stats['start_time']
        rate = total_callbacks / elapsed if elapsed > 0 else 0
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"[MONITOR] Tempo: {elapsed/60:.1f} min | Taxa: {rate:.0f} callbacks/seg")
        self.logger.info(f"Total callbacks: {total_callbacks:,}")
        
        # Detalhes por tipo
        for key, value in self.callbacks.items():
            if value > 0:
                self.logger.info(f"  {key:12}: {value:,}")
                
        with self.data_lock:
            self.logger.info(f"\nRegistros em memória: {len(self.data):,}")
            self.logger.info(f"Total salvo: {self.stats['total_saved']:,}")
            self.logger.info(f"Arquivos criados: {self.stats['files_created']}")
            
        if self.stats['last_price'] > 0:
            self.logger.info(f"Último preço: R$ {self.stats['last_price']:.2f}")
            
        self.logger.info(f"{'='*60}")
        
    def save_data(self, force_save=False):
        """Salva dados coletados com rotação de arquivo"""
        with self.data_lock:
            if not self.data and not force_save:
                return
                
            data_to_save = self.data.copy()
            self.data.clear()
            
        if not data_to_save:
            self.logger.warning("Nenhum dado para salvar")
            return
            
        try:
            # Criar diretório
            save_dir = Path('data/realtime/book') / datetime.now().strftime('%Y%m%d')
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Converter para DataFrame
            try:
                df = pd.DataFrame(data_to_save)
            except Exception as e:
                self.logger.error(f"Erro ao criar DataFrame: {e}")
                # Salvar como JSON de emergência
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                json_file = save_dir / f'wdo_continuous_{timestamp}_emergency.json'
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Dados salvos como JSON de emergência: {json_file}")
                return
                
            # Salvar em parquet
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            parquet_file = save_dir / f'wdo_continuous_{timestamp}.parquet'
            df.to_parquet(parquet_file, compression='snappy')
            
            self.stats['total_saved'] += len(df)
            self.stats['files_created'] += 1
            self.stats['last_save'] = datetime.now()
            
            self.logger.info(f"\n[SAVE] {len(df):,} registros salvos")
            self.logger.info(f"Arquivo: {parquet_file}")
            
            # Criar resumo
            summary = {
                'records': len(df),
                'total_saved': self.stats['total_saved'],
                'start_time': df['timestamp'].min() if not df.empty else None,
                'end_time': df['timestamp'].max() if not df.empty else None,
                'ticker': self.target_ticker,
                'types': df['type'].value_counts().to_dict() if 'type' in df.columns else {},
                'session_stats': {
                    'files_created': self.stats['files_created'],
                    'runtime_minutes': (time.time() - self.stats['start_time']) / 60,
                    'rate_per_minute': self.stats['total_saved'] / ((time.time() - self.stats['start_time']) / 60)
                }
            }
            
            # Estatísticas de preço
            if 'price' in df.columns:
                price_data = df[df['price'] > 0]['price']
                if not price_data.empty:
                    summary['price_stats'] = {
                        'min': float(price_data.min()),
                        'max': float(price_data.max()),
                        'mean': float(price_data.mean()),
                        'last': float(price_data.iloc[-1])
                    }
                    
            # Salvar resumo
            summary_file = save_dir / f'summary_continuous_{timestamp}.json'
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Erro ao salvar: {e}")
            self.stats['errors'] += 1
            import traceback
            traceback.print_exc()
            
    def consolidate_data(self):
        """Executa consolidacao automatica"""
        try:
            self.logger.info("\n" + "="*50)
            self.logger.info("INICIANDO CONSOLIDACAO AUTOMATICA")
            self.logger.info("="*50)
            
            import subprocess
            date = datetime.now().strftime('%Y%m%d')
            
            result = subprocess.run(
                ['python', 'auto_consolidate_book_data.py', date],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info("[OK] Consolidacao concluida com sucesso")
            else:
                self.logger.error(f"[ERRO] Falha na consolidacao: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"[ERRO] ao consolidar: {e}")
            
    def run_continuous(self):
        """Executa coleta contínua até o fim do pregão"""
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"INICIANDO COLETA CONTÍNUA - {self.target_ticker}")
        self.logger.info(f"Horário de mercado: {self.market_open} às {self.market_close}")
        self.logger.info(f"Salvamento a cada: {self.save_interval/60:.0f} minutos")
        self.logger.info(f"Rotação a cada: {self.rotate_size:,} registros")
        self.logger.info(f"{'='*70}\n")
        
        last_save_time = time.time()
        last_monitor_time = time.time()
        monitor_interval = 300  # Mostrar status a cada 5 minutos
        
        try:
            while self.is_running and not self.force_stop:
                current_time = time.time()
                
                # Verificar se mercado está aberto
                if not self.is_market_open():
                    self.logger.info("Mercado fechado. Aguardando abertura...")
                    # Salvar dados pendentes
                    self.save_data(force_save=True)
                    
                    # Aguardar 60 segundos antes de verificar novamente
                    time.sleep(60)
                    continue
                
                # Salvar periodicamente
                if (current_time - last_save_time) >= self.save_interval:
                    current_datetime = datetime.now()
                    self.logger.info("\n" + "="*50)
                    self.logger.info(f"Salvando dados as {current_datetime.strftime('%H:%M:%S')}")
                    self.logger.info("="*50)
                    self.save_data()
                    last_save_time = current_time
                    
                    # Consolidar dados automaticamente a cada hora
                    if current_datetime.hour != self.last_consolidation_hour and current_datetime.minute <= 5:
                        self.consolidate_data()
                        self.last_consolidation_hour = current_datetime.hour
                    
                # Rotação por tamanho
                with self.data_lock:
                    if len(self.data) >= self.rotate_size:
                        self.logger.info(f"Rotação de arquivo: {len(self.data):,} registros")
                        self.save_data()
                        
                # Monitor periódico
                if (current_time - last_monitor_time) >= monitor_interval:
                    self.monitor_status()
                    last_monitor_time = current_time
                    
                # Pequena pausa para não sobrecarregar CPU
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("\nColeta interrompida pelo usuário")
            
        except Exception as e:
            self.logger.error(f"Erro durante coleta: {e}")
            self.stats['errors'] += 1
            import traceback
            traceback.print_exc()
            
        finally:
            # Salvar dados finais
            self.logger.info("\nFinalizando coleta...")
            self.save_data(force_save=True)
            
            # Estatísticas finais
            self.logger.info(f"\n{'='*70}")
            self.logger.info("ESTATÍSTICAS FINAIS DA SESSÃO")
            self.logger.info(f"{'='*70}")
            self.monitor_status()
            
            runtime = (time.time() - self.stats['start_time']) / 3600
            self.logger.info(f"\nTempo total: {runtime:.2f} horas")
            self.logger.info(f"Taxa média: {self.stats['total_saved']/runtime/3600:.0f} registros/segundo")
            self.logger.info(f"Erros: {self.stats['errors']}")
            
    def stop(self):
        """Para a coleta de forma segura"""
        self.logger.info("Parando coleta...")
        self.is_running = False
        self.force_stop = True
        
    def cleanup(self):
        """Finaliza DLL"""
        if self.dll and hasattr(self.dll, 'DLLFinalize'):
            result = self.dll.DLLFinalize()
            self.logger.info(f"\n[CLEANUP] DLLFinalize: {result}")

def signal_handler(signum, frame):
    """Handler para Ctrl+C"""
    global collector
    print("\n\nRecebido sinal de interrupção. Finalizando...")
    if collector:
        collector.stop()
    sys.exit(0)

# Variável global para o handler
collector = None

def main():
    global collector
    
    print("\n" + "="*70)
    print("COLETOR CONTÍNUO WDO - ATÉ O FIM DO PREGÃO")
    print("="*70)
    print(f"Horário: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Verificar horário de mercado
    now = datetime.now()
    weekday = now.weekday()
    
    if weekday >= 5:
        print(f"[AVISO] FIM DE SEMANA - Mercado FECHADO")
    elif 9 <= now.hour < 18:
        print(f"[OK] Mercado ABERTO - Coletando até 18:00")
    else:
        print(f"[INFO] Fora do horário de mercado - Aguardará abertura")
    
    print("="*70 + "\n")
    
    # Configurar handler de sinal
    signal.signal(signal.SIGINT, signal_handler)
    
    collector = ContinuousBookCollector()
    
    if not collector.initialize():
        print("\n[ERRO] Falha na inicialização")
        return 1
        
    # Aguardar estabilização
    print("\nAguardando estabilização do sistema...")
    time.sleep(3)
    
    # Subscrever WDO
    print("\nSubscrevendo WDOU25...")
    collector.subscribe_wdo()
    
    # Aguardar dados começarem
    time.sleep(2)
    
    # Executar coleta contínua
    print("\nIniciando coleta contínua...")
    print("Pressione Ctrl+C para parar\n")
    
    try:
        collector.run_continuous()
    finally:
        # Finalizar
        collector.cleanup()
        
    print("\n[FIM] Coleta finalizada")
    print(f"Dados salvos em: data/realtime/book/{datetime.now().strftime('%Y%m%d')}/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())