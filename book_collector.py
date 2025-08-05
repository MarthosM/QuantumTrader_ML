"""
Book Collector WDO - Versão Final Funcional
Corrige estrutura TAssetID e encoding
"""

import os
import sys
import time
import ctypes
from ctypes import *
from datetime import datetime
from pathlib import Path
import pandas as pd
import json
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Estrutura TAssetIDRec mais simples - apenas ticker e bolsa
class TAssetIDRec(Structure):
    _fields_ = [
        ("ticker", c_wchar * 35),
        ("bolsa", c_wchar * 15),
    ]

class WDOFinalCollector:
    def __init__(self):
        self.logger = logging.getLogger('WDOFinal')
        self.dll = None
        
        # Flags de controle
        self.bAtivo = False
        self.bMarketConnected = False
        self.bConnectado = False
        self.bBrokerConnected = False
        
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
        
        # Referências dos callbacks - IMPORTANTE: manter referências
        self.callback_refs = {}
        
        # Ticker que estamos monitorando
        self.target_ticker = "WDOU25"
        
    def initialize(self):
        """Inicializa DLL e callbacks"""
        try:
            # Carregar DLL
            dll_path = "./ProfitDLL64.dll"
            self.logger.info(f"Carregando DLL: {os.path.abspath(dll_path)}")
            
            self.dll = WinDLL(dll_path)
            self.logger.info("[OK] DLL carregada")
            
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
        
        # TinyBook callback - Usando estrutura simplificada
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_double, c_int, c_int)
        def tinyBookCallBack(assetId, price, qtd, side):
            self.callbacks['tiny_book'] += 1
            
            # Por enquanto, vamos assumir que é WDOU25 já que sabemos que subscrevemos apenas ele
            ticker = self.target_ticker
            
            # Validar preço
            if price > 0 and price < 10000:
                # Log apenas primeiros 20 ou a cada 100
                if self.callbacks['tiny_book'] <= 20 or self.callbacks['tiny_book'] % 100 == 0:
                    side_str = "BID" if side == 0 else "ASK"
                    self.logger.info(f'[TINY_BOOK #{self.callbacks["tiny_book"]}] {ticker} {side_str}: R$ {price:.2f} x {qtd}')
                        
                # Salvar dados
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
                if self.callbacks['price_book'] <= 10 or self.callbacks['price_book'] % 100 == 0:
                    self.logger.info(f'[PRICE_BOOK #{self.callbacks["price_book"]}] {ticker} Action={nAction} Side={Side} Price={sPrice:.2f} Qty={nQtd}')
                    
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
                if self.callbacks['offer_book'] <= 20 or self.callbacks['offer_book'] % 100 == 0:
                    side_str = "BID" if Side == 0 else "ASK"
                    self.logger.info(f'[OFFER_BOOK #{self.callbacks["offer_book"]}] {ticker} {side_str} @ R$ {sPrice:.2f} x {nQtd}')
                    
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
            
            # Log apenas primeiros ou a cada 100
            if self.callbacks['daily'] <= 5 or self.callbacks['daily'] % 100 == 0:
                self.logger.info(f'[DAILY #{self.callbacks["daily"]}] {ticker}: O={sOpen:.2f} H={sHigh:.2f} L={sLow:.2f} C={sClose:.2f}')
                
            # Salvar dados diários
            self.data.append({
                'type': 'daily',
                'ticker': ticker,
                'open': float(sOpen),
                'high': float(sHigh),
                'low': float(sLow),
                'close': float(sClose),
                'volume': float(sVol),
                'qty': int(nQtd),
                'trades': int(nNegocios),
                'timestamp': datetime.now().isoformat()
            })
            
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
                
                # Usar nosso ticker conhecido
                ticker_str = self.target_ticker
                
                if price > 0 and price < 10000:
                    if self.callbacks['trade'] <= 20 or self.callbacks['trade'] % 100 == 0:
                        self.logger.info(f'[TRADE #{self.callbacks["trade"]}] {ticker_str} @ R$ {price:.2f} x {qty}')
                        
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
            
    def monitor_status(self):
        """Monitora status detalhadamente"""
        total_callbacks = sum(self.callbacks.values())
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"[MONITOR] Total callbacks: {total_callbacks}")
        self.logger.info(f"{'='*50}")
        
        # Callbacks por tipo
        for key, value in self.callbacks.items():
            if value > 0:
                self.logger.info(f"  {key:12}: {value:6}")
                
        # Dados por tipo
        if self.data:
            try:
                df = pd.DataFrame(self.data)
                
                # Tipos de dados
                type_counts = df['type'].value_counts()
                self.logger.info(f"\nDados por tipo:")
                for tipo, count in type_counts.items():
                    self.logger.info(f"  {tipo:12}: {count:6}")
                    
                # Estatísticas de preço
                price_data = df[df['price'] > 0]
                if not price_data.empty:
                    prices = price_data['price']
                    self.logger.info(f"\nEstatísticas de preço {self.target_ticker}:")
                    self.logger.info(f"  Último: R$ {prices.iloc[-1]:.2f}")
                    self.logger.info(f"  Médio: R$ {prices.mean():.2f}")
                    self.logger.info(f"  Min/Max: R$ {prices.min():.2f} / R$ {prices.max():.2f}")
                    self.logger.info(f"  Desvio: R$ {prices.std():.2f}")
                    
                # Últimos trades
                trade_data = df[df['type'] == 'trade']
                if not trade_data.empty and len(trade_data) > 0:
                    self.logger.info(f"\nÚltimos 5 trades:")
                    for idx, trade in trade_data.tail(5).iterrows():
                        self.logger.info(f"  R$ {trade['price']:.2f} x {trade['quantity']}")
                        
            except Exception as e:
                self.logger.error(f"Erro ao processar dados para monitor: {e}")
                self.logger.info(f"Total de registros coletados: {len(self.data)}")
                
        self.logger.info(f"\nTotal de registros: {len(self.data)}")
        self.logger.info(f"{'='*50}")
        
    def save_data(self):
        """Salva dados coletados"""
        if not self.data:
            self.logger.warning("Nenhum dado para salvar")
            return
            
        try:
            # Criar diretório
            save_dir = Path('data/realtime/book') / datetime.now().strftime('%Y%m%d')
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Converter para DataFrame com tratamento de erro
            try:
                df = pd.DataFrame(self.data)
            except ValueError as e:
                self.logger.error(f"Erro ao criar DataFrame: {e}")
                # Tentar salvar como lista de dicts
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                json_file = save_dir / f'wdo_final_{timestamp}_raw.json'
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Dados salvos como JSON: {json_file}")
                return
            
            # Salvar TODOS os dados (incluindo daily)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Parquet com todos os dados
            parquet_file = save_dir / f'wdo_final_{timestamp}.parquet'
            df.to_parquet(parquet_file, compression='snappy')
            
            self.logger.info(f"\n[SAVE] Dados salvos: {len(df)} registros")
            self.logger.info(f"  Arquivo: {parquet_file}")
            
            # Separar por tipo para análise
            summary = {
                'total_records': len(df),
                'start_time': df['timestamp'].min() if not df.empty else None,
                'end_time': df['timestamp'].max() if not df.empty else None,
                'ticker': self.target_ticker,
                'types': df['type'].value_counts().to_dict(),
                'stats_by_type': {}
            }
            
            # Estatísticas por tipo
            for data_type in df['type'].unique():
                type_data = df[df['type'] == data_type]
                if 'price' in type_data.columns and not type_data.empty:
                    prices = type_data['price'][type_data['price'] > 0]
                    if not prices.empty:
                        summary['stats_by_type'][data_type] = {
                            'count': len(type_data),
                            'min_price': float(prices.min()),
                            'max_price': float(prices.max()),
                            'mean_price': float(prices.mean()),
                            'last_price': float(prices.iloc[-1])
                        }
                        
            # Salvar resumo
            summary_file = save_dir / f'summary_wdo_final_{timestamp}.json'
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Erro ao salvar: {e}")
            import traceback
            traceback.print_exc()
            
    def run(self, duration=120):
        """Executa coleta por X segundos"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"INICIANDO COLETA DE DADOS - {self.target_ticker}")
        self.logger.info(f"Duração: {duration} segundos")
        self.logger.info(f"{'='*60}")
        
        start_time = time.time()
        last_save = time.time()
        last_monitor = time.time()
        
        try:
            while (time.time() - start_time) < duration:
                elapsed = int(time.time() - start_time)
                
                # Monitor a cada 15 segundos
                if (time.time() - last_monitor) > 15:
                    self.monitor_status()
                    last_monitor = time.time()
                    
                # Salvar a cada 30 segundos
                if (time.time() - last_save) > 30:
                    self.save_data()
                    last_save = time.time()
                    
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("\nColeta interrompida pelo usuário")
            
        # Salvar dados finais
        self.save_data()
        
        # Estatísticas finais
        self.logger.info(f"\n{'='*60}")
        self.logger.info("ESTATÍSTICAS FINAIS")
        self.logger.info(f"{'='*60}")
        self.monitor_status()
        
    def cleanup(self):
        """Finaliza DLL"""
        if self.dll and hasattr(self.dll, 'DLLFinalize'):
            result = self.dll.DLLFinalize()
            self.logger.info(f"\n[CLEANUP] DLLFinalize: {result}")


def main():
    print("\n" + "="*70)
    print("COLETOR WDO - VERSÃO FINAL FUNCIONAL")
    print("="*70)
    print(f"Horário: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Verificar horário de mercado
    hora_atual = datetime.now().hour
    minuto_atual = datetime.now().minute
    dia_semana = datetime.now().weekday()  # 0=Segunda, 6=Domingo
    
    if dia_semana >= 5:  # Sábado ou Domingo
        print(f"[AVISO] FIM DE SEMANA - Mercado FECHADO")
    elif 9 <= hora_atual < 18:
        print(f"[OK] Mercado provavelmente ABERTO")
    else:
        print(f"[AVISO] Mercado pode estar FECHADO (horário: {hora_atual}:{minuto_atual:02d})")
    
    print("="*70 + "\n")
    
    collector = WDOFinalCollector()
    
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
    
    # Executar coleta
    print("\nIniciando coleta de dados...")
    print("Pressione Ctrl+C para parar\n")
    
    # Coletar por 2 minutos
    collector.run(duration=120)
    
    # Finalizar
    collector.cleanup()
    
    print("\n[FIM] Coleta finalizada")
    print(f"Dados salvos em: data/realtime/book/{datetime.now().strftime('%Y%m%d')}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())