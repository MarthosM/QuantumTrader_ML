"""
Monitor completo de dados - Testa todos os callbacks simultaneamente
"""

import os
import sys
import time
import ctypes
from ctypes import c_int, c_int64, c_double, c_char, c_char_p, c_wchar_p, c_void_p, WINFUNCTYPE, Structure
from pathlib import Path
from datetime import datetime
import threading
from collections import defaultdict

# Estrutura TAssetIDRec
class TAssetIDRec(Structure):
    _fields_ = [
        ("pwcTicker", c_wchar_p),
        ("pwcBolsa", c_wchar_p),
        ("nFeed", c_int)
    ]

# Estatísticas globais
stats = defaultdict(int)
last_data = {}

def format_time():
    return datetime.now().strftime("%H:%M:%S")

def main():
    print("\n" + "="*60)
    print("MONITOR COMPLETO DE DADOS PROFITDLL")
    print("="*60)
    print(f"Início: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("="*60)
    
    try:
        # Carregar DLL
        dll_path = Path("ProfitDLL64.dll").absolute()
        dll = ctypes.WinDLL(str(dll_path))
        
        # Configurar tipos
        dll.DLLInitializeLogin.argtypes = [c_char_p, c_char_p, c_char_p]
        dll.DLLInitializeLogin.restype = c_int
        
        # Inicializar
        result = dll.DLLInitializeLogin(b"HMARL", b"29936354842", b"L30n@rd0sp")
        print(f"\nDLLInitializeLogin: {result} {'[OK]' if result == 0 else '[ERRO]'}")
        
        if result != 0:
            return 1
            
        print("\nConfigurando callbacks...")
        
        # 1. Trade Callback
        TRADE_CALLBACK = WINFUNCTYPE(None, c_char_p, c_double, c_int, c_int, c_int)
        
        @TRADE_CALLBACK
        def trade_callback(ticker, price, quantity, buyer, seller):
            stats['trades'] += 1
            ticker_str = ticker.decode() if ticker else 'N/A'
            last_data['trade'] = {
                'ticker': ticker_str,
                'price': price,
                'qty': quantity,
                'time': format_time()
            }
            if stats['trades'] <= 3:
                print(f"[{format_time()}] TRADE: {ticker_str} - "
                      f"Preço: {price:.2f} | Qtd: {quantity}")
                
        dll.SetTradeCallback(trade_callback)
        print("[OK] SetTradeCallback")
        
        # 2. State Callback
        STATE_CALLBACK = WINFUNCTYPE(None, c_int)
        
        @STATE_CALLBACK
        def state_callback(status):
            stats['states'] += 1
            last_data['state'] = status
            print(f"[{format_time()}] STATE: Código {status}")
            
        dll.SetStateCallback(state_callback)
        print("[OK] SetStateCallback")
        
        # 3. Offer Book Callback V2
        OFFER_BOOK_CALLBACK_V2 = WINFUNCTYPE(
            None, TAssetIDRec, c_int, c_int, c_int, c_int64, c_int,
            c_int64, c_double, c_char, c_char, c_char, c_char, c_char,
            c_wchar_p, c_void_p, c_void_p
        )
        
        @OFFER_BOOK_CALLBACK_V2
        def offer_book_callback(asset_id, action, position, side, qtd, agent,
                               offer_id, price, has_price, has_qtd, has_date,
                               has_offer_id, has_agent, date, array_sell, array_buy):
            stats['offer_book'] += 1
            if stats['offer_book'] <= 3:
                ticker = asset_id.pwcTicker if asset_id.pwcTicker else 'N/A'
                side_str = 'BID' if side == 0 else 'ASK'
                print(f"[{format_time()}] OFFER BOOK: {ticker} - "
                      f"{side_str} | Preço: {price:.2f} | Qtd: {qtd}")
                      
        dll.SetOfferBookCallbackV2(offer_book_callback)
        print("[OK] SetOfferBookCallbackV2")
        
        # 4. Price Book Callback V2
        PRICE_BOOK_CALLBACK_V2 = WINFUNCTYPE(
            None, TAssetIDRec, c_int, c_int, c_int, c_int64, c_int,
            c_int64, c_double, c_char, c_char, c_char, c_char, c_char,
            c_wchar_p, c_void_p, c_void_p
        )
        
        @PRICE_BOOK_CALLBACK_V2
        def price_book_callback(asset_id, action, position, side, qtd, agent,
                               offer_id, price, has_price, has_qtd, has_date,
                               has_offer_id, has_agent, date, array_sell, array_buy):
            stats['price_book'] += 1
            if stats['price_book'] <= 3:
                ticker = asset_id.pwcTicker if asset_id.pwcTicker else 'N/A'
                print(f"[{format_time()}] PRICE BOOK: {ticker} - "
                      f"Preço: {price:.2f}")
                      
        dll.SetPriceBookCallbackV2(price_book_callback)
        print("[OK] SetPriceBookCallbackV2")
        
        # 5. History Callback (se existir)
        if hasattr(dll, 'SetHistoryCallback'):
            HISTORY_CALLBACK = WINFUNCTYPE(None, c_char_p, c_double, c_int, c_int64)
            
            @HISTORY_CALLBACK
            def history_callback(ticker, price, quantity, timestamp):
                stats['history'] += 1
                if stats['history'] <= 3:
                    print(f"[{format_time()}] HISTORY: {ticker.decode()} - "
                          f"Preço: {price:.2f}")
                          
            dll.SetHistoryCallback(history_callback)
            print("[OK] SetHistoryCallback")
            
        # Subscrever a múltiplos tickers
        print("\nSubscrevendo tickers...")
        
        tickers = ["WDOU25", "WINQ25", "PETR4", "VALE3"]
        dll.SubscribeTicker.argtypes = [c_char_p]
        dll.SubscribeTicker.restype = c_int
        
        for ticker in tickers:
            result = dll.SubscribeTicker(ticker.encode())
            status = "[OK]" if result == 0 else f"[AVISO: {result}]"
            print(f"  {ticker}: {status}")
            
        # Monitorar por 60 segundos
        print(f"\nMonitorando dados por 60 segundos...")
        print("(Pressione Ctrl+C para parar)\n")
        
        start_time = time.time()
        last_report = 0
        
        try:
            while time.time() - start_time < 60:
                current_time = time.time() - start_time
                
                # Relatório a cada 10 segundos
                if current_time - last_report >= 10:
                    print(f"\n--- Relatório {int(current_time)}s ---")
                    print(f"Trades: {stats['trades']}")
                    print(f"States: {stats['states']}")
                    print(f"Offer Book: {stats['offer_book']}")
                    print(f"Price Book: {stats['price_book']}")
                    print(f"History: {stats['history']}")
                    print(f"TOTAL: {sum(stats.values())}")
                    
                    if last_data.get('trade'):
                        t = last_data['trade']
                        print(f"Último trade: {t['ticker']} R${t['price']:.2f} "
                              f"({t['qty']} @ {t['time']})")
                              
                    last_report = current_time
                    
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\nInterrompido pelo usuário")
            
        # Relatório final
        print("\n" + "="*60)
        print("RELATÓRIO FINAL")
        print("="*60)
        
        total = sum(stats.values())
        print(f"\nTotal de callbacks recebidos: {total}")
        
        for callback_type, count in stats.items():
            if count > 0:
                percentage = (count / total) * 100 if total > 0 else 0
                print(f"  {callback_type}: {count} ({percentage:.1f}%)")
                
        if total > 0:
            print(f"\n[SUCESSO] Sistema recebendo dados!")
            print("Tipos de dados ativos:")
            for k, v in stats.items():
                if v > 0:
                    print(f"  - {k}")
        else:
            print("\n[ATENÇÃO] Nenhum dado recebido")
            print("\nPossíveis causas:")
            print("1. Domingo - mercado fechado")
            print("2. Fora do horário de pregão (09:00-18:00)")
            print("3. Conta sem permissão para dados em tempo real")
            print("4. Necessário contratar market data na corretora")
            
        # Desconectar
        if hasattr(dll, 'DLLFinalize'):
            dll.DLLFinalize()
            print("\nDLL finalizada")
            
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()
        
    return 0


if __name__ == "__main__":
    sys.exit(main())