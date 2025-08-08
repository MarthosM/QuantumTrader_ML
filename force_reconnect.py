"""
Script para forçar reconexão e testar diferentes configurações
"""
import os
import sys
import time
import ctypes
from ctypes import *
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Estruturas
class TAssetIDRec(Structure):
    _fields_ = [
        ("ticker", c_wchar * 35),
        ("bolsa", c_wchar * 15),
    ]

# Variáveis globais
data_received = False
last_price = 0
callback_count = 0

# Callbacks
@WINFUNCTYPE(None, c_int32, c_int32)
def stateCallback(nType, nResult):
    states = {0: "Login", 1: "Broker", 2: "Market", 3: "Ativacao"}
    print(f"[STATE] {states.get(nType, f'Type{nType}')}: {nResult}")
    return None

@WINFUNCTYPE(None, POINTER(TAssetIDRec), c_double, c_int, c_int)
def tinyBookCallBack(assetId, price, qtd, side):
    global data_received, last_price, callback_count
    callback_count += 1
    
    if price > 0 and price < 10000:
        data_received = True
        last_price = float(price)
        
        # Sempre mostrar os primeiros 5 e depois a cada 100
        if callback_count <= 5 or callback_count % 100 == 0:
            side_str = "BID" if side == 0 else "ASK"
            print(f'[TINY #{callback_count}] {datetime.now().strftime("%H:%M:%S")} - Price: R$ {price:.2f} {side_str} x {qtd}')
    return None

@WINFUNCTYPE(None, POINTER(TAssetIDRec), c_wchar_p, c_double, c_double, 
            c_double, c_double, c_double, c_double, c_double, c_double, 
            c_double, c_double, c_int, c_int, c_int, c_int, c_int, c_int, c_int)
def dailyCallback(assetId, date, sOpen, sHigh, sLow, sClose, sVol, sAjuste, 
                 sMaxLimit, sMinLimit, sVolBuyer, sVolSeller, nQtd, nNegocios, 
                 nContratosOpen, nQtdBuyer, nQtdSeller, nNegBuyer, nNegSeller):
    global data_received
    data_received = True
    print(f"[DAILY] {datetime.now().strftime('%H:%M:%S')} - Close: {sClose:.2f} Vol: {sVol}")
    return None

def test_connection(ticker="WDOU25", wait_time=30):
    """Testa conexão com um ticker específico"""
    global data_received, last_price, callback_count
    
    # Reset
    data_received = False
    last_price = 0
    callback_count = 0
    
    print(f"\n{'='*60}")
    print(f"TESTANDO TICKER: {ticker}")
    print(f"Hora: {datetime.now()}")
    print(f"{'='*60}")
    
    # Carregar DLL
    dll_path = os.getenv('PROFIT_DLL_PATH', './ProfitDLL64.dll')
    dll = WinDLL(dll_path)
    
    # Login
    key = c_wchar_p("HMARL")
    user = c_wchar_p(os.getenv('PROFIT_USERNAME', '29936354842'))
    pwd = c_wchar_p(os.getenv('PROFIT_PASSWORD', 'Ultra3376!'))
    
    print("Fazendo login...")
    
    result = dll.DLLInitializeLogin(
        key, user, pwd,
        stateCallback,    # state
        None,             # history
        None,             # orderChange
        None,             # account
        None,             # accountInfo
        dailyCallback,    # daily
        None,             # priceBook
        None,             # offerBook
        None,             # historyTrade
        None,             # progress
        tinyBookCallBack  # tinyBook
    )
    
    print(f"Login result: {result}")
    
    if result != 0:
        print(f"ERRO no login: {result}")
        dll.DLLFinalize()
        return False
    
    # Aguardar conexão
    print("Aguardando conexão...")
    time.sleep(5)
    
    # Subscrever
    print(f"Subscrevendo {ticker}...")
    
    result = dll.SubscribeTicker(c_wchar_p(ticker), c_wchar_p("F"))
    print(f"Subscribe result: {result}")
    
    if result != 0:
        print(f"ERRO ao subscrever: {result}")
        dll.DLLFinalize()
        return False
    
    # Monitorar
    print(f"\nMonitorando por {wait_time} segundos...")
    print("Aguardando dados...\n")
    
    start = time.time()
    last_update = start
    
    while (time.time() - start) < wait_time:
        current = time.time()
        
        # Status a cada 5 segundos
        if (current - last_update) > 5:
            if data_received:
                print(f"\n[STATUS] {datetime.now().strftime('%H:%M:%S')} - Recebendo dados! Último preço: R$ {last_price:.2f}")
                print(f"Total callbacks: {callback_count}")
            else:
                print(f"\n[AGUARDANDO] {datetime.now().strftime('%H:%M:%S')} - Nenhum dado recebido ainda...")
            last_update = current
            
        time.sleep(0.1)
    
    # Resultado
    print(f"\n{'='*60}")
    print(f"RESULTADO PARA {ticker}:")
    print(f"Dados recebidos: {'SIM' if data_received else 'NÃO'}")
    print(f"Total callbacks: {callback_count}")
    print(f"Último preço: R$ {last_price:.2f}")
    print(f"{'='*60}")
    
    # Finalizar
    dll.DLLFinalize()
    time.sleep(2)
    
    return data_received

def main():
    print("\n*** TESTE DE RECONEXÃO E TICKERS ***")
    print(f"Hora atual: {datetime.now()}")
    
    # Lista de tickers para testar
    tickers_to_test = [
        "WDOU25",  # Setembro
        "WDOV25",  # Outubro  
        "WINQ25",  # WIN agosto
        "WINU25",  # WIN setembro
        "DOLU25",  # Dólar setembro
    ]
    
    results = {}
    
    # Testar cada ticker
    for ticker in tickers_to_test:
        success = test_connection(ticker, wait_time=20)
        results[ticker] = success
        
        # Pausa entre testes
        if ticker != tickers_to_test[-1]:
            print("\nAguardando 5 segundos antes do próximo teste...")
            time.sleep(5)
    
    # Resumo
    print("\n" + "="*60)
    print("RESUMO DOS TESTES:")
    print("="*60)
    
    for ticker, success in results.items():
        status = "✓ FUNCIONANDO" if success else "✗ SEM DADOS"
        print(f"{ticker}: {status}")
    
    # Recomendação
    working_tickers = [t for t, s in results.items() if s]
    if working_tickers:
        print(f"\nRECOMENDAÇÃO: Use {working_tickers[0]} que está funcionando!")
    else:
        print("\nATENÇÃO: Nenhum ticker está recebendo dados. Possíveis causas:")
        print("1. Mercado fechado")
        print("2. Problema de conectividade")
        print("3. Manutenção do servidor")
        print("4. Firewall/Antivírus bloqueando")

if __name__ == "__main__":
    main()