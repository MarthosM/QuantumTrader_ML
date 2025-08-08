"""
Teste de Conexão com ProfitDLL
Verifica se consegue conectar e receber dados
"""

import os
import ctypes
from ctypes import *
import time
from datetime import datetime
from dotenv import load_dotenv

# Carregar variáveis
load_dotenv()

# Estrutura
class TAssetIDRec(Structure):
    _fields_ = [
        ("ticker", c_wchar * 35),
        ("bolsa", c_wchar * 15),
    ]

# Variáveis globais
connected = False
market_connected = False
data_received = False

# Callbacks
@WINFUNCTYPE(None, c_int32, c_int32)
def stateCallback(nType, nResult):
    global connected, market_connected
    
    states = {0: "Login", 1: "Broker", 2: "Market", 3: "Activation"}
    results = {
        0: "OK",
        1: "Erro genérico",
        200: "Credenciais inválidas",
        201: "Usuário bloqueado"
    }
    
    print(f"[STATE] {states.get(nType, f'Type{nType}')} = {results.get(nResult, f'Result{nResult}')} ({nResult})")
    
    if nType == 0:  # Login
        connected = (nResult == 0)
    elif nType == 2:  # Market
        market_connected = (nResult in [2, 3, 4])

@WINFUNCTYPE(None, POINTER(TAssetIDRec), c_double, c_int, c_int)
def tinyBookCallBack(assetId, price, qtd, side):
    global data_received
    if 0 < price < 10000:
        data_received = True
        side_str = "BID" if side == 0 else "ASK"
        print(f"[PRICE] {side_str}: R$ {price:.2f} x {qtd}")

def main():
    print("\n" + "="*60)
    print("TESTE DE CONEXÃO PROFITDLL")
    print("="*60)
    print(f"Data: {datetime.now()}")
    print("="*60)
    
    # Mostrar credenciais (sem senha)
    print("\nCredenciais:")
    print(f"PROFIT_KEY: {os.getenv('PROFIT_KEY', 'NÃO DEFINIDO')}")
    print(f"PROFIT_USERNAME: {os.getenv('PROFIT_USERNAME', 'NÃO DEFINIDO')}")
    print(f"PROFIT_PASSWORD: {'*' * len(os.getenv('PROFIT_PASSWORD', '')) if os.getenv('PROFIT_PASSWORD') else 'NÃO DEFINIDO'}")
    print("="*60)
    
    # Usar credenciais do .env
    key = os.getenv('PROFIT_KEY', 'HMARL')
    username = os.getenv('PROFIT_USERNAME', '')
    password = os.getenv('PROFIT_PASSWORD', '')
    
    # Testar DLL local primeiro
    dll_paths = [
        './ProfitDLL64.dll',
        'ProfitDLL64.dll',
        os.getenv('PROFIT_DLL_PATH', '')
    ]
    
    dll = None
    dll_path_used = None
    
    for path in dll_paths:
        if path and os.path.exists(path):
            try:
                print(f"\n[1] Tentando carregar DLL: {path}")
                dll = WinDLL(path)
                dll_path_used = path
                print(f"[OK] DLL carregada: {path}")
                break
            except Exception as e:
                print(f"[ERRO] Falha ao carregar {path}: {e}")
    
    if not dll:
        print("\n[ERRO] Nenhuma DLL encontrada!")
        return 1
    
    # Login
    print(f"\n[2] Fazendo login...")
    print(f"Key: {key}")
    print(f"Username: {username}")
    
    result = dll.DLLInitializeLogin(
        c_wchar_p(key),
        c_wchar_p(username),
        c_wchar_p(password),
        stateCallback,
        None, None, None, None, None, None, None, None, None,
        tinyBookCallBack
    )
    
    print(f"[3] Resultado do login: {result}")
    
    # Aguardar callbacks
    print("\n[4] Aguardando conexão (15 segundos)...")
    
    timeout = 15
    start = time.time()
    
    while (time.time() - start) < timeout:
        if connected and market_connected:
            print("\n[OK] CONECTADO COM SUCESSO!")
            break
        time.sleep(0.5)
    
    if not connected:
        print("\n[ERRO] Falha na conexão")
        print("Possíveis causas:")
        print("1. ProfitChart não está aberto")
        print("2. Credenciais incorretas")
        print("3. Conta bloqueada ou sem permissão")
        return 1
    
    # Subscrever ticker
    print("\n[5] Subscrevendo WDOU25...")
    result = dll.SubscribeTicker(c_wchar_p("WDOU25"), c_wchar_p("F"))
    print(f"Resultado: {result}")
    
    # Aguardar dados
    print("\n[6] Aguardando dados (10 segundos)...")
    time.sleep(10)
    
    # Status final
    print("\n" + "="*60)
    print("RESULTADO DO TESTE")
    print("="*60)
    print(f"DLL: {dll_path_used}")
    print(f"Login: {'OK' if connected else 'FALHOU'}")
    print(f"Market: {'OK' if market_connected else 'FALHOU'}")
    print(f"Dados: {'RECEBENDO' if data_received else 'NÃO RECEBIDO'}")
    print("="*60)
    
    # Finalizar
    if hasattr(dll, 'DLLFinalize'):
        dll.DLLFinalize()
    
    return 0

if __name__ == "__main__":
    exit(main())