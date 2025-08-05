"""
Inspeciona todas as funções disponíveis na ProfitDLL
"""

import ctypes
import os
from pathlib import Path

def inspect_dll():
    """Inspeciona completamente a DLL"""
    print("\n" + "="*60)
    print("INSPEÇÃO COMPLETA DA PROFITDLL")
    print("="*60)
    
    # Carregar DLL
    dll_path = Path('./ProfitDLL64.dll').absolute()
    print(f"\nDLL: {dll_path}")
    print(f"Existe: {dll_path.exists()}")
    print(f"Tamanho: {dll_path.stat().st_size if dll_path.exists() else 0} bytes")
    
    try:
        dll = ctypes.WinDLL(str(dll_path))
    except Exception as e:
        print(f"\nErro ao carregar DLL: {e}")
        return
        
    # Listar TODAS as funções
    print("\n" + "-"*60)
    print("TODAS AS FUNÇÕES DISPONÍVEIS:")
    print("-"*60)
    
    all_functions = []
    for attr in dir(dll):
        if not attr.startswith('_'):
            all_functions.append(attr)
            
    all_functions.sort()
    
    # Agrupar por categoria
    categories = {
        'Initialize': [],
        'Subscribe': [],
        'Get': [],
        'Set': [],
        'Send': [],
        'DLL': [],
        'Enable': [],
        'Request': [],
        'Others': []
    }
    
    for func in all_functions:
        categorized = False
        for prefix in ['Initialize', 'Subscribe', 'Get', 'Set', 'Send', 'DLL', 'Enable', 'Request']:
            if func.startswith(prefix):
                categories[prefix].append(func)
                categorized = True
                break
        if not categorized:
            categories['Others'].append(func)
            
    # Mostrar por categoria
    for category, funcs in categories.items():
        if funcs:
            print(f"\n{category} ({len(funcs)} funções):")
            for func in funcs:
                print(f"  - {func}")
                
    # Testar funções específicas de dados
    print("\n" + "-"*60)
    print("TESTANDO FUNÇÕES DE DADOS:")
    print("-"*60)
    
    # 1. GetTicker
    if hasattr(dll, 'GetTicker'):
        print("\n1. GetTicker:")
        try:
            dll.GetTicker.argtypes = [ctypes.c_int]
            dll.GetTicker.restype = ctypes.c_wchar_p
            for i in range(5):
                ticker = dll.GetTicker(i)
                if ticker:
                    print(f"   [{i}] = {ticker}")
        except Exception as e:
            print(f"   Erro: {e}")
            
    # 2. GetLastTrade
    if hasattr(dll, 'GetLastTrade'):
        print("\n2. GetLastTrade:")
        try:
            dll.GetLastTrade.argtypes = [ctypes.c_wchar_p]
            dll.GetLastTrade.restype = ctypes.c_double
            price = dll.GetLastTrade(b'WDOQ25')
            print(f"   WDOQ25: {price}")
        except Exception as e:
            print(f"   Erro: {e}")
            
    # 3. GetLastCotation
    if hasattr(dll, 'GetLastCotation'):
        print("\n3. GetLastCotation:")
        try:
            # Estrutura de cotação
            class Cotation(ctypes.Structure):
                _fields_ = [
                    ("last", ctypes.c_double),
                    ("bid", ctypes.c_double),
                    ("ask", ctypes.c_double),
                    ("volume", ctypes.c_double)
                ]
                
            dll.GetLastCotation.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(Cotation)]
            dll.GetLastCotation.restype = ctypes.c_int
            
            cot = Cotation()
            result = dll.GetLastCotation(ctypes.c_wchar_p('WDOQ25'), ctypes.pointer(cot))
            print(f"   Result: {result}")
            if result == 0:
                print(f"   Last: {cot.last} | Bid: {cot.bid} | Ask: {cot.ask}")
        except Exception as e:
            print(f"   Erro: {e}")
            
    # 4. Verificar versão
    version_funcs = ['GetVersion', 'GetDLLVersion', 'DLLVersion', 'Version']
    for func_name in version_funcs:
        if hasattr(dll, func_name):
            print(f"\n{func_name}:")
            try:
                func = getattr(dll, func_name)
                func.restype = ctypes.c_wchar_p
                version = func()
                print(f"   {version}")
            except:
                try:
                    func.restype = ctypes.c_int
                    version = func()
                    print(f"   {version}")
                except Exception as e:
                    print(f"   Erro: {e}")
                    
    # 5. Estado da conexão
    print("\n" + "-"*60)
    print("ESTADO DA CONEXÃO:")
    print("-"*60)
    
    state_funcs = ['GetState', 'GetConnectionState', 'IsConnected', 'GetMarketState']
    for func_name in state_funcs:
        if hasattr(dll, func_name):
            try:
                func = getattr(dll, func_name)
                func.restype = ctypes.c_int
                state = func()
                print(f"{func_name}: {state}")
            except Exception as e:
                print(f"{func_name}: Erro - {e}")
                
    # 6. Callbacks alternativos
    print("\n" + "-"*60)
    print("CALLBACKS ALTERNATIVOS:")
    print("-"*60)
    
    callback_patterns = [
        'Callback', 'Handler', 'Event', 'Notify', 'OnData', 'OnTrade', 'OnBook'
    ]
    
    found_callbacks = []
    for func in all_functions:
        if any(pattern in func for pattern in callback_patterns):
            found_callbacks.append(func)
            
    if found_callbacks:
        print("\nCallbacks encontrados:")
        for cb in sorted(found_callbacks):
            print(f"  - {cb}")
    else:
        print("\nNenhum callback alternativo encontrado")
        
    return dll


def test_alternative_data_access(dll):
    """Testa formas alternativas de acessar dados"""
    print("\n" + "="*60)
    print("TESTANDO ACESSO ALTERNATIVO AOS DADOS")
    print("="*60)
    
    # 1. Polling direto
    print("\n1. Tentando polling direto...")
    
    poll_funcs = [
        ('GetQuote', [ctypes.c_wchar_p], ctypes.c_double),
        ('GetPrice', [ctypes.c_wchar_p], ctypes.c_double),
        ('GetMarketData', [ctypes.c_wchar_p], ctypes.c_void_p),
        ('ReadTicker', [ctypes.c_wchar_p], ctypes.c_int)
    ]
    
    for func_name, argtypes, restype in poll_funcs:
        if hasattr(dll, func_name):
            print(f"\n{func_name}:")
            try:
                func = getattr(dll, func_name)
                func.argtypes = argtypes
                func.restype = restype
                
                if restype == ctypes.c_double:
                    result = func(b'WDOQ25')
                    print(f"  WDOQ25: {result}")
                else:
                    result = func(b'WDOQ25')
                    print(f"  Result: {result}")
            except Exception as e:
                print(f"  Erro: {e}")
                
    # 2. Requisições específicas
    print("\n2. Funções de requisição...")
    
    request_funcs = ['RequestData', 'RequestQuote', 'RequestMarketData', 'UpdateData']
    for func_name in request_funcs:
        if hasattr(dll, func_name):
            print(f"  ✓ {func_name} disponível")
            

def main():
    # Inspecionar DLL
    dll = inspect_dll()
    
    if dll:
        # Fazer login básico
        print("\n" + "="*60)
        print("FAZENDO LOGIN")
        print("="*60)
        
        dll.DLLInitializeLogin.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        dll.DLLInitializeLogin.restype = ctypes.c_int
        
        result = dll.DLLInitializeLogin(b'HMARL', b'29936354842', b'Ultrajiu33!')
        print(f"Login result: {result}")
        
        if result == 0:
            print("✓ Login bem sucedido")
            
            # Subscrever
            if hasattr(dll, 'SubscribeTicker'):
                dll.SubscribeTicker.argtypes = [ctypes.c_char_p]
                dll.SubscribeTicker.restype = ctypes.c_int
                result = dll.SubscribeTicker(b'WDOQ25')
                print(f"Subscribe WDOQ25: {result}")
                
            # Testar acesso alternativo
            test_alternative_data_access(dll)
            
        # Finalizar
        if hasattr(dll, 'DLLFinalize'):
            dll.DLLFinalize()
            

if __name__ == "__main__":
    main()