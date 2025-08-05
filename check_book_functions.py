"""
Verifica funções de book disponíveis na DLL
"""

import ctypes
from pathlib import Path

def check_book_functions():
    """Verifica todas as funções relacionadas a book"""
    
    dll_path = Path("ProfitDLL64.dll").absolute()
    dll = ctypes.WinDLL(str(dll_path))
    
    # Lista de possíveis funções de book
    book_functions = [
        # Conforme o guia
        'DLL_SetBookOfferCallback',
        'DLL_EnableBookOffer',
        'DLL_SubscribeBookOffer',
        
        # Variações que encontramos
        'SetOfferBookCallback',
        'SetOfferBookCallbackV2',
        'SetPriceBookCallback',
        'SetPriceBookCallbackV2',
        
        # Outras possibilidades
        'EnableBookOffer',
        'SubscribeBookOffer',
        'SetBookCallback',
        'SetOrderBookCallback',
        'EnableOrderBook',
        'SubscribeOrderBook',
        
        # Funções de subscrição
        'SubscribeTicker',
        'SubscribeMarketData',
        'Subscribe',
        
        # Funções genéricas
        'SetCallback',
        'RegisterCallback'
    ]
    
    print("="*60)
    print("FUNÇÕES DE BOOK NA DLL")
    print("="*60)
    
    found = []
    not_found = []
    
    for func_name in book_functions:
        try:
            func = getattr(dll, func_name)
            found.append(func_name)
            print(f"[OK] {func_name}")
        except AttributeError:
            not_found.append(func_name)
            
    print("\n" + "-"*60)
    print(f"Encontradas: {len(found)}")
    print(f"Não encontradas: {len(not_found)}")
    
    # Mostrar agrupadas
    print("\n" + "="*60)
    print("FUNÇÕES DISPONÍVEIS PARA BOOK:")
    print("="*60)
    
    # Agrupar por tipo
    callbacks = [f for f in found if 'Callback' in f]
    subscribe = [f for f in found if 'Subscribe' in f]
    enable = [f for f in found if 'Enable' in f]
    others = [f for f in found if f not in callbacks + subscribe + enable]
    
    if callbacks:
        print("\nCallbacks:")
        for f in callbacks:
            print(f"  - {f}")
            
    if subscribe:
        print("\nSubscrição:")
        for f in subscribe:
            print(f"  - {f}")
            
    if enable:
        print("\nHabilitação:")
        for f in enable:
            print(f"  - {f}")
            
    if others:
        print("\nOutras:")
        for f in others:
            print(f"  - {f}")

if __name__ == "__main__":
    check_book_functions()