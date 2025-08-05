"""
Script para listar todas as funções exportadas pela DLL
"""

import ctypes
import subprocess
import os
from pathlib import Path

def list_dll_exports_dumpbin():
    """Usa dumpbin se disponível"""
    try:
        result = subprocess.run(
            ['dumpbin', '/exports', 'ProfitDLL64.dll'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("Funções exportadas (via dumpbin):")
            for line in result.stdout.split('\n'):
                if line.strip() and not line.startswith(' '):
                    print(line)
            return True
    except:
        pass
    return False

def list_dll_exports_manual():
    """Lista funções conhecidas do ProfitDLL"""
    print("Testando funções conhecidas do ProfitDLL:")
    
    dll_path = Path("ProfitDLL64.dll").absolute()
    dll = ctypes.WinDLL(str(dll_path))
    
    # Lista de possíveis nomes de funções
    possible_functions = [
        # Login/Inicialização
        'DLLInitializeLogin',
        'DLLInitialize',
        'Initialize',
        'InitializeLogin',
        
        # Conexão
        'SendLogin',
        'Login',
        'Connect',
        'SendConnect',
        
        # Market Data
        'SubscribeTicker',
        'Subscribe',
        'SubscribeMarketData',
        
        # Callbacks - Várias possibilidades
        'SetAggTradeCallback',
        'SetAggradeCallback',
        'SetTradeCallback',
        'SetNewTradeCallback',
        'SetTradeCallbackV2',
        
        'SetTickerCallback',
        'SetTickCallback', 
        'SetPriceCallback',
        'SetNewTickerCallback',
        
        'SetOfferBookCallback',
        'SetOfferBookCallbackV2',
        'SetBookCallback',
        
        'SetPriceBookCallback',
        'SetPriceBookCallbackV2',
        
        # History
        'GetHistoryTrades',
        'GetHistory',
        'RequestHistory',
        
        # Orders
        'SendOrder',
        'SendNewOrder',
        'CancelOrder',
        
        # Outras
        'GetAgentNameByID',
        'GetLastError',
        'GetErrorMessage',
        'Disconnect',
        'SendLogout'
    ]
    
    found_functions = []
    
    for func_name in sorted(possible_functions):
        try:
            func = getattr(dll, func_name)
            found_functions.append(func_name)
            print(f"  [OK] {func_name}")
        except AttributeError:
            pass
            
    print(f"\nTotal de funções encontradas: {len(found_functions)}")
    
    # Salvar lista de funções
    with open("profitdll_functions.txt", "w") as f:
        f.write("Funções encontradas na ProfitDLL64.dll:\n")
        f.write("="*50 + "\n")
        for func in found_functions:
            f.write(f"{func}\n")
            
    print(f"\nLista salva em: profitdll_functions.txt")
    
    return found_functions

def check_callback_variations():
    """Testa variações específicas de callbacks"""
    print("\n" + "="*60)
    print("TESTANDO VARIAÇÕES DE CALLBACKS")
    print("="*60)
    
    dll_path = Path("ProfitDLL64.dll").absolute()
    dll = ctypes.WinDLL(str(dll_path))
    
    # Variações de callbacks para testar
    callback_variations = {
        'Trade': [
            'SetTradeCallback',
            'SetNewTradeCallback', 
            'SetTradeCallbackV2',
            'SetAggTradeCallback',
            'SetAggradeCallback',
            'SetNewAggTradeCallback',
            'SetHistoryTradeCallback'
        ],
        'Ticker': [
            'SetTickerCallback',
            'SetTickCallback',
            'SetPriceCallback',
            'SetNewTickerCallback',
            'SetTickerCallbackV2',
            'SetNewPriceCallback'
        ],
        'Book': [
            'SetOfferBookCallback',
            'SetOfferBookCallbackV2',
            'SetBookCallback',
            'SetOrderBookCallback',
            'SetPriceBookCallback',
            'SetPriceBookCallbackV2'
        ]
    }
    
    found_callbacks = {}
    
    for category, variations in callback_variations.items():
        print(f"\n{category} Callbacks:")
        found_callbacks[category] = []
        
        for callback in variations:
            try:
                func = getattr(dll, callback)
                found_callbacks[category].append(callback)
                print(f"  [OK] {callback}")
            except AttributeError:
                pass
                
        if not found_callbacks[category]:
            print(f"  [ERRO] Nenhum callback de {category} encontrado")

    return found_callbacks

def main():
    print("="*60)
    print("ANÁLISE DE FUNÇÕES DA PROFITDLL")
    print("="*60)
    
    # Verificar se DLL existe
    if not Path("ProfitDLL64.dll").exists():
        print("[ERRO] ProfitDLL64.dll não encontrada!")
        return
        
    # Tentar listar exports com dumpbin
    if not list_dll_exports_dumpbin():
        print("dumpbin não disponível, usando método manual\n")
        
    # Listar funções manualmente
    functions = list_dll_exports_manual()
    
    # Verificar callbacks específicos
    callbacks = check_callback_variations()
    
    # Resumo
    print("\n" + "="*60)
    print("RESUMO DE CALLBACKS ENCONTRADOS")
    print("="*60)
    
    for category, found in callbacks.items():
        if found:
            print(f"\n{category}: {', '.join(found)}")
        else:
            print(f"\n{category}: NENHUM ENCONTRADO")

if __name__ == "__main__":
    main()