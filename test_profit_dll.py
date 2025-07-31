"""
Script de teste para verificar integração com ProfitDLL
"""

import os
import sys
from ctypes import WinDLL, c_int, c_wchar_p, c_char_p, WINFUNCTYPE
import time
import logging

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_profit_dll():
    """Testa conexão básica com ProfitDLL"""
    
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    
    print("="*60)
    print("TESTE DE CONEXÃO COM PROFITDLL")
    print("="*60)
    print(f"DLL Path: {dll_path}")
    print(f"Existe: {os.path.exists(dll_path)}")
    
    try:
        # 1. Carregar DLL
        print("\n1. Carregando DLL...")
        dll = WinDLL(dll_path)
        print("[OK] DLL carregada com sucesso")
        
        # 2. Listar funções disponíveis (tentativa)
        print("\n2. Verificando funções disponíveis...")
        
        # Funções documentadas no manual
        functions_to_test = [
            'DLLInitialize',
            'DLLInitializeLogin', 
            'DLLInitializeMarketLogin',
            'DLLFinalize',
            'GetHistoryTrades',
            'GetHistoryTradesReplay',
            'SetLoginStateCallback',
            'SetTradeCallback',
            'SetAssetListCallback'
        ]
        
        available_functions = []
        for func_name in functions_to_test:
            try:
                func = getattr(dll, func_name)
                available_functions.append(func_name)
                print(f"[OK] {func_name} - encontrada")
            except AttributeError:
                print(f"[X] {func_name} - não encontrada")
        
        print(f"\nFunções disponíveis: {len(available_functions)}/{len(functions_to_test)}")
        
        # 3. Tentar inicialização básica
        print("\n3. Tentando inicialização básica...")
        
        # Callback de estado
        @WINFUNCTYPE(None, c_int, c_int)
        def state_callback(state_type, state_value):
            logger.info(f"Estado recebido: tipo={state_type}, valor={state_value}")
        
        # Registrar callback se disponível
        if 'SetLoginStateCallback' in available_functions:
            dll.SetLoginStateCallback(state_callback)
            print("[OK] Callback de estado registrado")
        
        # Tentar DLLInitialize primeiro
        if 'DLLInitialize' in available_functions:
            print("\nTentando DLLInitialize()...")
            try:
                result = dll.DLLInitialize()
                print(f"DLLInitialize retornou: {result}")
            except Exception as e:
                print(f"Erro em DLLInitialize: {e}")
        
        # Aguardar um pouco
        time.sleep(2)
        
        # Tentar DLLInitializeMarketLogin
        if 'DLLInitializeMarketLogin' in available_functions:
            print("\nTentando DLLInitializeMarketLogin()...")
            try:
                result = dll.DLLInitializeMarketLogin()
                print(f"DLLInitializeMarketLogin retornou: {result}")
            except Exception as e:
                print(f"Erro em DLLInitializeMarketLogin: {e}")
        
        # Aguardar callbacks
        print("\nAguardando 5 segundos para receber callbacks...")
        time.sleep(5)
        
        # Finalizar
        if 'DLLFinalize' in available_functions:
            print("\nFinalizando DLL...")
            try:
                dll.DLLFinalize()
                print("[OK] DLL finalizada")
            except Exception as e:
                print(f"Erro ao finalizar: {e}")
        
    except Exception as e:
        print(f"\n[ERRO] Erro geral: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_profit_dll()