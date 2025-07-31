"""
Script de diagnóstico para conexão com ProfitDLL
"""

import os
import ctypes
from ctypes import WINFUNCTYPE, c_int, c_wchar_p
import time
import logging

# Configurar logging detalhado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_connection():
    """Testa conexão passo a passo"""
    
    print("="*60)
    print("DIAGNÓSTICO DE CONEXÃO PROFITDLL")
    print("="*60)
    
    # 1. Verificar DLL
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    print(f"\n1. Verificando DLL...")
    print(f"   Caminho: {dll_path}")
    print(f"   Existe: {os.path.exists(dll_path)}")
    
    if not os.path.exists(dll_path):
        print("   [ERRO] DLL não encontrada!")
        return
    
    # 2. Carregar DLL
    print("\n2. Carregando DLL...")
    try:
        dll = ctypes.CDLL(dll_path)
        print("   [OK] DLL carregada")
    except Exception as e:
        print(f"   [ERRO] Falha ao carregar: {e}")
        return
    
    # 3. Verificar funções
    print("\n3. Verificando funções disponíveis...")
    functions = [
        'DLLInitializeMarketLogin',
        'DLLInitializeLogin',
        'DLLFinalize',
        'GetHistoryTrades',
        'SubscribeTicker',
        'GetServerClock'
    ]
    
    for func_name in functions:
        try:
            func = getattr(dll, func_name)
            print(f"   [OK] {func_name}")
        except AttributeError:
            print(f"   [X] {func_name} não encontrada")
    
    # 4. Testar callback simples
    print("\n4. Configurando callbacks...")
    
    # Estado global
    connection_states = []
    
    # Callback de estado
    @WINFUNCTYPE(None, c_int, c_int)
    def state_callback(conn_type, result):
        state_info = f"Estado: tipo={conn_type}, resultado={result}"
        connection_states.append(state_info)
        logger.info(state_info)
        
        # Interpretar estados
        if conn_type == 0:  # LOGIN
            if result == 0:
                print("   [LOGIN] Conectado com sucesso!")
            elif result == 1:
                print("   [LOGIN] Login inválido")
            elif result == 2:
                print("   [LOGIN] Senha inválida")
        elif conn_type == 2:  # MARKET_DATA
            if result == 4:
                print("   [MARKET] Dados de mercado conectados!")
        elif conn_type == 3:  # MARKET_LOGIN
            if result == 0:
                print("   [MARKET_LOGIN] Ativação válida!")
    
    print("   [OK] Callbacks configurados")
    
    # 5. Tentar inicialização sem credenciais
    print("\n5. Tentando conexão sem credenciais (modo demo)...")
    
    try:
        # Configurar assinatura
        dll.DLLInitializeMarketLogin.argtypes = [
            c_wchar_p, c_wchar_p, c_wchar_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p
        ]
        dll.DLLInitializeMarketLogin.restype = c_int
        
        # Chamar com strings vazias
        result = dll.DLLInitializeMarketLogin(
            "",  # activation key
            "",  # user
            "",  # password
            state_callback,
            None,  # trade callback
            None,  # daily callback
            None,  # price book
            None,  # offer book
            None,  # history trade
            None,  # progress
            None   # tiny book
        )
        
        print(f"   Resultado da inicialização: {result}")
        
        if result == 0:
            print("   [OK] Inicialização aceita")
        else:
            print(f"   [ERRO] Código de erro: {result}")
            print("   Possíveis significados:")
            print("   - 0x80000001: Erro interno")
            print("   - 0x80000003: Argumentos inválidos")
            print("   - 0x80000004: Chave de ativação inválida")
    
    except Exception as e:
        print(f"   [ERRO] Exceção: {e}")
    
    # 6. Aguardar callbacks
    print("\n6. Aguardando callbacks (10 segundos)...")
    for i in range(10):
        time.sleep(1)
        print(f"   {i+1}s...")
        if connection_states:
            print(f"   Estados recebidos: {len(connection_states)}")
    
    # 7. Verificar horário do servidor (se conectado)
    print("\n7. Tentando obter horário do servidor...")
    try:
        if hasattr(dll, 'GetServerClock'):
            # Configurar tipos
            dll.GetServerClock.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int)
            ]
            dll.GetServerClock.restype = c_int
            
            # Variáveis
            dtDate = ctypes.c_double()
            year = ctypes.c_int()
            month = ctypes.c_int()
            day = ctypes.c_int()
            hour = ctypes.c_int()
            minute = ctypes.c_int()
            sec = ctypes.c_int()
            milisec = ctypes.c_int()
            
            result = dll.GetServerClock(
                ctypes.byref(dtDate),
                ctypes.byref(year),
                ctypes.byref(month),
                ctypes.byref(day),
                ctypes.byref(hour),
                ctypes.byref(minute),
                ctypes.byref(sec),
                ctypes.byref(milisec)
            )
            
            if result == 0:
                print(f"   [OK] Horário: {day.value}/{month.value}/{year.value} "
                      f"{hour.value}:{minute.value}:{sec.value}")
            else:
                print(f"   [ERRO] Não foi possível obter horário (código: {result})")
    except Exception as e:
        print(f"   [ERRO] {e}")
    
    # 8. Finalizar
    print("\n8. Finalizando DLL...")
    try:
        dll.DLLFinalize()
        print("   [OK] DLL finalizada")
    except Exception as e:
        print(f"   [ERRO] {e}")
    
    # Resumo
    print("\n" + "="*60)
    print("RESUMO DO DIAGNÓSTICO")
    print("="*60)
    print(f"Estados de conexão recebidos: {len(connection_states)}")
    for state in connection_states:
        print(f"  - {state}")
    
    print("\nPróximos passos:")
    print("1. Verificar se o ProfitChart está aberto")
    print("2. Configurar credenciais nas variáveis de ambiente:")
    print("   - PROFIT_KEY (chave de ativação)")
    print("   - PROFIT_USER (usuário)")
    print("   - PROFIT_PASSWORD (senha)")
    print("3. Verificar firewall/antivírus")
    print("4. Confirmar versão da DLL (32/64 bits)")

if __name__ == "__main__":
    test_connection()