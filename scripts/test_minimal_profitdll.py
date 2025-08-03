"""
Teste mínimo para verificar se conseguimos conectar ao ProfitDLL
sem travar o processo
"""

import os
import sys
import ctypes
import logging
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MinimalTest')

# Carregar variáveis
load_dotenv()


def test_dll_load():
    """Testa apenas carregar a DLL"""
    logger.info("="*60)
    logger.info("TESTE 1: Carregar DLL")
    logger.info("="*60)
    
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    
    try:
        logger.info(f"Tentando carregar: {dll_path}")
        dll = ctypes.CDLL(dll_path)
        logger.info("✅ DLL carregada com sucesso!")
        logger.info(f"DLL object: {dll}")
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao carregar DLL: {e}")
        return False


def test_dll_functions():
    """Testa se as funções existem na DLL"""
    logger.info("\n" + "="*60)
    logger.info("TESTE 2: Verificar funções")
    logger.info("="*60)
    
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    
    try:
        dll = ctypes.CDLL(dll_path)
        
        # Lista de funções esperadas
        functions = [
            'DLLInitializeLogin',
            'DLLFinalize',
            'GetHistoryTrades',
            'SubscribeTicker',
            'GetServerClock'
        ]
        
        for func_name in functions:
            try:
                func = getattr(dll, func_name)
                logger.info(f"✅ {func_name} encontrada: {func}")
            except AttributeError:
                logger.error(f"❌ {func_name} não encontrada")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro verificando funções: {e}")
        return False


def test_server_config():
    """Testa configurar servidor sem conectar"""
    logger.info("\n" + "="*60)
    logger.info("TESTE 3: Configurar servidor")
    logger.info("="*60)
    
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    
    try:
        dll = ctypes.CDLL(dll_path)
        
        # Tentar função SetServerAndPort
        if hasattr(dll, 'SetServerAndPort'):
            logger.info("Tentando configurar servidor...")
            
            # Definir tipos de parâmetros
            dll.SetServerAndPort.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
            dll.SetServerAndPort.restype = ctypes.c_int
            
            result = dll.SetServerAndPort("producao.nelogica.com.br", "8184")
            logger.info(f"Resultado: {result}")
            
            if result == 0:
                logger.info("✅ Servidor configurado!")
            else:
                logger.warning(f"⚠️ Código de retorno: {result}")
                
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro configurando servidor: {e}")
        return False


def test_simple_function():
    """Testa função simples que não deveria travar"""
    logger.info("\n" + "="*60)
    logger.info("TESTE 4: Função simples (GetServerClock)")
    logger.info("="*60)
    
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    
    try:
        dll = ctypes.CDLL(dll_path)
        
        if hasattr(dll, 'GetServerClock'):
            logger.info("Tentando GetServerClock...")
            
            # Criar variáveis para receber os valores
            date = ctypes.c_double()
            year = ctypes.c_int()
            month = ctypes.c_int()
            day = ctypes.c_int()
            hour = ctypes.c_int()
            minute = ctypes.c_int()
            second = ctypes.c_int()
            millisec = ctypes.c_int()
            
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
            dll.GetServerClock.restype = ctypes.c_int
            
            result = dll.GetServerClock(
                ctypes.byref(date),
                ctypes.byref(year),
                ctypes.byref(month),
                ctypes.byref(day),
                ctypes.byref(hour),
                ctypes.byref(minute),
                ctypes.byref(second),
                ctypes.byref(millisec)
            )
            
            logger.info(f"Resultado: {result}")
            
            if result == 0:
                logger.info(f"✅ Data/hora: {year.value}/{month.value}/{day.value} {hour.value}:{minute.value}:{second.value}")
            else:
                logger.warning(f"⚠️ Código de retorno: {result}")
                
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro em GetServerClock: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info("🧪 TESTE MÍNIMO DO PROFITDLL")
    logger.info("="*80)
    logger.info("Objetivo: Identificar qual operação causa o crash")
    logger.info("")
    
    # Executar testes em sequência
    tests = [
        ("Carregar DLL", test_dll_load),
        ("Verificar funções", test_dll_functions),
        ("Configurar servidor", test_server_config),
        ("Função simples", test_simple_function)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n🔍 Executando: {test_name}")
            success = test_func()
            results.append((test_name, success))
            
            if not success:
                logger.warning(f"⚠️ Parando testes após falha em: {test_name}")
                break
                
        except Exception as e:
            logger.error(f"❌ Erro fatal em {test_name}: {e}")
            results.append((test_name, False))
            break
    
    # Resumo
    logger.info("\n" + "="*80)
    logger.info("📋 RESUMO DOS TESTES")
    logger.info("="*80)
    
    for test_name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        logger.info(f"{test_name}: {status}")
    
    # Análise
    logger.info("\n💡 ANÁLISE:")
    
    if all(success for _, success in results):
        logger.info("✅ Todos os testes passaram!")
        logger.info("O problema pode estar na inicialização completa (DLLInitializeLogin)")
    else:
        failed_test = next((name for name, success in results if not success), None)
        logger.info(f"❌ Problema identificado em: {failed_test}")
        
        if failed_test == "Carregar DLL":
            logger.info("   - Verifique se a DLL está no caminho correto")
            logger.info("   - Verifique se é a versão correta (Win64)")
            logger.info("   - Verifique dependências da DLL")
        elif failed_test == "Função simples":
            logger.info("   - A DLL pode precisar ser inicializada antes")
            logger.info("   - Pode haver problema com os tipos de dados")


if __name__ == "__main__":
    main()