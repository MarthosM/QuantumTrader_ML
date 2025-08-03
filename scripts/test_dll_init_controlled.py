"""
Teste controlado da inicialização do ProfitDLL
Tenta identificar exatamente o que causa o crash
"""

import os
import sys
import ctypes
import logging
import threading
import time
import signal
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DLLInitTest')

# Carregar variáveis
load_dotenv()


# Definir callbacks como funções Python
def state_callback(nType, nResult):
    """Callback de estado da conexão"""
    logger.info(f"StateCallback: Type={nType}, Result={nResult}")
    return 0

def history_callback(*args):
    """Callback de histórico"""
    logger.info(f"HistoryCallback chamado")
    return 0

def order_change_callback(*args):
    """Callback de mudança de ordem"""
    logger.info(f"OrderChangeCallback chamado")
    return 0

def account_callback(*args):
    """Callback de conta"""
    logger.info(f"AccountCallback chamado")
    return 0

def new_trade_callback(*args):
    """Callback de novo trade"""
    logger.info(f"NewTradeCallback chamado")
    return 0

def new_daily_callback(*args):
    """Callback de dados diários"""
    logger.info(f"NewDailyCallback chamado")
    return 0

def price_book_callback(*args):
    """Callback de book de preços"""
    logger.info(f"PriceBookCallback chamado")
    return 0

def offer_book_callback(*args):
    """Callback de book de ofertas"""
    logger.info(f"OfferBookCallback chamado")
    return 0

def history_trade_callback(*args):
    """Callback de histórico de trades"""
    logger.info(f"HistoryTradeCallback chamado")
    return 0

def progress_callback(nProgress):
    """Callback de progresso"""
    logger.info(f"ProgressCallback: {nProgress}%")
    return 0

def tiny_book_callback(*args):
    """Callback de tiny book"""
    logger.info(f"TinyBookCallback chamado")
    return 0


def test_dll_init_minimal():
    """Testa inicialização mínima da DLL"""
    logger.info("="*60)
    logger.info("TESTE: Inicialização Mínima")
    logger.info("="*60)
    
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    
    try:
        # Carregar DLL
        logger.info("1. Carregando DLL...")
        dll = ctypes.CDLL(dll_path)
        logger.info("✅ DLL carregada")
        
        # Configurar servidor primeiro
        logger.info("\n2. Configurando servidor...")
        dll.SetServerAndPort.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        dll.SetServerAndPort.restype = ctypes.c_int
        result = dll.SetServerAndPort("producao.nelogica.com.br", "8184")
        logger.info(f"✅ Servidor configurado: {result}")
        
        # Criar tipos de callback
        logger.info("\n3. Criando tipos de callback...")
        
        # Tipos de callback conforme documentação
        StateCallbackType = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)
        HistoryCallbackType = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
        OrderChangeCallbackType = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
        AccountCallbackType = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p)
        NewTradeCallbackType = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
        NewDailyCallbackType = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
        PriceBookCallbackType = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
        OfferBookCallbackType = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
        HistoryTradeCallbackType = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
        ProgressCallbackType = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int)
        TinyBookCallbackType = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
        
        # Criar callbacks
        state_cb = StateCallbackType(state_callback)
        history_cb = HistoryCallbackType(history_callback)
        order_change_cb = OrderChangeCallbackType(order_change_callback)
        account_cb = AccountCallbackType(account_callback)
        new_trade_cb = NewTradeCallbackType(new_trade_callback)
        new_daily_cb = NewDailyCallbackType(new_daily_callback)
        price_book_cb = PriceBookCallbackType(price_book_callback)
        offer_book_cb = OfferBookCallbackType(offer_book_callback)
        history_trade_cb = HistoryTradeCallbackType(history_trade_callback)
        progress_cb = ProgressCallbackType(progress_callback)
        tiny_book_cb = TinyBookCallbackType(tiny_book_callback)
        
        logger.info("✅ Callbacks criados")
        
        # Configurar DLLInitializeLogin
        logger.info("\n4. Configurando DLLInitializeLogin...")
        dll.DLLInitializeLogin.argtypes = [
            ctypes.c_wchar_p,  # pwcActivationKey
            ctypes.c_wchar_p,  # pwcUser
            ctypes.c_wchar_p,  # pwcPassword
            StateCallbackType,
            HistoryCallbackType,
            OrderChangeCallbackType,
            AccountCallbackType,
            NewTradeCallbackType,
            NewDailyCallbackType,
            PriceBookCallbackType,
            OfferBookCallbackType,
            HistoryTradeCallbackType,
            ProgressCallbackType,
            TinyBookCallbackType
        ]
        dll.DLLInitializeLogin.restype = ctypes.c_int
        
        # Preparar credenciais
        key = os.getenv('PROFIT_KEY')
        username = os.getenv('PROFIT_USERNAME')
        password = os.getenv('PROFIT_PASSWORD')
        
        logger.info(f"Key: {key[:10]}...")
        logger.info(f"Username: {username}")
        logger.info(f"Password: {'*' * len(password)}")
        
        # Criar thread para timeout
        init_result = None
        init_error = None
        
        def init_with_timeout():
            nonlocal init_result, init_error
            try:
                logger.info("\n5. Chamando DLLInitializeLogin...")
                init_result = dll.DLLInitializeLogin(
                    key,
                    username,
                    password,
                    state_cb,
                    history_cb,
                    order_change_cb,
                    account_cb,
                    new_trade_cb,
                    new_daily_cb,
                    price_book_cb,
                    offer_book_cb,
                    history_trade_cb,
                    progress_cb,
                    tiny_book_cb
                )
                logger.info(f"✅ DLLInitializeLogin retornou: {init_result}")
            except Exception as e:
                init_error = e
                logger.error(f"❌ Erro em DLLInitializeLogin: {e}")
        
        # Executar em thread
        init_thread = threading.Thread(target=init_with_timeout)
        init_thread.start()
        
        # Aguardar com timeout
        init_thread.join(timeout=30)
        
        if init_thread.is_alive():
            logger.error("❌ Timeout na inicialização!")
            return False
        
        if init_error:
            logger.error(f"❌ Erro na inicialização: {init_error}")
            return False
        
        if init_result == 0:
            logger.info("✅ Inicialização bem-sucedida!")
            
            # Aguardar callbacks
            logger.info("\n6. Aguardando callbacks...")
            time.sleep(5)
            
            # Finalizar
            logger.info("\n7. Finalizando DLL...")
            dll.DLLFinalize()
            logger.info("✅ DLL finalizada")
            
            return True
        else:
            logger.error(f"❌ Inicialização falhou com código: {init_result}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro geral: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_in_subprocess():
    """Executa teste em subprocess para isolar crashes"""
    logger.info("="*60)
    logger.info("TESTE EM SUBPROCESS")
    logger.info("="*60)
    
    import subprocess
    
    # Criar script temporário
    test_script = """
import sys
sys.path.append(r'C:\\Users\\marth\\OneDrive\\Programacao\\Python\\Projetos\\QuantumTrader_ML')
from scripts.test_dll_init_controlled import test_dll_init_minimal
result = test_dll_init_minimal()
sys.exit(0 if result else 1)
"""
    
    # Executar em subprocess
    result = subprocess.run(
        [sys.executable, "-c", test_script],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    logger.info(f"Exit code: {result.returncode}")
    
    if result.stdout:
        logger.info("STDOUT:")
        for line in result.stdout.split('\n'):
            if line.strip():
                logger.info(f"  {line}")
    
    if result.stderr:
        logger.info("STDERR:")
        for line in result.stderr.split('\n'):
            if line.strip():
                logger.info(f"  {line}")
    
    return result.returncode == 0


def main():
    logger.info("🧪 TESTE CONTROLADO DE INICIALIZAÇÃO DO PROFITDLL")
    logger.info("="*80)
    
    # Primeiro tentar diretamente
    logger.info("Tentando inicialização direta...")
    
    try:
        success = test_dll_init_minimal()
        
        if success:
            logger.info("\n✅ SUCESSO! DLL inicializada sem crashes!")
        else:
            logger.info("\n❌ Falha na inicialização")
            
    except Exception as e:
        logger.error(f"\n❌ Crash detectado: {e}")
        
        # Se crashar, tentar em subprocess
        logger.info("\nTentando em subprocess isolado...")
        subprocess_success = test_in_subprocess()
        
        if subprocess_success:
            logger.info("\n✅ Funciona em subprocess!")
            logger.info("💡 Use arquitetura de processo isolado")
        else:
            logger.info("\n❌ Falha mesmo em subprocess")


if __name__ == "__main__":
    main()