"""
Teste direto da função GetHistoryTrades
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from ctypes import c_wchar_p, c_int
from dotenv import load_dotenv

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DirectTest')


def test_direct_history():
    """Testa GetHistoryTrades diretamente"""
    logger.info("="*60)
    logger.info("🔍 TESTE DIRETO GetHistoryTrades")
    logger.info("="*60)
    
    # Importar ConnectionManager
    from src.connection_manager_v4 import ConnectionManagerV4
    
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    
    try:
        # Criar connection manager
        conn = ConnectionManagerV4(dll_path)
        logger.info("ConnectionManager criado")
        
        # Conectar
        if not conn.initialize(
            key=os.getenv("PROFIT_KEY"),
            username=os.getenv("PROFIT_USERNAME"),
            password=os.getenv("PROFIT_PASSWORD")
        ):
            logger.error("Falha ao conectar")
            return
            
        logger.info("✅ Conectado ao ProfitDLL")
        
        # Aguardar conexão completa
        time.sleep(3)
        
        # Configurar callback para receber dados
        trades_received = []
        
        def on_history_trade(data):
            trades_received.append(data)
            logger.info(f"Trade histórico recebido: {data}")
        
        conn.register_history_trade_callback(on_history_trade)
        
        # Testar diferentes configurações
        tests = [
            # Teste 1: WDOU25 com exchange "F"
            {
                'name': 'WDOU25 com exchange F',
                'ticker': 'WDOU25',
                'exchange': 'F',
                'start': '01/08/2025',
                'end': '01/08/2025'
            },
            # Teste 2: WDOU25 com exchange "BMF"
            {
                'name': 'WDOU25 com exchange BMF',
                'ticker': 'WDOU25',
                'exchange': 'BMF',
                'start': '01/08/2025',
                'end': '01/08/2025'
            },
            # Teste 3: WDOU25 com exchange vazia
            {
                'name': 'WDOU25 com exchange vazia',
                'ticker': 'WDOU25',
                'exchange': '',
                'start': '01/08/2025',
                'end': '01/08/2025'
            },
            # Teste 4: WDO genérico
            {
                'name': 'WDO genérico com exchange F',
                'ticker': 'WDO',
                'exchange': 'F',
                'start': '01/08/2025',
                'end': '01/08/2025'
            },
            # Teste 5: Formato de data alternativo (YYYYMMDD)
            {
                'name': 'WDOU25 com data YYYYMMDD',
                'ticker': 'WDOU25',
                'exchange': 'F',
                'start': '20250801',
                'end': '20250801'
            },
            # Teste 6: Período maior (3 dias)
            {
                'name': 'WDOU25 últimos 3 dias',
                'ticker': 'WDOU25',
                'exchange': 'F',
                'start': '30/07/2025',
                'end': '01/08/2025'
            }
        ]
        
        # Executar testes
        for test in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"📊 TESTE: {test['name']}")
            logger.info(f"Ticker: {test['ticker']}")
            logger.info(f"Exchange: '{test['exchange']}'")
            logger.info(f"Período: {test['start']} até {test['end']}")
            logger.info(f"{'='*50}")
            
            trades_received.clear()
            
            # Chamar GetHistoryTrades diretamente
            if hasattr(conn.dll, 'GetHistoryTrades'):
                # Configurar tipos de argumentos
                conn.dll.GetHistoryTrades.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p, c_wchar_p]
                conn.dll.GetHistoryTrades.restype = c_int
                
                result = conn.dll.GetHistoryTrades(
                    test['ticker'],
                    test['exchange'],
                    test['start'],
                    test['end']
                )
                
                logger.info(f"Resultado GetHistoryTrades: {result}")
                
                # Interpretar resultado
                if result == 0:
                    logger.info("✅ Solicitação enviada com sucesso!")
                    
                    # Aguardar dados
                    logger.info("Aguardando dados (10 segundos)...")
                    time.sleep(10)
                    
                    if trades_received:
                        logger.info(f"✅ {len(trades_received)} trades recebidos!")
                        # Mostrar primeiros trades
                        for i, trade in enumerate(trades_received[:5]):
                            logger.info(f"Trade {i+1}: {trade}")
                    else:
                        logger.warning("⚠️ Nenhum dado recebido")
                        
                elif result == -2147483645:
                    logger.error("❌ Erro -2147483645 (parâmetros inválidos ou dados não disponíveis)")
                elif result == -1:
                    logger.error("❌ Erro -1 (erro geral)")
                else:
                    logger.error(f"❌ Erro desconhecido: {result}")
            else:
                logger.error("GetHistoryTrades não disponível na DLL")
            
            # Pausa entre testes
            time.sleep(2)
        
        # Desconectar
        conn.disconnect()
        logger.info("\n✅ Teste concluído")
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}", exc_info=True)


def check_dll_functions():
    """Verifica funções disponíveis na DLL"""
    logger.info("\n" + "="*60)
    logger.info("🔍 VERIFICANDO FUNÇÕES DA DLL")
    logger.info("="*60)
    
    import ctypes
    
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    
    try:
        dll = ctypes.windll.LoadLibrary(dll_path)
        logger.info("✅ DLL carregada")
        
        # Lista de funções esperadas
        functions = [
            'GetHistoryTrades',
            'GetHistoryTradesV2',
            'GetHistoricalData',
            'GetTrades',
            'GetTradesHistory',
            'RequestHistoryTrades',
            'RequestHistoricalData'
        ]
        
        logger.info("\nFunções relacionadas a histórico:")
        for func_name in functions:
            if hasattr(dll, func_name):
                logger.info(f"✅ {func_name} - DISPONÍVEL")
            else:
                logger.info(f"❌ {func_name} - não encontrada")
                
    except Exception as e:
        logger.error(f"Erro ao carregar DLL: {e}")


if __name__ == "__main__":
    # Verificar funções disponíveis
    check_dll_functions()
    
    # Testar GetHistoryTrades
    test_direct_history()