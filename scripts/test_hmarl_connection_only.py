"""
Teste mínimo apenas de conexão sem processamento de dados
Para identificar exatamente onde ocorre o Segmentation Fault
"""

import os
import sys
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestConnectionOnly')


def dummy_callback(data):
    """Callback vazio para não processar nada"""
    pass


def test_connection_only():
    """Teste apenas conexão sem processamento"""
    
    logger.info("🔬 Teste de Conexão Isolada (sem processamento)")
    
    # Credenciais
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    username = os.getenv("PROFIT_USERNAME")
    password = os.getenv("PROFIT_PASSWORD")
    key = os.getenv("PROFIT_KEY")
    
    # 1. Testar apenas conexão básica
    logger.info("\n=== TESTE 1: Conexão Básica ===")
    try:
        from src.connection_manager_v4 import ConnectionManagerV4
        
        connection = ConnectionManagerV4(dll_path)
        
        # Registrar callback vazio ANTES de conectar
        connection.register_trade_callback(dummy_callback)
        
        logger.info("Callback vazio registrado")
        
        # Conectar
        if connection.initialize(key=key, username=username, password=password):
            logger.info("✅ Conectado com sucesso!")
            
            # Aguardar para ver se crash ocorre apenas com conexão
            logger.info("Aguardando 10 segundos com conexão ativa...")
            time.sleep(10)
            
            logger.info("✅ Conexão estável por 10 segundos")
            
        else:
            logger.error("❌ Falha na conexão")
            
    except Exception as e:
        logger.error(f"❌ Erro: {e}", exc_info=True)
    finally:
        if 'connection' in locals():
            connection.disconnect()
            logger.info("Desconectado")
    
    # 2. Testar com subscrição mas sem processamento
    logger.info("\n=== TESTE 2: Com Subscrição (sem processar) ===")
    try:
        connection2 = ConnectionManagerV4(dll_path)
        
        # Callback que apenas conta
        counter = {'count': 0}
        def count_callback(data):
            counter['count'] += 1
            if counter['count'] % 1000 == 0:
                logger.info(f"Recebidos {counter['count']} eventos")
        
        connection2.register_trade_callback(count_callback)
        
        if connection2.initialize(key=key, username=username, password=password):
            logger.info("✅ Conectado")
            
            # Subscrever
            if connection2.subscribe_ticker('WDOQ25'):
                logger.info("✅ Subscrito para WDOQ25")
                
                logger.info("Aguardando 30 segundos recebendo dados...")
                time.sleep(30)
                
                logger.info(f"✅ Total de eventos recebidos: {counter['count']}")
            else:
                logger.warning("⚠️ Falha na subscrição")
                
    except Exception as e:
        logger.error(f"❌ Erro: {e}", exc_info=True)
    finally:
        if 'connection2' in locals():
            connection2.disconnect()
            logger.info("Desconectado")
    
    logger.info("\n✨ Teste de conexão concluído")


if __name__ == "__main__":
    test_connection_only()