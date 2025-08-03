"""
Script para testar servidor isolado e coleta
"""

import os
import sys
import time
import logging
import multiprocessing
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.integration.profit_dll_server import run_server

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestServerCollection')


def test_server_only():
    """Testa apenas o servidor isolado"""
    logger.info("="*60)
    logger.info("🧪 Teste do Servidor Isolado")
    logger.info("="*60)
    
    server_config = {
        'dll_path': r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll",
        'username': os.getenv("PROFIT_USERNAME"),
        'password': os.getenv("PROFIT_PASSWORD"),
        'key': os.getenv("PROFIT_KEY"),
        'port': 6791  # Porta diferente para evitar conflitos
    }
    
    logger.info("Iniciando servidor em processo isolado...")
    
    # Iniciar servidor
    server_process = multiprocessing.Process(
        target=run_server,
        args=(server_config,),
        name="TestProfitDLLServer"
    )
    server_process.start()
    
    logger.info(f"Servidor iniciado - PID: {server_process.pid}")
    
    # Monitorar servidor
    for i in range(30):  # 30 segundos
        time.sleep(1)
        if server_process.is_alive():
            logger.info(f"✅ Servidor rodando... ({i+1}/30)")
        else:
            logger.error(f"❌ Servidor morreu após {i+1} segundos")
            logger.info(f"Exit code: {server_process.exitcode}")
            break
    
    # Parar servidor
    if server_process.is_alive():
        logger.info("\n🛑 Parando servidor...")
        server_process.terminate()
        server_process.join(timeout=5)
        
        if server_process.is_alive():
            server_process.kill()
            server_process.join()
    
    logger.info("✅ Teste concluído")


def test_client_connection():
    """Testa conexão de cliente ao servidor"""
    from multiprocessing.connection import Client
    
    logger.info("\n"+"="*60)
    logger.info("🧪 Teste de Conexão Cliente-Servidor")
    logger.info("="*60)
    
    # Configuração
    server_config = {
        'dll_path': r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll",
        'username': os.getenv("PROFIT_USERNAME"),
        'password': os.getenv("PROFIT_PASSWORD"),
        'key': os.getenv("PROFIT_KEY"),
        'port': 6792
    }
    
    # 1. Iniciar servidor
    logger.info("1️⃣ Iniciando servidor...")
    server_process = multiprocessing.Process(
        target=run_server,
        args=(server_config,),
        name="TestServer"
    )
    server_process.start()
    
    # 2. Aguardar servidor inicializar
    logger.info("2️⃣ Aguardando servidor...")
    time.sleep(8)
    
    if not server_process.is_alive():
        logger.error("❌ Servidor morreu durante inicialização")
        return
    
    # 3. Conectar cliente
    logger.info("3️⃣ Conectando cliente...")
    try:
        client = Client(('localhost', server_config['port']), authkey=b'profit_dll_secret')
        
        # Receber mensagem de boas-vindas
        welcome = client.recv()
        logger.info(f"✅ Conectado! Resposta: {welcome}")
        
        # 4. Testar comando simples
        logger.info("4️⃣ Enviando comando ping...")
        client.send({'type': 'ping'})
        
        # Aguardar resposta
        timeout = 5
        start = time.time()
        while time.time() - start < timeout:
            if client.poll(0.5):
                response = client.recv()
                logger.info(f"✅ Resposta recebida: {response}")
                break
        else:
            logger.warning("⚠️ Timeout aguardando resposta")
        
        # 5. Solicitar coleta histórica
        logger.info("5️⃣ Solicitando dados históricos...")
        command = {
            'type': 'collect_historical',
            'symbol': 'WDOQ25',
            'start_date': '01/08/2025',
            'end_date': '02/08/2025',
            'data_types': ['trades']
        }
        client.send(command)
        
        # Aguardar dados
        timeout = 30
        start = time.time()
        while time.time() - start < timeout:
            if client.poll(1):
                response = client.recv()
                msg_type = response.get('type')
                
                if msg_type == 'historical_data':
                    count = response.get('count', 0)
                    logger.info(f"✅ Dados recebidos: {count} trades")
                    break
                elif msg_type == 'error':
                    logger.error(f"❌ Erro: {response.get('message')}")
                    break
                else:
                    logger.info(f"Mensagem recebida: {msg_type}")
        else:
            logger.warning("⚠️ Timeout aguardando dados históricos")
        
        # Fechar conexão
        client.send({'type': 'done'})
        client.close()
        
    except Exception as e:
        logger.error(f"❌ Erro no cliente: {e}")
    
    # 6. Parar servidor
    logger.info("6️⃣ Parando servidor...")
    if server_process.is_alive():
        server_process.terminate()
        server_process.join(timeout=5)
        
        if server_process.is_alive():
            server_process.kill()
            server_process.join()
    
    logger.info("✨ Teste concluído")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # Teste 1: Apenas servidor
    test_server_only()
    
    # Pequena pausa entre testes
    time.sleep(5)
    
    # Teste 2: Cliente-Servidor
    test_client_connection()