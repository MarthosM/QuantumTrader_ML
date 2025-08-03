"""
Script simplificado para testar coleta de dados históricos
Diagnóstico do problema de inicialização do servidor
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from multiprocessing import Process, Queue
from multiprocessing.connection import Client
from dotenv import load_dotenv

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SimpleCollector')


def run_isolated_server(queue: Queue):
    """Executa servidor ProfitDLL em processo isolado"""
    try:
        # Importar dentro do processo isolado
        from src.integration.profit_dll_server import run_server
        import os
        from dotenv import load_dotenv
        
        # Carregar variáveis de ambiente
        load_dotenv()
        
        logger.info("🚀 Iniciando servidor isolado...")
        
        # Configuração do servidor
        config = {
            'dll_path': r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll",
            'username': os.getenv('PROFIT_USERNAME'),
            'password': os.getenv('PROFIT_PASSWORD'),
            'key': os.getenv('PROFIT_KEY'),
            'pipe_name': 'profit_dll_pipe',
            'port': 6789  # Porta padrão do servidor
        }
        
        # Reportar status
        queue.put({"status": "initialized"})
        
        # Executar servidor (função que não retorna até shutdown)
        run_server(config)
        
    except Exception as e:
        logger.error(f"❌ Erro no servidor: {e}")
        queue.put({"status": "error", "message": str(e)})


def test_simple_collection():
    """Testa coleta simples de dados históricos"""
    logger.info("="*80)
    logger.info("🧪 TESTE SIMPLIFICADO DE COLETA HISTÓRICA")
    logger.info("="*80)
    
    # Verificar se pyzmq está instalado
    try:
        import zmq
        logger.info(f"✅ pyzmq instalado - versão: {zmq.zmq_version()}")
    except ImportError:
        logger.error("❌ pyzmq não está instalado!")
        logger.error("Execute: pip install pyzmq")
        return
    
    # Criar fila para comunicação
    queue = Queue()
    
    # Iniciar servidor em processo separado
    logger.info("\n🔧 Iniciando servidor ProfitDLL...")
    server_process = Process(target=run_isolated_server, args=(queue,))
    server_process.start()
    
    # Aguardar inicialização
    logger.info("⏳ Aguardando servidor inicializar...")
    initialized = False
    
    for i in range(30):  # 30 segundos timeout
        if not queue.empty():
            msg = queue.get()
            if msg.get("status") == "initialized":
                logger.info("✅ Servidor inicializado!")
                initialized = True
                break
            elif msg.get("status") == "error":
                logger.error(f"❌ Erro na inicialização: {msg.get('message')}")
                break
        
        time.sleep(1)
        if i % 5 == 0:
            logger.info(f"   Aguardando... {i}s")
    
    if not initialized:
        logger.error("❌ Servidor não inicializou no tempo esperado")
        server_process.terminate()
        server_process.join()
        return
    
    # Aguardar estabilização
    logger.info("⏳ Aguardando estabilização...")
    time.sleep(5)
    
    try:
        # Conectar ao servidor via IPC
        logger.info("\n🔌 Conectando ao servidor...")
        server_address = ('localhost', 6789)  # Usar porta correta
        client = Client(server_address, authkey=b'profit_dll_secret')
        logger.info("✅ Conectado ao servidor IPC!")
        
        # Solicitar status
        logger.info("\n📊 Verificando status do servidor...")
        client.send({'type': 'status'})
        
        # Aguardar resposta
        if client.poll(timeout=5):
            response = client.recv()
            logger.info(f"📥 Status: {response}")
        else:
            logger.warning("⚠️ Sem resposta do servidor")
        
        # Fazer coleta simples
        logger.info("\n📈 Solicitando dados históricos...")
        
        # Usar data de ontem (01/08/2025)
        command = {
            'type': 'collect_historical',
            'symbol': 'WDOU25',
            'start_date': '01/08/2025',
            'end_date': '01/08/2025',
            'data_types': ['trades']
        }
        
        logger.info(f"📤 Enviando comando: {command}")
        client.send(command)
        
        # Aguardar resposta
        logger.info("⏳ Aguardando dados...")
        
        if client.poll(timeout=60):  # 60 segundos timeout
            result = client.recv()
            logger.info(f"\n📥 Resposta recebida!")
            
            if result.get('success'):
                logger.info("✅ Coleta bem-sucedida!")
                
                # Analisar dados recebidos
                for data_type, data in result.get('data', {}).items():
                    if data:
                        logger.info(f"\n📊 {data_type.upper()}:")
                        logger.info(f"   Total de registros: {len(data)}")
                        
                        if len(data) > 0:
                            logger.info(f"   Primeiro registro: {data[0]}")
                            if len(data) > 1:
                                logger.info(f"   Último registro: {data[-1]}")
                    else:
                        logger.info(f"⚠️ Nenhum dado de {data_type}")
            else:
                logger.error(f"❌ Falha na coleta: {result.get('error')}")
        else:
            logger.error("❌ Timeout aguardando resposta do servidor")
        
        # Fechar conexão
        client.close()
        logger.info("\n🔌 Conexão fechada")
        
    except Exception as e:
        logger.error(f"❌ Erro na comunicação: {e}", exc_info=True)
    
    finally:
        # Finalizar servidor
        logger.info("\n🛑 Finalizando servidor...")
        
        try:
            # Tentar finalizar gracefully
            client = Client(('localhost', 6789), authkey=b'profit_dll_secret')
            client.send({'type': 'shutdown'})
            client.close()
        except:
            pass
        
        # Aguardar término
        server_process.join(timeout=5)
        
        if server_process.is_alive():
            logger.warning("⚠️ Forçando término do servidor...")
            server_process.terminate()
            server_process.join()
        
        logger.info("✅ Servidor finalizado")
    
    logger.info("\n" + "="*80)
    logger.info("🏁 TESTE CONCLUÍDO")
    logger.info("="*80)


if __name__ == "__main__":
    test_simple_collection()