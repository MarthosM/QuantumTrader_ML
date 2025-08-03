"""
Script simplificado para testar coleta de dados históricos
Versão 2 - Com porta dinâmica e melhor tratamento de erros
"""

import os
import sys
import time
import socket
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
logger = logging.getLogger('SimpleCollectorV2')


def find_free_port():
    """Encontra uma porta livre no sistema"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def run_isolated_server(queue: Queue, port: int):
    """Executa servidor ProfitDLL em processo isolado"""
    try:
        # Importar dentro do processo isolado
        from src.integration.profit_dll_server import ProfitDLLServer
        import os
        from dotenv import load_dotenv
        
        # Reconfigurar logging no processo filho
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger('IsolatedServer')
        
        # Carregar variáveis de ambiente
        load_dotenv()
        
        logger.info(f"🚀 Iniciando servidor isolado na porta {port}...")
        
        # Criar servidor com porta específica
        server = ProfitDLLServer(pipe_name='profit_dll_pipe', port=port)
        
        # Configuração
        dll_config = {
            'dll_path': r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll",
            'username': os.getenv('PROFIT_USERNAME'),
            'password': os.getenv('PROFIT_PASSWORD'),
            'key': os.getenv('PROFIT_KEY')
        }
        
        # Reportar status
        queue.put({"status": "initialized", "port": port})
        
        # Iniciar servidor
        server.start(dll_config)
        
    except Exception as e:
        logger.error(f"❌ Erro no servidor: {e}")
        import traceback
        traceback.print_exc()
        queue.put({"status": "error", "message": str(e)})


def test_direct_connection():
    """Testa conexão direta ao ProfitDLL (para comparação)"""
    logger.info("\n" + "="*80)
    logger.info("🧪 TESTE DIRETO (sem isolamento)")
    logger.info("="*80)
    
    try:
        from src.connection_manager_v4 import ConnectionManagerV4
        
        dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
        conn = ConnectionManagerV4(dll_path)
        
        logger.info("🔌 Tentando conectar diretamente...")
        result = conn.initialize(
            key=os.getenv("PROFIT_KEY"),
            username=os.getenv("PROFIT_USERNAME"),
            password=os.getenv("PROFIT_PASSWORD")
        )
        
        if result:
            logger.info("✅ Conexão direta bem-sucedida!")
            
            # Testar histórico
            logger.info("📊 Testando GetHistoryTrades...")
            
            trades_received = []
            def on_history_trade(data):
                trades_received.append(data)
            
            conn.register_history_trade_callback(on_history_trade)
            
            success = conn.get_history_trades(
                ticker='WDOU25',
                exchange='F',
                date_start='01/08/2025',
                date_end='01/08/2025'
            )
            
            logger.info(f"Resultado da solicitação: {success}")
            
            # Aguardar dados
            time.sleep(10)
            
            logger.info(f"Dados recebidos: {len(trades_received)}")
            
            conn.disconnect()
        else:
            logger.error("❌ Falha na conexão direta")
            
    except Exception as e:
        logger.error(f"❌ Erro no teste direto: {e}")


def test_simple_collection():
    """Testa coleta simples de dados históricos com servidor isolado"""
    logger.info("="*80)
    logger.info("🧪 TESTE COM SERVIDOR ISOLADO")
    logger.info("="*80)
    
    # Verificar se pyzmq está instalado
    try:
        import zmq
        logger.info(f"✅ pyzmq instalado - versão: {zmq.zmq_version()}")
    except ImportError:
        logger.error("❌ pyzmq não está instalado!")
        logger.error("Execute: pip install pyzmq")
        return
    
    # Encontrar porta livre
    port = find_free_port()
    logger.info(f"🔍 Usando porta livre: {port}")
    
    # Criar fila para comunicação
    queue = Queue()
    
    # Iniciar servidor em processo separado
    logger.info("\n🔧 Iniciando servidor ProfitDLL...")
    server_process = Process(target=run_isolated_server, args=(queue, port))
    server_process.start()
    
    # Aguardar inicialização
    logger.info("⏳ Aguardando servidor inicializar...")
    initialized = False
    server_port = None
    
    for i in range(30):  # 30 segundos timeout
        if not queue.empty():
            msg = queue.get()
            if msg.get("status") == "initialized":
                server_port = msg.get("port", port)
                logger.info(f"✅ Servidor inicializado na porta {server_port}!")
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
        logger.info(f"\n🔌 Conectando ao servidor na porta {server_port}...")
        server_address = ('localhost', server_port)
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
            client = Client(('localhost', server_port), authkey=b'profit_dll_secret')
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
    # Primeiro testa conexão direta para ver se há problemas básicos
    test_direct_connection()
    
    # Depois testa com servidor isolado
    logger.info("\n\n")
    test_simple_collection()