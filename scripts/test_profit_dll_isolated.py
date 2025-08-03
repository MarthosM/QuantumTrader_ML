"""
Teste do servidor ProfitDLL isolado com diagnóstico detalhado
"""

import os
import sys
import time
import socket
import logging
import signal
from datetime import datetime
from multiprocessing import Process, Queue
from multiprocessing.connection import Client, Listener
import threading
from dotenv import load_dotenv

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ProfitDLLTest')


def find_free_port():
    """Encontra uma porta livre no sistema"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def profit_dll_server_with_diagnostics(queue: Queue, port: int):
    """Servidor ProfitDLL com diagnóstico detalhado"""
    # Reconfigurar logging no processo filho
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('DiagnosticServer')
    
    # Ignorar sinais que podem matar o processo
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    
    try:
        logger.info(f"🚀 Iniciando servidor diagnóstico na porta {port}...")
        
        # Importar ConnectionManager
        from src.connection_manager_v4 import ConnectionManagerV4
        from dotenv import load_dotenv
        
        # Carregar variáveis
        load_dotenv()
        
        # Criar listener primeiro
        address = ('localhost', port)
        listener = Listener(address, authkey=b'profit_dll_secret')
        logger.info(f"✅ Listener criado em {address}")
        
        # Reportar que servidor está pronto para receber conexões
        queue.put({"status": "initialized", "port": port})
        
        # Variáveis de controle
        connection_manager = None
        client_conn = None
        is_running = True
        dll_connected = False
        
        # Thread para manter servidor vivo
        def keep_alive():
            while is_running:
                try:
                    logger.debug("Servidor ainda vivo...")
                    time.sleep(10)
                except:
                    pass
        
        alive_thread = threading.Thread(target=keep_alive, daemon=True)
        alive_thread.start()
        
        logger.info("⏳ Aguardando cliente IPC...")
        
        # Aceitar conexão com timeout
        listener._listener._socket.settimeout(30.0)
        try:
            client_conn = listener.accept()
            logger.info("✅ Cliente IPC conectado!")
        except socket.timeout:
            logger.error("❌ Timeout aguardando cliente IPC")
            return
        
        # Loop principal do servidor
        while is_running:
            try:
                if client_conn.poll(timeout=1):
                    msg = client_conn.recv()
                    logger.info(f"📥 Comando recebido: {msg.get('type')}")
                    
                    if msg.get('type') == 'connect_dll':
                        # Conectar ao ProfitDLL
                        logger.info("🔌 Conectando ao ProfitDLL...")
                        
                        try:
                            dll_path = msg.get('dll_path', r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll")
                            
                            # Criar connection manager
                            connection_manager = ConnectionManagerV4(dll_path)
                            logger.info("✅ ConnectionManager criado")
                            
                            # Tentar conectar com timeout
                            def connect_with_timeout():
                                nonlocal dll_connected
                                try:
                                    result = connection_manager.initialize(
                                        key=os.getenv('PROFIT_KEY'),
                                        username=os.getenv('PROFIT_USERNAME'),
                                        password=os.getenv('PROFIT_PASSWORD')
                                    )
                                    dll_connected = result
                                except Exception as e:
                                    logger.error(f"Erro na thread de conexão: {e}")
                                    dll_connected = False
                            
                            # Executar conexão em thread separada
                            connect_thread = threading.Thread(target=connect_with_timeout)
                            connect_thread.start()
                            connect_thread.join(timeout=30)  # 30 segundos timeout
                            
                            if connect_thread.is_alive():
                                logger.error("❌ Timeout conectando ao ProfitDLL")
                                client_conn.send({
                                    'type': 'connect_response',
                                    'success': False,
                                    'error': 'Timeout na conexão'
                                })
                            elif dll_connected:
                                logger.info("✅ ProfitDLL conectado!")
                                client_conn.send({
                                    'type': 'connect_response',
                                    'success': True,
                                    'connected': True
                                })
                            else:
                                logger.error("❌ Falha ao conectar ProfitDLL")
                                client_conn.send({
                                    'type': 'connect_response',
                                    'success': False,
                                    'error': 'Falha na conexão'
                                })
                                
                        except Exception as e:
                            logger.error(f"❌ Erro conectando DLL: {e}", exc_info=True)
                            client_conn.send({
                                'type': 'connect_response',
                                'success': False,
                                'error': str(e)
                            })
                    
                    elif msg.get('type') == 'status':
                        # Status do servidor
                        response = {
                            'type': 'status_response',
                            'alive': True,
                            'dll_connected': dll_connected,
                            'timestamp': datetime.now().isoformat()
                        }
                        client_conn.send(response)
                        logger.info("📤 Status enviado")
                    
                    elif msg.get('type') == 'collect_historical':
                        # Coleta histórica
                        if not dll_connected or not connection_manager:
                            client_conn.send({
                                'type': 'historical_response',
                                'success': False,
                                'error': 'DLL não conectada'
                            })
                        else:
                            logger.info(f"📊 Coletando histórico: {msg}")
                            
                            # Configurar callback
                            trades_received = []
                            def on_history_trade(data):
                                trades_received.append(data)
                                logger.debug(f"Trade recebido: {data}")
                            
                            connection_manager.register_history_trade_callback(on_history_trade)
                            
                            # Solicitar dados
                            success = connection_manager.get_history_trades(
                                ticker=msg.get('symbol'),
                                exchange='F',
                                date_start=msg.get('start_date'),
                                date_end=msg.get('end_date')
                            )
                            
                            if success:
                                logger.info("✅ Solicitação enviada, aguardando dados...")
                                # Aguardar dados
                                time.sleep(10)
                                
                                client_conn.send({
                                    'type': 'historical_response',
                                    'success': True,
                                    'data': {
                                        'trades': trades_received
                                    }
                                })
                            else:
                                client_conn.send({
                                    'type': 'historical_response',
                                    'success': False,
                                    'error': 'Falha ao solicitar dados'
                                })
                    
                    elif msg.get('type') == 'shutdown':
                        logger.info("🛑 Shutdown solicitado")
                        is_running = False
                        break
                        
            except Exception as e:
                logger.error(f"Erro no loop principal: {e}", exc_info=True)
                if "EOFError" in str(e) or "broken pipe" in str(e):
                    logger.info("Cliente desconectou")
                    break
        
        # Cleanup
        logger.info("🧹 Limpando recursos...")
        
        if connection_manager:
            try:
                connection_manager.disconnect()
                logger.info("✅ DLL desconectada")
            except:
                pass
        
        if client_conn:
            try:
                client_conn.close()
            except:
                pass
                
        if listener:
            try:
                listener.close()
            except:
                pass
        
        logger.info("✅ Servidor finalizado normalmente")
        
    except Exception as e:
        logger.error(f"❌ Erro fatal no servidor: {e}", exc_info=True)
        queue.put({"status": "error", "message": str(e)})


def main():
    """Função principal do teste"""
    logger.info("="*80)
    logger.info("🧪 TESTE SERVIDOR PROFITDLL ISOLADO COM DIAGNÓSTICO")
    logger.info("="*80)
    
    # Encontrar porta livre
    port = find_free_port()
    logger.info(f"🔍 Usando porta: {port}")
    
    # Criar fila
    queue = Queue()
    
    # Iniciar servidor
    logger.info("\n🚀 Iniciando servidor isolado...")
    server_process = Process(target=profit_dll_server_with_diagnostics, args=(queue, port))
    server_process.start()
    
    # Aguardar inicialização
    logger.info("⏳ Aguardando servidor...")
    initialized = False
    
    for i in range(30):
        if not queue.empty():
            msg = queue.get()
            if msg.get("status") == "initialized":
                logger.info("✅ Servidor pronto!")
                initialized = True
                break
        
        if not server_process.is_alive():
            logger.error("❌ Servidor morreu durante inicialização!")
            break
            
        time.sleep(1)
        if i % 5 == 0:
            logger.info(f"   Aguardando... {i}s (processo vivo: {server_process.is_alive()})")
    
    if not initialized:
        logger.error("❌ Servidor não inicializou")
        if server_process.is_alive():
            server_process.terminate()
        return
    
    # Conectar ao servidor
    try:
        logger.info(f"\n🔌 Conectando ao servidor...")
        client = Client(('localhost', port), authkey=b'profit_dll_secret')
        logger.info("✅ Cliente conectado!")
        
        # Teste 1: Status inicial
        logger.info("\n📊 Teste 1: Status inicial")
        client.send({'type': 'status'})
        if client.poll(timeout=5):
            response = client.recv()
            logger.info(f"Status: {response}")
        
        # Teste 2: Conectar DLL
        logger.info("\n📊 Teste 2: Conectar ao ProfitDLL")
        client.send({
            'type': 'connect_dll',
            'dll_path': r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
        })
        
        if client.poll(timeout=60):  # 60 segundos para conectar
            response = client.recv()
            logger.info(f"Resposta conexão: {response}")
            
            if response.get('success'):
                # Teste 3: Coleta histórica
                logger.info("\n📊 Teste 3: Coleta histórica")
                client.send({
                    'type': 'collect_historical',
                    'symbol': 'WDOU25',
                    'start_date': '01/08/2025',
                    'end_date': '01/08/2025',
                    'data_types': ['trades']
                })
                
                if client.poll(timeout=30):
                    response = client.recv()
                    logger.info(f"Resposta coleta: {response}")
                    
                    if response.get('success'):
                        data = response.get('data', {})
                        for dtype, records in data.items():
                            logger.info(f"   {dtype}: {len(records) if records else 0} registros")
        
        # Verificar se servidor ainda está vivo
        time.sleep(2)
        logger.info(f"\n🔍 Servidor ainda vivo? {server_process.is_alive()}")
        
        # Shutdown
        logger.info("\n🛑 Enviando shutdown...")
        client.send({'type': 'shutdown'})
        client.close()
        
    except Exception as e:
        logger.error(f"❌ Erro no cliente: {e}", exc_info=True)
    
    # Aguardar servidor
    logger.info("\n⏳ Aguardando servidor finalizar...")
    server_process.join(timeout=10)
    
    if server_process.is_alive():
        logger.warning("⚠️ Forçando término...")
        server_process.terminate()
        server_process.join()
    
    logger.info("\n" + "="*80)
    logger.info("🏁 TESTE CONCLUÍDO")
    logger.info(f"Exit code do servidor: {server_process.exitcode}")
    logger.info("="*80)


if __name__ == "__main__":
    main()