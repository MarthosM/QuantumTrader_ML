"""
Teste simples de IPC sem ProfitDLL
Para verificar se a comunicação entre processos está funcionando
"""

import os
import sys
import time
import socket
import logging
from datetime import datetime
from multiprocessing import Process, Queue
from multiprocessing.connection import Client, Listener
from dotenv import load_dotenv

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('IPCTest')


def find_free_port():
    """Encontra uma porta livre no sistema"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def simple_server(queue: Queue, port: int):
    """Servidor simples sem ProfitDLL"""
    # Reconfigurar logging no processo filho
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('SimpleServer')
    
    try:
        logger.info(f"🚀 Iniciando servidor simples na porta {port}...")
        
        # Criar listener
        address = ('localhost', port)
        listener = Listener(address, authkey=b'profit_dll_secret')
        logger.info(f"✅ Listener criado em {address}")
        
        # Reportar status
        queue.put({"status": "initialized", "port": port})
        
        # Aguardar conexão
        logger.info("⏳ Aguardando cliente...")
        conn = listener.accept()
        logger.info("✅ Cliente conectado!")
        
        # Loop principal
        while True:
            try:
                if conn.poll(timeout=1):
                    msg = conn.recv()
                    logger.info(f"📥 Mensagem recebida: {msg}")
                    
                    if msg.get('type') == 'status':
                        # Responder status
                        response = {
                            'type': 'status_response',
                            'connected': True,
                            'ready': True,
                            'timestamp': datetime.now().isoformat()
                        }
                        conn.send(response)
                        logger.info("📤 Status enviado")
                        
                    elif msg.get('type') == 'collect_historical':
                        # Simular coleta histórica
                        logger.info(f"📊 Simulando coleta: {msg.get('symbol')} de {msg.get('start_date')} até {msg.get('end_date')}")
                        
                        # Simular alguns dados
                        fake_data = []
                        for i in range(10):
                            fake_data.append({
                                'timestamp': f"2025-08-01 09:{i:02d}:00",
                                'price': 5000 + i * 10,
                                'volume': 100 + i * 5,
                                'type': 'trade'
                            })
                        
                        response = {
                            'type': 'historical_response',
                            'success': True,
                            'data': {
                                'trades': fake_data
                            }
                        }
                        conn.send(response)
                        logger.info("📤 Dados simulados enviados")
                        
                    elif msg.get('type') == 'shutdown':
                        logger.info("🛑 Comando de shutdown recebido")
                        break
                        
            except Exception as e:
                logger.error(f"Erro no loop: {e}")
                break
        
        # Fechar conexões
        conn.close()
        listener.close()
        logger.info("✅ Servidor finalizado normalmente")
        
    except Exception as e:
        logger.error(f"❌ Erro no servidor: {e}")
        import traceback
        traceback.print_exc()
        queue.put({"status": "error", "message": str(e)})


def test_ipc():
    """Testa comunicação IPC básica"""
    logger.info("="*80)
    logger.info("🧪 TESTE DE COMUNICAÇÃO IPC")
    logger.info("="*80)
    logger.info("⚠️  Teste sem ProfitDLL - apenas IPC")
    logger.info("")
    
    # Encontrar porta livre
    port = find_free_port()
    logger.info(f"🔍 Usando porta livre: {port}")
    
    # Criar fila para comunicação
    queue = Queue()
    
    # Iniciar servidor simples
    logger.info("\n🔧 Iniciando servidor simples...")
    server_process = Process(target=simple_server, args=(queue, port))
    server_process.start()
    
    # Aguardar inicialização
    logger.info("⏳ Aguardando servidor inicializar...")
    initialized = False
    server_port = None
    
    for i in range(10):
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
    
    if not initialized:
        logger.error("❌ Servidor não inicializou")
        server_process.terminate()
        server_process.join()
        return
    
    # Aguardar estabilização
    time.sleep(2)
    
    # Verificar se servidor ainda está vivo
    if not server_process.is_alive():
        logger.error("❌ Servidor morreu!")
        return
    
    logger.info("✅ Servidor estável!")
    
    try:
        # Conectar ao servidor
        logger.info(f"\n🔌 Conectando ao servidor na porta {server_port}...")
        server_address = ('localhost', server_port)
        client = Client(server_address, authkey=b'profit_dll_secret')
        logger.info("✅ Conectado!")
        
        # Teste 1: Status
        logger.info("\n📊 Teste 1: Solicitando status...")
        client.send({'type': 'status'})
        
        if client.poll(timeout=5):
            response = client.recv()
            logger.info(f"✅ Resposta: {response}")
        else:
            logger.error("❌ Sem resposta")
        
        # Teste 2: Coleta histórica simulada
        logger.info("\n📊 Teste 2: Simulando coleta histórica...")
        client.send({
            'type': 'collect_historical',
            'symbol': 'WDOU25',
            'start_date': '01/08/2025',
            'end_date': '01/08/2025',
            'data_types': ['trades']
        })
        
        if client.poll(timeout=5):
            response = client.recv()
            if response.get('success'):
                logger.info("✅ Dados recebidos!")
                data = response.get('data', {})
                for dtype, records in data.items():
                    logger.info(f"   {dtype}: {len(records)} registros")
                    if records:
                        logger.info(f"   Primeiro: {records[0]}")
                        logger.info(f"   Último: {records[-1]}")
            else:
                logger.error(f"❌ Falha: {response}")
        else:
            logger.error("❌ Sem resposta")
        
        # Finalizar
        logger.info("\n🛑 Enviando shutdown...")
        client.send({'type': 'shutdown'})
        client.close()
        
    except Exception as e:
        logger.error(f"❌ Erro no cliente: {e}", exc_info=True)
    
    # Aguardar servidor finalizar
    server_process.join(timeout=5)
    
    if server_process.is_alive():
        logger.warning("⚠️ Forçando término...")
        server_process.terminate()
        server_process.join()
    
    logger.info("\n" + "="*80)
    logger.info("🏁 TESTE CONCLUÍDO")
    logger.info("="*80)
    
    # Análise
    logger.info("\n📋 ANÁLISE:")
    if server_process.exitcode == 0:
        logger.info("✅ IPC funcionando corretamente")
        logger.info("✅ Servidor terminou normalmente")
        logger.info("❌ Problema está na integração com ProfitDLL")
        logger.info("\n💡 SOLUÇÃO: Implementar timeout e proteções no servidor ProfitDLL")
    else:
        logger.info("❌ Problema no IPC ou multiprocessing")


if __name__ == "__main__":
    test_ipc()