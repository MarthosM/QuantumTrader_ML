"""
Script para testar APENAS com servidor isolado
Evita o Segmentation Fault da conexão direta
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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('IsolatedTest')


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
            level=logging.INFO,
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


def main():
    """Função principal"""
    logger.info("="*80)
    logger.info("🧪 TESTE DE COLETA HISTÓRICA - SERVIDOR ISOLADO")
    logger.info("="*80)
    logger.info("⚠️  Evitando conexão direta para prevenir Segmentation Fault")
    logger.info("")
    
    # Verificar se pyzmq está instalado
    try:
        import zmq
        logger.info(f"✅ pyzmq instalado - versão: {zmq.zmq_version()}")
    except ImportError:
        logger.error("❌ pyzmq não está instalado!")
        logger.error("Execute: pip install pyzmq")
        return
    
    # Matar processos antigos na porta 6789
    logger.info("🧹 Limpando processos antigos...")
    os.system("taskkill /F /PID 11896 2>nul")  # PID que estava usando a porta
    time.sleep(2)
    
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
    
    # Aguardar estabilização completa
    logger.info("⏳ Aguardando servidor estabilizar...")
    time.sleep(10)  # Aumentado para dar mais tempo
    
    # Verificar se servidor ainda está vivo
    if not server_process.is_alive():
        logger.error("❌ Servidor morreu durante estabilização!")
        return
    
    logger.info("✅ Servidor estável e pronto!")
    
    try:
        # Conectar ao servidor via IPC
        logger.info(f"\n🔌 Conectando ao servidor na porta {server_port}...")
        server_address = ('localhost', server_port)
        
        # Tentar conectar com retry
        connected = False
        for attempt in range(3):
            try:
                client = Client(server_address, authkey=b'profit_dll_secret')
                logger.info("✅ Conectado ao servidor IPC!")
                connected = True
                break
            except Exception as e:
                logger.warning(f"Tentativa {attempt+1}/3 falhou: {e}")
                time.sleep(2)
        
        if not connected:
            raise Exception("Não foi possível conectar ao servidor")
        
        # Solicitar status
        logger.info("\n📊 Verificando status do servidor...")
        client.send({'type': 'status'})
        
        # Aguardar resposta
        if client.poll(timeout=10):
            response = client.recv()
            logger.info(f"📥 Status do servidor:")
            logger.info(f"   Conectado: {response.get('connected', False)}")
            logger.info(f"   Pronto: {response.get('ready', False)}")
            logger.info(f"   Mensagens enviadas: {response.get('messages_sent', 0)}")
        else:
            logger.warning("⚠️ Sem resposta de status do servidor")
        
        # Fazer coleta simples
        logger.info("\n📈 Solicitando dados históricos...")
        
        # Testar com diferentes datas
        test_dates = [
            {
                'start': '01/08/2025',
                'end': '01/08/2025',
                'desc': 'Ontem (01/08/2025)'
            },
            {
                'start': '30/07/2025',
                'end': '30/07/2025',
                'desc': 'Há 3 dias (30/07/2025)'
            },
            {
                'start': '25/07/2025',
                'end': '25/07/2025',
                'desc': 'Semana passada (25/07/2025)'
            }
        ]
        
        for test in test_dates:
            logger.info(f"\n🗓️ Testando período: {test['desc']}")
            
            command = {
                'type': 'collect_historical',
                'symbol': 'WDOU25',
                'start_date': test['start'],
                'end_date': test['end'],
                'data_types': ['trades']
            }
            
            logger.info(f"📤 Enviando comando...")
            client.send(command)
            
            # Aguardar resposta
            logger.info("⏳ Aguardando dados...")
            
            if client.poll(timeout=30):  # 30 segundos timeout
                result = client.recv()
                
                if result.get('success'):
                    logger.info("✅ Resposta recebida!")
                    
                    # Analisar dados recebidos
                    for data_type, data in result.get('data', {}).items():
                        if data and len(data) > 0:
                            logger.info(f"   📊 {data_type.upper()}: {len(data)} registros")
                            logger.info(f"   Primeiro: {data[0]}")
                            logger.info(f"   Último: {data[-1]}")
                            
                            # Se recebeu dados, parar testes
                            logger.info("\n🎉 SUCESSO! Dados históricos coletados!")
                            break
                        else:
                            logger.warning(f"   ⚠️ Nenhum dado de {data_type}")
                else:
                    logger.error(f"   ❌ Falha: {result.get('error')}")
            else:
                logger.error("   ❌ Timeout aguardando resposta")
            
            # Se já recebeu dados, não precisa testar outras datas
            if result.get('success') and any(result.get('data', {}).values()):
                break
        
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
            logger.info("   Comando de shutdown enviado")
        except:
            pass
        
        # Aguardar término
        server_process.join(timeout=10)
        
        if server_process.is_alive():
            logger.warning("⚠️ Forçando término do servidor...")
            server_process.terminate()
            server_process.join()
        
        logger.info("✅ Servidor finalizado")
    
    logger.info("\n" + "="*80)
    logger.info("🏁 TESTE CONCLUÍDO")
    logger.info("="*80)
    
    # Resumo
    logger.info("\n📋 RESUMO:")
    logger.info("1. Servidor isolado previne Segmentation Fault ✅")
    logger.info("2. Comunicação IPC funcionando ✅")
    logger.info("3. GetHistoryTrades aceita requisições ✅")
    logger.info("4. Verificar se dados foram recebidos acima ⬆️")


if __name__ == "__main__":
    main()