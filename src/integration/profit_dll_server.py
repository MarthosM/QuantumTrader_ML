"""
ProfitDLL Server - Processo Isolado
Executa ProfitDLL em processo separado e envia dados via IPC
"""

import os
import sys
import time
import json
import logging
import multiprocessing
import queue
import signal
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from multiprocessing.connection import Listener, Client
import threading

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.connection_manager_v4 import ConnectionManagerV4


class ProfitDLLServer:
    """
    Servidor que executa ProfitDLL em processo isolado
    Comunica via Named Pipe com o cliente HMARL
    """
    
    def __init__(self, pipe_name: str = "profit_dll_pipe", port: int = 6789):
        self.pipe_name = pipe_name
        self.port = port
        self.logger = self._setup_logger()
        
        # Gerenciamento de conexão
        self.connection_manager = None
        self.listener = None
        self.client_conn = None
        
        # Controle
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Filas para comunicação thread-safe
        self.send_queue = queue.Queue(maxsize=10000)
        self.stats = {
            'messages_sent': 0,
            'errors': 0,
            'start_time': None
        }
        
        self.logger.info(f"ProfitDLL Server inicializado - Porta: {port}")
    
    def _setup_logger(self) -> logging.Logger:
        """Configura logger para o processo servidor"""
        logger = logging.getLogger('ProfitDLLServer')
        logger.setLevel(logging.INFO)
        
        # Handler para arquivo específico do servidor
        fh = logging.FileHandler('logs/profit_dll_server.log')
        fh.setLevel(logging.INFO)
        
        # Handler para console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formato
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def start(self, dll_config: Dict[str, Any]):
        """Inicia o servidor ProfitDLL"""
        try:
            self.is_running = True
            self.stats['start_time'] = datetime.now()
            
            # 1. Iniciar listener para conexões
            self._start_listener()
            
            # 2. Conectar ao ProfitDLL
            if not self._connect_profit_dll(dll_config):
                raise RuntimeError("Falha ao conectar ProfitDLL")
            
            # 3. Thread para enviar dados ao cliente
            send_thread = threading.Thread(target=self._send_loop, daemon=True)
            send_thread.start()
            
            # 4. Aguardar conexão do cliente
            self.logger.info("Aguardando conexão do cliente HMARL...")
            self._wait_for_client()
            
            # 5. Loop principal
            self._main_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Servidor interrompido pelo usuário")
            raise
            
        except Exception as e:
            self.logger.error(f"Erro no servidor: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self.shutdown()
    
    def _start_listener(self):
        """Inicia listener para conexões IPC"""
        try:
            # Windows usa endereço localhost:porta
            address = ('localhost', self.port)
            self.listener = Listener(address, authkey=b'profit_dll_secret')
            self.logger.info(f"Listener iniciado em {address}")
        except Exception as e:
            self.logger.error(f"Erro iniciando listener: {e}")
            raise
    
    def _wait_for_client(self):
        """Aguarda conexão do cliente com timeout"""
        try:
            self.logger.info("Esperando cliente conectar (timeout 10s)...")
            
            # Configurar timeout no listener
            import socket
            self.listener._listener._socket.settimeout(10.0)
            
            try:
                self.client_conn = self.listener.accept()
                self.logger.info(f"Cliente conectado de {self.listener.last_accepted}")
                
                # Enviar mensagem de boas-vindas
                welcome_msg = {
                    'type': 'connection',
                    'status': 'connected',
                    'server_version': '1.0.0',
                    'timestamp': datetime.now().isoformat()
                }
                self.client_conn.send(welcome_msg)
                
            except socket.timeout:
                self.logger.warning("Timeout aguardando cliente - continuando sem cliente")
                self.client_conn = None
                
        except Exception as e:
            self.logger.error(f"Erro aguardando cliente: {e}")
            # Não raise - continuar sem cliente
    
    def _connect_profit_dll(self, config: Dict) -> bool:
        """Conecta ao ProfitDLL"""
        try:
            # Extrair configurações
            dll_path = config['dll_path']
            username = config['username']
            password = config['password']
            key = config['key']
            
            # Criar connection manager
            self.connection_manager = ConnectionManagerV4(dll_path)
            
            # Registrar callback que adiciona à fila
            self.connection_manager.register_trade_callback(self._on_trade_data)
            
            # Conectar
            self.logger.info("Conectando ao ProfitDLL...")
            if self.connection_manager.initialize(
                key=key,
                username=username,
                password=password
            ):
                self.logger.info("✅ ProfitDLL conectado com sucesso!")
                
                # Notificar cliente
                self._queue_message({
                    'type': 'dll_status',
                    'status': 'connected',
                    'timestamp': datetime.now().isoformat()
                })
                
                return True
            else:
                self.logger.error("❌ Falha ao conectar ProfitDLL")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro conectando ProfitDLL: {e}")
            return False
    
    def _on_trade_data(self, trade_data: Dict):
        """Callback para dados de trade - adiciona à fila"""
        try:
            # Adicionar timestamp se não existir
            if 'timestamp' not in trade_data:
                trade_data['timestamp'] = datetime.now().isoformat()
            
            # Adicionar tipo de mensagem
            message = {
                'type': 'trade',
                'data': trade_data
            }
            
            # Adicionar à fila (não bloqueia)
            if not self.send_queue.full():
                self.send_queue.put_nowait(message)
            else:
                # Se fila cheia, descartar mensagens antigas
                try:
                    self.send_queue.get_nowait()
                    self.send_queue.put_nowait(message)
                except:
                    pass
                    
        except Exception as e:
            self.stats['errors'] += 1
    
    def _queue_message(self, message: Dict):
        """Adiciona mensagem à fila de envio"""
        try:
            self.send_queue.put_nowait(message)
        except queue.Full:
            # Remover mensagem mais antiga se fila cheia
            try:
                self.send_queue.get_nowait()
                self.send_queue.put_nowait(message)
            except:
                pass
    
    def _send_loop(self):
        """Loop para enviar dados ao cliente"""
        self.logger.info("Thread de envio iniciada")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Pegar mensagem da fila
                message = self.send_queue.get(timeout=0.1)
                
                # Enviar ao cliente se conectado
                if self.client_conn:
                    try:
                        self.client_conn.send(message)
                        self.stats['messages_sent'] += 1
                    except (EOFError, ConnectionError):
                        self.logger.warning("Cliente desconectado")
                        self.client_conn = None
                        # Tentar reconectar
                        self._wait_for_client()
                        
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Erro no loop de envio: {e}")
                self.stats['errors'] += 1
                time.sleep(0.1)
    
    def _main_loop(self):
        """Loop principal do servidor"""
        self.logger.info("Servidor ProfitDLL rodando...")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Verificar comandos do cliente
                if self.client_conn and self.client_conn.poll(0.1):
                    try:
                        command = self.client_conn.recv()
                        self._handle_command(command)
                    except (EOFError, ConnectionError):
                        self.logger.warning("Cliente desconectado durante recepção")
                        self.client_conn = None
                
                # Enviar heartbeat periodicamente
                if hasattr(self, '_last_heartbeat'):
                    if time.time() - self._last_heartbeat > 30:
                        self._send_heartbeat()
                else:
                    self._last_heartbeat = time.time()
                
                time.sleep(0.01)  # Evitar uso excessivo de CPU
                
            except KeyboardInterrupt:
                self.logger.info("Interrupção do usuário detectada")
                break
            except Exception as e:
                self.logger.error(f"Erro no loop principal: {e}")
                time.sleep(1)
    
    def _handle_command(self, command: Dict):
        """Processa comandos do cliente"""
        try:
            cmd_type = command.get('type')
            
            if cmd_type == 'subscribe':
                ticker = command.get('ticker')
                if ticker and self.connection_manager:
                    success = self.connection_manager.subscribe_ticker(ticker)
                    self._queue_message({
                        'type': 'subscribe_response',
                        'ticker': ticker,
                        'success': success,
                        'timestamp': datetime.now().isoformat()
                    })
                    
            elif cmd_type == 'unsubscribe':
                ticker = command.get('ticker')
                if ticker and self.connection_manager:
                    success = self.connection_manager.unsubscribe_ticker(ticker)
                    self._queue_message({
                        'type': 'unsubscribe_response',
                        'ticker': ticker,
                        'success': success,
                        'timestamp': datetime.now().isoformat()
                    })
                    
            elif cmd_type == 'get_stats':
                self._send_stats()
                
            elif cmd_type == 'ping':
                self._queue_message({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                })
                
            elif cmd_type == 'shutdown':
                self.logger.info("Comando de shutdown recebido")
                self.shutdown_event.set()
                
        except Exception as e:
            self.logger.error(f"Erro processando comando: {e}")
    
    def _send_heartbeat(self):
        """Envia heartbeat ao cliente"""
        self._queue_message({
            'type': 'heartbeat',
            'timestamp': datetime.now().isoformat(),
            'stats': {
                'messages_sent': self.stats['messages_sent'],
                'errors': self.stats['errors'],
                'queue_size': self.send_queue.qsize()
            }
        })
        self._last_heartbeat = time.time()
    
    def _send_stats(self):
        """Envia estatísticas ao cliente"""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds() if self.stats['start_time'] else 0
        
        self._queue_message({
            'type': 'stats',
            'data': {
                'uptime_seconds': uptime,
                'messages_sent': self.stats['messages_sent'],
                'errors': self.stats['errors'],
                'queue_size': self.send_queue.qsize(),
                'dll_connected': self.connection_manager is not None and self.connection_manager.is_connected(),
                'timestamp': datetime.now().isoformat()
            }
        })
    
    def shutdown(self):
        """Desliga o servidor de forma limpa"""
        self.logger.info("Desligando servidor ProfitDLL...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Desconectar ProfitDLL
        if self.connection_manager:
            try:
                self.connection_manager.disconnect()
                self.logger.info("ProfitDLL desconectado")
            except:
                pass
        
        # Fechar conexão com cliente
        if self.client_conn:
            try:
                self.client_conn.send({'type': 'shutdown', 'timestamp': datetime.now().isoformat()})
                self.client_conn.close()
            except:
                pass
        
        # Fechar listener
        if self.listener:
            try:
                self.listener.close()
            except:
                pass
        
        self.logger.info("Servidor ProfitDLL finalizado")


def run_server(config: Dict):
    """Função para executar servidor em processo separado"""
    # Configurar signal handler
    def signal_handler(signum, frame):
        print("\nSinal de interrupção recebido")
        server.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Criar e executar servidor
    server = ProfitDLLServer(port=config.get('port', 6789))
    server.start(config)


if __name__ == "__main__":
    # Teste direto do servidor
    from dotenv import load_dotenv
    load_dotenv()
    
    config = {
        'dll_path': r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll",
        'username': os.getenv("PROFIT_USERNAME"),
        'password': os.getenv("PROFIT_PASSWORD"),
        'key': os.getenv("PROFIT_KEY"),
        'port': 6789
    }
    
    print("Iniciando ProfitDLL Server...")
    run_server(config)