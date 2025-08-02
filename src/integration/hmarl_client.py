"""
HMARL Client - Conecta ao ProfitDLL Server via IPC
Recebe dados do servidor isolado e processa com sistema HMARL
"""

import time
import logging
import json
import queue
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from multiprocessing.connection import Client
import traceback

# Componentes HMARL
from src.infrastructure.zmq_valkey_flow_setup import TradingInfrastructureWithFlow
from src.agents.order_flow_specialist import OrderFlowSpecialistAgent
from src.agents.footprint_pattern_agent import FootprintPatternAgent
from src.coordination.flow_aware_coordinator import FlowAwareCoordinator
from src.features.flow_feature_system import FlowFeatureSystem


class HMARLClient:
    """
    Cliente HMARL que recebe dados do ProfitDLL Server
    Processa dados com sistema multi-agente sem risco de Segmentation Fault
    """
    
    def __init__(self, server_address: tuple = ('localhost', 6789), config: Dict = None):
        self.server_address = server_address
        self.config = config or {}
        self.logger = logging.getLogger('HMARLClient')
        
        # Conexão com servidor
        self.connection = None
        self.is_connected = False
        
        # Componentes HMARL
        self.infrastructure = None
        self.flow_feature_system = None
        self.flow_coordinator = None
        self.agents = {}
        
        # Controle
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Fila para processar dados
        self.data_queue = queue.Queue(maxsize=1000)
        
        # Métricas
        self.metrics = {
            'messages_received': 0,
            'trades_processed': 0,
            'features_calculated': 0,
            'agent_signals': 0,
            'errors': 0,
            'start_time': None
        }
        
        self.logger.info("HMARL Client inicializado")
    
    def initialize(self) -> bool:
        """Inicializa componentes HMARL"""
        try:
            self.logger.info("Inicializando componentes HMARL...")
            
            # 1. Infraestrutura
            self.infrastructure = TradingInfrastructureWithFlow(self.config)
            if not self.infrastructure.initialize():
                raise RuntimeError("Falha ao inicializar infraestrutura")
            
            # 2. Sistema de features
            self.flow_feature_system = FlowFeatureSystem(self.infrastructure.valkey_client)
            
            # 3. Coordenador
            valkey_config = self.config.get('valkey', {
                'host': 'localhost',
                'port': 6379
            })
            self.flow_coordinator = FlowAwareCoordinator(valkey_config)
            
            # 4. Agentes
            self._initialize_agents()
            
            self.logger.info("✅ Componentes HMARL inicializados com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro inicializando HMARL: {e}")
            return False
    
    def _initialize_agents(self):
        """Inicializa agentes especializados"""
        try:
            # Configuração base para agentes
            agent_config = {
                'use_registry': False,
                'min_signal_interval': 1.0
            }
            
            # Order Flow Specialist
            self.agents['order_flow'] = OrderFlowSpecialistAgent({
                **agent_config,
                'ofi_threshold': 0.3,
                'delta_threshold': 1000,
                'aggression_threshold': 0.6
            })
            
            # Footprint Pattern
            self.agents['footprint'] = FootprintPatternAgent({
                **agent_config,
                'pattern_confidence_threshold': 0.7
            })
            
            # Registrar agentes no Valkey
            for agent_name, agent in self.agents.items():
                self.infrastructure.valkey_client.xadd(
                    f'agent_registry:{agent_name}',
                    {
                        'agent_id': agent.agent_id,
                        'agent_type': agent.agent_type,
                        'status': 'active',
                        'timestamp': str(datetime.now())
                    }
                )
            
            self.logger.info(f"✅ {len(self.agents)} agentes inicializados")
            
        except Exception as e:
            self.logger.error(f"Erro inicializando agentes: {e}")
            raise
    
    def connect_to_server(self) -> bool:
        """Conecta ao servidor ProfitDLL"""
        try:
            self.logger.info(f"Conectando ao servidor em {self.server_address}...")
            
            # Conectar usando multiprocessing.connection
            self.connection = Client(self.server_address, authkey=b'profit_dll_secret')
            
            # Receber mensagem de boas-vindas
            welcome = self.connection.recv()
            self.logger.info(f"Servidor respondeu: {welcome}")
            
            if welcome.get('status') == 'connected':
                self.is_connected = True
                self.logger.info("✅ Conectado ao servidor ProfitDLL")
                return True
            else:
                self.logger.error("❌ Resposta inesperada do servidor")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro conectando ao servidor: {e}")
            return False
    
    def start(self):
        """Inicia o cliente HMARL"""
        try:
            self.is_running = True
            self.metrics['start_time'] = datetime.now()
            
            # Thread para receber dados do servidor
            receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            receive_thread.start()
            
            # Thread para processar dados
            process_thread = threading.Thread(target=self._process_loop, daemon=True)
            process_thread.start()
            
            # Thread para análise de fluxo
            flow_thread = threading.Thread(target=self._flow_analysis_loop, daemon=True)
            flow_thread.start()
            
            # Thread para coordenação de agentes
            coord_thread = threading.Thread(target=self._coordination_loop, daemon=True)
            coord_thread.start()
            
            # Iniciar agentes
            for agent_name, agent in self.agents.items():
                agent_thread = threading.Thread(
                    target=agent.run_enhanced_agent_loop,
                    name=f"Agent-{agent_name}",
                    daemon=True
                )
                agent_thread.start()
            
            self.logger.info("🚀 Cliente HMARL iniciado com sucesso!")
            
            # Loop principal
            self._main_loop()
            
        except Exception as e:
            self.logger.error(f"Erro iniciando cliente: {e}")
            self.stop()
    
    def _receive_loop(self):
        """Loop para receber dados do servidor"""
        self.logger.info("Thread de recepção iniciada")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                if self.connection and self.connection.poll(0.1):
                    message = self.connection.recv()
                    self.metrics['messages_received'] += 1
                    
                    # Processar mensagem
                    msg_type = message.get('type')
                    
                    if msg_type == 'trade':
                        # Adicionar à fila de processamento
                        if not self.data_queue.full():
                            self.data_queue.put(message['data'])
                            
                    elif msg_type == 'heartbeat':
                        self.logger.debug(f"Heartbeat recebido: {message.get('stats')}")
                        
                    elif msg_type == 'dll_status':
                        self.logger.info(f"Status DLL: {message.get('status')}")
                        
                    elif msg_type == 'shutdown':
                        self.logger.warning("Servidor solicitou shutdown")
                        self.shutdown_event.set()
                        
            except EOFError:
                self.logger.error("Conexão com servidor perdida")
                self.is_connected = False
                # Tentar reconectar
                time.sleep(5)
                if self.connect_to_server():
                    self.logger.info("Reconectado ao servidor")
                    
            except Exception as e:
                self.logger.error(f"Erro recebendo dados: {e}")
                self.metrics['errors'] += 1
                time.sleep(0.1)
    
    def _process_loop(self):
        """Loop para processar dados de trade"""
        self.logger.info("Thread de processamento iniciada")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Pegar dado da fila
                trade_data = self.data_queue.get(timeout=0.1)
                
                # Processar trade
                self._process_trade(trade_data)
                
                self.metrics['trades_processed'] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Erro processando trade: {e}")
                self.metrics['errors'] += 1
    
    def _process_trade(self, trade_data: Dict):
        """Processa dados de trade com HMARL"""
        try:
            # Enriquecer dados
            enriched_data = {
                'timestamp': trade_data.get('timestamp', datetime.now()),
                'ticker': trade_data.get('ticker', ''),
                'symbol': trade_data.get('ticker', ''),  # Adicionar symbol para compatibilidade
                'price': trade_data.get('price', 0),
                'volume': trade_data.get('volume', 0),
                'quantity': trade_data.get('quantity', 0),
                'trade_type': trade_data.get('trade_type', 0)
            }
            
            # Publicar via infraestrutura
            if self.infrastructure:
                self.infrastructure.publish_tick_with_flow(enriched_data)
            
            # Armazenar no Valkey
            if self.infrastructure and self.infrastructure.valkey_client:
                self.infrastructure.valkey_client.xadd(
                    f'market_data:{enriched_data["ticker"]}',
                    {
                        'timestamp': str(enriched_data['timestamp']),
                        'price': enriched_data['price'],
                        'volume': enriched_data['volume']
                    },
                    maxlen=100000
                )
            
            self.metrics['features_calculated'] += 1
            
        except Exception as e:
            self.logger.error(f"Erro em _process_trade: {e}")
    
    def _flow_analysis_loop(self):
        """Loop de análise de fluxo"""
        self.logger.info("Thread de análise de fluxo iniciada")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                time.sleep(5)  # Análise a cada 5 segundos
                
                # Análise de fluxo seria implementada aqui
                # Por enquanto, apenas log
                self.logger.debug("Análise de fluxo executada")
                
            except Exception as e:
                self.logger.error(f"Erro na análise de fluxo: {e}")
                time.sleep(1)
    
    def _coordination_loop(self):
        """Loop de coordenação de agentes"""
        self.logger.info("Thread de coordenação iniciada")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                time.sleep(10)  # Coordenação a cada 10 segundos
                
                # Coordenar decisões dos agentes
                if self.flow_coordinator:
                    best_strategy = self.flow_coordinator.coordinate_with_flow_analysis()
                    
                    if best_strategy:
                        self.logger.info(f"Estratégia selecionada: {best_strategy}")
                        self.metrics['agent_signals'] += 1
                        
            except Exception as e:
                self.logger.error(f"Erro na coordenação: {e}")
                time.sleep(1)
    
    def _main_loop(self):
        """Loop principal do cliente"""
        self.logger.info("Cliente HMARL rodando...")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Enviar ping periodicamente
                if hasattr(self, '_last_ping'):
                    if time.time() - self._last_ping > 30:
                        self.send_command({'type': 'ping'})
                        self._last_ping = time.time()
                else:
                    self._last_ping = time.time()
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                self.logger.info("Interrupção do usuário")
                break
            except Exception as e:
                self.logger.error(f"Erro no loop principal: {e}")
    
    def send_command(self, command: Dict) -> bool:
        """Envia comando ao servidor"""
        try:
            if self.connection and self.is_connected:
                self.connection.send(command)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Erro enviando comando: {e}")
            return False
    
    def subscribe_ticker(self, ticker: str) -> bool:
        """Solicita subscrição de ticker ao servidor"""
        return self.send_command({
            'type': 'subscribe',
            'ticker': ticker
        })
    
    def get_metrics(self) -> Dict:
        """Retorna métricas do cliente"""
        uptime = (datetime.now() - self.metrics['start_time']).total_seconds() if self.metrics['start_time'] else 0
        
        return {
            **self.metrics,
            'uptime_seconds': uptime,
            'queue_size': self.data_queue.qsize(),
            'is_connected': self.is_connected,
            'active_agents': len(self.agents)
        }
    
    def stop(self):
        """Para o cliente de forma limpa"""
        self.logger.info("Parando cliente HMARL...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Notificar servidor
        if self.connection:
            try:
                self.send_command({'type': 'shutdown'})
                self.connection.close()
            except:
                pass
        
        # Parar agentes
        for agent in self.agents.values():
            agent.is_active = False
        
        # Parar infraestrutura
        if self.infrastructure:
            self.infrastructure.stop()
        
        self.logger.info("Cliente HMARL finalizado")


if __name__ == "__main__":
    # Teste direto do cliente
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = {
        'ticker': 'WDOQ25',
        'valkey': {
            'host': 'localhost',
            'port': 6379
        }
    }
    
    client = HMARLClient(config=config)
    
    if client.initialize() and client.connect_to_server():
        client.start()
    else:
        print("Falha ao inicializar cliente")