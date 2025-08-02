"""
Agent Registry - Sistema de Registro e Descoberta de Agentes HMARL
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import zmq
import orjson


class AgentInfo:
    """Informações sobre um agente registrado"""
    
    def __init__(self, agent_id: str, agent_type: str, capabilities: List[str],
                 pub_address: str, sub_topics: List[str]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.pub_address = pub_address
        self.sub_topics = sub_topics
        self.registered_at = time.time()
        self.last_heartbeat = time.time()
        self.status = 'active'
        self.performance_metrics = {
            'signals_generated': 0,
            'avg_confidence': 0.0,
            'success_rate': 0.0,
            'last_signal_time': None
        }
        
    def update_heartbeat(self):
        """Atualiza timestamp do heartbeat"""
        self.last_heartbeat = time.time()
        
    def is_alive(self, timeout: float = 30.0) -> bool:
        """Verifica se agente está vivo baseado em heartbeat"""
        return (time.time() - self.last_heartbeat) < timeout
        
    def to_dict(self) -> Dict:
        """Converte para dicionário"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'capabilities': self.capabilities,
            'pub_address': self.pub_address,
            'sub_topics': self.sub_topics,
            'status': self.status,
            'registered_at': self.registered_at,
            'last_heartbeat': self.last_heartbeat,
            'performance_metrics': self.performance_metrics
        }


class AgentRegistry:
    """Registro central de agentes HMARL"""
    
    def __init__(self, heartbeat_timeout: float = 30.0):
        self.logger = logging.getLogger(f"{__name__}.AgentRegistry")
        self.agents = {}  # agent_id -> AgentInfo
        self.agents_by_type = defaultdict(set)  # agent_type -> set of agent_ids
        self.agents_by_capability = defaultdict(set)  # capability -> set of agent_ids
        self.heartbeat_timeout = heartbeat_timeout
        self.lock = threading.RLock()
        
        # ZMQ para receber registros e heartbeats
        self.context = zmq.Context()
        self.registry_socket = self.context.socket(zmq.REP)
        self.registry_socket.bind("tcp://*:5560")
        
        # Thread para monitorar heartbeats
        self.monitor_thread = threading.Thread(target=self._monitor_agents, daemon=True)
        self.monitor_thread.start()
        
        # Thread para processar requisições
        self.request_thread = threading.Thread(target=self._process_requests, daemon=True)
        self.request_thread.start()
        
        self.logger.info("AgentRegistry inicializado")
        
    def register_agent(self, agent_info: Dict) -> Dict:
        """Registra um novo agente"""
        with self.lock:
            agent_id = agent_info['agent_id']
            
            # Criar objeto AgentInfo
            info = AgentInfo(
                agent_id=agent_id,
                agent_type=agent_info['agent_type'],
                capabilities=agent_info.get('capabilities', []),
                pub_address=agent_info['pub_address'],
                sub_topics=agent_info.get('sub_topics', [])
            )
            
            # Registrar
            self.agents[agent_id] = info
            self.agents_by_type[info.agent_type].add(agent_id)
            
            # Indexar por capacidades
            for capability in info.capabilities:
                self.agents_by_capability[capability].add(agent_id)
                
            self.logger.info(f"Agente registrado: {agent_id} ({info.agent_type})")
            
            return {
                'status': 'registered',
                'agent_id': agent_id,
                'registry_time': info.registered_at
            }
            
    def unregister_agent(self, agent_id: str) -> Dict:
        """Remove agente do registro"""
        with self.lock:
            if agent_id not in self.agents:
                return {'status': 'not_found', 'agent_id': agent_id}
                
            info = self.agents[agent_id]
            
            # Remover de todos os índices
            del self.agents[agent_id]
            self.agents_by_type[info.agent_type].discard(agent_id)
            
            for capability in info.capabilities:
                self.agents_by_capability[capability].discard(agent_id)
                
            self.logger.info(f"Agente removido: {agent_id}")
            
            return {'status': 'unregistered', 'agent_id': agent_id}
            
    def heartbeat(self, agent_id: str) -> Dict:
        """Processa heartbeat de agente"""
        with self.lock:
            if agent_id not in self.agents:
                return {'status': 'not_registered', 'agent_id': agent_id}
                
            self.agents[agent_id].update_heartbeat()
            
            return {
                'status': 'ok',
                'agent_id': agent_id,
                'timestamp': time.time()
            }
            
    def get_agents_by_type(self, agent_type: str) -> List[Dict]:
        """Retorna agentes de um tipo específico"""
        with self.lock:
            agent_ids = self.agents_by_type.get(agent_type, set())
            return [
                self.agents[aid].to_dict() 
                for aid in agent_ids 
                if aid in self.agents and self.agents[aid].is_alive(self.heartbeat_timeout)
            ]
            
    def get_agents_by_capability(self, capability: str) -> List[Dict]:
        """Retorna agentes com uma capacidade específica"""
        with self.lock:
            agent_ids = self.agents_by_capability.get(capability, set())
            return [
                self.agents[aid].to_dict() 
                for aid in agent_ids 
                if aid in self.agents and self.agents[aid].is_alive(self.heartbeat_timeout)
            ]
            
    def get_all_agents(self, only_active: bool = True) -> List[Dict]:
        """Retorna todos os agentes registrados"""
        with self.lock:
            if only_active:
                return [
                    info.to_dict() 
                    for info in self.agents.values() 
                    if info.is_alive(self.heartbeat_timeout)
                ]
            else:
                return [info.to_dict() for info in self.agents.values()]
                
    def update_agent_metrics(self, agent_id: str, metrics: Dict) -> Dict:
        """Atualiza métricas de performance do agente"""
        with self.lock:
            if agent_id not in self.agents:
                return {'status': 'not_found', 'agent_id': agent_id}
                
            agent = self.agents[agent_id]
            agent.performance_metrics.update(metrics)
            
            return {'status': 'updated', 'agent_id': agent_id}
            
    def _monitor_agents(self):
        """Thread para monitorar status dos agentes"""
        while True:
            try:
                with self.lock:
                    # Verificar agentes inativos
                    for agent_id, info in list(self.agents.items()):
                        if not info.is_alive(self.heartbeat_timeout):
                            if info.status == 'active':
                                info.status = 'inactive'
                                self.logger.warning(f"Agente inativo detectado: {agent_id}")
                        else:
                            if info.status == 'inactive':
                                info.status = 'active'
                                self.logger.info(f"Agente reativado: {agent_id}")
                                
                time.sleep(5)  # Verificar a cada 5 segundos
                
            except Exception as e:
                self.logger.error(f"Erro no monitor de agentes: {e}")
                time.sleep(1)
                
    def _process_requests(self):
        """Thread para processar requisições do registro"""
        while True:
            try:
                # Receber requisição
                message = self.registry_socket.recv()
                request = orjson.loads(message)
                
                # Processar por tipo
                request_type = request.get('type')
                response = {}
                
                if request_type == 'register':
                    response = self.register_agent(request['data'])
                    
                elif request_type == 'unregister':
                    response = self.unregister_agent(request['agent_id'])
                    
                elif request_type == 'heartbeat':
                    response = self.heartbeat(request['agent_id'])
                    
                elif request_type == 'get_by_type':
                    response = {
                        'agents': self.get_agents_by_type(request['agent_type'])
                    }
                    
                elif request_type == 'get_by_capability':
                    response = {
                        'agents': self.get_agents_by_capability(request['capability'])
                    }
                    
                elif request_type == 'get_all':
                    response = {
                        'agents': self.get_all_agents(request.get('only_active', True))
                    }
                    
                elif request_type == 'update_metrics':
                    response = self.update_agent_metrics(
                        request['agent_id'],
                        request['metrics']
                    )
                    
                else:
                    response = {'status': 'unknown_request', 'type': request_type}
                    
                # Enviar resposta
                self.registry_socket.send(orjson.dumps(response))
                
            except Exception as e:
                self.logger.error(f"Erro processando requisição: {e}")
                # Enviar erro como resposta
                error_response = {'status': 'error', 'message': str(e)}
                self.registry_socket.send(orjson.dumps(error_response))
                
    def get_registry_stats(self) -> Dict:
        """Retorna estatísticas do registro"""
        with self.lock:
            total_agents = len(self.agents)
            active_agents = sum(1 for a in self.agents.values() if a.is_alive(self.heartbeat_timeout))
            
            type_distribution = {
                agent_type: len(agents) 
                for agent_type, agents in self.agents_by_type.items()
            }
            
            capability_distribution = {
                cap: len(agents) 
                for cap, agents in self.agents_by_capability.items()
            }
            
            return {
                'total_agents': total_agents,
                'active_agents': active_agents,
                'inactive_agents': total_agents - active_agents,
                'type_distribution': type_distribution,
                'capability_distribution': capability_distribution,
                'registry_uptime': time.time() - self.agents[next(iter(self.agents))].registered_at if self.agents else 0
            }
            
    def shutdown(self):
        """Desliga o registro"""
        self.logger.info("Desligando AgentRegistry...")
        self.registry_socket.close()
        self.context.term()


class AgentRegistryClient:
    """Cliente para interagir com o AgentRegistry"""
    
    def __init__(self, registry_address: str = "tcp://localhost:5560"):
        self.logger = logging.getLogger(f"{__name__}.RegistryClient")
        self.registry_address = registry_address
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(registry_address)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # Timeout de 5 segundos
        
    def register(self, agent_info: Dict) -> Dict:
        """Registra agente no registro"""
        request = {
            'type': 'register',
            'data': agent_info
        }
        return self._send_request(request)
        
    def unregister(self, agent_id: str) -> Dict:
        """Remove agente do registro"""
        request = {
            'type': 'unregister',
            'agent_id': agent_id
        }
        return self._send_request(request)
        
    def heartbeat(self, agent_id: str) -> Dict:
        """Envia heartbeat"""
        request = {
            'type': 'heartbeat',
            'agent_id': agent_id
        }
        return self._send_request(request)
        
    def get_agents_by_type(self, agent_type: str) -> List[Dict]:
        """Busca agentes por tipo"""
        request = {
            'type': 'get_by_type',
            'agent_type': agent_type
        }
        response = self._send_request(request)
        return response.get('agents', [])
        
    def get_agents_by_capability(self, capability: str) -> List[Dict]:
        """Busca agentes por capacidade"""
        request = {
            'type': 'get_by_capability',
            'capability': capability
        }
        response = self._send_request(request)
        return response.get('agents', [])
        
    def get_all_agents(self, only_active: bool = True) -> List[Dict]:
        """Busca todos os agentes"""
        request = {
            'type': 'get_all',
            'only_active': only_active
        }
        response = self._send_request(request)
        return response.get('agents', [])
        
    def update_metrics(self, agent_id: str, metrics: Dict) -> Dict:
        """Atualiza métricas do agente"""
        request = {
            'type': 'update_metrics',
            'agent_id': agent_id,
            'metrics': metrics
        }
        return self._send_request(request)
        
    def _send_request(self, request: Dict) -> Dict:
        """Envia requisição e recebe resposta"""
        try:
            self.socket.send(orjson.dumps(request))
            response = self.socket.recv()
            return orjson.loads(response)
        except zmq.Again:
            self.logger.error("Timeout ao comunicar com registro")
            return {'status': 'timeout'}
        except Exception as e:
            self.logger.error(f"Erro na comunicação: {e}")
            return {'status': 'error', 'message': str(e)}
            
    def close(self):
        """Fecha conexão"""
        self.socket.close()
        self.context.term()


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Criar registro
    registry = AgentRegistry()
    
    # Simular registro de agentes
    client = AgentRegistryClient()
    
    # Registrar agente de order flow
    order_flow_info = {
        'agent_id': 'order_flow_001',
        'agent_type': 'order_flow_specialist',
        'capabilities': ['ofi_analysis', 'delta_analysis', 'sweep_detection'],
        'pub_address': 'tcp://localhost:5561',
        'sub_topics': ['market_data', 'orderbook_updates']
    }
    
    result = client.register(order_flow_info)
    print(f"Registro Order Flow: {result}")
    
    # Registrar agente de liquidez
    liquidity_info = {
        'agent_id': 'liquidity_001',
        'agent_type': 'liquidity_specialist',
        'capabilities': ['depth_analysis', 'iceberg_detection', 'hidden_liquidity'],
        'pub_address': 'tcp://localhost:5562',
        'sub_topics': ['orderbook_updates', 'trade_executions']
    }
    
    result = client.register(liquidity_info)
    print(f"Registro Liquidity: {result}")
    
    # Buscar agentes
    print(f"\nTodos os agentes: {client.get_all_agents()}")
    print(f"\nAgentes com capacidade 'delta_analysis': {client.get_agents_by_capability('delta_analysis')}")
    
    # Enviar heartbeat
    result = client.heartbeat('order_flow_001')
    print(f"\nHeartbeat: {result}")
    
    # Estatísticas
    print(f"\nEstatísticas do registro: {registry.get_registry_stats()}")
    
    # Limpar
    client.close()
    time.sleep(1)
    registry.shutdown()