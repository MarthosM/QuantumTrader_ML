"""
Flow-Aware Base Agent - HMARL Fase 1 Semana 3
Classe base para agentes com análise de fluxo
"""

import zmq
import orjson
import time
import uuid
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from datetime import datetime
import numpy as np
import threading
import sys
import os

# Adicionar diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from coordination.agent_registry import AgentRegistryClient
except ImportError:
    AgentRegistryClient = None


class FlowMemory:
    """Memória para armazenar experiências de fluxo"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def add(self, experience: Dict):
        """Adiciona experiência à memória"""
        self.memory.append(experience)
        
    def sample(self, batch_size: int) -> List[Dict]:
        """Amostra aleatória de experiências"""
        if len(self.memory) < batch_size:
            return list(self.memory)
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in indices]
        
    def __len__(self):
        return len(self.memory)


class FlowPatternInterpreter:
    """Interpreta padrões de fluxo"""
    
    def interpret(self, flow_data: Dict) -> Dict:
        """Interpreta dados de fluxo e retorna padrões identificados"""
        patterns = {
            'trend_strength': 0.0,
            'reversal_probability': 0.0,
            'accumulation': False,
            'distribution': False,
            'breakout_imminent': False,
            'sentiment': 'neutral'
        }
        
        # Análise de OFI
        ofi = flow_data.get('order_flow_imbalance', 0)
        if abs(ofi) > 0.3:
            patterns['trend_strength'] = abs(ofi)
            patterns['sentiment'] = 'bullish' if ofi > 0 else 'bearish'
            
        # Análise de agressão
        aggression = flow_data.get('aggression_score', 0)
        if aggression > 0.7:
            patterns['breakout_imminent'] = True
            
        # Análise de delta
        delta = flow_data.get('delta', 0)
        if delta > 100:
            patterns['accumulation'] = True
        elif delta < -100:
            patterns['distribution'] = True
            
        # Análise de padrões
        footprint = flow_data.get('footprint_pattern', '')
        if footprint == 'absorption':
            patterns['reversal_probability'] = 0.7
            
        return patterns


class FlowPatternRecognizer:
    """Reconhecedor de padrões aprendidos"""
    
    def __init__(self):
        self.known_patterns = {}
        self.pattern_performance = {}
        
    def recognize(self, flow_state: Dict) -> List[Dict]:
        """Reconhece padrões conhecidos no estado atual"""
        recognized = []
        
        # Comparar com padrões conhecidos
        for pattern_id, pattern in self.known_patterns.items():
            similarity = self._calculate_similarity(flow_state, pattern)
            if similarity > 0.8:
                recognized.append({
                    'pattern_id': pattern_id,
                    'similarity': similarity,
                    'expected_outcome': pattern.get('outcome'),
                    'confidence': pattern.get('confidence', 0.5)
                })
                
        return recognized
        
    def _calculate_similarity(self, state1: Dict, state2: Dict) -> float:
        """Calcula similaridade entre dois estados"""
        # Simplificado - seria mais complexo na prática
        common_keys = set(state1.keys()) & set(state2.keys())
        if not common_keys:
            return 0.0
            
        differences = []
        for key in common_keys:
            if isinstance(state1[key], (int, float)) and isinstance(state2[key], (int, float)):
                diff = abs(state1[key] - state2[key])
                differences.append(diff)
                
        if not differences:
            return 0.0
            
        avg_diff = np.mean(differences)
        similarity = 1.0 / (1.0 + avg_diff)
        return similarity
        
    def learn_pattern(self, pattern: Dict, outcome: Dict):
        """Aprende novo padrão"""
        pattern_id = str(uuid.uuid4())
        self.known_patterns[pattern_id] = {
            **pattern,
            'outcome': outcome,
            'confidence': 0.5
        }
        self.pattern_performance[pattern_id] = {
            'successes': 0,
            'failures': 0
        }


class FlowAwareBaseAgent(ABC):
    """Classe base para agentes com análise de fluxo"""
    
    def __init__(self, agent_type: str, config: Optional[Dict] = None):
        self.agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
        self.agent_type = agent_type
        self.config = config or {}
        
        # Configurar logging
        self.logger = logging.getLogger(f"Agent.{self.agent_type}.{self.agent_id}")
        
        # ZMQ connections expandidas
        self.context = zmq.Context()
        self.setup_connections()
        
        # Flow analysis components
        self.flow_interpreter = FlowPatternInterpreter()
        self.pattern_recognizer = FlowPatternRecognizer()
        
        # State tracking expandido
        self.state = {
            'price_state': {},
            'flow_state': {},
            'microstructure_state': {},
            'liquidity_state': {}
        }
        
        # Learning components
        self.flow_memory = FlowMemory(capacity=10000)
        
        # Control
        self.is_active = True
        
        # Registry integration
        self.registry_client = None
        self.heartbeat_thread = None
        self._setup_registry()
        
        self.logger.info(f"FlowAwareBaseAgent {self.agent_id} inicializado")
        
    def setup_connections(self):
        """Setup conexões ZMQ incluindo streams de fluxo"""
        # Data subscribers
        self.market_data_sub = self.context.socket(zmq.SUB)
        self.flow_data_sub = self.context.socket(zmq.SUB)
        self.footprint_sub = self.context.socket(zmq.SUB)
        
        # Conectar aos publishers
        self.market_data_sub.connect("tcp://localhost:5555")
        self.flow_data_sub.connect("tcp://localhost:5557")
        self.footprint_sub.connect("tcp://localhost:5558")
        
        # Subscribe to all relevant data
        self.market_data_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.flow_data_sub.setsockopt(zmq.SUBSCRIBE, b"flow_")
        self.footprint_sub.setsockopt(zmq.SUBSCRIBE, b"footprint_")
        
        # Publishers
        self.decision_publisher = self.context.socket(zmq.PUB)
        self.decision_publisher.connect("tcp://localhost:5559")
        
        self.logger.info("Conexões ZMQ estabelecidas")
        
    def process_market_data(self, market_data: Dict):
        """Processa dados de mercado"""
        self.state['price_state'].update({
            'price': market_data.get('price', 0),
            'volume': market_data.get('volume', 0),
            'timestamp': market_data.get('timestamp', time.time())
        })
        
    def process_flow_data(self, flow_data: Dict):
        """Processa dados de fluxo em tempo real"""
        # Atualizar estado de fluxo
        self.state['flow_state'].update({
            'last_ofi': flow_data.get('order_flow_imbalance'),
            'aggression_score': flow_data.get('aggression_score'),
            'delta': flow_data.get('delta'),
            'footprint_pattern': flow_data.get('footprint_pattern')
        })
        
        # Detectar padrões de fluxo
        patterns = self.flow_interpreter.interpret(flow_data)
        self.state['flow_patterns'] = patterns
        
        # Armazenar em memória para aprendizado
        self.flow_memory.add({
            'timestamp': time.time(),
            'flow_data': flow_data,
            'patterns': patterns
        })
        
        self.logger.debug(f"Flow data processado: OFI={flow_data.get('order_flow_imbalance', 0):.3f}")
        
    def process_footprint_data(self, footprint_data: Dict):
        """Processa dados de footprint"""
        self.state['microstructure_state'].update({
            'footprint': footprint_data.get('pattern'),
            'absorption': footprint_data.get('absorption', 0),
            'imbalance': footprint_data.get('imbalance', 0)
        })
        
    @abstractmethod
    def generate_signal_with_flow(self, price_state: Dict, flow_state: Dict) -> Dict:
        """Gerar sinal considerando análise de fluxo - deve ser implementado"""
        pass
        
    def _should_generate_signal(self) -> bool:
        """Verifica se deve gerar sinal"""
        # Verificar se tem dados suficientes
        if not self.state['price_state'] or not self.state['flow_state']:
            return False
            
        # Verificar intervalo mínimo entre sinais
        last_signal_time = self.state.get('last_signal_time', 0)
        min_interval = self.config.get('min_signal_interval', 1.0)  # segundos
        
        if time.time() - last_signal_time < min_interval:
            return False
            
        return True
        
    def _publish_enhanced_signal(self, signal: Dict):
        """Publica sinal aprimorado com contexto de fluxo"""
        enhanced_signal = {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'timestamp': time.time(),
            'signal': signal,
            'flow_context': self.state['flow_state'].copy(),
            'patterns_detected': self.state.get('flow_patterns', {})
        }
        
        # Publicar via ZMQ
        topic = f"signal_{self.agent_type}".encode()
        message = orjson.dumps(enhanced_signal)
        self.decision_publisher.send_multipart([topic, message])
        
        # Atualizar timestamp do último sinal
        self.state['last_signal_time'] = time.time()
        
        self.logger.info(f"Sinal publicado: {signal['action']} com confiança {signal['confidence']:.2f}")
        
    def run_enhanced_agent_loop(self):
        """Loop principal com processamento de fluxo"""
        poller = zmq.Poller()
        poller.register(self.market_data_sub, zmq.POLLIN)
        poller.register(self.flow_data_sub, zmq.POLLIN)
        poller.register(self.footprint_sub, zmq.POLLIN)
        
        self.logger.info(f"Agent {self.agent_id} iniciando loop principal")
        
        while self.is_active:
            try:
                socks = dict(poller.poll(100))  # 100ms timeout
                
                # Processar market data
                if self.market_data_sub in socks:
                    topic, data = self.market_data_sub.recv_multipart()
                    market_data = orjson.loads(data)
                    self.process_market_data(market_data)
                    
                # Processar flow data
                if self.flow_data_sub in socks:
                    topic, data = self.flow_data_sub.recv_multipart()
                    flow_data = orjson.loads(data)
                    self.process_flow_data(flow_data)
                    
                # Processar footprint
                if self.footprint_sub in socks:
                    topic, data = self.footprint_sub.recv_multipart()
                    footprint_data = orjson.loads(data)
                    self.process_footprint_data(footprint_data)
                    
                # Gerar sinal se condições adequadas
                if self._should_generate_signal():
                    signal = self.generate_signal_with_flow(
                        self.state['price_state'],
                        self.state['flow_state']
                    )
                    
                    if signal and signal.get('confidence', 0) > self.config.get('min_confidence', 0.3):
                        self._publish_enhanced_signal(signal)
                        
            except Exception as e:
                self.logger.error(f"Error in agent loop: {e}")
                
    def learn_from_feedback(self, feedback: Dict):
        """Aprende com feedback sobre decisões tomadas"""
        # Extrair informações relevantes
        decision_id = feedback.get('decision_id')
        reward = feedback.get('reward', 0)
        flow_context = feedback.get('flow_context', {})
        
        # Atualizar padrões reconhecidos
        if 'patterns' in feedback:
            for pattern in feedback['patterns']:
                self.pattern_recognizer.learn_pattern(
                    pattern,
                    {'reward': reward, 'action': feedback.get('action')}
                )
                
        self.logger.info(f"Aprendizado com reward={reward:.2f}")
        
    def get_state_summary(self) -> Dict:
        """Retorna resumo do estado atual"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'price_state': self.state['price_state'].copy(),
            'flow_state': self.state['flow_state'].copy(),
            'patterns': self.state.get('flow_patterns', {}),
            'memory_size': len(self.flow_memory),
            'is_active': self.is_active
        }
        
    def _setup_registry(self):
        """Configura integração com registro de agentes"""
        if AgentRegistryClient and self.config.get('use_registry', True):
            try:
                self.registry_client = AgentRegistryClient()
                
                # Registrar agente
                capabilities = self._get_agent_capabilities()
                registry_info = {
                    'agent_id': self.agent_id,
                    'agent_type': self.agent_type,
                    'capabilities': capabilities,
                    'pub_address': 'tcp://localhost:5559',
                    'sub_topics': ['market_data', 'flow_data', 'footprint_data']
                }
                
                result = self.registry_client.register(registry_info)
                if result.get('status') == 'registered':
                    self.logger.info(f"Agente registrado com sucesso no registry")
                    
                    # Iniciar thread de heartbeat
                    self.heartbeat_thread = threading.Thread(
                        target=self._heartbeat_loop,
                        daemon=True
                    )
                    self.heartbeat_thread.start()
                else:
                    self.logger.warning(f"Falha ao registrar agente: {result}")
                    
            except Exception as e:
                self.logger.warning(f"Registry não disponível: {e}")
                self.registry_client = None
                
    def _get_agent_capabilities(self) -> List[str]:
        """Retorna capacidades do agente"""
        # Base capabilities
        capabilities = ['flow_analysis', 'pattern_recognition']
        
        # Adicionar capacidades específicas por tipo
        if self.agent_type == 'order_flow_specialist':
            capabilities.extend(['ofi_analysis', 'delta_analysis', 'sweep_detection'])
        elif self.agent_type == 'liquidity_specialist':
            capabilities.extend(['depth_analysis', 'iceberg_detection', 'hidden_liquidity'])
        elif self.agent_type == 'tape_reading':
            capabilities.extend(['tape_speed', 'tape_patterns', 'momentum_tracking'])
        elif self.agent_type == 'footprint_pattern':
            capabilities.extend(['footprint_patterns', 'reversal_detection', 'absorption'])
            
        return capabilities
        
    def _heartbeat_loop(self):
        """Thread para enviar heartbeats ao registry"""
        while self.is_active and self.registry_client:
            try:
                result = self.registry_client.heartbeat(self.agent_id)
                if result.get('status') != 'ok':
                    self.logger.warning(f"Heartbeat falhou: {result}")
                    
                # Atualizar métricas
                metrics = {
                    'signals_generated': self.state.get('signals_count', 0),
                    'avg_confidence': self.state.get('avg_confidence', 0),
                    'last_signal_time': self.state.get('last_signal_time')
                }
                self.registry_client.update_metrics(self.agent_id, metrics)
                
            except Exception as e:
                self.logger.error(f"Erro no heartbeat: {e}")
                
            time.sleep(10)  # Heartbeat a cada 10 segundos
            
    def shutdown(self):
        """Desliga o agente de forma limpa"""
        self.logger.info(f"Desligando agent {self.agent_id}")
        self.is_active = False
        
        # Desregistrar do registry
        if self.registry_client:
            try:
                self.registry_client.unregister(self.agent_id)
                self.registry_client.close()
            except Exception as e:
                self.logger.error(f"Erro ao desregistrar: {e}")
                
        # Fechar conexões ZMQ
        self.market_data_sub.close()
        self.flow_data_sub.close()
        self.footprint_sub.close()
        self.decision_publisher.close()
        self.context.term()


# Exemplo de implementação concreta
class SimpleFlowAgent(FlowAwareBaseAgent):
    """Implementação simples de agente com análise de fluxo"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__('simple_flow', config)
        
    def generate_signal_with_flow(self, price_state: Dict, flow_state: Dict) -> Dict:
        """Gera sinal simples baseado em fluxo"""
        signal = {
            'action': 'hold',
            'confidence': 0.0,
            'metadata': {}
        }
        
        # Análise simples de OFI
        ofi = flow_state.get('last_ofi', 0)
        
        if ofi > 0.3:
            signal['action'] = 'buy'
            signal['confidence'] = min(ofi * 2, 1.0)
            signal['metadata']['reason'] = 'positive_ofi'
        elif ofi < -0.3:
            signal['action'] = 'sell'
            signal['confidence'] = min(abs(ofi) * 2, 1.0)
            signal['metadata']['reason'] = 'negative_ofi'
            
        return signal


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Criar e testar agente
    config = {
        'min_confidence': 0.3,
        'min_signal_interval': 1.0
    }
    
    agent = SimpleFlowAgent(config)
    
    # Simular alguns dados
    agent.process_market_data({
        'price': 5000.0,
        'volume': 100,
        'timestamp': time.time()
    })
    
    agent.process_flow_data({
        'order_flow_imbalance': 0.45,
        'aggression_score': 0.7,
        'delta': 150,
        'footprint_pattern': 'accumulation'
    })
    
    # Testar geração de sinal
    signal = agent.generate_signal_with_flow(
        agent.state['price_state'],
        agent.state['flow_state']
    )
    
    print(f"Sinal gerado: {signal}")
    print(f"Estado do agente: {agent.get_state_summary()}")
    
    agent.shutdown()