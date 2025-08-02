"""
Testes de Integração HMARL - Fase 1
"""

import unittest
import time
import threading
import logging
import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
import zmq
import orjson

# Adicionar diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.flow_feature_system import FlowFeatureSystem
from agents.order_flow_specialist import OrderFlowSpecialistAgent
from agents.footprint_pattern_agent import FootprintPatternAgent
from agents.liquidity_agent import LiquidityAgent
from agents.tape_reading_agent import TapeReadingAgent
from systems.flow_aware_feedback_system import FlowAwareFeedbackSystem
from coordination.agent_registry import AgentRegistry, AgentRegistryClient
from coordination.flow_aware_coordinator import FlowAwareCoordinator


class MockDataPublisher:
    """Publisher de dados mock para testes"""
    
    def __init__(self):
        self.context = zmq.Context()
        self.market_data_pub = self.context.socket(zmq.PUB)
        self.market_data_pub.bind("tcp://*:5555")
        
        self.flow_data_pub = self.context.socket(zmq.PUB)
        self.flow_data_pub.bind("tcp://*:5557")
        
        self.footprint_pub = self.context.socket(zmq.PUB)
        self.footprint_pub.bind("tcp://*:5558")
        
        self.is_running = False
        self.publish_thread = None
        
    def start(self):
        """Inicia publicação de dados mock"""
        self.is_running = True
        self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.publish_thread.start()
        
    def stop(self):
        """Para publicação"""
        self.is_running = False
        if self.publish_thread:
            self.publish_thread.join(timeout=2)
            
    def _publish_loop(self):
        """Loop de publicação de dados"""
        base_price = 5000.0
        
        while self.is_running:
            try:
                # Market data
                market_data = {
                    'timestamp': time.time(),
                    'price': base_price + np.random.randn() * 5,
                    'volume': np.random.randint(50, 200),
                    'bid': base_price - 0.5,
                    'ask': base_price + 0.5
                }
                self.market_data_pub.send_multipart([
                    b"market_data",
                    orjson.dumps(market_data)
                ])
                
                # Flow data
                flow_data = {
                    'timestamp': time.time(),
                    'order_flow_imbalance': np.random.uniform(-0.8, 0.8),
                    'aggression_score': np.random.uniform(0, 1),
                    'delta': np.random.randint(-200, 200),
                    'buy_volume': np.random.randint(100, 500),
                    'sell_volume': np.random.randint(100, 500)
                }
                self.flow_data_pub.send_multipart([
                    b"flow_ofi",
                    orjson.dumps(flow_data)
                ])
                
                # Footprint data
                patterns = ['p_reversal', 'b_reversal', 'continuation', 'absorption', 'exhaustion']
                footprint_data = {
                    'timestamp': time.time(),
                    'pattern': np.random.choice(patterns),
                    'absorption': np.random.uniform(0, 1),
                    'imbalance': np.random.uniform(-1, 1),
                    'confidence': np.random.uniform(0.5, 1)
                }
                self.footprint_pub.send_multipart([
                    b"footprint_pattern",
                    orjson.dumps(footprint_data)
                ])
                
                # Orderbook mock
                orderbook = self._generate_mock_orderbook(base_price)
                self.flow_data_pub.send_multipart([
                    b"orderbook",
                    orjson.dumps(orderbook)
                ])
                
                # Trades mock
                trades = self._generate_mock_trades(base_price)
                self.flow_data_pub.send_multipart([
                    b"recent_trades",
                    orjson.dumps(trades)
                ])
                
                time.sleep(0.1)  # 10Hz
                
            except Exception as e:
                print(f"Erro no publisher: {e}")
                
    def _generate_mock_orderbook(self, base_price):
        """Gera orderbook mock"""
        bids = []
        asks = []
        
        for i in range(10):
            bids.append({
                'price': base_price - (i+1) * 0.5,
                'volume': np.random.randint(50, 200)
            })
            asks.append({
                'price': base_price + (i+1) * 0.5,
                'volume': np.random.randint(50, 200)
            })
            
        return {'bids': bids, 'asks': asks}
        
    def _generate_mock_trades(self, base_price):
        """Gera trades mock"""
        trades = []
        current_time = time.time()
        
        for i in range(20):
            trades.append({
                'timestamp': current_time - (20-i),
                'price': base_price + np.random.randn() * 2,
                'volume': np.random.randint(10, 100),
                'side': np.random.choice(['buy', 'sell'])
            })
            
        return trades
        
    def close(self):
        """Fecha conexões"""
        self.stop()
        self.market_data_pub.close()
        self.flow_data_pub.close()
        self.footprint_pub.close()
        self.context.term()


class TestHMARLIntegration(unittest.TestCase):
    """Testes de integração do sistema HMARL"""
    
    @classmethod
    def setUpClass(cls):
        """Setup do ambiente de teste"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Iniciar publisher de dados mock
        cls.publisher = MockDataPublisher()
        cls.publisher.start()
        time.sleep(1)  # Dar tempo para ZMQ inicializar
        
        # Iniciar registry
        cls.registry = AgentRegistry()
        time.sleep(0.5)
        
    @classmethod
    def tearDownClass(cls):
        """Limpeza após testes"""
        cls.publisher.close()
        cls.registry.shutdown()
        
    def test_01_flow_feature_extraction(self):
        """Testa extração de features de fluxo"""
        print("\n=== Teste 1: Extração de Features ===")
        
        flow_system = FlowFeatureSystem()
        
        # Extrair features
        features = flow_system.extract_comprehensive_features(
            symbol='WDOH25',
            timestamp=datetime.now()
        )
        
        # Validações
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 200, "Deve ter 200+ features")
        
        # Verificar categorias de features
        feature_categories = {
            'ofi': 0,
            'delta': 0,
            'tape': 0,
            'footprint': 0,
            'liquidity': 0,
            'microstructure': 0
        }
        
        for feature_name in features.keys():
            for category in feature_categories:
                if category in feature_name.lower():
                    feature_categories[category] += 1
                    
        print(f"Total de features: {len(features)}")
        print(f"Distribuição por categoria: {feature_categories}")
        
        # Cada categoria deve ter features
        for category, count in feature_categories.items():
            self.assertGreater(count, 0, f"Categoria {category} deve ter features")
            
    def test_02_agent_registration(self):
        """Testa registro de agentes"""
        print("\n=== Teste 2: Registro de Agentes ===")
        
        client = AgentRegistryClient()
        
        # Registrar agente teste
        test_agent = {
            'agent_id': 'test_agent_001',
            'agent_type': 'test_type',
            'capabilities': ['test_cap1', 'test_cap2'],
            'pub_address': 'tcp://localhost:9999',
            'sub_topics': ['test_topic']
        }
        
        result = client.register(test_agent)
        self.assertEqual(result['status'], 'registered')
        
        # Buscar agente
        agents = client.get_agents_by_type('test_type')
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0]['agent_id'], 'test_agent_001')
        
        # Heartbeat
        result = client.heartbeat('test_agent_001')
        self.assertEqual(result['status'], 'ok')
        
        # Limpar
        result = client.unregister('test_agent_001')
        self.assertEqual(result['status'], 'unregistered')
        
        client.close()
        
    def test_03_individual_agents(self):
        """Testa agentes individuais"""
        print("\n=== Teste 3: Agentes Individuais ===")
        
        # Configuração comum
        config = {
            'min_confidence': 0.3,
            'use_registry': False  # Desabilitar registry para teste isolado
        }
        
        # Estado mock
        price_state = {
            'price': 5000.0,
            'prev_price': 4995.0,
            'sma_20': 4980.0
        }
        
        flow_state = {
            'last_ofi': 0.5,
            'delta': 150,
            'volume': 300,
            'avg_volume': 200,
            'absorption': 0.7,
            'footprint_pattern': 'p_reversal',
            'orderbook': {
                'bids': [{'price': 4999, 'volume': 100}],
                'asks': [{'price': 5001, 'volume': 100}]
            },
            'recent_trades': [
                {'price': 5000, 'volume': 50, 'side': 'buy', 'timestamp': time.time()}
            ]
        }
        
        # Testar cada agente
        agents = [
            ('OrderFlowSpecialist', OrderFlowSpecialistAgent),
            ('FootprintPattern', FootprintPatternAgent),
            ('Liquidity', LiquidityAgent),
            ('TapeReading', TapeReadingAgent)
        ]
        
        for agent_name, AgentClass in agents:
            print(f"\nTestando {agent_name}...")
            
            agent = AgentClass(config)
            signal = agent.generate_signal_with_flow(price_state, flow_state)
            
            # Validações
            self.assertIsInstance(signal, dict)
            self.assertIn('action', signal)
            self.assertIn('confidence', signal)
            self.assertIn(signal['action'], ['buy', 'sell', 'hold'])
            self.assertGreaterEqual(signal['confidence'], 0)
            self.assertLessEqual(signal['confidence'], 1)
            
            print(f"  Sinal: {signal['action']} (conf: {signal['confidence']:.2f})")
            
            agent.shutdown()
            
    def test_04_feedback_system(self):
        """Testa sistema de feedback"""
        print("\n=== Teste 4: Sistema de Feedback ===")
        
        feedback_system = FlowAwareFeedbackSystem()
        
        # Criar decisão mock
        decision = {
            'decision_id': 'test_dec_001',
            'agent_id': 'test_agent',
            'symbol': 'WDOH25',
            'timestamp': datetime.now(),
            'signal': {
                'action': 'buy',
                'confidence': 0.75
            },
            'metadata': {
                'ofi_signal': True,
                'delta_signal': True,
                'pattern': 'p_reversal'
            }
        }
        
        # Cachear decisão
        feedback_system.cache_decision(decision)
        
        # Criar execução mock
        execution = {
            'decision_id': 'test_dec_001',
            'pnl': 0.015,
            'price_movement': 0.012,
            'profitable': True,
            'return': 0.015,
            'entry_slippage': 0.0002,
            'exit_type': 'target'
        }
        
        # Processar feedback
        feedback = feedback_system.process_execution_feedback_with_flow(execution)
        
        # Validações
        self.assertIsNotNone(feedback)
        self.assertIn('reward', feedback)
        self.assertIn('flow_reward_component', feedback)
        self.assertIn('performance_analysis', feedback)
        self.assertIn('learning_insights', feedback)
        
        print(f"  Reward total: {feedback['reward']:.2f}")
        print(f"  Componente flow: {feedback['flow_reward_component']:.2f}")
        print(f"  Insights: {feedback['learning_insights']['key_lessons']}")
        
    def test_05_coordinator_basic(self):
        """Testa coordenador básico"""
        print("\n=== Teste 5: Coordenador Básico ===")
        
        coordinator = FlowAwareCoordinator()
        
        # Simular sinais de agentes
        test_signals = [
            {
                'agent_id': 'flow_agent_001',
                'agent_type': 'order_flow',
                'signal': {
                    'action': 'buy',
                    'confidence': 0.7,
                    'timestamp': time.time()
                }
            },
            {
                'agent_id': 'footprint_agent_001',
                'agent_type': 'footprint',
                'signal': {
                    'action': 'buy',
                    'confidence': 0.8,
                    'timestamp': time.time()
                }
            },
            {
                'agent_id': 'tape_agent_001',
                'agent_type': 'tape_reading',
                'signal': {
                    'action': 'sell',
                    'confidence': 0.6,
                    'timestamp': time.time()
                }
            }
        ]
        
        # Adicionar sinais ao buffer do coordenador
        current_window = int(time.time() / coordinator.coordination_window)
        coordinator.signal_buffer[current_window] = test_signals
        
        # Coordenar
        decision = coordinator.coordinate_with_flow_analysis()
        
        # Validações
        self.assertIsNotNone(decision)
        self.assertIn('action', decision)
        self.assertIn('confidence', decision)
        self.assertIn('flow_consensus', decision)
        
        print(f"  Decisão coordenada: {decision['action']} (conf: {decision['confidence']:.2f})")
        print(f"  Consenso de fluxo: {decision['flow_consensus']['direction']}")
        
        # Estatísticas
        stats = coordinator.get_coordination_stats()
        print(f"  Estatísticas: {stats}")
        
        coordinator.shutdown()
        
    def test_06_end_to_end_flow(self):
        """Teste end-to-end do fluxo completo"""
        print("\n=== Teste 6: Fluxo End-to-End ===")
        
        # Configuração
        config = {
            'min_confidence': 0.3,
            'use_registry': True
        }
        
        # 1. Criar agentes
        agents = []
        agents.append(OrderFlowSpecialistAgent(config))
        agents.append(FootprintPatternAgent(config))
        agents.append(LiquidityAgent(config))
        agents.append(TapeReadingAgent(config))
        
        print(f"  {len(agents)} agentes criados")
        
        # 2. Aguardar registro
        time.sleep(1)
        
        # 3. Verificar registro
        client = AgentRegistryClient()
        registered_agents = client.get_all_agents()
        print(f"  {len(registered_agents)} agentes registrados")
        
        # 4. Criar coordenador
        coordinator = FlowAwareCoordinator()
        
        # 5. Criar sistema de feedback
        feedback_system = FlowAwareFeedbackSystem()
        
        # 6. Simular ciclo completo
        print("\n  Simulando ciclo de operação...")
        
        # Simular por 5 segundos
        start_time = time.time()
        signals_generated = 0
        decisions_made = 0
        
        while time.time() - start_time < 5:
            # Coletar sinais (simulado - normalmente viria via ZMQ)
            coordinator.collect_agent_signals(timeout=0.1)
            
            # Tentar coordenar
            if time.time() % coordinator.coordination_window < 0.1:
                decision = coordinator.coordinate_with_flow_analysis()
                if decision:
                    decisions_made += 1
                    print(f"    Decisão {decisions_made}: {decision['action']}")
                    
            time.sleep(0.1)
            
        print(f"\n  Decisões tomadas: {decisions_made}")
        
        # 7. Limpar
        for agent in agents:
            agent.shutdown()
            
        coordinator.shutdown()
        client.close()
        
        print("\n  Teste end-to-end concluído!")
        

def run_integration_tests():
    """Executa todos os testes de integração"""
    print("\n" + "="*60)
    print("TESTES DE INTEGRAÇÃO HMARL - FASE 1")
    print("="*60)
    
    # Criar suite de testes
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHMARLIntegration)
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumo
    print("\n" + "="*60)
    print("RESUMO DOS TESTES")
    print("="*60)
    print(f"Testes executados: {result.testsRun}")
    print(f"Sucessos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Falhas: {len(result.failures)}")
    print(f"Erros: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ TODOS OS TESTES PASSARAM!")
    else:
        print("\n❌ ALGUNS TESTES FALHARAM")
        
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)