"""
Teste de Integração com Valkey
"""

import unittest
import time
import sys
import os
from datetime import datetime
import logging

# Adicionar diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from infrastructure.valkey_connection import ValkeyConnectionManager
from systems.flow_aware_feedback_system import FlowAwareFeedbackSystem
from coordination.flow_aware_coordinator import FlowAwareCoordinator


class TestValkeyIntegration(unittest.TestCase):
    """Testes de integração com Valkey"""
    
    @classmethod
    def setUpClass(cls):
        """Setup do ambiente de teste"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Configuração do Valkey
        cls.valkey_config = {
            'host': 'localhost',
            'port': 6379,
            'db': 1  # Use DB 1 para testes
        }
        
    def test_01_valkey_connection(self):
        """Testa conexão básica com Valkey"""
        print("\n=== Teste 1: Conexão com Valkey ===")
        
        valkey = ValkeyConnectionManager(**self.valkey_config)
        
        # Tentar conectar
        connected = valkey.connect()
        
        if not connected:
            self.skipTest("Valkey não está disponível - pulando testes")
            
        self.assertTrue(connected, "Deve conectar com sucesso")
        
        # Health check
        health = valkey.health_check()
        self.assertTrue(health['connected'])
        print(f"  Ping: {health.get('ping_ms', 'N/A')}ms")
        print(f"  Memória: {health.get('used_memory_mb', 'N/A')}MB")
        
        valkey.disconnect()
        
    def test_02_decision_persistence(self):
        """Testa persistência de decisões"""
        print("\n=== Teste 2: Persistência de Decisões ===")
        
        valkey = ValkeyConnectionManager(**self.valkey_config)
        if not valkey.connect():
            self.skipTest("Valkey não disponível")
            
        # Criar decisão teste
        decision = {
            'decision_id': f'test_decision_{int(time.time())}',
            'agent_id': 'test_agent',
            'action': 'buy',
            'confidence': 0.75,
            'metadata': {
                'test': True,
                'timestamp': time.time()
            }
        }
        
        # Armazenar
        stored = valkey.store_decision(decision)
        self.assertTrue(stored)
        
        # Recuperar
        retrieved = valkey.get_decision(decision['decision_id'])
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['action'], 'buy')
        self.assertEqual(retrieved['confidence'], 0.75)
        
        print(f"  Decisão armazenada e recuperada com sucesso")
        
        # Buscar decisões recentes
        recent = valkey.get_recent_decisions(limit=10)
        self.assertIsInstance(recent, list)
        print(f"  {len(recent)} decisões recentes encontradas")
        
        valkey.disconnect()
        
    def test_03_feedback_system_integration(self):
        """Testa integração do sistema de feedback com Valkey"""
        print("\n=== Teste 3: Sistema de Feedback + Valkey ===")
        
        # Criar sistema de feedback com Valkey
        feedback_system = FlowAwareFeedbackSystem(self.valkey_config)
        
        # Verificar se conectou
        stats = feedback_system.get_feedback_statistics()
        
        if not stats.get('valkey_connected', False):
            self.skipTest("Valkey não disponível")
            
        print(f"  Valkey conectado: {stats['valkey_connected']}")
        
        # Criar e cachear decisão
        decision = {
            'decision_id': f'feedback_test_{int(time.time())}',
            'agent_id': 'feedback_test_agent',
            'symbol': 'WDOH25',
            'timestamp': datetime.now(),
            'signal': {
                'action': 'buy',
                'confidence': 0.8
            },
            'metadata': {
                'pattern': 'test_pattern'
            }
        }
        
        feedback_system.cache_decision(decision)
        
        # Criar execução
        execution = {
            'decision_id': decision['decision_id'],
            'pnl': 0.02,
            'profitable': True,
            'return': 0.02,
            'price_movement': 0.015
        }
        
        # Processar feedback
        feedback = feedback_system.process_execution_feedback_with_flow(execution)
        
        self.assertIsNotNone(feedback)
        self.assertIn('reward', feedback)
        print(f"  Feedback processado: reward={feedback['reward']:.2f}")
        
        # Verificar persistência
        stats = feedback_system.get_feedback_statistics()
        print(f"  Decisões armazenadas: {stats.get('stored_decisions', 0)}")
        print(f"  Feedbacks armazenados: {stats.get('stored_feedback', 0)}")
        
    def test_04_coordinator_integration(self):
        """Testa integração do coordenador com Valkey"""
        print("\n=== Teste 4: Coordenador + Valkey ===")
        
        # Criar coordenador com Valkey
        coordinator = FlowAwareCoordinator(self.valkey_config)
        
        # Verificar se Valkey está conectado
        if not coordinator.valkey or not coordinator.valkey.is_connected():
            self.skipTest("Valkey não disponível")
            
        # Simular decisão
        test_decision = {
            'decision_id': f'coord_test_{int(time.time())}',
            'selected_agent': 'test_agent',
            'action': 'buy',
            'confidence': 0.7,
            'flow_consensus': {
                'direction': 'bullish',
                'strength': 0.8,
                'confidence': 0.75
            },
            'metadata': {
                'symbol': 'WDOH25'
            }
        }
        
        # Persistir decisão
        coordinator._persist_decision(test_decision)
        
        # Recuperar estatísticas do Valkey
        stats = coordinator.get_coordination_stats_from_valkey()
        
        self.assertIn('recent_decisions', stats)
        self.assertIn('valkey_health', stats)
        
        print(f"  Decisões recentes: {len(stats.get('recent_decisions', []))}")
        print(f"  Health Valkey: {stats.get('valkey_health', {}).get('connected', False)}")
        
        coordinator.shutdown()
        
    def test_05_flow_state_persistence(self):
        """Testa persistência de estado de fluxo"""
        print("\n=== Teste 5: Persistência de Flow State ===")
        
        valkey = ValkeyConnectionManager(**self.valkey_config)
        if not valkey.connect():
            self.skipTest("Valkey não disponível")
            
        # Criar flow state
        flow_state = {
            'dominant_flow_direction': 'bullish',
            'flow_strength': 0.75,
            'ofi': 0.45,
            'delta': 250,
            'aggression': 0.8,
            'volume_profile': {
                'poc': 5000,
                'value_area_high': 5010,
                'value_area_low': 4990
            }
        }
        
        symbol = 'WDOH25'
        
        # Armazenar
        stored = valkey.store_flow_state(symbol, flow_state)
        self.assertTrue(stored)
        
        # Recuperar estado atual
        retrieved = valkey.get_flow_state(symbol)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['dominant_flow_direction'], 'bullish')
        self.assertEqual(retrieved['flow_strength'], 0.75)
        
        print(f"  Flow state armazenado e recuperado")
        
        # Recuperar por timestamp (histórico)
        time.sleep(0.1)
        historical = valkey.get_flow_state(symbol, time.time())
        self.assertIsNotNone(historical)
        
        print(f"  Flow state histórico recuperado")
        
        valkey.disconnect()
        
    def test_06_pattern_storage(self):
        """Testa armazenamento de padrões"""
        print("\n=== Teste 6: Armazenamento de Padrões ===")
        
        valkey = ValkeyConnectionManager(**self.valkey_config)
        if not valkey.connect():
            self.skipTest("Valkey não disponível")
            
        # Criar padrão
        pattern = {
            'type': 'reversal',
            'name': 'p_reversal_test',
            'confidence': 0.85,
            'features': {
                'absorption': 0.9,
                'delta': 300,
                'volume_spike': True
            },
            'outcome': {
                'success_rate': 0.7,
                'avg_return': 0.015
            }
        }
        
        # Armazenar
        stored = valkey.store_pattern(pattern)
        self.assertTrue(stored)
        self.assertIn('pattern_id', pattern)  # ID deve ser adicionado
        
        print(f"  Padrão armazenado: {pattern['pattern_id']}")
        
        # Buscar padrões por tipo
        reversal_patterns = valkey.get_patterns_by_type('reversal')
        self.assertIsInstance(reversal_patterns, list)
        self.assertGreater(len(reversal_patterns), 0)
        
        print(f"  {len(reversal_patterns)} padrões 'reversal' encontrados")
        
        valkey.disconnect()
        
    def test_07_performance_metrics(self):
        """Testa métricas de performance"""
        print("\n=== Teste 7: Métricas de Performance ===")
        
        valkey = ValkeyConnectionManager(**self.valkey_config)
        if not valkey.connect():
            self.skipTest("Valkey não disponível")
            
        # Simular múltiplos feedbacks para um agente
        agent_id = 'perf_test_agent'
        
        for i in range(5):
            feedback = {
                'decision_id': f'perf_test_{i}',
                'agent_id': agent_id,
                'profitable': i % 2 == 0,  # 3 sucessos, 2 falhas
                'reward': 0.01 if i % 2 == 0 else -0.005
            }
            
            valkey.store_feedback(feedback)
            
        # Recuperar performance
        performance = valkey.get_agent_performance(agent_id)
        
        self.assertIsNotNone(performance)
        self.assertEqual(performance['total_decisions'], 5)
        self.assertEqual(performance['successful_decisions'], 3)
        self.assertEqual(performance['success_rate'], 0.6)
        
        print(f"  Performance do agente:")
        print(f"    Total: {performance['total_decisions']}")
        print(f"    Sucessos: {performance['successful_decisions']}")
        print(f"    Taxa: {performance['success_rate']:.1%}")
        
        valkey.disconnect()
        
    def test_08_cleanup(self):
        """Testa limpeza de dados antigos"""
        print("\n=== Teste 8: Limpeza de Dados ===")
        
        valkey = ValkeyConnectionManager(**self.valkey_config)
        if not valkey.connect():
            self.skipTest("Valkey não disponível")
            
        # Executar limpeza (dados com mais de 30 dias)
        deleted = valkey.cleanup_old_data(days=30)
        
        print(f"  {deleted} registros antigos removidos")
        
        # Verificar keyspace
        health = valkey.health_check()
        keyspace = health.get('keyspace', {})
        
        print(f"  Keyspace atual:")
        for key_type, count in keyspace.items():
            print(f"    {key_type}: {count}")
            
        valkey.disconnect()


def run_valkey_tests():
    """Executa testes de integração com Valkey"""
    print("\n" + "="*60)
    print("TESTES DE INTEGRAÇÃO COM VALKEY")
    print("="*60)
    
    # Criar suite de testes
    suite = unittest.TestLoader().loadTestsFromTestCase(TestValkeyIntegration)
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumo
    print("\n" + "="*60)
    print("RESUMO DOS TESTES VALKEY")
    print("="*60)
    print(f"Testes executados: {result.testsRun}")
    print(f"Sucessos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Falhas: {len(result.failures)}")
    print(f"Erros: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ INTEGRAÇÃO COM VALKEY FUNCIONANDO!")
    else:
        print("\n❌ PROBLEMAS NA INTEGRAÇÃO COM VALKEY")
        
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_valkey_tests()
    sys.exit(0 if success else 1)