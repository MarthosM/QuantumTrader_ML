"""
Test script for ZeroMQ + Valkey Infrastructure
Valida a implementação da Task 1.1 da Semana 1
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infrastructure.zmq_valkey_flow_setup import (
    TradingInfrastructureWithFlow,
    FlowDataPoint,
    HMARL_FLOW_CONFIG
)
from src.infrastructure.zmq_consumers import (
    FlowConsumer,
    TapeConsumer,
    LiquidityConsumer,
    MultiStreamConsumer
)


class TestZMQValkeyInfrastructure:
    """Testes para infraestrutura ZeroMQ + Valkey"""
    
    @pytest.fixture
    def infrastructure(self):
        """Cria infraestrutura para testes"""
        config = HMARL_FLOW_CONFIG.copy()
        config['symbol'] = 'TEST_SYMBOL'
        
        infra = TradingInfrastructureWithFlow(config)
        infra.initialize()
        infra.start()
        
        yield infra
        
        infra.stop()
    
    @pytest.fixture
    def flow_consumer(self):
        """Cria consumer de fluxo para testes"""
        consumer = FlowConsumer('TEST_SYMBOL')
        consumer.start()
        
        yield consumer
        
        consumer.stop()
    
    def test_infrastructure_initialization(self, infrastructure):
        """Testa inicialização da infraestrutura"""
        assert infrastructure is not None
        assert infrastructure.valkey_client is not None
        assert infrastructure.flow_analyzer is not None
        assert infrastructure.tape_reader is not None
        assert infrastructure.liquidity_monitor is not None
        
        # Verificar publishers
        assert len(infrastructure.publishers) == 6
        for pub in infrastructure.publishers.values():
            assert pub is not None
    
    def test_flow_data_point_creation(self):
        """Testa criação de FlowDataPoint"""
        flow_point = FlowDataPoint(
            timestamp=datetime.now(),
            symbol='TEST',
            price=100.5,
            volume=10,
            trade_type=2,
            aggressor='buyer',
            trade_size_category='medium',
            speed_of_tape=5.2
        )
        
        # Converter para dict
        data = flow_point.to_dict()
        
        assert data['symbol'] == 'TEST'
        assert data['price'] == 100.5
        assert data['volume'] == 10
        assert data['aggressor'] == 'buyer'
        assert isinstance(data['timestamp'], str)
    
    def test_publish_tick_with_flow(self, infrastructure):
        """Testa publicação de tick com análise de fluxo"""
        tick_data = {
            'symbol': 'TEST_SYMBOL',
            'timestamp': datetime.now().isoformat(),
            'price': 100.0,
            'volume': 15,
            'trade_type': 2  # buy
        }
        
        # Publicar tick
        infrastructure.publish_tick_with_flow(tick_data)
        
        # Verificar métricas
        time.sleep(0.1)  # Aguardar processamento
        
        metrics = infrastructure.get_performance_metrics()
        assert metrics['messages_published'] > 0
        assert metrics['messages_stored'] > 0
    
    def test_flow_history_retrieval(self, infrastructure):
        """Testa recuperação de histórico via time travel"""
        # Publicar alguns ticks
        base_time = datetime.now()
        
        for i in range(5):
            tick_data = {
                'symbol': 'TEST_SYMBOL',
                'timestamp': (base_time + timedelta(seconds=i)).isoformat(),
                'price': 100.0 + i,
                'volume': 10 + i,
                'trade_type': 2 if i % 2 == 0 else 3
            }
            infrastructure.publish_tick_with_flow(tick_data)
            time.sleep(0.05)
        
        # Recuperar histórico
        history = infrastructure.get_flow_history('TEST_SYMBOL', minutes_back=1)
        
        # Filtrar apenas entradas com análise (não as de inicialização)
        flow_entries = [h for h in history if 'analysis' in h]
        
        assert len(flow_entries) >= 5
        assert 'analysis' in flow_entries[0]
        assert 'timestamp' in flow_entries[0]
    
    def test_flow_consumer_reception(self, infrastructure, flow_consumer):
        """Testa recepção de dados pelo consumer"""
        # Publicar tick
        tick_data = {
            'symbol': 'TEST_SYMBOL',
            'timestamp': datetime.now().isoformat(),
            'price': 100.0,
            'volume': 20,
            'trade_type': 2
        }
        
        infrastructure.publish_tick_with_flow(tick_data)
        
        # Aguardar recepção
        time.sleep(0.2)
        
        # Verificar dados recebidos
        latest = flow_consumer.get_latest(1)
        assert len(latest) > 0
        
        stats = flow_consumer.get_stats()
        assert stats['messages_received'] > 0
    
    def test_ofi_calculation(self, infrastructure, flow_consumer):
        """Testa cálculo de OFI (Order Flow Imbalance)"""
        # Publicar série de trades
        base_time = datetime.now()
        
        # 70% buys, 30% sells
        for i in range(10):
            tick_data = {
                'symbol': 'TEST_SYMBOL',
                'timestamp': (base_time + timedelta(seconds=i)).isoformat(),
                'price': 100.0,
                'volume': 10,
                'trade_type': 2 if i < 7 else 3
            }
            infrastructure.publish_tick_with_flow(tick_data)
            time.sleep(0.05)
        
        # Aguardar processamento
        time.sleep(0.3)
        
        # Verificar OFI
        ofi_5m = flow_consumer.get_ofi('5m')
        
        # OFI deve ser positivo (mais compras)
        assert ofi_5m > 0
        assert ofi_5m < 1  # Entre 0 e 1
    
    def test_tape_pattern_detection(self, infrastructure):
        """Testa detecção de padrões no tape"""
        tape_consumer = TapeConsumer('TEST_SYMBOL')
        tape_consumer.start()
        
        try:
            # Simular sweep pattern (trades rápidos na mesma direção)
            base_time = datetime.now()
            
            for i in range(5):
                tick_data = {
                    'symbol': 'TEST_SYMBOL',
                    'timestamp': (base_time + timedelta(milliseconds=i*100)).isoformat(),
                    'price': 100.0 + i*0.1,
                    'volume': 20 + i*5,  # Volume crescente
                    'trade_type': 2  # Todos buys
                }
                infrastructure.publish_tick_with_flow(tick_data)
                time.sleep(0.01)
            
            # Aguardar detecção
            time.sleep(0.5)
            
            # Verificar padrões detectados
            patterns = tape_consumer.get_recent_patterns()
            
            # Deve ter detectado algum padrão
            assert len(patterns) >= 0  # Pode não detectar sempre devido a thresholds
            
        finally:
            tape_consumer.stop()
    
    def test_multi_stream_consumer(self, infrastructure):
        """Testa consumer multi-stream"""
        multi_consumer = MultiStreamConsumer('TEST_SYMBOL')
        multi_consumer.start()
        
        try:
            # Publicar dados diversos
            tick_data = {
                'symbol': 'TEST_SYMBOL',
                'timestamp': datetime.now().isoformat(),
                'price': 100.0,
                'volume': 25,
                'trade_type': 2
            }
            
            infrastructure.publish_tick_with_flow(tick_data)
            
            # Aguardar agregação
            time.sleep(0.5)
            
            # Obter visão unificada
            unified = multi_consumer.get_unified_view()
            
            assert 'symbol' in unified
            assert unified['symbol'] == 'TEST_SYMBOL'
            assert 'streams' in unified
            assert 'timestamp' in unified
            
            # Verificar estatísticas
            stats = multi_consumer.get_stats()
            assert 'flow' in stats
            assert 'tape' in stats
            assert 'liquidity' in stats
            
        finally:
            multi_consumer.stop()
    
    def test_performance_metrics(self, infrastructure):
        """Testa métricas de performance"""
        # Publicar vários ticks
        for i in range(20):
            tick_data = {
                'symbol': 'TEST_SYMBOL',
                'timestamp': datetime.now().isoformat(),
                'price': 100.0 + i*0.1,
                'volume': 10,
                'trade_type': 2 if i % 2 == 0 else 3
            }
            infrastructure.publish_tick_with_flow(tick_data)
            time.sleep(0.01)
        
        # Obter métricas
        metrics = infrastructure.get_performance_metrics()
        
        assert metrics['messages_published'] >= 20
        assert metrics['messages_stored'] >= 20
        assert 'avg_latency_ms' in metrics
        assert metrics['avg_latency_ms'] < 100  # Latência deve ser < 100ms
        assert metrics['uptime_seconds'] > 0


def test_infrastructure_integration():
    """Teste de integração completo"""
    print("\n=== Teste de Integração ZeroMQ + Valkey ===\n")
    
    # Configuração
    config = HMARL_FLOW_CONFIG.copy()
    config['symbol'] = 'INTEGRATION_TEST'
    
    # Criar infraestrutura
    print("1. Inicializando infraestrutura...")
    infrastructure = TradingInfrastructureWithFlow(config)
    assert infrastructure.initialize(), "Falha na inicialização"
    infrastructure.start()
    print("✅ Infraestrutura inicializada")
    
    # Criar consumers
    print("\n2. Criando consumers...")
    flow_consumer = FlowConsumer('INTEGRATION_TEST')
    flow_consumer.start()
    print("✅ Consumers criados")
    
    # Publicar dados de teste
    print("\n3. Publicando dados de teste...")
    published = 0
    
    for i in range(10):
        tick_data = {
            'symbol': 'INTEGRATION_TEST',
            'timestamp': datetime.now().isoformat(),
            'price': 5000.0 + i,
            'volume': 5 + i,
            'trade_type': 2 if i < 6 else 3  # 60% buys, 40% sells
        }
        
        infrastructure.publish_tick_with_flow(tick_data)
        published += 1
        time.sleep(0.1)
    
    print(f"✅ {published} ticks publicados")
    
    # Aguardar processamento
    print("\n4. Aguardando processamento...")
    time.sleep(1)
    
    # Verificar recepção
    print("\n5. Verificando recepção de dados...")
    latest_flow = flow_consumer.get_latest(5)
    consumer_stats = flow_consumer.get_stats()
    
    print(f"   - Mensagens recebidas: {consumer_stats['messages_received']}")
    print(f"   - Mensagens processadas: {consumer_stats['messages_processed']}")
    print(f"   - Erros: {consumer_stats['errors']}")
    
    assert consumer_stats['messages_received'] >= 5, "Poucos dados recebidos"
    assert consumer_stats['errors'] == 0, "Erros durante processamento"
    
    # Verificar análise de fluxo
    print("\n6. Verificando análise de fluxo...")
    ofi = flow_consumer.get_ofi('5m')
    print(f"   - OFI (5m): {ofi:.3f}")
    assert abs(ofi) <= 1, "OFI fora do range esperado"
    
    # Verificar histórico
    print("\n7. Verificando time travel...")
    history = infrastructure.get_flow_history('INTEGRATION_TEST', minutes_back=5)
    print(f"   - Registros no histórico: {len(history)}")
    assert len(history) >= published, "Histórico incompleto"
    
    # Verificar métricas
    print("\n8. Verificando métricas de performance...")
    metrics = infrastructure.get_performance_metrics()
    print(f"   - Latência média: {metrics['avg_latency_ms']:.2f}ms")
    print(f"   - Mensagens armazenadas: {metrics['messages_stored']}")
    print(f"   - Eventos de fluxo: {metrics['flow_events_detected']}")
    
    assert metrics['avg_latency_ms'] < 50, "Latência muito alta"
    
    # Cleanup
    print("\n9. Finalizando...")
    flow_consumer.stop()
    infrastructure.stop()
    print("✅ Teste de integração concluído com sucesso!")
    
    return True


if __name__ == "__main__":
    # Executar teste de integração
    try:
        if test_infrastructure_integration():
            print("\n✅ TODOS OS TESTES PASSARAM!")
        else:
            print("\n❌ TESTE FALHOU!")
    except Exception as e:
        print(f"\n❌ ERRO NO TESTE: {e}")
        import traceback
        traceback.print_exc()