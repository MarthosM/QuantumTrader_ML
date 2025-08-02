"""
Teste da infraestrutura HMARL com mock do Valkey
Permite testar sem ter o Valkey rodando
"""

import sys
import os
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infrastructure.zmq_valkey_flow_setup import (
    TradingInfrastructureWithFlow,
    FlowDataPoint,
    FlowAnalysisEngine,
    AutomatedTapeReader,
    LiquidityMonitor,
    HMARL_FLOW_CONFIG
)
from src.infrastructure.zmq_consumers import FlowConsumer


class MockValkeyClient:
    """Mock do cliente Valkey para testes"""
    
    def __init__(self, *args, **kwargs):
        self.data = {}
        self.streams = {}
    
    def ping(self):
        return True
    
    def xadd(self, stream, data, id=None, maxlen=None):
        if stream not in self.streams:
            self.streams[stream] = []
        
        entry_id = id or f"{int(time.time() * 1000)}-0"
        
        # Converter dict para formato compatível com Valkey (bytes)
        if isinstance(data, dict):
            converted_data = {}
            for k, v in data.items():
                key = k.encode() if isinstance(k, str) else k
                val = str(v).encode() if not isinstance(v, bytes) else v
                converted_data[key] = val
            data = converted_data
        
        self.streams[stream].append((entry_id, data))
        
        if maxlen and len(self.streams[stream]) > maxlen:
            self.streams[stream] = self.streams[stream][-maxlen:]
        
        return entry_id
    
    def xrange(self, stream, start, end):
        if stream not in self.streams:
            return []
        
        return [(eid, data) for eid, data in self.streams[stream]
                if start <= eid <= end]
    
    def close(self):
        pass


def test_flow_analysis_engine():
    """Testa o motor de análise de fluxo"""
    print("\n=== Teste: Flow Analysis Engine ===")
    
    engine = FlowAnalysisEngine(HMARL_FLOW_CONFIG)
    
    # Criar pontos de fluxo
    for i in range(10):
        flow_point = FlowDataPoint(
            timestamp=datetime.now() - timedelta(seconds=10-i),
            symbol='TEST',
            price=100.0 + i*0.1,
            volume=10 + i,
            trade_type=2 if i < 7 else 3,  # 70% compras
            aggressor='buyer' if i < 7 else 'seller',
            trade_size_category='medium',
            speed_of_tape=5.0
        )
        
        analysis = engine.analyze(flow_point)
    
    # Verificar análise
    assert 'ofi' in analysis
    assert 'volume_imbalance' in analysis
    assert 'aggression_ratio' in analysis
    
    # OFI deve ser positivo (mais compras)
    ofi_1m = analysis['ofi'].get(1, 0)
    print(f"OFI (1m): {ofi_1m:.3f}")
    assert ofi_1m > 0
    
    print("[OK] Flow Analysis Engine funcionando!")
    return True


def test_tape_reader():
    """Testa o leitor de tape"""
    print("\n=== Teste: Tape Reader ===")
    
    reader = AutomatedTapeReader(HMARL_FLOW_CONFIG)
    
    # Simular sweep pattern
    for i in range(5):
        tick = {
            'symbol': 'TEST',
            'timestamp': datetime.now().isoformat(),
            'price': 100.0 + i*0.1,
            'volume': 20 + i*5,  # Volume crescente
            'trade_type': 2  # Todos buys
        }
        
        pattern = reader.analyze_tick(tick)
        time.sleep(0.01)
    
    # Verificar velocidade
    speed = reader.get_current_speed()
    print(f"Velocidade do tape: {speed:.2f} trades/segundo")
    
    # Verificar buffer
    assert len(reader.trade_buffer) == 5
    
    print("[OK] Tape Reader funcionando!")
    return True


def test_liquidity_monitor():
    """Testa o monitor de liquidez"""
    print("\n=== Teste: Liquidity Monitor ===")
    
    monitor = LiquidityMonitor(HMARL_FLOW_CONFIG)
    
    # Simular book
    book_data = {
        'bids': {
            99.5: 100,
            99.0: 200,
            98.5: 150
        },
        'asks': {
            100.5: 120,
            101.0: 180,
            101.5: 140
        }
    }
    
    monitor.update_book(book_data)
    
    # Verificar métricas
    assert len(monitor.liquidity_history) == 1
    
    metrics = monitor.liquidity_history[0]['metrics']
    assert metrics['bid_depth'] == 450  # 100+200+150
    assert metrics['ask_depth'] == 440  # 120+180+140
    assert metrics['spread'] == 1.0     # 100.5-99.5
    
    print(f"Liquidez total: {metrics['bid_depth'] + metrics['ask_depth']}")
    print(f"Spread: {metrics['spread']}")
    print(f"Score de liquidez: {metrics['liquidity_score']:.2f}")
    
    print("[OK] Liquidity Monitor funcionando!")
    return True


@patch('valkey.Valkey', MockValkeyClient)
def test_infrastructure_with_mock():
    """Testa infraestrutura completa com mock"""
    print("\n=== Teste: Infraestrutura com Mock ===")
    
    config = HMARL_FLOW_CONFIG.copy()
    config['symbol'] = 'MOCK_TEST'
    
    # Criar infraestrutura
    infrastructure = TradingInfrastructureWithFlow(config)
    
    # Inicializar
    assert infrastructure.initialize()
    infrastructure.start()
    
    # Publicar ticks
    for i in range(5):
        tick_data = {
            'symbol': 'MOCK_TEST',
            'timestamp': datetime.now().isoformat(),
            'price': 100.0 + i,
            'volume': 10 + i,
            'trade_type': 2 if i % 2 == 0 else 3
        }
        
        infrastructure.publish_tick_with_flow(tick_data)
        time.sleep(0.05)
    
    # Verificar métricas
    metrics = infrastructure.get_performance_metrics()
    assert metrics['messages_published'] >= 5
    assert metrics['messages_stored'] >= 5
    
    print(f"Mensagens publicadas: {metrics['messages_published']}")
    print(f"Mensagens armazenadas: {metrics['messages_stored']}")
    print(f"Latência média: {metrics['avg_latency_ms']:.2f}ms")
    
    # Verificar histórico
    history = infrastructure.get_flow_history('MOCK_TEST', minutes_back=1)
    assert len(history) >= 5
    
    print(f"Registros no histórico: {len(history)}")
    
    infrastructure.stop()
    print("[OK] Infraestrutura funcionando com mock!")
    return True


def test_zmq_publishers():
    """Testa apenas os publishers ZeroMQ (sem Valkey)"""
    print("\n=== Teste: ZeroMQ Publishers ===")
    
    import zmq
    
    context = zmq.Context()
    
    # Criar publisher
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5999")  # Porta de teste
    
    # Criar subscriber
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://localhost:5999")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "test")
    subscriber.setsockopt(zmq.RCVTIMEO, 1000)  # Timeout 1s
    
    time.sleep(0.1)  # Aguardar conexão
    
    # Publicar mensagem
    test_data = {'data': 'test', 'value': 123}
    import orjson
    publisher.send_multipart([
        b"test_topic",
        orjson.dumps(test_data)
    ])
    
    # Receber mensagem
    try:
        topic, message = subscriber.recv_multipart()
        data = orjson.loads(message)
        
        assert topic == b"test_topic"
        assert data['value'] == 123
        
        print("[OK] ZeroMQ funcionando!")
        
    except zmq.Again:
        print("[ERRO] Timeout ao receber mensagem ZMQ")
        return False
    
    finally:
        publisher.close()
        subscriber.close()
        context.term()
    
    return True


def test_flow_data_structures():
    """Testa estruturas de dados sem dependências externas"""
    print("\n=== Teste: Estruturas de Dados ===")
    
    # FlowDataPoint
    flow_point = FlowDataPoint(
        timestamp=datetime.now(),
        symbol='TEST',
        price=100.5,
        volume=50,
        trade_type=2,
        aggressor='buyer',
        trade_size_category='large',
        speed_of_tape=8.5
    )
    
    # Converter para dict
    data = flow_point.to_dict()
    
    assert data['symbol'] == 'TEST'
    assert data['price'] == 100.5
    assert data['volume'] == 50
    assert data['trade_size_category'] == 'large'
    assert isinstance(data['timestamp'], str)
    
    print("[OK] Estruturas de dados funcionando!")
    return True


def run_all_tests():
    """Executa todos os testes"""
    print("\n" + "="*50)
    print("   TESTES DA INFRAESTRUTURA HMARL")
    print("="*50)
    
    tests = [
        ("Estruturas de Dados", test_flow_data_structures),
        ("Flow Analysis Engine", test_flow_analysis_engine),
        ("Tape Reader", test_tape_reader),
        ("Liquidity Monitor", test_liquidity_monitor),
        ("ZeroMQ Publishers", test_zmq_publishers),
        ("Infraestrutura com Mock", test_infrastructure_with_mock)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"[ERRO] {name} falhou")
        except Exception as e:
            failed += 1
            print(f"[ERRO] {name} erro: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*50)
    print(f"RESULTADO: {passed} passou, {failed} falhou")
    print("="*50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n[OK] TODOS OS TESTES PASSARAM!")
    else:
        print("\n[ERRO] ALGUNS TESTES FALHARAM!")
    
    # Informar sobre Valkey
    print("\nNOTA: Para testes completos com Valkey:")
    print("1. Instale Docker: https://www.docker.com/")
    print("2. Execute: docker run -d -p 6379:6379 --name valkey valkey/valkey:latest")
    print("3. Execute novamente: pytest tests/test_zmq_valkey_infrastructure.py")