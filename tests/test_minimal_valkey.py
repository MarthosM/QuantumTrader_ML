"""
Teste mínimo para identificar o problema
"""

from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infrastructure.zmq_valkey_flow_setup import TradingInfrastructureWithFlow, HMARL_FLOW_CONFIG

def test_minimal():
    """Teste mínimo para debugar"""
    
    print("1. Criando infraestrutura...")
    config = HMARL_FLOW_CONFIG.copy()
    config['symbol'] = 'DEBUG_TEST'
    
    infra = TradingInfrastructureWithFlow(config)
    
    print("\n2. Inicializando...")
    try:
        result = infra.initialize()
        print(f"   Inicialização: {'OK' if result else 'FALHOU'}")
        
        if not result:
            print("   Falha na inicialização!")
            return
    except Exception as e:
        print(f"   ERRO na inicialização: {e}")
        return
    
    print("\n3. Iniciando...")
    infra.start()
    
    print("\n4. Criando tick de teste...")
    tick_data = {
        'symbol': 'DEBUG_TEST',
        'timestamp': datetime.now().isoformat(),
        'price': 100.0,
        'volume': 10,
        'trade_type': 2
    }
    print(f"   Tick: {tick_data}")
    
    print("\n5. Publicando tick...")
    try:
        infra.publish_tick_with_flow(tick_data)
        print("   [OK] Tick publicado!")
    except Exception as e:
        print(f"   [ERRO] {e}")
        import traceback
        traceback.print_exc()
    
    print("\n6. Verificando métricas...")
    metrics = infra.get_performance_metrics()
    print(f"   Mensagens publicadas: {metrics['messages_published']}")
    print(f"   Mensagens armazenadas: {metrics['messages_stored']}")
    print(f"   Eventos detectados: {metrics['flow_events_detected']}")
    
    print("\n7. Parando infraestrutura...")
    infra.stop()
    print("   [OK] Parado")

if __name__ == "__main__":
    test_minimal()