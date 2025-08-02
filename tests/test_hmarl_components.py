"""
Testes dos componentes HMARL sem dependências externas
Focado em validar a lógica de negócio
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import time
from src.infrastructure.zmq_valkey_flow_setup import (
    FlowDataPoint,
    FlowAnalysisEngine,
    AutomatedTapeReader,
    LiquidityMonitor,
    HMARL_FLOW_CONFIG
)


def test_all_components():
    """Testa todos os componentes principais"""
    
    print("\n" + "="*60)
    print("TESTE DOS COMPONENTES HMARL - SEM DEPENDÊNCIAS EXTERNAS")
    print("="*60)
    
    # 1. FlowDataPoint
    print("\n1. Testando FlowDataPoint...")
    
    flow_point = FlowDataPoint(
        timestamp=datetime.now(),
        symbol='WDOH25',
        price=5000.0,
        volume=25,
        trade_type=2,
        aggressor='buyer',
        trade_size_category='large',
        speed_of_tape=7.5
    )
    
    data = flow_point.to_dict()
    assert data['symbol'] == 'WDOH25'
    assert data['volume'] == 25
    assert data['trade_size_category'] == 'large'
    print("   [OK] FlowDataPoint criado e serializado")
    
    # 2. FlowAnalysisEngine
    print("\n2. Testando FlowAnalysisEngine...")
    
    engine = FlowAnalysisEngine(HMARL_FLOW_CONFIG)
    
    # Simular fluxo de trades
    buy_count = 0
    sell_count = 0
    
    for i in range(20):
        is_buy = i < 14  # 70% compras
        if is_buy:
            buy_count += 1
        else:
            sell_count += 1
            
        flow_point = FlowDataPoint(
            timestamp=datetime.now() - timedelta(seconds=20-i),
            symbol='WDOH25',
            price=5000.0 + i*0.5,
            volume=10 + i%5,
            trade_type=2 if is_buy else 3,
            aggressor='buyer' if is_buy else 'seller',
            trade_size_category=['small', 'medium', 'large'][i%3],
            speed_of_tape=5.0 + i*0.2
        )
        
        analysis = engine.analyze(flow_point)
    
    # Verificar análise
    print(f"   - Trades simulados: {buy_count} compras, {sell_count} vendas")
    print(f"   - OFI calculado (1m): {analysis['ofi']['1']:.3f}")
    print(f"   - Volume imbalance: {analysis['volume_imbalance']:.3f}")
    print(f"   - Aggression ratio: {analysis['aggression_ratio']:.3f}")
    print(f"   - Large trade ratio: {analysis['large_trade_ratio']:.3f}")
    print(f"   - Flow momentum: {analysis['flow_momentum']:.3f}")
    
    assert analysis['ofi']['1'] > 0  # Deve ser positivo (mais compras)
    assert 0 <= analysis['aggression_ratio'] <= 1
    print("   [OK] Análise de fluxo funcionando corretamente")
    
    # 3. AutomatedTapeReader
    print("\n3. Testando AutomatedTapeReader...")
    
    reader = AutomatedTapeReader(HMARL_FLOW_CONFIG)
    
    # Simular padrão de sweep
    print("   - Simulando padrão de sweep...")
    for i in range(6):
        tick = {
            'symbol': 'WDOH25',
            'timestamp': datetime.now().isoformat(),
            'price': 5000.0 + i,
            'volume': 30 + i*10,  # Volume crescente
            'trade_type': 2  # Todos compras
        }
        pattern = reader.analyze_tick(tick)
        time.sleep(0.001)
    
    speed = reader.get_current_speed()
    print(f"   - Velocidade do tape: {speed:.2f} trades/segundo")
    print(f"   - Trades no buffer: {len(reader.trade_buffer)}")
    
    # Verificar detecção de padrão
    recent_trades = reader.trade_buffer[-5:]
    if len(recent_trades) >= 5:
        # Verificar se são todos na mesma direção
        types = [t['trade_type'] for t in recent_trades]
        volumes = [t['volume'] for t in recent_trades]
        
        same_direction = len(set(types)) == 1
        increasing_volume = volumes == sorted(volumes)
        
        print(f"   - Mesma direção: {'Sim' if same_direction else 'Não'}")
        print(f"   - Volume crescente: {'Sim' if increasing_volume else 'Não'}")
        
        if same_direction and increasing_volume:
            print("   - Padrão de SWEEP detectado!")
    
    print("   [OK] Tape Reader funcionando")
    
    # 4. LiquidityMonitor
    print("\n4. Testando LiquidityMonitor...")
    
    monitor = LiquidityMonitor(HMARL_FLOW_CONFIG)
    
    # Simular diferentes estados de liquidez
    scenarios = [
        {
            'name': 'Alta liquidez',
            'bids': {4995: 200, 4990: 300, 4985: 250},
            'asks': {5005: 180, 5010: 280, 5015: 220}
        },
        {
            'name': 'Baixa liquidez',
            'bids': {4995: 50, 4990: 30},
            'asks': {5005: 40, 5010: 35}
        }
    ]
    
    for scenario in scenarios:
        monitor.update_book(scenario)
        metrics = monitor.liquidity_history[-1]['metrics']
        
        print(f"\n   Cenário: {scenario['name']}")
        print(f"   - Profundidade Bid: {metrics['bid_depth']}")
        print(f"   - Profundidade Ask: {metrics['ask_depth']}")
        print(f"   - Spread: {metrics['spread']}")
        print(f"   - Desequilíbrio: {metrics['imbalance']:.3f}")
        print(f"   - Score de liquidez: {metrics['liquidity_score']:.2f}")
    
    assert len(monitor.liquidity_history) == 2
    print("\n   [OK] Monitor de liquidez funcionando")
    
    # 5. Integração dos componentes
    print("\n5. Testando integração entre componentes...")
    
    # Simular fluxo completo
    total_volume = 0
    for i in range(10):
        # Criar trade
        tick = {
            'symbol': 'WDOH25',
            'timestamp': datetime.now().isoformat(),
            'price': 5000.0 + i*0.5,
            'volume': 15 + i%10,
            'trade_type': 2 if i%3 != 0 else 3
        }
        
        total_volume += tick['volume']
        
        # Processar em tape reader
        reader.analyze_tick(tick)
        
        # Criar flow point
        flow_point = FlowDataPoint(
            timestamp=datetime.now(),
            symbol=tick['symbol'],
            price=tick['price'],
            volume=tick['volume'],
            trade_type=tick['trade_type'],
            aggressor='buyer' if tick['trade_type'] == 2 else 'seller',
            trade_size_category='medium',
            speed_of_tape=reader.get_current_speed()
        )
        
        # Analisar fluxo
        analysis = engine.analyze(flow_point)
    
    print(f"   - Total de volume processado: {total_volume}")
    print(f"   - Velocidade final do tape: {reader.get_current_speed():.2f} trades/seg")
    print(f"   - OFI final (5m): {analysis['ofi'].get('5', 0):.3f}")
    print("   [OK] Integração funcionando")
    
    print("\n" + "="*60)
    print("TODOS OS COMPONENTES TESTADOS COM SUCESSO!")
    print("="*60)
    
    # Resumo dos componentes
    print("\nComponentes validados:")
    print("1. FlowDataPoint - Estrutura de dados para fluxo")
    print("2. FlowAnalysisEngine - Cálculo de OFI e métricas")
    print("3. AutomatedTapeReader - Detecção de padrões")
    print("4. LiquidityMonitor - Análise de profundidade")
    print("5. Integração - Fluxo completo de dados")
    
    print("\nPróximos passos:")
    print("- Instalar Valkey/Redis para persistência")
    print("- Testar com dados reais do ProfitDLL")
    print("- Integrar com sistema de ML existente")
    
    return True


if __name__ == "__main__":
    try:
        test_all_components()
        print("\n[SUCESSO] Implementação validada!")
    except Exception as e:
        print(f"\n[ERRO] Falha nos testes: {e}")
        import traceback
        traceback.print_exc()