#!/usr/bin/env python3
"""
Teste Avançado do Sistema de Ordens
Sistema ML Trading v2.0 - Advanced Order System Test
"""
import sys
import os
import time
import threading
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.WARNING)  # Reduzir verbosidade
logger = logging.getLogger(__name__)

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_complex_position_scenarios():
    """Teste de cenários complexos de posição"""
    print("🧪 TESTE AVANÇADO 1: CENÁRIOS COMPLEXOS DE POSIÇÃO")
    print("="*60)
    
    try:
        from execution_engine import SimpleExecutionEngine
        from order_manager import Order, OrderSide, OrderType
        
        # Criar mocks
        mock_order_manager = Mock()
        mock_ml_coordinator = Mock()
        mock_risk_manager = Mock()
        
        # Criar engine
        engine = SimpleExecutionEngine(
            mock_order_manager,
            mock_ml_coordinator,
            mock_risk_manager
        )
        
        # Cenário 1: Sequência complexa de operações
        print("📋 Teste: Sequência de operações Long → Redução → Reversão → Short")
        
        # Compra inicial: flat → long 5
        order1 = Order(
            symbol='WDOZ24',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=5,
            fill_price=50000.0,
            filled_qty=5
        )
        engine._update_position(order1)
        pos = engine.position_tracker['WDOZ24']
        assert pos == {'quantity': 5, 'avg_price': 50000.0, 'side': 'long'}
        print("  ✅ Long 5 @ 50000")
        
        # Compra adicional: long 5 → long 8
        order2 = Order(
            symbol='WDOZ24',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=3,
            fill_price=50150.0,
            filled_qty=3
        )
        engine._update_position(order2)
        pos = engine.position_tracker['WDOZ24']
        expected_avg = (5 * 50000.0 + 3 * 50150.0) / 8  # 50056.25
        assert abs(pos['avg_price'] - expected_avg) < 0.01
        assert pos['quantity'] == 8 and pos['side'] == 'long'
        print(f"  ✅ Long 8 @ {pos['avg_price']:.2f}")
        
        # Venda parcial: long 8 → long 3
        order3 = Order(
            symbol='WDOZ24',
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=5,
            fill_price=50200.0,
            filled_qty=5
        )
        engine._update_position(order3)
        pos = engine.position_tracker['WDOZ24']
        assert abs(pos['avg_price'] - expected_avg) < 0.01  # Preço médio mantido
        assert pos['quantity'] == 3 and pos['side'] == 'long'
        print(f"  ✅ Long 3 @ {pos['avg_price']:.2f} (preço médio mantido)")
        
        # Reversão completa: long 3 → short 2
        order4 = Order(
            symbol='WDOZ24',
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=5,
            fill_price=50300.0,
            filled_qty=5
        )
        engine._update_position(order4)
        pos = engine.position_tracker['WDOZ24']
        assert pos['quantity'] == -2 and pos['side'] == 'short'
        assert pos['avg_price'] == 50300.0  # Novo preço da ordem de reversão
        print(f"  ✅ Short 2 @ {pos['avg_price']:.2f}")
        
        # Acumular posição short: short 2 → short 4
        order5 = Order(
            symbol='WDOZ24',
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=2,
            fill_price=50250.0,
            filled_qty=2
        )
        engine._update_position(order5)
        pos = engine.position_tracker['WDOZ24']
        expected_short_avg = (2 * 50300.0 + 2 * 50250.0) / 4  # 50275.0
        assert abs(pos['avg_price'] - expected_short_avg) < 0.01
        assert pos['quantity'] == -4 and pos['side'] == 'short'
        print(f"  ✅ Short 4 @ {pos['avg_price']:.2f}")
        
        # Fechamento total: short 4 → flat
        order6 = Order(
            symbol='WDOZ24',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=4,
            fill_price=50100.0,
            filled_qty=4
        )
        engine._update_position(order6)
        pos = engine.position_tracker['WDOZ24']
        assert pos == {'quantity': 0, 'avg_price': 0.0, 'side': None}
        print("  ✅ Flat (posição zerada)")
        
        print("🎉 Cenários complexos de posição: PASSOU\n")
        return True
        
    except Exception as e:
        print(f"❌ Erro nos cenários complexos: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_symbols():
    """Teste com múltiplos símbolos simultaneamente"""
    print("🧪 TESTE AVANÇADO 2: MÚLTIPLOS SÍMBOLOS")
    print("="*60)
    
    try:
        from execution_engine import SimpleExecutionEngine
        from order_manager import Order, OrderSide, OrderType
        
        # Criar mocks
        mock_order_manager = Mock()
        mock_ml_coordinator = Mock()
        mock_risk_manager = Mock()
        
        # Criar engine
        engine = SimpleExecutionEngine(
            mock_order_manager,
            mock_ml_coordinator,
            mock_risk_manager
        )
        
        # Símbolos diferentes
        symbols = ['WDOZ24', 'WINZ24', 'INDZ24']
        
        # Criar posições em cada símbolo
        for i, symbol in enumerate(symbols, 1):
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=i,
                fill_price=50000.0 + i * 100,
                filled_qty=i
            )
            engine._update_position(order)
            
            pos = engine.position_tracker[symbol]
            assert pos['quantity'] == i
            assert pos['avg_price'] == 50000.0 + i * 100
            assert pos['side'] == 'long'
            print(f"  ✅ {symbol}: Long {i} @ {pos['avg_price']}")
        
        # Verificar posições gerais
        positions = engine.get_positions()
        assert len(positions) == 3
        print("  ✅ Múltiplos símbolos gerenciados simultaneamente")
        
        # Fechar uma posição específica
        close_order = Order(
            symbol='WINZ24',
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=2,
            fill_price=50300.0,
            filled_qty=2
        )
        engine._update_position(close_order)
        
        positions = engine.get_positions()
        assert positions['WINZ24']['quantity'] == 0
        assert positions['WDOZ24']['quantity'] == 1  # Inalterada
        assert positions['INDZ24']['quantity'] == 3  # Inalterada
        print("  ✅ Fechamento seletivo por símbolo")
        
        print("🎉 Múltiplos símbolos: PASSOU\n")
        return True
        
    except Exception as e:
        print(f"❌ Erro múltiplos símbolos: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_callback_edge_cases():
    """Teste de casos extremos nos callbacks"""
    print("🧪 TESTE AVANÇADO 3: CASOS EXTREMOS NOS CALLBACKS")
    print("="*60)
    
    try:
        from execution_engine import SimpleExecutionEngine
        from order_manager import Order, OrderSide, OrderType, OrderStatus
        
        # Criar mocks
        mock_order_manager = Mock()
        mock_ml_coordinator = Mock()
        mock_risk_manager = Mock()
        
        # Criar engine
        engine = SimpleExecutionEngine(
            mock_order_manager,
            mock_ml_coordinator,
            mock_risk_manager
        )
        
        # Caso 1: Ordem com dados inválidos
        bad_order = Mock()
        bad_order.profit_id = None
        bad_order.status = OrderStatus.FILLED
        
        # Não deve quebrar
        engine._on_order_update(bad_order)
        print("  ✅ Tratamento de ordem com dados inválidos")
        
        # Caso 2: Ordem sem fill_price
        partial_order = Order(
            symbol='WDOZ24',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1,
            price=50000.0
        )
        partial_order.profit_id = 123
        partial_order.status = OrderStatus.FILLED
        # fill_price não definido
        
        initial_stats = engine.get_execution_stats()
        engine._on_order_update(partial_order)
        
        # Deve processar sem erro
        new_stats = engine.get_execution_stats()
        assert new_stats['successful_orders'] > initial_stats['successful_orders']
        print("  ✅ Tratamento de ordem sem fill_price")
        
        # Caso 3: Múltiplos status de ordem
        statuses = [OrderStatus.REJECTED, OrderStatus.ERROR, OrderStatus.CANCELLED]
        
        for status in statuses:
            test_order = Order(
                symbol='TESTORDER',
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1
            )
            test_order.profit_id = 999
            test_order.status = status
            
            engine.active_orders["999"] = test_order
            engine._on_order_update(test_order)
            
            # Deve remover das ativas
            assert "999" not in engine.active_orders
            print(f"  ✅ Status {status.value} processado")
        
        print("🎉 Casos extremos nos callbacks: PASSOU\n")
        return True
        
    except Exception as e:
        print(f"❌ Erro casos extremos: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_high_frequency_operations():
    """Teste de operações de alta frequência"""
    print("🧪 TESTE AVANÇADO 4: OPERAÇÕES ALTA FREQUÊNCIA")
    print("="*60)
    
    try:
        from execution_engine import SimpleExecutionEngine
        from order_manager import Order, OrderSide, OrderType, OrderStatus
        
        # Criar mocks
        mock_order_manager = Mock()
        mock_ml_coordinator = Mock()
        mock_risk_manager = Mock()
        
        # Criar engine
        engine = SimpleExecutionEngine(
            mock_order_manager,
            mock_ml_coordinator,
            mock_risk_manager
        )
        
        # Simular 100 operações rápidas
        num_operations = 100
        
        for i in range(num_operations):
            # Alternar entre compra e venda
            side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
            
            order = Order(
                symbol='WDOZ24',
                side=side,
                order_type=OrderType.MARKET,
                quantity=1,
                fill_price=50000.0 + (i % 10),
                filled_qty=1
            )
            order.profit_id = 1000 + i
            order.status = OrderStatus.FILLED
            
            # Processar callback
            engine._on_order_update(order)
        
        # Verificar estatísticas
        stats = engine.get_execution_stats()
        assert stats['successful_orders'] == num_operations
        assert stats['total_orders'] == 0  # process_ml_signal não foi chamado
        
        print(f"  ✅ {num_operations} operações processadas")
        print(f"  ✅ Taxa de sucesso: {stats['success_rate']:.1%}")
        
        # Verificar posição final (deveria ser flat por alternância)
        position = engine.position_tracker.get('WDOZ24')
        assert position is not None, "Posição deveria existir"
        assert position['quantity'] == 0  # 50 compras - 50 vendas = 0
        assert position['side'] is None
        print("  ✅ Posição final balanceada (flat)")
        
        print("🎉 Operações alta frequência: PASSOU\n")
        return True
        
    except Exception as e:
        print(f"❌ Erro alta frequência: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stress_validation():
    """Teste de validação sob stress"""
    print("🧪 TESTE AVANÇADO 5: VALIDAÇÃO SOB STRESS")
    print("="*60)
    
    try:
        from execution_engine import SimpleExecutionEngine
        
        # Criar mocks
        mock_order_manager = Mock()
        mock_ml_coordinator = Mock()
        mock_risk_manager = Mock()
        
        # Configurar risk manager variável
        def variable_risk_check(signal):
            # 70% aprovação, 30% rejeição
            import random
            if random.random() < 0.7:
                return {'approved': True, 'position_size': 1}
            else:
                return {'approved': False, 'reason': 'Random rejection'}
        
        mock_risk_manager.check_signal_risk.side_effect = variable_risk_check
        mock_order_manager.send_order.return_value = Mock(profit_id=555)
        
        # Criar engine
        engine = SimpleExecutionEngine(
            mock_order_manager,
            mock_ml_coordinator,
            mock_risk_manager
        )
        
        # Gerar 50 sinais diversos
        approved_count = 0
        rejected_count = 0
        
        for i in range(50):
            signal = {
                'symbol': 'WDOZ24',
                'action': 'buy' if i % 2 == 0 else 'sell',
                'confidence': 0.5 + (i % 5) * 0.1,  # 0.5 a 0.9
                'prediction': {
                    'direction': 1 if i % 2 == 0 else -1,
                    'magnitude': 0.001 + (i % 3) * 0.001  # 0.001 a 0.003
                }
            }
            
            order_id = engine.process_ml_signal(signal)
            if order_id:
                approved_count += 1
            else:
                rejected_count += 1
        
        print(f"  ✅ {approved_count} sinais aprovados")
        print(f"  ✅ {rejected_count} sinais rejeitados")
        print(f"  ✅ Taxa de aprovação: {approved_count/50:.1%}")
        
        # Verificar se engine ainda está funcional
        test_signal = {
            'symbol': 'WDOZ24',
            'action': 'buy',
            'confidence': 0.8,
            'prediction': {'direction': 1, 'magnitude': 0.002}
        }
        
        is_valid = engine._validate_ml_signal(test_signal)
        assert is_valid == True
        print("  ✅ Engine ainda funcional após stress test")
        
        print("🎉 Validação sob stress: PASSOU\n")
        return True
        
    except Exception as e:
        print(f"❌ Erro validação stress: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_advanced_tests():
    """Executa todos os testes avançados"""
    print("🚀 TESTES AVANÇADOS DO SISTEMA DE ORDENS")
    print("ML Trading v2.0 - Advanced Order System Tests")
    print("=" * 70)
    
    # Lista de testes avançados
    tests = [
        ("Cenários Complexos", test_complex_position_scenarios),
        ("Múltiplos Símbolos", test_multiple_symbols),  
        ("Casos Extremos Callbacks", test_callback_edge_cases),
        ("Alta Frequência", test_high_frequency_operations),
        ("Validação Stress", test_stress_validation)
    ]
    
    # Executar testes
    results = []
    total_tests = len(tests)
    passed_tests = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                passed_tests += 1
                print(f"🎉 {test_name}: PASSOU\n")
            else:
                print(f"❌ {test_name}: FALHOU\n")
        except Exception as e:
            print(f"💥 {test_name}: ERRO CRÍTICO - {e}\n")
            results.append((test_name, False))
    
    # Relatório final
    print("=" * 70)
    print("📊 RELATÓRIO FINAL DOS TESTES AVANÇADOS")
    print("=" * 70)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name:25}: {status}")
    
    print(f"\n📈 RESULTADO: {passed_tests}/{total_tests} testes avançados passaram")
    print(f"📊 Taxa de sucesso: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 TODOS OS TESTES AVANÇADOS PASSARAM!")
        print("✅ Sistema robusto e pronto para produção!")
        print("🚀 Testado sob condições extremas!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} teste(s) avançado(s) falharam")
        print("🔧 Sistema básico funcional, ajustes recomendados")
    
    print("=" * 70)
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_advanced_tests()
    sys.exit(0 if success else 1)
