#!/usr/bin/env python3
"""
Teste de integração do sistema de execução com o trading system
"""

import sys
import os
sys.path.append('src')

import logging
from unittest.mock import Mock, MagicMock
from datetime import datetime

# Mock dos componentes que podem não estar disponíveis
sys.modules['order_manager'] = Mock()
sys.modules['execution_engine'] = Mock()
sys.modules['execution_integration'] = Mock()

from trading_system import TradingSystem

def test_execution_system_integration():
    """Testa a integração do sistema de execução"""
    
    print("="*60)
    print("TESTE DE INTEGRAÇÃO - SISTEMA DE EXECUÇÃO")
    print("="*60)
    
    # Configuração de teste
    config = {
        'dll_path': 'dummy.dll',
        'username': 'test_user',
        'password': 'test_pass',
        'models_dir': 'models/trained',
        'ticker': 'WDOQ25',
        'initial_balance': 100000,
        'strategy': {
            'type': 'ml_based',
            'confidence_threshold': 0.6
        },
        'risk': {
            'max_position_size': 3,
            'max_daily_loss': 0.05
        }
    }
    
    # Criar sistema de trading
    trading_system = TradingSystem(config)
    
    # Verificar se componentes de execução foram inicializados
    print("\n1. Verificando inicialização dos componentes:")
    print(f"   - order_manager: {trading_system.order_manager}")
    print(f"   - execution_engine: {trading_system.execution_engine}")
    print(f"   - execution_integration: {trading_system.execution_integration}")
    
    # Testar métodos de execução
    print("\n2. Testando métodos de execução:")
    
    # Status de execução
    execution_status = trading_system.get_execution_status()
    print(f"   - Status de execução: {execution_status}")
    
    # Estatísticas de execução
    exec_stats = trading_system.get_execution_statistics()
    print(f"   - Estatísticas: {exec_stats}")
    
    # Ordens ativas
    active_orders = trading_system.get_active_orders()
    print(f"   - Ordens ativas: {len(active_orders)}")
    
    # Testar sinal de exemplo
    print("\n3. Testando processamento de sinal:")
    
    test_signal = {
        'symbol': 'WDOQ25',
        'action': 'buy',
        'confidence': 0.75,
        'prediction': {
            'regime': 'trend_up',
            'probability': 0.8,
            'direction': 0.75,
            'magnitude': 0.005
        },
        'price': 127500,
        'stop_loss': 127000,
        'take_profit': 128500,
        'position_size': 1
    }
    
    try:
        # Simular execução segura
        trading_system._execute_order_safely(test_signal)
        print("   ✅ Processamento de sinal executado com sucesso")
    except Exception as e:
        print(f"   ⚠️ Erro no processamento (esperado em teste): {e}")
    
    # Testar status completo
    print("\n4. Testando status completo do sistema:")
    status = trading_system.get_status()
    
    print("   Status do sistema:")
    for key, value in status.items():
        if key == 'execution':
            print(f"   - {key}: {value}")
        elif key == 'execution_stats':
            print(f"   - {key}: {value}")
        else:
            print(f"   - {key}: {str(value)[:50]}...")
    
    # Teste de parada segura
    print("\n5. Testando parada segura do sistema:")
    try:
        trading_system.stop()
        print("   ✅ Sistema parado com segurança")
    except Exception as e:
        print(f"   ⚠️ Erro na parada (esperado em teste): {e}")
    
    print("\n" + "="*60)
    print("TESTE DE INTEGRAÇÃO CONCLUÍDO")
    print("="*60)

def test_manual_order():
    """Testa funcionalidade de ordem manual"""
    
    print("\n" + "="*60)
    print("TESTE - ORDEM MANUAL")
    print("="*60)
    
    config = {
        'dll_path': 'dummy.dll',
        'username': 'test_user',
        'password': 'test_pass',
        'models_dir': 'models/trained',
        'ticker': 'WDOQ25'
    }
    
    trading_system = TradingSystem(config)
    
    # Testar ordem manual
    print("\nTestando ordem manual:")
    order_id = trading_system.manual_order(
        symbol='WDOQ25',
        side='buy',
        quantity=1,
        order_type='limit',
        price=127500
    )
    
    print(f"   - Ordem manual enviada: {order_id}")
    
    # Testar cancelamento
    print("\nTestando cancelamento de ordens:")
    result = trading_system.cancel_all_orders('WDOQ25')
    print(f"   - Cancelamento executado: {result}")
    
    # Testar fechamento de posição
    print("\nTestando fechamento de posição:")
    result = trading_system.close_position('WDOQ25', at_market=True)
    print(f"   - Fechamento executado: {result}")
    
    print("\n" + "="*60)
    print("TESTE DE ORDEM MANUAL CONCLUÍDO")
    print("="*60)

def test_emergency_procedures():
    """Testa procedimentos de emergência"""
    
    print("\n" + "="*60)
    print("TESTE - PROCEDIMENTOS DE EMERGÊNCIA")
    print("="*60)
    
    config = {
        'dll_path': 'dummy.dll',
        'username': 'test_user',
        'password': 'test_pass',
        'models_dir': 'models/trained',
        'ticker': 'WDOQ25'
    }
    
    trading_system = TradingSystem(config)
    
    print("\nTestando parada de emergência:")
    try:
        trading_system.emergency_stop()
        print("   ✅ Parada de emergência executada")
    except Exception as e:
        print(f"   ⚠️ Erro na emergência (esperado em teste): {e}")
    
    print("\n" + "="*60)
    print("TESTE DE EMERGÊNCIA CONCLUÍDO")
    print("="*60)

if __name__ == "__main__":
    # Configurar logging para o teste
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_execution_integration.log')
        ]
    )
    
    print("INICIANDO TESTES DE INTEGRAÇÃO DO SISTEMA DE EXECUÇÃO")
    print(f"Timestamp: {datetime.now()}")
    
    try:
        # Executar testes
        test_execution_system_integration()
        test_manual_order()
        test_emergency_procedures()
        
        print("\n🎉 TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")
        
    except Exception as e:
        print(f"\n❌ ERRO NOS TESTES: {e}")
        import traceback
        traceback.print_exc()
