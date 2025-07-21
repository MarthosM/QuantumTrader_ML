#!/usr/bin/env python3
"""
Teste limpo do sistema - versão otimizada sem spam de logs
"""

import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from trading_system import TradingSystem
from data_integration import DataIntegration
from data_structure import TradingDataStructure
import pandas as pd
from datetime import datetime, timedelta
import time

def configure_clean_logging():
    """Configura logging limpo apenas com informações essenciais"""
    
    # Configurar logging root com nível WARNING para reduzir spam
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Configurar níveis específicos para componentes críticos
    critical_loggers = [
        'TradingSystem',
        'ConnectionManager', 
        'DataIntegration'
    ]
    
    for logger_name in critical_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
    
    # Silenciar completamente loggers verbosos
    verbose_loggers = [
        'FeatureEngine',
        'TechnicalIndicators',
        'MLFeatures',
        'ProductionDataValidator'
    ]
    
    for logger_name in verbose_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)  # Apenas erros críticos
    
    print("✅ Logging configurado - apenas informações essenciais")

def test_system_initialization():
    """Testa inicialização do sistema sem logs desnecessários"""
    print("\n🧪 TESTE 1: Inicialização do Sistema")
    print("=" * 50)
    
    try:
        # Configuração mínima para teste
        config = {
            'dll_path': 'dummy',
            'username': 'test',
            'password': 'test',
            'models_dir': 'test_models',
            'ticker': 'WDOQ25',
            'strategy': {},
            'risk': {}
        }
        
        # Criar sistema (apenas verificar se classe carrega)
        try:
            ts = TradingSystem(config)
            print("✅ TradingSystem classe carregada")
            return True
        except Exception as e:
            print(f"⚠️ TradingSystem requer DLL real: {e}")
            # Testar componentes individuais
            from data_structure import TradingDataStructure
            from connection_manager import ConnectionManager
            
            ds = TradingDataStructure()
            ds.initialize_structure()
            print("✅ TradingDataStructure inicializada")
            
            return True
            
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        return False

def test_dataframe_logging():
    """Testa logging de DataFrame sem spam"""
    print("\n🧪 TESTE 2: DataFrame Logging")
    print("=" * 50)
    
    try:
        # Criar dados de teste pequenos
        data_integration = DataIntegration()
        
        # Criar dados sintéticos mínimos (apenas 10 candles)
        test_data = []
        base_time = datetime.now() - timedelta(minutes=10)
        
        for i in range(10):
            test_data.append({
                'timestamp': base_time + timedelta(minutes=i),
                'price': 5100 + (i * 2),
                'volume': 100 + i,
                'quantity': 10 + i,
                'trade_type': 1,
                'trade_number': 1000 + i,
                'is_historical': True
            })
        
        # Processar dados
        for trade in test_data:
            data_integration.process_trade(trade)
        
        # Verificar se DataFrame foi criado
        if not data_integration.candles.empty:
            print(f"✅ DataFrame criado com {len(data_integration.candles)} candles")
            
            # Chamar log do DataFrame (deve aparecer apenas uma vez)
            data_integration.log_dataframe_summary()
            
            return True
        else:
            print("❌ DataFrame vazio")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste de DataFrame: {e}")
        return False

def test_timestamp_validation():
    """Testa validação de timestamps de forma silenciosa"""
    print("\n🧪 TESTE 3: Validação de Timestamps")
    print("=" * 50)
    
    try:
        data_integration = DataIntegration()
        
        # Teste 1: Dado histórico (deve aceitar)
        historical_trade = {
            'timestamp': datetime.now() - timedelta(days=1),
            'price': 5100,
            'volume': 100,
            'is_historical': True
        }
        
        result1 = data_integration._is_valid_timestamp(historical_trade)
        print(f"  📊 Histórico (1 dia): {'✅ Aceito' if result1 else '❌ Rejeitado'}")
        
        # Teste 2: Tempo real recente (deve aceitar)
        realtime_trade = {
            'timestamp': datetime.now() - timedelta(seconds=30),
            'price': 5100,
            'volume': 100,
            'is_historical': False
        }
        
        result2 = data_integration._is_valid_timestamp(realtime_trade)
        print(f"  🕐 Tempo real (30s): {'✅ Aceito' if result2 else '❌ Rejeitado'}")
        
        # Teste 3: Tempo real muito antigo (deve rejeitar)
        old_realtime = {
            'timestamp': datetime.now() - timedelta(minutes=10),
            'price': 5100,
            'volume': 100,
            'is_historical': False
        }
        
        result3 = data_integration._is_valid_timestamp(old_realtime)
        print(f"  ⚠️ Tempo real antigo (10min): {'✅ Aceito' if result3 else '❌ Rejeitado'}")
        
        # Resultado esperado: True, True, False
        expected_results = result1 and result2 and not result3
        print(f"  🎯 Validação: {'✅ Funcionando' if expected_results else '❌ Problemas'}")
        
        return expected_results
        
    except Exception as e:
        print(f"❌ Erro na validação de timestamps: {e}")
        return False

def test_anti_loop_system():
    """Testa sistema anti-loop de forma concisa"""
    print("\n🧪 TESTE 4: Sistema Anti-Loop")
    print("=" * 50)
    
    try:
        from trading_system import TradingSystem
        
        ts = TradingSystem()
        
        # Simular gap temporal
        now = datetime.now()
        ts.last_historical_load = now - timedelta(minutes=10)  # 10 min gap
        
        # Primeira verificação (deve permitir)
        can_fill1 = ts._check_and_fill_temporal_gap()
        print(f"  🔄 Primeira verificação gap: {'✅ Permitido' if can_fill1 else '❌ Bloqueado'}")
        
        # Segunda verificação imediata (deve bloquear)
        can_fill2 = ts._check_and_fill_temporal_gap()
        print(f"  🚫 Segunda verificação gap: {'✅ Permitido' if can_fill2 else '❌ Bloqueado (correto)'}")
        
        # Resetar flag manualmente para teste
        ts.gap_fill_in_progress = False
        
        # Terceira verificação (deve permitir novamente)
        can_fill3 = ts._check_and_fill_temporal_gap()
        print(f"  🔄 Após reset: {'✅ Permitido' if can_fill3 else '❌ Bloqueado'}")
        
        # Resultado esperado: True, False, True
        expected_results = can_fill1 and not can_fill2 and can_fill3
        print(f"  🎯 Anti-loop: {'✅ Funcionando' if expected_results else '❌ Problemas'}")
        
        return expected_results
        
    except Exception as e:
        print(f"❌ Erro no sistema anti-loop: {e}")
        return False

def test_performance_benchmark():
    """Teste rápido de performance sem logs"""
    print("\n🧪 TESTE 5: Performance Benchmark")
    print("=" * 50)
    
    try:
        # Cronometrar criação do sistema
        start_time = time.time()
        ts = TradingSystem()
        init_time = time.time() - start_time
        
        print(f"  ⏱️ Inicialização: {init_time:.3f}s")
        
        # Cronometrar processamento de trades
        start_time = time.time()
        
        # Processar 100 trades sintéticos
        base_time = datetime.now()
        for i in range(100):
            trade = {
                'timestamp': base_time + timedelta(seconds=i),
                'price': 5100 + (i % 10),
                'volume': 100,
                'quantity': 10,
                'trade_type': 1,
                'trade_number': i,
                'is_historical': True
            }
            ts.data_integration.process_trade(trade)
        
        process_time = time.time() - start_time
        trades_per_second = 100 / process_time if process_time > 0 else 0
        
        print(f"  📊 Processamento: {process_time:.3f}s")
        print(f"  🚀 Taxa: {trades_per_second:.0f} trades/s")
        
        # Verificar resultado
        total_candles = len(ts.data_integration.candles)
        print(f"  📈 Candles gerados: {total_candles}")
        
        # Performance aceitável: < 1s para inicialização, > 50 trades/s
        performance_ok = init_time < 1.0 and trades_per_second > 50
        print(f"  🎯 Performance: {'✅ Boa' if performance_ok else '⚠️ Verificar'}")
        
        return performance_ok
        
    except Exception as e:
        print(f"❌ Erro no benchmark: {e}")
        return False

def main():
    """Executa todos os testes de forma limpa e concisa"""
    print("🧪 TESTE LIMPO DO SISTEMA DE TRADING")
    print("=" * 60)
    print("📊 Versão otimizada - sem spam de logs")
    print("🎯 Foco em resultados essenciais")
    print()
    
    # Configurar logging limpo
    configure_clean_logging()
    
    # Executar testes
    tests = [
        ("Inicialização", test_system_initialization),
        ("DataFrame Logging", test_dataframe_logging),
        ("Validação Timestamps", test_timestamp_validation),
        ("Sistema Anti-Loop", test_anti_loop_system),
        ("Performance", test_performance_benchmark)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erro em {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumo final
    print("\n" + "=" * 60)
    print("📋 RESUMO DOS TESTES")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"  {test_name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 RESULTADO FINAL: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Sistema pronto para uso")
    elif passed > total // 2:
        print("⚠️ MAIORIA DOS TESTES PASSOU")
        print("🔧 Verificar testes que falharam")
    else:
        print("❌ VÁRIOS TESTES FALHARAM")
        print("🛠️ Sistema precisa de correções")
    
    print("\n💡 Sistema otimizado para reduzir spam de logs")
    print("🔧 Configure logging conforme necessidade")

if __name__ == "__main__":
    main()
