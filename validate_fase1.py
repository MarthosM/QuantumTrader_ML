"""
Script de Validação da Fase 1 - Infraestrutura de Dados
Executa testes completos e valida se a Fase 1 está pronta
"""

import sys
import os
from datetime import datetime
import logging

# Adicionar src e tests ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

def setup_logging():
    """Configura logging para validação"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_basic_imports():
    """Testa se todos os módulos podem ser importados"""
    
    print("\n1. TESTANDO IMPORTS DOS MÓDULOS...")
    
    try:
        from data.real_data_collector import RealDataCollector
        print("[OK] RealDataCollector importado")
    except Exception as e:
        print(f"[ERROR] Erro importando RealDataCollector: {e}")
        return False
    
    try:
        from data.trading_data_structure_v3 import TradingDataStructureV3
        print("[OK] TradingDataStructureV3 importado")
    except Exception as e:
        print(f"[ERROR] Erro importando TradingDataStructureV3: {e}")
        return False
    
    try:
        from test_real_data_collection import TestRealDataCollection, TestTradingDataStructureV3
        print("[OK] Testes importados")
    except Exception as e:
        print(f"[ERROR] Erro importando testes: {e}")
        return False
    
    return True

def run_unit_tests():
    """Executa testes unitários"""
    
    print("\n2. EXECUTANDO TESTES UNITÁRIOS...")
    
    try:
        from test_real_data_collection import run_all_tests
        success = run_all_tests()
        return success
    except Exception as e:
        print(f"[ERROR] Erro executando testes: {e}")
        return False

def test_integration():
    """Testa integração entre componentes"""
    
    print("\n3. TESTANDO INTEGRAÇÃO DOS COMPONENTES...")
    
    try:
        from data.real_data_collector import RealDataCollector
        from data.trading_data_structure_v3 import TradingDataStructureV3
        import pandas as pd
        import numpy as np
        
        # Teste de integração completa
        print("   Criando collector...")
        collector = RealDataCollector()
        
        print("   Criando dados simulados...")
        # Simular dados de trades
        trades_data = []
        base_time = datetime(2025, 1, 27, 10, 0)
        
        for i in range(500):
            trade = {
                'datetime': base_time + pd.Timedelta(seconds=i*10),
                'price': 5900 + np.random.randn() * 2,
                'volume': np.random.randint(1000, 10000),
                'quantity': np.random.randint(1, 10),
                'side': np.random.choice(['BUY', 'SELL'])
            }
            trades_data.append(trade)
        
        trades_df = pd.DataFrame(trades_data)
        trades_df.set_index('datetime', inplace=True)
        
        print("   Agregando trades...")
        candles = collector.aggregate_to_candles(trades_df, '1min')
        microstructure = collector.calculate_microstructure_metrics(trades_df, '1min')
        
        print("   Criando TradingDataStructure...")
        data_struct = TradingDataStructureV3()
        
        historical_data = {
            'candles': candles,
            'microstructure': microstructure,
            'trades': trades_df
        }
        
        print("   Inicializando com dados históricos...")
        data_struct.initialize_from_historical_data(historical_data)
        
        print("   Calculando features...")
        success = data_struct.calculate_all_features()
        
        if not success:
            print("[ERROR] Falha no cálculo de features")
            return False
        
        print("   Obtendo features...")
        features = data_struct.get_latest_features(50)
        
        if features.empty:
            print("[ERROR] Nenhuma feature retornada")
            return False
        
        print("   Testando modo tempo real...")
        for i in range(5):
            trade_data = {
                'datetime': datetime.now(),
                'price': 5900 + np.random.randn(),
                'volume': np.random.randint(1000, 10000),
                'quantity': np.random.randint(1, 10),
                'side': np.random.choice(['BUY', 'SELL'])
            }
            data_struct.add_tick_data(trade_data)
        
        print("   Verificando resumo...")
        summary = data_struct.get_data_summary()
        
        # Validações finais
        if not summary['initialized']:
            print("[ERROR] Estrutura não inicializada")
            return False
        
        if summary['metadata']['data_quality_score'] < 0.5:
            print(f"[ERROR] Quality score muito baixo: {summary['metadata']['data_quality_score']:.3f}")
            return False
        
        print(f"[OK] Integração completa validada (Quality: {summary['metadata']['data_quality_score']:.3f})")
        print(f"   - Candles: {summary['shapes']['candles']}")
        print(f"   - Features: {summary['shapes']['features']}")
        print(f"   - Tempo real: {summary['real_time_mode']}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Erro na integração: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Testa performance dos componentes"""
    
    print("\n4. TESTANDO PERFORMANCE...")
    
    try:
        import time
        from data.real_data_collector import RealDataCollector
        from data.trading_data_structure_v3 import TradingDataStructureV3
        import pandas as pd
        import numpy as np
        
        # Teste de performance com dados maiores
        print("   Criando dataset de performance (2000 trades)...")
        
        trades_data = []
        base_time = datetime(2025, 1, 27, 9, 0)
        
        for i in range(2000):
            trade = {
                'datetime': base_time + pd.Timedelta(seconds=i*5),
                'price': 5900 + np.random.randn() * 2,
                'volume': np.random.randint(1000, 10000),
                'quantity': np.random.randint(1, 10),
                'side': np.random.choice(['BUY', 'SELL'])
            }
            trades_data.append(trade)
        
        trades_df = pd.DataFrame(trades_data)
        trades_df.set_index('datetime', inplace=True)
        
        collector = RealDataCollector()
        
        # Teste 1: Agregação de candles
        start_time = time.time()
        candles = collector.aggregate_to_candles(trades_df, '1min')
        aggregation_time = time.time() - start_time
        
        print(f"   Agregação de candles: {aggregation_time:.3f}s ({len(trades_df)} trades -> {len(candles)} candles)")
        
        if aggregation_time > 5.0:
            print("[WARNING] Agregação lenta (> 5s)")
        
        # Teste 2: Cálculo de microestrutura
        start_time = time.time()
        microstructure = collector.calculate_microstructure_metrics(trades_df, '1min')
        micro_time = time.time() - start_time
        
        print(f"   Microestrutura: {micro_time:.3f}s")
        
        if micro_time > 3.0:
            print("[WARNING] Microestrutura lenta (> 3s)")
        
        # Teste 3: Inicialização da estrutura
        start_time = time.time()
        data_struct = TradingDataStructureV3()
        data_struct.initialize_from_historical_data({
            'candles': candles,
            'microstructure': microstructure
        })
        init_time = time.time() - start_time
        
        print(f"   Inicialização estrutura: {init_time:.3f}s")
        
        if init_time > 2.0:
            print("[WARNING] Inicialização lenta (> 2s)")
        
        # Teste 4: Cálculo de features
        start_time = time.time()
        success = data_struct.calculate_all_features()
        features_time = time.time() - start_time
        
        print(f"   Cálculo de features: {features_time:.3f}s")
        
        if features_time > 5.0:
            print("[WARNING] Features lentas (> 5s)")
        
        # Performance total
        total_time = aggregation_time + micro_time + init_time + features_time
        print(f"   Tempo total: {total_time:.3f}s")
        
        if total_time > 15.0:
            print("[WARNING] Performance geral lenta (> 15s)")
            return False
        
        print("[OK] Performance aceitável")
        return True
        
    except Exception as e:
        print(f"[ERROR] Erro no teste de performance: {e}")
        return False

def validate_file_structure():
    """Valida estrutura de arquivos necessários"""
    
    print("\n5. VALIDANDO ESTRUTURA DE ARQUIVOS...")
    
    required_files = [
        'src/data/real_data_collector.py',
        'src/data/trading_data_structure_v3.py',
        'tests/test_real_data_collection.py',
        'DEVELOPER_GUIDE_V3_REFACTORING.md',
        'NOVO_MAPA_FLUXO_DADOS.md'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"[OK] {file_path}")
    
    if missing_files:
        print(f"[ERROR] Arquivos faltando: {missing_files}")
        return False
    
    print("[OK] Estrutura de arquivos completa")
    return True

def generate_validation_report():
    """Gera relatório de validação"""
    
    print("\n" + "="*80)
    print("RELATÓRIO DE VALIDAÇÃO DA FASE 1")
    print("="*80)
    
    report = {
        'validation_time': datetime.now().isoformat(),
        'phase': 'Fase 1 - Infraestrutura de Dados',
        'tests_completed': [],
        'overall_status': 'PENDING'
    }
    
    # Executar todos os testes
    tests = [
        ('Imports', test_basic_imports),
        ('Unit Tests', run_unit_tests),
        ('Integration', test_integration),
        ('Performance', test_performance),
        ('File Structure', validate_file_structure)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            report['tests_completed'].append({
                'name': test_name,
                'status': 'PASS' if result else 'FAIL',
                'timestamp': datetime.now().isoformat()
            })
            
            if not result:
                all_passed = False
                
        except Exception as e:
            report['tests_completed'].append({
                'name': test_name,
                'status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            all_passed = False
    
    # Status geral
    if all_passed:
        report['overall_status'] = 'APPROVED'
        print("\n[SUCCESS] FASE 1 APROVADA!")
        print("[OK] Infraestrutura de dados funcionando corretamente")
        print("[READY] Pronto para prosseguir para FASE 2 - Pipeline ML")
        
        next_steps = [
            "1. Executar coleta de dados reais: python src/data/real_data_collector.py",
            "2. Validar dados coletados",
            "3. Iniciar FASE 2: Implementar MLFeaturesV3",
            "4. Atualizar pipeline de treinamento"
        ]
        
        print("\n[TODO] PRÓXIMOS PASSOS:")
        for step in next_steps:
            print(f"   {step}")
            
    else:
        report['overall_status'] = 'FAILED'
        print("\n[ERROR] FASE 1 REPROVADA!")
        print("Corrigir problemas antes de prosseguir para FASE 2")
        
        failed_tests = [test for test in report['tests_completed'] if test['status'] != 'PASS']
        print("\n[FIX] PROBLEMAS ENCONTRADOS:")
        for test in failed_tests:
            print(f"   - {test['name']}: {test['status']}")
            if 'error' in test:
                print(f"     Erro: {test['error']}")
    
    # Salvar relatório
    import json
    report_file = f"fase1_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[REPORT] Relatório salvo em: {report_file}")
    
    return all_passed


def main():
    """Função principal de validação"""
    
    setup_logging()
    
    print("="*80)
    print("VALIDAÇÃO DA FASE 1 - INFRAESTRUTURA DE DADOS")
    print("="*80)
    print(f"Iniciado em: {datetime.now()}")
    
    try:
        success = generate_validation_report()
        exit_code = 0 if success else 1
        
    except Exception as e:
        print(f"\n[CRITICAL] ERRO CRÍTICO NA VALIDAÇÃO: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 2
    
    print(f"\nValidação finalizada em: {datetime.now()}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()