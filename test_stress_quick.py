"""
Script de teste rápido de stress
"""

import sys
import os
import logging

sys.path.insert(0, os.path.abspath('src'))

from testing.stress_test_v3 import StressTestV3, StressScenario

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def quick_stress_test():
    """Executa versão reduzida do stress test"""
    print("="*60)
    print("STRESS TEST RÁPIDO - VERSÃO REDUZIDA")
    print("="*60)
    
    stress_test = StressTestV3()
    
    # Limpar cenários padrão
    stress_test.scenarios = []
    
    # Adicionar apenas cenários rápidos
    
    # 1. Teste de alta frequência (reduzido)
    high_freq = StressScenario(
        "High Frequency Data (Quick)",
        "Testa com 100 trades/segundo por 5 segundos"
    )
    high_freq.add_parameter('trades_per_second', 100)
    high_freq.add_parameter('duration_seconds', 5)
    stress_test.scenarios.append(high_freq)
    
    # 2. Processamento paralelo (reduzido)
    parallel = StressScenario(
        "Parallel Processing (Quick)",
        "Testa com 10 threads simultâneas"
    )
    parallel.add_parameter('num_threads', 10)
    parallel.add_parameter('calculations_per_thread', 5)
    stress_test.scenarios.append(parallel)
    
    # 3. Dados extremos (rápido)
    extreme = StressScenario(
        "Extreme Market Data (Quick)",
        "Testa com volatilidade de 5%"
    )
    extreme.add_parameter('price_volatility', 0.05)
    extreme.add_parameter('gap_percentage', 0.02)
    stress_test.scenarios.append(extreme)
    
    # Executar testes
    report = stress_test.run_all_tests()
    
    # Exibir resultados detalhados
    print("\nRESULTADOS DETALHADOS:")
    print("-"*60)
    
    for scenario in stress_test.scenarios:
        print(f"\nCenário: {scenario.name}")
        print(f"Status: {scenario.results.get('status', 'UNKNOWN')}")
        
        # Métricas específicas
        if 'High Frequency' in scenario.name:
            print(f"Trades processados: {scenario.results.get('trades_processed', 0)}")
            print(f"Throughput: {scenario.results.get('throughput', 0):.2f} trades/s")
            print(f"Latência média: {scenario.results.get('avg_latency_ms', 0):.2f} ms")
            print(f"Erros: {scenario.results.get('errors', 0)}")
        
        elif 'Parallel' in scenario.name:
            print(f"Total de cálculos: {scenario.results.get('total_calculations', 0)}")
            print(f"Taxa: {scenario.results.get('calcs_per_second', 0):.2f} calc/s")
            print(f"Erros: {scenario.results.get('total_errors', 0)}")
        
        elif 'Extreme' in scenario.name:
            print(f"Features calculadas: {scenario.results.get('features_calculated', 0)}")
            print(f"Tempo de cálculo: {scenario.results.get('calc_time_seconds', 0):.3f} s")
            print(f"Taxa de NaN: {scenario.results.get('nan_rate', 0):.2%}")

if __name__ == "__main__":
    quick_stress_test()