"""
Runner para executar todos os testes de integração
Gera relatório consolidado dos resultados
"""

import pytest
import sys
import os
import time
from datetime import datetime
import json

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def run_integration_tests():
    """Executa todos os testes de integração e gera relatório"""
    
    print("="*80)
    print("EXECUTANDO TESTES DE INTEGRAÇÃO END-TO-END")
    print("="*80)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Lista de módulos de teste
    test_modules = [
        "test_complete_flow.py",
        "test_risk_integration.py", 
        "test_performance.py",
        "test_failure_recovery.py"
    ]
    
    # Diretório dos testes
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resultados
    results = {
        'start_time': datetime.now().isoformat(),
        'modules': {},
        'summary': {
            'total_passed': 0,
            'total_failed': 0,
            'total_skipped': 0,
            'total_time': 0
        }
    }
    
    # Executar cada módulo
    for module in test_modules:
        print(f"\n{'='*60}")
        print(f"Executando: {module}")
        print("="*60)
        
        module_path = os.path.join(test_dir, module)
        start_time = time.time()
        
        # Executar pytest para o módulo
        # -v: verbose, -s: mostrar prints, --tb=short: traceback curto
        exit_code = pytest.main([module_path, "-v", "-s", "--tb=short"])
        
        duration = time.time() - start_time
        
        # Registrar resultado
        results['modules'][module] = {
            'exit_code': exit_code,
            'duration': duration,
            'status': 'PASSED' if exit_code == 0 else 'FAILED'
        }
        
        results['summary']['total_time'] += duration
        
        if exit_code == 0:
            print(f"\n[PASSED] {module} - PASSOU ({duration:.2f}s)")
        else:
            print(f"\n[FAILED] {module} - FALHOU ({duration:.2f}s)")
    
    # Gerar relatório final
    print("\n" + "="*80)
    print("RELATÓRIO FINAL DOS TESTES DE INTEGRAÇÃO")
    print("="*80)
    
    # Estatísticas por módulo
    print("\nResultados por módulo:")
    for module, result in results['modules'].items():
        status_icon = "[PASS]" if result['status'] == 'PASSED' else "[FAIL]"
        print(f"  {status_icon} {module:<30} {result['status']:<10} ({result['duration']:.2f}s)")
    
    # Resumo geral
    total_modules = len(results['modules'])
    passed_modules = sum(1 for r in results['modules'].values() if r['status'] == 'PASSED')
    failed_modules = total_modules - passed_modules
    
    print(f"\nResumo:")
    print(f"  Total de módulos: {total_modules}")
    print(f"  Passou: {passed_modules}")
    print(f"  Falhou: {failed_modules}")
    print(f"  Tempo total: {results['summary']['total_time']:.2f}s")
    print(f"  Tempo médio: {results['summary']['total_time']/total_modules:.2f}s")
    
    # Taxa de sucesso
    success_rate = (passed_modules / total_modules) * 100 if total_modules > 0 else 0
    print(f"  Taxa de sucesso: {success_rate:.1f}%")
    
    # Status geral
    print(f"\nStatus Geral: {'[SUCESSO]' if failed_modules == 0 else '[FALHA]'}")
    
    # Salvar relatório em arquivo
    results['end_time'] = datetime.now().isoformat()
    results['summary']['success_rate'] = success_rate
    
    report_file = f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRelatório salvo em: {report_file}")
    print(f"\nFinalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return 0 if failed_modules == 0 else 1


def run_specific_test_category(category):
    """Executa categoria específica de testes"""
    
    categories = {
        'flow': ['test_complete_flow.py'],
        'risk': ['test_risk_integration.py'],
        'performance': ['test_performance.py'],
        'recovery': ['test_failure_recovery.py'],
        'quick': ['test_complete_flow.py', 'test_risk_integration.py'],  # Testes rápidos
        'full': ['test_complete_flow.py', 'test_risk_integration.py', 
                'test_performance.py', 'test_failure_recovery.py']  # Todos
    }
    
    if category not in categories:
        print(f"Categoria '{category}' não encontrada.")
        print(f"Categorias disponíveis: {', '.join(categories.keys())}")
        return 1
    
    print(f"\nExecutando categoria: {category}")
    test_modules = categories[category]
    
    # Executar testes da categoria
    for module in test_modules:
        pytest.main([os.path.join(os.path.dirname(__file__), module), "-v"])
    
    return 0


if __name__ == "__main__":
    # Verificar argumentos
    if len(sys.argv) > 1:
        # Executar categoria específica
        category = sys.argv[1]
        exit_code = run_specific_test_category(category)
    else:
        # Executar todos os testes
        exit_code = run_integration_tests()
    
    sys.exit(exit_code)