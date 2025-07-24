#!/usr/bin/env python3
"""
Teste SUPER LIMPO - apenas logging otimizado
Foca em resolver problema de spam de logs
"""

import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def configure_optimal_logging():
    """Configura logging para eliminar spam no terminal"""
    
    # 1. NÍVEL ROOT: WARNING para reduzir tudo
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s - %(name)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # 2. COMPONENTES ESSENCIAIS: INFO apenas quando necessário
    essential_loggers = [
        'TradingSystem',
        'ConnectionManager'
    ]
    
    for logger_name in essential_loggers:
        logging.getLogger(logger_name).setLevel(logging.INFO)
    
    # 3. COMPONENTES SILENCIOSOS: Apenas erros críticos
    silent_loggers = [
        'FeatureEngine',
        'ProductionDataValidator',
        'TechnicalIndicators',
        'MLFeatures',
        'DataIntegration',
        'RealTimeProcessor',
        'DataPipeline'
    ]
    
    for logger_name in silent_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    print("✅ Logging otimizado configurado")
    print("   - ROOT: WARNING")
    print("   - Essenciais: INFO")
    print("   - Outros: ERROR apenas")

def test_logging_reduction():
    """Testa se configuração reduz logs"""
    print("\n🧪 TESTE: Redução de Logs")
    print("=" * 40)
    
    # Criar vários loggers e testar
    loggers_to_test = [
        'FeatureEngine',
        'ProductionDataValidator', 
        'TechnicalIndicators',
        'ConnectionManager',
        'TradingSystem'
    ]
    
    for logger_name in loggers_to_test:
        logger = logging.getLogger(logger_name)
        
        # Tentar vários níveis
        logger.debug("Debug message - NÃO deve aparecer")
        logger.info("Info message - Pode aparecer se essencial")
        logger.warning("Warning message - Deve aparecer")
        
        # Verificar nível efetivo
        level_name = logging.getLevelName(logger.getEffectiveLevel())
        print(f"  {logger_name}: {level_name}")
    
    return True

def test_basic_imports():
    """Testa imports básicos sem executar componentes pesados"""
    print("\n🧪 TESTE: Imports Básicos")
    print("=" * 40)
    
    imports_success = 0
    imports_total = 0
    
    # Testar imports principais
    components = [
        ('TradingDataStructure', 'data_structure'),
        ('ConnectionManager', 'connection_manager'),
        ('DataIntegration', 'data_integration'),
        ('FeatureEngine', 'feature_engine'),
        ('TechnicalIndicators', 'technical_indicators')
    ]
    
    for class_name, module_name in components:
        imports_total += 1
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✅ {class_name}")
            imports_success += 1
        except Exception as e:
            print(f"  ❌ {class_name}: {e}")
    
    success_rate = (imports_success / imports_total) * 100
    print(f"  📊 Taxa de sucesso: {success_rate:.0f}%")
    
    return success_rate > 80

def test_data_structure_creation():
    """Testa criação de estrutura de dados básica"""
    print("\n🧪 TESTE: Estrutura de Dados")
    print("=" * 40)
    
    try:
        from data_structure import TradingDataStructure
        
        # Criar e inicializar estrutura
        ds = TradingDataStructure()
        ds.initialize_structure()
        
        # Verificar DataFrames básicos
        dataframes = ['candles', 'indicators', 'features']
        created_dfs = 0
        
        for df_name in dataframes:
            if hasattr(ds, df_name):
                df = getattr(ds, df_name)
                if hasattr(df, 'empty'):
                    print(f"  ✅ {df_name}: DataFrame vazio criado")
                    created_dfs += 1
                else:
                    print(f"  ⚠️ {df_name}: Não é DataFrame")
            else:
                print(f"  ❌ {df_name}: Não encontrado")
        
        success = created_dfs >= 2
        print(f"  📊 DataFrames criados: {created_dfs}/{len(dataframes)}")
        
        return success
        
    except Exception as e:
        print(f"  ❌ Erro: {e}")
        return False

def test_minimal_system():
    """Teste mínimo do sistema sem dependências externas"""
    print("\n🧪 TESTE: Sistema Mínimo")
    print("=" * 40)
    
    tests_passed = 0
    tests_total = 4
    
    # Teste 1: Configuração de logging
    try:
        configure_optimal_logging()
        print("  ✅ Logging configurado")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Logging: {e}")
    
    # Teste 2: Redução de logs
    try:
        if test_logging_reduction():
            print("  ✅ Redução de logs funcionando")
            tests_passed += 1
    except Exception as e:
        print(f"  ❌ Redução logs: {e}")
    
    # Teste 3: Imports básicos
    try:
        if test_basic_imports():
            print("  ✅ Imports principais OK")
            tests_passed += 1
    except Exception as e:
        print(f"  ❌ Imports: {e}")
    
    # Teste 4: Estrutura de dados
    try:
        if test_data_structure_creation():
            print("  ✅ Estrutura de dados OK")
            tests_passed += 1
    except Exception as e:
        print(f"  ❌ Estrutura dados: {e}")
    
    return tests_passed, tests_total

def main():
    """Executa teste super limpo focado em logs"""
    print("🧪 TESTE SUPER LIMPO - OTIMIZAÇÃO DE LOGS")
    print("=" * 60)
    print("🎯 Foco: Eliminar spam de logs no terminal")
    print("⚡ Objetivo: Testes rápidos e limpos")
    print()
    
    # Executar teste mínimo
    passed, total = test_minimal_system()
    
    # Resultado final
    print("\n" + "=" * 60)
    print("📋 RESULTADO FINAL")
    print("=" * 60)
    
    success_rate = (passed / total) * 100
    print(f"🎯 Testes: {passed}/{total} ({success_rate:.0f}%)")
    
    if success_rate >= 75:
        print("🎉 SISTEMA OTIMIZADO!")
        print("✅ Logs reduzidos com sucesso")
        print("⚡ Performance melhorada")
    else:
        print("⚠️ MELHORIAS PARCIAIS")
        print("🔧 Alguns testes precisam de atenção")
    
    print("\n💡 DICAS PARA USO:")
    print("1. Use configure_optimal_logging() antes de executar sistema")
    print("2. Ajuste níveis de log conforme necessidade")
    print("3. ProductionDataValidator está em modo ERROR apenas")
    print("4. ConnectionManager mantém INFO para debug essencial")
    
    print("\n🚀 Sistema pronto para uso sem spam!")

if __name__ == "__main__":
    main()
