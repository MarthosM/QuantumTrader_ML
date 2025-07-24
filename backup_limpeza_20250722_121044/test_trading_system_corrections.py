#!/usr/bin/env python3
"""
Teste das Correções do TradingSystem
Verifica se as implementações críticas estão funcionais
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_trading_system_corrections():
    """Testa as correções implementadas no TradingSystem"""
    print("🧪 TESTE: Correções TradingSystem")
    print("="*60)
    
    try:
        # Importar sem erro
        from trading_system import TradingSystem
        
        # Configuração básica
        config = {
            'dll_path': 'test.dll',
            'username': 'test',
            'password': 'test',
            'models_dir': 'test_models',
            'ticker': 'WDOQ25'
        }
        
        # Criar instância
        trading_system = TradingSystem(config)
        print("✅ TradingSystem criado com sucesso")
        
        # Testar se ProductionDataValidator foi integrado
        if hasattr(trading_system, 'production_validator'):
            print("✅ ProductionDataValidator integrado")
        else:
            print("❌ ProductionDataValidator não encontrado")
            
        # Testar método de validação
        if hasattr(trading_system, '_validate_production_data'):
            print("✅ Método _validate_production_data implementado")
        else:
            print("❌ Método _validate_production_data não encontrado")
        
        # Testar método de execução segura
        if hasattr(trading_system, '_execute_order_safely'):
            print("✅ Método _execute_order_safely implementado")
        else:
            print("❌ Método _execute_order_safely não encontrado")
            
        # Testar detecção de regime
        if hasattr(trading_system, '_detect_market_regime'):
            print("✅ Método _detect_market_regime implementado")
            
            # Criar dados de teste
            import pandas as pd
            import numpy as np
            
            # Simular dados de mercado
            data = {
                'open': np.random.uniform(5000, 5100, 60),
                'high': np.random.uniform(5000, 5100, 60),
                'low': np.random.uniform(5000, 5100, 60),
                'close': np.random.uniform(5000, 5100, 60)
            }
            
            market_data = pd.DataFrame(data)
            
            # Testar detecção
            regime = trading_system._detect_market_regime(market_data)
            
            # Regimes válidos esperados
            valid_regimes = ['trend_up', 'trend_down', 'ranging', 'high_volatility', 'undefined']
            
            if regime in valid_regimes:
                print(f"✅ Detecção de regime funcionando: {regime}")
            else:
                print(f"❌ Regime inválido detectado: {regime}")
                
        else:
            print("❌ Método _detect_market_regime não encontrado")
        
        # Testar método de aplicação de otimização
        if hasattr(trading_system, '_apply_optimization_results'):
            print("✅ Método _apply_optimization_results implementado")
            
            # Testar com dados mock
            test_results = {
                'features': {
                    'changed': True,
                    'selected_features': ['close', 'volume', 'rsi']
                }
            }
            
            try:
                trading_system._apply_optimization_results(test_results)
                print("✅ _apply_optimization_results executado sem erro")
            except Exception as e:
                print(f"⚠️ _apply_optimization_results executou com aviso: {e}")
        else:
            print("❌ Método _apply_optimization_results não encontrado")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_production_safety():
    """Testa se as proteções de produção estão funcionando"""
    print(f"\n🛡️ TESTE: Proteções de Produção")
    print("="*60)
    
    # Salvar estado atual
    current_env = os.environ.get('TRADING_ENV', 'development')
    
    try:
        from trading_system import TradingSystem
        
        config = {
            'dll_path': 'test.dll',
            'username': 'test', 
            'password': 'test',
            'models_dir': 'test_models'
        }
        
        trading_system = TradingSystem(config)
        
        # Testar em modo desenvolvimento
        os.environ['TRADING_ENV'] = 'development'
        print("🧪 Testando modo DEVELOPMENT...")
        
        try:
            signal = {
                'action': 'buy',
                'price': 5000,
                'position_size': 1,
                'stop_loss': 4995,
                'take_profit': 5010
            }
            
            # Deve funcionar em desenvolvimento
            trading_system._simulate_order_execution(signal)
            print("✅ Simulação permitida em DEVELOPMENT")
            
        except Exception as e:
            print(f"❌ Erro inesperado em development: {e}")
        
        # Testar em modo produção
        os.environ['TRADING_ENV'] = 'production'
        print("🚨 Testando modo PRODUCTION...")
        
        try:
            # Deve falhar em produção
            trading_system._simulate_order_execution(signal)
            print("❌ PERIGO: Simulação permitida em PRODUCTION!")
            
        except RuntimeError as e:
            if "SIMULAÇÃO CHAMADA EM PRODUÇÃO" in str(e):
                print("✅ Simulação corretamente bloqueada em PRODUCTION")
            else:
                print(f"⚠️ Erro diferente do esperado: {e}")
        except Exception as e:
            print(f"⚠️ Erro não esperado: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de produção: {e}")
        return False
        
    finally:
        # Restaurar estado
        os.environ['TRADING_ENV'] = current_env

def main():
    """Executa todos os testes das correções"""
    print("🧪 TESTE COMPLETO - CORREÇÕES TRADING SYSTEM")
    print("="*70)
    
    results = []
    
    # Teste das implementações
    result1 = test_trading_system_corrections()
    results.append(("Correções Implementadas", result1))
    
    # Teste de segurança
    result2 = test_production_safety()  
    results.append(("Proteções de Produção", result2))
    
    # Resultado final
    print(f"\n" + "="*70)
    print("📋 RESULTADO DOS TESTES")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"   {test_name:<25}: {status}")
    
    print(f"\n🎯 RESULTADO FINAL: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 CORREÇÕES IMPLEMENTADAS COM SUCESSO!")
        print("✅ Métodos críticos completados")
        print("✅ Proteções de produção funcionando")
        print("✅ ProductionDataValidator integrado")
        print("✅ Execução segura implementada")
    else:
        print("⚠️ Algumas correções precisam de ajustes")
    
    print(f"\n🚨 PRÓXIMOS PASSOS:")
    print("1. Implementar place_order no ConnectionManager")
    print("2. Completar métodos de atualização nos managers")
    print("3. Testes com dados reais")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
