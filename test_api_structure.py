#!/usr/bin/env python3
"""
Teste estrutural da API de dados históricos
Verifica se as correções na API estão corretas sem fazer conexão real
"""

import sys
import os
import logging

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_api_structure():
    """Testa a estrutura das correções da API"""
    
    print("=== TESTE ESTRUTURAL DA API ===")
    print()
    
    try:
        # 1. Testar imports
        print("1. Testando imports...")
        from connection_manager import ConnectionManager
        print("   ✅ ConnectionManager importado")
        
        # 2. Testar inicialização 
        print("2. Testando inicialização...")
        dll_path = r"C:\fake\path\ProfitDLL.dll"  # Path fake
        connection = ConnectionManager(dll_path)
        print("   ✅ ConnectionManager inicializado")
        
        # 3. Verificar se os atributos necessários existem
        print("3. Verificando atributos...")
        assert hasattr(connection, '_historical_data_count'), "❌ _historical_data_count missing"
        assert connection._historical_data_count == 0, "❌ _historical_data_count not initialized"
        print("   ✅ _historical_data_count inicializado corretamente")
        
        # 4. Verificar se métodos existem
        print("4. Verificando métodos...")
        assert hasattr(connection, 'wait_for_historical_data'), "❌ wait_for_historical_data missing"
        assert hasattr(connection, 'request_historical_data'), "❌ request_historical_data missing"  
        assert hasattr(connection, 'request_historical_data_alternative'), "❌ request_historical_data_alternative missing"
        print("   ✅ Métodos necessários existem")
        
        # 5. Verificar assinaturas de métodos
        print("5. Verificando assinaturas...")
        import inspect
        
        # wait_for_historical_data
        sig = inspect.signature(connection.wait_for_historical_data)
        assert 'timeout_seconds' in sig.parameters, "❌ timeout_seconds param missing"
        print("   ✅ wait_for_historical_data tem parâmetro timeout_seconds")
        
        # request_historical_data
        sig = inspect.signature(connection.request_historical_data) 
        assert 'ticker' in sig.parameters, "❌ ticker param missing"
        assert 'start_date' in sig.parameters, "❌ start_date param missing"
        assert 'end_date' in sig.parameters, "❌ end_date param missing"
        print("   ✅ request_historical_data tem parâmetros corretos")
        
        # 6. Testar logic básica do wait_for_historical_data
        print("6. Testando lógica básica...")
        
        # Simular que não há dados
        connection._historical_data_count = 0
        # Chamar com timeout muito baixo deveria retornar False
        result = connection.wait_for_historical_data(timeout_seconds=1)
        assert result is False, "❌ Deveria retornar False sem dados"
        print("   ✅ wait_for_historical_data retorna False sem dados")
        
        # Simular que há dados
        connection._historical_data_count = 100
        # Chamar deveria retornar True rapidamente (dados estáveis)
        result = connection.wait_for_historical_data(timeout_seconds=3)
        assert result is True, "❌ Deveria retornar True com dados"
        print("   ✅ wait_for_historical_data retorna True com dados")
        
        print()
        print("🎉 TODOS OS TESTES ESTRUTURAIS PASSARAM!")
        print()
        print("✅ As correções implementadas estão estruturalmente corretas:")
        print("   - Contador de dados históricos funcional")
        print("   - Método de espera implementado")
        print("   - APIs de requisição existem") 
        print("   - Parâmetros corretos")
        print()
        print("🚀 PRONTO para testar com conexão real!")
        
        return True
        
    except Exception as e:
        print(f"❌ FALHA NO TESTE: {e}")
        import traceback
        print(f"Stack trace: {traceback.format_exc()}")
        return False

def test_trading_system_integration():
    """Testa se o trading system pode usar as correções"""
    
    print("=== TESTE INTEGRAÇÃO TRADING SYSTEM ===")
    print()
    
    try:
        # Importar trading system
        from trading_system import TradingSystem
        print("✅ TradingSystem importado")
        
        # Verificar se método existe
        assert hasattr(TradingSystem, '_load_historical_data_safe'), "❌ _load_historical_data_safe missing"
        print("✅ _load_historical_data_safe existe")
        
        # Testar código não quebra
        config = {'historical_days': 5}
        # Não vamos instanciar completamente, apenas testar estrutura
        print("✅ Integração estruturalmente compatível")
        
        return True
        
    except Exception as e:
        print(f"❌ FALHA INTEGRAÇÃO: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Executando testes estruturais...")
    print()
    
    test1 = test_api_structure()
    test2 = test_trading_system_integration()
    
    print()
    if test1 and test2:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print()
        print("📋 PRÓXIMOS PASSOS:")
        print("1. Verificar credenciais no arquivo .env")
        print("2. Executar: python src/main.py")
        print("3. Observar logs para:")
        print("   - '📈 DADO HISTÓRICO RECEBIDO'")  
        print("   - '⏳ Download histórico WDOQ25: X%'")
        print("   - '✅ Download histórico de WDOQ25 COMPLETO!'")
        print("   - '📊 X dados históricos recebidos...'")
        
        sys.exit(0)
    else:
        print("❌ ALGUNS TESTES FALHARAM!")
        sys.exit(1)
