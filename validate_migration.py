#!/usr/bin/env python3
"""
Script de Validação Pós-Migração
Verifica se a migração foi bem-sucedida
"""

import sys
import importlib
import traceback

def validate_imports():
    """Valida que os novos módulos podem ser importados"""
    print("🔍 Validando imports...")
    
    modules_to_test = [
        'profit_dll_structures',
        'connection_manager_v4',
        'order_manager_v4',
    ]
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module} importado com sucesso")
        except ImportError as e:
            print(f"  ❌ Erro ao importar {module}: {e}")
            return False
    
    return True

def validate_structures():
    """Valida que as estruturas estão corretas"""
    print("\n🔍 Validando estruturas...")
    
    try:
        from profit_dll_structures import (
            TConnectorSendOrder, TConnectorAccountIdentifier,
            create_account_identifier, create_send_order,
            OrderSide, OrderType
        )
        
        # Testar criação de estruturas
        account = create_account_identifier(1, "12345")
        assert account.BrokerID == 1
        assert account.AccountID == "12345"
        print("  ✅ TConnectorAccountIdentifier OK")
        
        # Testar ordem
        order = create_send_order(
            account=account,
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=50.0
        )
        assert order.Ticker == "TEST"
        assert order.Quantity == 100
        print("  ✅ TConnectorSendOrder OK")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro validando estruturas: {e}")
        traceback.print_exc()
        return False

def validate_main_system():
    """Valida que o sistema principal pode ser importado"""
    print("\n🔍 Validando sistema principal...")
    
    try:
        # Tentar importar trading_system
        import trading_system
        print("  ✅ trading_system importado")
        
        # Verificar que usa as classes v4
        if hasattr(trading_system, 'ConnectionManager'):
            conn_module = trading_system.ConnectionManager.__module__
            if 'v4' in conn_module:
                print(f"  ✅ Usando ConnectionManagerV4 de {conn_module}")
            else:
                print(f"  ⚠️  ConnectionManager de {conn_module} - verificar se é v4")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro ao importar trading_system: {e}")
        return False

def main():
    """Executa todas as validações"""
    print("=" * 60)
    print("VALIDAÇÃO PÓS-MIGRAÇÃO PROFITDLL v4.0.0.30")
    print("=" * 60)
    
    all_ok = True
    
    # Adicionar src ao path
    sys.path.insert(0, 'src')
    
    # Executar validações
    all_ok &= validate_imports()
    all_ok &= validate_structures()
    all_ok &= validate_main_system()
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ MIGRAÇÃO VALIDADA COM SUCESSO!")
    else:
        print("❌ PROBLEMAS ENCONTRADOS NA MIGRAÇÃO")
        print("Por favor, verifique os erros acima")
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
