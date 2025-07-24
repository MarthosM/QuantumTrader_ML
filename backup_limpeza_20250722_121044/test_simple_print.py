#!/usr/bin/env python3
"""
Teste simples do print do DataFrame
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_simple_dataframe_print():
    """Teste simples do print do DataFrame"""
    print("🧪 TESTE SIMPLES - PRINT DO DATAFRAME")
    print("="*60)
    
    try:
        from data_integration import DataIntegration
        from data_loader import DataLoader
        
        # Criar componentes
        data_loader = DataLoader()
        data_integration = DataIntegration(None, data_loader)
        
        print("📊 Criando DataFrame de teste...")
        
        # Forçar criação de DataFrame de teste
        success = data_integration.force_create_test_dataframe()
        
        if success:
            print("\n✅ DataFrame criado e impresso com sucesso!")
            
            # Testar método manual
            print("\n🔍 TESTANDO PRINT MANUAL:")
            data_integration.print_current_dataframe()
            
            # Mostrar estatísticas
            print("\n📊 ESTATÍSTICAS FINAIS:")
            stats = data_integration.get_dataframe_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            return True
        else:
            print("❌ Falha ao criar DataFrame de teste")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def main():
    """Executa teste simples"""
    success = test_simple_dataframe_print()
    
    if success:
        print("\n🎉 SUCESSO!")
        print("✅ Print do DataFrame está funcionando")
        print("\n💡 No sistema real, o print será ativado:")
        print("  - Automaticamente a cada 10 candles")
        print("  - A cada 5 minutos")
        print("  - Ao final do carregamento histórico")
        print("  - Manualmente via print_current_dataframe()")
        
        print("\n🚀 Para usar no seu sistema:")
        print("  data_integration.print_current_dataframe()")
        
    else:
        print("\n❌ Teste falhou")
    
    return success

if __name__ == "__main__":
    main()
