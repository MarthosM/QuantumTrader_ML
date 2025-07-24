#!/usr/bin/env python3
"""
Script para imprimir DataFrame do sistema em execução
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def print_system_dataframe():
    """Imprime o DataFrame do sistema se estiver rodando"""
    print("🔍 VERIFICANDO SISTEMA EM EXECUÇÃO...")
    print("="*60)
    
    try:
        # Tentar acessar o sistema se estiver rodando
        # Isso pode ser usado durante execução real
        
        from trading_system import TradingSystem
        from data_integration import DataIntegration
        from data_loader import DataLoader
        
        print("✅ Módulos carregados com sucesso")
        
        # Para uso imediato: criar uma instância para demonstração
        print("\n📊 CRIANDO INSTÂNCIA PARA DEMONSTRAÇÃO:")
        
        data_loader = DataLoader()
        data_integration = DataIntegration(None, data_loader)
        
        # Criar DataFrame de demonstração
        data_integration.force_create_test_dataframe()
        
        print("\n" + "="*60)
        print("💡 INSTRUÇÕES PARA USO REAL:")
        print("="*60)
        print("1. No seu sistema em execução, tenha acesso ao data_integration")
        print("2. Chame: data_integration.print_current_dataframe()")
        print("3. Ou para estatísticas: data_integration.get_dataframe_stats()")
        print("")
        print("📝 EXEMPLO DE CÓDIGO:")
        print("```python")
        print("# No seu sistema principal:")
        print("trading_system = TradingSystem(config)")
        print("data_integration = trading_system.data_integration")
        print("")
        print("# Para imprimir DataFrame:")
        print("data_integration.print_current_dataframe()")
        print("")
        print("# Para obter apenas estatísticas:")
        print("stats = data_integration.get_dataframe_stats()")
        print("print(stats)")
        print("```")
        
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    print_system_dataframe()
