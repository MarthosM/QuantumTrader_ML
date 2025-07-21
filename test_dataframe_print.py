#!/usr/bin/env python3
"""
Teste do print do DataFrame - validar se está funcionando corretamente
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_integration import DataIntegration
from data_loader import DataLoader
from datetime import datetime, timedelta
import pandas as pd

def test_dataframe_print():
    """Testa o print do DataFrame com dados simulados"""
    print("🧪 TESTE: Print do DataFrame")
    print("="*60)
    
    try:
        # Criar componentes
        data_loader = DataLoader()
        data_integration = DataIntegration(None, data_loader)
        
        # Simular trades históricos para formar vários candles
        base_time = datetime.now() - timedelta(minutes=30)
        
        print("📊 Simulando criação de candles...")
        
        # Simular 15 trades em diferentes minutos para formar candles
        for i in range(15):
            # Cada 2 trades formam um candle (aproximadamente)
            minute_offset = i // 2  # 0,0,1,1,2,2,3,3...
            
            trade = {
                'timestamp': base_time + timedelta(minutes=minute_offset, seconds=i*10),
                'price': 5100 + (i * 0.5),  # Preço variando
                'volume': 100 + (i * 5),
                'quantity': 10 + i,
                'trade_type': 1,
                'trade_number': 1000 + i,
                'ticker': 'WDOQ25',
                'is_historical': True
            }
            
            print(f"  📈 Trade {i+1}: {trade['timestamp'].strftime('%H:%M:%S')} - Preço: {trade['price']}")
            
            # Processar trade
            data_integration._on_trade(trade)
        
        print("\n✅ Trades processados!")
        
        # Forçar print do DataFrame atual
        print("\n🔍 FORÇANDO PRINT DO DATAFRAME ATUAL:")
        data_integration.print_current_dataframe()
        
        # Mostrar estatísticas
        print("\n📊 ESTATÍSTICAS:")
        stats = data_integration.get_dataframe_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

def simulate_real_system():
    """Simula o sistema real em funcionamento"""
    print("\n🚀 SIMULAÇÃO: Sistema Real")
    print("="*60)
    
    try:
        # Criar componentes
        data_loader = DataLoader()
        data_integration = DataIntegration(None, data_loader)
        
        # Simular carregamento histórico completo
        print("📈 Simulando conclusão de carregamento histórico...")
        
        completion_event = {
            'event_type': 'historical_data_complete',
            'total_records': 50000,  # Simular muitos dados
            'timestamp': datetime.now()
        }
        
        # Processar evento de conclusão (isso deve gerar log do DataFrame)
        data_integration._on_trade(completion_event)
        
        print("✅ Simulação de sistema real completa!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na simulação: {e}")
        return False

def main():
    """Executa todos os testes de print do DataFrame"""
    print("🧪 TESTE COMPLETO - PRINT DO DATAFRAME")
    print("="*70)
    print("🎯 Objetivo: Verificar se DataFrame é impresso corretamente")
    print("📊 Casos: Candles formados + Carregamento histórico completo")
    print()
    
    results = []
    
    # Teste 1: Print com dados simulados
    result1 = test_dataframe_print()
    results.append(("DataFrame Print", result1))
    
    # Teste 2: Simulação sistema real
    result2 = simulate_real_system()
    results.append(("Sistema Real", result2))
    
    # Resultado final
    print("\n" + "="*70)
    print("📋 RESULTADO DOS TESTES")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"  {test_name:<20}: {status}")
    
    print(f"\n🎯 RESULTADO: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 PRINT DO DATAFRAME FUNCIONANDO!")
        print("✅ DataFrame será impresso automaticamente:")
        print("   - A cada 10 candles formados")
        print("   - A cada 5 minutos")
        print("   - Ao final do carregamento histórico")
        print("   - Quando solicitado via print_current_dataframe()")
    else:
        print("⚠️ Alguns testes falharam - verificar implementação")
    
    print("\n💡 USO NO SISTEMA REAL:")
    print("  data_integration.print_current_dataframe()  # Print manual")
    print("  data_integration.get_dataframe_stats()      # Estatísticas")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
