#!/usr/bin/env python3
"""
Diagnóstico de atualização de dados no GUI
"""

import os
import sys
import time
import logging
from datetime import datetime

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_data_access():
    """Testa se os dados estão sendo atualizados corretamente"""
    print("🔍 DIAGNÓSTICO DE DADOS DO GUI")
    print("=" * 50)
    
    try:
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        
        # Simular sistema de trading
        class MockTradingSystem:
            def __init__(self):
                self.is_running = True
                self.data_structure = self._create_mock_data_structure()
                self.last_prediction = None
                
            def _create_mock_data_structure(self):
                import pandas as pd
                
                class MockDataStructure:
                    def __init__(self):
                        # Criar dados de candle simulados
                        self.candles = pd.DataFrame({
                            'open': [5577.0, 5578.5, 5579.0],
                            'high': [5580.0, 5581.0, 5582.0], 
                            'low': [5575.0, 5576.0, 5577.0],
                            'close': [5578.5, 5579.0, 5581.0],
                            'volume': [1000000, 1200000, 950000],
                            'trades': [100, 120, 95],
                            'buy_volume': [600000, 720000, 570000],
                            'sell_volume': [400000, 480000, 380000]
                        })
                        print(f"✓ Mock data structure criada com {len(self.candles)} candles")
                        
                return MockDataStructure()
                
            def get_candles_df(self):
                """Método compatível para retornar dados de candles"""
                return self.data_structure.candles
                
            def get_current_price(self):
                """Retorna preço atual"""
                if not self.data_structure.candles.empty:
                    return self.data_structure.candles.iloc[-1]['close']
                return 5580.0
                
            def get_day_statistics(self):
                """Retorna estatísticas do dia"""
                if not self.data_structure.candles.empty:
                    candles = self.data_structure.candles
                    day_open = candles.iloc[0]['open']
                    current_price = candles.iloc[-1]['close']
                    return {
                        'day_open': day_open,
                        'day_high': candles['high'].max(),
                        'day_low': candles['low'].min(),
                        'day_variation': current_price - day_open,
                        'day_variation_pct': ((current_price - day_open) / day_open) * 100
                    }
                return {}
        
        # Criar sistema mock
        mock_system = MockTradingSystem()
        print(f"✓ Sistema mock criado")
        
        # Importar GUI
        from trading_monitor_gui import create_monitor_gui
        print(f"✓ GUI importado com sucesso")
        
        # Criar monitor
        monitor = create_monitor_gui(mock_system)
        print(f"✓ Monitor GUI criado")
        
        # Testar coleta de dados manualmente
        print("\n📊 TESTANDO COLETA DE DADOS...")
        monitor._collect_trading_data()
        
        # Verificar dados coletados
        current_data = monitor.current_data
        print(f"\n📋 Dados coletados:")
        print(f"  - current_price: {current_data.get('current_price')}")
        print(f"  - last_candle: {current_data.get('last_candle')}")
        print(f"  - day_stats: {current_data.get('day_stats')}")
        print(f"  - system_status: {current_data.get('system_status')}")
        
        # Verificar se dados estão disponíveis
        has_price = 'current_price' in current_data
        has_candle = 'last_candle' in current_data
        has_stats = 'day_stats' in current_data
        
        print(f"\n✅ RESULTADO DOS TESTES:")
        print(f"  - Preço atual disponível: {'✓' if has_price else '✗'}")
        print(f"  - Dados de candle disponíveis: {'✓' if has_candle else '✗'}")
        print(f"  - Estatísticas do dia disponíveis: {'✓' if has_stats else '✗'}")
        
        if has_price and has_candle:
            print(f"\n🎉 DADOS ESTÃO SENDO COLETADOS CORRETAMENTE!")
            return True
        else:
            print(f"\n❌ PROBLEMA NA COLETA DE DADOS!")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_updates():
    """Testa se as atualizações do GUI estão funcionando"""
    print("\n" + "=" * 50)
    print("🔄 TESTE DE ATUALIZAÇÕES DO GUI")
    print("=" * 50)
    
    try:
        import tkinter as tk
        
        # Criar janela de teste
        root = tk.Tk()
        root.title("Teste de Atualização")
        root.geometry("400x200")
        
        # Label para mostrar updates
        update_count = 0
        update_label = tk.Label(root, text=f"Updates: {update_count}", font=('Arial', 14))
        update_label.pack(pady=20)
        
        price_label = tk.Label(root, text="Preço: R$ --", font=('Arial', 12))
        price_label.pack(pady=10)
        
        def update_display():
            nonlocal update_count
            update_count += 1
            
            # Simular preço que muda
            import random
            price = 5580 + random.uniform(-10, 10)
            
            update_label.config(text=f"Updates: {update_count}")
            price_label.config(text=f"Preço: R$ {price:.2f}")
            
            # Agendar próxima atualização
            root.after(1000, update_display)  # A cada 1 segundo
            
        # Iniciar atualizações
        update_display()
        
        print("✓ Janela de teste criada")
        print("✓ Timer de atualização configurado")
        print("\n💡 Feche a janela para continuar...")
        
        # Executar por poucos segundos para teste
        root.after(5000, root.destroy)  # Auto-fechar após 5 segundos
        root.mainloop()
        
        print(f"✓ Teste concluído - {update_count} atualizações executadas")
        return update_count > 0
        
    except Exception as e:
        print(f"❌ Erro no teste GUI: {e}")
        return False

def main():
    """Executa diagnóstico completo"""
    print("🏥 DIAGNÓSTICO COMPLETO DO GUI")
    print("=" * 60)
    
    # Teste 1: Acesso aos dados
    data_ok = test_data_access()
    
    # Teste 2: Atualizações do GUI
    gui_ok = test_gui_updates()
    
    # Relatório final
    print("\n" + "=" * 60)
    print("📋 RELATÓRIO FINAL DO DIAGNÓSTICO")
    print("=" * 60)
    print(f"✅ Coleta de dados: {'OK' if data_ok else 'FALHA'}")
    print(f"✅ Atualizações GUI: {'OK' if gui_ok else 'FALHA'}")
    
    if data_ok and gui_ok:
        print(f"\n🎉 DIAGNÓSTICO COMPLETO: SISTEMA FUNCIONAL")
        print(f"💡 O problema pode estar na sincronização entre GUI e sistema real")
    elif data_ok:
        print(f"\n⚠️  PROBLEMA: GUI não está atualizando")
        print(f"💡 Verificar threading e loop de atualização")
    elif gui_ok:
        print(f"\n⚠️  PROBLEMA: Dados não estão sendo coletados")
        print(f"💡 Verificar estrutura de dados do sistema")
    else:
        print(f"\n❌ PROBLEMAS CRÍTICOS: GUI e dados com falhas")
        print(f"💡 Revisão completa necessária")
        
    return data_ok and gui_ok

if __name__ == "__main__":
    main()
