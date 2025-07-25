#!/usr/bin/env python3
"""
Teste final para verificar processamento correto de dados históricos
Verifica:
1. Todos os trades históricos são convertidos em candles
2. Candles são sincronizados corretamente 
3. Não há erro de validação de dados
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import time

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trading_system import TradingSystem

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_historical_candles():
    """Testa processamento de candles históricos"""
    
    print("\n" + "="*80)
    print("TESTE DE PROCESSAMENTO DE CANDLES HISTÓRICOS")
    print("="*80 + "\n")
    
    # Definir ambiente
    os.environ['TRADING_ENV'] = 'development'
    
    # Configuração
    config = {
        'dll_path': r'C:\Profit\ProfitDLL64.dll',
        'username': 'MTBR118A',
        'password': 'Jac118A',
        'ticker': 'WDOZ25',
        'historical_days': 1,
        'use_gui': False,
        'models_dir': 'models/',
        'ml_interval': 60,
        'features_config': {
            'technical_indicators': True,
            'ml_features': True
        }
    }
    
    print("1. CRIANDO SISTEMA...")
    system = TradingSystem(config)
    
    print("\n2. INICIALIZANDO SISTEMA...")
    if not system.initialize():
        print("❌ Falha na inicialização")
        return False
    
    print("\n3. INICIANDO SISTEMA...")
    start_time = datetime.now()
    
    # Iniciar em thread separada
    import threading
    system_thread = threading.Thread(target=system.start, daemon=True)
    system_thread.start()
    
    # Monitorar carregamento
    print("\n4. MONITORANDO CARREGAMENTO DE DADOS HISTÓRICOS...")
    
    # Aguardar dados históricos
    if not system.historical_data_ready.wait(timeout=60):
        print("❌ Timeout aguardando dados históricos")
        return False
    
    print("\n✅ Dados históricos carregados!")
    
    # Aguardar processamento adicional
    print("\n5. AGUARDANDO PROCESSAMENTO COMPLETO...")
    time.sleep(5)
    
    # Verificar resultados
    print("\n6. VERIFICANDO RESULTADOS:")
    
    # Verificar connection_manager
    if hasattr(system.connection, '_historical_data_count'):
        trades_count = system.connection._historical_data_count
        print(f"   - Trades históricos recebidos: {trades_count}")
    
    # Verificar data_integration
    if system.data_integration:
        if hasattr(system.data_integration, 'candles_1min'):
            candles_count_integration = len(system.data_integration.candles_1min)
            print(f"   - Candles em data_integration: {candles_count_integration}")
            
            if candles_count_integration > 0:
                first_candle = system.data_integration.candles_1min.index.min()
                last_candle = system.data_integration.candles_1min.index.max()
                print(f"   - Período: {first_candle} até {last_candle}")
                print(f"   - Duração: {last_candle - first_candle}")
    
    # Verificar data_loader
    if system.data_loader:
        if hasattr(system.data_loader, 'candles_df'):
            candles_count_loader = len(system.data_loader.candles_df)
            print(f"   - Candles em data_loader: {candles_count_loader}")
    
    # Verificar data_structure
    if system.data_structure:
        candles_count_structure = len(system.data_structure.candles)
        print(f"   - Candles em data_structure: {candles_count_structure}")
        
        if candles_count_structure > 0:
            # Verificar sell_volume
            if 'sell_volume' in system.data_structure.candles.columns:
                sell_vol_zeros = (system.data_structure.candles['sell_volume'] == 0).sum()
                sell_vol_pct = sell_vol_zeros / len(system.data_structure.candles) * 100
                print(f"   - sell_volume zeros: {sell_vol_zeros}/{len(system.data_structure.candles)} ({sell_vol_pct:.1f}%)")
    
    # Verificar features
    print("\n7. VERIFICANDO CÁLCULO DE FEATURES:")
    time.sleep(5)  # Aguardar cálculo
    
    if hasattr(system.data_structure, 'features') and system.data_structure.features is not None:
        features_count = len(system.data_structure.features.columns)
        print(f"   ✅ Features calculadas: {features_count}")
    else:
        print("   ⚠️ Nenhuma feature calculada ainda")
    
    # Análise final
    print("\n8. ANÁLISE:")
    if trades_count > 0 and candles_count_structure > 0:
        ratio = trades_count / candles_count_structure
        print(f"   - Média de trades por candle: {ratio:.1f}")
        
        expected_candles = trades_count / 60  # Assumindo ~60 trades por minuto
        efficiency = candles_count_structure / expected_candles * 100
        print(f"   - Eficiência de conversão: {efficiency:.1f}%")
        
        if efficiency < 50:
            print("   ⚠️ Baixa eficiência - verificar processamento de candles")
        else:
            print("   ✅ Boa eficiência de conversão")
    
    # Parar sistema
    print("\n9. PARANDO SISTEMA...")
    system.stop()
    
    print("\n" + "="*80)
    print("✅ TESTE CONCLUÍDO!")
    print("="*80)
    
    return True

if __name__ == "__main__":
    try:
        test_historical_candles()
        
        print("\n💡 PRÓXIMOS PASSOS:")
        print("1. Se poucos candles foram criados, verificar:")
        print("   - Os trades históricos estão distribuídos em poucos minutos?")
        print("   - O processamento de candles está agrupando corretamente?")
        print("2. Se sell_volume está 100% zero:")
        print("   - Verificar se trade_type está sendo processado")
        print("   - Verificar mapeamento buy/sell no connection_manager")
        
    except Exception as e:
        print(f"\n❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()