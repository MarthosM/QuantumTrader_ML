#!/usr/bin/env python3
"""
Teste para verificar sincronização de dados históricos
Garante que features só são calculadas após dados completos
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trading_system import TradingSystemV2
from connection_manager import ConnectionManager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_historical_data_sync():
    """Testa sincronização de dados históricos"""
    
    print("\n=== TESTE DE SINCRONIZAÇÃO DE DADOS HISTÓRICOS ===\n")
    
    # Configuração
    config = {
        'dll_path': r'C:\Profit\ProfitDLL64.dll',
        'username': 'MTBR118A',
        'password': 'Jac118A',
        'ticker': 'WDOZ25',  # Contrato atual
        'historical_days': 1,
        'use_gui': False,
        'models_dir': 'models/',
        'features_config': {
            'technical_indicators': True,
            'ml_features': True
        }
    }
    
    # Criar sistema
    print("1. Criando sistema de trading...")
    system = TradingSystemV2(config)
    
    # Inicializar
    print("\n2. Inicializando sistema...")
    if not system.initialize():
        print("❌ Falha na inicialização")
        return False
    
    # Verificar estado inicial
    print(f"\n3. Estado inicial:")
    print(f"   - Dados históricos carregados: {system.historical_data_loaded}")
    print(f"   - Evento historical_data_ready: {system.historical_data_ready.is_set()}")
    print(f"   - Candles no data_structure: {len(system.data_structure.candles) if system.data_structure else 0}")
    
    # Iniciar sistema
    print("\n4. Iniciando sistema (carregando dados)...")
    start_time = datetime.now()
    
    if not system.start():
        print("❌ Falha ao iniciar sistema")
        return False
    
    load_time = (datetime.now() - start_time).total_seconds()
    
    # Verificar estado após carregamento
    print(f"\n5. Estado após carregamento (tempo: {load_time:.2f}s):")
    print(f"   - Dados históricos carregados: {system.historical_data_loaded}")
    print(f"   - Evento historical_data_ready: {system.historical_data_ready.is_set()}")
    print(f"   - Candles no data_structure: {len(system.data_structure.candles) if system.data_structure else 0}")
    
    # Verificar features
    print("\n6. Verificando features calculadas:")
    if hasattr(system.data_structure, 'features') and system.data_structure.features is not None:
        print(f"   - Features disponíveis: {len(system.data_structure.features.columns)}")
        print(f"   - Primeiras 5 features: {list(system.data_structure.features.columns[:5])}")
    else:
        print("   - Nenhuma feature calculada ainda")
    
    # Aguardar um pouco para ver logs
    print("\n7. Aguardando processamento ML (10 segundos)...")
    import time
    time.sleep(10)
    
    # Verificar novamente
    print("\n8. Estado final:")
    print(f"   - Candles: {len(system.data_structure.candles) if system.data_structure else 0}")
    print(f"   - Features: {len(system.data_structure.features.columns) if hasattr(system.data_structure, 'features') and system.data_structure.features is not None else 0}")
    
    # Parar sistema
    print("\n9. Parando sistema...")
    system.stop()
    
    print("\n✅ Teste concluído!")
    return True

if __name__ == "__main__":
    try:
        # Definir ambiente
        os.environ['TRADING_ENV'] = 'development'  # Para permitir testes
        
        test_historical_data_sync()
        
    except Exception as e:
        print(f"\n❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()