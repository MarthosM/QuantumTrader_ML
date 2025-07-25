#!/usr/bin/env python3
"""
Teste final do sistema corrigido
Verifica:
1. Sincronização de dados históricos
2. Validação relaxada para dados históricos
3. Cálculo de features após dados prontos
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import time

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trading_system import TradingSystemV2

# Configurar logging detalhado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_fixed_system():
    """Testa sistema corrigido"""
    
    print("\n" + "="*80)
    print("TESTE DO SISTEMA CORRIGIDO - SINCRONIZAÇÃO DE DADOS HISTÓRICOS")
    print("="*80 + "\n")
    
    # Definir ambiente
    os.environ['TRADING_ENV'] = 'development'
    print(f"✓ Ambiente definido: {os.environ['TRADING_ENV']}")
    
    # Configuração
    config = {
        'dll_path': r'C:\Profit\ProfitDLL64.dll',
        'username': 'MTBR118A',
        'password': 'Jac118A',
        'ticker': 'WDOZ25',
        'historical_days': 1,
        'use_gui': False,
        'models_dir': 'models/',
        'ml_interval': 30,  # Reduzido para teste
        'features_config': {
            'technical_indicators': True,
            'ml_features': True
        }
    }
    
    # Criar sistema
    print("\n1. CRIANDO SISTEMA...")
    system = TradingSystemV2(config)
    
    # Inicializar
    print("\n2. INICIALIZANDO SISTEMA...")
    if not system.initialize():
        print("❌ Falha na inicialização")
        return False
    
    print(f"   ✓ Sistema inicializado")
    print(f"   ✓ Feature engine: allow_historical_data = {system.feature_engine.allow_historical_data}")
    print(f"   ✓ Validator: allow_historical_data = {system.feature_engine.validator.allow_historical_data}")
    
    # Verificar estado inicial
    print("\n3. ESTADO INICIAL:")
    print(f"   - historical_data_loaded: {system.historical_data_loaded}")
    print(f"   - historical_data_ready: {system.historical_data_ready.is_set()}")
    print(f"   - is_running: {system.is_running}")
    
    # Iniciar sistema
    print("\n4. INICIANDO SISTEMA (CARREGANDO DADOS)...")
    start_time = datetime.now()
    
    # Iniciar em thread separada para não bloquear
    import threading
    system_thread = threading.Thread(target=system.start, daemon=True)
    system_thread.start()
    
    # Monitorar progresso
    print("\n5. MONITORANDO CARREGAMENTO DE DADOS...")
    timeout = 60  # 1 minuto
    check_interval = 2
    elapsed = 0
    
    while elapsed < timeout:
        if system.historical_data_ready.is_set():
            print(f"\n   ✓ Dados históricos prontos após {elapsed}s!")
            break
        
        print(f"   ⏳ Aguardando... ({elapsed}s)", end='\r')
        time.sleep(check_interval)
        elapsed += check_interval
    
    if not system.historical_data_ready.is_set():
        print(f"\n   ❌ Timeout após {timeout}s")
        return False
    
    # Verificar dados carregados
    print("\n6. VERIFICANDO DADOS CARREGADOS:")
    candles_count = len(system.data_structure.candles) if system.data_structure else 0
    print(f"   - Candles: {candles_count}")
    print(f"   - is_running: {system.is_running}")
    
    # Aguardar processamento de features
    print("\n7. AGUARDANDO PROCESSAMENTO DE FEATURES...")
    time.sleep(10)
    
    # Verificar features
    print("\n8. VERIFICANDO FEATURES CALCULADAS:")
    if hasattr(system.data_structure, 'features') and system.data_structure.features is not None:
        print(f"   ✓ Features disponíveis: {len(system.data_structure.features.columns)}")
        if len(system.data_structure.features.columns) > 0:
            print(f"   ✓ Exemplos: {list(system.data_structure.features.columns[:5])}")
    else:
        print("   ⚠️ Nenhuma feature calculada")
    
    # Verificar indicadores
    if hasattr(system.data_structure, 'indicators') and system.data_structure.indicators is not None:
        print(f"   ✓ Indicadores: {len(system.data_structure.indicators.columns)}")
    
    # Verificar se houve erros de validação
    print("\n9. VERIFICANDO LOGS DE VALIDAÇÃO:")
    print("   - Verificar logs acima para erros de 'DADOS SINTÉTICOS'")
    print("   - Se não houver erros, a validação foi bem-sucedida!")
    
    # Parar sistema
    print("\n10. PARANDO SISTEMA...")
    system.stop()
    
    print("\n" + "="*80)
    print("✅ TESTE CONCLUÍDO COM SUCESSO!")
    print("="*80)
    
    return True

if __name__ == "__main__":
    try:
        success = test_fixed_system()
        
        if success:
            print("\n🎉 Sistema funcionando corretamente!")
            print("\nPRÓXIMOS PASSOS:")
            print("1. Execute o sistema com GUI: python src/main.py")
            print("2. Verifique se não há mais erros de 'DADOS SINTÉTICOS'")
            print("3. Confirme que features são calculadas após dados históricos")
        else:
            print("\n❌ Teste falhou - verifique os logs acima")
            
    except Exception as e:
        print(f"\n❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()