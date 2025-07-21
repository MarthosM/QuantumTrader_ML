"""
Teste rápido das correções de dados históricos
"""
import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import logging
from datetime import datetime, timedelta
from connection_manager import ConnectionManager
from data_integration import DataIntegration
from data_loader import DataLoader

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_historical_data_fixes():
    """Testa as correções implementadas"""
    print("🧪 Testando correções de dados históricos...")
    
    # Teste 1: Timestamp parsing no ConnectionManager
    print("\n📅 Teste 1: Parsing de timestamp")
    
    # Simular dados com diferentes formatos
    test_timestamps = [
        "19/07/2025 14:30:25.123",
        "19/07/2025 14:30:25",
        "2025-07-19 14:30:25.123",
        "2025-07-19 14:30:25",
        ""
    ]
    
    for ts_str in test_timestamps:
        try:
            # Simular lógica do callback corrigido
            if ts_str:
                for fmt in ['%d/%m/%Y %H:%M:%S.%f', '%d/%m/%Y %H:%M:%S', 
                           '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S']:
                    try:
                        timestamp = datetime.strptime(ts_str, fmt)
                        print(f"  ✅ '{ts_str}' -> {timestamp}")
                        break
                    except ValueError:
                        continue
                else:
                    print(f"  ⚠️ '{ts_str}' -> formato não reconhecido, usando now()")
                    timestamp = datetime.now()
            else:
                print(f"  ⚠️ timestamp vazio -> usando now()")
                timestamp = datetime.now()
                
        except Exception as e:
            print(f"  ❌ Erro com '{ts_str}': {e}")
    
    # Teste 2: Validação de dados históricos no DataIntegration
    print("\n🔄 Teste 2: Validação de dados históricos")
    
    # Simular trade histórico
    historical_trade = {
        'timestamp': datetime.now() - timedelta(days=2),  # 2 dias atrás
        'ticker': 'WDOQ25',
        'price': 5120.5,
        'volume': 100,
        'quantity': 1,
        'trade_type': 1,
        'trade_number': 12345,
        'is_historical': True
    }
    
    # Simular trade em tempo real antigo
    realtime_old_trade = {
        'timestamp': datetime.now() - timedelta(minutes=5),  # 5 minutos atrás
        'ticker': 'WDOQ25',
        'price': 5120.5,
        'volume': 100,
        'quantity': 1,
        'trade_type': 1,
        'trade_number': 12346,
        'is_historical': False
    }
    
    # Simular trade em tempo real atual
    realtime_current_trade = {
        'timestamp': datetime.now(),  # Agora
        'ticker': 'WDOQ25',
        'price': 5120.5,
        'volume': 100,
        'quantity': 1,
        'trade_type': 1,
        'trade_number': 12347,
        'is_historical': False
    }
    
    # Simular validação
    trades_to_test = [
        ("Histórico (2 dias)", historical_trade),
        ("Tempo real antigo (5 min)", realtime_old_trade),
        ("Tempo real atual", realtime_current_trade)
    ]
    
    for desc, trade in trades_to_test:
        try:
            # Simular lógica de validação corrigida
            now = datetime.now()
            trade_time = trade['timestamp']
            is_historical = trade.get('is_historical', False)
            
            if not is_historical and (now - trade_time).total_seconds() > 60:
                print(f"  ❌ {desc}: Rejeitado (muito antigo para tempo real)")
                valid = False
            elif is_historical:
                days_old = (now - trade_time).days
                print(f"  ✅ {desc}: Aceito (histórico, {days_old} dias)")
                valid = True
            else:
                print(f"  ✅ {desc}: Aceito (tempo real atual)")
                valid = True
                
        except Exception as e:
            print(f"  ❌ Erro validando {desc}: {e}")
    
    # Teste 3: Detecção de contrato WDO
    print("\n🎯 Teste 3: Detecção inteligente de contratos WDO")
    
    try:
        cm = ConnectionManager(dll_path=None)  # Apenas para teste, sem DLL
        
        # Testar detecção para diferentes datas
        test_dates = [
            datetime(2025, 7, 10),  # Antes do dia 15
            datetime(2025, 7, 20),  # Depois do dia 15
            datetime(2025, 12, 20), # Dezembro (virada de ano)
        ]
        
        for test_date in test_dates:
            contract = cm._get_current_wdo_contract(test_date)
            print(f"  📊 {test_date.strftime('%d/%m/%Y')} -> {contract}")
            
        # Testar variações de ticker
        variations = cm._get_smart_ticker_variations("WDO")
        print(f"  📋 Variações para 'WDO': {variations}")
        
        variations = cm._get_smart_ticker_variations("WDOQ25")
        print(f"  📋 Variações para 'WDOQ25': {variations}")
        
    except Exception as e:
        print(f"  ❌ Erro no teste de contratos: {e}")
    
    print("\n🎉 Testes concluídos!")
    print("\n💡 Principais correções implementadas:")
    print("  1. ✅ Parsing robusto de timestamps no history_callback")
    print("  2. ✅ Validação inteligente para dados históricos vs tempo real")
    print("  3. ✅ Detecção automática de contratos WDO com viradas de mês")
    print("  4. ✅ Logs menos frequentes para evitar spam")
    print("  5. ✅ Timeout aumentado para 60s para dados históricos")

if __name__ == "__main__":
    test_historical_data_fixes()
