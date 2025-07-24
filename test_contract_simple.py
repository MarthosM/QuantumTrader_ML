"""Teste simplificado da lógica de contratos WDO"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.connection_manager import ConnectionManager

def test_contract():
    print("=" * 60)
    print("TESTE DA LOGICA CORRIGIDA DE CONTRATOS WDO")
    print("=" * 60)
    
    dll_path = "dummy"
    cm = ConnectionManager(dll_path)
    
    # Teste para hoje (24/07/2025)
    today = datetime.now()
    contract = cm._get_current_wdo_contract(today)
    
    print(f"Data atual: {today.strftime('%d/%m/%Y')}")
    print(f"Contrato detectado: {contract}")
    print(f"Esperado para 24/07 (apos dia 15): WDOQ25")
    
    if contract == "WDOQ25":
        print("SUCESSO: Logica corrigida!")
    else:
        print("ERRO: Logica ainda incorreta!")
    
    # Testar variações
    variations = cm._get_smart_ticker_variations("WDO")
    print(f"\nVariacoes: {variations}")
    
    # Teste rápido para outros meses
    print(f"\nTestes rapidos:")
    
    test_dates = [
        (datetime(2025, 8, 1), "WDOU25"),   # Agosto usa setembro
        (datetime(2025, 9, 1), "WDOV25"),   # Setembro usa outubro
        (datetime(2025, 12, 1), "WDOF26"),  # Dezembro usa janeiro/26
    ]
    
    for test_date, expected in test_dates:
        result = cm._get_current_wdo_contract(test_date)
        status = "OK" if result == expected else "ERRO"
        print(f"  {test_date.strftime('%m/%Y')}: {result} ({status})")
    
    print(f"\nRegra aplicada:")
    print(f"  SEMPRE usa contrato do PROXIMO mes")
    print(f"  Julho usa agosto, agosto usa setembro, etc.")
    print("=" * 60)

if __name__ == "__main__":
    test_contract()