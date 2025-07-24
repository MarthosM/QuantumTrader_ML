"""
Teste para validar a lógica corrigida de contratos WDO
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.connection_manager import ConnectionManager

def test_wdo_contract_logic():
    print("=" * 80)
    print("TESTE DA LOGICA CORRIGIDA DE CONTRATOS WDO")
    print("=" * 80)
    
    # Criar ConnectionManager para testar a lógica
    dll_path = "dummy_path"  # Não precisamos da DLL para testar a lógica
    cm = ConnectionManager(dll_path)
    
    # Mapear códigos de mês esperados
    expected_codes = {
        1: 'F',   # Janeiro
        2: 'G',   # Fevereiro  
        3: 'H',   # Março
        4: 'J',   # Abril
        5: 'K',   # Maio
        6: 'M',   # Junho
        7: 'N',   # Julho
        8: 'Q',   # Agosto
        9: 'U',   # Setembro
        10: 'V',  # Outubro
        11: 'X',  # Novembro
        12: 'Z'   # Dezembro
    }
    
    print("1. TESTANDO TODOS OS MESES DO ANO 2025:")
    print("-" * 50)
    
    for month in range(1, 13):
        # Testar para diferentes dias do mês
        for day in [1, 15, 28]:
            try:
                test_date = datetime(2025, month, day)
                contract = cm._get_current_wdo_contract(test_date)
                expected_contract = f"WDO{expected_codes[month]}25"
                
                status = "✅" if contract == expected_contract else "❌"
                print(f"   {status} {test_date.strftime('%d/%m/2025')}: {contract} (esperado: {expected_contract})")
                
                if contract != expected_contract:
                    print(f"      ERRO: Contrato incorreto!")
                    
            except ValueError:
                # Dia inválido para o mês (ex: 30 de fevereiro)
                continue
    
    print(f"\n2. TESTE ESPECÍFICO PARA JULHO 2025 (HOJE):")
    print("-" * 50)
    
    today = datetime.now()
    contract = cm._get_current_wdo_contract(today)
    expected_july = "WDON25"  # Julho = N
    
    print(f"   Data atual: {today.strftime('%d/%m/%Y')}")
    print(f"   Contrato detectado: {contract}")
    print(f"   Contrato esperado: {expected_july}")
    
    if contract == expected_july:
        print("   ✅ SUCESSO: Lógica corrigida funciona para julho!")
    else:
        print("   ❌ ERRO: Lógica ainda não está correta!")
    
    print(f"\n3. TESTANDO VARIAÇÕES DE TICKER:")
    print("-" * 50)
    
    variations = cm._get_smart_ticker_variations("WDO")
    print(f"   Variações geradas: {variations}")
    
    # Deve conter pelo menos o contrato correto
    if expected_july in variations:
        print("   ✅ SUCESSO: Contrato correto nas variações!")
    else:
        print("   ❌ ERRO: Contrato correto não encontrado nas variações!")
    
    print(f"\n4. RESUMO DA CORREÇÃO:")
    print("-" * 50)
    print("   ANTES: Usava próximo mês após dia 15")
    print("   AGORA: Sempre usa mês atual (regra correta)")
    print("   ")
    print("   Exemplos:")
    print("   - 24/07/2025 → WDON25 (Julho)")
    print("   - 01/08/2025 → WDOQ25 (Agosto)")
    print("   - 31/12/2025 → WDOZ25 (Dezembro)")
    
    print("\n" + "=" * 80)
    print("TESTE CONCLUÍDO")
    print("=" * 80)

if __name__ == "__main__":
    test_wdo_contract_logic()