"""
Script para comparar estrutura de callbacks entre scripts
Identifica diferenças entre o que funciona e o que não funciona
"""

import os
import re
from pathlib import Path

def extract_dll_init(file_path):
    """Extrai a chamada DLLInitializeLogin de um arquivo"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Encontrar DLLInitializeLogin
        pattern = r'self\.dll\.DLLInitializeLogin\s*\(([\s\S]*?)\)'
        matches = re.findall(pattern, content)
        
        if matches:
            # Limpar e formatar
            call = matches[0]
            params = [p.strip() for p in call.split(',')]
            return params
            
    except Exception as e:
        print(f"Erro ao ler {file_path}: {e}")
        
    return None

def main():
    print("\n" + "="*60)
    print("COMPARAÇÃO DE CALLBACKS - PROFITDLL")
    print("="*60)
    
    scripts = {
        'book_collector_continuous.py': 'FUNCIONA',
        'production_debug.py': 'NÃO FUNCIONA',
        'production_complete.py': 'TALVEZ FUNCIONE',
        'start_production_simple_ml.py': 'CONECTA MAS SEM DADOS'
    }
    
    results = {}
    
    for script, status in scripts.items():
        if os.path.exists(script):
            params = extract_dll_init(script)
            if params:
                results[script] = {
                    'status': status,
                    'params': params
                }
                
    # Comparar estruturas
    print("\nESTRUTURA DE CALLBACKS:")
    print("-"*60)
    
    # Ordem dos parâmetros esperada
    param_names = [
        'key', 'user', 'pwd',
        'stateCallback',
        'historyCallback',
        'orderChangeCallback',
        'accountCallback', 
        'accountInfoCallback',
        'dailyCallback',
        'priceBookCallback',
        'offerBookCallback',
        'historyTradeCallback',
        'progressCallback',
        'tinyBookCallback'
    ]
    
    for script, data in results.items():
        print(f"\n{script} ({data['status']}):")
        print("-"*40)
        
        for i, param in enumerate(data['params']):
            if i < len(param_names):
                param_name = param_names[i]
            else:
                param_name = f'param_{i}'
                
            if param.strip() != 'None':
                print(f"  {param_name:20} = {param}")
                
    # Identificar diferenças
    print("\n" + "="*60)
    print("ANÁLISE DE DIFERENÇAS:")
    print("-"*60)
    
    working = None
    not_working = []
    
    for script, data in results.items():
        if 'FUNCIONA' in data['status']:
            working = (script, data['params'])
        else:
            not_working.append((script, data['params']))
            
    if working and not_working:
        working_script, working_params = working
        
        print(f"\nScript funcional: {working_script}")
        print("\nDiferenças encontradas:")
        
        for script, params in not_working:
            print(f"\n{script}:")
            
            # Comparar parâmetros
            for i in range(max(len(working_params), len(params))):
                if i < len(working_params) and i < len(params):
                    if working_params[i] != params[i]:
                        print(f"  Posição {i} ({param_names[i] if i < len(param_names) else 'unknown'}):")
                        print(f"    Funcional: {working_params[i]}")
                        print(f"    Atual:     {params[i]}")
                        
    # Recomendações
    print("\n" + "="*60)
    print("RECOMENDAÇÕES:")
    print("-"*60)
    
    if working:
        print(f"\n1. Use a estrutura EXATA de {working[0]}")
        print("2. Callbacks essenciais que devem estar presentes:")
        print("   - stateCallback (posição 3)")
        print("   - tinyBookCallback (posição 13)")
        print("   - dailyCallback (posição 8)")
        print("   - priceBookCallback (posição 9)")
        print("3. Crie TODOS os callbacks ANTES do DLLInitializeLogin")
        print("4. Use a mesma ordem de parâmetros")

if __name__ == "__main__":
    main()