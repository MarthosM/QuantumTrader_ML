"""
Script para corrigir os callbacks no ConnectionManagerV4
Adiciona return 0 em todas as funções callback
"""

import os
import shutil
from datetime import datetime


def fix_callbacks():
    """Corrige os callbacks para retornar 0"""
    
    file_path = "src/connection_manager_v4.py"
    backup_path = f"src/connection_manager_v4_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    print(f"Aplicando correção em {file_path}")
    
    # Fazer backup
    print(f"Criando backup: {backup_path}")
    shutil.copy2(file_path, backup_path)
    
    # Ler arquivo
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Corrigir callbacks
    in_callback = False
    callback_names = ['state_callback', 'trade_callback', 'history_callback', 
                      'progress_callback', 'account_callback', 'order_callback',
                      'daily_callback', 'price_book_callback', 'offer_book_callback',
                      'tiny_book_callback']
    
    for i in range(len(lines)):
        line = lines[i]
        
        # Detectar início de callback
        for cb_name in callback_names:
            if f'def {cb_name}(' in line:
                in_callback = True
                print(f"Encontrado callback: {cb_name}")
                break
        
        # Se estamos em um callback e encontramos o except
        if in_callback and 'except Exception as e:' in line:
            # Verificar se já tem return antes do except
            has_return = False
            j = i - 1
            while j >= 0 and '    try:' not in lines[j]:
                if 'return' in lines[j]:
                    has_return = True
                    break
                j -= 1
            
            if not has_return:
                # Adicionar return 0 antes do except
                indent = '                '  # 16 espaços
                lines.insert(i, f'{indent}return 0\n')
                print(f"  Adicionado return 0 na linha {i}")
                i += 1
            
            # Adicionar return 0 depois do logger.error também
            if i + 1 < len(lines) and 'self.logger.error' in lines[i + 1]:
                indent = '                '  # 16 espaços
                lines.insert(i + 2, f'{indent}return 0\n')
                print(f"  Adicionado return 0 após erro na linha {i + 2}")
            
            in_callback = False
    
    # Salvar arquivo corrigido
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"\nCorreção aplicada com sucesso!")
    print(f"Backup salvo em: {backup_path}")
    
    return True


def main():
    print("CORRIGINDO CALLBACKS DO CONNECTIONMANAGERV4")
    print("="*60)
    print("Problema: Callbacks não retornam valor")
    print("Solução: Adicionar return 0 em todos os callbacks")
    print("="*60)
    
    if fix_callbacks():
        print("\n✅ CORREÇÃO APLICADA!")
        print("\nAgora execute novamente o teste:")
        print("python scripts/test_final_connection.py")
    else:
        print("\n❌ Erro ao aplicar correção")


if __name__ == "__main__":
    main()