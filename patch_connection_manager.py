"""
Patch temporário para connection_manager_v4.py
Remove o callback de conta que está causando segmentation fault
"""

import os
import shutil
from datetime import datetime

def patch_connection_manager():
    """Aplica patch no connection_manager_v4.py"""
    
    file_path = "src/connection_manager_v4.py"
    backup_path = f"src/connection_manager_v4_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    print(f"Fazendo backup: {backup_path}")
    shutil.copy2(file_path, backup_path)
    
    print("Lendo arquivo...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Procurar o callback de conta e comentá-lo
    print("Aplicando patch...")
    
    # Encontrar a linha com account_callback e adicionar return antes
    lines = content.split('\n')
    new_lines = []
    in_account_callback = False
    skip_count = 0
    
    for i, line in enumerate(lines):
        # Detectar início do account_callback
        if 'def account_callback(' in line and '@TAccountCallback' in lines[i-1]:
            print(f"Encontrado account_callback na linha {i}")
            # Adicionar comentário e return vazio
            new_lines.append(lines[i-1])  # @TAccountCallback
            new_lines.append(line)  # def account_callback
            new_lines.append("            # PATCH: Desabilitado temporariamente devido a segfault")
            new_lines.append("            return 0")
            in_account_callback = True
            skip_count = 0
            continue
            
        # Se estamos dentro do account_callback, pular até o próximo método
        if in_account_callback:
            # Detectar fim do método (próxima definição ou decorador)
            if (line.strip().startswith('def ') or 
                line.strip().startswith('@') or 
                (not line.strip() and skip_count > 10)):
                in_account_callback = False
                new_lines.append(line)
            else:
                skip_count += 1
                # Comentar a linha original
                if line.strip():
                    new_lines.append("        # " + line)
                else:
                    new_lines.append(line)
        else:
            new_lines.append(line)
    
    # Salvar arquivo patchado
    print("Salvando arquivo patchado...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    
    print("[OK] Patch aplicado com sucesso!")
    print(f"Backup salvo em: {backup_path}")
    
def restore_connection_manager():
    """Restaura o arquivo original do último backup"""
    import glob
    
    backups = sorted(glob.glob("src/connection_manager_v4_backup_*.py"))
    if not backups:
        print("Nenhum backup encontrado!")
        return
        
    latest_backup = backups[-1]
    print(f"Restaurando de: {latest_backup}")
    
    shutil.copy2(latest_backup, "src/connection_manager_v4.py")
    print("[OK] Arquivo restaurado!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore_connection_manager()
    else:
        patch_connection_manager()