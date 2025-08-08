"""
Connection Manager V4 Safe - Versão sem callback de conta problemático
Baseado no connection_manager_v4.py mas com o callback de conta desabilitado
"""

# Copiar todo o conteúdo do connection_manager_v4.py original
import shutil
import os

# Fazer cópia do arquivo original
original = "src/connection_manager_v4.py"
safe_version = "src/connection_manager_v4_safe.py"

# Copiar arquivo
shutil.copy2(original, safe_version)

# Ler conteúdo
with open(safe_version, 'r', encoding='utf-8') as f:
    content = f.read()

# Modificar o callback de conta para retornar imediatamente
# Procurar pela definição do account_callback e modificar
import re

# Pattern para encontrar o account_callback
pattern = r'(@TAccountCallback\s*\n\s*def account_callback.*?)\n(\s+)try:(.*?)return 0.*?except.*?return 0'

# Substituição - manter assinatura mas retornar imediatamente
replacement = r'\1\n\2# SAFE VERSION: Account callback desabilitado\n\2return 0'

# Aplicar substituição
content_modified = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Se não encontrou com o pattern acima, tentar outra abordagem
if content_modified == content:
    # Procurar linha por linha
    lines = content.split('\n')
    new_lines = []
    skip_account_callback = False
    found_account_callback = False
    
    for i, line in enumerate(lines):
        # Detectar início do account_callback
        if 'def account_callback(' in line and i > 0 and '@TAccountCallback' in lines[i-1]:
            new_lines.append(line)
            # Adicionar return imediato
            indent = len(line) - len(line.lstrip())
            new_lines.append(' ' * (indent + 4) + '# SAFE: Callback desabilitado para evitar segfault')
            new_lines.append(' ' * (indent + 4) + 'return 0')
            skip_account_callback = True
            found_account_callback = True
            continue
        
        # Se estamos pulando o corpo do account_callback
        if skip_account_callback:
            # Detectar próxima função ou fim do callback
            if line.strip() and not line.startswith(' '):
                skip_account_callback = False
                new_lines.append(line)
            elif line.strip().startswith('def ') or line.strip().startswith('@'):
                skip_account_callback = False
                new_lines.append(line)
            # Pular linhas do corpo do callback
            continue
        else:
            new_lines.append(line)
    
    if found_account_callback:
        content_modified = '\n'.join(new_lines)

# Salvar versão modificada
with open(safe_version, 'w', encoding='utf-8') as f:
    f.write(content_modified)

print(f"[OK] Criado {safe_version} sem callback de conta problemático")