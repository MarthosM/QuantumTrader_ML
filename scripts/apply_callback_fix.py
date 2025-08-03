"""
Script para aplicar a correção dos callbacks no profit_dll_structures.py
Muda todos os callbacks de retorno None para retorno c_int
"""

import os
import sys
import shutil
from datetime import datetime

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def apply_fix():
    """Aplica a correção nos callbacks"""
    
    file_path = "src/profit_dll_structures.py"
    backup_path = f"src/profit_dll_structures_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    print(f"📋 Aplicando correção em {file_path}")
    
    # Fazer backup
    print(f"💾 Criando backup: {backup_path}")
    shutil.copy2(file_path, backup_path)
    
    # Ler arquivo
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Aplicar correções
    replacements = [
        # Callbacks principais
        ("TStateCallback = WINFUNCTYPE(None, c_int, c_int)", 
         "TStateCallback = WINFUNCTYPE(c_int, c_int, c_int)"),
        
        ("TNewTradeCallback = WINFUNCTYPE(\n    None,", 
         "TNewTradeCallback = WINFUNCTYPE(\n    c_int,"),
        
        ("THistoryTradeCallback = WINFUNCTYPE(\n    None,", 
         "THistoryTradeCallback = WINFUNCTYPE(\n    c_int,"),
        
        ("TProgressCallback = WINFUNCTYPE(None, TAssetID, c_int)", 
         "TProgressCallback = WINFUNCTYPE(c_int, TAssetID, c_int)"),
        
        ("TAccountCallback = WINFUNCTYPE(\n    None,", 
         "TAccountCallback = WINFUNCTYPE(\n    c_int,"),
        
        ("TPriceBookCallback = WINFUNCTYPE(\n    None,", 
         "TPriceBookCallback = WINFUNCTYPE(\n    c_int,"),
        
        ("TOfferBookCallback = WINFUNCTYPE(\n    None,", 
         "TOfferBookCallback = WINFUNCTYPE(\n    c_int,"),
        
        ("TConnectorOrderCallback = WINFUNCTYPE(\n    None,", 
         "TConnectorOrderCallback = WINFUNCTYPE(\n    c_int,"),
        
        ("TConnectorAccountCallback = WINFUNCTYPE(\n    None,", 
         "TConnectorAccountCallback = WINFUNCTYPE(\n    c_int,"),
        
        ("TConnectorAssetPositionListCallback = WINFUNCTYPE(\n    None,", 
         "TConnectorAssetPositionListCallback = WINFUNCTYPE(\n    c_int,"),
        
        ("TConnectorEnumerateOrdersProc = WINFUNCTYPE(\n    None,", 
         "TConnectorEnumerateOrdersProc = WINFUNCTYPE(\n    c_int,"),
        
        ("TConnectorEnumerateAssetProc = WINFUNCTYPE(\n    None,", 
         "TConnectorEnumerateAssetProc = WINFUNCTYPE(\n    c_int,"),
        
        ("TConnectorTradeCallback = WINFUNCTYPE(\n    None,", 
         "TConnectorTradeCallback = WINFUNCTYPE(\n    c_int,"),
    ]
    
    # Aplicar cada substituição
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"✅ Corrigido: {old.split('=')[0].strip()}")
        else:
            print(f"⚠️  Não encontrado: {old.split('=')[0].strip()}")
    
    # Salvar arquivo corrigido
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ Correção aplicada com sucesso!")
    print(f"💾 Backup salvo em: {backup_path}")
    
    # Adicionar comentário no topo do arquivo
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Adicionar comentário após a docstring
    for i, line in enumerate(lines):
        if '"""' in line and i > 0:  # Segunda ocorrência de """
            lines.insert(i+1, "\n# CORREÇÃO APLICADA: Todos callbacks retornam c_int para evitar crashes\n")
            break
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("\n📝 Comentário adicionado ao arquivo")
    
    return True


def test_import():
    """Testa se a importação funciona após a correção"""
    print("\n🧪 Testando importação...")
    
    try:
        from src.profit_dll_structures import TStateCallback
        print("✅ Importação bem-sucedida!")
        print(f"   TStateCallback: {TStateCallback}")
        return True
    except Exception as e:
        print(f"❌ Erro na importação: {e}")
        return False


def main():
    print("APLICANDO CORRECAO DOS CALLBACKS DO PROFITDLL")
    print("="*60)
    print("Problema: Callbacks com retorno None causam Segmentation Fault")
    print("Solucao: Mudar todos para retorno c_int")
    print("="*60)
    
    # Aplicar correção
    if apply_fix():
        # Testar
        if test_import():
            print("\n✅ CORREÇÃO APLICADA COM SUCESSO!")
            print("\n📋 Próximos passos:")
            print("1. Execute novamente o teste de conexão")
            print("2. Se funcionar, a coleta histórica deve funcionar")
            print("3. Para reverter, use o arquivo de backup criado")
        else:
            print("\n❌ Erro após aplicar correção")
    else:
        print("\n❌ Erro ao aplicar correção")


if __name__ == "__main__":
    main()