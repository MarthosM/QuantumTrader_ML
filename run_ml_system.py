#!/usr/bin/env python3
"""
Script Universal de Inicialização - ML Trading v2.0
Funciona tanto na raiz quanto no src
Ativa ambiente virtual automaticamente
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def find_project_root():
    """Encontra a raiz do projeto procurando por .env ou .venv"""
    current = Path(os.getcwd())
    
    # Procurar para cima até encontrar a raiz
    for path in [current] + list(current.parents):
        if (path / '.env').exists() or (path / '.venv').exists():
            return str(path)
    
    # Se não encontrar, assumir diretório atual
    return str(current)

def activate_venv_and_run():
    """Ativa ambiente virtual e executa o sistema"""
    project_root = find_project_root()
    print(f"🔍 Projeto detectado em: {project_root}")
    
    # Verificar se estamos no ambiente virtual
    if 'VIRTUAL_ENV' not in os.environ:
        print("⚡ Ativando ambiente virtual...")
        
        # Caminhos possíveis para o ambiente virtual
        venv_paths = [
            os.path.join(project_root, '.venv', 'Scripts', 'Activate.ps1'),
            os.path.join(project_root, '.venv', 'Scripts', 'activate.bat'),
            os.path.join(project_root, 'venv', 'Scripts', 'Activate.ps1'),
            os.path.join(project_root, 'venv', 'Scripts', 'activate.bat'),
        ]
        
        for venv_path in venv_paths:
            if os.path.exists(venv_path):
                print(f"✅ Ambiente virtual encontrado: {venv_path}")
                
                # Construir comando para ativar e executar
                if venv_path.endswith('.ps1'):
                    cmd = f'powershell -Command "& {{.\.venv\Scripts\Activate.ps1; python start_ml_trading_clean.py}}"'
                else:
                    cmd = f'call "{venv_path}" && python start_ml_trading_clean.py'
                
                print(f"🚀 Executando: {cmd}")
                os.chdir(project_root)
                subprocess.run(cmd, shell=True)
                return
        
        print("⚠️ Ambiente virtual não encontrado, continuando sem ativação...")
    else:
        print("✅ Ambiente virtual já está ativo")
    
    # Executar sistema diretamente
    os.chdir(project_root)
    
    # Procurar pelo script de inicialização
    startup_scripts = [
        'start_ml_trading_clean.py',
        'start_ml_trading_integrated.py', 
        'start_ml_trading.py'
    ]
    
    for script in startup_scripts:
        if os.path.exists(script):
            print(f"🎯 Executando script: {script}")
            subprocess.run([sys.executable, script])
            return
    
    # Se não encontrar scripts na raiz, tentar no src
    src_path = os.path.join(project_root, 'src')
    if os.path.exists(src_path):
        os.chdir(src_path)
        main_py = os.path.join(src_path, 'main.py')
        if os.path.exists(main_py):
            print(f"🎯 Executando main.py do src")
            subprocess.run([sys.executable, 'main.py'])
            return
    
    print("❌ Nenhum script de inicialização encontrado!")
    print("Scripts procurados:")
    for script in startup_scripts:
        print(f"  - {script}")
    print(f"  - src/main.py")

if __name__ == "__main__":
    try:
        activate_venv_and_run()
    except KeyboardInterrupt:
        print("\n🛑 Execução interrompida pelo usuário")
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
