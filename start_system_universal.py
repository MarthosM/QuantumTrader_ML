#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema Universal de Inicialização ML Trading v2.0
Funciona tanto na raiz quanto no src, com ativação automática do ambiente virtual
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

def detect_project_structure():
    """Detecta se estamos na raiz ou no src e ajusta os caminhos"""
    current_dir = Path.cwd()
    
    # Verificar se estamos no src
    if current_dir.name == 'src' and (current_dir.parent / '.venv').exists():
        project_root = current_dir.parent
        is_in_src = True
    # Verificar se estamos na raiz
    elif (current_dir / '.venv').exists() and (current_dir / 'src').exists():
        project_root = current_dir
        is_in_src = False
    else:
        # Tentar encontrar a raiz do projeto
        check_dir = current_dir
        while check_dir.parent != check_dir:
            if (check_dir / '.venv').exists() and (check_dir / 'src').exists():
                project_root = check_dir
                is_in_src = current_dir != project_root
                break
            check_dir = check_dir.parent
        else:
            raise RuntimeError("Não foi possível encontrar a raiz do projeto ML_Tradingv2.0")
    
    return {
        'project_root': project_root,
        'src_dir': project_root / 'src',
        'venv_dir': project_root / '.venv',
        'is_in_src': is_in_src,
        'current_dir': current_dir
    }

def activate_virtual_environment(venv_dir):
    """Verifica se o ambiente virtual está ativo e retorna o caminho do Python"""
    # Verificar se já estamos no venv
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Ambiente virtual já está ativo")
        return sys.executable
    
    # Encontrar o executável Python no venv
    if os.name == 'nt':  # Windows
        python_exe = venv_dir / 'Scripts' / 'python.exe'
        activate_script = venv_dir / 'Scripts' / 'Activate.ps1'
    else:  # Linux/Mac
        python_exe = venv_dir / 'bin' / 'python'
        activate_script = venv_dir / 'bin' / 'activate'
    
    if not python_exe.exists():
        raise RuntimeError(f"Python do ambiente virtual não encontrado: {python_exe}")
    
    print(f"✓ Ambiente virtual encontrado: {venv_dir}")
    print(f"✓ Python do venv: {python_exe}")
    
    return str(python_exe)

def setup_python_path(paths):
    """Configura o PYTHONPATH com os diretórios necessários"""
    dirs_to_add = [
        str(paths['project_root']),
        str(paths['src_dir'])
    ]
    
    for dir_path in dirs_to_add:
        if dir_path not in sys.path:
            sys.path.insert(0, dir_path)
            print(f"✓ Adicionado ao PYTHONPATH: {dir_path}")

def run_with_venv(script_path, paths, use_clean_version=False):
    """Executa um script Python usando o ambiente virtual"""
    venv_python = activate_virtual_environment(paths['venv_dir'])
    
    # Definir variáveis de ambiente
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{paths['project_root']};{paths['src_dir']}"
    
    # Escolher qual versão executar
    if use_clean_version:
        script_name = 'start_ml_trading_clean.py'
    else:
        script_name = 'start_ml_trading_integrated.py'
    
    script_full_path = paths['project_root'] / script_name
    
    if not script_full_path.exists():
        print(f"❌ Script não encontrado: {script_full_path}")
        return False
    
    print(f"🚀 Executando: {script_full_path}")
    print(f"🐍 Python: {venv_python}")
    print(f"📁 Diretório de trabalho: {paths['project_root']}")
    
    try:
        # Mudar para o diretório raiz
        original_cwd = os.getcwd()
        os.chdir(paths['project_root'])
        
        # Executar o script
        result = subprocess.run(
            [venv_python, str(script_name)],
            env=env,
            cwd=paths['project_root'],
            capture_output=False,
            text=True
        )
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Erro executando script: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def run_main_directly(paths):
    """Executa main.py diretamente no src"""
    venv_python = activate_virtual_environment(paths['venv_dir'])
    
    main_script = paths['src_dir'] / 'main.py'
    
    if not main_script.exists():
        print(f"❌ main.py não encontrado: {main_script}")
        return False
    
    # Definir variáveis de ambiente
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{paths['project_root']};{paths['src_dir']}"
    
    print(f"🚀 Executando main.py diretamente")
    print(f"🐍 Python: {venv_python}")
    print(f"📁 Diretório de trabalho: {paths['src_dir']}")
    
    try:
        # Executar main.py
        result = subprocess.run(
            [venv_python, 'main.py'],
            env=env,
            cwd=paths['src_dir'],
            capture_output=False,
            text=True
        )
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Erro executando main.py: {e}")
        return False

def main():
    """Função principal de entrada"""
    print("=" * 80)
    print("    SISTEMA UNIVERSAL ML TRADING v2.0")
    print("=" * 80)
    print(f"    Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("    Detecção automática de ambiente e estrutura")
    print("=" * 80)
    
    try:
        # 1. Detectar estrutura do projeto
        print("\n🔍 1. Detectando estrutura do projeto...")
        paths = detect_project_structure()
        
        print(f"   📂 Raiz do projeto: {paths['project_root']}")
        print(f"   📂 Diretório src: {paths['src_dir']}")
        print(f"   📂 Ambiente virtual: {paths['venv_dir']}")
        print(f"   📍 Executando de: {paths['current_dir']}")
        print(f"   🔍 Está no src: {'Sim' if paths['is_in_src'] else 'Não'}")
        
        # 2. Configurar ambiente virtual
        print("\n🐍 2. Configurando ambiente virtual...")
        venv_python = activate_virtual_environment(paths['venv_dir'])
        
        # 3. Configurar caminhos Python
        print("\n📝 3. Configurando caminhos Python...")
        setup_python_path(paths)
        
        # 4. Apresentar opções
        print("\n🎯 4. Opções de execução disponíveis:")
        print("   [1] Sistema integrado ML (Recomendado)")
        print("   [2] Sistema integrado ML (Versão limpa)")
        print("   [3] main.py diretamente")
        print("   [4] Apenas configurar ambiente e sair")
        
        # 5. Executar opção padrão ou solicitar escolha
        try:
            choice = input("\n   Escolha uma opção (1-4) [Enter = 1]: ").strip()
            if not choice:
                choice = "1"
        except (KeyboardInterrupt, EOFError):
            choice = "1"
        
        success = False
        
        if choice == "1":
            print("\n🚀 5. Executando sistema integrado ML...")
            success = run_with_venv('start_ml_trading_integrated.py', paths, use_clean_version=False)
            
        elif choice == "2":
            print("\n🚀 5. Executando sistema integrado ML (versão limpa)...")
            success = run_with_venv('start_ml_trading_clean.py', paths, use_clean_version=True)
            
        elif choice == "3":
            print("\n🚀 5. Executando main.py diretamente...")
            success = run_main_directly(paths)
            
        elif choice == "4":
            print("\n✅ Ambiente configurado com sucesso!")
            print(f"   🐍 Python: {venv_python}")
            print(f"   📂 Projeto: {paths['project_root']}")
            print(f"   📂 Src: {paths['src_dir']}")
            success = True
            
        else:
            print(f"\n❌ Opção inválida: {choice}")
            success = False
        
        if success:
            print("\n✅ Execução concluída com sucesso!")
        else:
            print("\n❌ Execução falhou!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
