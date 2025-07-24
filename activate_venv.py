#!/usr/bin/env python3
"""
Ativador de Ambiente Virtual para ML Trading v2.0
Ativa o ambiente virtual corretamente no Windows/PowerShell
"""

import os
import sys
import subprocess
from pathlib import Path

def find_project_root():
    """Encontra a raiz do projeto"""
    current = Path.cwd()
    
    # Se estamos no src, subir um nível
    if current.name == 'src':
        root = current.parent
    else:
        root = current
    
    # Verificar se é realmente a raiz do projeto
    if (root / '.venv').exists() and (root / 'src').exists():
        return root
    
    raise RuntimeError("Não foi possível encontrar a raiz do projeto")

def activate_venv_and_run(script_name=None):
    """Ativa o ambiente virtual e executa um script"""
    try:
        project_root = find_project_root()
        venv_dir = project_root / '.venv'
        
        print(f"📂 Projeto: {project_root}")
        print(f"🐍 Venv: {venv_dir}")
        
        # Comando para ativar o venv no PowerShell
        activate_script = venv_dir / 'Scripts' / 'Activate.ps1'
        
        if not activate_script.exists():
            print(f"❌ Script de ativação não encontrado: {activate_script}")
            return False
        
        # Se não foi especificado script, apenas ativar o ambiente
        if not script_name:
            cmd = f"""
            Set-Location "{project_root}"
            & "{activate_script}"
            Write-Host "✅ Ambiente virtual ativado!"
            Write-Host "📂 Diretório: $(Get-Location)"
            Write-Host "🐍 Python: $(where.exe python)"
            """
        else:
            # Ativar venv e executar script
            script_path = project_root / script_name
            if not script_path.exists():
                print(f"❌ Script não encontrado: {script_path}")
                return False
                
            cmd = f"""
            Set-Location "{project_root}"
            & "{activate_script}"
            Write-Host "✅ Ambiente virtual ativado!"
            Write-Host "🚀 Executando: {script_name}"
            python "{script_name}"
            """
        
        # Executar no PowerShell
        result = subprocess.run(
            ['powershell', '-Command', cmd],
            cwd=project_root,
            capture_output=False,
            text=True
        )
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def main():
    """Função principal"""
    print("🔧 ATIVADOR DE AMBIENTE VIRTUAL - ML TRADING v2.0")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        script_name = sys.argv[1]
        print(f"🎯 Script solicitado: {script_name}")
    else:
        script_name = None
        print("🎯 Apenas ativando ambiente virtual")
    
    success = activate_venv_and_run(script_name)
    
    if success:
        print("✅ Concluído com sucesso!")
    else:
        print("❌ Falha na execução!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
