"""
Script para reiniciar o coletor de book com verificações
"""

import os
import sys
import time
import psutil
import subprocess
from pathlib import Path

def kill_old_processes():
    """Mata processos antigos do coletor"""
    print("1. Limpando processos antigos...")
    
    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any('book_collector' in str(arg) or 'profit_dll_server' in str(arg) for arg in cmdline):
                print(f"   Matando processo {proc.info['name']} (PID: {proc.info['pid']})")
                proc.kill()
                killed += 1
                time.sleep(0.5)
        except:
            pass
    
    if killed:
        print(f"   [OK] {killed} processos encerrados")
    else:
        print("   [OK] Nenhum processo antigo encontrado")

def check_dll():
    """Verifica se a DLL está presente"""
    print("\n2. Verificando ProfitDLL...")
    
    dll_path = Path("ProfitDLL64.dll")
    if not dll_path.exists():
        print("   [ERRO] ProfitDLL64.dll não encontrada!")
        print("\n   AÇÃO NECESSÁRIA:")
        print("   1. Copie o arquivo ProfitDLL64.dll para:")
        print(f"      {Path.cwd()}")
        print("   2. Execute este script novamente")
        return False
    
    print(f"   [OK] DLL encontrada: {dll_path.absolute()}")
    return True

def check_directories():
    """Verifica e cria diretórios necessários"""
    print("\n3. Verificando diretórios...")
    
    dirs = [
        'data/realtime/book',
        'logs',
        'tmp'
    ]
    
    for dir_path in dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True)
            print(f"   [+] Criado: {dir_path}")
        else:
            print(f"   [OK] Existe: {dir_path}")

def start_collector():
    """Inicia o coletor"""
    print("\n4. Iniciando coletor de book...")
    
    # Comando para iniciar o coletor
    cmd = [sys.executable, "scripts/book_collector.py", "--symbol", "WDOU25"]
    
    print(f"   Executando: {' '.join(cmd)}")
    print("\n" + "="*60)
    print("INSTRUÇÕES:")
    print("="*60)
    print("1. Escolha opção 1 (Book completo)")
    print("2. Deixe rodando durante o pregão")
    print("3. Para parar: Ctrl+C")
    print("="*60)
    
    # Executar
    subprocess.call(cmd)

def main():
    print("="*60)
    print("REINICIALIZANDO COLETOR DE BOOK")
    print("="*60)
    
    # 1. Limpar processos
    kill_old_processes()
    
    # 2. Verificar DLL
    if not check_dll():
        print("\n[ERRO] Correções necessárias antes de continuar")
        return 1
    
    # 3. Verificar diretórios
    check_directories()
    
    # 4. Iniciar coletor
    print("\n[OK] Sistema pronto!")
    time.sleep(2)
    
    try:
        start_collector()
    except KeyboardInterrupt:
        print("\n\n[INFO] Coleta interrompida pelo usuário")
    except Exception as e:
        print(f"\n\n[ERRO] Falha ao executar coletor: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())