"""
Script para limpar processos Python travados
"""

import os
import sys
import subprocess
import time
from datetime import datetime

print("\n" + "="*60)
print("LIMPEZA DE PROCESSOS PYTHON")
print("="*60)
print(f"Hora: {datetime.now()}")
print("="*60)

def get_python_processes():
    """Lista todos os processos Python rodando"""
    try:
        # Comando para listar processos Python
        cmd = 'wmic process where "name like \'%python%\'" get ProcessId,CommandLine /format:list'
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        processes = []
        current_process = {}
        
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line.startswith('CommandLine='):
                current_process['cmd'] = line.split('=', 1)[1]
            elif line.startswith('ProcessId='):
                current_process['pid'] = line.split('=', 1)[1]
                if current_process.get('cmd') and current_process.get('pid'):
                    processes.append(current_process)
                    current_process = {}
        
        return processes
        
    except Exception as e:
        print(f"Erro ao listar processos: {e}")
        return []

def kill_trading_processes():
    """Mata processos relacionados ao trading"""
    trading_keywords = [
        'book_collector',
        'production_fixed',
        'monitor_gui',
        'test_connection',
        'force_reconnect',
        'ProfitDLL',
        'start_book_collection'
    ]
    
    processes = get_python_processes()
    killed = 0
    
    print(f"\nProcessos Python encontrados: {len(processes)}")
    
    for proc in processes:
        cmd = proc.get('cmd', '')
        pid = proc.get('pid', '')
        
        # Pular este próprio script
        if 'clean_python_processes' in cmd:
            continue
            
        # Verificar se é um processo de trading
        is_trading = any(keyword in cmd for keyword in trading_keywords)
        
        if is_trading and pid:
            print(f"\nEncontrado processo de trading:")
            print(f"  PID: {pid}")
            print(f"  CMD: {cmd[:100]}...")
            
            try:
                # Matar processo
                subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
                print(f"  ✓ Processo {pid} finalizado")
                killed += 1
            except Exception as e:
                print(f"  ✗ Erro ao matar processo {pid}: {e}")
    
    return killed

def check_dll_locks():
    """Verifica se a DLL está travada"""
    dll_path = './ProfitDLL64.dll'
    
    if os.path.exists(dll_path):
        try:
            # Tentar abrir a DLL para verificar se está travada
            with open(dll_path, 'rb') as f:
                f.read(1)
            print(f"\n✓ DLL não está travada: {dll_path}")
        except Exception as e:
            print(f"\n✗ DLL pode estar travada: {e}")
            print("  Recomenda-se reiniciar o computador")
    else:
        print(f"\n⚠ DLL não encontrada: {dll_path}")

def main():
    # 1. Listar processos antes
    print("\n1. PROCESSOS ANTES DA LIMPEZA:")
    processes_before = get_python_processes()
    
    for i, proc in enumerate(processes_before):
        print(f"\n   [{i+1}] PID: {proc.get('pid', 'N/A')}")
        print(f"       CMD: {proc.get('cmd', 'N/A')[:80]}...")
    
    if not processes_before:
        print("   Nenhum processo Python encontrado")
    
    # 2. Matar processos de trading
    print("\n2. MATANDO PROCESSOS DE TRADING...")
    killed = kill_trading_processes()
    
    if killed > 0:
        print(f"\n   ✓ {killed} processos finalizados")
        print("   Aguardando 3 segundos...")
        time.sleep(3)
    else:
        print("\n   Nenhum processo de trading encontrado")
    
    # 3. Verificar DLL
    print("\n3. VERIFICANDO STATUS DA DLL...")
    check_dll_locks()
    
    # 4. Listar processos depois
    print("\n4. PROCESSOS APÓS LIMPEZA:")
    processes_after = get_python_processes()
    
    if processes_after:
        for i, proc in enumerate(processes_after):
            print(f"\n   [{i+1}] PID: {proc.get('pid', 'N/A')}")
            print(f"       CMD: {proc.get('cmd', 'N/A')[:80]}...")
    else:
        print("   ✓ Nenhum processo Python rodando")
    
    # 5. Recomendações
    print("\n" + "="*60)
    print("RECOMENDAÇÕES:")
    print("="*60)
    
    if killed > 0:
        print("\n✓ Processos limpos com sucesso!")
        print("\nAgora você pode:")
        print("1. Aguardar 1 minuto")
        print("2. Executar: python test_connection_minimal.py")
        print("3. Se ainda falhar, reiniciar o computador")
    else:
        print("\n⚠ Nenhum processo travado encontrado")
        print("\nSe ainda tem problemas de conexão:")
        print("1. Reiniciar o computador")
        print("2. Verificar se ProfitChart funciona")
        print("3. Aguardar 30-60 minutos (possível bloqueio temporário)")
        print("4. Contatar suporte da corretora")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()