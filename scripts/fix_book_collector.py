"""
Script para corrigir problemas do coletor de book
"""

import os
import sys
import psutil
import socket
import time
import subprocess

def kill_process_on_port(port):
    """Mata processo usando a porta especificada"""
    print(f"Verificando processos na porta {port}...")
    
    # Método 1: Usando psutil
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.net_connections(kind='inet'):
                if conn.laddr.port == port:
                    print(f"Matando processo {proc.info['name']} (PID: {proc.info['pid']}) na porta {port}")
                    proc.kill()
                    time.sleep(1)
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Método 2: Usando netstat e taskkill (Windows)
    if sys.platform == 'win32':
        try:
            # Encontrar PID usando netstat
            result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    pid = parts[-1]
                    if pid.isdigit():
                        print(f"Matando processo PID {pid} na porta {port}")
                        subprocess.run(['taskkill', '/F', '/PID', pid])
                        time.sleep(1)
                        return True
        except Exception as e:
            print(f"Erro usando netstat: {e}")
    
    return False

def check_port_available(port):
    """Verifica se a porta está disponível"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result != 0

def fix_encoding():
    """Corrige problemas de encoding no Windows"""
    if sys.platform == 'win32':
        # Configurar encoding UTF-8
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # Tentar configurar o console do Windows
        try:
            import subprocess
            subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
        except:
            pass

def main():
    print("=== CORRECAO DO COLETOR DE BOOK ===\n")
    
    # 1. Corrigir encoding
    print("1. Configurando encoding UTF-8...")
    fix_encoding()
    
    # 2. Verificar e liberar porta 6789
    print("\n2. Verificando porta 6789...")
    
    if not check_port_available(6789):
        print("   Porta 6789 em uso. Tentando liberar...")
        if kill_process_on_port(6789):
            print("   [OK] Porta liberada com sucesso!")
        else:
            print("   [!] Nao foi possivel liberar a porta automaticamente")
            print("   Tente executar como administrador ou use:")
            print("   netstat -ano | findstr :6789")
            print("   taskkill /F /PID [PID_ENCONTRADO]")
    else:
        print("   [OK] Porta 6789 esta disponivel!")
    
    # 3. Limpar arquivos temporários
    print("\n3. Limpando arquivos temporarios...")
    temp_files = [
        'profit_dll_server.lock',
        'server.pid'
    ]
    
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"   [OK] Removido: {file}")
            except:
                print(f"   [!] Nao foi possivel remover: {file}")
    
    # 4. Verificar novamente
    print("\n4. Verificacao final...")
    
    if check_port_available(6789):
        print("   [OK] Sistema pronto para coletar book!")
        print("\n[SUCESSO] CORRECOES APLICADAS")
        print("\nAgora execute novamente:")
        print("python scripts/book_collector.py")
    else:
        print("   [ERRO] Porta ainda em uso")
        print("\nTente reiniciar o computador ou execute:")
        print("tasklist | findstr python")
        print("taskkill /F /IM python.exe")
    
    print("\n" + "="*40)

if __name__ == "__main__":
    main()