"""
Diagnóstico rápido do sistema de coleta contínua
"""

import os
import sys
import subprocess
from datetime import datetime

print("\n" + "="*70)
print("DIAGNÓSTICO DO SISTEMA DE COLETA")
print(f"Horário: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("="*70 + "\n")

# 1. Verificar ambiente
print("1. VERIFICANDO AMBIENTE:")
print(f"   Python: {sys.version}")
print(f"   Diretório: {os.getcwd()}")
print(f"   Virtual Env: {os.getenv('VIRTUAL_ENV', 'NÃO ATIVADO')}")

# 2. Verificar arquivos necessários
print("\n2. VERIFICANDO ARQUIVOS:")
files_to_check = [
    "ProfitDLL64.dll",
    "book_collector.py",
    "book_collector_continuous.py",
    "start_continuous_collection.py"
]

for file in files_to_check:
    exists = os.path.exists(file)
    status = "✓" if exists else "✗"
    print(f"   {status} {file}")

# 3. Verificar processos Python rodando
print("\n3. PROCESSOS PYTHON ATIVOS:")
try:
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                          capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    python_count = sum(1 for line in lines if 'python.exe' in line.lower())
    print(f"   Encontrados: {python_count} processos Python")
except Exception as e:
    print(f"   Erro ao verificar processos: {e}")

# 4. Testar importação do módulo
print("\n4. TESTANDO IMPORTAÇÃO:")
try:
    import book_collector_continuous
    print("   ✓ book_collector_continuous importado com sucesso")
except Exception as e:
    print(f"   ✗ Erro ao importar: {e}")

# 5. Testar inicialização básica
print("\n5. TESTANDO INICIALIZAÇÃO BÁSICA:")
try:
    from ctypes import WinDLL
    dll = WinDLL("./ProfitDLL64.dll")
    print("   ✓ DLL carregada com sucesso")
except Exception as e:
    print(f"   ✗ Erro ao carregar DLL: {e}")

# 6. Verificar credenciais
print("\n6. VERIFICANDO CREDENCIAIS:")
username = os.getenv('PROFIT_USERNAME', '29936354842')
password = os.getenv('PROFIT_PASSWORD', 'Ultrajiu33!')
print(f"   Username: {username[:5]}...")
print(f"   Password: {'*' * len(password)}")

# 7. Arquivo PID
print("\n7. VERIFICANDO ARQUIVO PID:")
pid_file = "collection_manager.pid"
if os.path.exists(pid_file):
    with open(pid_file, 'r') as f:
        pid = f.read().strip()
    print(f"   ⚠️  Arquivo PID existe: {pid}")
    
    # Verificar se processo está rodando
    try:
        result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], 
                              capture_output=True, text=True)
        if pid in result.stdout:
            print(f"   ⚠️  Processo {pid} ainda está rodando!")
        else:
            print(f"   ✓ Processo {pid} não está mais ativo")
            os.remove(pid_file)
            print("   ✓ Arquivo PID removido")
    except:
        pass
else:
    print("   ✓ Nenhum arquivo PID encontrado")

print("\n" + "="*70)
print("DIAGNÓSTICO CONCLUÍDO")
print("="*70)

# Sugestão
print("\nSUGESTÃO:")
print("Execute diretamente: python book_collector_continuous.py")
print("Ou use o script simples: python start_book_collector_simple.py")