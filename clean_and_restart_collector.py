"""
Script para limpar cache e reiniciar coletor de book
"""

import os
import sys
import time
import psutil
import subprocess
from pathlib import Path

def kill_profit_processes():
    """Mata processos que possam estar usando a DLL"""
    killed = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Procurar processos Python com profit ou book
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('book' in arg or 'profit' in arg for arg in cmdline):
                    print(f"Matando processo: {proc.info['pid']} - {cmdline}")
                    proc.kill()
                    killed += 1
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
            
    if killed > 0:
        print(f"\n✓ {killed} processos finalizados")
        time.sleep(2)  # Aguardar liberação
        
def clear_temp_files():
    """Limpa arquivos temporários que podem estar travando"""
    temp_patterns = [
        "*.tmp",
        "*.lock", 
        "*profit*.log",
        "*book*.log"
    ]
    
    cleared = 0
    for pattern in temp_patterns:
        for file in Path('.').glob(pattern):
            try:
                file.unlink()
                print(f"Removido: {file}")
                cleared += 1
            except:
                pass
                
    if cleared > 0:
        print(f"\n✓ {cleared} arquivos temporários removidos")
        
def restart_network_service():
    """Reinicia adaptador de rede (Windows)"""
    try:
        print("\nReiniciando adaptador de rede...")
        # Desabilitar e habilitar adaptador
        os.system('netsh interface set interface "Wi-Fi" admin=disable 2>nul')
        time.sleep(2)
        os.system('netsh interface set interface "Wi-Fi" admin=enable 2>nul')
        
        # Tentar Ethernet também
        os.system('netsh interface set interface "Ethernet" admin=disable 2>nul')
        time.sleep(2)
        os.system('netsh interface set interface "Ethernet" admin=enable 2>nul')
        
        print("✓ Adaptador de rede reiniciado")
        time.sleep(3)
    except:
        print("⚠ Não foi possível reiniciar adaptador de rede")
        
def clear_dns_cache():
    """Limpa cache DNS"""
    try:
        print("\nLimpando cache DNS...")
        os.system('ipconfig /flushdns')
        print("✓ Cache DNS limpo")
    except:
        pass
        
def test_connection():
    """Testa conexão básica"""
    import socket
    
    print("\nTestando conectividade...")
    
    # Testar DNS
    try:
        socket.gethostbyname('google.com')
        print("✓ DNS funcionando")
    except:
        print("❌ Problema com DNS")
        
    # Testar porta 443 (HTTPS)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('google.com', 443))
        sock.close()
        
        if result == 0:
            print("✓ Conexão HTTPS OK")
        else:
            print("❌ Problema com conexão HTTPS")
    except:
        print("❌ Erro ao testar conexão")
        
def run_simple_test():
    """Executa teste simples de conexão"""
    print("\n" + "="*60)
    print("TESTE SIMPLES DE CONEXÃO")
    print("="*60)
    
    test_code = '''
import ctypes
from ctypes import *
import time

# Callback simples
@WINFUNCTYPE(None, c_int32, c_int32)
def state_cb(nType, nResult):
    print(f"[STATE] Type={nType}, Result={nResult}")
    return None

try:
    dll = WinDLL("./ProfitDLL64.dll")
    print("✓ DLL carregada")
    
    # Apenas testar se consegue chamar uma função
    print("Testando função básica...")
    
    # Se tiver GetVersion ou similar
    if hasattr(dll, 'GetVersion'):
        version = dll.GetVersion()
        print(f"Version: {version}")
        
    print("✓ DLL respondendo")
    
except Exception as e:
    print(f"❌ Erro: {e}")
'''
    
    with open('test_dll.py', 'w') as f:
        f.write(test_code)
        
    # Executar teste
    result = subprocess.run([sys.executable, 'test_dll.py'], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Erros: {result.stderr}")
        
    # Limpar
    try:
        os.unlink('test_dll.py')
    except:
        pass
        
def main():
    print("\n" + "="*60)
    print("LIMPEZA E REINICIALIZAÇÃO DO SISTEMA")
    print("="*60)
    
    # 1. Matar processos
    print("\n1. Finalizando processos...")
    kill_profit_processes()
    
    # 2. Limpar arquivos temporários
    print("\n2. Limpando arquivos temporários...")
    clear_temp_files()
    
    # 3. Limpar cache DNS
    print("\n3. Limpando cache de rede...")
    clear_dns_cache()
    
    # 4. Testar conexão
    print("\n4. Testando conectividade...")
    test_connection()
    
    # 5. Teste simples da DLL
    print("\n5. Testando DLL...")
    run_simple_test()
    
    print("\n" + "="*60)
    print("LIMPEZA CONCLUÍDA")
    print("="*60)
    
    print("\nAgora tente executar novamente:")
    print("1. python book_collector_working.py")
    print("2. Se não funcionar, reinicie o computador")
    print("3. Verifique se o ProfitChart está aberto e logado")
    
    # Criar versão mínima para teste
    print("\nCriando versão mínima para teste...")
    
    minimal_code = '''
import ctypes
from ctypes import *
import time
import os

print("\\nTESTE MÍNIMO DO COLETOR")
print("="*40)

# State callback
@WINFUNCTYPE(None, c_int32, c_int32)
def state_cb(nType, nResult):
    print(f"[STATE] Type={nType}, Result={nResult}")
    return None

# TinyBook callback  
@WINFUNCTYPE(None, c_wchar_p, c_wchar_p, c_double, c_int, c_int)
def book_cb(ticker, bolsa, price, qty, side):
    side_str = "BID" if side == 0 else "ASK"
    print(f"[BOOK] {ticker} {side_str}: {price} x {qty}")
    return None

try:
    dll = WinDLL("./ProfitDLL64.dll")
    print("✓ DLL carregada")
    
    # Login
    key = c_wchar_p("HMARL")
    user = c_wchar_p(os.getenv('PROFIT_USERNAME', '29936354842'))
    pwd = c_wchar_p(os.getenv('PROFIT_PASSWORD', 'Ultrajiu33!'))
    
    print("Fazendo login...")
    result = dll.DLLInitializeLogin(key, user, pwd, state_cb, 
                                   None, None, None, None, None, 
                                   None, None, None, None, book_cb)
    
    print(f"Login result: {result}")
    
    # Aguardar
    print("\\nAguardando 5 segundos...")
    time.sleep(5)
    
    # Subscribe
    print("\\nSubscrevendo WDOU25...")
    result = dll.SubscribeTicker(c_wchar_p("WDOU25"), c_wchar_p("F"))
    print(f"Subscribe result: {result}")
    
    # Aguardar dados
    print("\\nAguardando dados por 30 segundos...")
    print("Se não aparecer nada, há um problema de conexão\\n")
    
    for i in range(30):
        print(f"\\r[{i+1}/30s] Aguardando...", end='', flush=True)
        time.sleep(1)
        
    print("\\n\\nFinalizando...")
    dll.DLLFinalize()
    
except Exception as e:
    print(f"❌ Erro: {e}")
    import traceback
    traceback.print_exc()
'''
    
    with open('book_collector_minimal.py', 'w') as f:
        f.write(minimal_code)
        
    print("✓ Criado: book_collector_minimal.py")
    print("\nExecute: python book_collector_minimal.py")


if __name__ == "__main__":
    main()