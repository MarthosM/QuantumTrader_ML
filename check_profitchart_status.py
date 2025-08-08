"""
Verifica status do ProfitChart e conectividade
"""

import os
import subprocess
import socket
import time
from datetime import datetime
import winreg

print("\n" + "="*60)
print("VERIFICAÇÃO DE STATUS - PROFITCHART")
print("="*60)
print(f"Hora: {datetime.now()}")
print("="*60)

def check_profitchart_installed():
    """Verifica se ProfitChart está instalado"""
    print("\n1. VERIFICANDO INSTALAÇÃO DO PROFITCHART:")
    
    # Locais comuns de instalação
    common_paths = [
        r"C:\ProfitChart",
        r"C:\Program Files\ProfitChart",
        r"C:\Program Files (x86)\ProfitChart",
        r"C:\Nelogica\ProfitChart",
        os.path.expanduser(r"~\AppData\Local\ProfitChart")
    ]
    
    found = False
    for path in common_paths:
        if os.path.exists(path):
            print(f"   ✓ Encontrado em: {path}")
            found = True
            
            # Verificar executável
            exe_path = os.path.join(path, "ProfitChart.exe")
            if os.path.exists(exe_path):
                print(f"   ✓ Executável encontrado: {exe_path}")
            break
    
    if not found:
        print("   ✗ ProfitChart não encontrado nos locais padrão")
        
        # Tentar registro do Windows
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                               r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")
            
            for i in range(0, winreg.QueryInfoKey(key)[0]):
                subkey_name = winreg.EnumKey(key, i)
                subkey = winreg.OpenKey(key, subkey_name)
                
                try:
                    name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                    if "ProfitChart" in name:
                        location = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                        print(f"   ✓ Encontrado no registro: {location}")
                        found = True
                        break
                except:
                    pass
                    
                winreg.CloseKey(subkey)
                
            winreg.CloseKey(key)
        except:
            pass
    
    return found

def check_profitchart_running():
    """Verifica se ProfitChart está rodando"""
    print("\n2. VERIFICANDO SE PROFITCHART ESTÁ RODANDO:")
    
    try:
        # Verificar processo
        cmd = 'tasklist /FI "IMAGENAME eq ProfitChart.exe" /FO CSV'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if "ProfitChart.exe" in result.stdout:
            print("   ✓ ProfitChart está rodando")
            
            # Pegar PID
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                pid = lines[1].split(',')[1].strip('"')
                print(f"   PID: {pid}")
            
            return True
        else:
            print("   ✗ ProfitChart NÃO está rodando")
            return False
            
    except Exception as e:
        print(f"   ✗ Erro ao verificar processo: {e}")
        return False

def check_network_connectivity():
    """Verifica conectividade de rede"""
    print("\n3. VERIFICANDO CONECTIVIDADE DE REDE:")
    
    # Servidores conhecidos da Nelogica/ProfitChart
    servers = [
        ("profitchart.com.br", 80),
        ("profitchart.com.br", 443),
        ("login.nelogica.com.br", 443),
        ("8.8.8.8", 53)  # Google DNS para testar internet
    ]
    
    connected = 0
    for host, port in servers:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                print(f"   ✓ {host}:{port} - Conectado")
                connected += 1
            else:
                print(f"   ✗ {host}:{port} - Falha na conexão")
                
        except socket.gaierror:
            print(f"   ✗ {host}:{port} - Host não encontrado")
        except Exception as e:
            print(f"   ✗ {host}:{port} - Erro: {e}")
    
    return connected > 0

def check_firewall_rules():
    """Verifica regras de firewall"""
    print("\n4. VERIFICANDO FIREWALL DO WINDOWS:")
    
    try:
        # Verificar se firewall está ativo
        cmd = 'netsh advfirewall show allprofiles state'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if "ON" in result.stdout:
            print("   ⚠ Firewall está ATIVO")
            
            # Verificar regras para ProfitChart
            cmd = 'netsh advfirewall firewall show rule name=all | findstr /i "profitchart"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.stdout:
                print("   ✓ Regras encontradas para ProfitChart:")
                for line in result.stdout.split('\n')[:3]:
                    if line.strip():
                        print(f"     {line.strip()}")
            else:
                print("   ⚠ Nenhuma regra específica para ProfitChart")
                print("   Considere adicionar exceção no firewall")
        else:
            print("   ✓ Firewall está DESATIVADO")
            
    except Exception as e:
        print(f"   ✗ Erro ao verificar firewall: {e}")

def check_dll_files():
    """Verifica arquivos DLL necessários"""
    print("\n5. VERIFICANDO ARQUIVOS DLL:")
    
    required_dlls = [
        "ProfitDLL64.dll",
        "libssl-1_1-x64.dll",
        "libcrypto-1_1-x64.dll"
    ]
    
    current_dir = os.getcwd()
    all_found = True
    
    for dll in required_dlls:
        dll_path = os.path.join(current_dir, dll)
        if os.path.exists(dll_path):
            size = os.path.getsize(dll_path)
            print(f"   ✓ {dll} - {size:,} bytes")
        else:
            print(f"   ✗ {dll} - NÃO ENCONTRADO")
            all_found = False
    
    return all_found

def main():
    # Verificações
    profitchart_installed = check_profitchart_installed()
    profitchart_running = check_profitchart_running()
    network_ok = check_network_connectivity()
    check_firewall_rules()
    dlls_ok = check_dll_files()
    
    # Diagnóstico
    print("\n" + "="*60)
    print("DIAGNÓSTICO:")
    print("="*60)
    
    issues = []
    
    if not profitchart_installed:
        issues.append("ProfitChart não está instalado")
    
    if not profitchart_running:
        issues.append("ProfitChart não está rodando")
    
    if not network_ok:
        issues.append("Problemas de conectividade de rede")
    
    if not dlls_ok:
        issues.append("DLLs necessárias não encontradas")
    
    if issues:
        print("\n⚠ PROBLEMAS ENCONTRADOS:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print("\nRECOMENDAÇÕES:")
        
        if not profitchart_running and profitchart_installed:
            print("\n1. Abra o ProfitChart manualmente")
            print("2. Faça login com suas credenciais")
            print("3. Verifique se consegue ver cotações")
            print("4. Se funcionar, tente novamente o sistema")
        
        if not network_ok:
            print("\n1. Verifique sua conexão com a internet")
            print("2. Desative temporariamente antivírus/firewall")
            print("3. Tente ping profitchart.com.br")
        
        if not dlls_ok:
            print("\n1. Verifique se as DLLs estão no diretório correto")
            print("2. Baixe as DLLs do repositório oficial")
    else:
        print("\n✓ TODOS OS COMPONENTES PARECEM ESTAR OK!")
        print("\nSe ainda tem problemas de conexão:")
        print("1. O servidor pode estar temporariamente indisponível")
        print("2. Sua conta pode ter limite de conexões atingido")
        print("3. Aguarde 30-60 minutos e tente novamente")
        print("4. Entre em contato com o suporte da corretora")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()