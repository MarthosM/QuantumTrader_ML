import psutil
import subprocess
import time

print("\nVERIFICANDO PROFITCHART")
print("="*50)

# Verificar se ProfitChart está rodando
profit_found = False
for proc in psutil.process_iter(['pid', 'name']):
    try:
        if proc.info['name'] and 'profit' in proc.info['name'].lower():
            print(f"[OK] Encontrado: {proc.info['name']} (PID: {proc.info['pid']})")
            profit_found = True
    except:
        pass

if not profit_found:
    print("[!] ProfitChart NAO encontrado")
    print("\nO ProfitChart precisa estar aberto e logado!")
    print("Abra o ProfitChart primeiro e faca login.")
else:
    print("\n[OK] ProfitChart esta rodando")
    
# Verificar portas em uso
print("\nPORTAS EM USO:")
connections = psutil.net_connections()
profit_ports = []

for conn in connections:
    if conn.status == 'LISTEN' and conn.laddr.port in [443, 80, 8080, 9090, 5000]:
        print(f"Porta {conn.laddr.port}: {conn.status}")
        
# Testar conexao simples
print("\nTESTE DE REDE:")
import socket

test_hosts = [
    ("google.com", 80),
    ("google.com", 443),
]

for host, port in test_hosts:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"[OK] {host}:{port} acessivel")
        else:
            print(f"[!] {host}:{port} inacessivel")
    except:
        print(f"[ERRO] Teste {host}:{port}")
        
print("\n" + "="*50)
print("RECOMENDACOES:")
print("="*50)

if not profit_found:
    print("1. Abra o ProfitChart")
    print("2. Faca login com sua conta") 
    print("3. Aguarde carregar completamente")
    print("4. Tente novamente o coletor")
else:
    print("1. Feche o ProfitChart")
    print("2. Aguarde 10 segundos")
    print("3. Abra novamente e faca login")
    print("4. Aguarde carregar os dados")
    print("5. Execute o coletor novamente")
    
print("\nSe continuar com problemas:")
print("- Reinicie o computador")
print("- Desative temporariamente antivirus/firewall")
print("- Verifique se ha atualizacao do ProfitChart")