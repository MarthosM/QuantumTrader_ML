"""
Script para configurar e iniciar Valkey (Redis fork) para o sistema HMARL
"""

import subprocess
import time
import sys
import os

def run_command(cmd, shell=True):
    """Executa comando e retorna resultado"""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_docker():
    """Verifica se Docker está rodando"""
    print("1. Verificando Docker...")
    success, stdout, stderr = run_command("docker info")
    
    if not success:
        print("   [ERRO] Docker não está rodando!")
        print("   Por favor, inicie o Docker Desktop")
        return False
    
    print("   [OK] Docker está rodando")
    return True

def stop_existing_valkey():
    """Para container Valkey existente se houver"""
    print("\n2. Verificando containers existentes...")
    
    # Verificar se existe
    success, stdout, stderr = run_command("docker ps -a --filter name=valkey-trading --format {{.Names}}")
    
    if stdout.strip() == "valkey-trading":
        print("   Container 'valkey-trading' encontrado")
        
        # Parar se estiver rodando
        print("   Parando container...")
        run_command("docker stop valkey-trading")
        
        # Remover container
        print("   Removendo container antigo...")
        run_command("docker rm valkey-trading")
        print("   [OK] Container antigo removido")
    else:
        print("   [OK] Nenhum container existente")

def create_valkey_volume():
    """Cria volume para persistência"""
    print("\n3. Criando volume para dados...")
    
    success, stdout, stderr = run_command("docker volume create valkey-data")
    
    if success:
        print("   [OK] Volume 'valkey-data' criado/verificado")
    else:
        print(f"   [AVISO] {stderr}")
    
    return True

def start_valkey():
    """Inicia container Valkey"""
    print("\n4. Iniciando Valkey...")
    
    cmd = [
        "docker", "run", "-d",
        "--name", "valkey-trading",
        "-p", "6379:6379",
        "-v", "valkey-data:/data",
        "--restart", "unless-stopped",
        "valkey/valkey:latest",
        "--maxmemory", "2gb",
        "--maxmemory-policy", "allkeys-lru",
        "--save", "60", "1000",  # Salvar a cada 60s se 1000+ mudanças
        "--save", "300", "100",  # Salvar a cada 5min se 100+ mudanças
        "--save", "900", "1"     # Salvar a cada 15min se 1+ mudança
    ]
    
    success, stdout, stderr = run_command(" ".join(cmd))
    
    if success:
        container_id = stdout.strip()[:12]
        print(f"   [OK] Container iniciado: {container_id}")
        return True
    else:
        print(f"   [ERRO] Falha ao iniciar: {stderr}")
        return False

def wait_for_valkey():
    """Aguarda Valkey estar pronto"""
    print("\n5. Aguardando Valkey inicializar...")
    
    max_attempts = 30
    for i in range(max_attempts):
        success, stdout, stderr = run_command('docker exec valkey-trading redis-cli ping')
        
        if success and "PONG" in stdout:
            print("   [OK] Valkey respondendo!")
            return True
        
        print(f"   Tentativa {i+1}/{max_attempts}...", end='\r')
        time.sleep(1)
    
    print("\n   [ERRO] Timeout aguardando Valkey")
    return False

def test_valkey_connection():
    """Testa conexão com Valkey via Python"""
    print("\n6. Testando conexão Python...")
    
    try:
        import valkey
        client = valkey.Valkey(host='localhost', port=6379, decode_responses=True)
        
        # Teste básico
        client.set('test_key', 'HMARL_Test')
        value = client.get('test_key')
        
        if value == 'HMARL_Test':
            print("   [OK] Conexão Python funcionando!")
            
            # Informações do servidor
            info = client.info()
            print(f"   - Versão: {info.get('redis_version', 'Unknown')}")
            print(f"   - Memória usada: {info.get('used_memory_human', 'Unknown')}")
            print(f"   - Clientes conectados: {info.get('connected_clients', 0)}")
            
            client.delete('test_key')
            return True
        else:
            print("   [ERRO] Teste de leitura/escrita falhou")
            return False
            
    except ImportError:
        print("   [ERRO] Biblioteca 'valkey' não instalada")
        print("   Execute: pip install valkey")
        return False
    except Exception as e:
        print(f"   [ERRO] Conexão falhou: {e}")
        return False

def show_useful_commands():
    """Mostra comandos úteis"""
    print("\n" + "="*60)
    print("VALKEY INSTALADO COM SUCESSO!")
    print("="*60)
    
    print("\nComandos úteis:")
    print("\n1. Status do container:")
    print("   docker ps --filter name=valkey-trading")
    
    print("\n2. Logs do Valkey:")
    print("   docker logs valkey-trading")
    
    print("\n3. Acessar CLI do Valkey:")
    print("   docker exec -it valkey-trading redis-cli")
    
    print("\n4. Parar Valkey:")
    print("   docker stop valkey-trading")
    
    print("\n5. Iniciar Valkey:")
    print("   docker start valkey-trading")
    
    print("\n6. Monitorar em tempo real:")
    print("   docker exec -it valkey-trading redis-cli monitor")
    
    print("\n7. Ver estatísticas:")
    print("   docker exec -it valkey-trading redis-cli info stats")
    
    print("\n" + "="*60)

def main():
    """Função principal"""
    print("="*60)
    print("SETUP VALKEY PARA HMARL")
    print("="*60)
    
    # Verificações
    if not check_docker():
        print("\n[ERRO] Configure o Docker primeiro!")
        sys.exit(1)
    
    # Setup
    stop_existing_valkey()
    create_valkey_volume()
    
    if not start_valkey():
        print("\n[ERRO] Falha ao iniciar Valkey!")
        sys.exit(1)
    
    if not wait_for_valkey():
        print("\n[ERRO] Valkey não respondeu!")
        # Mostrar logs para debug
        print("\nLogs do container:")
        os.system("docker logs valkey-trading --tail 20")
        sys.exit(1)
    
    # Teste
    if test_valkey_connection():
        show_useful_commands()
        
        print("\nPróximo passo:")
        print("Execute os testes completos:")
        print("  python -m pytest tests/test_zmq_valkey_infrastructure.py -v")
    else:
        print("\n[AVISO] Valkey está rodando mas conexão Python falhou")
        print("Instale a biblioteca: pip install valkey")

if __name__ == "__main__":
    main()