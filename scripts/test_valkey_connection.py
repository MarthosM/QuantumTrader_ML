"""
Teste simples de conexão com Valkey/Redis
"""

import sys

def test_connection():
    """Testa conexão com Valkey"""
    print("="*60)
    print("TESTE DE CONEXÃO VALKEY/REDIS")
    print("="*60)
    
    try:
        import valkey
        print("\n[OK] Biblioteca 'valkey' instalada")
    except ImportError:
        print("\n[ERRO] Biblioteca 'valkey' não instalada")
        print("Execute: pip install valkey")
        return False
    
    # Tentar conectar
    print("\nTentando conectar em localhost:6379...")
    
    try:
        client = valkey.Valkey(
            host='localhost', 
            port=6379, 
            decode_responses=True,
            socket_connect_timeout=5
        )
        
        # Teste ping
        response = client.ping()
        
        if response:
            print("[OK] Conexão estabelecida!")
            
            # Informações do servidor
            try:
                info = client.info()
                print(f"\nInformações do servidor:")
                print(f"  - Versão: {info.get('redis_version', 'Unknown')}")
                print(f"  - Modo: {info.get('redis_mode', 'standalone')}")
                print(f"  - Porta: {info.get('tcp_port', 6379)}")
                print(f"  - PID: {info.get('process_id', 'Unknown')}")
                print(f"  - Uptime: {info.get('uptime_in_seconds', 0)} segundos")
                print(f"  - Memória usada: {info.get('used_memory_human', 'Unknown')}")
                print(f"  - Clientes conectados: {info.get('connected_clients', 0)}")
                
                # Teste de escrita/leitura
                print("\nTestando escrita/leitura...")
                client.set('hmarl:test', 'Sistema HMARL Funcionando!')
                value = client.get('hmarl:test')
                
                if value == 'Sistema HMARL Funcionando!':
                    print("[OK] Escrita/leitura funcionando!")
                    client.delete('hmarl:test')
                    
                    print("\n✅ VALKEY/REDIS ESTÁ FUNCIONANDO!")
                    print("\nVocê pode executar os testes completos:")
                    print("  python -m pytest tests/test_zmq_valkey_infrastructure.py -v")
                    return True
                else:
                    print("[ERRO] Teste de escrita/leitura falhou")
                    return False
                    
            except Exception as e:
                print(f"\n[AVISO] Conectado mas erro ao obter informações: {e}")
                print("Pode ser um Redis com configurações restritivas")
                return True
                
    except valkey.ConnectionError as e:
        print(f"\n[ERRO] Não foi possível conectar: {e}")
        print("\nPossíveis soluções:")
        print("1. Inicie o Docker Desktop")
        print("2. Execute: scripts\\setup_valkey_windows.bat")
        print("3. Ou instale Redis localmente:")
        print("   - Windows: https://github.com/microsoftarchive/redis/releases")
        print("   - Ou use WSL2: sudo apt install redis-server")
        return False
        
    except Exception as e:
        print(f"\n[ERRO] Erro inesperado: {e}")
        return False

def test_docker_status():
    """Verifica status do Docker"""
    print("\n" + "-"*60)
    print("Verificando Docker...")
    
    import subprocess
    
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] Docker instalado: {result.stdout.strip()}")
            
            # Verificar se está rodando
            result = subprocess.run(['docker', 'info'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("[OK] Docker está rodando")
                
                # Verificar containers
                result = subprocess.run(['docker', 'ps', '-a', '--filter', 'name=valkey'], 
                                      capture_output=True, text=True)
                if 'valkey' in result.stdout:
                    print("[INFO] Container Valkey encontrado")
                    print("Execute: docker start valkey-trading")
                else:
                    print("[INFO] Nenhum container Valkey encontrado")
                    print("Execute: scripts\\setup_valkey_windows.bat")
            else:
                print("[ERRO] Docker não está rodando")
                print("Inicie o Docker Desktop")
        else:
            print("[ERRO] Docker não está instalado")
    except:
        print("[ERRO] Docker não encontrado no PATH")

if __name__ == "__main__":
    # Testar conexão
    connected = test_connection()
    
    # Se não conectou, verificar Docker
    if not connected:
        test_docker_status()
    
    print("\n" + "="*60)
    
    sys.exit(0 if connected else 1)