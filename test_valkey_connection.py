#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste de conexão com Valkey
"""

import valkey
import sys
import time

def test_valkey_connection():
    """Testa conexão com Valkey"""
    
    print("[Teste] Conectando ao Valkey...")
    
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Conectar ao Valkey
            client = valkey.Valkey(
                host='localhost',
                port=6379,
                decode_responses=True,
                socket_connect_timeout=5
            )
            
            # Teste ping
            response = client.ping()
            if response:
                print(f"[OK] Conexao estabelecida! Resposta: {response}")
                
                # Teste de escrita
                test_key = "test:connection"
                test_value = "success"
                client.set(test_key, test_value, ex=60)  # Expira em 60s
                
                # Teste de leitura
                retrieved = client.get(test_key)
                if retrieved == test_value:
                    print("[OK] Teste de escrita/leitura bem sucedido")
                
                # Teste de streams
                stream_key = "test:stream"
                stream_id = client.xadd(stream_key, {"test": "data", "timestamp": str(time.time())})
                print(f"[OK] Stream criado com ID: {stream_id}")
                
                # Ler stream
                entries = client.xrange(stream_key, count=1)
                if entries:
                    print(f"[OK] Stream lido: {entries[0]}")
                
                # Info do servidor
                info = client.info('server')
                print(f"\n[Info] Servidor Valkey:")
                print(f"  - Versao: {info.get('redis_version', 'N/A')}")
                print(f"  - Modo: {info.get('redis_mode', 'N/A')}")
                print(f"  - OS: {info.get('os', 'N/A')}")
                
                # Cleanup
                client.delete(test_key, stream_key)
                
                print("\n[SUCESSO] Todos os testes passaram!")
                return True
                
        except valkey.ConnectionError as e:
            retry_count += 1
            print(f"[Aviso] Tentativa {retry_count}/{max_retries} - Erro: {e}")
            if retry_count < max_retries:
                print(f"[Info] Aguardando 2 segundos antes de tentar novamente...")
                time.sleep(2)
        
        except Exception as e:
            print(f"[ERRO] Erro inesperado: {type(e).__name__}: {e}")
            return False
    
    print(f"\n[ERRO] Nao foi possivel conectar ao Valkey apos {max_retries} tentativas")
    print("\nVerifique se:")
    print("1. Docker Desktop esta rodando")
    print("2. Valkey foi iniciado com: start_valkey.bat")
    print("3. Porta 6379 nao esta sendo usada por outro processo")
    
    return False

if __name__ == "__main__":
    success = test_valkey_connection()
    sys.exit(0 if success else 1)