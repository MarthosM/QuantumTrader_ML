"""
Debug do problema com Valkey
"""

import valkey
from datetime import datetime

def test_valkey_xadd():
    """Testa diferentes formatos de xadd"""
    
    client = valkey.Valkey(host='localhost', port=6379, decode_responses=False)
    
    print("Testando diferentes formatos de xadd...\n")
    
    # Teste 1: Tudo como bytes
    print("1. Teste com tudo em bytes:")
    try:
        result = client.xadd(
            "test:stream1",
            {b"key1": b"value1", b"key2": b"value2"},
            maxlen=100
        )
        print(f"   [OK] Resultado: {result}")
    except Exception as e:
        print(f"   [ERRO] {e}")
    
    # Teste 2: Stream como string, dados como bytes
    print("\n2. Teste com stream string, dados bytes:")
    try:
        result = client.xadd(
            "test:stream2",
            {b"key1": b"value1", b"key2": b"value2"},
            maxlen=100
        )
        print(f"   [OK] Resultado: {result}")
    except Exception as e:
        print(f"   [ERRO] {e}")
    
    # Teste 3: Cliente com decode_responses=True
    print("\n3. Teste com decode_responses=True:")
    client2 = valkey.Valkey(host='localhost', port=6379, decode_responses=True)
    try:
        result = client2.xadd(
            "test:stream3",
            {"key1": "value1", "key2": "value2"},
            maxlen=100
        )
        print(f"   [OK] Resultado: {result}")
    except Exception as e:
        print(f"   [ERRO] {e}")
    
    # Teste 4: Misturando tipos (erro esperado)
    print("\n4. Teste misturando tipos (deve dar erro):")
    try:
        result = client.xadd(
            "test:stream4",
            {"key1": "value1", b"key2": b"value2"},  # Misturado
            maxlen=100
        )
        print(f"   [OK?] Resultado: {result}")
    except Exception as e:
        print(f"   [ERRO ESPERADO] {e}")
    
    # Teste 5: Com ID específico
    print("\n5. Teste com ID específico:")
    try:
        timestamp_ms = int(datetime.now().timestamp() * 1000)
        result = client2.xadd(
            "test:stream5",
            {"timestamp": datetime.now().isoformat(), "value": "123"},
            id=f"{timestamp_ms}-0",
            maxlen=100
        )
        print(f"   [OK] Resultado: {result}")
    except Exception as e:
        print(f"   [ERRO] {e}")
    
    # Limpar streams de teste
    print("\n6. Limpando streams de teste...")
    for i in range(1, 6):
        try:
            client.delete(f"test:stream{i}")
        except:
            pass
    print("   [OK] Streams removidos")
    
    print("\n" + "="*50)
    print("CONCLUSÃO: Use decode_responses=True para strings")
    print("ou decode_responses=False com todos os valores em bytes")
    print("="*50)

if __name__ == "__main__":
    test_valkey_xadd()