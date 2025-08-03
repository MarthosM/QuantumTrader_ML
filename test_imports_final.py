#!/usr/bin/env python3
"""
Teste final de importação do schedule
"""

# Teste individual de cada import
try:
    import schedule
    print("✅ schedule: OK")
except ImportError as e:
    print(f"❌ schedule: {e}")

try:
    import time
    print("✅ time: OK")
except ImportError as e:
    print(f"❌ time: {e}")

try:
    import multiprocessing
    print("✅ multiprocessing: OK")
except ImportError as e:
    print(f"❌ multiprocessing: {e}")

try:
    from dotenv import load_dotenv
    print("✅ dotenv: OK")
except ImportError as e:
    print(f"❌ dotenv: {e}")

# Teste de funcionalidade básica do schedule
if 'schedule' in locals():
    try:
        def test_job():
            print("Job de teste")
        
        schedule.every(10).seconds.do(test_job)
        print("✅ schedule: Configuração básica OK")
    except Exception as e:
        print(f"❌ schedule: Erro na configuração - {e}")

print("\nTeste concluído!")
