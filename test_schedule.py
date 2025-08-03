#!/usr/bin/env python3
"""
Teste simples do módulo schedule
"""
import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")

try:
    import schedule
    print("✅ schedule importado com sucesso!")
    print(f"schedule version: {schedule.__version__ if hasattr(schedule, '__version__') else 'unknown'}")
    
    # Teste básico
    def test_job():
        print("Job executado!")
    
    schedule.every(10).seconds.do(test_job)
    print("✅ schedule configurado com sucesso!")
    
except ImportError as e:
    print(f"❌ Erro ao importar schedule: {e}")
except Exception as e:
    print(f"❌ Erro geral: {e}")
