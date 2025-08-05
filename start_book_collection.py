"""
Script simples para iniciar coleta de book
"""

import subprocess
import sys
import time
import os

def main():
    print("\n" + "="*60)
    print("INICIANDO COLETA DE BOOK - WDO")
    print("="*60)
    print("Ticker: WDOU25")
    print("Tipo: Book completo (offer + price)")
    print("="*60 + "\n")
    
    # Criar comando com input automático
    # Simula pressionar Enter para aceitar opção padrão (1)
    cmd = [sys.executable, "scripts/book_collector.py", "--symbol", "WDOU25"]
    
    try:
        # Executar com input automático
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Enviar "1" + Enter quando solicitado
        time.sleep(2)  # Aguardar o menu aparecer
        process.stdin.write("1\n")
        process.stdin.flush()
        
        # Mostrar output em tempo real
        print("Coleta iniciada. Pressione Ctrl+C para parar.\n")
        
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
                
    except KeyboardInterrupt:
        print("\n\nParando coleta...")
        process.terminate()
        process.wait(timeout=5)
        print("Coleta finalizada.")
    except Exception as e:
        print(f"Erro: {e}")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())