"""
Monitor de Log em Tempo Real
"""
import time
import sys
from pathlib import Path

def tail_log(log_file, keywords=['[ML]', '[ORDER]', '[STATUS]', '[STOP', '[TAKE']):
    """Monitor log em tempo real"""
    print(f"Monitorando: {log_file}")
    print("Filtros:", keywords)
    print("-" * 80)
    
    with open(log_file, 'r', encoding='utf-8') as f:
        # Ir para o final
        f.seek(0, 2)
        
        while True:
            line = f.readline()
            if line:
                # Verificar keywords
                if any(kw in line for kw in keywords):
                    print(line.strip())
            else:
                time.sleep(0.1)

if __name__ == "__main__":
    # Encontrar log mais recente
    logs_dir = Path("logs/production")
    logs = list(logs_dir.glob("fixed_*.log"))
    
    if not logs:
        print("Nenhum log encontrado!")
        sys.exit(1)
        
    # Pegar o mais recente
    latest_log = max(logs, key=lambda p: p.stat().st_mtime)
    
    print(f"Log mais recente: {latest_log}")
    print()
    
    try:
        tail_log(latest_log)
    except KeyboardInterrupt:
        print("\nMonitoramento interrompido")