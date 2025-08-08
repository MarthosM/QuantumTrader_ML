"""
Verifica status do sistema de produção
"""
import time
from pathlib import Path
from datetime import datetime

def check_status():
    """Verifica status do sistema"""
    # Encontrar log mais recente
    logs_dir = Path("logs/production")
    logs = list(logs_dir.glob("fixed_*.log"))
    
    if not logs:
        print("Nenhum log encontrado!")
        return
        
    latest_log = max(logs, key=lambda p: p.stat().st_mtime)
    
    print(f"Log: {latest_log.name}")
    print(f"Tamanho: {latest_log.stat().st_size / 1024:.1f} KB")
    print()
    
    # Ler últimas linhas
    with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    # Estatísticas
    ml_predictions = sum(1 for line in lines if '[ML]' in line)
    orders = sum(1 for line in lines if '[ORDER]' in line)
    closes = sum(1 for line in lines if '[CLOSE]' in line)
    status_lines = [line for line in lines if '[STATUS]' in line]
    
    print("📊 ESTATÍSTICAS:")
    print(f"Total de linhas: {len(lines)}")
    print(f"Predições ML: {ml_predictions}")
    print(f"Ordens: {orders}")
    print(f"Fechamentos: {closes}")
    print()
    
    # Último status
    if status_lines:
        print("📈 ÚLTIMO STATUS:")
        print(status_lines[-1].strip())
        print()
    
    # Tempo desde início
    if lines:
        first_time = lines[0].split(' - ')[0]
        last_time = lines[-1].split(' - ')[0]
        try:
            start = datetime.strptime(first_time, "%Y-%m-%d %H:%M:%S,%f")
            end = datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S,%f")
            duration = (end - start).total_seconds()
            print(f"⏱️ Tempo de execução: {duration:.1f} segundos")
            print(f"   Primeira linha: {first_time}")
            print(f"   Última linha: {last_time}")
        except:
            pass
    
    # Verificar predições ML
    print()
    print("🤖 PREDIÇÕES ML:")
    if ml_predictions == 0:
        # Calcular quando deveria aparecer primeira predição
        strategy_lines = [i for i, line in enumerate(lines) if '[STRATEGY] Iniciando' in line]
        if strategy_lines:
            strategy_start = strategy_lines[0]
            print(f"Estratégia iniciada na linha {strategy_start}")
            print("Aguardando 30 segundos + 20 candles para primeira predição...")
            
            # Contar candles após estratégia
            candles_after = sum(1 for line in lines[strategy_start:] if '[DAILY #' in line)
            print(f"Candles recebidos após início: {candles_after}")
            
            if candles_after >= 20:
                print("[OK] Dados suficientes - predicao deve ocorrer em breve!")
            else:
                print(f"Aguardando mais {20 - candles_after} candles...")
    else:
        ml_lines = [line for line in lines if '[ML]' in line]
        for ml in ml_lines[-3:]:  # Últimas 3 predições
            print(ml.strip())

if __name__ == "__main__":
    check_status()