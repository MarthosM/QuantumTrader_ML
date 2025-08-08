"""
Monitor de Console Simples para Sistema de Produção
Exibe status em tempo real sem interface gráfica
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
import threading

class ConsoleMonitor:
    """Monitor simples via console"""
    
    def __init__(self):
        self.running = True
        self.last_metrics = {}
        self.last_log_line = 0
        
    def clear_screen(self):
        """Limpa a tela"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def read_latest_metrics(self):
        """Lê métricas mais recentes"""
        metrics_file = Path('metrics/current_metrics.json')
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def read_recent_logs(self, num_lines=10):
        """Lê últimas linhas do log"""
        log_file = Path(f"logs/production_{datetime.now().strftime('%Y%m%d')}.log")
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    return lines[-num_lines:]
            except:
                pass
        return []
    
    def count_data_records(self):
        """Conta registros gravados hoje"""
        data_dir = Path('data/book_tick_data')
        book_count = 0
        tick_count = 0
        
        if data_dir.exists():
            today = datetime.now().strftime('%Y%m%d')
            for file in data_dir.glob(f'*{today}*.csv'):
                try:
                    with open(file, 'r') as f:
                        line_count = sum(1 for _ in f) - 1  # -1 para header
                        if 'book' in file.name:
                            book_count += line_count
                        elif 'tick' in file.name:
                            tick_count += line_count
                except:
                    pass
        
        return book_count, tick_count
    
    def display(self):
        """Exibe informações no console"""
        self.clear_screen()
        
        print("=" * 70)
        print(" QUANTUM TRADER ML - MONITOR DE PRODUÇÃO")
        print("=" * 70)
        print(f" Horário: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Métricas
        metrics = self.read_latest_metrics()
        if metrics:
            print("\n[MÉTRICAS]")
            if 'metrics' in metrics:
                m = metrics['metrics']
                counters = m.get('counters', {})
                print(f"  Features calculadas: {counters.get('features_calculated', 0)}")
                print(f"  Predições feitas: {counters.get('predictions_made', 0)}")
                print(f"  Trades executados: {counters.get('trades_executed', 0)}")
                
                gauges = m.get('gauges', {})
                print(f"  Latência média: {gauges.get('avg_latency_ms', 0):.2f}ms")
                print(f"  Win rate: {gauges.get('win_rate', 0):.2%}")
        
        # Dados gravados
        book_count, tick_count = self.count_data_records()
        print(f"\n[GRAVAÇÃO DE DADOS]")
        print(f"  Book records: {book_count:,}")
        print(f"  Tick records: {tick_count:,}")
        
        # Status do sistema
        pid_file = Path('quantum_trader.pid')
        if pid_file.exists():
            print(f"\n[STATUS]")
            print(f"  Sistema: RODANDO")
            with open(pid_file, 'r') as f:
                pid = f.read().strip()
            print(f"  PID: {pid}")
        else:
            print(f"\n[STATUS]")
            print(f"  Sistema: PARADO")
        
        # Logs recentes
        print(f"\n[LOGS RECENTES]")
        logs = self.read_recent_logs(5)
        for log in logs:
            # Limitar tamanho da linha
            log = log.strip()[:100]
            # Colorir por nível
            if 'ERROR' in log:
                print(f"  [ERRO] {log}")
            elif 'WARNING' in log or 'AVISO' in log:
                print(f"  [AVISO] {log}")
            elif 'TRADE' in log:
                print(f"  [TRADE] {log}")
            else:
                print(f"  {log[:100]}")
        
        print("\n" + "=" * 70)
        print(" Pressione Ctrl+C para sair do monitor")
        print("=" * 70)
    
    def run(self):
        """Loop principal do monitor"""
        print("Iniciando monitor de console...")
        print("Atualizando a cada 5 segundos...")
        
        while self.running:
            try:
                self.display()
                time.sleep(5)
            except KeyboardInterrupt:
                print("\n\nMonitor encerrado.")
                break
            except Exception as e:
                print(f"Erro no monitor: {e}")
                time.sleep(5)


def main():
    """Função principal"""
    monitor = ConsoleMonitor()
    monitor.run()


if __name__ == "__main__":
    main()