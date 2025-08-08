"""
Monitor GUI v2 - Dashboard atualizado para QuantumTrader ML
Lê dados do arquivo JSON compartilhado e logs
"""

import tkinter as tk
from tkinter import ttk
import json
import time
from datetime import datetime
from pathlib import Path
import threading

class TradingMonitorV2:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("QuantumTrader ML - Monitor v2")
        self.root.geometry("800x600")
        
        # Estilo
        self.root.configure(bg='#1e1e1e')
        
        # Dados
        self.shared_data_file = Path('data/monitor_data.json')
        self.last_update = None
        
        # Criar interface
        self._create_widgets()
        
        # Iniciar thread de atualização
        self.running = True
        self.update_thread = threading.Thread(target=self._update_data, daemon=True)
        self.update_thread.start()
        
        # Atualizar GUI
        self._refresh_gui()
        
    def _create_widgets(self):
        """Cria interface simplificada"""
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#1e1e1e', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title = tk.Label(main_frame, text="QUANTUMTRADER ML - MONITOR", 
                        bg='#1e1e1e', fg='#00ff00', font=('Arial', 18, 'bold'))
        title.pack(pady=(0, 20))
        
        # Frame de status
        status_frame = tk.LabelFrame(main_frame, text="STATUS DO SISTEMA", 
                                   bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold'))
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Labels de status
        self.status_label = tk.Label(status_frame, text="Status: Aguardando...", 
                                   bg='#2d2d2d', fg='yellow', font=('Arial', 11))
        self.status_label.pack(pady=5)
        
        self.ticker_label = tk.Label(status_frame, text="Ticker: --", 
                                   bg='#2d2d2d', fg='white', font=('Arial', 11))
        self.ticker_label.pack(pady=5)
        
        # Frame de preço
        price_frame = tk.LabelFrame(main_frame, text="DADOS DE MERCADO", 
                                  bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold'))
        price_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.price_label = tk.Label(price_frame, text="R$ 0.00", 
                                  bg='#2d2d2d', fg='#00ff00', font=('Arial', 24, 'bold'))
        self.price_label.pack(pady=10)
        
        self.update_time_label = tk.Label(price_frame, text="Última atualização: --", 
                                        bg='#2d2d2d', fg='gray', font=('Arial', 9))
        self.update_time_label.pack(pady=(0, 10))
        
        # Frame de trading
        trading_frame = tk.LabelFrame(main_frame, text="TRADING", 
                                    bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold'))
        trading_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Grid de métricas
        metrics = [
            ("Posição:", "position_label"),
            ("P&L Diário:", "daily_pnl_label"),
            ("P&L Total:", "total_pnl_label"),
            ("Trades:", "trades_label"),
            ("Win Rate:", "winrate_label"),
            ("Predições ML:", "predictions_label")
        ]
        
        for i, (label_text, attr_name) in enumerate(metrics):
            row = i // 2
            col = (i % 2) * 2
            
            label = tk.Label(trading_frame, text=label_text, bg='#2d2d2d', fg='gray', 
                           font=('Arial', 10))
            label.grid(row=row, column=col, sticky='w', padx=10, pady=5)
            
            value_label = tk.Label(trading_frame, text="--", bg='#2d2d2d', fg='white', 
                                 font=('Arial', 11, 'bold'))
            value_label.grid(row=row, column=col+1, sticky='w', padx=(0, 20), pady=5)
            
            setattr(self, attr_name, value_label)
        
        # Frame de callbacks
        callback_frame = tk.LabelFrame(main_frame, text="CALLBACKS", 
                                     bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold'))
        callback_frame.pack(fill=tk.BOTH, expand=True)
        
        self.callbacks_text = tk.Text(callback_frame, bg='#1e1e1e', fg='white', 
                                    font=('Consolas', 10), height=10)
        self.callbacks_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def _update_data(self):
        """Thread que lê dados compartilhados"""
        while self.running:
            try:
                if self.shared_data_file.exists():
                    with open(self.shared_data_file, 'r') as f:
                        data = json.load(f)
                        self.last_update = data
                else:
                    # Tentar ler do log se JSON não existe
                    self._read_from_logs()
                    
            except Exception as e:
                print(f"Erro ao ler dados: {e}")
                
            time.sleep(0.5)
            
    def _read_from_logs(self):
        """Fallback: lê dados dos logs"""
        try:
            log_dir = Path('logs/production')
            log_files = list(log_dir.glob('final_*.log'))
            
            if log_files:
                latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
                
                # Ler últimas linhas para extrair dados
                with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()[-100:]  # Últimas 100 linhas
                    
                    for line in reversed(lines):
                        if '[STATUS]' in line and 'Price:' in line:
                            try:
                                price = float(line.split('Price:')[1].split()[0])
                                if not self.last_update:
                                    self.last_update = {}
                                self.last_update['price'] = price
                                break
                            except:
                                pass
                                
        except Exception as e:
            print(f"Erro ao ler logs: {e}")
            
    def _refresh_gui(self):
        """Atualiza interface com dados mais recentes"""
        if self.last_update:
            try:
                # Status
                status = self.last_update.get('status', 'Desconhecido')
                color = '#00ff00' if status == 'Operacional' else 'yellow'
                self.status_label.config(text=f"Status: {status}", fg=color)
                
                # Ticker
                ticker = self.last_update.get('ticker', '--')
                self.ticker_label.config(text=f"Ticker: {ticker}")
                
                # Preço
                price = self.last_update.get('price', 0)
                self.price_label.config(text=f"R$ {price:,.2f}")
                
                # Hora da atualização
                timestamp = self.last_update.get('timestamp', '')
                if timestamp:
                    dt = datetime.fromisoformat(timestamp)
                    self.update_time_label.config(text=f"Última atualização: {dt.strftime('%H:%M:%S')}")
                
                # Métricas de trading
                self.position_label.config(text=str(self.last_update.get('position', 0)))
                
                daily_pnl = self.last_update.get('daily_pnl', 0)
                color = '#00ff00' if daily_pnl >= 0 else '#ff4444'
                self.daily_pnl_label.config(text=f"R$ {daily_pnl:,.2f}", fg=color)
                
                total_pnl = self.last_update.get('total_pnl', 0)
                color = '#00ff00' if total_pnl >= 0 else '#ff4444'
                self.total_pnl_label.config(text=f"R$ {total_pnl:,.2f}", fg=color)
                
                trades = self.last_update.get('trades', 0)
                self.trades_label.config(text=str(trades))
                
                wins = self.last_update.get('wins', 0)
                losses = self.last_update.get('losses', 0)
                winrate = (wins / trades * 100) if trades > 0 else 0
                self.winrate_label.config(text=f"{winrate:.1f}%")
                
                predictions = self.last_update.get('predictions', 0)
                self.predictions_label.config(text=str(predictions))
                
                # Callbacks
                callbacks = self.last_update.get('callbacks', {})
                if callbacks:
                    cb_text = "Callbacks recebidos:\n\n"
                    for cb_type, count in callbacks.items():
                        if count > 0:
                            cb_text += f"{cb_type:>15}: {count:,}\n"
                    
                    self.callbacks_text.delete(1.0, tk.END)
                    self.callbacks_text.insert(1.0, cb_text)
                    
            except Exception as e:
                print(f"Erro ao atualizar GUI: {e}")
                
        # Agendar próxima atualização
        self.root.after(500, self._refresh_gui)
        
    def run(self):
        """Inicia o monitor"""
        self.root.mainloop()
        
    def stop(self):
        """Para o monitor"""
        self.running = False
        self.root.quit()

if __name__ == "__main__":
    monitor = TradingMonitorV2()
    monitor.run()