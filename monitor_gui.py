"""
Monitor GUI - Dashboard em tempo real para o QuantumTrader ML
Acompanha predições, trades e status do sistema
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import json
import time
from datetime import datetime
from pathlib import Path
import threading
import queue
from collections import deque

class TradingMonitorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("QuantumTrader ML - Monitor")
        self.root.geometry("1200x800")
        
        # Estilo
        self.root.configure(bg='#1e1e1e')
        style = ttk.Style()
        style.theme_use('clam')
        
        # Queue para thread-safe updates
        self.update_queue = queue.Queue()
        
        # Dados
        self.predictions_history = deque(maxlen=50)
        self.trades_history = deque(maxlen=100)
        self.current_data = {
            'price': 0,
            'position': 0,
            'pnl': 0,
            'daily_pnl': 0,
            'last_prediction': None,
            'status': 'Desconectado'
        }
        
        # Criar interface
        self._create_widgets()
        
        # Iniciar thread de leitura de logs
        self.running = True
        self.log_thread = threading.Thread(target=self._read_logs, daemon=True)
        self.log_thread.start()
        
        # Atualizar GUI
        self._update_gui()
        
    def _create_widgets(self):
        """Cria todos os widgets da interface"""
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top Frame - Status e Métricas
        top_frame = tk.Frame(main_frame, bg='#1e1e1e')
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Status Cards
        self._create_status_card(top_frame, "Status", "status", 0, 0)
        self._create_status_card(top_frame, "Preço", "price", 0, 1)
        self._create_status_card(top_frame, "Posição", "position", 0, 2)
        self._create_status_card(top_frame, "P&L Diário", "daily_pnl", 0, 3)
        
        # Middle Frame - Gráficos
        middle_frame = tk.Frame(main_frame, bg='#1e1e1e')
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Predictions Frame
        pred_frame = tk.LabelFrame(middle_frame, text="Predições ML", 
                                  bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold'))
        pred_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Canvas para gráfico de predições
        self.pred_canvas = tk.Canvas(pred_frame, bg='#2d2d2d', highlightthickness=0)
        self.pred_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Trades Frame
        trades_frame = tk.LabelFrame(middle_frame, text="Histórico de Trades", 
                                    bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold'))
        trades_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Treeview para trades
        self.trades_tree = ttk.Treeview(trades_frame, columns=('time', 'side', 'price', 'pnl'),
                                       show='headings', height=10)
        self.trades_tree.heading('time', text='Hora')
        self.trades_tree.heading('side', text='Lado')
        self.trades_tree.heading('price', text='Preço')
        self.trades_tree.heading('pnl', text='P&L')
        
        self.trades_tree.column('time', width=100)
        self.trades_tree.column('side', width=80)
        self.trades_tree.column('price', width=100)
        self.trades_tree.column('pnl', width=100)
        
        self.trades_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bottom Frame - Logs
        bottom_frame = tk.LabelFrame(main_frame, text="Logs do Sistema", 
                                   bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold'))
        bottom_frame.pack(fill=tk.BOTH, expand=True)
        
        # Text area para logs
        self.log_text = scrolledtext.ScrolledText(bottom_frame, bg='#1e1e1e', fg='#00ff00',
                                                 font=('Consolas', 9), height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tags para colorir logs
        self.log_text.tag_config('error', foreground='#ff4444')
        self.log_text.tag_config('warning', foreground='#ffaa00')
        self.log_text.tag_config('success', foreground='#44ff44')
        self.log_text.tag_config('ml', foreground='#44aaff')
        self.log_text.tag_config('trade', foreground='#ffff44')
        
    def _create_status_card(self, parent, title, key, row, col):
        """Cria um card de status"""
        frame = tk.Frame(parent, bg='#2d2d2d', relief=tk.RAISED, bd=1)
        frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
        parent.grid_columnconfigure(col, weight=1)
        
        # Título
        title_label = tk.Label(frame, text=title, bg='#2d2d2d', fg='#888888',
                             font=('Arial', 10))
        title_label.pack(pady=(10, 5))
        
        # Valor
        value_label = tk.Label(frame, text="--", bg='#2d2d2d', fg='white',
                             font=('Arial', 16, 'bold'))
        value_label.pack(pady=(0, 10))
        
        # Guardar referência
        setattr(self, f'{key}_label', value_label)
        
    def _read_logs(self):
        """Thread que lê os logs mais recentes"""
        # Encontrar arquivo de log mais recente
        log_dir = Path('logs/production')
        
        while self.running:
            try:
                # Procurar arquivo de log mais recente
                log_files = list(log_dir.glob('final_*.log'))
                if not log_files:
                    time.sleep(1)
                    continue
                    
                latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
                
                # Ler últimas linhas
                with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                    # Ir para o final do arquivo
                    f.seek(0, 2)
                    
                    while self.running:
                        line = f.readline()
                        if line:
                            self._process_log_line(line)
                        else:
                            time.sleep(0.1)
                            
            except Exception as e:
                print(f"Erro ao ler logs: {e}")
                time.sleep(1)
                
    def _process_log_line(self, line):
        """Processa uma linha de log"""
        try:
            # Adicionar linha ao queue de logs
            self.update_queue.put(('log', line))
            
            # Extrair informações importantes
            if '[STATUS]' in line:
                # Extrair métricas de status
                if 'Price:' in line:
                    price = float(line.split('Price:')[1].split()[0])
                    self.update_queue.put(('price', price))
                    
                if 'Pos:' in line:
                    pos = int(line.split('Pos:')[1].split()[0])
                    self.update_queue.put(('position', pos))
                    
            elif '[ML]' in line:
                # Extrair predição
                if 'Dir:' in line:
                    parts = line.split('|')
                    direction = float(parts[0].split('Dir:')[1].strip())
                    confidence = float(parts[1].split('Conf:')[1].strip())
                    models = int(parts[2].split('Models:')[1].strip())
                    
                    prediction = {
                        'time': datetime.now(),
                        'direction': direction,
                        'confidence': confidence,
                        'models': models
                    }
                    self.update_queue.put(('prediction', prediction))
                    
            elif '[ORDER]' in line:
                # Extrair ordem
                side = 'BUY' if 'COMPRA' in line else 'SELL'
                price = float(line.split('@')[1].strip())
                
                trade = {
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'side': side,
                    'price': price,
                    'pnl': 0
                }
                self.update_queue.put(('trade', trade))
                
            elif '[P&L]' in line:
                # Extrair P&L
                if 'Diário:' in line:
                    daily_pnl = float(line.split('R$')[2].strip())
                    self.update_queue.put(('daily_pnl', daily_pnl))
                    
            elif 'SISTEMA OPERACIONAL' in line:
                self.update_queue.put(('status', 'Operacional'))
                
        except Exception as e:
            print(f"Erro ao processar linha: {e}")
            
    def _update_gui(self):
        """Atualiza a interface com dados do queue"""
        try:
            # Processar até 10 items do queue
            for _ in range(10):
                if self.update_queue.empty():
                    break
                    
                update_type, data = self.update_queue.get_nowait()
                
                if update_type == 'log':
                    self._add_log(data)
                elif update_type == 'price':
                    self.current_data['price'] = data
                    self.price_label.config(text=f"R$ {data:.2f}")
                elif update_type == 'position':
                    self.current_data['position'] = data
                    self.position_label.config(text=str(data))
                    color = '#44ff44' if data == 0 else '#ffaa00'
                    self.position_label.config(fg=color)
                elif update_type == 'daily_pnl':
                    self.current_data['daily_pnl'] = data
                    color = '#44ff44' if data >= 0 else '#ff4444'
                    self.daily_pnl_label.config(text=f"R$ {data:.2f}", fg=color)
                elif update_type == 'status':
                    self.current_data['status'] = data
                    self.status_label.config(text=data, fg='#44ff44')
                elif update_type == 'prediction':
                    self.predictions_history.append(data)
                    self._update_prediction_chart()
                elif update_type == 'trade':
                    self.trades_history.append(data)
                    self._add_trade(data)
                    
        except queue.Empty:
            pass
            
        # Agendar próxima atualização
        if self.running:
            self.root.after(100, self._update_gui)
            
    def _add_log(self, line):
        """Adiciona linha ao log com coloração"""
        self.log_text.insert(tk.END, line)
        
        # Aplicar tags de cor
        if '[ERROR]' in line or 'ERRO' in line:
            tag = 'error'
        elif '[WARNING]' in line or 'WARN' in line:
            tag = 'warning'
        elif '[OK]' in line or 'SUCCESS' in line:
            tag = 'success'
        elif '[ML]' in line:
            tag = 'ml'
        elif '[ORDER]' in line or '[TRADE]' in line:
            tag = 'trade'
        else:
            tag = None
            
        if tag:
            # Colorir última linha
            line_start = self.log_text.index("end-2c linestart")
            line_end = self.log_text.index("end-1c")
            self.log_text.tag_add(tag, line_start, line_end)
            
        # Auto-scroll
        self.log_text.see(tk.END)
        
        # Limitar tamanho do log
        if int(self.log_text.index('end-1c').split('.')[0]) > 1000:
            self.log_text.delete('1.0', '100.0')
            
    def _add_trade(self, trade):
        """Adiciona trade ao histórico"""
        values = (trade['time'], trade['side'], f"R$ {trade['price']:.2f}", 
                 f"R$ {trade.get('pnl', 0):.2f}")
        self.trades_tree.insert('', 0, values=values)
        
        # Limitar número de trades mostrados
        if len(self.trades_tree.get_children()) > 50:
            self.trades_tree.delete(self.trades_tree.get_children()[-1])
            
    def _update_prediction_chart(self):
        """Atualiza gráfico de predições"""
        self.pred_canvas.delete('all')
        
        if len(self.predictions_history) < 2:
            return
            
        # Dimensões do canvas
        width = self.pred_canvas.winfo_width()
        height = self.pred_canvas.winfo_height()
        
        if width < 100 or height < 100:
            return
            
        # Margens
        margin = 20
        chart_width = width - 2 * margin
        chart_height = height - 2 * margin
        
        # Desenhar grid
        self.pred_canvas.create_line(margin, height/2, width-margin, height/2, 
                                   fill='#444444', dash=(2, 2))
        self.pred_canvas.create_text(margin-10, height/2, text="0.5", 
                                   fill='white', anchor='e')
        
        # Desenhar predições
        points = []
        conf_points = []
        
        for i, pred in enumerate(self.predictions_history):
            x = margin + (i * chart_width / (len(self.predictions_history) - 1))
            
            # Direction (0-1)
            y_dir = margin + (1 - pred['direction']) * chart_height
            points.append((x, y_dir))
            
            # Confidence (0-1)
            y_conf = margin + (1 - pred['confidence']) * chart_height
            conf_points.append((x, y_conf))
            
        # Desenhar linha de direção
        if len(points) > 1:
            for i in range(len(points) - 1):
                self.pred_canvas.create_line(points[i][0], points[i][1],
                                           points[i+1][0], points[i+1][1],
                                           fill='#44aaff', width=2)
                                           
        # Desenhar linha de confiança
        if len(conf_points) > 1:
            for i in range(len(conf_points) - 1):
                self.pred_canvas.create_line(conf_points[i][0], conf_points[i][1],
                                           conf_points[i+1][0], conf_points[i+1][1],
                                           fill='#ffaa00', width=1, dash=(3, 2))
                                           
        # Última predição
        if self.predictions_history:
            last = self.predictions_history[-1]
            text = f"Dir: {last['direction']:.3f} | Conf: {last['confidence']:.3f}"
            self.pred_canvas.create_text(width/2, margin-5, text=text,
                                       fill='white', font=('Arial', 10))
                                       
    def run(self):
        """Inicia o monitor"""
        self.root.mainloop()
        
    def stop(self):
        """Para o monitor"""
        self.running = False
        self.root.quit()

if __name__ == "__main__":
    monitor = TradingMonitorGUI()
    try:
        monitor.run()
    except KeyboardInterrupt:
        monitor.stop()