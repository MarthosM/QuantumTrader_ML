"""
Monitor visual do sistema de trading
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
from datetime import datetime
from typing import Dict, Optional
import queue


class TradingMonitor:
    """Interface gráfica para monitoramento do sistema"""
    
    def __init__(self, trading_system):
        """
        Inicializa o monitor
        
        Args:
            trading_system: Instância do sistema de trading
        """
        self.system = trading_system
        self.root = None
        self.is_running = False
        
        # Queue para atualizações thread-safe
        self.update_queue = queue.Queue()
        
        # Widgets
        self.widgets = {}
        
        # Thread de atualização
        self.update_thread = None
        
        # Cores padrão
        self.colors = {
            'bg': '#1e1e1e',
            'fg': '#ffffff',
            'accent': '#007acc',
            'success': '#4caf50',
            'danger': '#f44336',
            'warning': '#ff9800',
            'frame': '#2d2d2d',
            'input': '#3c3c3c'
        }
        
    def start(self):
        """Inicia o monitor"""
        self.is_running = True
        
        # Criar janela principal
        self.root = tk.Tk()
        self.root.title("Trading System Monitor v2.0")
        self.root.geometry("1200x800")
        
        # Configurar estilo
        self._setup_style()
        
        # Criar interface
        self._create_interface()
        
        # Iniciar thread de atualização
        self.update_thread = threading.Thread(
            target=self._update_worker,
            daemon=True
        )
        self.update_thread.start()
        
        # Iniciar loop de atualização da GUI
        self._update_gui()
        
        # Iniciar mainloop
        self.root.mainloop()
        
    def stop(self):
        """Para o monitor"""
        self.is_running = False
        if self.root:
            self.root.quit()
            
    def _setup_style(self):
        """Configura estilo da interface"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Cores
        self.colors = {
            'bg': '#1e1e1e',
            'fg': '#ffffff',
            'accent': '#007acc',
            'success': '#4caf50',
            'danger': '#f44336',
            'warning': '#ff9800',
            'frame': '#2d2d2d',
            'input': '#3c3c3c'
        }
        
        if self.root:
            self.root.configure(bg=self.colors['bg'])
        
    def _create_interface(self):
        """Cria a interface completa"""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Grid layout
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        
        # 1. Header com informações do sistema
        self._create_header(main_frame)
        
        # 2. Painel de preços e posição
        self._create_price_panel(main_frame)
        
        # 3. Painel de ML e sinais
        self._create_ml_panel(main_frame)
        
        # 4. Painel de métricas
        self._create_metrics_panel(main_frame)
        
        # 5. Log de atividades
        self._create_log_panel(main_frame)
        
        # 6. Controles
        self._create_controls(main_frame)
        
    def _create_header(self, parent):
        """Cria header com informações do sistema"""
        header_frame = tk.Frame(parent, bg=self.colors['frame'], height=60)
        header_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        
        # Status do sistema
        self.widgets['status_label'] = tk.Label(
            header_frame,
            text="SISTEMA INICIANDO...",
            font=('Arial', 16, 'bold'),
            bg=self.colors['frame'],
            fg=self.colors['fg']
        )
        self.widgets['status_label'].pack(side=tk.LEFT, padx=20, pady=10)
        
        # Contrato atual
        self.widgets['contract_label'] = tk.Label(
            header_frame,
            text="Contrato: --",
            font=('Arial', 12),
            bg=self.colors['frame'],
            fg=self.colors['fg']
        )
        self.widgets['contract_label'].pack(side=tk.LEFT, padx=20)
        
        # Uptime
        self.widgets['uptime_label'] = tk.Label(
            header_frame,
            text="Uptime: 00:00:00",
            font=('Arial', 12),
            bg=self.colors['frame'],
            fg=self.colors['fg']
        )
        self.widgets['uptime_label'].pack(side=tk.RIGHT, padx=20)
        
    def _create_price_panel(self, parent):
        """Cria painel de preços e posição"""
        price_frame = tk.LabelFrame(
            parent,
            text="MERCADO & POSIÇÃO",
            font=('Arial', 12, 'bold'),
            bg=self.colors['frame'],
            fg=self.colors['fg']
        )
        price_frame.grid(row=1, column=0, sticky='nsew', padx=(0, 5))
        
        # Preço atual
        price_container = tk.Frame(price_frame, bg=self.colors['frame'])
        price_container.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            price_container,
            text="Preço Atual:",
            font=('Arial', 10),
            bg=self.colors['frame'],
            fg=self.colors['fg']
        ).pack(side=tk.LEFT)
        
        self.widgets['current_price'] = tk.Label(
            price_container,
            text="0.00",
            font=('Arial', 24, 'bold'),
            bg=self.colors['frame'],
            fg=self.colors['accent']
        )
        self.widgets['current_price'].pack(side=tk.LEFT, padx=10)
        
        # Variação
        self.widgets['price_change'] = tk.Label(
            price_container,
            text="0.00 (0.00%)",
            font=('Arial', 12),
            bg=self.colors['frame'],
            fg=self.colors['fg']
        )
        self.widgets['price_change'].pack(side=tk.LEFT, padx=10)
        
        # Máxima/Mínima
        high_low_frame = tk.Frame(price_frame, bg=self.colors['frame'])
        high_low_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.widgets['session_high'] = tk.Label(
            high_low_frame,
            text="Máx: 0.00",
            font=('Arial', 10),
            bg=self.colors['frame'],
            fg=self.colors['success']
        )
        self.widgets['session_high'].pack(side=tk.LEFT, padx=10)
        
        self.widgets['session_low'] = tk.Label(
            high_low_frame,
            text="Mín: 0.00",
            font=('Arial', 10),
            bg=self.colors['frame'],
            fg=self.colors['danger']
        )
        self.widgets['session_low'].pack(side=tk.LEFT, padx=10)
        
        # Separador
        ttk.Separator(price_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)
        
        # Posição atual
        position_label = tk.Label(
            price_frame,
            text="POSIÇÃO ATUAL",
            font=('Arial', 11, 'bold'),
            bg=self.colors['frame'],
            fg=self.colors['fg']
        )
        position_label.pack(pady=5)
        
        self.widgets['position_info'] = tk.Label(
            price_frame,
            text="Sem posição",
            font=('Arial', 10),
            bg=self.colors['frame'],
            fg=self.colors['warning']
        )
        self.widgets['position_info'].pack(pady=5)
        
        # P&L
        self.widgets['position_pnl'] = tk.Label(
            price_frame,
            text="P&L: R$ 0.00",
            font=('Arial', 12, 'bold'),
            bg=self.colors['frame'],
            fg=self.colors['fg']
        )
        self.widgets['position_pnl'].pack(pady=5)
        
    def _create_ml_panel(self, parent):
        """Cria painel de ML e sinais"""
        ml_frame = tk.LabelFrame(
            parent,
            text="ML & SINAIS",
            font=('Arial', 12, 'bold'),
            bg=self.colors['frame'],
            fg=self.colors['fg']
        )
        ml_frame.grid(row=1, column=1, sticky='nsew', padx=(5, 0))
        
        # Última predição
        pred_label = tk.Label(
            ml_frame,
            text="ÚLTIMA PREDIÇÃO",
            font=('Arial', 11, 'bold'),
            bg=self.colors['frame'],
            fg=self.colors['fg']
        )
        pred_label.pack(pady=10)
        
        pred_frame = tk.Frame(ml_frame, bg=self.colors['frame'])
        pred_frame.pack(fill=tk.X, padx=10)
        
        # Direção
        self.widgets['pred_direction'] = tk.Label(
            pred_frame,
            text="Direção: --",
            font=('Arial', 10),
            bg=self.colors['frame'],
            fg=self.colors['fg']
        )
        self.widgets['pred_direction'].pack(anchor=tk.W, pady=2)
        
        # Magnitude
        self.widgets['pred_magnitude'] = tk.Label(
            pred_frame,
            text="Magnitude: --",
            font=('Arial', 10),
            bg=self.colors['frame'],
            fg=self.colors['fg']
        )
        self.widgets['pred_magnitude'].pack(anchor=tk.W, pady=2)
        
        # Confiança
        self.widgets['pred_confidence'] = tk.Label(
            pred_frame,
            text="Confiança: --",
            font=('Arial', 10),
            bg=self.colors['frame'],
            fg=self.colors['fg']
        )
        self.widgets['pred_confidence'].pack(anchor=tk.W, pady=2)
        
        # Tempo
        self.widgets['pred_time'] = tk.Label(
            pred_frame,
            text="Tempo: --",
            font=('Arial', 9),
            bg=self.colors['frame'],
            fg=self.colors['fg']
        )
        self.widgets['pred_time'].pack(anchor=tk.W, pady=2)
        
        # Separador
        ttk.Separator(ml_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)
        
        # Último sinal
        signal_label = tk.Label(
            ml_frame,
            text="ÚLTIMO SINAL",
            font=('Arial', 11, 'bold'),
            bg=self.colors['frame'],
            fg=self.colors['fg']
        )
        signal_label.pack(pady=5)
        
        self.widgets['last_signal'] = tk.Label(
            ml_frame,
            text="Aguardando sinal...",
            font=('Arial', 10),
            bg=self.colors['frame'],
            fg=self.colors['warning']
        )
        self.widgets['last_signal'].pack(pady=5)
        
        # Detalhes do sinal
        signal_details_frame = tk.Frame(ml_frame, bg=self.colors['frame'])
        signal_details_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.widgets['signal_details'] = tk.Label(
            signal_details_frame,
            text="",
            font=('Arial', 9),
            bg=self.colors['frame'],
            fg=self.colors['fg'],
            justify=tk.LEFT
        )
        self.widgets['signal_details'].pack(anchor=tk.W)
        
    def _create_metrics_panel(self, parent):
        """Cria painel de métricas"""
        metrics_frame = tk.LabelFrame(
            parent,
            text="MÉTRICAS DE PERFORMANCE",
            font=('Arial', 12, 'bold'),
            bg=self.colors['frame'],
            fg=self.colors['fg']
        )
        metrics_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=10)
        
        # Grid de métricas
        metrics_grid = tk.Frame(metrics_frame, bg=self.colors['frame'])
        metrics_grid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configurar grid
        for i in range(4):
            metrics_grid.grid_columnconfigure(i, weight=1)
            
        # Métricas
        metrics_info = [
            ('trades_count', 'Trades Processados:', '0'),
            ('predictions_count', 'Predições ML:', '0'),
            ('signals_count', 'Sinais Gerados:', '0'),
            ('executions_count', 'Sinais Executados:', '0'),
            ('win_rate', 'Taxa de Acerto:', '0.0%'),
            ('total_pnl', 'P&L Total:', 'R$ 0.00'),
            ('drawdown', 'Drawdown:', '0.0%'),
            ('errors', 'Erros:', '0')
        ]
        
        row = 0
        col = 0
        for key, label, default in metrics_info:
            container = tk.Frame(metrics_grid, bg=self.colors['frame'])
            container.grid(row=row, column=col, padx=5, pady=5, sticky='w')
            
            tk.Label(
                container,
                text=label,
                font=('Arial', 9),
                bg=self.colors['frame'],
                fg=self.colors['fg']
            ).pack(side=tk.LEFT)
            
            self.widgets[f'metric_{key}'] = tk.Label(
                container,
                text=default,
                font=('Arial', 9, 'bold'),
                bg=self.colors['frame'],
                fg=self.colors['accent']
            )
            self.widgets[f'metric_{key}'].pack(side=tk.LEFT, padx=5)
            
            col += 1
            if col >= 4:
                col = 0
                row += 1
                
    def _create_log_panel(self, parent):
        """Cria painel de log"""
        log_frame = tk.LabelFrame(
            parent,
            text="LOG DE ATIVIDADES",
            font=('Arial', 12, 'bold'),
            bg=self.colors['frame'],
            fg=self.colors['fg']
        )
        log_frame.grid(row=3, column=0, columnspan=2, sticky='nsew', pady=10)
        parent.grid_rowconfigure(3, weight=1)
        
        # Text widget com scroll
        log_container = tk.Frame(log_frame, bg=self.colors['frame'])
        log_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(log_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.widgets['log_text'] = tk.Text(
            log_container,
            height=10,
            font=('Consolas', 9),
            bg=self.colors['input'],
            fg=self.colors['fg'],
            yscrollcommand=scrollbar.set
        )
        self.widgets['log_text'].pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.widgets['log_text'].yview)
        
    def _create_controls(self, parent):
        """Cria controles do sistema"""
        control_frame = tk.Frame(parent, bg=self.colors['frame'])
        control_frame.grid(row=4, column=0, columnspan=2, sticky='ew', pady=10)
        
        # Botão de pausa/resume
        self.widgets['pause_button'] = tk.Button(
            control_frame,
            text="PAUSAR TRADING",
            command=self._toggle_trading,
            font=('Arial', 10, 'bold'),
            bg=self.colors['warning'],
            fg=self.colors['bg'],
            padx=20,
            pady=5
        )
        self.widgets['pause_button'].pack(side=tk.LEFT, padx=10)
        
        # Botão de fechar posições
        self.widgets['close_button'] = tk.Button(
            control_frame,
            text="FECHAR POSIÇÕES",
            command=self._close_all_positions,
            font=('Arial', 10, 'bold'),
            bg=self.colors['danger'],
            fg=self.colors['fg'],
            padx=20,
            pady=5
        )
        self.widgets['close_button'].pack(side=tk.LEFT, padx=10)
        
        # Botão de sair
        self.widgets['exit_button'] = tk.Button(
            control_frame,
            text="SAIR",
            command=self._exit_system,
            font=('Arial', 10, 'bold'),
            bg=self.colors['frame'],
            fg=self.colors['fg'],
            padx=20,
            pady=5
        )
        self.widgets['exit_button'].pack(side=tk.RIGHT, padx=10)
        
    def _update_worker(self):
        """Thread que coleta atualizações do sistema"""
        while self.is_running:
            try:
                # Obter status do sistema
                if self.system:
                    status = self.system.get_status()
                    
                    # Obter métricas
                    if self.system.metrics:
                        metrics = self.system.metrics.get_summary()
                        status['metrics'] = metrics
                        
                    # Adicionar à fila
                    self.update_queue.put(status)
                    
                time.sleep(0.5)  # Atualizar 2x por segundo
                
            except Exception as e:
                print(f"Erro no update worker: {e}")
                
    def _update_gui(self):
        """Atualiza a GUI com dados da fila"""
        try:
            # Processar todas as atualizações pendentes
            while not self.update_queue.empty():
                status = self.update_queue.get_nowait()
                self._apply_updates(status)
                
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Erro atualizando GUI: {e}")
            
        # Agendar próxima atualização
        if self.is_running and self.root is not None:
            self.root.after(100, self._update_gui)
            
    def _apply_updates(self, status: Dict):
        """Aplica atualizações na interface"""
        try:
            # Status do sistema
            if 'status_label' in self.widgets and status.get('running'):
                self.widgets['status_label'].config(
                    text="SISTEMA OPERACIONAL",
                    fg=self.colors['success']
                )
            elif 'status_label' in self.widgets:
                self.widgets['status_label'].config(
                    text="SISTEMA PARADO",
                    fg=self.colors['danger']
                )
                
            # Contrato
            if 'contract_label' in self.widgets:
                self.widgets['contract_label'].config(
                    text=f"Contrato: {status.get('ticker', '--')}"
                )
            
            # Métricas se disponíveis
            if 'metrics' in status:
                metrics = status['metrics']
                
                # Uptime
                if 'uptime_label' in self.widgets:
                    self.widgets['uptime_label'].config(
                        text=f"Uptime: {metrics.get('uptime', '00:00:00')}"
                    )
                
                # Preço
                if 'current_price' in self.widgets:
                    price = metrics.get('current_price', 0)
                    self.widgets['current_price'].config(text=f"{price:.2f}")
                
                # Máxima/Mínima
                if 'session_high' in self.widgets:
                    self.widgets['session_high'].config(
                        text=f"Máx: {metrics.get('session_high', 0):.2f}"
                    )
                if 'session_low' in self.widgets:
                    self.widgets['session_low'].config(
                        text=f"Mín: {metrics.get('session_low', 0):.2f}"
                    )
                
                # Métricas de trading
                if 'metric_trades_count' in self.widgets:
                    self.widgets['metric_trades_count'].config(
                        text=str(metrics.get('trades_processed', 0))
                    )
                if 'metric_predictions_count' in self.widgets:
                    self.widgets['metric_predictions_count'].config(
                        text=str(metrics.get('predictions_made', 0))
                    )
                if 'metric_signals_count' in self.widgets:
                    self.widgets['metric_signals_count'].config(
                        text=str(metrics.get('signals_generated', 0))
                    )
                if 'metric_executions_count' in self.widgets:
                    self.widgets['metric_executions_count'].config(
                        text=str(metrics.get('signals_executed', 0))
                    )
                if 'metric_win_rate' in self.widgets:
                    self.widgets['metric_win_rate'].config(
                        text=f"{metrics.get('win_rate', 0):.1f}%"
                    )
                if 'metric_total_pnl' in self.widgets:
                    self.widgets['metric_total_pnl'].config(
                        text=f"R$ {metrics.get('total_pnl', 0):.2f}"
                    )
                if 'metric_drawdown' in self.widgets:
                    self.widgets['metric_drawdown'].config(
                        text=f"{metrics.get('current_drawdown', 0):.1f}%"
                    )
                if 'metric_errors' in self.widgets:
                    self.widgets['metric_errors'].config(
                        text=str(metrics.get('errors_count', 0))
                    )
                
            # Posições
            positions = status.get('active_positions', {})
            if positions and 'position_info' in self.widgets:
                for ticker, pos in positions.items():
                    self.widgets['position_info'].config(
                        text=f"{pos['side'].upper()} {pos.get('size', 1)} @ {pos['entry_price']:.2f}",
                        fg=self.colors['success'] if pos['side'] == 'buy' else self.colors['danger']
                    )
                    
                    # Calcular P&L (simplificado)
                    if 'metrics' in status and 'position_pnl' in self.widgets:
                        current_price = status['metrics'].get('current_price', 0)
                        if pos['side'] == 'buy':
                            pnl = (current_price - pos['entry_price']) * pos.get('size', 1)
                        else:
                            pnl = (pos['entry_price'] - current_price) * pos.get('size', 1)
                            
                        self.widgets['position_pnl'].config(
                            text=f"P&L: R$ {pnl:.2f}",
                            fg=self.colors['success'] if pnl > 0 else self.colors['danger']
                        )
            else:
                if 'position_info' in self.widgets:
                    self.widgets['position_info'].config(
                        text="Sem posição",
                        fg=self.colors['warning']
                    )
                if 'position_pnl' in self.widgets:
                    self.widgets['position_pnl'].config(
                        text="P&L: R$ 0.00",
                        fg=self.colors['fg']
                    )
                
            # Última predição
            if status.get('last_prediction'):
                pred = status['last_prediction']
                
                # Direção
                direction = pred.get('direction', 0)
                if direction > 0:
                    dir_text = f"ALTA ({direction:.2f})"
                    dir_color = self.colors['success']
                elif direction < 0:
                    dir_text = f"BAIXA ({direction:.2f})"
                    dir_color = self.colors['danger']
                else:
                    dir_text = f"NEUTRO ({direction:.2f})"
                    dir_color = self.colors['warning']
                    
                if 'pred_direction' in self.widgets:
                    self.widgets['pred_direction'].config(
                        text=f"Direção: {dir_text}",
                        fg=dir_color
                    )
                
                # Magnitude
                if 'pred_magnitude' in self.widgets:
                    self.widgets['pred_magnitude'].config(
                        text=f"Magnitude: {pred.get('magnitude', 0):.4f}"
                    )
                
                # Confiança
                if 'pred_confidence' in self.widgets:
                    confidence = pred.get('confidence', 0)
                    self.widgets['pred_confidence'].config(
                        text=f"Confiança: {confidence:.1%}"
                    )
                
                # Tempo
                if 'pred_time' in self.widgets and 'timestamp' in pred:
                    time_str = pred['timestamp'].strftime('%H:%M:%S')
                    self.widgets['pred_time'].config(text=f"Tempo: {time_str}")
                    
            # Últimos sinais
            if self.system.metrics:
                signals = self.system.metrics.get_recent_signals(1)
                if signals:
                    signal = signals[0]
                    action = signal.get('action', 'none')
                    
                    if action != 'none':
                        signal_text = f"{action.upper()} @ {signal.get('price', 0):.2f}"
                        signal_color = self.colors['success'] if action == 'buy' else self.colors['danger']
                        
                        self.widgets['last_signal'].config(
                            text=signal_text,
                            fg=signal_color
                        )
                        
                        # Detalhes
                        details = (
                            f"SL: {signal.get('stop_loss', 0):.2f}\n"
                            f"TP: {signal.get('take_profit', 0):.2f}\n"
                            f"R:R: {signal.get('risk_reward', 0):.1f}"
                        )
                        self.widgets['signal_details'].config(text=details)
                        
        except Exception as e:
            print(f"Erro aplicando updates: {e}")
            
    def _toggle_trading(self):
        """Pausa/resume trading"""
        # Implementar lógica de pausa
        pass
        
    def _close_all_positions(self):
        """Fecha todas as posições"""
        # Implementar fechamento de posições
        pass
        
    def _exit_system(self):
        """Sai do sistema"""
        self.stop()
        if self.system:
            self.system.stop()
            
    def log_message(self, message: str, level: str = 'INFO'):
        """Adiciona mensagem ao log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Determinar cor baseada no nível
        color_tags = {
            'INFO': 'info',
            'WARNING': 'warning',
            'ERROR': 'error',
            'SUCCESS': 'success'
        }
        
        # Adicionar ao text widget
        self.widgets['log_text'].insert(
            tk.END,
            f"[{timestamp}] {level}: {message}\n"
        )
        
        # Auto-scroll
        self.widgets['log_text'].see(tk.END)
        
        # Limitar tamanho do log
        lines = int(self.widgets['log_text'].index('end-1c').split('.')[0])
        if lines > 1000:
            self.widgets['log_text'].delete('1.0', '2.0')