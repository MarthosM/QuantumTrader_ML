import tkinter as tk
from tkinter import ttk, font, messagebox
import threading
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from collections import deque
import numpy as np



# ML Flow Integration
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from gui_prediction_extension import extend_gui_with_prediction_display
except ImportError:
    def extend_gui_with_prediction_display(gui):
        return False


class TradingMonitorGUI:
    """
    Monitor GUI para Trading System ML v2.0
    Exibe predições ML, dados de candles e métricas em tempo real
    """
    
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.running = False
        self.update_interval = 1  # segundos
        
        # Buffers para dados históricos (últimos 100 registros)
        self.predictions_history = deque(maxlen=100)
        self.candles_history = deque(maxlen=100)
        self.alerts_history = deque(maxlen=50)
        
        # Setup da interface
        self.root = tk.Tk()
        self._setup_window()
        self._configure_styles()
        self._create_widgets()
        
        # Thread de atualização
        self.update_thread = None
        
        # Dados atuais
        self.current_data = {
            'last_prediction': None,
            'last_candle': None,
            'system_status': 'Inicializando...',
            'account_info': {},
            'positions': {}
        }
       
        
        # 🔧 ML FLOW INTEGRATION - Estender GUI com painéis de predição
        try:
            if extend_gui_with_prediction_display(self):
                self.logger.info("✅ GUI estendido com painéis ML")
            else:
                self.logger.info("ℹ️ Extensão ML não disponível")
        except Exception as e:
            self.logger.warning(f"Erro estendendo GUI: {e}")
 
    def _setup_window(self):
        """Configura janela principal"""
        self.root.title("Trading System ML v2.0 - Monitor")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Configurar para fechar adequadamente
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Ícone (opcional)
        try:
            # self.root.iconbitmap("trading_icon.ico")  # Descomente se tiver ícone
            pass
        except:
            pass
            
    def _configure_styles(self):
        """Configura estilos visuais"""
        self.style = ttk.Style()
        
        # Cores do tema
        self.colors = {
            'profit': '#00FF00',
            'loss': '#FF0000',
            'neutral': '#FFFF00',
            'bg_dark': '#2b2b2b',
            'bg_light': '#404040',
            'text': '#FFFFFF',
            'accent': '#007ACC'
        }
        
        # Configurar fontes
        self.fonts = {
            'title': font.Font(family='Arial', size=14, weight='bold'),
            'data': font.Font(family='Courier', size=10),
            'status': font.Font(family='Arial', size=12, weight='bold')
        }
        
    def _create_widgets(self):
        """Cria todos os widgets da interface"""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === HEADER ===
        self._create_header(main_frame)
        
        # === CONTENT FRAMES ===
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Frame esquerdo (Predições + Candles)
        left_frame = ttk.LabelFrame(content_frame, text="Dados de Trading", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Frame direito (Métricas + Alertas)
        right_frame = ttk.LabelFrame(content_frame, text="Monitoramento", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Criar seções específicas
        self._create_prediction_section(left_frame)
        self._create_candle_section(left_frame)
        self._create_metrics_section(right_frame)
        self._create_alerts_section(right_frame)
        
        # === FOOTER ===
        self._create_footer(main_frame)
        
    def _create_header(self, parent):
        """Cria header com status do sistema"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Título
        title_label = tk.Label(header_frame, text="Trading System ML v2.0", 
                              font=self.fonts['title'], fg=self.colors['accent'])
        title_label.pack(side=tk.LEFT)
        
        # Status
        self.status_label = tk.Label(header_frame, text="Sistema: Inicializando...", 
                                    font=self.fonts['status'], fg=self.colors['neutral'])
        self.status_label.pack(side=tk.RIGHT)
        
        # Separador
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=5)
        
    def _create_prediction_section(self, parent):
        """Cria seção de predições ML"""
        pred_frame = ttk.LabelFrame(parent, text="🎯 Última Predição ML", padding=10)
        pred_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Grid para organizar dados
        pred_grid = ttk.Frame(pred_frame)
        pred_grid.pack(fill=tk.X)
        
        # Labels para dados da predição
        self.pred_labels = {}
        
        # Linha 1: Direção e Confiança
        tk.Label(pred_grid, text="Direção:", font=self.fonts['data']).grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.pred_labels['direction'] = tk.Label(pred_grid, text="-", font=self.fonts['data'], fg=self.colors['neutral'])
        self.pred_labels['direction'].grid(row=0, column=1, sticky='w', padx=(0, 20))
        
        tk.Label(pred_grid, text="Confiança:", font=self.fonts['data']).grid(row=0, column=2, sticky='w', padx=(0, 10))
        self.pred_labels['confidence'] = tk.Label(pred_grid, text="-", font=self.fonts['data'])
        self.pred_labels['confidence'].grid(row=0, column=3, sticky='w')
        
        # Linha 2: Magnitude e Ação
        tk.Label(pred_grid, text="Magnitude:", font=self.fonts['data']).grid(row=1, column=0, sticky='w', padx=(0, 10))
        self.pred_labels['magnitude'] = tk.Label(pred_grid, text="-", font=self.fonts['data'])
        self.pred_labels['magnitude'].grid(row=1, column=1, sticky='w', padx=(0, 20))
        
        tk.Label(pred_grid, text="Ação:", font=self.fonts['data']).grid(row=1, column=2, sticky='w', padx=(0, 10))
        self.pred_labels['action'] = tk.Label(pred_grid, text="-", font=self.fonts['data'], fg=self.colors['neutral'])
        self.pred_labels['action'].grid(row=1, column=3, sticky='w')
        
        # Linha 3: Regime e Timestamp
        tk.Label(pred_grid, text="Regime:", font=self.fonts['data']).grid(row=2, column=0, sticky='w', padx=(0, 10))
        self.pred_labels['regime'] = tk.Label(pred_grid, text="-", font=self.fonts['data'])
        self.pred_labels['regime'].grid(row=2, column=1, sticky='w', padx=(0, 20))
        
        tk.Label(pred_grid, text="Timestamp:", font=self.fonts['data']).grid(row=2, column=2, sticky='w', padx=(0, 10))
        self.pred_labels['timestamp'] = tk.Label(pred_grid, text="-", font=self.fonts['data'])
        self.pred_labels['timestamp'].grid(row=2, column=3, sticky='w')
        
    def _create_candle_section(self, parent):
        """Cria seção de dados do último candle"""
        candle_frame = ttk.LabelFrame(parent, text="📈 Dados de Mercado em Tempo Real", padding=10)
        candle_frame.pack(fill=tk.X, pady=(0, 10))

        # === PREÇO ATUAL ===
        current_price_frame = ttk.LabelFrame(candle_frame, text="💰 Preço Atual", padding=5)
        current_price_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Grid para preço atual
        price_grid = ttk.Frame(current_price_frame)
        price_grid.pack(fill=tk.X)
        
        # Preço atual (destaque)
        tk.Label(price_grid, text="Preço:", font=self.fonts['title']).grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.current_price_label = tk.Label(price_grid, text="R$ -.--", font=font.Font(family='Arial', size=16, weight='bold'), 
                                          fg=self.colors['accent'])
        self.current_price_label.grid(row=0, column=1, sticky='w', padx=(0, 20))
        
        # Variação do dia
        tk.Label(price_grid, text="Var. Dia:", font=self.fonts['data']).grid(row=0, column=2, sticky='w', padx=(0, 10))
        self.day_variation_label = tk.Label(price_grid, text="-.--% (R$ -.--)", font=self.fonts['data'])
        self.day_variation_label.grid(row=0, column=3, sticky='w', padx=(0, 20))
        
        # Última atualização do preço
        tk.Label(price_grid, text="Atualizado:", font=self.fonts['data']).grid(row=0, column=4, sticky='w', padx=(0, 5))
        self.price_update_time = tk.Label(price_grid, text="--:--:--", font=self.fonts['data'])
        self.price_update_time.grid(row=0, column=5, sticky='w')

        # === ÚLTIMO CANDLE ===
        last_candle_frame = ttk.LabelFrame(candle_frame, text="📊 Último Candle (1 min)", padding=5)
        last_candle_frame.pack(fill=tk.X)

        # Grid para dados do candle
        candle_grid = ttk.Frame(last_candle_frame)
        candle_grid.pack(fill=tk.X)

        self.candle_labels = {}

        # Linha 1: OHLC
        labels_row1 = ['Open', 'High', 'Low', 'Close']
        for i, label in enumerate(labels_row1):
            tk.Label(candle_grid, text=f"{label}:", font=self.fonts['data']).grid(row=0, column=i*2, sticky='w', padx=(0, 5))
            self.candle_labels[label.lower()] = tk.Label(candle_grid, text="-.--", font=self.fonts['data'], fg=self.colors['text'])
            self.candle_labels[label.lower()].grid(row=0, column=i*2+1, sticky='w', padx=(0, 15))

        # Linha 2: Volume e Dados adicionais
        tk.Label(candle_grid, text="Volume:", font=self.fonts['data']).grid(row=1, column=0, sticky='w', padx=(0, 5))
        self.candle_labels['volume'] = tk.Label(candle_grid, text="-", font=self.fonts['data'])
        self.candle_labels['volume'].grid(row=1, column=1, sticky='w', padx=(0, 15))

        tk.Label(candle_grid, text="Trades:", font=self.fonts['data']).grid(row=1, column=2, sticky='w', padx=(0, 5))
        self.candle_labels['trades'] = tk.Label(candle_grid, text="-", font=self.fonts['data'])
        self.candle_labels['trades'].grid(row=1, column=3, sticky='w', padx=(0, 15))

        tk.Label(candle_grid, text="Var %:", font=self.fonts['data']).grid(row=1, column=4, sticky='w', padx=(0, 5))
        self.candle_labels['variation'] = tk.Label(candle_grid, text="-.--", font=self.fonts['data'])
        self.candle_labels['variation'].grid(row=1, column=5, sticky='w', padx=(0, 15))

        # Linha 3: Buy/Sell Volume e Timestamp
        tk.Label(candle_grid, text="Buy Vol:", font=self.fonts['data']).grid(row=2, column=0, sticky='w', padx=(0, 5))
        self.candle_labels['buy_volume'] = tk.Label(candle_grid, text="-", font=self.fonts['data'], fg=self.colors['profit'])
        self.candle_labels['buy_volume'].grid(row=2, column=1, sticky='w', padx=(0, 15))
        
        tk.Label(candle_grid, text="Sell Vol:", font=self.fonts['data']).grid(row=2, column=2, sticky='w', padx=(0, 5))
        self.candle_labels['sell_volume'] = tk.Label(candle_grid, text="-", font=self.fonts['data'], fg=self.colors['loss'])
        self.candle_labels['sell_volume'].grid(row=2, column=3, sticky='w', padx=(0, 15))

        tk.Label(candle_grid, text="Timestamp:", font=self.fonts['data']).grid(row=2, column=4, sticky='w', padx=(0, 5))
        self.candle_labels['timestamp'] = tk.Label(candle_grid, text="--:--:--", font=self.fonts['data'])
        self.candle_labels['timestamp'].grid(row=2, column=5, sticky='w')
        
    def _create_metrics_section(self, parent):
        """Cria seção de métricas do sistema"""
        metrics_frame = ttk.LabelFrame(parent, text="📊 Métricas do Sistema", padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Notebook para organizar métricas em abas
        self.metrics_notebook = ttk.Notebook(metrics_frame)
        self.metrics_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Aba Trading
        trading_frame = ttk.Frame(self.metrics_notebook)
        self.metrics_notebook.add(trading_frame, text="Trading")
        self._create_trading_metrics(trading_frame)
        
        # Aba Sistema
        system_frame = ttk.Frame(self.metrics_notebook)
        self.metrics_notebook.add(system_frame, text="Sistema")
        self._create_system_metrics(system_frame)
        
        # Aba Posições
        positions_frame = ttk.Frame(self.metrics_notebook)
        self.metrics_notebook.add(positions_frame, text="Posições")
        self._create_positions_display(positions_frame)
        
    def _create_trading_metrics(self, parent):
        """Cria métricas de trading"""
        self.trading_metrics = {}
        
        # Grid para métricas
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        metrics_data = [
            ('P&L Diário:', 'daily_pnl'),
            ('Trades Hoje:', 'trades_today'),
            ('Win Rate:', 'win_rate'),
            ('Posições Ativas:', 'active_positions'),
            ('Saldo:', 'balance'),
            ('Disponível:', 'available')
        ]
        
        for i, (label, key) in enumerate(metrics_data):
            tk.Label(grid, text=label, font=self.fonts['data']).grid(row=i, column=0, sticky='w', padx=(0, 10), pady=2)
            self.trading_metrics[key] = tk.Label(grid, text="-", font=self.fonts['data'])
            self.trading_metrics[key].grid(row=i, column=1, sticky='w', pady=2)
            
    def _create_system_metrics(self, parent):
        """Cria métricas do sistema"""
        self.system_metrics = {}
        
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        system_data = [
            ('CPU:', 'cpu_percent'),
            ('Memória:', 'memory_mb'),
            ('Threads:', 'threads'),
            ('Uptime:', 'uptime'),
            ('ML Predictions:', 'ml_predictions'),
            ('Sinais Gerados:', 'signals_generated')
        ]
        
        for i, (label, key) in enumerate(system_data):
            tk.Label(grid, text=label, font=self.fonts['data']).grid(row=i, column=0, sticky='w', padx=(0, 10), pady=2)
            self.system_metrics[key] = tk.Label(grid, text="-", font=self.fonts['data'])
            self.system_metrics[key].grid(row=i, column=1, sticky='w', pady=2)
            
    def _create_positions_display(self, parent):
        """Cria display de posições ativas"""
        # Treeview para mostrar posições
        columns = ('Symbol', 'Side', 'Entry', 'Current', 'P&L', 'Size')
        self.positions_tree = ttk.Treeview(parent, columns=columns, show='headings', height=6)
        
        # Configurar colunas
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=80, anchor='center')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack
        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
        
    def _create_alerts_section(self, parent):
        """Cria seção de alertas"""
        alerts_frame = ttk.LabelFrame(parent, text="⚠️ Alertas", padding=10)
        alerts_frame.pack(fill=tk.BOTH, expand=True)
        
        # Listbox para alertas
        listbox_frame = ttk.Frame(alerts_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        self.alerts_listbox = tk.Listbox(listbox_frame, font=self.fonts['data'], height=8)
        alerts_scrollbar = ttk.Scrollbar(listbox_frame, orient='vertical', command=self.alerts_listbox.yview)
        self.alerts_listbox.configure(yscrollcommand=alerts_scrollbar.set)
        
        self.alerts_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        alerts_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def _create_footer(self, parent):
        """Cria footer com controles"""
        footer_frame = ttk.Frame(parent)
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Separador
        ttk.Separator(footer_frame, orient='horizontal').pack(fill=tk.X, pady=(0, 10))
        
        # Botões de controle
        controls_frame = ttk.Frame(footer_frame)
        controls_frame.pack()
        
        self.start_button = ttk.Button(controls_frame, text="▶ Iniciar Monitor", command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(controls_frame, text="⏸ Parar Monitor", command=self.stop_monitoring, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status de atualização
        self.last_update_label = tk.Label(controls_frame, text="Última atualização: -", font=self.fonts['data'])
        self.last_update_label.pack(side=tk.RIGHT)
        
    def start_monitoring(self):
        """Inicia monitoramento em tempo real"""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            
            # Atualizar botões
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            
            self._update_status("Sistema: Monitorando", self.colors['profit'])
            
    def stop_monitoring(self):
        """Para monitoramento"""
        self.running = False
        
        # Atualizar botões
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        self._update_status("Sistema: Parado", self.colors['loss'])
        
    def stop(self):
        """Para o monitor GUI completamente"""
        self.stop_monitoring()
        if self.update_thread and self.update_thread.is_alive():
            self.running = False
            self.update_thread.join(timeout=2)
            
    def destroy(self):
        """Destroir window completely"""
        self.stop()
        if self.root:
            self.root.destroy()
        
    def _update_loop(self):
        """Loop principal de atualização"""
        while self.running:
            try:
                # Coletar dados do sistema de trading
                self._collect_trading_data()
                
                # Atualizar interface
                self.root.after(0, self._update_interface)
                
                # Aguardar intervalo
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Erro no loop de atualização GUI: {e}")
                # Continuar mesmo com erro
                time.sleep(1)
                
    def _collect_trading_data(self):
        """Coleta dados atuais do trading system"""
        try:
            # Última predição
            if hasattr(self.trading_system, 'last_prediction') and self.trading_system.last_prediction:
                self.current_data['last_prediction'] = self.trading_system.last_prediction.copy()

            # 🔧 CORREÇÃO: Tentar múltiplas fontes de dados de candle
            candles_df = None
            
            # Método 1: data_structure.candles (estrutura padrão)
            if (hasattr(self.trading_system, 'data_structure') and 
                self.trading_system.data_structure and 
                hasattr(self.trading_system.data_structure, 'candles') and 
                not self.trading_system.data_structure.candles.empty):
                candles_df = self.trading_system.data_structure.candles
                
            # Método 2: data_loader.candles_df (usado nos logs)
            elif (hasattr(self.trading_system, 'data_loader') and
                  self.trading_system.data_loader and
                  hasattr(self.trading_system.data_loader, 'candles_df') and
                  not self.trading_system.data_loader.candles_df.empty):
                candles_df = self.trading_system.data_loader.candles_df
                
            # Método 3: Verificar se há método get_candles_df
            elif hasattr(self.trading_system, 'get_candles_df'):
                try:
                    candles_df = self.trading_system.get_candles_df()
                except:
                    pass
                    
            # Se encontrou dados de candle, processar
            if candles_df is not None and not candles_df.empty:
                print(f"📊 GUI: Coletando dados - {len(candles_df)} candles disponíveis")
                
                # Último candle
                last_candle = candles_df.iloc[-1]
                self.current_data['last_candle'] = last_candle.to_dict()
                
                # Preço atual (close do último candle)
                current_price = last_candle['close']
                self.current_data['current_price'] = current_price
                print(f"💰 GUI: Preço atual = R$ {current_price}")
                
                # Calcular estatísticas do dia se tivermos dados suficientes
                if len(candles_df) > 1:
                    # Primeiro candle do dia (aproximação - primeiros candles disponíveis)
                    today_candles = candles_df.tail(min(600, len(candles_df)))  # Últimas ~10 horas de candles
                    if not today_candles.empty:
                        day_open = today_candles.iloc[0]['open']
                        day_high = today_candles['high'].max()
                        day_low = today_candles['low'].min()
                        day_volume = today_candles['volume'].sum()
                        
                        # Variação do dia
                        day_variation_pct = ((current_price - day_open) / day_open) * 100 if day_open > 0 else 0
                        day_variation_value = current_price - day_open
                        
                        self.current_data['day_stats'] = {
                            'open': day_open,
                            'high': day_high,
                            'low': day_low,
                            'current': current_price,
                            'variation_pct': day_variation_pct,
                            'variation_value': day_variation_value,
                            'volume': day_volume
                        }
                
                # Timestamp da última atualização
                self.current_data['price_update_time'] = datetime.now()
            else:
                # 🔍 DEBUG: Nenhum dado de candle encontrado
                print("⚠️  GUI: Nenhum dado de candle encontrado")
                if hasattr(self.trading_system, 'data_structure'):
                    print(f"   - data_structure existe: {self.trading_system.data_structure is not None}")
                if hasattr(self.trading_system, 'data_loader'):
                    print(f"   - data_loader existe: {self.trading_system.data_loader is not None}")

            # Métricas do sistema
            if hasattr(self.trading_system, '_get_trading_metrics_safe'):
                self.current_data['trading_metrics'] = self.trading_system._get_trading_metrics_safe()

            if hasattr(self.trading_system, '_get_system_metrics_safe'):
                self.current_data['system_metrics'] = self.trading_system._get_system_metrics_safe()

            # Posições ativas
            if hasattr(self.trading_system, 'active_positions'):
                self.current_data['positions'] = self.trading_system.active_positions.copy()

            # Account info
            if hasattr(self.trading_system, 'account_info'):
                self.current_data['account_info'] = self.trading_system.account_info.copy()

            # Status do sistema
            if hasattr(self.trading_system, 'is_running'):
                status = "Operacional" if self.trading_system.is_running else "Parado"
                self.current_data['system_status'] = status

        except Exception as e:
            print(f"Erro coletando dados: {e}")
            
    def _update_interface(self):
        """Atualiza todos os elementos da interface"""
        try:
            # Atualizar preço atual
            self._update_current_price_display()
            
            # Atualizar predição
            self._update_prediction_display()

            # Atualizar candle
            self._update_candle_display()

            # Atualizar métricas
            self._update_metrics_display()

            # Atualizar posições
            self._update_positions_display()

            # Atualizar timestamp
            self.last_update_label.config(text=f"Última atualização: {datetime.now().strftime('%H:%M:%S')}")

        except Exception as e:
            print(f"Erro atualizando interface: {e}")
            
    def _update_current_price_display(self):
        """Atualiza display do preço atual"""
        try:
            # Preço atual
            current_price = self.current_data.get('current_price')
            if current_price:
                self.current_price_label.config(text=f"R$ {current_price:,.2f}")
                
            # Estatísticas do dia
            day_stats = self.current_data.get('day_stats')
            if day_stats:
                variation_pct = day_stats.get('variation_pct', 0)
                variation_value = day_stats.get('variation_value', 0)
                
                # Texto da variação
                var_text = f"{variation_pct:+.2f}% (R$ {variation_value:+,.2f})"
                
                # Cor baseada na variação
                if variation_pct > 0:
                    var_color = self.colors['profit']
                elif variation_pct < 0:
                    var_color = self.colors['loss']
                else:
                    var_color = self.colors['neutral']
                    
                self.day_variation_label.config(text=var_text, fg=var_color)
            else:
                self.day_variation_label.config(text="-.--% (R$ -.--)", fg=self.colors['neutral'])
                
            # Timestamp da atualização do preço
            price_update_time = self.current_data.get('price_update_time')
            if price_update_time:
                self.price_update_time.config(text=price_update_time.strftime('%H:%M:%S'))
                
        except Exception as e:
            print(f"Erro atualizando preço atual: {e}")

    def _update_prediction_display(self):
        """Atualiza display da predição"""
        pred = self.current_data.get('last_prediction')
        if pred:
            # Direção
            direction = pred.get('direction', 0)
            direction_text = f"{direction:.3f}"
            direction_color = self.colors['profit'] if direction > 0 else self.colors['loss'] if direction < 0 else self.colors['neutral']
            self.pred_labels['direction'].config(text=direction_text, fg=direction_color)
            
            # Confiança
            confidence = pred.get('confidence', 0)
            confidence_text = f"{confidence:.1%}"
            confidence_color = self.colors['profit'] if confidence > 0.6 else self.colors['neutral'] if confidence > 0.5 else self.colors['loss']
            self.pred_labels['confidence'].config(text=confidence_text, fg=confidence_color)
            
            # Magnitude
            magnitude = pred.get('magnitude', 0)
            self.pred_labels['magnitude'].config(text=f"{magnitude:.4f}")
            
            # Ação
            action = pred.get('action', 'HOLD')
            action_color = self.colors['profit'] if action == 'BUY' else self.colors['loss'] if action == 'SELL' else self.colors['neutral']
            self.pred_labels['action'].config(text=action, fg=action_color)
            
            # Regime
            regime = pred.get('regime', 'unknown')
            self.pred_labels['regime'].config(text=regime.replace('_', ' ').title())
            
            # Timestamp
            timestamp = pred.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                self.pred_labels['timestamp'].config(text=timestamp[:19])
            else:
                self.pred_labels['timestamp'].config(text=timestamp.strftime('%H:%M:%S'))
                
    def _update_candle_display(self):
        """Atualiza display do candle"""
        candle = self.current_data.get('last_candle')
        if candle:
            # OHLC
            for field in ['open', 'high', 'low', 'close']:
                value = candle.get(field, 0)
                self.candle_labels[field].config(text=f"{value:,.2f}")

            # Volume
            volume = candle.get('volume', 0)
            self.candle_labels['volume'].config(text=f"{int(volume):,}")
            
            # Número de trades
            trades = candle.get('trades', 0)
            self.candle_labels['trades'].config(text=f"{int(trades):,}")

            # Buy Volume (se disponível)
            buy_volume = candle.get('buy_volume', 0)
            if buy_volume > 0:
                self.candle_labels['buy_volume'].config(text=f"{int(buy_volume):,}")
            else:
                self.candle_labels['buy_volume'].config(text="-")
                
            # Sell Volume (se disponível) 
            sell_volume = candle.get('sell_volume', 0)
            if sell_volume > 0:
                self.candle_labels['sell_volume'].config(text=f"{int(sell_volume):,}")
            else:
                self.candle_labels['sell_volume'].config(text="-")

            # Variação do candle
            open_price = candle.get('open', 0)
            close_price = candle.get('close', 0)
            if open_price > 0:
                variation = ((close_price - open_price) / open_price) * 100
                var_text = f"{variation:+.2f}%"
                var_color = self.colors['profit'] if variation > 0 else self.colors['loss'] if variation < 0 else self.colors['neutral']
                self.candle_labels['variation'].config(text=var_text, fg=var_color)
            else:
                self.candle_labels['variation'].config(text="0.00%")

            # Timestamp
            timestamp = candle.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    self.candle_labels['timestamp'].config(text=timestamp[-8:])  # Apenas HH:MM:SS
                else:
                    self.candle_labels['timestamp'].config(text=timestamp.strftime('%H:%M:%S'))
        else:
            # Se não há dados, limpar campos
            for field in ['open', 'high', 'low', 'close']:
                self.candle_labels[field].config(text="-.--")
            for field in ['volume', 'trades', 'buy_volume', 'sell_volume']:
                self.candle_labels[field].config(text="-")
            self.candle_labels['variation'].config(text="-.--", fg=self.colors['neutral'])
            self.candle_labels['timestamp'].config(text="--:--:--")
                    
    def _update_metrics_display(self):
        """Atualiza métricas"""
        # Trading metrics
        trading_metrics = self.current_data.get('trading_metrics', {})
        account_info = self.current_data.get('account_info', {})
        
        # P&L Diário
        daily_pnl = trading_metrics.get('pnl', 0)
        pnl_text = f"R$ {daily_pnl:,.2f}"
        pnl_color = self.colors['profit'] if daily_pnl > 0 else self.colors['loss'] if daily_pnl < 0 else self.colors['neutral']
        self.trading_metrics['daily_pnl'].config(text=pnl_text, fg=pnl_color)
        
        # Outras métricas de trading
        self.trading_metrics['trades_today'].config(text=str(trading_metrics.get('trades_count', 0)))
        
        win_rate = trading_metrics.get('win_rate', 0)
        self.trading_metrics['win_rate'].config(text=f"{win_rate:.1%}")
        
        self.trading_metrics['active_positions'].config(text=str(trading_metrics.get('positions', 0)))
        self.trading_metrics['balance'].config(text=f"R$ {account_info.get('balance', 0):,.2f}")
        self.trading_metrics['available'].config(text=f"R$ {account_info.get('available', 0):,.2f}")
        
        # System metrics
        system_metrics = self.current_data.get('system_metrics', {})
        
        self.system_metrics['cpu_percent'].config(text=f"{system_metrics.get('cpu_percent', 0):.1f}%")
        self.system_metrics['memory_mb'].config(text=f"{system_metrics.get('memory_mb', 0):.1f} MB")
        self.system_metrics['threads'].config(text=str(system_metrics.get('threads', 0)))
        
        uptime = system_metrics.get('uptime', 0)
        uptime_text = f"{int(uptime//3600):02d}:{int((uptime%3600)//60):02d}:{int(uptime%60):02d}"
        self.system_metrics['uptime'].config(text=uptime_text)
        
        # Métricas ML (se disponíveis)
        if hasattr(self.trading_system, 'metrics') and self.trading_system.metrics:
            metrics_obj = self.trading_system.metrics
            if hasattr(metrics_obj, 'metrics'):
                ml_predictions = metrics_obj.metrics.get('predictions_made', 0)
                signals_generated = metrics_obj.metrics.get('signals_generated', 0)
                self.system_metrics['ml_predictions'].config(text=str(ml_predictions))
                self.system_metrics['signals_generated'].config(text=str(signals_generated))
                
    def _update_positions_display(self):
        """Atualiza display de posições"""
        # Limpar árvore
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)
            
        # Adicionar posições ativas
        positions = self.current_data.get('positions', {})
        for symbol, position in positions.items():
            entry_price = position.get('entry_price', 0)
            current_price = position.get('current_price', entry_price)
            side = position.get('side', 'unknown')
            size = position.get('size', 0)
            
            # Calcular P&L
            if side.lower() in ['buy', 'long']:
                pnl = (current_price - entry_price) * size
            elif side.lower() in ['sell', 'short']:
                pnl = (entry_price - current_price) * size
            else:
                pnl = 0
                
            # Inserir na árvore
            self.positions_tree.insert('', 'end', values=(
                symbol,
                side.upper(),
                f"{entry_price:.2f}",
                f"{current_price:.2f}",
                f"{pnl:+.2f}",
                str(size)
            ))
            
    def _update_status(self, message: str, color: Optional[str] = None):
        """Atualiza status do sistema"""
        self.status_label.config(text=message)
        if color:
            self.status_label.config(fg=color)
            
    def _on_closing(self):
        """Trata fechamento da janela"""
        if self.running:
            if messagebox.askokcancel("Fechar", "Deseja parar o monitoramento e fechar?"):
                self.stop_monitoring()
                self.root.destroy()
        else:
            self.root.destroy()
            
    def run(self):
        """
        Inicia a interface gráfica na thread principal
        🔧 CORREÇÃO: Garante execução na thread principal
        """
        self.logger = logging.getLogger('GUI')
        self.logger.info("Iniciando GUI na thread principal...")
        
        try:
            # Configurar protocolo de fechamento
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            
            # 🔧 CORREÇÃO: Sempre iniciar monitoramento (independente do status)
            # O GUI precisa estar ativo para mostrar dados atualizados
            self.logger.info("Iniciando monitoramento automático...")
            self.root.after(2000, self.start_monitoring)  # Delay de 2s para garantir inicialização completa
                
            # Executar mainloop na thread principal
            self.root.mainloop()
            
        except Exception as e:
            self.logger.error(f"Erro executando GUI: {e}", exc_info=True)
        finally:
            self.logger.info("GUI finalizado")


def create_monitor_gui(trading_system):
    """
    Factory function para criar monitor GUI
    
    Args:
        trading_system: Instância do TradingSystem
        
    Returns:
        TradingMonitorGUI: Instância do monitor GUI
    """
    return TradingMonitorGUI(trading_system)


if __name__ == "__main__":
    # Teste básico da interface
    import sys
    
    class MockTradingSystem:
        """Sistema mock para teste da interface"""
        def __init__(self):
            self.last_prediction = {
                'direction': 0.75,
                'confidence': 0.82,
                'magnitude': 0.0045,
                'action': 'BUY',
                'regime': 'trend_up',
                'timestamp': datetime.now()
            }
            self.active_positions = {
                'WDOQ25': {
                    'side': 'long',
                    'entry_price': 123456.0,
                    'current_price': 123500.0,
                    'size': 1
                }
            }
            self.account_info = {
                'balance': 100000.0,
                'available': 95000.0,
                'daily_pnl': 250.0
            }
            self.is_running = True
            
        def _get_trading_metrics_safe(self):
            return {
                'trades_count': 3,
                'win_rate': 0.67,
                'pnl': 250.0,
                'positions': 1
            }
            
        def _get_system_metrics_safe(self):
            return {
                'cpu_percent': 15.2,
                'memory_mb': 245.8,
                'threads': 8,
                'uptime': 3665
            }
    
    # Criar sistema mock
    mock_system = MockTradingSystem()
    
    # Criar e executar GUI
    monitor = TradingMonitorGUI(mock_system)
    monitor.run()
