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
        
        # Setup do logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
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
        
        # Ajustar tamanho para melhor visualização
        # Detectar resolução da tela
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Definir tamanho otimizado (80% da tela ou máximo de 1400x950) - altura aumentada
        window_width = min(int(screen_width * 0.8), 1400)
        window_height = min(int(screen_height * 0.85), 950)
        
        # Centralizar janela
        pos_x = (screen_width - window_width) // 2
        pos_y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        self.root.resizable(True, True)
        
        # Definir tamanho mínimo (altura aumentada)
        self.root.minsize(1000, 700)
        
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
        
        # Cores do tema (otimizadas para melhor visibilidade)
        self.colors = {
            'profit': '#00C851',      # Verde mais suave
            'loss': '#FF4444',        # Vermelho mais suave
            'neutral': '#33B5E5',     # Azul claro ao invés de amarelo
            'warning': '#FF8800',     # Laranja para avisos
            'bg_dark': '#1E1E1E',     # Fundo escuro
            'bg_light': '#2D2D2D',    # Fundo claro
            'text': '#E0E0E0',        # Texto cinza claro ao invés de branco
            'text_dark': '#B0B0B0',   # Texto secundário
            'accent': '#0078D4',      # Azul Microsoft
            'border': '#404040'       # Bordas
        }
        
        # Configurar tema escuro profissional
        self._configure_dark_theme()
        
        # Configurar fontes (tamanhos otimizados)
        self.fonts = {
            'title': font.Font(family='Segoe UI', size=12, weight='bold'),
            'data': font.Font(family='Consolas', size=9),
            'status': font.Font(family='Segoe UI', size=10, weight='bold'),
            'small': font.Font(family='Segoe UI', size=8),
            'header': font.Font(family='Segoe UI', size=11, weight='bold')
        }
        
    def _configure_dark_theme(self):
        """Configura tema escuro profissional"""
        try:
            # Configurar tema escuro para TTK
            self.style.theme_use('clam')
            
            # Configurar cores para widgets TTK
            self.style.configure('TLabel', 
                               background=self.colors['bg_dark'],
                               foreground=self.colors['text'])
            
            self.style.configure('TFrame', 
                               background=self.colors['bg_dark'],
                               borderwidth=1,
                               relief='flat')
            
            self.style.configure('TLabelFrame', 
                               background=self.colors['bg_dark'],
                               foreground=self.colors['accent'],
                               borderwidth=1,
                               relief='solid')
                               
            self.style.configure('TButton',
                               background=self.colors['bg_light'],
                               foreground=self.colors['text'],
                               borderwidth=1,
                               focuscolor='none')
                               
            # Configurar root background
            self.root.configure(bg=self.colors['bg_dark'])
            
        except Exception as e:
            self.logger.warning(f"Erro configurando tema escuro: {e}")
        
    def _create_widgets(self):
        """Cria todos os widgets da interface com layout melhorado"""
        # Frame principal com melhor espaçamento
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # === HEADER COMPACTO ===
        self._create_header(main_frame)
        
        # === LAYOUT PRINCIPAL REORGANIZADO ===
        # Usar grid para melhor controle de layout
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Configurar grid para 3 linhas e 2 colunas
        content_frame.grid_rowconfigure(0, weight=0)  # Top row - altura fixa
        content_frame.grid_rowconfigure(1, weight=1)  # Middle row - expansível
        content_frame.grid_rowconfigure(2, weight=0)  # Bottom row - altura fixa
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
        
        # === LINHA 1: INFORMAÇÕES PRINCIPAIS ===
        # Predições ML (coluna 1)
        pred_frame = ttk.LabelFrame(content_frame, text="🎯 Predições ML", padding=8)
        pred_frame.grid(row=0, column=0, sticky='ew', padx=(0, 4), pady=(0, 6))
        
        # Status do Sistema (coluna 2)  
        status_frame = ttk.LabelFrame(content_frame, text="📊 Status Sistema", padding=8)
        status_frame.grid(row=0, column=1, sticky='ew', padx=(4, 0), pady=(0, 6))
        
        # === LINHA 2: DADOS PRINCIPAIS ===
        # Dados de Mercado (coluna 1)
        market_frame = ttk.LabelFrame(content_frame, text="💹 Dados de Mercado", padding=8)
        market_frame.grid(row=1, column=0, sticky='nsew', padx=(0, 4), pady=(0, 6))
        
        # Métricas e Performance (coluna 2)
        metrics_frame = ttk.LabelFrame(content_frame, text="📈 Métricas & Performance", padding=8)
        metrics_frame.grid(row=1, column=1, sticky='nsew', padx=(4, 0), pady=(0, 6))
        
        # === LINHA 3: ALERTAS E STATUS ===
        # Alertas e Log (largura total)
        alerts_frame = ttk.LabelFrame(content_frame, text="🔔 Alertas & Log do Sistema", padding=6)
        alerts_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(0, 6))
        
        # Criar seções com layout melhorado
        self._create_prediction_section_improved(pred_frame)
        self._create_system_status_section_improved(status_frame)
        self._create_market_data_section_improved(market_frame)
        self._create_metrics_section_improved(metrics_frame)
        self._create_alerts_section_improved(alerts_frame)
        
        # === FOOTER SIMPLIFICADO ===
        self._create_footer_improved(main_frame)
        
    def _create_header(self, parent):
        """Cria header com status do sistema"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Título
        title_label = tk.Label(header_frame, text="Trading System ML v2.0", 
                              font=self.fonts['title'], fg=self.colors['accent'],
                              bg=self.colors['bg_dark'])
        title_label.pack(side=tk.LEFT)
        
        # Status
        self.status_label = tk.Label(header_frame, text="Sistema: Inicializando...", 
                                    font=self.fonts['status'], fg=self.colors['neutral'],
                                    bg=self.colors['bg_dark'])
        self.status_label.pack(side=tk.RIGHT)
        
        # Separador
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=5)
        
    def _create_prediction_section(self, parent):
        """Cria seção de predições ML (versão compacta)"""
        # Grid para organizar dados de forma compacta
        pred_grid = ttk.Frame(parent)
        pred_grid.pack(fill=tk.X)
        
        # Labels para dados da predição
        self.pred_labels = {}
        
        # Linha 1: Direção e Confiança
        tk.Label(pred_grid, text="Direção:", font=self.fonts['small'], 
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=0, column=0, sticky='w', padx=(0, 5))
        self.pred_labels['direction'] = tk.Label(pred_grid, text="-", font=self.fonts['data'], 
                                                fg=self.colors['neutral'], bg=self.colors['bg_dark'])
        self.pred_labels['direction'].grid(row=0, column=1, sticky='w', padx=(0, 15))
        
        tk.Label(pred_grid, text="Conf:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=0, column=2, sticky='w', padx=(0, 5))
        self.pred_labels['confidence'] = tk.Label(pred_grid, text="-", font=self.fonts['data'],
                                                 fg=self.colors['text'], bg=self.colors['bg_dark'])
        self.pred_labels['confidence'].grid(row=0, column=3, sticky='w')
        
        # Linha 2: Ação e Magnitude
        tk.Label(pred_grid, text="Ação:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=1, column=0, sticky='w', padx=(0, 5))
        self.pred_labels['action'] = tk.Label(pred_grid, text="-", font=self.fonts['data'], 
                                             fg=self.colors['neutral'], bg=self.colors['bg_dark'])
        self.pred_labels['action'].grid(row=1, column=1, sticky='w', padx=(0, 15))
        
        tk.Label(pred_grid, text="Mag:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=1, column=2, sticky='w', padx=(0, 5))
        self.pred_labels['magnitude'] = tk.Label(pred_grid, text="-", font=self.fonts['data'],
                                                fg=self.colors['text'], bg=self.colors['bg_dark'])
        self.pred_labels['magnitude'].grid(row=1, column=3, sticky='w')
        
        # Linha 3: Regime
        tk.Label(pred_grid, text="Regime:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=2, column=0, sticky='w', padx=(0, 5))
        self.pred_labels['regime'] = tk.Label(pred_grid, text="-", font=self.fonts['data'],
                                             fg=self.colors['text'], bg=self.colors['bg_dark'])
        self.pred_labels['regime'].grid(row=2, column=1, sticky='w', padx=(0, 20))
        
        tk.Label(pred_grid, text="Hora:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=2, column=2, sticky='w', padx=(0, 5))
        self.pred_labels['timestamp'] = tk.Label(pred_grid, text="-", font=self.fonts['small'],
                                                fg=self.colors['text_dark'], bg=self.colors['bg_dark'])
        self.pred_labels['timestamp'].grid(row=2, column=3, sticky='w')
        
    def _create_prediction_section_improved(self, parent):
        """Seção de predições ML com layout melhorado"""
        # Container principal
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Labels para predições
        self.pred_labels = {}
        
        # Linha 1: Ação principal (destaque)
        action_frame = ttk.Frame(container)
        action_frame.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(action_frame, text="Ação:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).pack(side=tk.LEFT)
        self.pred_labels['action'] = tk.Label(action_frame, text="AGUARDANDO", 
                                             font=self.fonts['title'], fg=self.colors['neutral'],
                                             bg=self.colors['bg_dark'])
        self.pred_labels['action'].pack(side=tk.LEFT, padx=(8, 0))
        
        # Linha 2: Confiança e Magnitude
        metrics_frame = ttk.Frame(container)
        metrics_frame.pack(fill=tk.X, pady=(0, 4))
        
        # Grid 2x2 para métricas
        tk.Label(metrics_frame, text="Confiança:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=0, column=0, sticky='w', padx=(0, 8))
        self.pred_labels['confidence'] = tk.Label(metrics_frame, text="0%", font=self.fonts['data'],
                                                 fg=self.colors['text'], bg=self.colors['bg_dark'])
        self.pred_labels['confidence'].grid(row=0, column=1, sticky='w', padx=(0, 20))
        
        tk.Label(metrics_frame, text="Magnitude:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=0, column=2, sticky='w', padx=(0, 8))
        self.pred_labels['magnitude'] = tk.Label(metrics_frame, text="0.000", font=self.fonts['data'],
                                                fg=self.colors['text'], bg=self.colors['bg_dark'])
        self.pred_labels['magnitude'].grid(row=0, column=3, sticky='w')
        
        # Linha 3: Regime e Timestamp
        tk.Label(metrics_frame, text="Regime:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=1, column=0, sticky='w', padx=(0, 8))
        self.pred_labels['regime'] = tk.Label(metrics_frame, text="Detectando...", font=self.fonts['data'],
                                             fg=self.colors['warning'], bg=self.colors['bg_dark'])
        self.pred_labels['regime'].grid(row=1, column=1, sticky='w', padx=(0, 20))
        
        tk.Label(metrics_frame, text="Última:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=1, column=2, sticky='w', padx=(0, 8))
        self.pred_labels['timestamp'] = tk.Label(metrics_frame, text="--:--:--", font=self.fonts['small'],
                                                fg=self.colors['text_dark'], bg=self.colors['bg_dark'])
        self.pred_labels['timestamp'].grid(row=1, column=3, sticky='w')
        
        # Configurar expansão das colunas
        metrics_frame.grid_columnconfigure(1, weight=1)
        metrics_frame.grid_columnconfigure(3, weight=1)
        
        # Adicionar labels obrigatórios para compatibilidade
        self.pred_labels['direction'] = self.pred_labels.get('action', self.pred_labels['action'])  # Reuso do action
        
        # Label de preço atual (obrigatório)
        price_frame = ttk.Frame(container)
        price_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(price_frame, text="Preço Atual:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).pack(side=tk.LEFT)
        self.current_price_label = tk.Label(price_frame, text="R$ -.--", font=self.fonts['data'],
                                           fg=self.colors['accent'], bg=self.colors['bg_dark'])
        self.current_price_label.pack(side=tk.LEFT, padx=(8, 0))
        
    def _create_system_status_section(self, parent):
        """Cria seção de status do sistema (nova)"""
        # Grid para status
        status_grid = ttk.Frame(parent)
        status_grid.pack(fill=tk.X)
        
        # Labels para status do sistema
        self.status_labels = {}
        
        # Linha 1: Sistema e Conexão
        tk.Label(status_grid, text="Sistema:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=0, column=0, sticky='w', padx=(0, 5))
        self.status_labels['system'] = tk.Label(status_grid, text="Inicializando", font=self.fonts['data'], 
                                               fg=self.colors['warning'], bg=self.colors['bg_dark'])
        self.status_labels['system'].grid(row=0, column=1, sticky='w', padx=(0, 15))
        
        tk.Label(status_grid, text="Conexão:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=0, column=2, sticky='w', padx=(0, 5))
        self.status_labels['connection'] = tk.Label(status_grid, text="Offline", font=self.fonts['data'], 
                                                   fg=self.colors['loss'], bg=self.colors['bg_dark'])
        self.status_labels['connection'].grid(row=0, column=3, sticky='w')
        
        # Linha 2: Ticker e Última Atualização
        tk.Label(status_grid, text="Ticker:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=1, column=0, sticky='w', padx=(0, 5))
        self.status_labels['ticker'] = tk.Label(status_grid, text="-", font=self.fonts['data'],
                                               fg=self.colors['accent'], bg=self.colors['bg_dark'])
        self.status_labels['ticker'].grid(row=1, column=1, sticky='w', padx=(0, 15))
        
        tk.Label(status_grid, text="Atualiz:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=1, column=2, sticky='w', padx=(0, 5))
        self.status_labels['last_update'] = tk.Label(status_grid, text="-", font=self.fonts['small'],
                                                    fg=self.colors['text_dark'], bg=self.colors['bg_dark'])
        self.status_labels['last_update'].grid(row=1, column=3, sticky='w')
        
    def _create_candle_section(self, parent):
        """Cria seção de dados do último candle (versão compacta)"""
        # Dividir em duas colunas: OHLC + Volume/Tempo
        main_grid = ttk.Frame(parent)
        main_grid.pack(fill=tk.BOTH, expand=True)
        
        # === COLUNA ESQUERDA: PREÇOS OHLC ===
        prices_frame = ttk.LabelFrame(main_grid, text="💰 OHLC", padding=3)
        prices_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 2))
        
        # Grid para preços OHLC
        price_grid = ttk.Frame(prices_frame)
        price_grid.pack(fill=tk.X)
        
        # Labels para dados OHLC
        self.candle_labels = {}
        
        # OHLC em grid compacto
        ohlc_data = [('Open:', 'open'), ('High:', 'high'), ('Low:', 'low'), ('Close:', 'close')]
        for i, (label, key) in enumerate(ohlc_data):
            row = i // 2
            col = (i % 2) * 2
            tk.Label(price_grid, text=label, font=self.fonts['small']).grid(row=row, column=col, sticky='w', padx=(0, 3))
            self.candle_labels[key] = tk.Label(price_grid, text="-.--", font=self.fonts['data'], fg=self.colors['text'])
            self.candle_labels[key].grid(row=row, column=col+1, sticky='w', padx=(0, 10))
        
        # Configurar expansão das colunas
        main_grid.columnconfigure(0, weight=1)
        main_grid.columnconfigure(1, weight=1)
        
        # === COLUNA DIREITA: VOLUME + STATUS ===
        info_frame = ttk.LabelFrame(main_grid, text="📊 Info", padding=3)
        info_frame.grid(row=0, column=1, sticky='nsew', padx=(2, 0))
        
        info_grid = ttk.Frame(info_frame)
        info_grid.pack(fill=tk.X)
        
        # Volume e Variação
        tk.Label(info_grid, text="Volume:", font=self.fonts['small']).grid(row=0, column=0, sticky='w', padx=(0, 3))
        self.candle_labels['volume'] = tk.Label(info_grid, text="-", font=self.fonts['data'])
        self.candle_labels['volume'].grid(row=0, column=1, sticky='w', padx=(0, 10))
        
        tk.Label(info_grid, text="Var%:", font=self.fonts['small']).grid(row=1, column=0, sticky='w', padx=(0, 3))
        self.candle_labels['variation'] = tk.Label(info_grid, text="-.--", font=self.fonts['data'])
        self.candle_labels['variation'].grid(row=1, column=1, sticky='w', padx=(0, 10))
        
        # Preço atual (destaque)
        tk.Label(info_grid, text="Preço:", font=self.fonts['small']).grid(row=3, column=0, sticky='w', padx=(0, 3))
        self.current_price_label = tk.Label(info_grid, text="R$ -.--", font=self.fonts['data'], fg=self.colors['accent'])
        self.current_price_label.grid(row=3, column=1, sticky='w', padx=(0, 10))
        
        # Timestamp do último candle
        tk.Label(info_grid, text="Hora:", font=self.fonts['small']).grid(row=2, column=0, sticky='w', padx=(0, 3))
        self.candle_labels['timestamp'] = tk.Label(info_grid, text="--:--", font=self.fonts['small'])
        self.candle_labels['timestamp'].grid(row=2, column=1, sticky='w')
        
    def _create_metrics_section(self, parent):
        """Cria seção de métricas do sistema (versão compacta)"""
        # Grid principal para métricas
        metrics_grid = ttk.Frame(parent)
        metrics_grid.pack(fill=tk.BOTH, expand=True)
        
        # === MÉTRICAS DE TRADING (Superior) ===
        trading_frame = ttk.LabelFrame(metrics_grid, text="📈 Trading", padding=3)
        trading_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 2))
        self._create_trading_metrics_compact(trading_frame)
        
        # === MÉTRICAS DO SISTEMA (Inferior) ===
        system_frame = ttk.LabelFrame(metrics_grid, text="⚙️ Sistema", padding=3)
        system_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(2, 0))
        self._create_system_metrics_compact(system_frame)
        
        # Configurar expansão
        metrics_grid.columnconfigure(0, weight=1)
        metrics_grid.columnconfigure(1, weight=1)
        
    def _create_trading_metrics_compact(self, parent):
        """Métricas de trading em formato compacto"""
        trading_grid = ttk.Frame(parent)
        trading_grid.pack(fill=tk.X)
        
        # Labels para métricas de trading
        self.trading_metrics = {}
        
        # Linha 1: PnL e Win Rate
        tk.Label(trading_grid, text="PnL:", font=self.fonts['small']).grid(row=0, column=0, sticky='w', padx=(0, 3))
        self.trading_metrics['pnl'] = tk.Label(trading_grid, text="R$ 0.00", font=self.fonts['data'], fg=self.colors['neutral'])
        self.trading_metrics['pnl'].grid(row=0, column=1, sticky='w', padx=(0, 10))
        
        tk.Label(trading_grid, text="Win%:", font=self.fonts['small']).grid(row=0, column=2, sticky='w', padx=(0, 3))
        self.trading_metrics['win_rate'] = tk.Label(trading_grid, text="0%", font=self.fonts['data'])
        self.trading_metrics['win_rate'].grid(row=0, column=3, sticky='w')
        
        # Linha 2: Trades e Drawdown
        tk.Label(trading_grid, text="Trades:", font=self.fonts['small']).grid(row=1, column=0, sticky='w', padx=(0, 3))
        self.trading_metrics['trades'] = tk.Label(trading_grid, text="0", font=self.fonts['data'])
        self.trading_metrics['trades'].grid(row=1, column=1, sticky='w', padx=(0, 10))
        
        tk.Label(trading_grid, text="DD%:", font=self.fonts['small']).grid(row=1, column=2, sticky='w', padx=(0, 3))
        self.trading_metrics['drawdown'] = tk.Label(trading_grid, text="0%", font=self.fonts['data'])
        self.trading_metrics['drawdown'].grid(row=1, column=3, sticky='w')
        
    def _create_system_metrics_compact(self, parent):
        """Métricas do sistema em formato compacto"""
        system_grid = ttk.Frame(parent)
        system_grid.pack(fill=tk.X)
        
        # Labels para métricas do sistema
        self.system_metrics = {}
        
        # Linha 1: CPU e Memória
        tk.Label(system_grid, text="CPU:", font=self.fonts['small']).grid(row=0, column=0, sticky='w', padx=(0, 3))
        self.system_metrics['cpu'] = tk.Label(system_grid, text="0%", font=self.fonts['data'])
        self.system_metrics['cpu'].grid(row=0, column=1, sticky='w', padx=(0, 10))
        
        tk.Label(system_grid, text="RAM:", font=self.fonts['small']).grid(row=0, column=2, sticky='w', padx=(0, 3))
        self.system_metrics['memory'] = tk.Label(system_grid, text="0MB", font=self.fonts['data'])
        self.system_metrics['memory'].grid(row=0, column=3, sticky='w')
        
        # Linha 2: Uptime e Features
        tk.Label(system_grid, text="Uptime:", font=self.fonts['small']).grid(row=1, column=0, sticky='w', padx=(0, 3))
        self.system_metrics['uptime'] = tk.Label(system_grid, text="00:00:00", font=self.fonts['small'])
        self.system_metrics['uptime'].grid(row=1, column=1, sticky='w', padx=(0, 10))
        
        tk.Label(system_grid, text="Features:", font=self.fonts['small']).grid(row=1, column=2, sticky='w', padx=(0, 3))
        self.system_metrics['features'] = tk.Label(system_grid, text="0", font=self.fonts['small'])
        self.system_metrics['features'].grid(row=1, column=3, sticky='w')
        
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
        """Cria seção de alertas (versão compacta)"""
        # Listbox compacta para alertas com altura reduzida
        listbox_frame = ttk.Frame(parent)
        listbox_frame.pack(fill=tk.X, expand=False)
        
        self.alerts_listbox = tk.Listbox(listbox_frame, font=self.fonts['small'], height=4)
        alerts_scrollbar = ttk.Scrollbar(listbox_frame, orient='vertical', command=self.alerts_listbox.yview)
        self.alerts_listbox.configure(yscrollcommand=alerts_scrollbar.set)
        
        self.alerts_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        alerts_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Adicionar alertas de exemplo
        self.alerts_listbox.insert(0, "Sistema iniciado com sucesso")
        self.alerts_listbox.insert(1, "Aguardando dados do mercado...")
        
    def _create_footer(self, parent):
        """Cria footer com controles (versão compacta)"""
        footer_frame = ttk.Frame(parent)
        footer_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Separador
        ttk.Separator(footer_frame, orient='horizontal').pack(fill=tk.X, pady=(0, 5))
        
        # Status e controles em linha
        controls_frame = ttk.Frame(footer_frame)
        controls_frame.pack()
        
        # Status simplificado
        self.footer_status = tk.Label(controls_frame, text="Monitor Ativo", 
                                     font=self.fonts['small'], fg=self.colors['profit'],
                                     bg=self.colors['bg_dark'])
        self.footer_status.pack(side=tk.LEFT, padx=(0, 20))
        
        # Botões compactos
        self.start_button = ttk.Button(controls_frame, text="▶ Iniciar", command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
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
            
            # Atualizar status do sistema (NOVO)
            self._update_system_status()

            # Atualizar timestamp
            if hasattr(self, 'last_update_label'):
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
                    
                # Variação do dia removida do layout compacto
                pass
            else:
                # Variação do dia removida do layout compacto
                pass
                
            # Timestamp da atualização do preço removido do layout compacto
            # Informação mantida internamente mas não exibida
                
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
            
            # Buy/Sell Volume removidos do layout compacto para economia de espaço
            # Funcionalidade mantida internamente mas não exibida na interface

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
            # Limpar apenas campos que existem no layout compacto
            if 'volume' in self.candle_labels:
                self.candle_labels['volume'].config(text="-")
            # Campos removidos do layout compacto: trades, buy_volume, sell_volume
            self.candle_labels['variation'].config(text="-.--", fg=self.colors['neutral'])
            if 'timestamp' in self.candle_labels:
                self.candle_labels['timestamp'].config(text="--:--")
                    
    def _update_metrics_display(self):
        """Atualiza métricas"""
        # Trading metrics
        trading_metrics = self.current_data.get('trading_metrics', {})
        account_info = self.current_data.get('account_info', {})
        
        # P&L Diário (layout compacto usa 'pnl' ao invés de 'daily_pnl')
        daily_pnl = trading_metrics.get('pnl', 0)
        pnl_text = f"R$ {daily_pnl:,.2f}"
        pnl_color = self.colors['profit'] if daily_pnl > 0 else self.colors['loss'] if daily_pnl < 0 else self.colors['neutral']
        if 'pnl' in self.trading_metrics:
            self.trading_metrics['pnl'].config(text=pnl_text, fg=pnl_color)
        
        # Outras métricas de trading (layout compacto usa 'trades' ao invés de 'trades_today')
        if 'trades' in self.trading_metrics:
            self.trading_metrics['trades'].config(text=str(trading_metrics.get('trades_count', 0)))
        
        win_rate = trading_metrics.get('win_rate', 0)
        if 'win_rate' in self.trading_metrics:
            self.trading_metrics['win_rate'].config(text=f"{win_rate:.1%}")
        
        # Proteger referências a métricas que podem não existir no layout compacto
        if 'active_positions' in self.trading_metrics:
            self.trading_metrics['active_positions'].config(text=str(trading_metrics.get('positions', 0)))
        if 'balance' in self.trading_metrics:
            self.trading_metrics['balance'].config(text=f"R$ {account_info.get('balance', 0):,.2f}")
        if 'available' in self.trading_metrics:
            self.trading_metrics['available'].config(text=f"R$ {account_info.get('available', 0):,.2f}")
        if 'drawdown' in self.trading_metrics:
            drawdown = trading_metrics.get('drawdown', 0)
            self.trading_metrics['drawdown'].config(text=f"{drawdown:.1%}")
        
        # System metrics (proteger referências do layout compacto)
        system_metrics = self.current_data.get('system_metrics', {})
        
        if 'cpu' in self.system_metrics:
            self.system_metrics['cpu'].config(text=f"{system_metrics.get('cpu_percent', 0):.1f}%")
        if 'memory' in self.system_metrics:
            self.system_metrics['memory'].config(text=f"{system_metrics.get('memory_mb', 0):.1f}MB")
        if 'uptime' in self.system_metrics:
            uptime = system_metrics.get('uptime', 0)
            uptime_text = f"{int(uptime//3600):02d}:{int((uptime%3600)//60):02d}:{int(uptime%60):02d}"
            self.system_metrics['uptime'].config(text=uptime_text)
        if 'features' in self.system_metrics:
            self.system_metrics['features'].config(text=str(system_metrics.get('features_count', 0)))
        
        # Métricas ML (se disponíveis) - proteger referências do layout compacto
        if hasattr(self.trading_system, 'metrics') and self.trading_system.metrics:
            metrics_obj = self.trading_system.metrics
            if hasattr(metrics_obj, 'metrics'):
                ml_predictions = metrics_obj.metrics.get('predictions_made', 0)
                signals_generated = metrics_obj.metrics.get('signals_generated', 0)
                if 'ml_predictions' in self.system_metrics:
                    self.system_metrics['ml_predictions'].config(text=str(ml_predictions))
                if 'signals_generated' in self.system_metrics:
                    self.system_metrics['signals_generated'].config(text=str(signals_generated))
                
    def _update_positions_display(self):
        """Atualiza display de posições"""
        # Verificar se positions_tree existe (removido do layout compacto)
        if not hasattr(self, 'positions_tree'):
            return
            
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

    def _update_system_status(self):
        """Atualiza status do sistema em tempo real"""
        try:
            if not hasattr(self, 'status_labels'):
                return
                
            # Atualizar status do sistema
            if hasattr(self.trading_system, 'is_running'):
                if self.trading_system.is_running:
                    status_text = "Operacional"
                    status_color = self.colors['profit']
                else:
                    status_text = "Parado"
                    status_color = self.colors['loss']
                    
                if 'system' in self.status_labels:
                    self.status_labels['system'].config(text=status_text, fg=status_color)
            
            # Atualizar status da conexão
            connection_status = "Online"
            connection_color = self.colors['profit']
            
            # Verificar se há conexão ativa
            if hasattr(self.trading_system, 'connection_manager'):
                if hasattr(self.trading_system.connection_manager, 'connected'):
                    if not self.trading_system.connection_manager.connected:
                        connection_status = "Offline"
                        connection_color = self.colors['loss']
                elif hasattr(self.trading_system.connection_manager, 'is_connected'):
                    if not self.trading_system.connection_manager.is_connected():
                        connection_status = "Offline"
                        connection_color = self.colors['loss']
            
            if 'connection' in self.status_labels:
                self.status_labels['connection'].config(text=connection_status, fg=connection_color)
            
            # Atualizar ticker
            if hasattr(self.trading_system, 'ticker') and 'ticker' in self.status_labels:
                self.status_labels['ticker'].config(text=self.trading_system.ticker)
            
            # Atualizar última atualização
            if 'last_update' in self.status_labels:
                current_time = datetime.now().strftime('%H:%M:%S')
                self.status_labels['last_update'].config(text=current_time)
                
        except Exception as e:
            print(f"Erro atualizando status do sistema: {e}")
            
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
            
    def _create_system_status_section_improved(self, parent):
        """Cria seção de status do sistema melhorada"""
        # Container principal
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        
        # Labels para status do sistema
        self.status_labels = {}
        
        # Grid para organização compacta
        grid = ttk.Frame(container)
        grid.pack(fill=tk.X)
        
        # Status labels com cores dinâmicas e atualizações automáticas
        status_items = [
            ('system', 'Sistema:', 'Ativo'),
            ('connection', 'Conexão:', 'Online'),
            ('ticker', 'Ticker:', getattr(self.trading_system, 'ticker', 'WDOQ25')),
            ('last_update', 'Atualiz:', datetime.now().strftime('%H:%M:%S'))
        ]
        
        for i, (key, label, default_value) in enumerate(status_items):
            row = i // 2
            col = (i % 2) * 2
            
            tk.Label(grid, text=label, font=self.fonts['small'],
                    fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(
                        row=row, column=col, sticky='w', padx=(0, 5))
            
            self.status_labels[key] = tk.Label(grid, text=default_value, font=self.fonts['data'],
                                             fg=self.colors['profit'], bg=self.colors['bg_dark'])
            self.status_labels[key].grid(row=row, column=col+1, sticky='w', padx=(0, 20))
        
        # Configurar expansão
        grid.grid_columnconfigure(1, weight=1)
        grid.grid_columnconfigure(3, weight=1)

    def _create_market_data_section_improved(self, parent):
        """Cria seção de dados de mercado melhorada"""
        # Container com duas colunas
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        
        # OHLC compacto
        ohlc_frame = ttk.LabelFrame(container, text="💰 OHLC", padding=4)
        ohlc_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        
        # Labels para OHLC
        self.candle_labels = {}
        ohlc_grid = ttk.Frame(ohlc_frame)
        ohlc_grid.pack(fill=tk.X)
        
        ohlc_items = [('open', 'Open'), ('high', 'High'), ('low', 'Low'), ('close', 'Close')]
        for i, (key, label) in enumerate(ohlc_items):
            row = i // 2
            col = (i % 2) * 2
            
            tk.Label(ohlc_grid, text=f"{label}:", font=self.fonts['small'],
                    fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(
                        row=row, column=col, sticky='w', padx=(0, 3))
            
            self.candle_labels[key] = tk.Label(ohlc_grid, text="-", font=self.fonts['data'],
                                             fg=self.colors['text'], bg=self.colors['bg_dark'])
            self.candle_labels[key].grid(row=row, column=col+1, sticky='w', padx=(0, 10))
        
        # Volume e timestamp
        vol_frame = ttk.LabelFrame(container, text="📊 Info", padding=4)
        vol_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        
        vol_grid = ttk.Frame(vol_frame)
        vol_grid.pack(fill=tk.X)
        
        tk.Label(vol_grid, text="Volume:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=0, column=0, sticky='w')
        self.candle_labels['volume'] = tk.Label(vol_grid, text="-", font=self.fonts['data'],
                                               fg=self.colors['text'], bg=self.colors['bg_dark'])
        self.candle_labels['volume'].grid(row=0, column=1, sticky='w', padx=(5, 0))
        
        tk.Label(vol_grid, text="Hora:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=1, column=0, sticky='w')
        self.candle_labels['timestamp'] = tk.Label(vol_grid, text="-", font=self.fonts['small'],
                                                  fg=self.colors['text_dark'], bg=self.colors['bg_dark'])
        self.candle_labels['timestamp'].grid(row=1, column=1, sticky='w', padx=(5, 0))
        
        tk.Label(vol_grid, text="Variação:", font=self.fonts['small'],
                fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=2, column=0, sticky='w')
        self.candle_labels['variation'] = tk.Label(vol_grid, text="-.--", font=self.fonts['data'],
                                                  fg=self.colors['neutral'], bg=self.colors['bg_dark'])
        self.candle_labels['variation'].grid(row=2, column=1, sticky='w', padx=(5, 0))

    def _create_metrics_section_improved(self, parent):
        """Cria seção de métricas melhorada"""
        # Container principal
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        
        # Dividir em duas colunas: Trading e Sistema
        # Trading metrics
        trading_frame = ttk.LabelFrame(container, text="📈 Trading", padding=4)
        trading_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        
        self.trading_metrics = {}
        trading_grid = ttk.Frame(trading_frame)
        trading_grid.pack(fill=tk.X)
        
        trading_items = [('pnl', 'P&L:', 'R$ 0.00'), ('trades', 'Trades:', '0'), ('win_rate', 'Win Rate:', '0%')]
        for i, (key, label, default) in enumerate(trading_items):
            tk.Label(trading_grid, text=label, font=self.fonts['small'],
                    fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=i, column=0, sticky='w', padx=(0, 5))
            
            self.trading_metrics[key] = tk.Label(trading_grid, text=default, font=self.fonts['data'],
                                               fg=self.colors['text'], bg=self.colors['bg_dark'])
            self.trading_metrics[key].grid(row=i, column=1, sticky='w')
        
        # System metrics
        system_frame = ttk.LabelFrame(container, text="🖥️ Sistema", padding=4)
        system_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        
        self.system_metrics = {}
        system_grid = ttk.Frame(system_frame)
        system_grid.pack(fill=tk.X)
        
        system_items = [('cpu', 'CPU:', '0%'), ('memory', 'RAM:', '0 MB'), ('uptime', 'Uptime:', '0s')]
        for i, (key, label, default) in enumerate(system_items):
            tk.Label(system_grid, text=label, font=self.fonts['small'],
                    fg=self.colors['text_dark'], bg=self.colors['bg_dark']).grid(row=i, column=0, sticky='w', padx=(0, 5))
            
            self.system_metrics[key] = tk.Label(system_grid, text=default, font=self.fonts['data'],
                                              fg=self.colors['text'], bg=self.colors['bg_dark'])
            self.system_metrics[key].grid(row=i, column=1, sticky='w')

    def _create_alerts_section_improved(self, parent):
        """Cria seção de alertas melhorada e compacta"""
        # Container compacto com padding reduzido
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, padx=6, pady=2)
        
        # Listbox compacta com altura ainda menor
        self.alerts_listbox = tk.Listbox(container, font=self.fonts['small'], height=2,
                                        bg=self.colors['bg_light'], fg=self.colors['text'],
                                        selectbackground=self.colors['accent'])
        
        scrollbar = ttk.Scrollbar(container, orient='vertical', command=self.alerts_listbox.yview)
        self.alerts_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.alerts_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Alertas iniciais atualizados
        self.alerts_listbox.insert(0, f"Sistema iniciado - {datetime.now().strftime('%H:%M:%S')}")
        self.alerts_listbox.insert(1, "Aguardando dados do mercado...")

    def _create_footer_improved(self, parent):
        """Cria footer melhorado e compacto"""
        # Container principal com padding reduzido
        footer = ttk.Frame(parent)
        footer.pack(fill=tk.X, side=tk.BOTTOM, pady=(3, 0))
        
        # Separador
        ttk.Separator(footer, orient='horizontal').pack(fill=tk.X, pady=(0, 3))
        
        # Status e controles em linha
        controls = ttk.Frame(footer)
        controls.pack(fill=tk.X)
        
        # Status ativo/inativo
        self.footer_status = tk.Label(controls, text="Monitor Ativo", font=self.fonts['small'],
                                     fg=self.colors['profit'], bg=self.colors['bg_dark'])
        self.footer_status.pack(side=tk.LEFT)
        
        # Botões de controle
        self.start_button = ttk.Button(controls, text="▶ Iniciar", command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=(20, 5))
        
        self.stop_button = ttk.Button(controls, text="⏸ Parar", command=self.stop_monitoring, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=(0, 20))
        
        # Timestamp da última atualização
        self.last_update_label = tk.Label(controls, text="Última atualização: --:--:--", 
                                         font=self.fonts['small'], fg=self.colors['text_dark'],
                                         bg=self.colors['bg_dark'])
        self.last_update_label.pack(side=tk.RIGHT)

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
