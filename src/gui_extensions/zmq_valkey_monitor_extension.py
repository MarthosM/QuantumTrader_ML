# -*- coding: utf-8 -*-
"""
Extensão do Monitor GUI para incluir dados ZMQ + Valkey
Adiciona abas e painéis com métricas do sistema enhanced
"""

import tkinter as tk
from tkinter import ttk
import threading
from datetime import datetime
from typing import Dict, Optional, Any

class ZMQValkeyMonitorExtension:
    """
    Extensão para adicionar monitoramento ZMQ + Valkey ao GUI existente
    """
    
    def __init__(self, parent_gui, trading_system):
        self.parent_gui = parent_gui
        self.trading_system = trading_system
        self.dashboard = None
        
        # Verificar se sistema enhanced está ativo
        self.enhanced_active = self._check_enhanced_system()
        
        if self.enhanced_active:
            self._setup_dashboard()
            self._add_enhanced_tabs()
    
    def _check_enhanced_system(self) -> bool:
        """Verifica se sistema enhanced está ativo"""
        try:
            # Verificar se é TradingSystemEnhanced
            if hasattr(self.trading_system, 'get_enhanced_status'):
                status = self.trading_system.get_enhanced_status()
                return status['enhanced_features'].get('zmq_enabled', False) or \
                       status['enhanced_features'].get('valkey_enabled', False)
        except:
            pass
        return False
    
    def _setup_dashboard(self):
        """Configura dashboard se disponível"""
        try:
            if hasattr(self.trading_system, 'valkey_manager') and self.trading_system.valkey_manager:
                from monitoring.realtime_dashboard import RealTimeDashboard
                self.dashboard = RealTimeDashboard(self.trading_system.valkey_manager)
        except Exception as e:
            self.parent_gui.logger.error(f"Erro ao configurar dashboard: {e}")
    
    def _add_enhanced_tabs(self):
        """Adiciona novas abas ao notebook existente"""
        try:
            # Encontrar o notebook principal
            notebook = self._find_notebook()
            if not notebook:
                return
            
            # Adicionar aba ZMQ/Valkey
            self.zmq_frame = ttk.Frame(notebook)
            notebook.add(self.zmq_frame, text="ZMQ/Valkey")
            self._create_zmq_valkey_panel()
            
            # Adicionar aba Time Travel se disponível
            if self._has_time_travel():
                self.tt_frame = ttk.Frame(notebook)
                notebook.add(self.tt_frame, text="Time Travel")
                self._create_time_travel_panel()
                
        except Exception as e:
            self.parent_gui.logger.error(f"Erro ao adicionar abas: {e}")
    
    def _find_notebook(self) -> Optional[ttk.Notebook]:
        """Encontra o notebook no GUI pai"""
        for widget in self.parent_gui.root.winfo_children():
            if isinstance(widget, ttk.Notebook):
                return widget
            # Buscar recursivamente
            result = self._find_notebook_recursive(widget)
            if result:
                return result
        return None
    
    def _find_notebook_recursive(self, parent) -> Optional[ttk.Notebook]:
        """Busca recursiva por notebook"""
        try:
            for widget in parent.winfo_children():
                if isinstance(widget, ttk.Notebook):
                    return widget
                result = self._find_notebook_recursive(widget)
                if result:
                    return result
        except:
            pass
        return None
    
    def _create_zmq_valkey_panel(self):
        """Cria painel de monitoramento ZMQ/Valkey"""
        # Frame principal com grid
        main_frame = ttk.Frame(self.zmq_frame, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Título
        title_label = ttk.Label(main_frame, text="Sistema Enhanced Status", 
                               font=self.parent_gui.fonts['title'])
        title_label.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Status geral
        self._create_status_section(main_frame, row=1)
        
        # Estatísticas ZMQ
        self._create_zmq_stats_section(main_frame, row=2)
        
        # Estatísticas Valkey
        self._create_valkey_stats_section(main_frame, row=3)
        
        # Bridge stats
        self._create_bridge_stats_section(main_frame, row=4)
    
    def _create_status_section(self, parent, row):
        """Seção de status geral"""
        frame = ttk.LabelFrame(parent, text="Status Geral", padding=10)
        frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        # Labels de status
        self.status_labels = {}
        components = ['ZMQ', 'Valkey', 'Time Travel', 'Bridge']
        
        for i, component in enumerate(components):
            label = ttk.Label(frame, text=f"{component}:")
            label.grid(row=0, column=i*2, sticky='w', padx=5)
            
            status = ttk.Label(frame, text="OFF", foreground='red')
            status.grid(row=0, column=i*2+1, sticky='w', padx=5)
            self.status_labels[component] = status
    
    def _create_zmq_stats_section(self, parent, row):
        """Seção de estatísticas ZMQ"""
        frame = ttk.LabelFrame(parent, text="ZeroMQ Stats", padding=10)
        frame.grid(row=row, column=0, sticky='ew', padx=5, pady=5)
        
        self.zmq_stats_labels = {}
        stats = ['Ticks Publicados', 'Trades Publicados', 'Erros', 'Taxa/seg']
        
        for i, stat in enumerate(stats):
            label = ttk.Label(frame, text=f"{stat}:")
            label.grid(row=i, column=0, sticky='w', padx=5, pady=2)
            
            value = ttk.Label(frame, text="0", font=self.parent_gui.fonts['data'])
            value.grid(row=i, column=1, sticky='e', padx=5, pady=2)
            self.zmq_stats_labels[stat] = value
    
    def _create_valkey_stats_section(self, parent, row):
        """Seção de estatísticas Valkey"""
        frame = ttk.LabelFrame(parent, text="Valkey Stats", padding=10)
        frame.grid(row=row, column=1, sticky='ew', padx=5, pady=5)
        
        self.valkey_stats_labels = {}
        stats = ['Streams Ativos', 'Total Entries', 'Memória MB', 'Latência ms']
        
        for i, stat in enumerate(stats):
            label = ttk.Label(frame, text=f"{stat}:")
            label.grid(row=i, column=0, sticky='w', padx=5, pady=2)
            
            value = ttk.Label(frame, text="0", font=self.parent_gui.fonts['data'])
            value.grid(row=i, column=1, sticky='e', padx=5, pady=2)
            self.valkey_stats_labels[stat] = value
    
    def _create_bridge_stats_section(self, parent, row):
        """Seção de estatísticas da bridge"""
        frame = ttk.LabelFrame(parent, text="Bridge ZMQ → Valkey", padding=10)
        frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        self.bridge_stats_labels = {}
        
        # Layout em 2 colunas
        stats = [
            ('Ticks Bridged', 'ticks_bridged'),
            ('Features Bridged', 'features_bridged'),
            ('Taxa Sucesso', 'success_rate'),
            ('Último Tick', 'last_tick')
        ]
        
        for i, (display_name, key) in enumerate(stats):
            col = (i % 2) * 2
            row_pos = i // 2
            
            label = ttk.Label(frame, text=f"{display_name}:")
            label.grid(row=row_pos, column=col, sticky='w', padx=5, pady=2)
            
            value = ttk.Label(frame, text="0", font=self.parent_gui.fonts['data'])
            value.grid(row=row_pos, column=col+1, sticky='e', padx=5, pady=2)
            self.bridge_stats_labels[key] = value
    
    def _create_time_travel_panel(self):
        """Cria painel de Time Travel"""
        # Frame principal
        main_frame = ttk.Frame(self.tt_frame, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Título
        title_label = ttk.Label(main_frame, text="Time Travel Features", 
                               font=self.parent_gui.fonts['title'])
        title_label.pack(pady=5)
        
        # Frame para features
        features_frame = ttk.LabelFrame(main_frame, text="Enhanced Features", padding=10)
        features_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Criar labels para features
        self.tt_feature_labels = {}
        features = [
            'volume_pattern_score',
            'historical_momentum_rank',
            'microstructure_imbalance',
            'volatility_regime',
            'intraday_seasonality',
            'momentum_percentile',
            'volume_profile_score',
            'price_action_quality'
        ]
        
        # Grid de features (2 colunas)
        for i, feature in enumerate(features):
            row = i // 2
            col = (i % 2) * 2
            
            label = ttk.Label(features_frame, text=f"{feature}:")
            label.grid(row=row, column=col, sticky='w', padx=5, pady=2)
            
            value = ttk.Label(features_frame, text="N/A", 
                            font=self.parent_gui.fonts['data'],
                            foreground=self.parent_gui.colors['neutral'])
            value.grid(row=row, column=col+1, sticky='e', padx=5, pady=2)
            self.tt_feature_labels[feature] = value
        
        # Frame para métricas
        metrics_frame = ttk.LabelFrame(main_frame, text="Time Travel Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.tt_metrics_labels = {}
        metrics = ['Lookback Minutes', 'Data Points', 'Data Quality', 'Cache Hits']
        
        for i, metric in enumerate(metrics):
            label = ttk.Label(metrics_frame, text=f"{metric}:")
            label.grid(row=0, column=i*2, sticky='w', padx=5)
            
            value = ttk.Label(metrics_frame, text="0", font=self.parent_gui.fonts['data'])
            value.grid(row=0, column=i*2+1, sticky='e', padx=5)
            self.tt_metrics_labels[metric] = value
    
    def _has_time_travel(self) -> bool:
        """Verifica se time travel está disponível"""
        try:
            if hasattr(self.trading_system, 'time_travel_engine'):
                return self.trading_system.time_travel_engine is not None
        except:
            pass
        return False
    
    def update_display(self):
        """Atualiza displays com dados mais recentes"""
        if not self.enhanced_active:
            return
            
        try:
            # Obter status enhanced
            status = self.trading_system.get_enhanced_status()
            
            # Atualizar status geral
            self._update_general_status(status)
            
            # Atualizar stats ZMQ
            if 'zmq_stats' in status:
                self._update_zmq_stats(status['zmq_stats'])
            
            # Atualizar stats Valkey
            if 'valkey_stats' in status:
                self._update_valkey_stats(status['valkey_stats'])
            
            # Atualizar bridge stats
            if 'bridge_stats' in status:
                self._update_bridge_stats(status['bridge_stats'])
            
            # Atualizar time travel se disponível
            if self._has_time_travel() and self.dashboard:
                self._update_time_travel_display()
                
        except Exception as e:
            self.parent_gui.logger.error(f"Erro ao atualizar display enhanced: {e}")
    
    def _update_general_status(self, status):
        """Atualiza status geral dos componentes"""
        features = status.get('enhanced_features', {})
        
        # ZMQ
        if 'ZMQ' in self.status_labels:
            if features.get('zmq_enabled'):
                self.status_labels['ZMQ'].config(text="ON", foreground='green')
            else:
                self.status_labels['ZMQ'].config(text="OFF", foreground='red')
        
        # Valkey
        if 'Valkey' in self.status_labels:
            if features.get('valkey_enabled'):
                self.status_labels['Valkey'].config(text="ON", foreground='green')
            else:
                self.status_labels['Valkey'].config(text="OFF", foreground='red')
        
        # Time Travel
        if 'Time Travel' in self.status_labels:
            if features.get('time_travel_enabled'):
                self.status_labels['Time Travel'].config(text="ON", foreground='green')
            else:
                self.status_labels['Time Travel'].config(text="OFF", foreground='red')
        
        # Bridge
        if 'Bridge' in self.status_labels:
            if features.get('bridge_active'):
                self.status_labels['Bridge'].config(text="ON", foreground='green')
            else:
                self.status_labels['Bridge'].config(text="OFF", foreground='red')
    
    def _update_zmq_stats(self, stats):
        """Atualiza estatísticas ZMQ"""
        if 'Ticks Publicados' in self.zmq_stats_labels:
            self.zmq_stats_labels['Ticks Publicados'].config(
                text=f"{stats.get('ticks_published', 0):,}"
            )
        
        if 'Trades Publicados' in self.zmq_stats_labels:
            self.zmq_stats_labels['Trades Publicados'].config(
                text=f"{stats.get('trades_published', 0):,}"
            )
        
        if 'Erros' in self.zmq_stats_labels:
            errors = stats.get('errors', 0)
            self.zmq_stats_labels['Erros'].config(
                text=str(errors),
                foreground='red' if errors > 0 else self.parent_gui.colors['text']
            )
        
        # Calcular taxa
        if 'Taxa/seg' in self.zmq_stats_labels:
            # Implementar cálculo de taxa
            self.zmq_stats_labels['Taxa/seg'].config(text="N/A")
    
    def _update_valkey_stats(self, stats):
        """Atualiza estatísticas Valkey"""
        if 'Streams Ativos' in self.valkey_stats_labels:
            self.valkey_stats_labels['Streams Ativos'].config(
                text=str(stats.get('active_streams', 0))
            )
        
        if 'Total Entries' in self.valkey_stats_labels:
            total = sum(s.get('length', 0) for s in stats.get('streams', {}).values())
            self.valkey_stats_labels['Total Entries'].config(text=f"{total:,}")
        
        # Memória e latência seriam obtidas do dashboard
        if 'Memória MB' in self.valkey_stats_labels:
            self.valkey_stats_labels['Memória MB'].config(text="N/A")
        
        if 'Latência ms' in self.valkey_stats_labels:
            self.valkey_stats_labels['Latência ms'].config(text="N/A")
    
    def _update_bridge_stats(self, stats):
        """Atualiza estatísticas da bridge"""
        if 'ticks_bridged' in self.bridge_stats_labels:
            self.bridge_stats_labels['ticks_bridged'].config(
                text=f"{stats.get('ticks_bridged', 0):,}"
            )
        
        if 'features_bridged' in self.bridge_stats_labels:
            self.bridge_stats_labels['features_bridged'].config(
                text=f"{stats.get('features_bridged', 0):,}"
            )
        
        if 'success_rate' in self.bridge_stats_labels:
            ticks = stats.get('ticks_bridged', 0)
            errors = stats.get('errors', 0)
            if ticks > 0:
                rate = (ticks / (ticks + errors)) * 100
                self.bridge_stats_labels['success_rate'].config(text=f"{rate:.1f}%")
            else:
                self.bridge_stats_labels['success_rate'].config(text="N/A")
        
        if 'last_tick' in self.bridge_stats_labels:
            last_time = stats.get('last_tick_time')
            if last_time:
                if isinstance(last_time, datetime):
                    self.bridge_stats_labels['last_tick'].config(
                        text=last_time.strftime("%H:%M:%S")
                    )
                else:
                    self.bridge_stats_labels['last_tick'].config(text="N/A")
    
    def _update_time_travel_display(self):
        """Atualiza display de time travel"""
        try:
            # Obter últimas features do dashboard
            symbol = self.trading_system.ticker
            if self.dashboard and symbol:
                latest_features = self.dashboard.valkey_manager.get_latest_features(symbol)
                
                if latest_features and 'features' in latest_features:
                    features_data = latest_features['features']
                    
                    # Atualizar labels de features
                    for feature_name, label in self.tt_feature_labels.items():
                        if feature_name in features_data:
                            value = features_data[feature_name]
                            if isinstance(value, float):
                                label.config(text=f"{value:.4f}")
                                
                                # Colorir baseado no valor
                                if value > 0.7:
                                    label.config(foreground=self.parent_gui.colors['profit'])
                                elif value < 0.3:
                                    label.config(foreground=self.parent_gui.colors['loss'])
                                else:
                                    label.config(foreground=self.parent_gui.colors['neutral'])
                            else:
                                label.config(text=str(value))
                        else:
                            label.config(text="N/A")
                    
                    # Atualizar métricas
                    if 'lookback_minutes' in features_data:
                        self.tt_metrics_labels['Lookback Minutes'].config(
                            text=str(features_data['lookback_minutes'])
                        )
                    
                    if 'data_points' in features_data:
                        self.tt_metrics_labels['Data Points'].config(
                            text=f"{features_data['data_points']:,}"
                        )
                    
                    if 'data_quality' in features_data:
                        quality = features_data['data_quality']
                        self.tt_metrics_labels['Data Quality'].config(
                            text=f"{quality:.2%}"
                        )
                        
        except Exception as e:
            self.parent_gui.logger.error(f"Erro ao atualizar time travel: {e}")


def integrate_zmq_valkey_monitor(parent_gui, trading_system):
    """
    Função helper para integrar a extensão ao monitor existente
    """
    try:
        extension = ZMQValkeyMonitorExtension(parent_gui, trading_system)
        
        # Adicionar ao loop de atualização do GUI pai
        if hasattr(parent_gui, 'update_functions'):
            parent_gui.update_functions.append(extension.update_display)
        else:
            # Criar lista se não existir
            parent_gui.update_functions = [extension.update_display]
        
        return extension
        
    except Exception as e:
        parent_gui.logger.error(f"Erro ao integrar extensão ZMQ/Valkey: {e}")
        return None