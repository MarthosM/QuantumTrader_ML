#!/usr/bin/env python3
"""
Extensão do GUI para Exibição de Predições ML
Sistema de Trading v2.0

Esta extensão adiciona funcionalidades ao monitor GUI para exibir:
1. Resultados de predições em tempo real
2. Features calculadas
3. Fluxo de dados detalhado
4. Métricas de performance
"""

import tkinter as tk
from tkinter import ttk, font
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import threading
import time


class PredictionDisplayPanel:
    """Painel para exibir predições ML em tempo real"""
    
    def __init__(self, parent_frame, logger=None):
        self.parent = parent_frame
        self.logger = logger or logging.getLogger('PredictionPanel')
        
        # Dados atuais
        self.current_prediction = None
        self.prediction_history = []
        self.features_data = None
        
        # Setup do painel
        self.setup_panel()
        
    def setup_panel(self):
        """Configura o painel de predições"""
        # Frame principal para predições
        self.main_frame = ttk.LabelFrame(self.parent, text="🎯 Predições ML", padding="10")
        self.main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Frame superior - Predição atual
        self.current_frame = ttk.LabelFrame(self.main_frame, text="Predição Atual", padding="5")
        self.current_frame.pack(fill="x", pady=(0, 10))
        
        self.setup_current_prediction_display()
        
        # Frame do meio - Features importantes
        self.features_frame = ttk.LabelFrame(self.main_frame, text="Features Principais", padding="5")
        self.features_frame.pack(fill="x", pady=(0, 10))
        
        self.setup_features_display()
        
        # Frame inferior - Histórico
        self.history_frame = ttk.LabelFrame(self.main_frame, text="Histórico de Predições", padding="5")
        self.history_frame.pack(fill="both", expand=True)
        
        self.setup_history_display()
        
    def setup_current_prediction_display(self):
        """Configura exibição da predição atual"""
        # Grid para organizar informações
        
        # Linha 1: Timestamp e Status
        tk.Label(self.current_frame, text="Timestamp:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 5)
        )
        self.timestamp_label = tk.Label(self.current_frame, text="--", fg="blue")
        self.timestamp_label.grid(row=0, column=1, sticky="w", padx=(0, 20))
        
        tk.Label(self.current_frame, text="Status:", font=("Arial", 10, "bold")).grid(
            row=0, column=2, sticky="w", padx=(0, 5)
        )
        self.status_label = tk.Label(self.current_frame, text="Aguardando...", fg="orange")
        self.status_label.grid(row=0, column=3, sticky="w")
        
        # Linha 2: Direção e Magnitude
        tk.Label(self.current_frame, text="Direção:", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0)
        )
        self.direction_label = tk.Label(self.current_frame, text="--", font=("Arial", 12, "bold"))
        self.direction_label.grid(row=1, column=1, sticky="w", padx=(0, 20), pady=(5, 0))
        
        tk.Label(self.current_frame, text="Magnitude:", font=("Arial", 10, "bold")).grid(
            row=1, column=2, sticky="w", padx=(0, 5), pady=(5, 0)
        )
        self.magnitude_label = tk.Label(self.current_frame, text="--", font=("Arial", 12))
        self.magnitude_label.grid(row=1, column=3, sticky="w", pady=(5, 0))
        
        # Linha 3: Confiança e Regime
        tk.Label(self.current_frame, text="Confiança:", font=("Arial", 10, "bold")).grid(
            row=2, column=0, sticky="w", padx=(0, 5), pady=(5, 0)
        )
        self.confidence_label = tk.Label(self.current_frame, text="--", font=("Arial", 12))
        self.confidence_label.grid(row=2, column=1, sticky="w", padx=(0, 20), pady=(5, 0))
        
        tk.Label(self.current_frame, text="Regime:", font=("Arial", 10, "bold")).grid(
            row=2, column=2, sticky="w", padx=(0, 5), pady=(5, 0)
        )
        self.regime_label = tk.Label(self.current_frame, text="--", font=("Arial", 12))
        self.regime_label.grid(row=2, column=3, sticky="w", pady=(5, 0))
        
        # Linha 4: Modelo e Tempo de Processamento
        tk.Label(self.current_frame, text="Modelo:", font=("Arial", 10, "bold")).grid(
            row=3, column=0, sticky="w", padx=(0, 5), pady=(5, 0)
        )
        self.model_label = tk.Label(self.current_frame, text="--", font=("Arial", 10))
        self.model_label.grid(row=3, column=1, sticky="w", padx=(0, 20), pady=(5, 0))
        
        tk.Label(self.current_frame, text="Tempo:", font=("Arial", 10, "bold")).grid(
            row=3, column=2, sticky="w", padx=(0, 5), pady=(5, 0)
        )
        self.processing_time_label = tk.Label(self.current_frame, text="--", font=("Arial", 10))
        self.processing_time_label.grid(row=3, column=3, sticky="w", pady=(5, 0))
        
    def setup_features_display(self):
        """Configura exibição das features principais"""
        # Frame com scroll para features
        self.features_canvas = tk.Canvas(self.features_frame, height=120)
        self.features_scrollbar = ttk.Scrollbar(
            self.features_frame, orient="vertical", command=self.features_canvas.yview
        )
        self.features_scrollable_frame = ttk.Frame(self.features_canvas)
        
        self.features_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.features_canvas.configure(scrollregion=self.features_canvas.bbox("all"))
        )
        
        self.features_canvas.create_window((0, 0), window=self.features_scrollable_frame, anchor="nw")
        self.features_canvas.configure(yscrollcommand=self.features_scrollbar.set)
        
        self.features_canvas.pack(side="left", fill="both", expand=True)
        self.features_scrollbar.pack(side="right", fill="y")
        
        # Labels para features (serão criados dinamicamente)
        self.features_labels = {}
        
    def setup_history_display(self):
        """Configura exibição do histórico de predições"""
        # Treeview para histórico
        columns = ("Timestamp", "Direção", "Magnitude", "Confiança", "Regime")
        self.history_tree = ttk.Treeview(self.history_frame, columns=columns, show="headings", height=8)
        
        # Configurar colunas
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=100, anchor="center")
        
        # Scrollbar para histórico
        history_scrollbar = ttk.Scrollbar(self.history_frame, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scrollbar.set)
        
        self.history_tree.pack(side="left", fill="both", expand=True)
        history_scrollbar.pack(side="right", fill="y")
        
    def update_prediction_data(self, prediction_data: Dict):
        """Atualiza dados da predição atual"""
        try:
            if not prediction_data or 'prediction' not in prediction_data:
                return
                
            pred = prediction_data['prediction']
            
            # Atualizar predição atual
            self.timestamp_label.config(text=pred.get('timestamp', '--'))
            
            # Status baseado na confiança
            confidence = pred.get('confidence', 0)
            if confidence > 0.7:
                status = "Alta Confiança"
                status_color = "green"
            elif confidence > 0.5:
                status = "Média Confiança"
                status_color = "orange"
            else:
                status = "Baixa Confiança"
                status_color = "red"
                
            self.status_label.config(text=status, fg=status_color)
            
            # Direção com cores
            direction = pred.get('direction', 0)
            if direction > 0.1:
                direction_text = f"↗ COMPRA ({direction:.3f})"
                direction_color = "green"
            elif direction < -0.1:
                direction_text = f"↘ VENDA ({direction:.3f})"
                direction_color = "red"
            else:
                direction_text = f"→ NEUTRO ({direction:.3f})"
                direction_color = "gray"
                
            self.direction_label.config(text=direction_text, fg=direction_color)
            
            # Magnitude
            magnitude = pred.get('magnitude', 0)
            self.magnitude_label.config(text=f"{magnitude:.3f}")
            
            # Confiança com cor
            confidence_text = f"{confidence:.1%}"
            if confidence > 0.7:
                confidence_color = "green"
            elif confidence > 0.5:
                confidence_color = "orange"
            else:
                confidence_color = "red"
            self.confidence_label.config(text=confidence_text, fg=confidence_color)
            
            # Regime
            regime = pred.get('regime', 'unknown')
            self.regime_label.config(text=regime.title())
            
            # Modelo e tempo
            self.model_label.config(text=pred.get('model', '--'))
            self.processing_time_label.config(text=pred.get('processing_time', '--'))
            
            # Atualizar features se disponível
            if 'features' in prediction_data:
                self.update_features_display(prediction_data['features'])
                
            # Adicionar ao histórico
            self.add_to_history(pred)
            
            # Armazenar dados atuais
            self.current_prediction = pred
            
        except Exception as e:
            self.logger.error(f"Erro atualizando dados de predição: {e}")
            
    def update_features_display(self, features_data: Dict):
        """Atualiza exibição das features"""
        try:
            # Limpar labels existentes
            for widget in self.features_scrollable_frame.winfo_children():
                widget.destroy()
            self.features_labels.clear()
            
            # Obter features para exibir
            features_to_show = {}
            
            # Features da predição atual
            if 'sample' in features_data:
                features_to_show.update(features_data['sample'])
                
            # Últimos valores se disponível
            if 'last_values' in features_data:
                features_to_show.update(features_data['last_values'])
                
            # Mostrar contagem de features
            count = features_data.get('count', 0)
            count_label = tk.Label(
                self.features_scrollable_frame,
                text=f"Total de Features: {count}",
                font=("Arial", 10, "bold")
            )
            count_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
            
            # Exibir features principais (máximo 10)
            row = 1
            for feature_name, value in list(features_to_show.items())[:10]:
                if isinstance(value, (int, float)):
                    # Nome da feature
                    name_label = tk.Label(
                        self.features_scrollable_frame,
                        text=f"{feature_name}:",
                        font=("Arial", 9)
                    )
                    name_label.grid(row=row, column=0, sticky="w", padx=(0, 10))
                    
                    # Valor da feature
                    if isinstance(value, float):
                        value_text = f"{value:.4f}"
                    else:
                        value_text = str(value)
                        
                    value_label = tk.Label(
                        self.features_scrollable_frame,
                        text=value_text,
                        font=("Arial", 9, "bold"),
                        fg="blue"
                    )
                    value_label.grid(row=row, column=1, sticky="w")
                    
                    self.features_labels[feature_name] = (name_label, value_label)
                    row += 1
                    
            # Atualizar canvas scroll region
            self.features_scrollable_frame.update_idletasks()
            self.features_canvas.configure(scrollregion=self.features_canvas.bbox("all"))
            
        except Exception as e:
            self.logger.error(f"Erro atualizando features: {e}")
            
    def add_to_history(self, prediction: Dict):
        """Adiciona predição ao histórico"""
        try:
            # Preparar dados para inserção
            timestamp = prediction.get('timestamp', '--')
            direction = prediction.get('direction', 0)
            magnitude = prediction.get('magnitude', 0)
            confidence = prediction.get('confidence', 0)
            regime = prediction.get('regime', 'unknown')
            
            # Formatar direção para exibição
            if direction > 0.1:
                direction_text = f"↗ {direction:.3f}"
            elif direction < -0.1:
                direction_text = f"↘ {direction:.3f}"
            else:
                direction_text = f"→ {direction:.3f}"
                
            # Inserir no treeview (no topo)
            self.history_tree.insert(
                '', 0,  # Inserir no início
                values=(
                    timestamp,
                    direction_text,
                    f"{magnitude:.3f}",
                    f"{confidence:.1%}",
                    regime.title()
                )
            )
            
            # Manter apenas últimas 50 entradas
            children = self.history_tree.get_children()
            if len(children) > 50:
                for item in children[50:]:
                    self.history_tree.delete(item)
                    
            # Adicionar à lista de histórico
            self.prediction_history.insert(0, prediction)
            if len(self.prediction_history) > 50:
                self.prediction_history = self.prediction_history[:50]
                
        except Exception as e:
            self.logger.error(f"Erro adicionando ao histórico: {e}")
            
    def get_current_prediction(self) -> Optional[Dict]:
        """Retorna predição atual"""
        return self.current_prediction
        
    def get_prediction_history(self) -> List[Dict]:
        """Retorna histórico de predições"""
        return self.prediction_history.copy()
        
    def clear_history(self):
        """Limpa histórico de predições"""
        # Limpar treeview
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
            
        # Limpar lista
        self.prediction_history.clear()
        
        self.logger.info("Histórico de predições limpo")


class DataFlowStatusPanel:
    """Painel para exibir status do fluxo de dados"""
    
    def __init__(self, parent_frame, logger=None):
        self.parent = parent_frame
        self.logger = logger or logging.getLogger('DataFlowPanel')
        
        self.setup_panel()
        
    def setup_panel(self):
        """Configura painel de status do fluxo"""
        # Frame principal
        self.main_frame = ttk.LabelFrame(self.parent, text="🔄 Fluxo de Dados", padding="10")
        self.main_frame.pack(fill="x", padx=5, pady=5)
        
        # Grid de status
        # Linha 1: Status dos componentes
        tk.Label(self.main_frame, text="Candles:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 5)
        )
        self.candles_status = tk.Label(self.main_frame, text="⏳", font=("Arial", 12))
        self.candles_status.grid(row=0, column=1, sticky="w", padx=(0, 20))
        
        tk.Label(self.main_frame, text="Features:", font=("Arial", 10, "bold")).grid(
            row=0, column=2, sticky="w", padx=(0, 5)
        )
        self.features_status = tk.Label(self.main_frame, text="⏳", font=("Arial", 12))
        self.features_status.grid(row=0, column=3, sticky="w", padx=(0, 20))
        
        tk.Label(self.main_frame, text="Predição:", font=("Arial", 10, "bold")).grid(
            row=0, column=4, sticky="w", padx=(0, 5)
        )
        self.prediction_status = tk.Label(self.main_frame, text="⏳", font=("Arial", 12))
        self.prediction_status.grid(row=0, column=5, sticky="w")
        
        # Linha 2: Contadores
        tk.Label(self.main_frame, text="Total Fluxos:", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky="w", padx=(0, 5), pady=(10, 0)
        )
        self.total_flows_label = tk.Label(self.main_frame, text="0", font=("Arial", 10))
        self.total_flows_label.grid(row=1, column=1, sticky="w", padx=(0, 20), pady=(10, 0))
        
        tk.Label(self.main_frame, text="Predições:", font=("Arial", 10, "bold")).grid(
            row=1, column=2, sticky="w", padx=(0, 5), pady=(10, 0)
        )
        self.total_predictions_label = tk.Label(self.main_frame, text="0", font=("Arial", 10))
        self.total_predictions_label.grid(row=1, column=3, sticky="w", padx=(0, 20), pady=(10, 0))
        
        tk.Label(self.main_frame, text="Erros:", font=("Arial", 10, "bold")).grid(
            row=1, column=4, sticky="w", padx=(0, 5), pady=(10, 0)
        )
        self.total_errors_label = tk.Label(self.main_frame, text="0", font=("Arial", 10))
        self.total_errors_label.grid(row=1, column=5, sticky="w", pady=(10, 0))
        
    def update_flow_status(self, status_data: Dict):
        """Atualiza status do fluxo de dados"""
        try:
            current_status = status_data.get('current_status', {})
            
            # Status dos componentes
            self.candles_status.config(
                text="✅" if current_status.get('has_candle') else "❌",
                fg="green" if current_status.get('has_candle') else "red"
            )
            
            self.features_status.config(
                text="✅" if current_status.get('has_features') else "❌",
                fg="green" if current_status.get('has_features') else "red"
            )
            
            self.prediction_status.config(
                text="✅" if current_status.get('has_prediction') else "❌",
                fg="green" if current_status.get('has_prediction') else "red"
            )
            
            # Contadores
            self.total_flows_label.config(text=str(status_data.get('total_flows_processed', 0)))
            self.total_predictions_label.config(text=str(status_data.get('total_predictions', 0)))
            
            errors_count = status_data.get('total_errors', 0)
            self.total_errors_label.config(
                text=str(errors_count),
                fg="red" if errors_count > 0 else "black"
            )
            
        except Exception as e:
            self.logger.error(f"Erro atualizando status do fluxo: {e}")


def extend_gui_with_prediction_display(gui_instance):
    """
    Estende o GUI existente com painéis de predição
    
    Args:
        gui_instance: Instância do TradingMonitorGUI
    """
    try:
        logger = logging.getLogger('GUIExtension')
        logger.info("Estendendo GUI com display de predições...")
        
        # Verificar se já tem notebook (tabs)
        if hasattr(gui_instance, 'notebook'):
            # Adicionar nova aba para predições
            prediction_frame = ttk.Frame(gui_instance.notebook)
            gui_instance.notebook.add(prediction_frame, text="🎯 Predições ML")
        else:
            # Criar frame dentro do layout existente
            prediction_frame = ttk.Frame(gui_instance.root)
            prediction_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
        # Criar painéis
        # Painel de status do fluxo (superior)
        flow_status_panel = DataFlowStatusPanel(prediction_frame, logger)
        
        # Painel de predições (principal)
        prediction_panel = PredictionDisplayPanel(prediction_frame, logger)
        
        # Adicionar referências ao GUI
        gui_instance.prediction_panel = prediction_panel
        gui_instance.flow_status_panel = flow_status_panel
        
        # Adicionar métodos de atualização ao GUI
        def update_prediction_data(prediction_data):
            try:
                gui_instance.prediction_panel.update_prediction_data(prediction_data)
            except Exception as e:
                logger.error(f"Erro atualizando dados de predição: {e}")
                
        def update_flow_status(status_data):
            try:
                gui_instance.flow_status_panel.update_flow_status(status_data)
            except Exception as e:
                logger.error(f"Erro atualizando status do fluxo: {e}")
                
        gui_instance.update_prediction_data = update_prediction_data
        gui_instance.update_flow_status = update_flow_status
        
        logger.info("✅ GUI estendido com sucesso com display de predições")
        return True
        
    except Exception as e:
        logger.error(f"Erro estendendo GUI: {e}")
        return False


def main():
    """Teste dos painéis de predição"""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('PredictionDisplayTest')
    
    # Criar janela de teste
    root = tk.Tk()
    root.title("Teste - Painéis de Predição")
    root.geometry("800x600")
    
    # Criar painéis
    flow_panel = DataFlowStatusPanel(root, logger)
    pred_panel = PredictionDisplayPanel(root, logger)
    
    # Dados de teste
    test_prediction_data = {
        'prediction': {
            'timestamp': '14:30:15',
            'direction': 0.75,
            'magnitude': 0.23,
            'confidence': 0.85,
            'regime': 'trending',
            'model': 'XGBoost',
            'processing_time': '0.045s'
        },
        'features': {
            'count': 127,
            'sample': {
                'ema_9': 125430.5,
                'rsi': 68.3,
                'atr': 245.7,
                'momentum_1': 0.0034
            },
            'last_values': {
                'volume': 1250,
                'close': 125425.0,
                'high': 125440.0,
                'low': 125380.0
            }
        }
    }
    
    test_flow_status = {
        'current_status': {
            'has_candle': True,
            'has_features': True,
            'has_prediction': True
        },
        'total_flows_processed': 45,
        'total_predictions': 23,
        'total_errors': 1
    }
    
    # Atualizar com dados de teste
    pred_panel.update_prediction_data(test_prediction_data)
    flow_panel.update_flow_status(test_flow_status)
    
    logger.info("Interface de teste criada. Feche a janela para encerrar.")
    root.mainloop()


if __name__ == "__main__":
    main()
