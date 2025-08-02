"""
Rate Limiter para controlar fluxo de dados
Previne overflow do buffer em mercados de alta frequência
"""

import time
from typing import Dict, Any
from collections import deque
from datetime import datetime, timedelta


class RateLimiter:
    """
    Limita taxa de processamento para evitar overflow
    """
    
    def __init__(self, max_per_second: int = 100):
        self.max_per_second = max_per_second
        self.min_interval = 1.0 / max_per_second
        self.last_allowed = 0
        
        # Janela deslizante para contagem
        self.window = deque()
        self.window_size = 1.0  # 1 segundo
        
    def should_process(self) -> bool:
        """Verifica se deve processar baseado na taxa"""
        now = time.time()
        
        # Limpar eventos antigos da janela
        cutoff = now - self.window_size
        while self.window and self.window[0] < cutoff:
            self.window.popleft()
        
        # Verificar se excedeu limite
        if len(self.window) >= self.max_per_second:
            return False
        
        # Verificar intervalo mínimo
        if now - self.last_allowed < self.min_interval:
            return False
        
        # Permitir e registrar
        self.window.append(now)
        self.last_allowed = now
        return True
    
    def get_rate(self) -> float:
        """Retorna taxa atual (eventos/segundo)"""
        now = time.time()
        cutoff = now - self.window_size
        
        # Limpar eventos antigos
        while self.window and self.window[0] < cutoff:
            self.window.popleft()
        
        return len(self.window)


class AdaptiveRateLimiter(RateLimiter):
    """
    Rate limiter adaptativo que ajusta baseado em performance
    """
    
    def __init__(self, initial_rate: int = 100, target_buffer_usage: float = 0.7):
        super().__init__(initial_rate)
        
        self.target_buffer_usage = target_buffer_usage
        self.min_rate = 10
        self.max_rate = 1000
        
        # Histórico para ajuste
        self.adjustment_history = deque(maxlen=10)
        self.last_adjustment = time.time()
        self.adjustment_interval = 5.0  # segundos
        
    def adjust_rate(self, buffer_usage: float):
        """Ajusta taxa baseado no uso do buffer"""
        now = time.time()
        
        # Ajustar apenas periodicamente
        if now - self.last_adjustment < self.adjustment_interval:
            return
        
        # Calcular ajuste
        if buffer_usage > self.target_buffer_usage * 1.2:
            # Buffer muito cheio - reduzir taxa
            new_rate = int(self.max_per_second * 0.8)
        elif buffer_usage < self.target_buffer_usage * 0.5:
            # Buffer muito vazio - aumentar taxa
            new_rate = int(self.max_per_second * 1.2)
        else:
            # Taxa ok
            return
        
        # Aplicar limites
        new_rate = max(self.min_rate, min(self.max_rate, new_rate))
        
        if new_rate != self.max_per_second:
            self.max_per_second = new_rate
            self.min_interval = 1.0 / new_rate
            self.adjustment_history.append((now, new_rate, buffer_usage))
            self.last_adjustment = now