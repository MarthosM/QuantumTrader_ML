"""
RiskManager - Sistema de gestão de risco em tempo real
Valida sinais, gerencia exposição e protege capital
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from enum import Enum
import threading
import logging
import json
from dataclasses import dataclass, field

class RiskLevel(Enum):
    """Níveis de risco"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class RiskAction(Enum):
    """Ações de risco"""
    ALLOW = "ALLOW"
    REDUCE = "REDUCE"
    BLOCK = "BLOCK"
    CLOSE_ALL = "CLOSE_ALL"

@dataclass
class RiskMetrics:
    """Métricas de risco atuais"""
    current_exposure: float = 0.0
    max_exposure: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    open_positions: int = 0
    daily_loss: float = 0.0
    daily_trades: int = 0
    win_rate: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RiskLimits:
    """Limites de risco configuráveis"""
    max_position_size: int = 10
    max_open_positions: int = 5
    max_daily_loss: float = 1000.0
    max_drawdown: float = 0.10  # 10%
    max_exposure: float = 50000.0
    max_daily_trades: int = 50
    min_win_rate: float = 0.40  # 40%
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.03  # 3%
    trailing_stop_pct: float = 0.015  # 1.5%

class RiskManager:
    """
    Gerenciador de risco completo
    
    Features:
    - Validação pré-trade
    - Stop loss/Take profit automáticos
    - Trailing stop
    - Limites de exposição
    - Circuit breakers
    - Position sizing dinâmico
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Limites de risco
        self.limits = RiskLimits(
            max_position_size=config.get('max_position_size', 10),
            max_open_positions=config.get('max_open_positions', 5),
            max_daily_loss=config.get('max_daily_loss', 1000.0),
            max_drawdown=config.get('max_drawdown', 0.10),
            max_exposure=config.get('max_exposure', 50000.0),
            max_daily_trades=config.get('max_daily_trades', 50),
            min_win_rate=config.get('min_win_rate', 0.40),
            stop_loss_pct=config.get('stop_loss_pct', 0.02),
            take_profit_pct=config.get('take_profit_pct', 0.03),
            trailing_stop_pct=config.get('trailing_stop_pct', 0.015)
        )
        
        # Capital e métricas
        self.initial_capital = config.get('initial_capital', 100000.0)
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        
        # Posições abertas
        self.open_positions = {}  # symbol -> position_info
        self.position_history = deque(maxlen=1000)
        
        # Métricas diárias
        self.daily_metrics = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0,
            'reset_time': datetime.now().replace(hour=0, minute=0, second=0)
        }
        
        # Histórico para análise
        self.trade_history = deque(maxlen=100)
        self.risk_events = deque(maxlen=50)
        
        # Estado
        self.is_locked = False  # Circuit breaker ativado
        self.lock_reason = None
        self.lock_time = None
        
        # Threading
        self.lock = threading.RLock()
        self.monitor_thread = None
        self.is_running = False
        
        # Callbacks
        self.callbacks = {
            'on_risk_alert': [],
            'on_stop_loss': [],
            'on_take_profit': [],
            'on_circuit_break': []
        }
        
    def start(self):
        """Inicia o RiskManager"""
        self.logger.info("Iniciando RiskManager")
        self.is_running = True
        
        # Thread de monitoramento
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="RiskMonitor"
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info(f"[OK] RiskManager iniciado - Capital: ${self.initial_capital:,.2f}")
        
    def stop(self):
        """Para o RiskManager"""
        self.logger.info("Parando RiskManager...")
        self.is_running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
            
        self._log_final_report()
        
    def validate_signal(self, signal_info: Dict, current_price: float) -> Tuple[bool, Optional[str]]:
        """
        Valida se um sinal pode ser executado
        
        Args:
            signal_info: Informações do sinal
            current_price: Preço atual do ativo
            
        Returns:
            (aprovado, motivo_rejeição)
        """
        with self.lock:
            # 1. Verificar circuit breaker
            if self.is_locked:
                return False, f"Sistema bloqueado: {self.lock_reason}"
            
            # 2. Verificar limites diários
            if self.daily_metrics['trades'] >= self.limits.max_daily_trades:
                return False, "Limite diário de trades atingido"
            
            if self.daily_metrics['pnl'] <= -self.limits.max_daily_loss:
                return False, "Limite de perda diária atingido"
            
            # 3. Verificar número de posições abertas
            symbol = signal_info.get('symbol', 'WDOU25')
            side = signal_info.get('side', 'BUY')
            
            if len(self.open_positions) >= self.limits.max_open_positions:
                # Permitir apenas se for fechamento
                if not self._is_closing_position(symbol, side):
                    return False, "Limite de posições abertas atingido"
            
            # 4. Verificar exposição
            position_size = signal_info.get('quantity', 1)
            position_value = position_size * current_price
            
            current_exposure = self._calculate_total_exposure(current_price)
            new_exposure = current_exposure + position_value
            
            if new_exposure > self.limits.max_exposure:
                return False, f"Exposição máxima excedida: ${new_exposure:,.2f}"
            
            # 5. Verificar drawdown
            current_metrics = self.get_current_metrics()
            if current_metrics.current_drawdown > self.limits.max_drawdown:
                return False, f"Drawdown máximo atingido: {current_metrics.current_drawdown:.2%}"
            
            # 6. Verificar win rate (se houver histórico suficiente)
            if len(self.trade_history) >= 20:
                recent_win_rate = self._calculate_recent_win_rate()
                if recent_win_rate < self.limits.min_win_rate:
                    return False, f"Win rate abaixo do mínimo: {recent_win_rate:.2%}"
            
            # 7. Validação específica por tipo de sinal
            if signal_info.get('confidence', 0) < 0.6:
                return False, "Confiança do sinal muito baixa"
            
            # Aprovado
            return True, None
            
    def calculate_position_size(self, signal_info: Dict, current_price: float,
                              available_capital: float) -> int:
        """
        Calcula tamanho apropriado da posição
        
        Args:
            signal_info: Informações do sinal
            current_price: Preço atual
            available_capital: Capital disponível
            
        Returns:
            Tamanho da posição
        """
        with self.lock:
            # 1. Kelly Criterion simplificado
            confidence = signal_info.get('confidence', 0.6)
            win_rate = self._calculate_recent_win_rate() or 0.5
            
            # Kelly % = (p * b - q) / b
            # p = probabilidade de ganho, q = probabilidade de perda, b = odds
            p = win_rate
            q = 1 - p
            b = self.limits.take_profit_pct / self.limits.stop_loss_pct  # Risk/reward ratio
            
            kelly_pct = max(0, (p * b - q) / b)
            kelly_pct = min(kelly_pct, 0.25)  # Máximo 25% (Kelly/4 para ser conservador)
            
            # 2. Ajustar por confiança do sinal
            confidence_factor = (confidence - 0.5) * 2  # Mapeia 0.5-1.0 para 0-1
            adjusted_pct = kelly_pct * confidence_factor
            
            # 3. Ajustar por condições de mercado
            risk_metrics = self.get_current_metrics()
            
            # Reduzir tamanho se drawdown alto
            if risk_metrics.current_drawdown > 0.05:  # 5%
                adjusted_pct *= 0.5
            
            # Reduzir se muitas perdas recentes
            if self.daily_metrics['losses'] > self.daily_metrics['wins'] + 2:
                adjusted_pct *= 0.7
            
            # 4. Calcular posição em contratos
            position_value = available_capital * adjusted_pct
            position_size = int(position_value / current_price)
            
            # 5. Aplicar limites
            position_size = max(1, position_size)  # Mínimo 1
            position_size = min(position_size, self.limits.max_position_size)
            
            # 6. Verificar se não excede exposição
            new_exposure = self._calculate_total_exposure(current_price) + (position_size * current_price)
            if new_exposure > self.limits.max_exposure:
                # Reduzir para caber no limite
                available_exposure = self.limits.max_exposure - self._calculate_total_exposure(current_price)
                position_size = max(1, int(available_exposure / current_price))
            
            self.logger.info(f"Position sizing: Kelly={kelly_pct:.2%}, "
                           f"Adjusted={adjusted_pct:.2%}, Size={position_size}")
            
            return position_size
            
    def register_trade(self, trade_info: Dict):
        """
        Registra um novo trade
        
        Args:
            trade_info: Informações do trade executado
        """
        with self.lock:
            symbol = trade_info.get('symbol')
            side = trade_info.get('side')
            quantity = trade_info.get('quantity', 0)
            price = trade_info.get('price', 0)
            
            # Atualizar métricas diárias
            self.daily_metrics['trades'] += 1
            
            # Adicionar ao histórico
            self.trade_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'signal_info': trade_info.get('signal_info', {})
            })
            
            # Atualizar posições abertas
            if side == 'BUY':
                self._open_or_increase_position(symbol, quantity, price)
            else:  # SELL
                self._close_or_reduce_position(symbol, quantity, price)
                
            self.logger.info(f"Trade registrado: {side} {quantity} {symbol} @ ${price:.2f}")
            
    def update_position_price(self, symbol: str, current_price: float) -> Optional[RiskAction]:
        """
        Atualiza preço da posição e verifica stops
        
        Args:
            symbol: Símbolo do ativo
            current_price: Preço atual
            
        Returns:
            Ação de risco recomendada
        """
        with self.lock:
            if symbol not in self.open_positions:
                return None
                
            position = self.open_positions[symbol]
            entry_price = position['entry_price']
            quantity = position['quantity']
            
            # Calcular P&L
            if quantity > 0:  # Long
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # Short
                pnl_pct = (entry_price - current_price) / entry_price
                
            position['current_price'] = current_price
            position['pnl'] = pnl_pct * abs(quantity) * entry_price
            position['pnl_pct'] = pnl_pct
            
            # Verificar stop loss
            if pnl_pct <= -self.limits.stop_loss_pct:
                self._trigger_stop_loss(symbol, position)
                return RiskAction.CLOSE_ALL
                
            # Verificar take profit
            if pnl_pct >= self.limits.take_profit_pct:
                self._trigger_take_profit(symbol, position)
                return RiskAction.CLOSE_ALL
                
            # Atualizar trailing stop
            if pnl_pct > 0:
                self._update_trailing_stop(position, current_price)
                
                # Verificar trailing stop
                if position.get('trailing_stop'):
                    if quantity > 0 and current_price <= position['trailing_stop']:
                        self._trigger_trailing_stop(symbol, position)
                        return RiskAction.CLOSE_ALL
                    elif quantity < 0 and current_price >= position['trailing_stop']:
                        self._trigger_trailing_stop(symbol, position)
                        return RiskAction.CLOSE_ALL
                        
            return RiskAction.ALLOW
            
    def check_circuit_breakers(self) -> bool:
        """
        Verifica se algum circuit breaker deve ser ativado
        
        Returns:
            True se sistema deve ser bloqueado
        """
        with self.lock:
            # 1. Verificar perda diária
            if self.daily_metrics['pnl'] <= -self.limits.max_daily_loss:
                self._activate_circuit_breaker("Perda diária máxima atingida")
                return True
                
            # 2. Verificar drawdown
            current_metrics = self.get_current_metrics()
            if current_metrics.current_drawdown > self.limits.max_drawdown:
                self._activate_circuit_breaker(f"Drawdown máximo atingido: {current_metrics.current_drawdown:.2%}")
                return True
                
            # 3. Verificar sequência de perdas
            recent_trades = list(self.trade_history)[-10:]
            if len(recent_trades) >= 5:
                losses = sum(1 for t in recent_trades[-5:] if t.get('pnl', 0) < 0)
                if losses >= 5:
                    self._activate_circuit_breaker("5 perdas consecutivas")
                    return True
                    
            # 4. Verificar volatilidade extrema
            if self._detect_extreme_volatility():
                self._activate_circuit_breaker("Volatilidade extrema detectada")
                return True
                
            return False
            
    def get_current_metrics(self) -> RiskMetrics:
        """Retorna métricas de risco atuais"""
        with self.lock:
            # Calcular exposição total
            current_exposure = self._calculate_total_exposure()
            
            # Calcular drawdown
            current_drawdown = 0
            if self.peak_capital > 0:
                current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
                
            # Calcular win rate
            win_rate = 0
            if self.daily_metrics['trades'] > 0:
                win_rate = self.daily_metrics['wins'] / self.daily_metrics['trades']
                
            # Determinar nível de risco
            risk_level = self._calculate_risk_level(current_drawdown, current_exposure)
            
            return RiskMetrics(
                current_exposure=current_exposure,
                max_exposure=self.limits.max_exposure,
                current_drawdown=current_drawdown,
                max_drawdown=self.limits.max_drawdown,
                open_positions=len(self.open_positions),
                daily_loss=min(0, self.daily_metrics['pnl']),
                daily_trades=self.daily_metrics['trades'],
                win_rate=win_rate,
                risk_level=risk_level
            )
            
    def get_position_info(self, symbol: str) -> Optional[Dict]:
        """Retorna informações de uma posição específica"""
        with self.lock:
            return self.open_positions.get(symbol)
            
    def close_all_positions(self, reason: str = "Manual"):
        """Fecha todas as posições abertas"""
        with self.lock:
            self.logger.warning(f"Fechando todas as posições: {reason}")
            
            for symbol, position in list(self.open_positions.items()):
                self._close_position(symbol, position, reason)
                
            self._record_risk_event("CLOSE_ALL", reason)
            
    def register_callback(self, event: str, callback):
        """Registra callback para eventos de risco"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            
    def _monitor_loop(self):
        """Loop de monitoramento contínuo"""
        self.logger.info("Monitor de risco iniciado")
        
        last_check = datetime.now()
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Reset diário
                if current_time.date() > self.daily_metrics['reset_time'].date():
                    self._reset_daily_metrics()
                    
                # Verificar circuit breakers a cada 5 segundos
                if (current_time - last_check).total_seconds() > 5:
                    self.check_circuit_breakers()
                    last_check = current_time
                    
                # Pequena pausa
                threading.Event().wait(1)
                
            except Exception as e:
                self.logger.error(f"Erro no monitor de risco: {e}")
                
    def _open_or_increase_position(self, symbol: str, quantity: int, price: float):
        """Abre ou aumenta uma posição"""
        
        if symbol in self.open_positions:
            # Aumentar posição existente
            position = self.open_positions[symbol]
            old_qty = position['quantity']
            old_price = position['entry_price']
            
            # Calcular novo preço médio
            new_qty = old_qty + quantity
            new_price = ((old_qty * old_price) + (quantity * price)) / new_qty
            
            position['quantity'] = new_qty
            position['entry_price'] = new_price
            position['last_update'] = datetime.now()
            
        else:
            # Nova posição
            self.open_positions[symbol] = {
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': price,
                'current_price': price,
                'entry_time': datetime.now(),
                'last_update': datetime.now(),
                'pnl': 0,
                'pnl_pct': 0,
                'peak_price': price,
                'trailing_stop': None
            }
            
    def _close_or_reduce_position(self, symbol: str, quantity: int, exit_price: float):
        """Fecha ou reduz uma posição"""
        
        if symbol not in self.open_positions:
            self.logger.warning(f"Tentativa de fechar posição inexistente: {symbol}")
            return
            
        position = self.open_positions[symbol]
        position_qty = position['quantity']
        
        # Calcular P&L realizado
        if position_qty > 0:  # Fechando long
            pnl = (exit_price - position['entry_price']) * quantity
        else:  # Fechando short
            pnl = (position['entry_price'] - exit_price) * abs(quantity)
            
        # Atualizar métricas
        self.daily_metrics['pnl'] += pnl
        if pnl > 0:
            self.daily_metrics['wins'] += 1
        else:
            self.daily_metrics['losses'] += 1
            
        # Atualizar capital
        self.current_capital += pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        # Adicionar ao histórico
        self.position_history.append({
            'symbol': symbol,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'pnl_pct': pnl / (position['entry_price'] * quantity),
            'holding_time': datetime.now() - position['entry_time'],
            'exit_time': datetime.now()
        })
        
        # Reduzir ou remover posição
        if abs(quantity) >= abs(position_qty):
            # Fechar completamente
            del self.open_positions[symbol]
        else:
            # Reduzir posição
            if position_qty > 0:
                position['quantity'] -= quantity
            else:
                position['quantity'] += quantity
                
    def _close_position(self, symbol: str, position: Dict, reason: str):
        """Força fechamento de uma posição"""
        
        quantity = abs(position['quantity'])
        current_price = position.get('current_price', position['entry_price'])
        
        # Registrar fechamento
        self._close_or_reduce_position(symbol, quantity, current_price)
        
        self.logger.info(f"Posição fechada: {symbol} - {reason}")
        
    def _calculate_total_exposure(self, reference_price: Optional[float] = None) -> float:
        """Calcula exposição total em todas as posições"""
        
        total = 0
        for position in self.open_positions.values():
            price = position.get('current_price', position['entry_price'])
            if reference_price and position['symbol'] == 'WDOU25':
                price = reference_price
            total += abs(position['quantity']) * price
            
        return total
        
    def _calculate_recent_win_rate(self, lookback: int = 20) -> float:
        """Calcula win rate recente"""
        
        recent_positions = list(self.position_history)[-lookback:]
        if not recent_positions:
            return 0.5  # Default 50%
            
        wins = sum(1 for p in recent_positions if p['pnl'] > 0)
        return wins / len(recent_positions)
        
    def _is_closing_position(self, symbol: str, side: str) -> bool:
        """Verifica se ordem fecha posição existente"""
        
        if symbol not in self.open_positions:
            return False
            
        position = self.open_positions[symbol]
        
        # Se posição é long e ordem é sell, está fechando
        if position['quantity'] > 0 and side == 'SELL':
            return True
            
        # Se posição é short e ordem é buy, está fechando
        if position['quantity'] < 0 and side == 'BUY':
            return True
            
        return False
        
    def _calculate_risk_level(self, drawdown: float, exposure: float) -> RiskLevel:
        """Calcula nível de risco atual"""
        
        # Pontuação de risco
        risk_score = 0
        
        # Drawdown
        if drawdown > 0.08:
            risk_score += 3
        elif drawdown > 0.05:
            risk_score += 2
        elif drawdown > 0.02:
            risk_score += 1
            
        # Exposição
        exposure_pct = exposure / self.limits.max_exposure
        if exposure_pct > 0.8:
            risk_score += 2
        elif exposure_pct > 0.6:
            risk_score += 1
            
        # Perdas diárias
        daily_loss_pct = abs(self.daily_metrics['pnl']) / self.initial_capital
        if daily_loss_pct > 0.02:
            risk_score += 2
        elif daily_loss_pct > 0.01:
            risk_score += 1
            
        # Mapear para nível
        if risk_score >= 5:
            return RiskLevel.CRITICAL
        elif risk_score >= 3:
            return RiskLevel.HIGH
        elif risk_score >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
            
    def _trigger_stop_loss(self, symbol: str, position: Dict):
        """Dispara stop loss"""
        
        self.logger.warning(f"STOP LOSS ativado: {symbol} - "
                          f"P&L: {position['pnl_pct']:.2%}")
        
        # Disparar callbacks
        for callback in self.callbacks['on_stop_loss']:
            try:
                callback(symbol, position)
            except Exception as e:
                self.logger.error(f"Erro em callback stop loss: {e}")
                
        self._record_risk_event("STOP_LOSS", f"{symbol}: {position['pnl_pct']:.2%}")
        
    def _trigger_take_profit(self, symbol: str, position: Dict):
        """Dispara take profit"""
        
        self.logger.info(f"TAKE PROFIT ativado: {symbol} - "
                        f"P&L: {position['pnl_pct']:.2%}")
        
        # Disparar callbacks
        for callback in self.callbacks['on_take_profit']:
            try:
                callback(symbol, position)
            except Exception as e:
                self.logger.error(f"Erro em callback take profit: {e}")
                
        self._record_risk_event("TAKE_PROFIT", f"{symbol}: {position['pnl_pct']:.2%}")
        
    def _trigger_trailing_stop(self, symbol: str, position: Dict):
        """Dispara trailing stop"""
        
        self.logger.info(f"TRAILING STOP ativado: {symbol} - "
                        f"P&L: {position['pnl_pct']:.2%}")
        
        self._record_risk_event("TRAILING_STOP", f"{symbol}: {position['pnl_pct']:.2%}")
        
    def _update_trailing_stop(self, position: Dict, current_price: float):
        """Atualiza trailing stop"""
        
        quantity = position['quantity']
        
        if quantity > 0:  # Long
            # Atualizar peak
            if current_price > position['peak_price']:
                position['peak_price'] = current_price
                # Calcular novo trailing stop
                position['trailing_stop'] = current_price * (1 - self.limits.trailing_stop_pct)
                
        else:  # Short
            # Atualizar peak (mínimo para short)
            if current_price < position['peak_price']:
                position['peak_price'] = current_price
                # Calcular novo trailing stop
                position['trailing_stop'] = current_price * (1 + self.limits.trailing_stop_pct)
                
    def _activate_circuit_breaker(self, reason: str):
        """Ativa circuit breaker"""
        
        self.is_locked = True
        self.lock_reason = reason
        self.lock_time = datetime.now()
        
        self.logger.critical(f"CIRCUIT BREAKER ATIVADO: {reason}")
        
        # Disparar callbacks
        for callback in self.callbacks['on_circuit_break']:
            try:
                callback(reason)
            except Exception as e:
                self.logger.error(f"Erro em callback circuit breaker: {e}")
                
        self._record_risk_event("CIRCUIT_BREAKER", reason)
        
    def _detect_extreme_volatility(self) -> bool:
        """Detecta volatilidade extrema no mercado"""
        
        # Simplificado - em produção seria mais sofisticado
        recent_positions = list(self.position_history)[-10:]
        if len(recent_positions) < 5:
            return False
            
        # Verificar variação de P&L
        pnl_values = [abs(p['pnl_pct']) for p in recent_positions]
        avg_pnl_variation = np.mean(pnl_values)
        
        # Se variação média > 5%, considerar volátil
        return avg_pnl_variation > 0.05
        
    def _reset_daily_metrics(self):
        """Reset métricas diárias"""
        
        self.logger.info("Reset de métricas diárias")
        
        self.daily_metrics = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0,
            'reset_time': datetime.now().replace(hour=0, minute=0, second=0)
        }
        
        # Desbloquear se estava bloqueado por limites diários
        if self.is_locked and "diária" in (self.lock_reason or ""):
            self.is_locked = False
            self.lock_reason = None
            self.logger.info("Circuit breaker desativado (novo dia)")
            
    def _record_risk_event(self, event_type: str, details: str):
        """Registra evento de risco"""
        
        self.risk_events.append({
            'timestamp': datetime.now(),
            'type': event_type,
            'details': details,
            'metrics': self.get_current_metrics().current_drawdown
        })
        
    def _log_final_report(self):
        """Gera relatório final"""
        
        metrics = self.get_current_metrics()
        
        self.logger.info("="*60)
        self.logger.info("Relatório Final - RiskManager")
        self.logger.info("="*60)
        self.logger.info(f"Capital inicial: ${self.initial_capital:,.2f}")
        self.logger.info(f"Capital final: ${self.current_capital:,.2f}")
        self.logger.info(f"Retorno: {((self.current_capital / self.initial_capital) - 1):.2%}")
        self.logger.info(f"Drawdown máximo: {metrics.max_drawdown:.2%}")
        self.logger.info(f"Total de trades: {len(self.position_history)}")
        
        if self.position_history:
            wins = sum(1 for p in self.position_history if p['pnl'] > 0)
            win_rate = wins / len(self.position_history)
            self.logger.info(f"Win rate: {win_rate:.2%}")
            
        if self.risk_events:
            self.logger.info(f"\nEventos de risco: {len(self.risk_events)}")
            for event in list(self.risk_events)[-5:]:
                self.logger.info(f"  {event['timestamp'].strftime('%Y-%m-%d %H:%M')} - "
                               f"{event['type']}: {event['details']}")
                
    def get_statistics(self) -> Dict:
        """Retorna estatísticas completas"""
        
        with self.lock:
            metrics = self.get_current_metrics()
            
            stats = {
                'current_capital': self.current_capital,
                'initial_capital': self.initial_capital,
                'total_return': (self.current_capital / self.initial_capital) - 1,
                'current_drawdown': metrics.current_drawdown,
                'max_drawdown': metrics.max_drawdown,
                'open_positions': len(self.open_positions),
                'total_trades': len(self.position_history),
                'daily_trades': self.daily_metrics['trades'],
                'daily_pnl': self.daily_metrics['pnl'],
                'win_rate': self._calculate_recent_win_rate(),
                'risk_level': metrics.risk_level.value,
                'is_locked': self.is_locked,
                'lock_reason': self.lock_reason,
                'risk_events': len(self.risk_events)
            }
            
            return stats