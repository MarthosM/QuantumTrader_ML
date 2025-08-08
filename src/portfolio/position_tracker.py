"""
PositionTracker - Sistema de rastreamento de posições e P&L em tempo real
Monitora posições abertas, calcula lucros/perdas e mantém histórico
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
from enum import Enum
import threading
import logging
import json
from dataclasses import dataclass, field

@dataclass
class Position:
    """Representa uma posição aberta"""
    symbol: str
    quantity: int  # Positivo para long, negativo para short
    entry_price: float
    entry_time: datetime
    entry_order_id: str
    current_price: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    commission_paid: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    
    def update_price(self, price: float):
        """Atualiza preço e P&L não realizado"""
        self.current_price = price
        self.last_update = datetime.now()
        
        # Calcular P&L não realizado
        if self.quantity > 0:  # Long
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:  # Short
            self.unrealized_pnl = (self.entry_price - price) * abs(self.quantity)
            
        # Atualizar máximos
        self.max_profit = max(self.max_profit, self.unrealized_pnl)
        self.max_loss = min(self.max_loss, self.unrealized_pnl)

@dataclass
class Trade:
    """Representa um trade completo (entrada + saída)"""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # 'LONG' ou 'SHORT'
    pnl: float
    pnl_pct: float
    commission: float
    holding_time: timedelta
    entry_order_id: str
    exit_order_id: str
    max_profit: float = 0.0
    max_loss: float = 0.0
    
    def to_dict(self) -> Dict:
        """Converte para dicionário"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'side': self.side,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'commission': self.commission,
            'holding_time_seconds': self.holding_time.total_seconds(),
            'max_profit': self.max_profit,
            'max_loss': self.max_loss
        }

@dataclass
class PortfolioMetrics:
    """Métricas do portfolio em tempo real"""
    total_value: float = 0.0
    cash_balance: float = 0.0
    positions_value: float = 0.0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    open_positions: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class PositionTracker:
    """
    Sistema completo de rastreamento de posições
    
    Features:
    - Rastreamento de posições abertas
    - Cálculo de P&L realizado e não realizado
    - Histórico detalhado de trades
    - Métricas de performance em tempo real
    - Suporte a múltiplos símbolos
    - Thread-safe
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Capital inicial
        self.initial_capital = config.get('initial_capital', 100000.0)
        self.cash_balance = self.initial_capital
        
        # Posições abertas
        self.positions = {}  # symbol -> Position
        
        # Histórico
        self.trades = deque(maxlen=10000)  # Trades completos
        self.trade_history = deque(maxlen=50000)  # Todas as transações
        
        # Métricas diárias
        self.daily_metrics = {
            'start_balance': self.initial_capital,
            'pnl': 0.0,
            'trades': 0,
            'volume': 0,
            'reset_time': datetime.now().replace(hour=0, minute=0, second=0)
        }
        
        # Performance tracking
        self.equity_curve = deque(maxlen=10000)
        self.peak_equity = self.initial_capital
        
        # Estatísticas por símbolo
        self.symbol_stats = defaultdict(lambda: {
            'trades': 0,
            'pnl': 0.0,
            'volume': 0,
            'win_rate': 0.0
        })
        
        # Threading
        self.lock = threading.RLock()
        self.update_thread = None
        self.is_running = False
        
        # Callbacks
        self.callbacks = {
            'on_position_opened': [],
            'on_position_closed': [],
            'on_pnl_update': [],
            'on_drawdown_alert': []
        }
        
        # Configurações
        self.drawdown_alert_threshold = config.get('drawdown_alert_threshold', 0.05)
        self.update_interval_seconds = config.get('update_interval_seconds', 1)
        
    def start(self):
        """Inicia o PositionTracker"""
        self.logger.info("Iniciando PositionTracker")
        self.is_running = True
        
        # Thread de atualização
        self.update_thread = threading.Thread(
            target=self._update_loop,
            name="PositionUpdate"
        )
        self.update_thread.daemon = True
        self.update_thread.start()
        
        self.logger.info(f"[OK] PositionTracker iniciado - Capital: ${self.initial_capital:,.2f}")
        
    def stop(self):
        """Para o PositionTracker"""
        self.logger.info("Parando PositionTracker...")
        self.is_running = False
        
        if self.update_thread:
            self.update_thread.join(timeout=2)
            
        self._log_final_report()
        
    def open_position(self, order_data: Dict):
        """
        Abre uma nova posição ou aumenta uma existente
        
        Args:
            order_data: Dados da ordem executada
        """
        with self.lock:
            symbol = order_data['symbol']
            side = order_data['side']
            quantity = order_data['quantity']
            price = order_data['price']
            commission = order_data.get('commission', 0)
            order_id = order_data.get('order_id', '')
            
            # Determinar quantidade com sinal
            signed_quantity = quantity if side == 'BUY' else -quantity
            
            if symbol in self.positions:
                # Aumentar ou reduzir posição existente
                self._adjust_position(symbol, signed_quantity, price, commission, order_id)
            else:
                # Nova posição
                self._create_position(symbol, signed_quantity, price, commission, order_id)
                
            # Atualizar cash
            self.cash_balance -= (quantity * price + commission)
            
            # Registrar transação
            self._record_transaction(order_data)
            
            # Disparar callback
            self._trigger_callback('on_position_opened', symbol, self.positions.get(symbol))
            
            self.logger.info(f"Posição aberta: {side} {quantity} {symbol} @ ${price:.2f}")
            
    def close_position(self, order_data: Dict):
        """
        Fecha uma posição parcial ou totalmente
        
        Args:
            order_data: Dados da ordem executada
        """
        with self.lock:
            symbol = order_data['symbol']
            side = order_data['side']
            quantity = order_data['quantity']
            price = order_data['price']
            commission = order_data.get('commission', 0)
            order_id = order_data.get('order_id', '')
            
            if symbol not in self.positions:
                self.logger.warning(f"Tentativa de fechar posição inexistente: {symbol}")
                return
                
            position = self.positions[symbol]
            
            # Calcular P&L realizado
            if position.quantity > 0:  # Fechando long
                if side != 'SELL':
                    self.logger.error(f"Side incorreto para fechar long: {side}")
                    return
                pnl = (price - position.entry_price) * min(quantity, position.quantity)
            else:  # Fechando short
                if side != 'BUY':
                    self.logger.error(f"Side incorreto para fechar short: {side}")
                    return
                pnl = (position.entry_price - price) * min(quantity, abs(position.quantity))
                
            # Subtrair comissões (entrada + saída)
            close_quantity = min(quantity, abs(position.quantity))
            proportional_entry_commission = position.commission_paid * (close_quantity / abs(position.quantity))
            total_commission = proportional_entry_commission + commission
            pnl -= total_commission
            
            # Atualizar cash
            self.cash_balance += (quantity * price - commission)
            
            # Registrar trade se fechar completamente
            close_quantity = min(quantity, abs(position.quantity))
            
            if close_quantity >= abs(position.quantity):
                # Fechamento total
                self._complete_trade(position, price, pnl, total_commission, order_id)
                del self.positions[symbol]
                self.logger.info(f"Posição fechada: {symbol} - P&L: ${pnl:.2f}")
                
                # Disparar callback
                self._trigger_callback('on_position_closed', symbol, pnl)
            else:
                # Fechamento parcial
                self._reduce_position(position, close_quantity, pnl)
                self.logger.info(f"Posição reduzida: {symbol} - P&L parcial: ${pnl:.2f}")
                
            # Registrar transação
            self._record_transaction(order_data)
            
            # Atualizar métricas
            self.daily_metrics['pnl'] += pnl
            
    def update_price(self, symbol: str, price: float):
        """
        Atualiza preço de uma posição
        
        Args:
            symbol: Símbolo do ativo
            price: Novo preço
        """
        with self.lock:
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
                
    def get_position(self, symbol: str) -> Optional[Position]:
        """Retorna posição de um símbolo"""
        with self.lock:
            return self.positions.get(symbol)
            
    def get_all_positions(self) -> Dict[str, Position]:
        """Retorna todas as posições abertas"""
        with self.lock:
            return self.positions.copy()
            
    def get_portfolio_value(self) -> float:
        """Calcula valor total do portfolio"""
        with self.lock:
            positions_value = sum(
                abs(pos.quantity) * pos.current_price
                for pos in self.positions.values()
            )
            return self.cash_balance + positions_value
            
    def get_metrics(self) -> PortfolioMetrics:
        """Retorna métricas atuais do portfolio"""
        with self.lock:
            # Calcular valores
            positions_value = sum(
                abs(pos.quantity) * pos.current_price
                for pos in self.positions.values()
            )
            
            total_value = self.cash_balance + positions_value
            
            # P&L não realizado
            unrealized_pnl = sum(
                pos.unrealized_pnl
                for pos in self.positions.values()
            )
            
            # P&L total
            total_pnl = total_value - self.initial_capital
            realized_pnl = total_pnl - unrealized_pnl
            
            # Estatísticas de trades
            winning_trades = sum(1 for t in self.trades if t.pnl > 0)
            losing_trades = sum(1 for t in self.trades if t.pnl < 0)
            total_trades = len(self.trades)
            
            # Win rate
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Profit factor
            gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Sharpe ratio (simplificado)
            if len(self.equity_curve) > 30:
                returns = pd.Series([
                    (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
                    for i in range(1, len(self.equity_curve))
                ])
                sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
                
            # Drawdown
            current_drawdown = 0
            if self.peak_equity > 0:
                current_drawdown = (self.peak_equity - total_value) / self.peak_equity
                
            max_drawdown = self._calculate_max_drawdown()
            
            return PortfolioMetrics(
                total_value=total_value,
                cash_balance=self.cash_balance,
                positions_value=positions_value,
                total_pnl=total_pnl,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                daily_pnl=self.daily_metrics['pnl'],
                open_positions=len(self.positions),
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown
            )
            
    def get_trade_history(self, symbol: Optional[str] = None, 
                         days: Optional[int] = None) -> List[Trade]:
        """
        Retorna histórico de trades
        
        Args:
            symbol: Filtrar por símbolo (opcional)
            days: Número de dias para retornar (opcional)
            
        Returns:
            Lista de trades
        """
        with self.lock:
            trades = list(self.trades)
            
            # Filtrar por símbolo
            if symbol:
                trades = [t for t in trades if t.symbol == symbol]
                
            # Filtrar por data
            if days:
                cutoff_date = datetime.now() - timedelta(days=days)
                trades = [t for t in trades if t.exit_time > cutoff_date]
                
            return trades
            
    def get_statistics(self, symbol: Optional[str] = None) -> Dict:
        """
        Calcula estatísticas detalhadas
        
        Args:
            symbol: Símbolo específico ou None para todos
            
        Returns:
            Dicionário com estatísticas
        """
        with self.lock:
            # Filtrar trades
            if symbol:
                trades = [t for t in self.trades if t.symbol == symbol]
            else:
                trades = list(self.trades)
                
            if not trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'total_pnl': 0,
                    'best_trade': 0,
                    'worst_trade': 0,
                    'avg_holding_time': 0
                }
                
            # Calcular estatísticas
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl < 0]
            
            avg_win = np.mean([t.pnl for t in wins]) if wins else 0
            avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
            
            gross_profit = sum(t.pnl for t in wins)
            gross_loss = abs(sum(t.pnl for t in losses))
            
            holding_times = [t.holding_time.total_seconds() / 3600 for t in trades]  # Em horas
            
            return {
                'total_trades': len(trades),
                'win_rate': len(wins) / len(trades) if trades else 0,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0,
                'total_pnl': sum(t.pnl for t in trades),
                'best_trade': max(t.pnl for t in trades) if trades else 0,
                'worst_trade': min(t.pnl for t in trades) if trades else 0,
                'avg_holding_time': np.mean(holding_times) if holding_times else 0,
                'total_commission': sum(t.commission for t in trades)
            }
            
    def register_callback(self, event: str, callback):
        """Registra callback para eventos"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            
    def _update_loop(self):
        """Loop de atualização contínua"""
        self.logger.info("Loop de atualização iniciado")
        
        last_reset = datetime.now().date()
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Reset diário
                if current_time.date() > last_reset:
                    self._reset_daily_metrics()
                    last_reset = current_time.date()
                    
                # Atualizar equity curve
                with self.lock:
                    total_value = self.get_portfolio_value()
                    self.equity_curve.append(total_value)
                    
                    # Atualizar peak
                    if total_value > self.peak_equity:
                        self.peak_equity = total_value
                    
                # Verificar drawdown
                if self.peak_equity > 0:
                    current_dd = (self.peak_equity - total_value) / self.peak_equity
                    if current_dd > self.drawdown_alert_threshold:
                        self._trigger_callback('on_drawdown_alert', current_dd)
                        
                # Disparar atualização de P&L
                metrics = self.get_metrics()
                self._trigger_callback('on_pnl_update', metrics)
                
                # Aguardar intervalo
                threading.Event().wait(self.update_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Erro no loop de atualização: {e}")
                threading.Event().wait(1)
                
    def _create_position(self, symbol: str, quantity: int, price: float, 
                        commission: float, order_id: str):
        """Cria nova posição"""
        
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
            entry_order_id=order_id,
            current_price=price,
            commission_paid=commission
        )
        
        self.positions[symbol] = position
        
    def _adjust_position(self, symbol: str, quantity: int, price: float,
                        commission: float, order_id: str):
        """Ajusta posição existente"""
        
        position = self.positions[symbol]
        
        # Se invertendo posição
        if (position.quantity > 0 and quantity < 0) or \
           (position.quantity < 0 and quantity > 0):
            
            # Fechar posição atual primeiro
            close_quantity = min(abs(quantity), abs(position.quantity))
            
            # Calcular P&L da parte fechada
            if position.quantity > 0:
                pnl = (price - position.entry_price) * close_quantity
            else:
                pnl = (position.entry_price - price) * close_quantity
                
            pnl -= commission * (close_quantity / abs(quantity))
            
            # Se fechar completamente e ainda sobrar, criar nova posição
            if abs(quantity) > abs(position.quantity):
                remaining = abs(quantity) - abs(position.quantity)
                remaining_signed = remaining if quantity > 0 else -remaining
                
                # Completar trade
                self._complete_trade(position, price, pnl, 
                                   commission * (close_quantity / abs(quantity)), order_id)
                
                # Criar nova posição com o restante
                self._create_position(symbol, remaining_signed, price,
                                    commission * (remaining / abs(quantity)), order_id)
            else:
                # Apenas reduzir
                self._reduce_position(position, close_quantity, pnl)
                
        else:
            # Aumentar posição (mesmo lado)
            # Calcular novo preço médio
            total_quantity = position.quantity + quantity
            total_value = (position.quantity * position.entry_price) + (quantity * price)
            
            position.quantity = total_quantity
            position.entry_price = total_value / total_quantity
            position.commission_paid += commission
            
    def _reduce_position(self, position: Position, quantity: int, pnl: float):
        """Reduz tamanho da posição"""
        
        if position.quantity > 0:
            position.quantity -= quantity
        else:
            position.quantity += quantity
            
        position.realized_pnl += pnl
        
    def _complete_trade(self, position: Position, exit_price: float,
                       pnl: float, commission: float, order_id: str):
        """Completa um trade e registra no histórico"""
        
        # Calcular P&L percentual
        pnl_pct = pnl / (abs(position.quantity) * position.entry_price)
        
        trade = Trade(
            trade_id=f"TRD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{position.symbol}",
            symbol=position.symbol,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=abs(position.quantity),
            side='LONG' if position.quantity > 0 else 'SHORT',
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,  # Total commission already calculated
            holding_time=datetime.now() - position.entry_time,
            entry_order_id=position.entry_order_id,
            exit_order_id=order_id,
            max_profit=position.max_profit,
            max_loss=position.max_loss
        )
        
        self.trades.append(trade)
        
        # Atualizar estatísticas por símbolo
        stats = self.symbol_stats[position.symbol]
        stats['trades'] += 1
        stats['pnl'] += pnl
        stats['volume'] += abs(position.quantity)
        
        # Atualizar métricas diárias
        self.daily_metrics['trades'] += 1
        self.daily_metrics['volume'] += abs(position.quantity)
        
    def _record_transaction(self, order_data: Dict):
        """Registra transação no histórico"""
        
        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': order_data['symbol'],
            'side': order_data['side'],
            'quantity': order_data['quantity'],
            'price': order_data['price'],
            'commission': order_data.get('commission', 0),
            'order_id': order_data.get('order_id', ''),
            'portfolio_value': self.get_portfolio_value()
        })
        
    def _calculate_max_drawdown(self) -> float:
        """Calcula drawdown máximo histórico"""
        
        if len(self.equity_curve) < 2:
            return 0
            
        equity = list(self.equity_curve)
        peak = equity[0]
        max_dd = 0
        
        for value in equity[1:]:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
                
        return max_dd
        
    def _reset_daily_metrics(self):
        """Reset métricas diárias"""
        
        self.logger.info("Reset de métricas diárias")
        
        current_balance = self.get_portfolio_value()
        
        self.daily_metrics = {
            'start_balance': current_balance,
            'pnl': 0.0,
            'trades': 0,
            'volume': 0,
            'reset_time': datetime.now().replace(hour=0, minute=0, second=0)
        }
        
    def _trigger_callback(self, event: str, *args):
        """Dispara callbacks registrados"""
        
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(*args)
                except Exception as e:
                    self.logger.error(f"Erro em callback {event}: {e}")
                    
    def _log_final_report(self):
        """Gera relatório final detalhado"""
        
        metrics = self.get_metrics()
        stats = self.get_statistics()
        
        self.logger.info("="*80)
        self.logger.info("Relatório Final - PositionTracker")
        self.logger.info("="*80)
        self.logger.info(f"Capital inicial: ${self.initial_capital:,.2f}")
        self.logger.info(f"Valor final: ${metrics.total_value:,.2f}")
        self.logger.info(f"P&L Total: ${metrics.total_pnl:,.2f} ({(metrics.total_pnl/self.initial_capital)*100:.2f}%)")
        self.logger.info(f"P&L Realizado: ${metrics.realized_pnl:,.2f}")
        self.logger.info(f"P&L Não Realizado: ${metrics.unrealized_pnl:,.2f}")
        self.logger.info("")
        self.logger.info(f"Total de trades: {stats['total_trades']}")
        self.logger.info(f"Taxa de acerto: {stats['win_rate']:.2%}")
        self.logger.info(f"Profit Factor: {stats['profit_factor']:.2f}")
        self.logger.info(f"Lucro médio: ${stats['avg_win']:.2f}")
        self.logger.info(f"Perda média: ${stats['avg_loss']:.2f}")
        self.logger.info(f"Melhor trade: ${stats['best_trade']:.2f}")
        self.logger.info(f"Pior trade: ${stats['worst_trade']:.2f}")
        self.logger.info(f"Tempo médio: {stats['avg_holding_time']:.1f} horas")
        self.logger.info(f"Drawdown máximo: {metrics.max_drawdown:.2%}")
        self.logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        
        # Estatísticas por símbolo
        if self.symbol_stats:
            self.logger.info("\nEstatísticas por símbolo:")
            for symbol, stats in self.symbol_stats.items():
                if stats['trades'] > 0:
                    self.logger.info(f"  {symbol}: {stats['trades']} trades, "
                                   f"P&L: ${stats['pnl']:.2f}, "
                                   f"Volume: {stats['volume']}")
                    
    def export_trades(self, filepath: str):
        """Exporta histórico de trades"""
        
        with self.lock:
            trades_data = [t.to_dict() for t in self.trades]
            
        df = pd.DataFrame(trades_data)
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Trades exportados para: {filepath}")
        
    def export_equity_curve(self, filepath: str):
        """Exporta curva de equity"""
        
        with self.lock:
            equity_data = list(self.equity_curve)
            
        df = pd.DataFrame({
            'timestamp': pd.date_range(
                end=datetime.now(),
                periods=len(equity_data),
                freq=f'{self.update_interval_seconds}S'
            ),
            'equity': equity_data
        })
        
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Equity curve exportada para: {filepath}")