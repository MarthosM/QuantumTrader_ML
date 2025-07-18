"""
Gerenciador de Risco - Sistema v2.0
Gerencia risco das operações de trading
"""

from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta
import logging


class RiskManager:
    """Gerencia risco das operações"""
    
    def __init__(self, config: Dict):
        """
        Inicializa o gerenciador de risco
        
        Args:
            config: Configuração de risco
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Limites de risco
        self.max_daily_loss = config.get('max_daily_loss', 0.05)  # 5% do capital
        self.max_positions = config.get('max_positions', 1)
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)  # 2% por trade
        self.min_risk_reward = config.get('min_risk_reward', 1.5)
        
        # Controles de posição
        self.position_sizing_method = config.get('position_sizing', 'fixed')
        self.max_correlation = config.get('max_correlation', 0.7)
        
        # Estado
        self.open_positions = []
        self.closed_positions = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.now().date()
        
        # Histórico
        self.risk_events = []
        
    def validate_signal(self, signal: Dict, account_balance: float, 
                       current_positions: Optional[List[Dict]] = None) -> Tuple[bool, str]:
        """
        Valida se sinal pode ser executado considerando risco
        
        Args:
            signal: Sinal a ser validado
            account_balance: Saldo da conta
            current_positions: Posições atuais (opcional)
            
        Returns:
            Tuple (is_valid, reason)
        """
        self.logger.info("[RISK] Validando sinal...")
        
        # Resetar contadores diários se necessário
        self._check_daily_reset()
        
        # Usar posições fornecidas ou internas
        positions = current_positions if current_positions is not None else self.open_positions
        
        # 1. Verificar limite de posições
        if len(positions) >= self.max_positions:
            self.logger.warning(f"[RISK] Limite de posições atingido: {len(positions)}/{self.max_positions}")
            return False, "max_positions_reached"
        
        # 2. Verificar perda diária
        max_loss_amount = self.max_daily_loss * account_balance
        if self.daily_pnl < -max_loss_amount:
            self.logger.warning(f"[RISK] Limite de perda diária atingido: {self.daily_pnl:.2f}")
            return False, "daily_loss_limit_reached"
        
        # 3. Verificar risco por trade
        trade_risk = self._calculate_trade_risk(signal, account_balance)
        if trade_risk > self.max_risk_per_trade:
            self.logger.warning(f"[RISK] Risco por trade muito alto: {trade_risk:.2%}")
            return False, "risk_per_trade_too_high"
        
        # 4. Verificar risk/reward
        risk_reward = signal.get('risk_reward', 0)
        if risk_reward < self.min_risk_reward:
            self.logger.warning(f"[RISK] Risk/reward muito baixo: {risk_reward:.2f}")
            return False, "risk_reward_too_low"
        
        # 5. Verificar correlação com posições existentes
        if positions and not self._check_correlation(signal, positions):
            self.logger.warning("[RISK] Alta correlação com posições existentes")
            return False, "high_correlation"
        
        # 6. Verificar horário de trading
        if not self._check_trading_hours():
            self.logger.warning("[RISK] Fora do horário de trading")
            return False, "outside_trading_hours"
        
        # 7. Verificar volatilidade do mercado
        if 'atr' in signal.get('metadata', {}):
            if not self._check_market_volatility(signal['metadata']['atr'], signal['entry_price']):
                self.logger.warning("[RISK] Volatilidade do mercado inadequada")
                return False, "inadequate_volatility"
        
        self.logger.info("[RISK] Sinal aprovado")
        return True, "approved"
    
    def register_position(self, position: Dict):
        """Registra nova posição aberta"""
        position_copy = position.copy()
        position_copy['open_time'] = datetime.now()
        position_copy['status'] = 'open'
        
        self.open_positions.append(position_copy)
        self.daily_trades += 1
        
        self.logger.info(
            f"[RISK] Posição registrada: {position['action']} @ {position['entry_price']:.2f}"
        )
    
    def close_position(self, position_id: str, exit_price: float, exit_reason: str) -> Dict:
        """Fecha posição e calcula P&L"""
        position = None
        for i, pos in enumerate(self.open_positions):
            if pos.get('id') == position_id or i == int(position_id):
                position = self.open_positions.pop(i)
                break
        
        if not position:
            self.logger.error(f"[RISK] Posição {position_id} não encontrada")
            return {}
        
        # Calcular P&L
        if position['action'] == 'buy':
            pnl = (exit_price - position['entry_price']) * position['position_size']
            pnl_pct = (exit_price / position['entry_price'] - 1)
        else:  # sell
            pnl = (position['entry_price'] - exit_price) * position['position_size']
            pnl_pct = (1 - exit_price / position['entry_price'])
        
        # Atualizar posição
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now()
        position['exit_reason'] = exit_reason
        position['pnl'] = pnl
        position['pnl_pct'] = pnl_pct
        position['status'] = 'closed'
        
        # Adicionar ao histórico
        self.closed_positions.append(position)
        
        # Atualizar P&L diário
        self.daily_pnl += pnl
        
        self.logger.info(
            f"[RISK] Posição fechada: {position['action']} @ {exit_price:.2f}, "
            f"P&L: {pnl:.2f} ({pnl_pct:.2%})"
        )
        
        return position
    
    def update_position_stop(self, position_id: str, new_stop: float) -> bool:
        """Atualiza stop loss de uma posição"""
        for position in self.open_positions:
            if position.get('id') == position_id:
                old_stop = position['stop_loss']
                position['stop_loss'] = new_stop
                position['stop_updated'] = datetime.now()
                
                self.logger.info(
                    f"[RISK] Stop atualizado: {old_stop:.2f} -> {new_stop:.2f}"
                )
                return True
        
        return False
    
    def get_risk_metrics(self) -> Dict:
        """Retorna métricas de risco atuais"""
        total_trades = len(self.closed_positions)
        winning_trades = sum(1 for p in self.closed_positions if p['pnl'] > 0)
        
        metrics = {
            'open_positions': len(self.open_positions),
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'total_trades': total_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'avg_win': self._calculate_avg_win(),
            'avg_loss': self._calculate_avg_loss(),
            'profit_factor': self._calculate_profit_factor(),
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }
        
        return metrics
    
    def _check_daily_reset(self):
        """Verifica e reseta contadores diários se necessário"""
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset = today
            self.logger.info("[RISK] Contadores diários resetados")
    
    def _calculate_trade_risk(self, signal: Dict, account_balance: float) -> float:
        """Calcula risco percentual do trade"""
        entry = signal['entry_price']
        stop = signal['stop_loss']
        size = signal['position_size']
        
        risk_amount = abs(entry - stop) * size
        risk_pct = risk_amount / account_balance
        
        return risk_pct
    
    def _check_correlation(self, signal: Dict, positions: List[Dict]) -> bool:
        """Verifica correlação com posições existentes"""
        # Por enquanto, apenas verificar se não há posição na mesma direção
        same_direction = any(
            p['action'] == signal['action'] for p in positions
        )
        
        return not same_direction or len(positions) < self.max_positions
    
    def _check_trading_hours(self) -> bool:
        """Verifica se está no horário de trading"""
        now = datetime.now()
        
        # WDO: 9:00 às 17:55
        if now.weekday() > 4:  # Fim de semana
            return False
        
        current_time = now.time()
        market_open = datetime.strptime("09:00", "%H:%M").time()
        market_close = datetime.strptime("17:55", "%H:%M").time()
        
        return market_open <= current_time <= market_close
    
    def _check_market_volatility(self, atr: float, price: float) -> bool:
        """Verifica se volatilidade está adequada"""
        atr_pct = atr / price
        
        # Evitar mercados muito voláteis ou muito parados
        min_vol = 0.0001  # 0.01%
        max_vol = 0.01   # 1%
        
        return min_vol <= atr_pct <= max_vol
    
    def _calculate_avg_win(self) -> float:
        """Calcula ganho médio"""
        wins = [p['pnl'] for p in self.closed_positions if p['pnl'] > 0]
        return sum(wins) / len(wins) if wins else 0.0
    
    def _calculate_avg_loss(self) -> float:
        """Calcula perda média"""
        losses = [abs(p['pnl']) for p in self.closed_positions if p['pnl'] < 0]
        return sum(losses) / len(losses) if losses else 0.0
    
    def _calculate_profit_factor(self) -> float:
        """Calcula fator de lucro"""
        gross_profit = sum(p['pnl'] for p in self.closed_positions if p['pnl'] > 0)
        gross_loss = abs(sum(p['pnl'] for p in self.closed_positions if p['pnl'] < 0))
        
        if gross_loss > 0:
            return gross_profit / gross_loss
        return float('inf') if gross_profit > 0 else 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calcula drawdown máximo"""
        if not self.closed_positions:
            return 0.0
        
        cumulative_pnl = []
        cum_sum = 0
        
        for position in self.closed_positions:
            cum_sum += position['pnl']
            cumulative_pnl.append(cum_sum)
        
        if not cumulative_pnl:
            return 0.0
        
        peak = cumulative_pnl[0]
        max_dd = 0
        
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = (peak - pnl) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calcula Sharpe Ratio simplificado"""
        if len(self.closed_positions) < 2:
            return 0.0
        
        returns = [p['pnl_pct'] for p in self.closed_positions]
        
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        
        if std_return > 0:
            return (avg_return * 252 ** 0.5) / std_return  # Anualizado
        return 0.0