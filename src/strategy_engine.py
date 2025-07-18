"""
Motor de Estratégia - Sistema v2.0
Motor principal que coordena geração de sinais e gerenciamento de risco
"""

from typing import Dict, Optional, List
from datetime import datetime
import logging


class StrategyEngine:
    """Motor principal de estratégia"""
    
    def __init__(self, signal_generator, risk_manager):
        """
        Inicializa o motor de estratégia
        
        Args:
            signal_generator: Gerador de sinais
            risk_manager: Gerenciador de risco
        """
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)
        
        # Estado
        self.current_signal = None
        self.signal_history = []
        self.active_positions = []
        
        # Configurações
        self.max_signal_history = 100
        self.position_id_counter = 0
        
    def process_prediction(self, prediction: Dict, market_data, 
                         account_info: Dict) -> Optional[Dict]:
        """
        Processa predição e gera sinal validado
        
        Args:
            prediction: Resultado da predição ML
            market_data: TradingDataStructure com dados do mercado
            account_info: Informações da conta (balance, etc)
            
        Returns:
            Sinal validado ou None
        """
        self.logger.info("="*60)
        self.logger.info("[STRATEGY] Processando predição para sinal")
        
        try:
            # 1. Gerar sinal
            signal = self.signal_generator.generate_signal(prediction, market_data)
            
            if signal['action'] == 'none':
                self.logger.info(f"[STRATEGY] Sem sinal: {signal['reason']}")
                return None
            
            # 2. Adicionar ID único ao sinal
            signal['id'] = f"SIG_{self.position_id_counter:06d}"
            self.position_id_counter += 1
            
            # 3. Validar com risk manager
            account_balance = account_info.get('balance', 100000)
            is_valid, reason = self.risk_manager.validate_signal(
                signal, account_balance, self.active_positions
            )
            
            if not is_valid:
                self.logger.warning(f"[STRATEGY] Sinal rejeitado: {reason}")
                signal['action'] = 'none'
                signal['rejected'] = True
                signal['rejected_reason'] = reason
                
                # Armazenar sinal rejeitado no histórico também
                self._store_signal(signal)
                return None
            
            # 4. Sinal aprovado - preparar para execução
            signal['approved'] = True
            signal['account_balance'] = account_balance
            
            # 5. Armazenar sinal
            self.current_signal = signal
            self._store_signal(signal)
            
            self.logger.info("[STRATEGY] Sinal aprovado e pronto para execução")
            self.logger.info(f"  ID: {signal['id']}")
            self.logger.info(f"  Ação: {signal['action']}")
            self.logger.info(f"  Entrada: {signal['entry_price']:.2f}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"[STRATEGY] Erro processando predição: {e}", exc_info=True)
            return None
    
    def execute_signal(self, signal: Dict) -> bool:
        """
        Registra execução de um sinal
        
        Args:
            signal: Sinal a ser executado
            
        Returns:
            True se executado com sucesso
        """
        try:
            # Criar posição a partir do sinal
            position = {
                'id': signal['id'],
                'action': signal['action'],
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'position_size': signal['position_size'],
                'confidence': signal['confidence'],
                'open_time': datetime.now(),
                'signal_reason': signal['reason']
            }
            
            # Registrar no risk manager
            self.risk_manager.register_position(position)
            
            # Adicionar à lista de posições ativas
            self.active_positions.append(position)
            
            self.logger.info(f"[STRATEGY] Sinal {signal['id']} executado")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[STRATEGY] Erro executando sinal: {e}")
            return False
    
    def check_positions(self, market_data) -> List[Dict]:
        """
        Verifica posições abertas para stops/targets
        
        Args:
            market_data: Dados atuais do mercado
            
        Returns:
            Lista de posições que devem ser fechadas
        """
        if not self.active_positions or market_data.candles.empty:
            return []
        
        current_price = float(market_data.candles['close'].iloc[-1])
        current_high = float(market_data.candles['high'].iloc[-1])
        current_low = float(market_data.candles['low'].iloc[-1])
        
        positions_to_close = []
        
        for position in self.active_positions[:]:  # Cópia para permitir remoção
            close_position = False
            exit_reason = None
            exit_price = current_price
            
            # Verificar stops e targets
            if position['action'] == 'buy':
                # Stop loss
                if current_low <= position['stop_loss']:
                    close_position = True
                    exit_reason = 'stop_loss'
                    exit_price = position['stop_loss']
                # Take profit
                elif current_high >= position['take_profit']:
                    close_position = True
                    exit_reason = 'take_profit'
                    exit_price = position['take_profit']
                    
            else:  # sell
                # Stop loss
                if current_high >= position['stop_loss']:
                    close_position = True
                    exit_reason = 'stop_loss'
                    exit_price = position['stop_loss']
                # Take profit
                elif current_low <= position['take_profit']:
                    close_position = True
                    exit_reason = 'take_profit'
                    exit_price = position['take_profit']
            
            if close_position:
                # Fechar posição
                closed = self.risk_manager.close_position(
                    position['id'], exit_price, exit_reason
                )
                
                if closed:
                    positions_to_close.append(closed)
                    self.active_positions.remove(position)
                    
                    self.logger.info(
                        f"[STRATEGY] Posição {position['id']} fechada: "
                        f"{exit_reason} @ {exit_price:.2f}"
                    )
        
        return positions_to_close
    
    def update_trailing_stops(self, market_data):
        """Atualiza trailing stops das posições"""
        if not self.active_positions or market_data.candles.empty:
            return
        
        current_price = float(market_data.candles['close'].iloc[-1])
        
        for position in self.active_positions:
            # Calcular se deve atualizar stop
            if position['action'] == 'buy':
                # Para compra, stop sobe mas nunca desce
                profit_points = (current_price - position['entry_price']) / 0.5
                
                if profit_points >= 10:  # 10 pontos de lucro
                    # Mover stop para breakeven + 2 pontos
                    new_stop = position['entry_price'] + 1.0  # 2 pontos
                    
                    if new_stop > position['stop_loss']:
                        self.risk_manager.update_position_stop(
                            position['id'], new_stop
                        )
                        position['stop_loss'] = new_stop
                        
            else:  # sell
                # Para venda, stop desce mas nunca sobe
                profit_points = (position['entry_price'] - current_price) / 0.5
                
                if profit_points >= 10:
                    new_stop = position['entry_price'] - 1.0
                    
                    if new_stop < position['stop_loss']:
                        self.risk_manager.update_position_stop(
                            position['id'], new_stop
                        )
                        position['stop_loss'] = new_stop
    
    def get_strategy_stats(self) -> Dict:
        """Retorna estatísticas da estratégia"""
        stats = {
            'total_signals': len(self.signal_history),
            'active_positions': len(self.active_positions),
            'current_signal': self.current_signal
        }
        
        # Adicionar estatísticas de sinais
        if self.signal_history:
            approved = sum(1 for s in self.signal_history if s.get('approved', False))
            rejected = sum(1 for s in self.signal_history if s.get('rejected', False))
            
            stats['signals_approved'] = approved
            stats['signals_rejected'] = rejected
            stats['approval_rate'] = approved / len(self.signal_history)
            
            # Distribuição de sinais
            buy_signals = sum(1 for s in self.signal_history if s['action'] == 'buy')
            sell_signals = sum(1 for s in self.signal_history if s['action'] == 'sell')
            
            stats['buy_signals'] = buy_signals
            stats['sell_signals'] = sell_signals
        
        # Adicionar métricas de risco
        risk_metrics = self.risk_manager.get_risk_metrics()
        stats.update(risk_metrics)
        
        return stats
    
    def _store_signal(self, signal: Dict):
        """Armazena sinal no histórico"""
        self.signal_history.append(signal.copy())
        
        # Limitar tamanho do histórico
        if len(self.signal_history) > self.max_signal_history:
            self.signal_history.pop(0)