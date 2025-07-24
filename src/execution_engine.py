# src/execution/execution_engine.py
import logging
from typing import Dict, List, Optional
from datetime import datetime
import threading

# Imports necessários para execução de ordens
try:
    from .order_manager import OrderStatus, OrderType, OrderSide, Order
except ImportError:
    # Fallback para desenvolvimento/testes
    from order_manager import OrderStatus, OrderType, OrderSide, Order

class SimpleExecutionEngine:
    """Engine de execução simplificado integrado com ML"""
    
    def __init__(self, order_manager, ml_coordinator, risk_manager):
        self.order_manager = order_manager
        self.ml_coordinator = ml_coordinator
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)
        
        # Estado
        self.active_orders = {}
        self.position_tracker = {}
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_slippage': 0.0
        }
        
        # Configurações
        self.config = {
            'max_position_size': 3,
            'max_orders_per_minute': 10,
            'enable_stop_orders': True,
            'enable_take_profit': True,
            'default_order_type': 'limit',
            'limit_price_offset': 1.0  # ticks
        }
        
    def process_ml_signal(self, ml_signal: Dict) -> Optional[str]:
        """
        Processa sinal do ML e executa ordem se apropriado
        
        Args:
            ml_signal: Sinal gerado pelo MLCoordinator
            
        Returns:
            order_id se ordem foi enviada, None caso contrário
        """
        try:
            # Validar sinal
            if not self._validate_ml_signal(ml_signal):
                return None
            
            # Verificar gestão de risco
            risk_check = self.risk_manager.check_signal_risk(ml_signal)
            if not risk_check['approved']:
                self.logger.warning(f"Sinal rejeitado pelo risco: {risk_check['reason']}")
                return None
            
            # Preparar ordem
            order_params = self._prepare_order_params(ml_signal, risk_check)
            
            # Enviar ordem
            order = self.order_manager.send_order(order_params)
            
            if order:
                self.active_orders[str(order.profit_id)] = order
                self.execution_stats['total_orders'] += 1
                
                # Registrar callback para acompanhar ordem
                self.order_manager.register_order_callback(self._on_order_update)
                
                return str(order.profit_id)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro processando sinal ML: {e}")
            return None
    
    def _validate_ml_signal(self, signal: Dict) -> bool:
        """Valida sinal do ML"""
        required_fields = ['symbol', 'action', 'confidence', 'prediction']
        
        for field in required_fields:
            if field not in signal:
                self.logger.error(f"Campo obrigatório ausente: {field}")
                return False
        
        # Validar confiança mínima
        if signal['confidence'] < 0.6:
            self.logger.info(f"Confiança insuficiente: {signal['confidence']}")
            return False
        
        # Validar ação
        if signal['action'] not in ['buy', 'sell', 'none']:
            self.logger.error(f"Ação inválida: {signal['action']}")
            return False
        
        # Ignorar sinais 'none'
        if signal['action'] == 'none':
            return False
        
        return True
    
    def _prepare_order_params(self, ml_signal: Dict, risk_check: Dict) -> Dict:
        """Prepara parâmetros da ordem baseado no sinal e risco"""
        # Calcular quantidade baseado no risco
        position_size = risk_check.get('position_size', 1)
        
        # Tipo de ordem baseado na configuração
        order_type = self.config['default_order_type']
        
        # Preparar parâmetros
        order_params = {
            'symbol': ml_signal['symbol'],
            'action': ml_signal['action'],
            'quantity': position_size,
            'order_type': order_type
        }
        
        # Se ordem limite, calcular preço
        if order_type == 'limit':
            current_price = self._get_current_price(ml_signal['symbol'])
            offset = self.config['limit_price_offset']
            
            if ml_signal['action'] == 'buy':
                order_params['price'] = current_price + offset
            else:
                order_params['price'] = current_price - offset
        
        # Adicionar stops se configurado
        if self.config['enable_stop_orders'] and 'stop_loss' in risk_check:
            order_params['stop_loss'] = risk_check['stop_loss']
        
        if self.config['enable_take_profit'] and 'take_profit' in risk_check:
            order_params['take_profit'] = risk_check['take_profit']
        
        return order_params
    
    def _on_order_update(self, order):
        """Callback para atualizações de ordem"""
        try:
            order_id = str(order.profit_id)
            
            if order.status == OrderStatus.FILLED:
                self.execution_stats['successful_orders'] += 1
                
                # Calcular slippage se ordem limite
                if order.order_type == OrderType.LIMIT and hasattr(order, 'fill_price'):
                    slippage = abs(order.fill_price - order.price)
                    self.execution_stats['total_slippage'] += slippage
                
                # Atualizar posição
                self._update_position(order)
                
                # Criar ordens de proteção se necessário
                self._create_protection_orders(order)
                
                self.logger.info(f"Ordem {order_id} executada com sucesso: {order.filled_qty} @ {order.fill_price}")
                
            elif order.status in [OrderStatus.REJECTED, OrderStatus.ERROR]:
                self.execution_stats['failed_orders'] += 1
                self.logger.warning(f"Ordem {order_id} falhou: {order.status}")
                
            # Remover de ordens ativas se finalizada
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                               OrderStatus.REJECTED, OrderStatus.ERROR]:
                self.active_orders.pop(order_id, None)
                
        except Exception as e:
            self.logger.error(f"Erro processando update de ordem: {e}")
            # Não interromper o fluxo, apenas logar o erro
    
    def _update_position(self, filled_order):
        """Atualiza tracking de posição"""
        symbol = filled_order.symbol
        
        if symbol not in self.position_tracker:
            self.position_tracker[symbol] = {
                'quantity': 0,
                'avg_price': 0.0,
                'side': None
            }
        
        position = self.position_tracker[symbol]
        current_qty = position['quantity']
        current_avg = position['avg_price']
        
        if filled_order.side == OrderSide.BUY:
            # Compra - aumenta posição long ou reduz short
            new_qty = current_qty + filled_order.filled_qty
            
            if current_qty >= 0:  # Long ou flat - soma à posição
                # Calcular novo preço médio ponderado
                if new_qty > 0:
                    total_value = (current_qty * current_avg + 
                                 filled_order.filled_qty * filled_order.fill_price)
                    position['avg_price'] = total_value / new_qty
                else:
                    position['avg_price'] = 0.0
            else:  # Era short - reduzindo posição short
                if new_qty <= 0:  # Ainda short ou zerou
                    if new_qty < 0:
                        # Mantém preço médio da posição short original
                        position['avg_price'] = current_avg
                    else:
                        # Posição zerou - zerar preço médio
                        position['avg_price'] = 0.0
                else:  # Virou long
                    # Novo preço é o da ordem atual
                    position['avg_price'] = filled_order.fill_price
            
            position['quantity'] = new_qty
            
        else:  # OrderSide.SELL
            # Venda - reduz long ou aumenta short
            new_qty = current_qty - filled_order.filled_qty
            
            if current_qty <= 0:  # Short ou flat - aumenta posição short
                # Calcular novo preço médio ponderado para short
                if new_qty < 0:
                    total_value = (abs(current_qty) * current_avg + 
                                 filled_order.filled_qty * filled_order.fill_price)
                    position['avg_price'] = total_value / abs(new_qty)
                else:
                    position['avg_price'] = 0.0  # Posição zerou
            else:  # Era long - reduzindo posição long
                if new_qty >= 0:  # Ainda long ou zerou
                    # Mantém preço médio da posição long original se ainda long
                    # Zera preço médio se posição zerou
                    if new_qty > 0:
                        position['avg_price'] = current_avg
                    else:
                        position['avg_price'] = 0.0
                else:  # Virou short
                    # Novo preço é o da ordem atual
                    position['avg_price'] = filled_order.fill_price
            
            position['quantity'] = new_qty
        
        # Atualizar lado da posição
        if position['quantity'] > 0:
            position['side'] = 'long'
        elif position['quantity'] < 0:
            position['side'] = 'short'
        else:
            position['side'] = None
    
    def _create_protection_orders(self, filled_order):
        """Cria ordens de stop loss e take profit"""
        try:
            if not self.config.get('enable_stop_orders', True):
                return
                
            # TODO: Implementar criação automática de stops baseado no risk manager
            # Por enquanto apenas log
            self.logger.info(f"Criação de stops para ordem {filled_order.profit_id} - TODO")
            
        except Exception as e:
            self.logger.error(f"Erro criando ordens de proteção: {e}")
    
    def _get_current_price(self, symbol: str) -> float:
        """Obtém preço atual do market data"""
        try:
            # TODO: Integrar com market data processor
            # Por enquanto retorna preço mock
            self.logger.debug(f"Obtendo preço atual para {symbol} - usando mock")
            return 50000.0  # Preço mock para WDO
            
        except Exception as e:
            self.logger.error(f"Erro obtendo preço atual: {e}")
            return 0.0
    
    def get_execution_stats(self) -> Dict:
        """Retorna estatísticas de execução"""
        stats = self.execution_stats.copy()
        
        # Calcular métricas adicionais
        if stats['total_orders'] > 0:
            stats['success_rate'] = stats['successful_orders'] / stats['total_orders']
            stats['avg_slippage'] = stats['total_slippage'] / max(stats['successful_orders'], 1)
        else:
            stats['success_rate'] = 0.0
            stats['avg_slippage'] = 0.0
        
        return stats
    
    def get_active_orders(self) -> List:
        """Retorna ordens ativas"""
        return list(self.active_orders.values())
    
    def get_positions(self) -> Dict:
        """Retorna posições atuais"""
        return self.position_tracker.copy()
    
    def emergency_close_all(self):
        """Fecha todas as posições em emergência"""
        self.logger.warning("Fechamento de emergência iniciado")
        
        # Cancelar ordens pendentes
        self.order_manager.cancel_all_orders()
        
        # Fechar todas as posições
        for symbol, position in self.position_tracker.items():
            if position['quantity'] != 0:
                self.order_manager.close_position(symbol, at_market=True)