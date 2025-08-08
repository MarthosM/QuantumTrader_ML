"""
OrderManager - Sistema completo de gerenciamento de ordens
Integra com ProfitDLL para execução real de ordens
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
from enum import Enum
import threading
import logging
import time
import json
import uuid
from dataclasses import dataclass, field

class OrderState(Enum):
    """Estados possíveis de uma ordem"""
    PENDING = "PENDING"           # Ordem criada, não enviada
    SUBMITTED = "SUBMITTED"       # Enviada para o broker
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # Parcialmente executada
    FILLED = "FILLED"            # Totalmente executada
    CANCELLED = "CANCELLED"       # Cancelada
    REJECTED = "REJECTED"        # Rejeitada pelo broker
    EXPIRED = "EXPIRED"          # Expirada
    FAILED = "FAILED"            # Falha no envio

class OrderType(Enum):
    """Tipos de ordem"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    """Lado da ordem"""
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Order:
    """Estrutura de uma ordem"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    state: OrderState = OrderState.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_quantity: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    signal_info: Optional[Dict] = None
    error_message: Optional[str] = None
    broker_order_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Converte ordem para dicionário"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'state': self.state.value,
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'commission': self.commission,
            'broker_order_id': self.broker_order_id
        }

class OrderManager:
    """
    Gerenciador completo de ordens
    
    Features:
    - Criação e validação de ordens
    - Envio via ProfitDLL
    - Rastreamento de estado
    - Retry automático
    - Callbacks para eventos
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configurações
        self.max_retry_attempts = config.get('max_retry_attempts', 3)
        self.retry_delay_ms = config.get('retry_delay_ms', 1000)
        self.order_timeout_seconds = config.get('order_timeout_seconds', 30)
        self.commission_per_contract = config.get('commission_per_contract', 5.0)
        
        # Ordens ativas
        self.orders = {}  # order_id -> Order
        self.pending_orders = deque()  # Fila de ordens pendentes
        
        # Mapeamento broker
        self.broker_order_map = {}  # broker_order_id -> order_id
        
        # Callbacks
        self.callbacks = {
            'on_submitted': [],
            'on_filled': [],
            'on_cancelled': [],
            'on_rejected': [],
            'on_error': []
        }
        
        # Estatísticas
        self.stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'failed_orders': 0,
            'total_commission': 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
        self.processing_thread = None
        self.is_running = False
        
        # Conexão com broker (será injetada)
        self.broker_connection = None
        
    def start(self):
        """Inicia o OrderManager"""
        self.logger.info("Iniciando OrderManager")
        self.is_running = True
        
        # Thread de processamento de ordens
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name="OrderProcessing"
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("[OK] OrderManager iniciado")
        
    def stop(self):
        """Para o OrderManager"""
        self.logger.info("Parando OrderManager...")
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
            
        self._log_statistics()
        
    def set_broker_connection(self, connection):
        """Define conexão com o broker"""
        self.broker_connection = connection
        self.logger.info("Conexão com broker configurada")
        
    def create_order(self, symbol: str, side: str, quantity: int,
                    order_type: str = "MARKET", price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    signal_info: Optional[Dict] = None) -> Optional[Order]:
        """
        Cria uma nova ordem
        
        Args:
            symbol: Símbolo do ativo
            side: 'BUY' ou 'SELL'
            quantity: Quantidade
            order_type: Tipo da ordem
            price: Preço limite (se aplicável)
            stop_price: Preço stop (se aplicável)
            signal_info: Informações do sinal que gerou a ordem
            
        Returns:
            Order criada ou None se inválida
        """
        try:
            # Validar parâmetros
            if not self._validate_order_params(symbol, side, quantity, order_type, price):
                return None
            
            # Criar ordem
            order = Order(
                order_id=self._generate_order_id(),
                symbol=symbol,
                side=OrderSide(side),
                order_type=OrderType(order_type),
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                signal_info=signal_info
            )
            
            with self.lock:
                # Adicionar ao registro
                self.orders[order.order_id] = order
                self.pending_orders.append(order.order_id)
                self.stats['total_orders'] += 1
                
            self.logger.info(f"Ordem criada: {order.order_id} - {side} {quantity} {symbol}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Erro ao criar ordem: {e}")
            return None
            
    def submit_order(self, order_id: str) -> bool:
        """
        Submete uma ordem para execução
        
        Args:
            order_id: ID da ordem
            
        Returns:
            True se submetida com sucesso
        """
        with self.lock:
            if order_id not in self.orders:
                self.logger.error(f"Ordem não encontrada: {order_id}")
                return False
                
            order = self.orders[order_id]
            
            if order.state != OrderState.PENDING:
                self.logger.error(f"Ordem {order_id} não está pendente: {order.state}")
                return False
                
        # Processar imediatamente
        return self._process_order(order)
        
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancela uma ordem
        
        Args:
            order_id: ID da ordem
            
        Returns:
            True se cancelamento enviado
        """
        with self.lock:
            if order_id not in self.orders:
                self.logger.error(f"Ordem não encontrada: {order_id}")
                return False
                
            order = self.orders[order_id]
            
            # Verificar se pode ser cancelada
            if order.state not in [OrderState.PENDING, OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED]:
                self.logger.error(f"Ordem {order_id} não pode ser cancelada: {order.state}")
                return False
                
        try:
            # Enviar cancelamento ao broker
            if self.broker_connection and order.broker_order_id:
                success = self.broker_connection.cancel_order(order.broker_order_id)
                
                if success:
                    self.logger.info(f"Cancelamento enviado: {order_id}")
                    return True
                else:
                    self.logger.error(f"Falha ao cancelar ordem {order_id}")
                    return False
            else:
                # Se não foi enviada ainda, cancelar localmente
                self._update_order_state(order, OrderState.CANCELLED)
                return True
                
        except Exception as e:
            self.logger.error(f"Erro ao cancelar ordem: {e}")
            return False
            
    def get_order(self, order_id: str) -> Optional[Order]:
        """Retorna uma ordem específica"""
        with self.lock:
            return self.orders.get(order_id)
            
    def get_open_orders(self) -> List[Order]:
        """Retorna todas as ordens abertas"""
        with self.lock:
            return [
                order for order in self.orders.values()
                if order.state in [OrderState.PENDING, OrderState.SUBMITTED, 
                                 OrderState.PARTIALLY_FILLED]
            ]
            
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Retorna ordens de um símbolo específico"""
        with self.lock:
            return [
                order for order in self.orders.values()
                if order.symbol == symbol
            ]
            
    def register_callback(self, event: str, callback: Callable):
        """
        Registra callback para eventos
        
        Args:
            event: Nome do evento (on_submitted, on_filled, etc.)
            callback: Função a ser chamada
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            self.logger.info(f"Callback registrado para {event}")
        else:
            self.logger.error(f"Evento inválido: {event}")
            
    def _processing_loop(self):
        """Loop de processamento de ordens pendentes"""
        self.logger.info("Loop de processamento iniciado")
        
        while self.is_running:
            try:
                # Processar ordens pendentes
                with self.lock:
                    if self.pending_orders:
                        order_id = self.pending_orders.popleft()
                        order = self.orders.get(order_id)
                        
                        if order and order.state == OrderState.PENDING:
                            # Processar fora do lock
                            self._process_order(order)
                
                # Verificar timeouts
                self._check_order_timeouts()
                
                # Pequena pausa
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Erro no loop de processamento: {e}")
                time.sleep(1)
                
    def _process_order(self, order: Order) -> bool:
        """Processa uma ordem individual"""
        
        retry_count = 0
        
        while retry_count < self.max_retry_attempts:
            try:
                # Validar ordem antes de enviar
                if not self._validate_order(order):
                    self._update_order_state(order, OrderState.REJECTED, 
                                          "Ordem inválida")
                    return False
                
                # Enviar ao broker
                if self.broker_connection:
                    # Preparar dados para ProfitDLL
                    broker_order = self._prepare_broker_order(order)
                    
                    # Enviar ordem
                    broker_order_id = self.broker_connection.send_order(broker_order)
                    
                    if broker_order_id:
                        # Sucesso no envio
                        order.broker_order_id = broker_order_id
                        order.submitted_at = datetime.now()
                        
                        with self.lock:
                            self.broker_order_map[broker_order_id] = order.order_id
                            
                        self._update_order_state(order, OrderState.SUBMITTED)
                        
                        self.logger.info(f"Ordem enviada: {order.order_id} -> "
                                       f"Broker ID: {broker_order_id}")
                        return True
                    else:
                        raise Exception("Broker retornou ID nulo")
                        
                else:
                    # Modo simulação (sem broker)
                    self._simulate_order_execution(order)
                    return True
                    
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Erro ao enviar ordem (tentativa {retry_count}): {e}")
                
                if retry_count < self.max_retry_attempts:
                    time.sleep(self.retry_delay_ms / 1000)
                else:
                    self._update_order_state(order, OrderState.FAILED, str(e))
                    return False
                    
        return False
        
    def _prepare_broker_order(self, order: Order) -> Dict:
        """Prepara ordem no formato do broker"""
        
        broker_order = {
            'symbol': order.symbol,
            'side': 0 if order.side == OrderSide.BUY else 1,  # ProfitDLL format
            'quantity': order.quantity,
            'order_type': self._convert_order_type(order.order_type)
        }
        
        # Adicionar preços se necessário
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            broker_order['price'] = order.price
            
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            broker_order['stop_price'] = order.stop_price
            
        return broker_order
        
    def _convert_order_type(self, order_type: OrderType) -> int:
        """Converte tipo de ordem para formato do broker"""
        
        # Mapeamento para ProfitDLL
        type_map = {
            OrderType.MARKET: 0,
            OrderType.LIMIT: 1,
            OrderType.STOP: 2,
            OrderType.STOP_LIMIT: 3
        }
        
        return type_map.get(order_type, 0)
        
    def _simulate_order_execution(self, order: Order):
        """Simula execução de ordem (para testes)"""
        
        self.logger.warning(f"SIMULAÇÃO: Executando ordem {order.order_id}")
        
        # Simular envio
        order.submitted_at = datetime.now()
        order.broker_order_id = f"SIM_{order.order_id}"
        self._update_order_state(order, OrderState.SUBMITTED)
        
        # Simular execução após pequeno delay
        def simulate_fill():
            time.sleep(0.5)
            
            # Simular preenchimento
            order.filled_at = datetime.now()
            order.filled_quantity = order.quantity
            order.filled_price = order.price or 5000.0  # Preço simulado
            order.commission = order.quantity * self.commission_per_contract
            
            self._update_order_state(order, OrderState.FILLED)
            
        threading.Thread(target=simulate_fill, daemon=True).start()
        
    def on_order_update(self, broker_order_id: str, update_data: Dict):
        """
        Callback para atualizações de ordem do broker
        
        Args:
            broker_order_id: ID da ordem no broker
            update_data: Dados da atualização
        """
        with self.lock:
            # Encontrar ordem local
            order_id = self.broker_order_map.get(broker_order_id)
            if not order_id:
                self.logger.warning(f"Ordem não encontrada para broker ID: {broker_order_id}")
                return
                
            order = self.orders.get(order_id)
            if not order:
                return
                
        # Processar atualização
        status = update_data.get('status', '').upper()
        
        if status == 'FILLED':
            order.filled_at = datetime.now()
            order.filled_quantity = update_data.get('filled_quantity', order.quantity)
            order.filled_price = update_data.get('filled_price', 0)
            order.commission = update_data.get('commission', 
                                             order.filled_quantity * self.commission_per_contract)
            
            self._update_order_state(order, OrderState.FILLED)
            
        elif status == 'PARTIALLY_FILLED':
            order.filled_quantity = update_data.get('filled_quantity', 0)
            self._update_order_state(order, OrderState.PARTIALLY_FILLED)
            
        elif status == 'CANCELLED':
            self._update_order_state(order, OrderState.CANCELLED)
            
        elif status == 'REJECTED':
            reason = update_data.get('reason', 'Desconhecido')
            self._update_order_state(order, OrderState.REJECTED, reason)
            
    def _update_order_state(self, order: Order, new_state: OrderState, 
                          error_message: Optional[str] = None):
        """Atualiza estado da ordem e dispara callbacks"""
        
        old_state = order.state
        order.state = new_state
        
        if error_message:
            order.error_message = error_message
            
        # Atualizar estatísticas
        with self.lock:
            if new_state == OrderState.FILLED:
                self.stats['filled_orders'] += 1
                self.stats['total_commission'] += order.commission
            elif new_state == OrderState.CANCELLED:
                self.stats['cancelled_orders'] += 1
            elif new_state == OrderState.REJECTED:
                self.stats['rejected_orders'] += 1
            elif new_state == OrderState.FAILED:
                self.stats['failed_orders'] += 1
                
        self.logger.info(f"Ordem {order.order_id}: {old_state.value} -> {new_state.value}")
        
        # Disparar callbacks
        self._trigger_callbacks(order, new_state)
        
    def _trigger_callbacks(self, order: Order, state: OrderState):
        """Dispara callbacks para mudança de estado"""
        
        event_map = {
            OrderState.SUBMITTED: 'on_submitted',
            OrderState.FILLED: 'on_filled',
            OrderState.CANCELLED: 'on_cancelled',
            OrderState.REJECTED: 'on_rejected',
            OrderState.FAILED: 'on_error'
        }
        
        event = event_map.get(state)
        if event and event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(order)
                except Exception as e:
                    self.logger.error(f"Erro em callback {event}: {e}")
                    
    def _check_order_timeouts(self):
        """Verifica timeouts de ordens"""
        
        current_time = datetime.now()
        
        with self.lock:
            for order in self.orders.values():
                if order.state == OrderState.SUBMITTED:
                    # Verificar timeout
                    if order.submitted_at:
                        elapsed = (current_time - order.submitted_at).total_seconds()
                        
                        if elapsed > self.order_timeout_seconds:
                            self.logger.warning(f"Timeout na ordem {order.order_id}")
                            self._update_order_state(order, OrderState.EXPIRED, 
                                                  "Timeout na execução")
                            
    def _validate_order_params(self, symbol: str, side: str, quantity: int,
                             order_type: str, price: Optional[float]) -> bool:
        """Valida parâmetros da ordem"""
        
        # Validar símbolo
        if not symbol:
            self.logger.error("Símbolo vazio")
            return False
            
        # Validar lado
        if side not in ['BUY', 'SELL']:
            self.logger.error(f"Lado inválido: {side}")
            return False
            
        # Validar quantidade
        if quantity <= 0:
            self.logger.error(f"Quantidade inválida: {quantity}")
            return False
            
        # Validar tipo
        if order_type not in ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']:
            self.logger.error(f"Tipo inválido: {order_type}")
            return False
            
        # Validar preço para ordens limit
        if order_type in ['LIMIT', 'STOP_LIMIT'] and not price:
            self.logger.error("Preço necessário para ordem limit")
            return False
            
        return True
        
    def _validate_order(self, order: Order) -> bool:
        """Valida ordem antes de enviar"""
        
        # Aqui podemos adicionar validações adicionais
        # Por exemplo: verificar margem, limites de posição, etc.
        
        return True
        
    def _generate_order_id(self) -> str:
        """Gera ID único para ordem"""
        return f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
    def get_statistics(self) -> Dict:
        """Retorna estatísticas do OrderManager"""
        
        with self.lock:
            stats = self.stats.copy()
            
            # Adicionar métricas calculadas
            if stats['total_orders'] > 0:
                stats['fill_rate'] = stats['filled_orders'] / stats['total_orders']
                stats['rejection_rate'] = stats['rejected_orders'] / stats['total_orders']
                stats['failure_rate'] = stats['failed_orders'] / stats['total_orders']
            else:
                stats['fill_rate'] = 0
                stats['rejection_rate'] = 0
                stats['failure_rate'] = 0
                
            # Ordens por estado
            state_counts = {}
            for order in self.orders.values():
                state = order.state.value
                state_counts[state] = state_counts.get(state, 0) + 1
                
            stats['orders_by_state'] = state_counts
            
            return stats
            
    def _log_statistics(self):
        """Loga estatísticas finais"""
        
        stats = self.get_statistics()
        
        self.logger.info("="*60)
        self.logger.info("Estatísticas do OrderManager")
        self.logger.info("="*60)
        self.logger.info(f"Total de ordens: {stats['total_orders']}")
        self.logger.info(f"Ordens executadas: {stats['filled_orders']}")
        self.logger.info(f"Taxa de execução: {stats['fill_rate']:.2%}")
        self.logger.info(f"Taxa de rejeição: {stats['rejection_rate']:.2%}")
        self.logger.info(f"Comissão total: ${stats['total_commission']:.2f}")
        
        if stats['orders_by_state']:
            self.logger.info("\nOrdens por estado:")
            for state, count in stats['orders_by_state'].items():
                self.logger.info(f"  {state}: {count}")
                
    def export_orders(self, filepath: str):
        """Exporta histórico de ordens para arquivo"""
        
        with self.lock:
            orders_data = [order.to_dict() for order in self.orders.values()]
            
        df = pd.DataFrame(orders_data)
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Ordens exportadas para: {filepath}")