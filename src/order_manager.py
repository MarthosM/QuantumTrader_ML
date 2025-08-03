# src/execution/order_manager.py
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import threading
import queue
from dataclasses import dataclass
from src.profit_dll_structures import (
    OrderSide as ProfitOrderSide, OrderType as ProfitOrderType, NResult,
    create_account_identifier, create_send_order
)


class OrderType(Enum):
    """Tipos de ordem suportados"""
    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Lado da ordem"""
    BUY = 1
    SELL = 2

class OrderStatus(Enum):
    """Status da ordem"""
    PENDING = "pending"
    SENT = "sent"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"

@dataclass
class Order:
    """Estrutura de dados para ordem"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float = 0.0
    stop_price: float = 0.0
    account_id: str = ""
    broker_id: str = ""
    password: str = ""
    cl_ord_id: str = ""
    profit_id: int = 0
    status: OrderStatus = OrderStatus.PENDING
    timestamp: Optional[datetime] = None
    fill_price: float = 0.0
    filled_qty: int = 0
    remaining_qty: int = 0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.remaining_qty == 0:
            self.remaining_qty = self.quantity

class OrderExecutionManager:
    """Gerenciador de execução de ordens com ProfitDLL"""
    
    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
        self.logger = logging.getLogger(__name__)
        
        # Filas de ordens
        self.order_queue = queue.Queue()
        self.pending_orders = {}  # cl_ord_id -> Order
        self.executed_orders = {}  # cl_ord_id -> Order
        
        # Informações da conta
        self.account_info = None
        self.position_tracker = {}  # symbol -> position
        
        # Thread de processamento
        self.processing_thread = None
        self.running = False
        
        # Callbacks
        self.order_callbacks = []
        
        # Configurações
        self.config = {
            'max_retry_attempts': 3,
            'retry_delay': 1.0,
            'order_timeout': 30.0,
            'use_market_orders': False,
            'default_broker_id': '1'  # Ajustar conforme necessário
        }
        
    def initialize(self):
        """Inicializa o gerenciador de ordens"""
        self.logger.info("Inicializando OrderExecutionManager")
        
        # Obter informações da conta
        self._get_account_info()
        
        # Configurar callbacks da DLL
        self._setup_dll_callbacks()
        
        # Iniciar thread de processamento
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_orders)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("OrderExecutionManager inicializado")
    
    def _get_account_info(self):
        """Obtém informações da conta via DLL"""
        try:
            # GetAccount retorna informações via callback
            self.connection_manager.dll.GetAccount()
            # Aguardar callback ser chamado
            threading.Event().wait(0.5)
        except Exception as e:
            self.logger.error(f"Erro obtendo informações da conta: {e}")
    
    def _setup_dll_callbacks(self):
        """Configura callbacks para receber atualizações de ordens"""
        # O connection_manager já deve ter configurado os callbacks
        # Aqui apenas registramos para receber notificações
        self.connection_manager.register_order_callback(self._on_order_update)
        self.connection_manager.register_account_callback(self._on_account_info)
    
    def send_order(self, signal: Dict) -> Optional[Order]:
        """
        Envia uma ordem baseada em um sinal de trading
        
        Args:
            signal: Dicionário com informações do sinal
                {
                    'symbol': str,
                    'action': 'buy' | 'sell' | 'close',
                    'quantity': int,
                    'price': float (opcional para market),
                    'stop_loss': float (opcional),
                    'take_profit': float (opcional),
                    'order_type': 'limit' | 'market'
                }
        
        Returns:
            Order object ou None se falhar
        """
        try:
            # Validar sinal
            if not self._validate_signal(signal):
                return None
            
            # Criar ordem
            order = self._create_order_from_signal(signal)
            
            # Adicionar à fila
            self.order_queue.put(order)
            
            self.logger.info(f"Ordem adicionada à fila: {order.symbol} {order.side.name} {order.quantity}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Erro ao enviar ordem: {e}")
            return None
    
    def _validate_signal(self, signal: Dict) -> bool:
        """Valida se o sinal contém informações necessárias"""
        required_fields = ['symbol', 'action', 'quantity']
        
        for field in required_fields:
            if field not in signal:
                self.logger.error(f"Campo obrigatório ausente no sinal: {field}")
                return False
        
        # Validar valores
        if signal['quantity'] <= 0:
            self.logger.error("Quantidade deve ser maior que zero")
            return False
        
        if signal['action'] not in ['buy', 'sell', 'close']:
            self.logger.error(f"Ação inválida: {signal['action']}")
            return False
        
        return True
    
    def _create_order_from_signal(self, signal: Dict) -> Order:
        """Cria objeto Order a partir do sinal"""
        # Determinar lado da ordem
        if signal['action'] == 'buy':
            side = OrderSide.BUY
        elif signal['action'] == 'sell':
            side = OrderSide.SELL
        else:  # close
            # Verificar posição atual para determinar lado
            position = self.get_position(signal['symbol'])
            if position and position['side'] == 'long':
                side = OrderSide.SELL
            else:
                side = OrderSide.BUY
        
        # Tipo de ordem
        order_type = OrderType.MARKET if signal.get('order_type') == 'market' else OrderType.LIMIT
        
        # Criar ordem
        order = Order(
            symbol=signal['symbol'],
            side=side,
            order_type=order_type,
            quantity=signal['quantity'],
            price=signal.get('price', 0.0),
            account_id=self.account_info.get('account_id', '') if self.account_info else '',
            broker_id=self.account_info.get('broker_id', self.config['default_broker_id']) if self.account_info else self.config['default_broker_id'],
            password=signal.get('password', '')  # Senha de roteamento se necessária
        )
        
        return order
    
    def _process_orders(self):
        """Thread principal de processamento de ordens"""
        while self.running:
            try:
                # Pegar ordem da fila (timeout de 1 segundo)
                order = self.order_queue.get(timeout=1.0)
                
                # Executar ordem
                self._execute_order(order)
                
            except queue.Empty:
                # Nada para processar
                continue
            except Exception as e:
                self.logger.error(f"Erro processando ordem: {e}")
    
    def _execute_order(self, order: Order):
        """Executa uma ordem via ProfitDLL"""
        try:
            self.logger.info(f"Executando ordem: {order.symbol} {order.side.name} {order.quantity}")
            
            # Preparar parâmetros
            symbol = order.symbol
            exchange = self._get_exchange_for_symbol(symbol)
            
            # Chamar método apropriado da DLL
            if order.order_type == OrderType.MARKET:
                profit_id = self._send_market_order(order, exchange)
            elif order.order_type == OrderType.LIMIT:
                profit_id = self._send_limit_order(order, exchange)
            elif order.order_type == OrderType.STOP:
                profit_id = self._send_stop_order(order, exchange)
            else:
                raise ValueError(f"Tipo de ordem não suportado: {order.order_type}")
            
            # Atualizar ordem com ID retornado
            if profit_id > 0:
                order.profit_id = profit_id
                order.status = OrderStatus.SENT
                self.pending_orders[str(profit_id)] = order
                self.logger.info(f"Ordem enviada com sucesso. ProfitID: {profit_id}")
            else:
                order.status = OrderStatus.ERROR
                self.logger.error(f"Falha ao enviar ordem. Retorno: {profit_id}")
                
        except Exception as e:
            self.logger.error(f"Erro executando ordem: {e}")
            order.status = OrderStatus.ERROR
    
    def _send_market_order(self, order: Order, exchange: str) -> int:
        """Envia ordem a mercado"""
        dll = self.connection_manager.dll
        
        if order.side == OrderSide.BUY:
            return dll.SendOrder(
                order.account_id,
                order.broker_id,
                order.password,
                order.symbol,
                exchange,
                order.quantity
            )
        else:
            return dll.SendOrder(
                order.account_id,
                order.broker_id,
                order.password,
                order.symbol,
                exchange,
                order.quantity
            )
    
    def _send_limit_order(self, order: Order, exchange: str) -> int:
        """Envia ordem limite"""
        dll = self.connection_manager.dll
        
        if order.side == OrderSide.BUY:
            return dll.SendOrder(
                order.account_id,
                order.broker_id,
                order.password,
                order.symbol,
                exchange,
                order.price,
                order.quantity
            )
        else:
            return dll.SendOrder(
                order.account_id,
                order.broker_id,
                order.password,
                order.symbol,
                exchange,
                order.price,
                order.quantity
            )
    
    def _send_stop_order(self, order: Order, exchange: str) -> int:
        """Envia ordem stop"""
        dll = self.connection_manager.dll
        
        if order.side == OrderSide.BUY:
            return dll.SendStopBuyOrder(
                order.account_id,
                order.broker_id,
                order.password,
                order.symbol,
                exchange,
                order.price,
                order.stop_price,
                order.quantity
            )
        else:
            return dll.SendStopSellOrder(
                order.account_id,
                order.broker_id,
                order.password,
                order.symbol,
                exchange,
                order.price,
                order.stop_price,
                order.quantity
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancela uma ordem"""
        try:
            order = self.pending_orders.get(order_id)
            if not order:
                self.logger.error(f"Ordem não encontrada: {order_id}")
                return False
            
            # Cancelar via DLL
            result = self.connection_manager.dll.SendCancelOrderV2(
                order.account_id,
                order.broker_id,
                order.cl_ord_id,
                order.password
            )
            
            if result == 0:  # NL_OK
                self.logger.info(f"Ordem cancelada: {order_id}")
                return True
            else:
                self.logger.error(f"Falha ao cancelar ordem: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro cancelando ordem: {e}")
            return False
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        """Cancela todas as ordens ou todas de um símbolo"""
        try:
            if not self.account_info:
                return False
            
            if symbol:
                # Cancelar ordens de um símbolo
                exchange = self._get_exchange_for_symbol(symbol)
                result = self.connection_manager.dll.SendCancelOrders(
                    self.account_info['account_id'],
                    self.account_info['broker_id'],
                    self.account_info.get('password', ''),
                    symbol,
                    exchange
                )
            else:
                # Cancelar todas as ordens
                result = self.connection_manager.dll.SendCancelAllOrders(
                    self.account_info['account_id'],
                    self.account_info['broker_id'],
                    self.account_info.get('password', '')
                )
            
            return result == 0  # NL_OK
            
        except Exception as e:
            self.logger.error(f"Erro cancelando ordens: {e}")
            return False
    
    def close_position(self, symbol: str, at_market: bool = False) -> Optional[Order]:
        """Fecha posição de um símbolo"""
        try:
            position = self.get_position(symbol)
            if not position or position['quantity'] == 0:
                self.logger.info(f"Sem posição em {symbol}")
                return None
            
            # Criar ordem de fechamento
            signal = {
                'symbol': symbol,
                'action': 'close',
                'quantity': abs(position['quantity']),
                'order_type': 'market' if at_market else 'limit'
            }
            
            if not at_market:
                # Usar preço atual para ordem limite
                current_price = self._get_current_price(symbol)
                if position['side'] == 'long':
                    signal['price'] = current_price - 1  # Melhorar preço para venda
                else:
                    signal['price'] = current_price + 1  # Melhorar preço para compra
            
            return self.send_order(signal)
            
        except Exception as e:
            self.logger.error(f"Erro fechando posição: {e}")
            return None
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Obtém posição atual de um símbolo"""
        try:
            if not self.account_info:
                return None
            
            exchange = self._get_exchange_for_symbol(symbol)
            
            # GetPosition retorna um ponteiro para estrutura
            position_ptr = self.connection_manager.dll.GetPositionV2(
                self.account_info['account_id'],
                self.account_info['broker_id'],
                symbol,
                exchange
            )
            
            if position_ptr:
                # Parsear estrutura de posição
                # TODO: Implementar parsing da estrutura conforme documentação
                return self.position_tracker.get(symbol, {
                    'symbol': symbol,
                    'quantity': 0,
                    'side': None,
                    'avg_price': 0.0
                })
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro obtendo posição: {e}")
            return None
    
    def _on_order_update(self, order_data: Dict):
        """Callback para atualizações de ordem"""
        try:
            profit_id = str(order_data.get('profit_id'))
            cl_ord_id = order_data.get('cl_ord_id')
            
            # Encontrar ordem
            order = self.pending_orders.get(profit_id)
            if not order and cl_ord_id:
                # Procurar por cl_ord_id
                for o in self.pending_orders.values():
                    if o.cl_ord_id == cl_ord_id:
                        order = o
                        break
            
            if order:
                # Atualizar status
                status = order_data.get('status', '').lower()
                if 'filled' in status:
                    order.status = OrderStatus.FILLED
                    order.fill_price = order_data.get('avg_price', 0.0)
                    order.filled_qty = order_data.get('traded_qty', 0)
                    
                    # Mover para executadas
                    self.executed_orders[profit_id] = order
                    del self.pending_orders[profit_id]
                    
                elif 'partial' in status:
                    order.status = OrderStatus.PARTIAL
                    order.filled_qty = order_data.get('traded_qty', 0)
                    order.remaining_qty = order_data.get('leaves_qty', 0)
                    
                elif 'cancelled' in status or 'canceled' in status:
                    order.status = OrderStatus.CANCELLED
                    del self.pending_orders[profit_id]
                    
                elif 'rejected' in status:
                    order.status = OrderStatus.REJECTED
                    del self.pending_orders[profit_id]
                
                # Notificar callbacks
                self._notify_order_update(order)
                
        except Exception as e:
            self.logger.error(f"Erro processando atualização de ordem: {e}")
    
    def _on_account_info(self, account_data: Dict):
        """Callback para informações da conta"""
        self.account_info = account_data
        self.logger.info(f"Informações da conta recebidas: {account_data.get('account_id')}")
    
    def _notify_order_update(self, order: Order):
        """Notifica callbacks registrados sobre atualização de ordem"""
        for callback in self.order_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"Erro em callback de ordem: {e}")
    
    def register_order_callback(self, callback):
        """Registra callback para atualizações de ordem"""
        self.order_callbacks.append(callback)
    
    def _get_exchange_for_symbol(self, symbol: str) -> str:
        """Determina exchange baseado no símbolo"""
        # Lógica simplificada - ajustar conforme necessário
        if 'FUT' in symbol or symbol.startswith('WDO') or symbol.startswith('WIN'):
            return 'F'  # BMF
        else:
            return 'B'  # Bovespa
    
    def _get_current_price(self, symbol: str) -> float:
        """Obtém preço atual do símbolo"""
        # TODO: Implementar obtenção de preço atual do market data
        return 0.0
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Obtém status de uma ordem"""
        order = self.pending_orders.get(order_id) or self.executed_orders.get(order_id)
        return order.status if order else None
    
    def get_pending_orders(self) -> List[Order]:
        """Retorna lista de ordens pendentes"""
        return list(self.pending_orders.values())
    
    def get_executed_orders(self) -> List[Order]:
        """Retorna lista de ordens executadas"""
        return list(self.executed_orders.values())
    
    def shutdown(self):
        """Finaliza o gerenciador de ordens"""
        self.logger.info("Finalizando OrderExecutionManager")
        
        # Cancelar ordens pendentes
        self.cancel_all_orders()
        
        # Parar thread
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        self.logger.info("OrderExecutionManager finalizado")