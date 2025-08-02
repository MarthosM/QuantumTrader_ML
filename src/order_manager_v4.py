# src/execution/order_manager_v4.py
"""
Gerenciador de Ordens compatível com ProfitDLL v4.0.0.30
Usa as novas estruturas e funções não-depreciadas
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import threading
import queue
from dataclasses import dataclass
import uuid
from ctypes import byref, c_wchar_p, c_int, c_double, POINTER

# Importar estruturas da API v4.0.0.30
from profit_dll_structures import (
    TConnectorSendOrder, TConnectorChangeOrder, TConnectorCancelOrder,
    TConnectorCancelOrders, TConnectorCancelAllOrders, TConnectorAccountIdentifier,
    TConnectorTradingAccountPosition, TConnectorOrderOut,
    OrderSide as DLLOrderSide, OrderType as DLLOrderType, 
    OrderStatus as DLLOrderStatus, NResult,
    create_account_identifier, create_send_order, create_cancel_order
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
        if not self.cl_ord_id:
            self.cl_ord_id = f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

class OrderExecutionManagerV4:
    """Gerenciador de execução de ordens com ProfitDLL v4.0.0.30"""
    
    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
        self.logger = logging.getLogger(__name__)
        
        # Filas de ordens
        self.order_queue = queue.Queue()
        self.pending_orders = {}  # cl_ord_id -> Order
        self.executed_orders = {}  # cl_ord_id -> Order
        
        # Informações da conta
        self.account_info = None
        self.account_identifier = None  # TConnectorAccountIdentifier
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
            'default_broker_id': 1  # Numérico para v4.0.0.30
        }
        
        self.logger.info("OrderExecutionManagerV4 criado - Compatível com ProfitDLL v4.0.0.30")
        
    def initialize(self):
        """Inicializa o gerenciador de ordens"""
        self.logger.info("Inicializando OrderExecutionManagerV4")
        
        # Obter informações da conta
        self._get_account_info()
        
        # Configurar callbacks da DLL
        self._setup_dll_callbacks()
        
        # Iniciar thread de processamento
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_orders)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("OrderExecutionManagerV4 inicializado")
    
    def _get_account_info(self):
        """Obtém informações da conta via DLL v4.0.0.30"""
        try:
            # GetAccount retorna informações via callback
            result = self.connection_manager.dll.GetAccount()
            if result == NResult.NL_OK:
                self.logger.info("Solicitação de informações de conta enviada")
                # Aguardar callback ser chamado
                threading.Event().wait(0.5)
            else:
                self.logger.error(f"Erro ao solicitar informações de conta: {result}")
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
            stop_price=signal.get('stop_price', 0.0),
            account_id=self.account_info.get('account_id', '') if self.account_info else '',
            broker_id=str(self.account_info.get('broker_id', self.config['default_broker_id'])) if self.account_info else str(self.config['default_broker_id']),
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
        """Executa uma ordem via ProfitDLL v4.0.0.30"""
        try:
            self.logger.info(f"Executando ordem v4.0.0.30: {order.symbol} {order.side.name} {order.quantity}")
            
            # Verificar se temos account_identifier
            if not self.account_identifier:
                self.logger.error("Account identifier não disponível")
                order.status = OrderStatus.ERROR
                return
            
            # Usar nova função SendOrder unificada
            profit_id = self._send_order_v4(order)
            
            # Atualizar ordem com ID retornado
            if profit_id > 0:
                order.profit_id = profit_id
                order.status = OrderStatus.SENT
                self.pending_orders[order.cl_ord_id] = order
                self.logger.info(f"Ordem enviada com sucesso. ProfitID: {profit_id}, ClientOrderID: {order.cl_ord_id}")
            else:
                order.status = OrderStatus.ERROR
                self.logger.error(f"Falha ao enviar ordem. Retorno: {profit_id}")
                
        except Exception as e:
            self.logger.error(f"Erro executando ordem: {e}")
            order.status = OrderStatus.ERROR
    
    def _send_order_v4(self, order: Order) -> int:
        """
        Envia ordem usando API v4.0.0.30 com SendOrder unificado
        
        Returns:
            int: LocalOrderID (>0 sucesso, <0 erro)
        """
        try:
            # Mapear tipo de ordem
            dll_order_type = DLLOrderType.MARKET
            if order.order_type == OrderType.LIMIT:
                dll_order_type = DLLOrderType.LIMIT
            elif order.order_type == OrderType.STOP:
                dll_order_type = DLLOrderType.STOP
            elif order.order_type == OrderType.STOP_LIMIT:
                dll_order_type = DLLOrderType.STOP_LIMIT
            
            # Criar estrutura de envio usando helper
            send_order = create_send_order(
                account=self.account_identifier,
                symbol=order.symbol,
                side=DLLOrderSide(order.side.value),
                order_type=dll_order_type,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price,
                password=order.password
            )
            
            # Adicionar client order ID
            send_order.ClientOrderID = order.cl_ord_id
            
            # Log de debug
            self.logger.debug(f"Enviando ordem: Symbol={order.symbol}, Side={order.side.value}, "
                            f"Type={dll_order_type}, Qty={order.quantity}, Price={order.price}")
            
            # Chamar SendOrder da DLL
            result = self.connection_manager.dll.SendOrder(byref(send_order))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro em _send_order_v4: {e}")
            return -1
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancela uma ordem usando SendCancelOrderV2"""
        try:
            order = self.pending_orders.get(order_id)
            if not order:
                self.logger.error(f"Ordem não encontrada: {order_id}")
                return False
            
            if not self.account_identifier:
                self.logger.error("Account identifier não disponível")
                return False
            
            # Criar estrutura de cancelamento
            cancel_order = create_cancel_order(
                account=self.account_identifier,
                client_order_id=order.cl_ord_id,
                password=order.password
            )
            
            # Cancelar via DLL v4.0.0.30
            result = self.connection_manager.dll.SendCancelOrderV2(byref(cancel_order))
            
            if result == NResult.NL_OK:
                self.logger.info(f"Ordem cancelada: {order_id}")
                return True
            else:
                self.logger.error(f"Falha ao cancelar ordem: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro cancelando ordem: {e}")
            return False
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        """Cancela todas as ordens usando SendCancelOrdersV2 ou SendCancelAllOrdersV2"""
        try:
            if not self.account_identifier:
                self.logger.error("Account identifier não disponível")
                return False
            
            if symbol:
                # Cancelar ordens de um símbolo específico
                cancel_orders = TConnectorCancelOrders()
                cancel_orders.AccountID = POINTER(TConnectorAccountIdentifier)(self.account_identifier)
                cancel_orders.Password = c_wchar_p("")  # Senha se necessária
                cancel_orders.Ticker = c_wchar_p(symbol)
                cancel_orders.Exchange = c_wchar_p(self._get_exchange_for_symbol(symbol))
                cancel_orders.Side = 0  # 0 = Ambos os lados
                
                result = self.connection_manager.dll.SendCancelOrdersV2(byref(cancel_orders))
            else:
                # Cancelar todas as ordens
                cancel_all = TConnectorCancelAllOrders()
                cancel_all.AccountID = POINTER(TConnectorAccountIdentifier)(self.account_identifier)
                cancel_all.Password = c_wchar_p("")  # Senha se necessária
                
                result = self.connection_manager.dll.SendCancelAllOrdersV2(byref(cancel_all))
            
            if result == NResult.NL_OK:
                self.logger.info(f"Ordens canceladas com sucesso")
                return True
            else:
                self.logger.error(f"Falha ao cancelar ordens: {result}")
                return False
            
        except Exception as e:
            self.logger.error(f"Erro cancelando ordens: {e}")
            return False
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Obtém posição atual usando GetPositionV2"""
        try:
            if not self.account_identifier:
                return None
            
            position = TConnectorTradingAccountPosition()
            position.AccountID = self.account_identifier
            position.AssetID.pwcTicker = c_wchar_p(symbol)
            position.AssetID.pwcBolsa = c_wchar_p(self._get_exchange_for_symbol(symbol))
            
            # GetPositionV2 preenche a estrutura
            result = self.connection_manager.dll.GetPositionV2(byref(position))
            
            if result == NResult.NL_OK:
                # Converter para dicionário
                return {
                    'symbol': symbol,
                    'quantity': position.Quantity,
                    'side': 'long' if position.Side == 1 else 'short',
                    'avg_price': position.AveragePrice,
                    'current_price': position.CurrentPrice,
                    'pnl': position.PnL,
                    'pnl_percent': position.PnLPercent,
                    'available_qty': position.AvailableQuantity
                }
            elif result == NResult.NL_NO_POSITION:
                self.logger.info(f"Sem posição em {symbol}")
                return {
                    'symbol': symbol,
                    'quantity': 0,
                    'side': None,
                    'avg_price': 0.0
                }
            else:
                self.logger.error(f"Erro obtendo posição: {result}")
                return None
            
        except Exception as e:
            self.logger.error(f"Erro obtendo posição: {e}")
            return None
    
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
                current_price = position.get('current_price', 0.0)
                if current_price > 0:
                    if position['side'] == 'long':
                        signal['price'] = current_price - 1  # Melhorar preço para venda
                    else:
                        signal['price'] = current_price + 1  # Melhorar preço para compra
            
            return self.send_order(signal)
            
        except Exception as e:
            self.logger.error(f"Erro fechando posição: {e}")
            return None
    
    def _on_order_update(self, order_data: Dict):
        """Callback para atualizações de ordem - compatível com v4.0.0.30"""
        try:
            # Na v4.0.0.30, o callback recebe um ponteiro para TConnectorOrderOut
            # order_data deve ser processado adequadamente
            
            cl_ord_id = order_data.get('client_order_id', '')
            profit_id = order_data.get('profit_id', 0)
            
            # Encontrar ordem por client order ID
            order = self.pending_orders.get(cl_ord_id)
            
            if order:
                # Mapear status da DLL para status interno
                dll_status = order_data.get('status', 0)
                
                if dll_status == DLLOrderStatus.FILLED:
                    order.status = OrderStatus.FILLED
                    order.fill_price = order_data.get('average_price', 0.0)
                    order.filled_qty = order_data.get('executed_quantity', 0)
                    
                    # Mover para executadas
                    self.executed_orders[cl_ord_id] = order
                    del self.pending_orders[cl_ord_id]
                    
                elif dll_status == DLLOrderStatus.PARTIALLY_FILLED:
                    order.status = OrderStatus.PARTIAL
                    order.filled_qty = order_data.get('executed_quantity', 0)
                    order.remaining_qty = order_data.get('remaining_quantity', 0)
                    
                elif dll_status == DLLOrderStatus.CANCELLED:
                    order.status = OrderStatus.CANCELLED
                    del self.pending_orders[cl_ord_id]
                    
                elif dll_status == DLLOrderStatus.REJECTED:
                    order.status = OrderStatus.REJECTED
                    del self.pending_orders[cl_ord_id]
                
                # Atualizar profit_id se ainda não temos
                if profit_id > 0 and order.profit_id == 0:
                    order.profit_id = profit_id
                
                # Notificar callbacks
                self._notify_order_update(order)
                
        except Exception as e:
            self.logger.error(f"Erro processando atualização de ordem: {e}")
    
    def _on_account_info(self, account_data: Dict):
        """Callback para informações da conta - compatível com v4.0.0.30"""
        try:
            self.account_info = account_data
            
            # Criar account identifier para v4.0.0.30
            if account_data.get('broker_id') and account_data.get('account_id'):
                self.account_identifier = create_account_identifier(
                    broker_id=int(account_data.get('broker_id', 0)),
                    account_id=account_data.get('account_id', ''),
                    sub_account_id=account_data.get('sub_account_id', '')
                )
                self.logger.info(f"Account identifier criado: Broker={self.account_identifier.BrokerID}, "
                               f"Account={self.account_identifier.AccountID}")
            
            self.logger.info(f"Informações da conta recebidas: {account_data.get('account_id')}")
        except Exception as e:
            self.logger.error(f"Erro processando informações da conta: {e}")
    
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
        self.logger.info("Finalizando OrderExecutionManagerV4")
        
        # Cancelar ordens pendentes
        self.cancel_all_orders()
        
        # Parar thread
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        self.logger.info("OrderExecutionManagerV4 finalizado")