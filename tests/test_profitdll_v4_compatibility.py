"""
Testes de compatibilidade com ProfitDLL v4.0.0.30
Valida as novas estruturas e implementações
"""

import pytest
import sys
import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from ctypes import c_int, c_wchar_p, POINTER

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Importar módulos para teste
from profit_dll_structures import (
    TConnectorAccountIdentifier, TConnectorSendOrder, TConnectorCancelOrder,
    OrderSide, OrderType, OrderStatus, NResult,
    create_account_identifier, create_send_order, create_cancel_order,
    PROFITDLL_VERSION
)
from order_manager_v4 import OrderExecutionManagerV4, Order
from connection_manager_v4 import ConnectionManagerV4


class TestProfitDLLStructures:
    """Testa as estruturas de dados da v4.0.0.30"""
    
    def test_version_constant(self):
        """Verifica se a versão está correta"""
        assert PROFITDLL_VERSION == "4.0.0.30"
    
    def test_create_account_identifier(self):
        """Testa criação de identificador de conta"""
        account = create_account_identifier(
            broker_id=1,
            account_id="12345",
            sub_account_id="001"
        )
        
        assert account.BrokerID == 1
        assert account.AccountID == "12345"
        assert account.SubAccountID == "001"
    
    def test_create_account_identifier_without_subaccount(self):
        """Testa criação sem sub-conta"""
        account = create_account_identifier(
            broker_id=2,
            account_id="54321"
        )
        
        assert account.BrokerID == 2
        assert account.AccountID == "54321"
        assert account.SubAccountID == ""
    
    def test_create_send_order_market(self):
        """Testa criação de ordem de mercado"""
        account = create_account_identifier(1, "12345")
        
        order = create_send_order(
            account=account,
            symbol="WDOQ25",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10,
            password="test123"
        )
        
        assert order.Ticker == "WDOQ25"
        assert order.Exchange == "F"  # BMF para WDO
        assert order.Side == OrderSide.BUY
        assert order.OrderType == OrderType.MARKET
        assert order.Quantity == 10
        assert order.Price == 0.0
        assert order.StopPrice == 0.0
        assert order.Password == "test123"
        assert order.ValidityType == 0  # Day order
    
    def test_create_send_order_limit(self):
        """Testa criação de ordem limite"""
        account = create_account_identifier(1, "12345")
        
        order = create_send_order(
            account=account,
            symbol="PETR4",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=35.50
        )
        
        assert order.Ticker == "PETR4"
        assert order.Exchange == ""  # Bovespa
        assert order.Side == OrderSide.SELL
        assert order.OrderType == OrderType.LIMIT
        assert order.Quantity == 100
        assert order.Price == 35.50
    
    def test_create_cancel_order(self):
        """Testa criação de estrutura de cancelamento"""
        account = create_account_identifier(1, "12345")
        
        cancel = create_cancel_order(
            account=account,
            client_order_id="ORD_20240131_123456",
            password="test123"
        )
        
        assert cancel.ClientOrderID == "ORD_20240131_123456"
        assert cancel.Password == "test123"
        # Verificar que o ponteiro está correto
        assert cancel.AccountID.contents.BrokerID == 1
        assert cancel.AccountID.contents.AccountID == "12345"


class TestOrderExecutionManagerV4:
    """Testa o gerenciador de ordens v4.0.0.30"""
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Cria mock do connection manager"""
        mock_cm = Mock()
        mock_cm.dll = Mock()
        mock_cm.dll.SendOrder = Mock(return_value=12345)  # LocalOrderID
        mock_cm.dll.SendCancelOrderV2 = Mock(return_value=NResult.NL_OK)
        mock_cm.dll.GetPositionV2 = Mock(return_value=NResult.NL_OK)
        return mock_cm
    
    @pytest.fixture
    def order_manager(self, mock_connection_manager):
        """Cria instância do order manager para testes"""
        manager = OrderExecutionManagerV4(mock_connection_manager)
        # Simular account info
        manager.account_info = {
            'broker_id': 1,
            'account_id': '12345',
            'account_name': 'Test Account'
        }
        manager.account_identifier = create_account_identifier(1, "12345")
        return manager
    
    def test_initialization(self, order_manager):
        """Testa inicialização do manager"""
        assert order_manager is not None
        assert order_manager.account_identifier is not None
        assert order_manager.account_identifier.BrokerID == 1
    
    def test_send_market_order(self, order_manager, mock_connection_manager):
        """Testa envio de ordem de mercado"""
        signal = {
            'symbol': 'WDOQ25',
            'action': 'buy',
            'quantity': 5,
            'order_type': 'market'
        }
        
        order = order_manager.send_order(signal)
        
        assert order is not None
        assert order.symbol == 'WDOQ25'
        assert order.side == order_manager.OrderSide.BUY
        assert order.order_type == order_manager.OrderType.MARKET
        assert order.quantity == 5
        assert order.cl_ord_id.startswith("ORD_")
    
    def test_send_limit_order(self, order_manager):
        """Testa envio de ordem limite"""
        signal = {
            'symbol': 'PETR4',
            'action': 'sell',
            'quantity': 100,
            'price': 35.50,
            'order_type': 'limit'
        }
        
        order = order_manager.send_order(signal)
        
        assert order is not None
        assert order.symbol == 'PETR4'
        assert order.side == order_manager.OrderSide.SELL
        assert order.order_type == order_manager.OrderType.LIMIT
        assert order.quantity == 100
        assert order.price == 35.50
    
    def test_validate_signal(self, order_manager):
        """Testa validação de sinais"""
        # Sinal válido
        valid_signal = {
            'symbol': 'WDOQ25',
            'action': 'buy',
            'quantity': 10
        }
        assert order_manager._validate_signal(valid_signal) is True
        
        # Sinal sem símbolo
        invalid_signal1 = {
            'action': 'buy',
            'quantity': 10
        }
        assert order_manager._validate_signal(invalid_signal1) is False
        
        # Sinal com quantidade inválida
        invalid_signal2 = {
            'symbol': 'WDOQ25',
            'action': 'buy',
            'quantity': 0
        }
        assert order_manager._validate_signal(invalid_signal2) is False
        
        # Sinal com ação inválida
        invalid_signal3 = {
            'symbol': 'WDOQ25',
            'action': 'invalid',
            'quantity': 10
        }
        assert order_manager._validate_signal(invalid_signal3) is False
    
    def test_cancel_order(self, order_manager, mock_connection_manager):
        """Testa cancelamento de ordem"""
        # Criar ordem primeiro
        order = Order(
            symbol='WDOQ25',
            side=order_manager.OrderSide.BUY,
            order_type=order_manager.OrderType.LIMIT,
            quantity=10,
            price=5000.0,
            cl_ord_id='TEST_ORDER_123'
        )
        
        # Adicionar à lista de pendentes
        order_manager.pending_orders['TEST_ORDER_123'] = order
        
        # Cancelar ordem
        result = order_manager.cancel_order('TEST_ORDER_123')
        
        assert result is True
        mock_connection_manager.dll.SendCancelOrderV2.assert_called_once()
    
    def test_get_exchange_for_symbol(self, order_manager):
        """Testa determinação correta da bolsa"""
        assert order_manager._get_exchange_for_symbol('WDOQ25') == 'F'
        assert order_manager._get_exchange_for_symbol('WINQ25') == 'F'
        assert order_manager._get_exchange_for_symbol('PETR4') == 'B'
        assert order_manager._get_exchange_for_symbol('VALE3') == 'B'


class TestConnectionManagerV4:
    """Testa o gerenciador de conexão v4.0.0.30"""
    
    @pytest.fixture
    def mock_dll(self):
        """Cria mock da DLL"""
        mock = MagicMock()
        mock.SetServerAndPort.return_value = NResult.NL_OK
        mock.DLLInitializeLogin.return_value = NResult.NL_OK
        mock.SetOrderCallback.return_value = NResult.NL_OK
        mock.SetOrderHistoryCallback.return_value = NResult.NL_OK
        mock.SubscribeTicker.return_value = NResult.NL_OK
        mock.GetHistoryTrades.return_value = 1
        return mock
    
    @pytest.fixture
    def connection_manager(self):
        """Cria instância do connection manager"""
        # Usar caminho dummy para testes
        cm = ConnectionManagerV4("dummy.dll")
        return cm
    
    def test_initialization(self, connection_manager):
        """Testa inicialização básica"""
        assert connection_manager is not None
        assert connection_manager.dll is None
        assert connection_manager.connected is False
        assert len(connection_manager.callbacks) == 0
    
    def test_get_current_wdo_contract(self, connection_manager):
        """Testa detecção automática de contratos WDO"""
        # Testar para julho de 2025 - deve retornar WDOQ25 (agosto)
        july_date = datetime(2025, 7, 15)
        contract = connection_manager._get_current_wdo_contract(july_date)
        assert contract == "WDOQ25"  # Q = Agosto
        
        # Testar para dezembro de 2025 - deve retornar WDOF26 (janeiro/26)
        december_date = datetime(2025, 12, 20)
        contract = connection_manager._get_current_wdo_contract(december_date)
        assert contract == "WDOF26"  # F = Janeiro, 26 = 2026
    
    def test_get_smart_ticker_variations(self, connection_manager):
        """Testa geração de variações de ticker"""
        # Para WDO, deve gerar variações inteligentes
        variations = connection_manager._get_smart_ticker_variations("WDO")
        assert len(variations) >= 2
        assert variations[0].startswith("WDO")  # Contrato atual
        assert "WDO" in variations  # Genérico
        
        # Para outros tickers, apenas o original
        variations = connection_manager._get_smart_ticker_variations("PETR4")
        assert variations == ["PETR4"]
    
    @patch('os.path.exists')
    @patch('ctypes.WinDLL')
    def test_load_dll(self, mock_windll, mock_exists, connection_manager):
        """Testa carregamento da DLL"""
        mock_exists.return_value = True
        mock_dll_instance = MagicMock()
        mock_windll.return_value = mock_dll_instance
        
        dll = connection_manager._load_dll()
        
        assert dll is not None
        mock_windll.assert_called_once_with(connection_manager.dll_path)
    
    def test_validate_market_data_production(self, connection_manager):
        """Testa validação de dados de mercado em produção"""
        # Simular ambiente de produção
        with patch.dict(os.environ, {'TRADING_ENV': 'PRODUCTION'}):
            # Sem conexão de market data
            connection_manager.market_connected = False
            data = {'price': 5000.0, 'timestamp': datetime.now()}
            assert connection_manager._validate_market_data(data) is False
            
            # Com conexão mas dados antigos
            connection_manager.market_connected = True
            old_data = {
                'price': 5000.0,
                'timestamp': datetime.now() - datetime.timedelta(seconds=10)
            }
            assert connection_manager._validate_market_data(old_data) is False
            
            # Dados válidos
            valid_data = {'price': 5000.0, 'timestamp': datetime.now()}
            assert connection_manager._validate_market_data(valid_data) is True
            
            # Preço suspeito
            suspicious_data = {'price': 20000.0, 'timestamp': datetime.now()}
            assert connection_manager._validate_market_data(suspicious_data) is False


class TestIntegration:
    """Testes de integração entre componentes"""
    
    @pytest.fixture
    def integrated_system(self):
        """Cria sistema integrado para testes"""
        # Mock da DLL
        mock_dll = MagicMock()
        mock_dll.SendOrder.return_value = 12345
        mock_dll.GetAccount.return_value = NResult.NL_OK
        
        # Connection Manager
        cm = ConnectionManagerV4("dummy.dll")
        cm.dll = mock_dll
        cm.connected = True
        cm.login_state = 0  # Connected
        
        # Order Manager
        om = OrderExecutionManagerV4(cm)
        om.account_identifier = create_account_identifier(1, "12345")
        
        return {'cm': cm, 'om': om, 'dll': mock_dll}
    
    def test_full_order_flow(self, integrated_system):
        """Testa fluxo completo de ordem"""
        om = integrated_system['om']
        dll = integrated_system['dll']
        
        # Enviar ordem
        signal = {
            'symbol': 'WDOQ25',
            'action': 'buy',
            'quantity': 10,
            'price': 5000.0,
            'order_type': 'limit'
        }
        
        order = om.send_order(signal)
        
        # Verificar ordem criada
        assert order is not None
        assert order.symbol == 'WDOQ25'
        assert order.side == om.OrderSide.BUY
        assert order.quantity == 10
        assert order.price == 5000.0
        
        # Processar fila (simular thread)
        if not om.order_queue.empty():
            queued_order = om.order_queue.get()
            om._execute_order(queued_order)
            
            # Verificar que SendOrder foi chamado
            dll.SendOrder.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])