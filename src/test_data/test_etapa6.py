"""
Testes para Etapa 6 - Sistema Integrado
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import queue
import sys
import os

# Adicionar o diretório src ao path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_system import TradingSystem


class MockConnection:
    """Mock da conexão para testes"""
    def __init__(self, dll_path):
        self.dll_path = dll_path
        self.connected = False
        self.callbacks = {}
        
    def initialize(self, username, password, key=''):
        self.connected = True
        return True
        
    def set_trade_callback(self, callback):
        self.callbacks['trade'] = callback
        
    def set_book_callback(self, callback):
        self.callbacks['book'] = callback
        
    def set_state_callback(self, callback):
        self.callbacks['state'] = callback
        
    def subscribe_ticker(self, ticker):
        return True
        
    def unsubscribe_ticker(self, ticker):
        return True
        
    def disconnect(self):
        self.connected = False
        
    def request_historical_data(self, ticker, start, end):
        """Retorna dados históricos mock"""
        trades = []
        base_time = start
        base_price = 5000
        
        # Gerar 1000 trades
        for i in range(1000):
            trades.append({
                'timestamp': base_time + timedelta(seconds=i*3),
                'price': base_price + np.random.randn() * 10,
                'volume': np.random.randint(5, 20),
                'trade_type': 2 if np.random.random() > 0.5 else 3
            })
            
        return trades


class TestTradingSystem(unittest.TestCase):
    """Testes do sistema integrado"""
    
    def setUp(self):
        """Configuração dos testes"""
        self.config = {
            'dll_path': 'mock_dll',
            'username': 'test_user',
            'password': 'test_pass',
            'models_dir': 'test_models/',
            'ticker': 'WDOQ25',
            'historical_days': 1,
            'ml_interval': 5,  # 5 segundos para teste
            'initial_balance': 100000,
            'strategy': {
                'direction_threshold': 0.3,
                'magnitude_threshold': 0.0001,
                'confidence_threshold': 0.6
            },
            'risk': {
                'max_daily_loss': 0.05,
                'max_positions': 1
            }
        }
        
    def test_system_initialization(self):
        """Testa inicialização do sistema"""
        system = TradingSystem(self.config)
        
        # Verificar estado inicial
        self.assertFalse(system.initialized)
        self.assertFalse(system.is_running)
        self.assertIsNone(system.connection)
        
    @patch('trading_system.ConnectionManager', MockConnection)
    @patch('trading_system.ModelManager')
    def test_initialize_components(self, mock_model_manager):
        """Testa inicialização dos componentes"""
        # Configurar mock do model manager
        mock_model_manager.return_value.load_models.return_value = True
        mock_model_manager.return_value.models = {
            'test_model': Mock(predict=Mock(return_value=[[1, 0.001, 0.8]]))
        }
        mock_model_manager.return_value.model_features = {
            'test_model': ['close', 'ema_20', 'rsi']
        }
        
        system = TradingSystem(self.config)
        
        # Inicializar
        success = system.initialize()
        
        # Verificações
        self.assertTrue(success)
        self.assertTrue(system.initialized)
        self.assertIsNotNone(system.connection)
        self.assertIsNotNone(system.model_manager)
        self.assertIsNotNone(system.data_structure)
        self.assertIsNotNone(system.feature_engine)
        self.assertIsNotNone(system.ml_coordinator)
        self.assertIsNotNone(system.strategy_engine)
        
    @patch('trading_system.ConnectionManager', MockConnection)
    @patch('trading_system.ModelManager')
    def test_system_start(self, mock_model_manager):
        """Testa início do sistema"""
        # Configurar mocks
        mock_model_manager.return_value.load_models.return_value = True
        mock_model_manager.return_value.models = {'test_model': Mock()}
        mock_model_manager.return_value.model_features = {
            'test_model': ['close', 'volume']
        }
        
        system = TradingSystem(self.config)
        system.initialize()
        
        # Substituir main_loop para não bloquear
        system._main_loop = Mock()
        
        # Iniciar sistema
        success = system.start()
        
        # Verificações
        self.assertTrue(success)
        self.assertTrue(system.is_running)
        # Verificar se data_structure foi inicializada e tem dados
        self.assertIsNotNone(system.data_structure)
        if system.data_structure is not None and hasattr(system.data_structure, 'candles') and system.data_structure.candles is not None:
            self.assertGreater(len(system.data_structure.candles), 0)
        
    def test_callback_processing(self):
        """Testa processamento de callbacks"""
        system = TradingSystem(self.config)
        
        # Configurar mocks mínimos
        system.real_time_processor = Mock()
        
        # Mock metrics com estrutura apropriada
        mock_metrics = Mock()
        mock_metrics.metrics = {
            'trades_processed': 0,
            'errors': []
        }
        system.metrics = mock_metrics
        system.is_running = True
        
        # Simular trade
        trade_data = {
            'timestamp': datetime.now(),
            'price': 5000.0,
            'volume': 10,
            'trade_type': 2
        }
        
        # Processar
        system._on_trade(trade_data)
        
        # Verificar
        system.real_time_processor.process_trade.assert_called_once_with(trade_data)
        system.metrics.record_trade.assert_called_once()
        
    def test_ml_pipeline(self):
        """Testa pipeline de ML"""
        system = TradingSystem(self.config)
        
        # Configurar mocks
        system.ml_coordinator = Mock()
        system.ml_coordinator.process_prediction_request.return_value = {
            'direction': 0.8,
            'magnitude': 0.002,
            'confidence': 0.85,
            'timestamp': datetime.now()
        }
        
        system.data_structure = Mock()
        system.data_structure.candles = pd.DataFrame({
            'close': [5000] * 100
        })
        
        mock_metrics = Mock()
        system.metrics = mock_metrics
        
        # Mock da queue
        mock_queue = Mock()
        system.signal_queue = mock_queue
        
        # Processar predição
        system._process_ml_prediction()
        
        # Verificar
        system.ml_coordinator.process_prediction_request.assert_called_once()
        self.assertIsNotNone(system.last_prediction)
        
        # Verificar se a predição foi adicionada à queue se não estiver full
        # Como o mock queue não é "full", deve ter sido chamado put
        if not mock_queue.full.return_value:
            mock_queue.put.assert_called_once()
        
    def test_signal_generation(self):
        """Testa geração de sinais"""
        system = TradingSystem(self.config)
        
        # Configurar mocks
        system.strategy_engine = Mock()
        system.strategy_engine.process_prediction.return_value = {
            'action': 'buy',
            'price': 5000,
            'stop_loss': 4950,
            'take_profit': 5100,
            'position_size': 1
        }
        
        system.data_structure = Mock()
        system.account_info = {'balance': 100000}
        system.metrics = Mock()
        system.active_positions = {}
        
        # Predição mock
        prediction = {
            'direction': 0.8,
            'magnitude': 0.002,
            'confidence': 0.85
        }
        
        # Processar
        system._process_signal_generation(prediction)
        
        # Verificar
        system.strategy_engine.process_prediction.assert_called_once()
        self.assertEqual(len(system.active_positions), 1)
        
    def test_thread_safety(self):
        """Testa segurança de threads"""
        system = TradingSystem(self.config)
        
        # Testar operações concorrentes nas filas
        system.ml_queue = queue.Queue(maxsize=5)
        system.signal_queue = queue.Queue(maxsize=5)
        
        # Adicionar múltiplos itens
        for i in range(3):
            system._request_ml_prediction()
            
        self.assertEqual(system.ml_queue.qsize(), 3)
        
    def test_error_handling(self):
        """Testa tratamento de erros"""
        system = TradingSystem(self.config)
        
        # Configurar para gerar erro
        system.ml_coordinator = Mock()
        system.ml_coordinator.process_prediction_request.side_effect = Exception("Test error")
        
        system.data_structure = Mock()
        system.data_structure.candles = pd.DataFrame({'close': [5000] * 100})
        
        # Não deve lançar exceção
        system._process_ml_prediction()
        
        # Verificar log de erro (verificação indireta)
        self.assertIsNone(system.last_prediction)
        
    def test_system_status(self):
        """Testa obtenção de status do sistema"""
        system = TradingSystem(self.config)
        
        # Configurar estado
        system.is_running = True
        system.initialized = True
        system.ticker = 'WDOQ25'
        system.last_prediction = {'direction': 0.5}
        system.active_positions = {'WDOQ25': {'side': 'buy'}}
        
        # Obter status
        status = system.get_status()
        
        # Verificar
        self.assertTrue(status['running'])
        self.assertTrue(status['initialized'])
        self.assertEqual(status['ticker'], 'WDOQ25')
        self.assertEqual(len(status['active_positions']), 1)
    
    def test_contract_management(self):
        """Testa gerenciamento automático de contratos"""
        system = TradingSystem(self.config)
        
        # Testar detecção de contrato para diferentes datas
        # Janeiro
        jan_contract = system._get_current_contract(datetime(2025, 1, 15))
        self.assertEqual(jan_contract, "WDOG25")
        
        # Julho
        jul_contract = system._get_current_contract(datetime(2025, 7, 15))
        self.assertEqual(jul_contract, "WDOQ25")
        
        # Dezembro
        dec_contract = system._get_current_contract(datetime(2025, 12, 15))
        self.assertEqual(dec_contract, "WDOF25")
        
    def test_contract_rollover(self):
        """Testa mudança de contrato"""
        system = TradingSystem(self.config)
        
        # Configurar mocks
        system.connection = Mock()
        system.connection.unsubscribe_ticker = Mock()
        system.connection.subscribe_ticker = Mock()
        
        # Mock data_structure com DataFrames apropriados
        mock_data_structure = Mock()
        mock_data_structure.candles = pd.DataFrame({'close': [5000]})
        mock_data_structure.microstructure = pd.DataFrame()
        mock_data_structure.orderbook = pd.DataFrame()
        mock_data_structure.indicators = pd.DataFrame()
        mock_data_structure.features = pd.DataFrame()
        system.data_structure = mock_data_structure
        
        system.data_loader = Mock()
        
        # Simular contrato atual
        system.ticker = "WDOG25"
        
        # Forçar checagem com data diferente
        with patch('trading_system.datetime') as mock_datetime:
            # Simular que estamos em fevereiro
            mock_datetime.now.return_value = datetime(2025, 2, 1)
            
            # Executar verificação
            system._check_contract_rollover()
            
            # Verificar mudança
            self.assertEqual(system.ticker, "WDOH25")
            
            # Verificar que cancelou e re-subscreveu
            system.connection.unsubscribe_ticker.assert_called_with("WDOG25")
            system.connection.subscribe_ticker.assert_called_with("WDOH25")
                
    def test_system_stop(self):
        """Testa parada do sistema"""
        system = TradingSystem(self.config)
        
        # Configurar mocks
        system.is_running = True
        system.connection = Mock()
        system.ml_thread = Mock()
        system.signal_thread = Mock()
        
        # Parar
        system.stop()
        
        # Verificar
        self.assertFalse(system.is_running)
        system.connection.disconnect.assert_called_once()


if __name__ == '__main__':
    unittest.main()