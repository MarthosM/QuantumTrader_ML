"""
Teste do Trading System corrigido
Valida as correções aplicadas
"""

import sys
import os
import tempfile
import unittest
from unittest.mock import Mock, patch
from datetime import datetime

# Adicionar src ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_system import TradingSystem


class TestTradingSystemFixed(unittest.TestCase):
    """Testa as correções aplicadas no TradingSystem"""
    
    def setUp(self):
        """Setup para testes"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Configuração mínima para teste
        self.config = {
            'dll_path': 'mock_dll.dll',
            'username': 'test_user',
            'password': 'test_pass',
            'models_dir': self.temp_dir,
            'ticker': 'WDOH25',
            'strategy': {},
            'risk': {},
            'initial_balance': 100000,
            'ml_interval': 30,
            'historical_days': 5,
            'use_gui': False
        }
    
    def tearDown(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_trading_system_creation(self):
        """Testa criação do sistema sem erros"""
        try:
            ts = TradingSystem(self.config)
            self.assertIsNotNone(ts)
            self.assertEqual(ts.config, self.config)
            self.assertFalse(ts.initialized)
            self.assertFalse(ts.is_running)
        except Exception as e:
            self.fail(f"Erro na criação do TradingSystem: {e}")
    
    def test_contract_detection(self):
        """Testa detecção de contrato correto"""
        ts = TradingSystem(self.config)
        
        # Testar para julho de 2025
        test_date = datetime(2025, 7, 18)
        contract = ts._get_current_contract(test_date)
        
        # Julho = Q, 2025 = 25
        expected = "WDOQ25"
        self.assertEqual(contract, expected)
    
    def test_safe_historical_data_loading(self):
        """Testa carregamento seguro de dados históricos"""
        ts = TradingSystem(self.config)
        
        # Simular data_structure inicializada
        from data_structure import TradingDataStructure
        ts.data_structure = TradingDataStructure()
        ts.data_structure.initialize_structure()
        
        # Testar carregamento
        result = ts._load_historical_data_safe('WDOH25', 5)
        self.assertTrue(result)
        
        # Verificar se dados foram gerados
        self.assertGreater(len(ts.data_structure.candles), 0)
    
    def test_setup_callbacks_safe(self):
        """Testa setup de callbacks sem erros"""
        ts = TradingSystem(self.config)
        
        # Simular conexão
        ts.connection = Mock()
        
        # Não deve gerar erro mesmo sem métodos de callback
        try:
            ts._setup_callbacks()
        except Exception as e:
            self.fail(f"Erro no setup de callbacks: {e}")
    
    def test_feature_calculation_safe(self):
        """Testa cálculo de features com verificações de None"""
        ts = TradingSystem(self.config)
        
        # Sem componentes inicializados
        ts._process_feature_calculation()  # Não deve gerar erro
        
        # Com feature_engine = None
        ts.feature_engine = None
        ts.data_structure = Mock()
        ts._process_feature_calculation()  # Não deve gerar erro
    
    def test_ml_prediction_safe(self):
        """Testa predição ML com verificações de None"""
        ts = TradingSystem(self.config)
        
        # Sem componentes inicializados
        ts._process_ml_prediction()  # Não deve gerar erro
        
        # Com data_structure = None
        ts.data_structure = None
        ts._process_ml_prediction()  # Não deve gerar erro
    
    def test_signal_generation_safe(self):
        """Testa geração de sinais com verificações de None"""
        ts = TradingSystem(self.config)
        
        mock_prediction = {
            'direction': 1.0,
            'confidence': 0.8,
            'can_trade': True
        }
        
        # Sem strategy_engine inicializado
        ts._process_signal_generation(mock_prediction)  # Não deve gerar erro
    
    def test_metrics_update_safe(self):
        """Testa atualização de métricas com verificações de None"""
        ts = TradingSystem(self.config)
        
        # Sem metrics inicializado
        ts._update_metrics()  # Não deve gerar erro
    
    def test_on_trade_safe(self):
        """Testa callback de trade com verificações de None"""
        ts = TradingSystem(self.config)
        ts.is_running = True
        
        mock_trade = {
            'price': 5000.0,
            'volume': 100,
            'timestamp': datetime.now()
        }
        
        # Sem real_time_processor inicializado
        ts._on_trade(mock_trade)  # Não deve gerar erro


if __name__ == '__main__':
    print("Testando correções do TradingSystem...")
    unittest.main(verbosity=2)
