"""
tests/test_etapa1.py - Testes para infraestrutura base
"""

import unittest
import tempfile
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

# Importar os módulos a serem testados
import sys
import os
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

from connection_manager import ConnectionManager
from model_manager import ModelManager
from data_structure import TradingDataStructure


class TestConnectionManager(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dll_path = os.path.join(self.temp_dir, "test_dll.dll")
        # Criar arquivo DLL fake
        with open(self.dll_path, 'w') as f:
            f.write("fake dll")
            
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_initialization(self):
        """Testa inicialização do gerenciador de conexão"""
        conn = ConnectionManager(self.dll_path)
        self.assertIsNotNone(conn)
        self.assertFalse(conn.connected)
        self.assertEqual(conn.dll_path, self.dll_path)
        
    def test_register_callbacks(self):
        """Testa registro de callbacks"""
        conn = ConnectionManager(self.dll_path)
        
        # Registrar callback de trade
        trade_callback = Mock()
        conn.register_trade_callback(trade_callback)
        self.assertIn(trade_callback, conn.trade_callbacks)
        
        # Registrar callback de estado
        state_callback = Mock()
        conn.register_state_callback(state_callback)
        self.assertIn(state_callback, conn.state_callbacks)
        
    def test_dll_loading(self):
        """Testa carregamento da DLL real"""
        # Caminho para a DLL real do Profit
        real_dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
        
        # Verificar se a DLL real existe
        if not os.path.exists(real_dll_path):
            self.skipTest(f"DLL real não encontrada em: {real_dll_path}")
        
        # Testar com DLL real
        conn = ConnectionManager(real_dll_path)
        dll = conn._load_dll()
        
        # Verificar se carregou com sucesso
        self.assertIsNotNone(dll)
        self.assertEqual(conn.dll_path, real_dll_path)
        
    def test_dll_loading_failure(self):
        """Testa comportamento quando DLL não existe"""
        fake_dll_path = r"C:\path\that\does\not\exist\fake.dll"
        
        conn = ConnectionManager(fake_dll_path)
        dll = conn._load_dll()
        
        # Deve retornar None quando DLL não existe
        self.assertIsNone(dll)


class TestModelManager(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.temp_dir, "models")
        os.makedirs(self.models_dir)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def create_dummy_model(self, model_name: str, features: list):
        """Cria um modelo dummy para testes"""
        # Criar modelo sklearn simples
        from sklearn.ensemble import RandomForestRegressor
        
        X = np.random.rand(100, len(features))
        y = np.random.rand(100)
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Adicionar nomes das features
        model.feature_names_in_ = np.array(features)
        
        # Salvar modelo
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        
        # Salvar features em arquivo separado
        import json
        features_path = os.path.join(self.models_dir, f"{model_name}_features.json")
        with open(features_path, 'w') as f:
            json.dump(features, f)
            
        return model_path
        
    def test_model_loading(self):
        """Testa carregamento de modelos"""
        # Criar modelos de teste
        features = ['close', 'volume', 'rsi', 'ema_20']
        self.create_dummy_model("test_model1", features)
        self.create_dummy_model("test_model2", features + ['momentum_5'])
        
        # Carregar modelos
        manager = ModelManager(self.models_dir)
        success = manager.load_models()
        
        self.assertTrue(success)
        self.assertEqual(len(manager.models), 2)
        self.assertIn("test_model1", manager.models)
        self.assertIn("test_model2", manager.models)
        
    def test_feature_extraction(self):
        """Testa extração de features"""
        features = ['close', 'volume', 'rsi', 'ema_20', 'momentum_5']
        self.create_dummy_model("test_model", features)
        
        manager = ModelManager(self.models_dir)
        manager.load_models()
        
        model_features = manager.get_model_features("test_model")
        self.assertEqual(set(model_features), set(features))
        
    def test_get_all_required_features(self):
        """Testa obtenção de todas as features necessárias"""
        # Criar modelos com features diferentes
        self.create_dummy_model("model1", ['close', 'volume', 'rsi'])
        self.create_dummy_model("model2", ['close', 'ema_20', 'momentum_5'])
        
        manager = ModelManager(self.models_dir)
        manager.load_models()
        
        all_features = manager.get_all_required_features()
        expected = {'close', 'volume', 'rsi', 'ema_20', 'momentum_5'}
        self.assertEqual(set(all_features), expected)


class TestTradingDataStructure(unittest.TestCase):
    
    def setUp(self):
        self.data = TradingDataStructure()
        
    def test_initialization(self):
        """Testa inicialização da estrutura"""
        self.data.initialize_structure()
        
        # Verificar que dataframes foram criados
        self.assertIsInstance(self.data.candles, pd.DataFrame)
        self.assertIsInstance(self.data.microstructure, pd.DataFrame)
        self.assertIsInstance(self.data.orderbook, pd.DataFrame)
        self.assertIsInstance(self.data.indicators, pd.DataFrame)
        self.assertIsInstance(self.data.features, pd.DataFrame)
        
        # Verificar colunas
        self.assertIn('close', self.data.candles.columns)
        self.assertIn('buy_volume', self.data.microstructure.columns)
        self.assertIn('rsi', self.data.indicators.columns)
        
    def create_test_candles(self, n_candles: int) -> pd.DataFrame:
        """Cria candles de teste"""
        dates = pd.date_range(end=datetime.now(), periods=n_candles, freq='1min')
        
        # Gerar preços com tendência
        base_price = 5000
        prices = base_price + np.cumsum(np.random.randn(n_candles) * 10)
        
        candles = pd.DataFrame({
            'open': prices + np.random.randn(n_candles),
            'high': prices + abs(np.random.randn(n_candles)) * 2,
            'low': prices - abs(np.random.randn(n_candles)) * 2,
            'close': prices,
            'volume': np.random.randint(100, 1000, n_candles),
            'quantidade': np.random.randint(10, 100, n_candles)
        }, index=dates)
        
        # Garantir consistência OHLC
        candles['high'] = candles[['open', 'high', 'close']].max(axis=1)
        candles['low'] = candles[['open', 'low', 'close']].min(axis=1)
        
        return candles
        
    def test_update_candles(self):
        """Testa atualização de candles"""
        self.data.initialize_structure()
        
        # Criar e adicionar candles
        new_candles = self.create_test_candles(10)
        success = self.data.update_candles(new_candles)
        
        self.assertTrue(success)
        self.assertEqual(len(self.data.candles), 10)
        self.assertIsNotNone(self.data.last_price)
        self.assertIsNotNone(self.data.last_update)
        
    def test_data_quality_check(self):
        """Testa verificação de qualidade de dados"""
        self.data.initialize_structure()
        
        # Adicionar dados válidos
        candles = self.create_test_candles(100)
        self.data.update_candles(candles)
        
        # Verificar qualidade
        quality = self.data.check_data_quality()
        
        self.assertIn('has_issues', quality)
        self.assertIn('quality_score', quality)
        self.assertGreaterEqual(quality['quality_score'], 0)
        self.assertLessEqual(quality['quality_score'], 100)
        
    def test_get_candles_window(self):
        """Testa obtenção de janela de candles"""
        self.data.initialize_structure()
        
        # Adicionar 100 candles
        candles = self.create_test_candles(100)
        self.data.update_candles(candles)
        
        # Obter últimos 20
        window = self.data.get_candles_window(20)
        
        self.assertEqual(len(window), 20)
        self.assertTrue(window.index[-1] == candles.index[-1])
        
    def test_get_summary(self):
        """Testa obtenção de resumo"""
        self.data.initialize_structure()
        
        # Adicionar alguns dados
        candles = self.create_test_candles(50)
        self.data.update_candles(candles)
        
        summary = self.data.get_summary()
        
        self.assertIn('last_price', summary)
        self.assertIn('dataframes', summary)
        self.assertEqual(summary['dataframes']['candles']['rows'], 50)


if __name__ == '__main__':
    unittest.main()