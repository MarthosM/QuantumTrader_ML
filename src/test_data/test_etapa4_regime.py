"""
Testes para Etapa 4 com foco em Regime
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys

# Adicionar paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mocks'))

from prediction_engine import PredictionEngine
from ml_coordinator import MLCoordinator
from mock_regime_trainer import MockRegimeTrainer


class MockModelManager:
    """Gerenciador de modelos mock com modelos de regime"""
    def __init__(self):
        self.models = {
            'trend_model_1': MockTrendModel(),
            'trend_model_2': MockTrendModel(),
            'range_model_1': MockRangeModel(),
            'range_model_2': MockRangeModel()
        }
        self.model_features = {
            'trend_model_1': ['close', 'ema_20', 'momentum_10'],
            'trend_model_2': ['close', 'rsi', 'atr'],
            'range_model_1': ['close', 'bb_upper_20', 'bb_lower_20'],
            'range_model_2': ['close', 'support_level', 'resistance_level']
        }


class MockTrendModel:
    """Modelo mock para tendência"""
    def predict_proba(self, X):
        # Retorna alta probabilidade para tendência
        return np.array([[0.2, 0.8]])  # 80% bullish


class MockRangeModel:
    """Modelo mock para range"""
    def predict_proba(self, X):
        # Retorna probabilidade moderada
        return np.array([[0.4, 0.6]])  # 60% para reversão


class TestPredictionEngineRegime(unittest.TestCase):
    """Testes para PredictionEngine com regime"""
    
    def setUp(self):
        self.model_manager = MockModelManager()
        self.engine = PredictionEngine(self.model_manager)
        
    def test_predict_trend_up(self):
        """Testa predição em tendência de alta"""
        # Criar features
        features = pd.DataFrame({
            'close': [5000, 5010, 5020],
            'ema_20': [4990, 5000, 5010],
            'momentum_10': [20, 25, 30],
            'rsi': [60, 65, 70],
            'atr': [10, 11, 12],
            'bb_upper_20': [5030, 5040, 5050],
            'bb_lower_20': [4970, 4980, 4990]
        }, index=pd.date_range(end=datetime.now(), periods=3, freq='1min'))
        
        # Adicionar features para completar 50 linhas (mínimo)
        features = pd.concat([features] * 20).reset_index(drop=True)
        features.index = pd.date_range(end=datetime.now(), periods=len(features), freq='1min')
        
        # Regime info para tendência de alta
        regime_info = {
            'regime': 'trend_up',
            'confidence': 0.85,
            'trend_strength': 0.8
        }
        
        # Executar predição
        result = self.engine.predict_by_regime(features, regime_info)
        
        # Verificar resultado
        self.assertIsNotNone(result, "PredictionEngine.predict_by_regime returned None. Check implementation and input data.")
        if result is not None:
            self.assertEqual(result['regime_type'], 'trend')
            self.assertGreater(result['direction'], 0)  # Deve ser positivo
            self.assertTrue(result['can_trade'])
        
    def test_predict_range_near_support(self):
        """Testa predição em range próximo ao suporte"""
        # Criar features
        features = pd.DataFrame({
            'close': [4980] * 50,
            'bb_upper_20': [5020] * 50,
            'bb_lower_20': [4980] * 50,
            'support_level': [4975] * 50,
            'resistance_level': [5025] * 50
        }, index=pd.date_range(end=datetime.now(), periods=50, freq='1min'))
        
        # Regime info para range
        regime_info = {
            'regime': 'range',
            'confidence': 0.75,
            'support_resistance_proximity': 'near_support',
            'support_level': 4975,
            'resistance_level': 5025
        }
        
        # Executar predição
        result = self.engine.predict_by_regime(features, regime_info)
        
        # Verificar resultado
        self.assertIsNotNone(result)
        if result is not None:
            self.assertEqual(result['regime_type'], 'range')
            self.assertGreater(result['direction'], 0)  # Compra no suporte
        else:
            self.fail("PredictionEngine.predict_by_regime returned None. Check implementation and input data.")
        
    def test_no_trade_middle_range(self):
        """Testa sem sinal no meio do range"""
        features = pd.DataFrame({
            'close': [5000] * 50,
            'bb_upper_20': [5050] * 50,
            'bb_lower_20': [4950] * 50,
            'support_level': [4950] * 50,
            'resistance_level': [5050] * 50
        }, index=pd.date_range(end=datetime.now(), periods=50, freq='1min'))
        
        regime_info = {
            'regime': 'range',
            'confidence': 0.8,
            'support_resistance_proximity': 'neutral'
        }
        
        result = self.engine.predict_by_regime(features, regime_info)
        
        self.assertIsNotNone(result, "PredictionEngine.predict_by_regime returned None. Check implementation and input data.")
        if result is not None:
            self.assertFalse(result['can_trade'])
        else:
            self.fail("PredictionEngine.predict_by_regime returned None. Check implementation and input data.")


class TestMLCoordinatorRegime(unittest.TestCase):
    """Testes para MLCoordinator com regime"""
    
    def setUp(self):
        self.model_manager = MockModelManager()
        self.regime_trainer = MockRegimeTrainer()
        self.engine = PredictionEngine(self.model_manager)
        
        # Mock feature engine
        class MockFeatureEngine:
            def calculate(self, data):
                # Retorna features básicas
                features = pd.DataFrame({
                    'close': data.candles['close'],
                    'ema_20': data.candles['close'].rolling(20).mean().fillna(data.candles['close']),
                    'momentum_10': 0.001,
                    'rsi': 60,
                    'atr': 10,
                    'bb_upper_20': data.candles['close'] * 1.02,
                    'bb_lower_20': data.candles['close'] * 0.98,
                    'support_level': data.candles['close'].min(),
                    'resistance_level': data.candles['close'].max()
                }, index=data.candles.index)
                
                return {'model_ready': features}
        
        self.coordinator = MLCoordinator(
            self.model_manager,
            MockFeatureEngine(),
            self.engine,
            self.regime_trainer
        )
        
    def test_full_flow_trend(self):
        """Testa fluxo completo em tendência"""
        # Criar dados em tendência de alta
        n = 100
        dates = pd.date_range(end=datetime.now(), periods=n, freq='1min')
        prices = 5000 + np.arange(n) * 0.5  # Tendência de alta clara
        
        class MockData:
            def __init__(self):
                self.candles = pd.DataFrame({
                    'open': prices,
                    'high': prices + 2,
                    'low': prices - 2,
                    'close': prices,
                    'volume': 100
                }, index=dates)
                self.indicators = pd.DataFrame(index=dates)
                self.features = pd.DataFrame(index=dates)
        
        data = MockData()
        
        # Processar predição
        result = self.coordinator.process_prediction_request(data)
        
        # Verificar resultado
        self.assertIsNotNone(result)
        self.assertIsNotNone(result, "PredictionEngine.predict_by_regime returned None. Check implementation and input data.")
        if result is not None:
            self.assertIn('regime', result)
            self.assertIn('trade_decision', result)
            
            # Em tendência de alta clara, deve sugerir compra
            if result['regime'] == 'trend_up' and result['can_trade']:
                self.assertEqual(result['trade_decision'], 'BUY')
                self.assertEqual(result['risk_reward_target'], 2.0)
        else:
            self.fail("PredictionEngine.predict_by_regime returned None. Check implementation and input data.")
            
    def test_full_flow_range(self):
        """Testa fluxo completo em lateralização"""
        # Criar dados lateralizados
        n = 100
        dates = pd.date_range(end=datetime.now(), periods=n, freq='1min')
        # Preços oscilando em torno de 5000
        prices = 5000 + np.sin(np.arange(n) * 0.1) * 10
        
        class MockData:
            def __init__(self):
                self.candles = pd.DataFrame({
                    'open': prices,
                    'high': prices + 2,
                    'low': prices - 2,
                    'close': prices,
                    'volume': 100
                }, index=dates)
                self.indicators = pd.DataFrame(index=dates)
                self.features = pd.DataFrame(index=dates)
        
        data = MockData()
        
        # Processar predição
        result = self.coordinator.process_prediction_request(data)
        
        # Verificar resultado
        self.assertIsNotNone(result)
        
        # Em range, risk/reward deve ser 1.5:1
        if result is not None and result.get('regime') == 'range' and result.get('can_trade'):
            self.assertEqual(result['risk_reward_target'], 1.5)
            
    def test_statistics(self):
        """Testa estatísticas do coordenador"""
        # Criar dados mock com OHLCV completo
        class MockData:
            def __init__(self):
                dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
                prices = 5000 + np.random.randn(100).cumsum()
                self.candles = pd.DataFrame({
                    'open': prices + np.random.randn(100) * 0.5,
                    'high': prices + abs(np.random.randn(100)) * 2,
                    'low': prices - abs(np.random.randn(100)) * 2,
                    'close': prices,
                    'volume': np.random.randint(50, 200, 100)
                }, index=dates)
                
                # Ajustar para consistência OHLC
                self.candles['high'] = self.candles[['open', 'high', 'close']].max(axis=1)
                self.candles['low'] = self.candles[['open', 'low', 'close']].min(axis=1)
                
                self.indicators = pd.DataFrame(index=dates)
                self.features = pd.DataFrame(index=dates)
        
        data = MockData()
        
        # Fazer várias predições
        successful_predictions = 0
        for i in range(5):
            result = self.coordinator.force_prediction(data)
            if result is not None:
                successful_predictions += 1
            
        # Obter estatísticas
        stats = self.coordinator.get_coordinator_stats()
        
        # Verificar que ao menos algumas predições foram bem-sucedidas
        self.assertGreaterEqual(stats['total_predictions'], 1)
        self.assertIn('trend_predictions', stats)
        self.assertIn('range_predictions', stats)
        self.assertIn('no_trade_predictions', stats)


if __name__ == '__main__':
    unittest.main()