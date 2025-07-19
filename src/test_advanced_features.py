import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engine import FeatureEngine, AdvancedFeatureProcessor, IntelligentFeatureSelector
from data_structure import TradingDataStructure

class TestAdvancedFeatures(unittest.TestCase):
    
    def setUp(self):
        """Prepara dados de teste"""
        self.data_structure = TradingDataStructure()
        
        # Initialize structure first
        self.data_structure.initialize_structure()
        
        # Cria dados sintéticos
        dates = pd.date_range(start='2024-01-01 09:00:00', 
                            end='2024-01-01 16:00:00', 
                            freq='1min')
        
        # Candles
        self.data_structure.candles = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 100,
            'high': np.random.randn(len(dates)).cumsum() + 101,
            'low': np.random.randn(len(dates)).cumsum() + 99,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(100, 1000, len(dates))
        }, index=dates)
        
        # Microestrutura
        self.data_structure.microstructure = pd.DataFrame({
            'buy_volume': np.random.randint(50, 500, len(dates)),
            'sell_volume': np.random.randint(50, 500, len(dates)),
            'buy_trades': np.random.randint(10, 100, len(dates)),
            'sell_trades': np.random.randint(10, 100, len(dates))
        }, index=dates)
        
        # Feature Engine
        self.feature_engine = FeatureEngine()
        self.feature_engine.logger = self._create_logger()
    
    def test_advanced_processor_initialization(self):
        """Testa inicialização do processador avançado"""
        processor = AdvancedFeatureProcessor(self._create_logger())
        self.assertIsNotNone(processor)
        self.assertEqual(processor.max_features_by_samples, 15)
    
    def test_microstructure_features(self):
        """Testa cálculo de features de microestrutura"""
        processor = self.feature_engine.advanced_processor
        
        features = processor.extract_all_features(
            self.data_structure.candles,
            self.data_structure.microstructure,
            timestamp=123456  # Força recalculo
        )
        
        # Verifica features essenciais
        expected_features = [
            'order_flow_imbalance_1m',
            'order_flow_imbalance_5m', 
            'buy_pressure',
            'volume_imbalance'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features.columns)
            self.assertFalse(features[feature].isna().all(), f"Feature {feature} contém apenas NaN")
    
    def test_feature_selection(self):
        """Testa seleção inteligente de features"""
        # Cria features dummy
        n_samples = 1000
        n_features = 50
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Target com relação com algumas features
        y = X['feature_0'] * 2 + X['feature_1'] * 3 + np.random.randn(n_samples) * 0.1
        
        selector = IntelligentFeatureSelector(self._create_logger())
        selected = selector.select_optimal_features(X, y, max_features=10)
        
        # Verifica seleção
        self.assertEqual(len(selected.columns), 10)
        self.assertIn('feature_0', selected.columns)  # Deve selecionar features importantes
        self.assertIn('feature_1', selected.columns)
    
    def test_cache_functionality(self):
        """Testa funcionamento do cache"""
        processor = self.feature_engine.advanced_processor
        
        # Primeira chamada
        features1 = processor.extract_all_features(
            self.data_structure.candles,
            self.data_structure.microstructure,
            timestamp=1234567890
        )
        
        # Segunda chamada (deve usar cache)
        features2 = processor.extract_all_features(
            self.data_structure.candles,
            self.data_structure.microstructure,
            timestamp=1234567890
        )
        
        # Verifica que são idênticas
        pd.testing.assert_frame_equal(features1, features2)
    
    def test_adaptive_indicators(self):
        """Testa indicadores adaptativos"""
        processor = self.feature_engine.advanced_processor
        
        features = processor._extract_adaptive_technical_features(
            self.data_structure.candles
        )
        
        # Verifica indicadores adaptativos
        self.assertIn('adaptive_rsi', features.columns)
        self.assertIn('dynamic_macd', features.columns)
        self.assertIn('adaptive_bb_position', features.columns)
        
        # Verifica valores válidos
        self.assertTrue((features['adaptive_rsi'] >= 0).all())
        self.assertTrue((features['adaptive_rsi'] <= 100).all())
    
    def test_feature_reduction(self):
        """Testa redução de features de 80+ para 15-20"""
        # Simula muitas features
        result = self.feature_engine.calculate(self.data_structure, use_advanced=True)
        
        if 'features' in result and not result['features'].empty:
            initial_features = len(result['features'].columns)
            
            # Força seleção
            if initial_features > 20:
                target = self.data_structure.candles['close'].pct_change().shift(-1).fillna(0)
                selected = self.feature_engine.feature_selector.select_optimal_features(
                    result['features'],
                    target,
                    max_features=15
                )
                
                self.assertLessEqual(len(selected.columns), 15)
                self.assertGreaterEqual(len(selected.columns), 5)
    
    def _create_logger(self):
        """Cria logger para testes"""
        import logging
        logger = logging.getLogger('test')
        logger.setLevel(logging.INFO)
        return logger

if __name__ == '__main__':
    unittest.main()