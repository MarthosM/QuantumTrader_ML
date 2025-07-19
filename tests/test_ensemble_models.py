import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_manager import ModelManager, MultiModalEnsemble, HyperparameterOptimizer

class TestEnsembleModels(unittest.TestCase):
    
    def setUp(self):
        """Prepara ambiente de teste"""
        self.model_manager = ModelManager(models_dir='test_models')
        
        # Dados sintéticos
        np.random.seed(42)
        self.n_samples = 1000
        self.n_features = 20
        
        # Features sintéticas
        self.X = np.random.randn(self.n_samples, self.n_features)
        
        # Labels: simular padrão de trading
        self.y = np.random.choice([0, 1, 2], size=self.n_samples, p=[0.3, 0.4, 0.3])
    
    def test_ensemble_initialization(self):
        """Testa inicialização do ensemble"""
        success = self.model_manager.initialize_ensemble(n_features=self.n_features)
        self.assertTrue(success)
        self.assertTrue(self.model_manager.ensemble_initialized)
        
        # Verificar modelos criados
        expected_models = [
            'xgboost_fast', 'lstm_intraday', 
            'transformer_attention', 'rf_stable'
        ]
        
        for model_name in expected_models:
            self.assertIn(model_name, self.model_manager.ensemble.models)
    
    def test_ensemble_training(self):
        """Testa treinamento do ensemble"""
        # Inicializar
        self.model_manager.initialize_ensemble(n_features=self.n_features)
        
        # Treinar
        success = self.model_manager.train_ensemble(
            self.X, self.y,
            optimize_hyperparams=False  # Rápido para teste
        )
        
        self.assertTrue(success)
    
    def test_ensemble_prediction(self):
        """Testa predição do ensemble"""
        # Inicializar e treinar com dados pequenos
        self.model_manager.initialize_ensemble(n_features=self.n_features)
        
        # Dados de treino pequenos
        X_small = self.X[:200]
        y_small = self.y[:200]
        
        self.model_manager.train_ensemble(X_small, y_small)
        
        # Criar DataFrame de features
        feature_names = [f'feature_{i}' for i in range(self.n_features)]
        features_df = pd.DataFrame(
            self.X[-100:],  # Últimas 100 amostras
            columns=feature_names
        )
        
        # Predizer
        result = self.model_manager.predict(features_df, use_ensemble=True)
        
        # Verificar resultado
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('ensemble', False))
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertIn('market_regime', result)
    
    def test_regime_detection(self):
        """Testa detecção de regime de mercado"""
        # Criar features com padrões conhecidos
        features = pd.DataFrame({
            'volatility_20': [0.01, 0.02, 0.05],  # Aumentando
            'ema_9': [100, 101, 102],
            'ema_20': [99, 100, 101],
            'ema_50': [98, 99, 100]
        })
        
        regime = self.model_manager._detect_market_regime(features)
        self.assertIn(regime, ['high_volatility', 'low_volatility', 
                              'trending', 'ranging', 'undefined'])
    
    def test_hyperparameter_optimization(self):
        """Testa otimização de hiperparâmetros"""
        hyperopt = HyperparameterOptimizer(self.model_manager.logger)
        
        # Teste rápido com poucos trials
        best_params, best_score = hyperopt.optimize_hyperparameters(
            'xgboost_fast',
            self.X[:100], self.y[:100],
            self.X[100:150], self.y[100:150],
            n_trials=5
        )
        
        self.assertIsInstance(best_params, dict)
        self.assertIsInstance(best_score, (int, float))
    
    def test_sequential_data_preparation(self):
        """Testa preparação de dados sequenciais"""
        ensemble = MultiModalEnsemble(self.model_manager.logger)
        
        # Dados de teste
        X = np.random.randn(100, 10)
        
        # Preparar sequências
        X_seq = ensemble.prepare_sequential_data(X, sequence_length=20)
        
        # Verificar shape
        self.assertEqual(X_seq.shape[0], 81)  # 100 - 20 + 1
        self.assertEqual(X_seq.shape[1], 20)   # sequence_length
        self.assertEqual(X_seq.shape[2], 10)   # n_features
    
    def test_model_agreement(self):
        """Testa cálculo de concordância entre modelos"""
        ensemble = MultiModalEnsemble(self.model_manager.logger)
        
        # Predições simuladas
        predictions = {
            'model1': np.array([0.7, 0.2, 0.1]),  # Vota classe 0
            'model2': np.array([0.6, 0.3, 0.1]),  # Vota classe 0
            'model3': np.array([0.2, 0.6, 0.2]),  # Vota classe 1
            'model4': np.array([0.8, 0.1, 0.1]),  # Vota classe 0
        }
        
        agreement = ensemble.get_model_agreement(predictions)
        
        # 3 de 4 modelos concordam (classe 0)
        self.assertAlmostEqual(agreement, 0.75, places=2)

if __name__ == '__main__':
    unittest.main()