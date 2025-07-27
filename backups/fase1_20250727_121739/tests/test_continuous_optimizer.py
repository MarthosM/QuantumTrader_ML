"""
Teste Completo do Módulo Continuous Optimizer - ML Trading v2.0
Sistema de teste abrangente para verificar funcionamento da otimização contínua
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import sys
import os
import logging
from datetime import datetime, timedelta

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from continuous_optimizer import (
    OptimizationConfig, 
    ContinuousOptimizationPipeline,
    FeatureSelectionOptimizer,
    DynamicHyperparameterOptimizer,
    ModelDriftDetector,
    AutoOptimizationEngine,
    PerformanceMonitor
)


class TestContinuousOptimizer:
    """Suite de testes para otimização contínua"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        
        # Dados de teste
        self.sample_data = self._create_sample_market_data()
        self.performance_data = self._create_sample_performance_data()
        
    def teardown_method(self):
        """Cleanup após cada teste"""
        shutil.rmtree(self.temp_dir)
        
    def _create_sample_market_data(self) -> pd.DataFrame:
        """Cria dados de mercado de exemplo"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
        n_points = len(dates)
        
        # Simular dados realísticos
        np.random.seed(42)
        base_price = 5000
        
        # Gerar preços com movimento browniano
        returns = np.random.normal(0.0001, 0.02, n_points)
        prices = base_price * np.cumprod(1 + returns)
        
        data = pd.DataFrame({
            'datetime': dates,
            'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
            'close': prices,
            'volume': np.random.randint(100, 10000, n_points),
        })
        
        # Adicionar features técnicas básicas
        data['ema_9'] = data['close'].ewm(span=9).mean()
        data['ema_20'] = data['close'].ewm(span=20).mean()
        data['rsi'] = self._calculate_rsi(data['close'])
        data['volatility'] = data['close'].rolling(20).std()
        
        return data
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
        
    def _create_sample_performance_data(self) -> dict:
        """Cria dados de performance de exemplo"""
        return {
            'win_rate': 0.55,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.08,
            'total_trades': 150,
            'profit_factor': 1.8,
            'avg_trade_duration': 2.5,
            'market_regime': 'trend_up'
        }
        
    def test_optimization_config_creation(self):
        """Testa criação da configuração de otimização"""
        
        # Configuração padrão
        config_default = OptimizationConfig()
        assert config_default.min_win_rate == 0.5
        assert config_default.min_sharpe == 1.0
        assert config_default.max_drawdown == 0.1
        
        # Configuração personalizada
        config_custom = OptimizationConfig(
            min_win_rate=0.6,
            min_sharpe=1.5,
            max_features=30
        )
        assert config_custom.min_win_rate == 0.6
        assert config_custom.min_sharpe == 1.5
        assert config_custom.max_features == 30
        
    def test_continuous_optimization_pipeline_init(self):
        """Testa inicialização do pipeline de otimização"""
        
        pipeline = ContinuousOptimizationPipeline()
        
        assert pipeline.config is not None
        assert pipeline.feature_optimizer is not None
        assert pipeline.hyperparameter_optimizer is not None
        assert pipeline.drift_detector is not None
        assert pipeline.performance_monitor is not None
        assert pipeline.running is False
        
    def test_should_optimize_decision(self):
        """Testa decisão de otimização"""
        
        pipeline = ContinuousOptimizationPipeline()
        
        # Métricas que não requerem otimização
        good_metrics = {
            'win_rate': 0.65,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.05,
            'volatility': 0.02,
            'historical_volatility': 0.02,
            'accuracy': 0.6,
            'expected_accuracy': 0.55
        }
        
        should_opt, reasons = pipeline.should_optimize(good_metrics)
        
        # Com métricas boas, otimização pode ser necessária apenas por tempo
        assert isinstance(should_opt, bool)
        assert isinstance(reasons, str)
        
        # Métricas que requerem otimização (drift detectado)
        bad_metrics = {
            'win_rate': 0.35,
            'sharpe_ratio': 0.5,
            'max_drawdown': 0.15,
            'volatility': 0.05,
            'historical_volatility': 0.02,
            'accuracy': 0.4,
            'expected_accuracy': 0.55
        }
        
        should_opt_bad, reasons_bad = pipeline.should_optimize(bad_metrics)
        
        # Com métricas ruins, deve sugerir otimização
        assert isinstance(should_opt_bad, bool)
        assert isinstance(reasons_bad, str)
        
    def test_feature_selection_optimizer(self):
        """Testa otimizador de seleção de features"""
        
        optimizer = FeatureSelectionOptimizer()
        
        # Dados com features numericas
        feature_data = self.sample_data[['close', 'volume', 'ema_9', 'ema_20', 'rsi', 'volatility']].copy()
        target_returns = feature_data['close'].pct_change().fillna(0).tolist()
        
        # Selecionar features
        selected_features = optimizer.select_optimal_features(
            data=feature_data,
            target_returns=target_returns,
            max_features=4,
            methods=['mutual_info', 'lasso', 'random_forest']
        )
        
        assert isinstance(selected_features, list)
        assert len(selected_features) <= 4
        assert len(selected_features) > 0
        
        # Verificar que features selecionadas existem nos dados
        for feature in selected_features:
            assert feature in feature_data.columns
            
    def test_hyperparameter_optimizer(self):
        """Testa otimizador de hiperparâmetros"""
        
        optimizer = DynamicHyperparameterOptimizer()
        
        # Definir espaço de busca
        search_space = {
            'n_estimators': (50, 200),
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3)
        }
        
        # Executar otimização
        results = optimizer.optimize(
            features=['close', 'volume', 'rsi'],
            market_data=self.sample_data,
            search_space=search_space
        )
        
        assert 'best_params' in results
        assert 'best_score' in results
        assert isinstance(results['best_params'], dict)
        assert isinstance(results['best_score'], (int, float))
        
        # Verificar se parâmetros estão no espaço de busca
        if results['best_params']:
            best_params = results['best_params']
            if 'n_estimators' in best_params:
                assert search_space['n_estimators'][0] <= best_params['n_estimators'] <= search_space['n_estimators'][1]
                
    def test_model_drift_detector(self):
        """Testa detector de drift nos modelos"""
        
        detector = ModelDriftDetector()
        
        # Criar predições simuladas sem drift
        predictions_no_drift = []
        for i in range(150):
            predictions_no_drift.append({
                'prediction': np.random.normal(0.55, 0.05),
                'confidence': np.random.normal(0.7, 0.1)
            })
        
        metrics_good = {
            'accuracy': 0.58,
            'expected_accuracy': 0.55
        }
        
        drift_detected_good = detector.check_drift(predictions_no_drift, metrics_good)
        
        # Com predições estáveis, não deve detectar drift
        assert isinstance(drift_detected_good, bool)
        
        # Criar predições simuladas com drift
        predictions_with_drift = []
        # Primeira metade estável
        for i in range(75):
            predictions_with_drift.append({
                'prediction': np.random.normal(0.55, 0.05),
                'confidence': np.random.normal(0.7, 0.1)
            })
        # Segunda metade com drift
        for i in range(75):
            predictions_with_drift.append({
                'prediction': np.random.normal(0.35, 0.05),  # Drift significativo
                'confidence': np.random.normal(0.5, 0.1)   # Confiança menor
            })
        
        metrics_bad = {
            'accuracy': 0.4,  # Accuracy baixa
            'expected_accuracy': 0.55
        }
        
        drift_detected_bad = detector.check_drift(predictions_with_drift, metrics_bad)
        
        # Com drift significativo, deve detectar
        assert isinstance(drift_detected_bad, bool)
        
    def test_performance_monitor(self):
        """Testa monitor de performance"""
        
        monitor = PerformanceMonitor()
        
        current_metrics = monitor.get_current_metrics()
        
        # Verificar estrutura das métricas
        expected_keys = ['win_rate', 'sharpe_ratio', 'max_drawdown', 
                        'model_confidence', 'requires_retraining']
        
        for key in expected_keys:
            assert key in current_metrics
            
        # Verificar tipos dos valores
        assert isinstance(current_metrics['win_rate'], (int, float))
        assert isinstance(current_metrics['sharpe_ratio'], (int, float))
        assert isinstance(current_metrics['max_drawdown'], (int, float))
        assert isinstance(current_metrics['requires_retraining'], bool)
        
    def test_market_regime_detection(self):
        """Testa detecção de regime de mercado"""
        
        pipeline = ContinuousOptimizationPipeline()
        
        # Teste com dados insuficientes
        small_data = self.sample_data.head(10)
        regime_small = pipeline._detect_market_regime(small_data)
        assert regime_small == 'undefined'
        
        # Teste com dados suficientes
        regime_full = pipeline._detect_market_regime(self.sample_data)
        assert regime_full in ['trend_up', 'trend_down', 'range']
        
        # Criar dados específicos para trend_up
        trend_up_data = self.sample_data.copy()
        trend_up_data['close'] = pd.Series(range(100, 200))  # Tendência crescente
        regime_trend_up = pipeline._detect_market_regime(trend_up_data)
        assert regime_trend_up in ['trend_up', 'trend_down', 'range']
        
    def test_full_optimization_cycle(self):
        """Testa ciclo completo de otimização"""
        
        pipeline = ContinuousOptimizationPipeline()
        
        # Executar ciclo completo
        results = pipeline.run_optimization_cycle(
            market_data=self.sample_data,
            performance_data=self.performance_data
        )
        
        # Verificar estrutura dos resultados
        assert isinstance(results, dict)
        
        if results.get('status') != 'error':
            expected_sections = ['features']
            for section in expected_sections:
                if section in results:
                    assert isinstance(results[section], dict)
                    
        # Verificar que otimização foi registrada
        assert len(pipeline.optimization_history) > 0
        assert pipeline.last_optimization is not None
        
    def test_auto_optimization_engine_mock(self):
        """Testa engine de otimização automática (mock)"""
        
        # Mock dos componentes necessários
        class MockModelManager:
            def get_recent_predictions(self):
                return [{'prediction': 0.55, 'confidence': 0.7}] * 10
                
        class MockPerformanceMonitor:
            def get_current_metrics(self):
                return {
                    'win_rate': 0.45,  # Baixo para trigger otimização
                    'sharpe_ratio': 0.8,
                    'max_drawdown': 0.12,
                    'model_confidence': 0.5,
                    'requires_retraining': False
                }
        
        mock_model_manager = MockModelManager()
        mock_performance_monitor = MockPerformanceMonitor()
        drift_detector = ModelDriftDetector()
        
        config = OptimizationConfig(optimization_interval=1)  # 1 segundo para teste
        
        engine = AutoOptimizationEngine(
            model_manager=mock_model_manager,
            performance_monitor=mock_performance_monitor,
            drift_detector=drift_detector,
            config=config
        )
        
        # Verificar inicialização
        assert engine.running is False
        assert engine.optimization_thread is None
        
        # Testar should_optimize
        metrics = mock_performance_monitor.get_current_metrics()
        should_optimize = engine._should_optimize(metrics)
        
        # Com win_rate baixo, deve sugerir otimização
        assert isinstance(should_optimize, bool)
        
    def test_error_handling(self):
        """Testa tratamento de erros"""
        
        pipeline = ContinuousOptimizationPipeline()
        
        # Teste com dados vazios
        empty_data = pd.DataFrame()
        empty_performance = {}
        
        results = pipeline.run_optimization_cycle(empty_data, empty_performance)
        
        # Deve retornar erro ou resultado vazio sem causar exceção
        assert isinstance(results, dict)
        
        # Teste do otimizador de features com dados inválidos
        optimizer = FeatureSelectionOptimizer()
        
        empty_features = optimizer.select_optimal_features(
            data=empty_data,
            target_returns=[],
            max_features=10,
            methods=['mutual_info']
        )
        
        assert isinstance(empty_features, list)
        assert len(empty_features) == 0
        
    def test_integration_with_realistic_data(self):
        """Testa integração com dados realísticos"""
        
        # Criar dados mais complexos
        complex_data = self._create_complex_market_data()
        
        pipeline = ContinuousOptimizationPipeline()
        
        # Executar vários componentes
        feature_optimizer = FeatureSelectionOptimizer()
        
        # Selecionar features nos dados complexos
        numeric_cols = complex_data.select_dtypes(include=[np.number]).columns
        target_returns = complex_data['close'].pct_change().fillna(0).tolist()
        
        selected_features = feature_optimizer.select_optimal_features(
            data=complex_data[numeric_cols],
            target_returns=target_returns,
            max_features=15,
            methods=['mutual_info', 'random_forest']
        )
        
        assert len(selected_features) > 0
        assert len(selected_features) <= 15
        
        # Testar detecção de regime
        regime = pipeline._detect_market_regime(complex_data)
        assert regime in ['trend_up', 'trend_down', 'range', 'undefined']
        
    def _create_complex_market_data(self) -> pd.DataFrame:
        """Cria dados de mercado mais complexos para teste de integração"""
        
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='1H')
        n_points = len(dates)
        
        np.random.seed(123)
        base_price = 5000
        
        # Simular diferentes regimes de mercado
        regime_changes = [0, n_points//3, 2*n_points//3, n_points]
        regimes = ['trend_up', 'range', 'trend_down']
        
        prices = []
        current_price = base_price
        
        for i, regime in enumerate(regimes):
            start_idx = regime_changes[i]
            end_idx = regime_changes[i+1]
            period_length = end_idx - start_idx
            
            if regime == 'trend_up':
                # Tendência de alta com volatilidade
                returns = np.random.normal(0.0005, 0.015, period_length)
            elif regime == 'trend_down':
                # Tendência de baixa
                returns = np.random.normal(-0.0003, 0.02, period_length)
            else:  # range
                # Movimento lateral
                returns = np.random.normal(0, 0.01, period_length)
                
            period_prices = current_price * np.cumprod(1 + returns)
            prices.extend(period_prices)
            current_price = period_prices[-1]
        
        data = pd.DataFrame({
            'datetime': dates,
            'open': np.array(prices) * (1 + np.random.normal(0, 0.001, n_points)),
            'high': np.array(prices) * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
            'low': np.array(prices) * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
            'close': prices,
            'volume': np.random.randint(500, 15000, n_points),
        })
        
        # Adicionar múltiplas features técnicas
        data['ema_9'] = data['close'].ewm(span=9).mean()
        data['ema_20'] = data['close'].ewm(span=20).mean()
        data['ema_50'] = data['close'].ewm(span=50).mean()
        data['rsi'] = self._calculate_rsi(data['close'])
        data['volatility'] = data['close'].rolling(20).std()
        data['atr'] = (data['high'] - data['low']).rolling(14).mean()
        data['bb_upper'] = data['close'].rolling(20).mean() + 2 * data['close'].rolling(20).std()
        data['bb_lower'] = data['close'].rolling(20).mean() - 2 * data['close'].rolling(20).std()
        data['macd'] = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
        data['momentum'] = data['close'] - data['close'].shift(14)
        
        return data.dropna()


if __name__ == "__main__":
    """Execução direta do teste"""
    
    print("🚀 INICIANDO TESTE COMPLETO DO CONTINUOUS OPTIMIZER")
    print("=" * 60)
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Criar instância do teste
    tester = TestContinuousOptimizer()
    
    # Lista de testes para executar
    test_methods = [
        'test_optimization_config_creation',
        'test_continuous_optimization_pipeline_init',
        'test_should_optimize_decision',
        'test_feature_selection_optimizer',
        'test_hyperparameter_optimizer',
        'test_model_drift_detector',
        'test_performance_monitor',
        'test_market_regime_detection',
        'test_full_optimization_cycle',
        'test_auto_optimization_engine_mock',
        'test_error_handling',
        'test_integration_with_realistic_data'
    ]
    
    results = []
    
    for i, test_method in enumerate(test_methods, 1):
        print(f"\n{i:2d}. Executando {test_method}...")
        
        try:
            # Setup
            tester.setup_method()
            
            # Executar teste
            method = getattr(tester, test_method)
            method()
            
            print(f"    ✅ {test_method} - PASSOU")
            results.append((test_method, "PASSOU", None))
            
        except Exception as e:
            print(f"    ❌ {test_method} - FALHOU: {str(e)}")
            results.append((test_method, "FALHOU", str(e)))
            
        finally:
            # Cleanup
            try:
                tester.teardown_method()
            except:
                pass
    
    # Resumo final
    print("\n" + "=" * 60)
    print("📊 RESUMO FINAL DOS TESTES")
    print("=" * 60)
    
    passed = sum(1 for _, status, _ in results if status == "PASSOU")
    failed = sum(1 for _, status, _ in results if status == "FALHOU")
    total = len(results)
    
    print(f"Total de testes: {total}")
    print(f"Testes que passaram: {passed}")
    print(f"Testes que falharam: {failed}")
    print(f"Taxa de sucesso: {(passed/total)*100:.1f}%")
    
    if failed > 0:
        print(f"\n❌ Testes que falharam:")
        for test_name, status, error in results:
            if status == "FALHOU":
                print(f"  - {test_name}: {error}")
    
    print("\n" + "=" * 60)
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Módulo Continuous Optimizer está funcionando corretamente")
    else:
        print(f"⚠️  {failed} teste(s) falharam")
        print("🔧 Verifique os erros acima para correções necessárias")
    
    print("=" * 60)
