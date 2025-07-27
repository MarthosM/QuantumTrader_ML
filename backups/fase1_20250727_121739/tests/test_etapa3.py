import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Adicionar diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_structure import TradingDataStructure
from technical_indicators import TechnicalIndicators
from ml_features import MLFeatures
from feature_engine import FeatureEngine


@pytest.fixture
def sample_candles():
    """Cria candles de teste com movimento realista"""
    n = 200
    dates = pd.date_range(end=datetime.now(), periods=n, freq='1min')
    
    # Gerar preço com tendência e ruído
    np.random.seed(42)
    trend = np.linspace(5000, 5100, n)
    noise = np.random.normal(0, 20, n)
    prices = trend + noise
    
    data = {
        'open': prices + np.random.normal(0, 5, n),
        'high': prices + abs(np.random.normal(10, 5, n)),
        'low': prices - abs(np.random.normal(10, 5, n)),
        'close': prices,
        'volume': np.random.randint(10, 100, n).astype(float)
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Garantir consistência OHLC
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def sample_microstructure(sample_candles):
    """Cria dados de microestrutura alinhados com candles"""
    n = len(sample_candles)
    
    data = {
        'buy_volume': np.random.uniform(20, 80, n),
        'sell_volume': np.random.uniform(20, 80, n),
        'buy_trades': np.random.randint(5, 20, n),
        'sell_trades': np.random.randint(5, 20, n),
        'avg_trade_size': np.random.uniform(2, 10, n)
    }
    
    df = pd.DataFrame(data, index=sample_candles.index)
    df['imbalance'] = df['buy_volume'] - df['sell_volume']
    
    return df


@pytest.fixture
def data_structure(sample_candles, sample_microstructure):
    """Cria estrutura de dados completa"""
    ds = TradingDataStructure()
    ds.initialize_structure()
    ds.candles = sample_candles
    ds.microstructure = sample_microstructure
    return ds


class TestTechnicalIndicators:
    """Testes para indicadores técnicos"""
    
    def test_calculate_all_indicators(self, sample_candles):
        """Testa cálculo de todos indicadores"""
        tech = TechnicalIndicators()
        indicators = tech.calculate_all(sample_candles)
        
        # Verificar que indicadores foram calculados
        assert not indicators.empty
        assert len(indicators) == len(sample_candles)
        
        # Verificar indicadores essenciais
        essential_indicators = ['ema_20', 'ema_50', 'rsi', 'macd', 
                              'bb_upper_20', 'bb_lower_20', 'atr']
        
        for indicator in essential_indicators:
            assert indicator in indicators.columns
    
    def test_moving_averages(self, sample_candles):
        """Testa cálculo de médias móveis"""
        tech = TechnicalIndicators()
        indicators = pd.DataFrame(index=sample_candles.index)
        
        tech._calculate_moving_averages(sample_candles, indicators)
        
        # Verificar EMAs
        assert 'ema_9' in indicators.columns
        assert 'ema_20' in indicators.columns
        assert 'ema_50' in indicators.columns
        
        # Verificar valores
        assert not indicators['ema_20'].iloc[30:].isna().any()
        
        # EMA deve estar próxima ao preço
        price_mean = sample_candles['close'].mean()
        ema_mean = indicators['ema_20'].mean()
        assert abs(price_mean - ema_mean) < price_mean * 0.1
    
    def test_rsi_calculation(self, sample_candles):
        """Testa cálculo do RSI"""
        tech = TechnicalIndicators()
        indicators = pd.DataFrame(index=sample_candles.index)
        
        tech._calculate_rsi(sample_candles, indicators)
        
        assert 'rsi' in indicators.columns
        
        # RSI deve estar entre 0 e 100
        rsi_values = indicators['rsi'].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
        
        # Verificar sinais
        assert 'rsi_oversold' in indicators.columns
        assert 'rsi_overbought' in indicators.columns
    
    def test_macd_calculation(self, sample_candles):
        """Testa cálculo do MACD"""
        tech = TechnicalIndicators()
        indicators = pd.DataFrame(index=sample_candles.index)
        
        tech._calculate_macd(sample_candles, indicators)
        
        # Verificar componentes
        assert 'macd' in indicators.columns
        assert 'macd_signal' in indicators.columns
        assert 'macd_hist' in indicators.columns
        
        # Verificar crossovers
        assert 'macd_bullish' in indicators.columns
        assert 'macd_bearish' in indicators.columns
        
        # Histograma deve ser diferença entre MACD e signal
        hist_calc = indicators['macd'] - indicators['macd_signal']
        pd.testing.assert_series_equal(
            indicators['macd_hist'].dropna(),
            hist_calc.dropna(),
            check_names=False
        )
    
    def test_bollinger_bands(self, sample_candles):
        """Testa cálculo das Bollinger Bands"""
        tech = TechnicalIndicators()
        indicators = pd.DataFrame(index=sample_candles.index)
        
        tech._calculate_bollinger_bands(sample_candles, indicators)
        
        # Verificar bandas
        assert 'bb_upper_20' in indicators.columns
        assert 'bb_middle_20' in indicators.columns
        assert 'bb_lower_20' in indicators.columns
        assert 'bb_width_20' in indicators.columns
        assert 'bb_position_20' in indicators.columns
        
        # Verificar ordem das bandas
        valid_idx = ~indicators['bb_upper_20'].isna()
        assert (indicators.loc[valid_idx, 'bb_upper_20'] >= 
                indicators.loc[valid_idx, 'bb_middle_20']).all()
        assert (indicators.loc[valid_idx, 'bb_middle_20'] >= 
                indicators.loc[valid_idx, 'bb_lower_20']).all()
    
    @pytest.mark.parametrize("indicator_func,expected_cols", [
        ('_calculate_stochastic', ['stoch_k', 'stoch_d', 'slow_k', 'slow_d']),
        ('_calculate_atr', ['atr', 'atr_pct']),
        ('_calculate_adx', ['adx', 'plus_di', 'minus_di'])
    ])
    def test_other_indicators(self, sample_candles, indicator_func, expected_cols):
        """Testa outros indicadores"""
        tech = TechnicalIndicators()
        indicators = pd.DataFrame(index=sample_candles.index)
        
        # Calcular ATR primeiro se necessário para ADX
        if indicator_func == '_calculate_adx':
            tech._calculate_atr(sample_candles, indicators)
        
        getattr(tech, indicator_func)(sample_candles, indicators)
        
        for col in expected_cols:
            assert col in indicators.columns


class TestMLFeatures:
    """Testes para features de ML"""
    
    def test_calculate_all_features(self, sample_candles, sample_microstructure):
        """Testa cálculo de todas as features ML"""
        ml = MLFeatures()
        features = ml.calculate_all(sample_candles, sample_microstructure)
        
        assert not features.empty
        assert len(features) == len(sample_candles)
        
        # Verificar categorias de features
        feature_cols = features.columns.tolist()
        
        # Momentum
        assert any('momentum_' in col for col in feature_cols)
        assert any('roc_' in col for col in feature_cols)
        
        # Volatilidade
        assert any('volatility_' in col for col in feature_cols)
        assert any('range_pct_' in col for col in feature_cols)
        
        # Microestrutura
        assert 'buy_pressure' in feature_cols
        assert 'volume_imbalance' in feature_cols
    
    def test_momentum_features(self, sample_candles):
        """Testa features de momentum"""
        ml = MLFeatures()
        features = pd.DataFrame(index=sample_candles.index)
        
        ml._calculate_momentum_features(sample_candles, features)
        
        # Verificar períodos
        for period in [1, 3, 5, 10, 15, 20]:
            assert f'momentum_{period}' in features.columns
            assert f'momentum_pct_{period}' in features.columns
            assert f'roc_{period}' in features.columns
        
        # Verificar aceleração
        assert 'momentum_acc' in features.columns
    
    def test_volatility_features(self, sample_candles):
        """Testa features de volatilidade"""
        ml = MLFeatures()
        features = pd.DataFrame(index=sample_candles.index)
        
        ml._calculate_volatility_features(sample_candles, features)
        
        # Verificar períodos
        for period in [5, 10, 20, 50]:
            assert f'volatility_{period}' in features.columns
            assert f'high_low_range_{period}' in features.columns
            assert f'range_pct_{period}' in features.columns
        
        # Verificar volatilidade realizada
        assert 'realized_vol_5' in features.columns
        assert 'realized_vol_20' in features.columns
        
        # Volatilidade deve ser positiva
        vol_cols = [col for col in features.columns if 'volatility' in col]
        for col in vol_cols:
            assert (features[col].dropna() >= 0).all()
    
    def test_microstructure_features(self, sample_candles, sample_microstructure):
        """Testa features de microestrutura"""
        ml = MLFeatures()
        features = pd.DataFrame(index=sample_candles.index)
        
        ml._calculate_microstructure_features(sample_microstructure, features)
        
        # Verificar features básicas
        assert 'buy_pressure' in features.columns
        assert 'volume_imbalance' in features.columns
        assert 'trade_imbalance' in features.columns
        
        # Buy pressure deve estar entre 0 e 1
        assert (features['buy_pressure'] >= 0).all()
        assert (features['buy_pressure'] <= 1).all()
    
    def test_composite_features(self, sample_candles):
        """Testa features compostas"""
        # Criar indicadores necessários
        tech = TechnicalIndicators()
        indicators = tech.calculate_all(sample_candles)
        
        ml = MLFeatures()
        features = pd.DataFrame(index=sample_candles.index)
        
        ml._calculate_composite_features(sample_candles, indicators, features)
        
        # Verificar features compostas
        if 'ema_20' in indicators.columns:
            assert 'price_to_ema20' in features.columns
        
        if 'rsi' in indicators.columns:
            assert 'rsi_ma_5' in features.columns
            assert 'rsi_distance_50' in features.columns
        
        if all(col in indicators.columns for col in ['macd', 'macd_signal']):
            assert 'macd_divergence' in features.columns
    
    def test_pattern_features(self, sample_candles):
        """Testa features de padrões"""
        ml = MLFeatures()
        features = pd.DataFrame(index=sample_candles.index)
        
        ml._calculate_pattern_features(sample_candles, features)
        
        # Verificar padrões de candles
        pattern_features = ['doji', 'hammer', 'bullish_engulfing', 'bearish_engulfing']
        for pattern in pattern_features:
            assert pattern in features.columns
            # Padrões devem ser binários
            assert set(features[pattern].dropna().unique()).issubset({0, 1})
        
        # Verificar níveis de suporte/resistência
        assert 'pivot_point' in features.columns
        assert 'distance_to_high' in features.columns
        assert 'distance_to_low' in features.columns


class TestFeatureEngine:
    """Testes para o motor de features"""
    
    def test_calculate_all(self, data_structure):
        """Testa cálculo completo de features"""
        engine = FeatureEngine()
        result = engine.calculate(data_structure)
        
        # Verificar estrutura do resultado
        assert 'indicators' in result
        assert 'features' in result
        assert 'model_ready' in result
        assert 'all' in result
        
        # Verificar que dados foram calculados
        assert not result['indicators'].empty
        assert not result['features'].empty
        assert not result['model_ready'].empty
    
    def test_model_specific_features(self, data_structure):
        """Testa preparação de features específicas do modelo"""
        model_features = ['close', 'volume', 'ema_20', 'rsi', 'momentum_5', 'buy_pressure']
        engine = FeatureEngine(model_features)
        
        result = engine.calculate(data_structure)
        model_ready = result['model_ready']
        
        # Verificar que apenas features solicitadas estão presentes
        assert list(model_ready.columns) == model_features
        
        # Verificar que não há NaN
        assert not model_ready.isnull().any().any()
    
    def test_cache_mechanism(self, data_structure):
        """Testa mecanismo de cache"""
        engine = FeatureEngine()
        
        # Primeira chamada
        result1 = engine.calculate(data_structure)
        
        # Segunda chamada sem mudanças
        result2 = engine.calculate(data_structure)
        
        # Devem ser idênticos (cache usado)
        pd.testing.assert_frame_equal(result1['model_ready'], result2['model_ready'])
        
        # Modificar dados
        data_structure.candles.iloc[-1, 0] += 100
        
        # Terceira chamada com dados modificados
        result3 = engine.calculate(data_structure)
        
        # Deve ser diferente (cache invalidado)
        assert not result1['model_ready'].equals(result3['model_ready'])
    
    def test_parallel_processing(self, data_structure):
        """Testa processamento paralelo"""
        engine = FeatureEngine()
        engine.parallel_processing = True
        
        result = engine.calculate(data_structure)
        
        assert not result['indicators'].empty
        assert not result['features'].empty
    
    def test_feature_dependencies(self):
        """Testa resolução de dependências de features"""
        engine = FeatureEngine()
        
        # Features que dependem de outras
        model_features = ['price_to_ema20', 'macd_divergence']
        required = engine.get_required_features_for_models(model_features)
        
        # Deve incluir dependências
        assert 'close' in required
        assert 'ema_20' in required
        assert 'macd' in required
        assert 'macd_signal' in required
    
    def test_specific_feature_calculation(self, data_structure):
        """Testa cálculo de features específicas"""
        engine = FeatureEngine()
        
        # Calcular apenas algumas features
        specific_features = ['ema_20', 'rsi', 'momentum_5']
        result = engine.calculate_specific_features(data_structure, specific_features)
        
        assert list(result.columns) == specific_features
        assert len(result) > 0
    
    def test_feature_statistics(self, data_structure):
        """Testa estatísticas de features"""
        engine = FeatureEngine()
        result = engine.calculate(data_structure)
        
        stats = engine.get_feature_statistics(result['model_ready'])
        
        assert not stats.empty
        assert all(col in stats.columns for col in ['mean', 'std', 'min', 'max'])
    
    def test_empty_data_handling(self):
        """Testa tratamento de dados vazios"""
        empty_data = TradingDataStructure()
        empty_data.initialize_structure()
        
        engine = FeatureEngine()
        result = engine.calculate(empty_data)
        
        assert result['model_ready'].empty
    
    def test_missing_features_handling(self, data_structure):
        """Testa tratamento de features ausentes"""
        # Solicitar features que não existem
        model_features = ['close', 'feature_inexistente', 'ema_20']
        engine = FeatureEngine(model_features)
        
        result = engine.calculate(data_structure)
        model_ready = result['model_ready']
        
        # Deve ter todas as colunas solicitadas
        assert all(f in model_ready.columns for f in model_features)
        
        # Feature inexistente deve ter valores default (0)
        assert (model_ready['feature_inexistente'] == 0).all()


class TestIntegration:
    """Testes de integração entre componentes"""
    
    def test_full_pipeline(self, data_structure):
        """Testa pipeline completo de features"""
        # Configurar engine com features específicas
        model_features = [
            'close', 'volume', 'ema_20', 'ema_50',
            'rsi', 'macd', 'macd_signal',
            'momentum_5', 'momentum_10',
            'volatility_20', 'buy_pressure'
        ]
        
        engine = FeatureEngine(model_features)
        result = engine.calculate(data_structure)
        
        # Verificar resultado
        model_ready = result['model_ready']
        
        assert model_ready.shape[1] == len(model_features)
        assert model_ready.shape[0] > 100  # Deve ter dados após warm-up
        
        # Verificar integridade dos dados
        assert not model_ready.isnull().any().any()
        assert not (model_ready == np.inf).any().any()
        assert not (model_ready == -np.inf).any().any()
    
    def test_feature_consistency(self, sample_candles):
        """Testa consistência entre diferentes métodos de cálculo"""
        # Calcular via engine
        data = TradingDataStructure()
        data.candles = sample_candles
        
        engine = FeatureEngine()
        result1 = engine.calculate(data)
        
        # Calcular diretamente
        tech = TechnicalIndicators()
        indicators = tech.calculate_all(sample_candles)
        
        # Comparar indicadores comuns
        common_cols = set(result1['indicators'].columns) & set(indicators.columns)
        for col in common_cols:
            pd.testing.assert_series_equal(
                result1['indicators'][col],
                indicators[col],
                check_names=False
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])