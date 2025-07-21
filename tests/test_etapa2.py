import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Adicionar diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_structure import TradingDataStructure
from data_pipeline import DataPipeline
from real_time_processor import RealTimeProcessor
from data_loader import DataLoader


@pytest.fixture
def data_structure():
    """Fixture para criar estrutura de dados"""
    ds = TradingDataStructure()
    ds.initialize_structure()
    return ds


@pytest.fixture
def pipeline(data_structure):
    """Fixture para criar pipeline de dados"""
    return DataPipeline(data_structure)


@pytest.fixture
def rt_processor(data_structure):
    """Fixture para criar processador em tempo real"""
    return RealTimeProcessor(data_structure)


@pytest.fixture
def loader():
    """Fixture para criar data loader"""
    return DataLoader("test_data/")


@pytest.fixture
def sample_trades():
    """Fixture para gerar trades de exemplo"""
    trades = []
    base_time = datetime.now().replace(second=0, microsecond=0)
    
    for i in range(100):
        trades.append({
            'timestamp': base_time + timedelta(seconds=i*0.6),
            'price': 5000 + i * 0.1,
            'volume': 10 + (i % 5),
            'trade_type': 2 if i % 2 == 0 else 3
        })
    
    return trades


class TestDataPipeline:
    """Testes para o pipeline de dados"""
    
    def test_historical_processing(self, pipeline, sample_trades):
        """Testa processamento de dados históricos"""
        # Processar
        result = pipeline.process_historical_trades(sample_trades)
        
        # Verificar resultados
        assert 'candles' in result
        assert 'microstructure' in result
        assert len(result['candles']) > 0
        assert len(result['microstructure']) > 0
        
        # Verificar estrutura dos candles
        candles = result['candles']
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            assert col in candles.columns
        
        # Verificar valores OHLC
        for idx in candles.index:
            row = candles.loc[idx]
            assert row['high'] >= row['low']
            assert row['high'] >= row['open']
            assert row['high'] >= row['close']
            assert row['low'] <= row['open']
            assert row['low'] <= row['close']
    
    def test_empty_trades_list(self, pipeline):
        """Testa processamento de lista vazia"""
        result = pipeline.process_historical_trades([])
        assert len(result['candles']) == 0
        assert len(result['microstructure']) == 0
    
    def test_single_trade(self, pipeline):
        """Testa processamento de trade único"""
        single_trade = [{
            'timestamp': datetime.now(),
            'price': 5000,
            'volume': 10,
            'trade_type': 2
        }]
        result = pipeline.process_historical_trades(single_trade)
        assert len(result['candles']) == 1
        assert len(result['microstructure']) == 1
    
    def test_data_alignment(self, pipeline, sample_trades):
        """Testa alinhamento de dados para ML"""
        # Processar trades
        pipeline.process_historical_trades(sample_trades)
        
        # Obter dados alinhados
        aligned = pipeline.get_aligned_data(limit=50)
        
        # Verificar alinhamento
        assert not aligned.empty
        assert len(aligned) <= 50
        
        # Verificar que não há NaN
        assert not aligned.isnull().any().any()
        
        # Verificar colunas esperadas
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_cols:
            assert col in aligned.columns
    
    def test_memory_management(self, pipeline):
        """Testa gerenciamento de memória do pipeline"""
        # Criar muitos trades
        trades = []
        base_time = datetime.now() - timedelta(days=10)
        
        for day in range(10):
            for hour in range(24):
                for minute in range(60):
                    trades.append({
                        'timestamp': base_time + timedelta(days=day, hours=hour, minutes=minute),
                        'price': 5000 + day * 10 + hour * 0.5,
                        'volume': 10,
                        'trade_type': 2
                    })
        
        # Processar
        pipeline.process_historical_trades(trades)
        
        # Verificar tamanho inicial
        initial_size = len(pipeline.data.candles)
        assert initial_size > 0
        
        # Limpar dados antigos
        pipeline.clear_old_data(keep_days=3)
        
        # Verificar que dados foram removidos
        final_size = len(pipeline.data.candles)
        assert final_size < initial_size
        
        # Verificar que dados recentes permanecem
        if not pipeline.data.candles.empty:
            oldest_candle = pipeline.data.candles.index.min()
            assert oldest_candle > datetime.now() - timedelta(days=4)


class TestRealTimeProcessor:
    """Testes para o processador em tempo real"""
    
    def test_single_trade_processing(self, rt_processor):
        """Testa processamento de trade único"""
        trade = {
            'timestamp': datetime.now(),
            'price': 5000.0,
            'volume': 10,
            'trade_type': 2
        }
        
        # Processar trade
        success = rt_processor.process_trade(trade)
        assert success is True
        
        # Verificar estado
        state = rt_processor.get_current_state()
        assert state['last_candle_time'] is not None
        assert state['current_candle']['trades_count'] == 1
        assert state['current_candle']['volume'] == 10
    
    @pytest.mark.parametrize("invalid_trade", [
        {'timestamp': datetime.now(), 'price': -100, 'volume': 10, 'trade_type': 2},
        {'timestamp': datetime.now(), 'price': 0, 'volume': 10, 'trade_type': 2},
        {'timestamp': datetime.now(), 'price': 5000, 'volume': -5, 'trade_type': 2},
        {'timestamp': datetime.now(), 'price': 5000, 'volume': 10, 'trade_type': 5},
        {'timestamp': datetime.now(), 'volume': 10, 'trade_type': 2},  # missing price
        {'price': 5000, 'volume': 10, 'trade_type': 2},  # missing timestamp
    ])
    def test_invalid_trade_validation(self, rt_processor, invalid_trade):
        """Testa validação de trades inválidos"""
        success = rt_processor.process_trade(invalid_trade)
        assert success is False
    
    def test_candle_formation(self, rt_processor, data_structure):
        """Testa formação de candles com múltiplos trades"""
        base_time = datetime.now().replace(second=0, microsecond=0)
        
        # Criar trades para 2 candles completos
        for minute in range(2):
            for second in range(0, 60, 10):
                trade = {
                    'timestamp': base_time + timedelta(minutes=minute, seconds=second),
                    'price': 5000 + minute * 10 + second * 0.1,
                    'volume': 10,
                    'trade_type': 2 if second % 20 == 0 else 3
                }
                rt_processor.process_trade(trade)
        
        # Forçar fechamento do último candle
        rt_processor.force_close_candle()
        
        # Verificar candles formados
        candles = data_structure.candles
        assert len(candles) == 2
        
        # Verificar dados de microestrutura
        micro = data_structure.microstructure
        assert len(micro) == 2
        
        # Verificar integridade dos dados
        for idx in candles.index:
            candle = candles.loc[idx]
            assert candle['volume'] > 0
            assert candle['high'] > 0
            assert candle['low'] > 0
    
    def test_state_tracking(self, rt_processor):
        """Testa rastreamento de estado"""
        # Estado inicial
        initial_state = rt_processor.get_current_state()
        assert initial_state['last_candle_time'] is None
        assert initial_state['candles_count'] == 0
        
        # Processar alguns trades
        base_time = datetime.now()
        for i in range(5):
            trade = {
                'timestamp': base_time + timedelta(seconds=i),
                'price': 5000 + i,
                'volume': 10,
                'trade_type': 2
            }
            rt_processor.process_trade(trade)
        
        # Verificar estado atualizado
        updated_state = rt_processor.get_current_state()
        assert updated_state['last_candle_time'] is not None
        assert updated_state['current_candle']['trades_count'] == 5
        assert updated_state['current_candle']['volume'] == 50


class TestDataLoader:
    """Testes para o data loader"""
    
    def test_sample_data_generation(self, loader):
        """Testa geração de dados de exemplo"""
        # Gerar dados
        sample_data = loader.generate_sample_data(n_candles=100)
        
        # Verificar estrutura
        assert 'candles' in sample_data
        assert 'microstructure' in sample_data
        
        candles = sample_data['candles']
        micro = sample_data['microstructure']
        
        # Verificar tamanho
        assert len(candles) == 100
        assert len(micro) == 100
        
        # Verificar índices alinhados
        pd.testing.assert_index_equal(candles.index, micro.index)
        
        # Verificar valores realistas
        assert (candles['close'] > 0).all()
        assert (candles['volume'] >= 0).all()
        assert (micro['buy_volume'] >= 0).all()
        assert (micro['sell_volume'] >= 0).all()
    
    def test_available_data_listing(self, loader):
        """Testa listagem de dados disponíveis"""
        available = loader.get_available_data()
        
        assert isinstance(available, dict)
        assert 'trades' in available
        assert 'candles' in available
        assert 'orderbook' in available
        assert 'cache' in available
        
        for key in available:
            assert isinstance(available[key], list)


class TestMicrostructure:
    """Testes específicos para cálculo de microestrutura"""
    
    def test_microstructure_calculation(self, pipeline):
        """Testa cálculo correto da microestrutura"""
        trades = []
        base_time = datetime.now().replace(second=0, microsecond=0)
        
        # Criar 10 trades: 6 buys, 4 sells
        for i in range(10):
            trades.append({
                'timestamp': base_time + timedelta(seconds=i),
                'price': 5000,
                'volume': 10 if i < 6 else 15,  # Buys: 60 volume, Sells: 60 volume
                'trade_type': 2 if i < 6 else 3
            })
        
        # Processar
        result = pipeline.process_historical_trades(trades)
        micro = result['microstructure']
        
        # Verificar cálculos
        assert len(micro) == 1  # Todos no mesmo minuto
        row = micro.iloc[0]
        
        assert row['buy_volume'] == 60
        assert row['sell_volume'] == 60
        assert row['buy_trades'] == 6
        assert row['sell_trades'] == 4
        assert row['imbalance'] == 0  # 60 - 60 = 0
        assert row['trade_count'] == 10
    
    def test_imbalance_calculation(self, pipeline):
        """Testa cálculo de imbalance"""
        trades = [
            {'timestamp': datetime.now(), 'price': 5000, 'volume': 100, 'trade_type': 2},
            {'timestamp': datetime.now(), 'price': 5000, 'volume': 30, 'trade_type': 3},
        ]
        
        result = pipeline.process_historical_trades(trades)
        micro = result['microstructure']
        
        assert len(micro) == 1
        assert micro.iloc[0]['imbalance'] == 70  # 100 - 30


class TestEdgeCases:
    """Testes para casos extremos"""
    
    def test_same_timestamp_trades(self, pipeline):
        """Testa trades com mesmo timestamp"""
        same_time = datetime.now()
        same_time_trades = [
            {'timestamp': same_time, 'price': 5000, 'volume': 10, 'trade_type': 2},
            {'timestamp': same_time, 'price': 5001, 'volume': 20, 'trade_type': 3},
            {'timestamp': same_time, 'price': 5002, 'volume': 30, 'trade_type': 2},
        ]
        result = pipeline.process_historical_trades(same_time_trades)
        
        # Deve criar apenas 1 candle
        assert len(result['candles']) == 1
        candle = result['candles'].iloc[0]
        assert candle['open'] == 5000
        assert candle['close'] == 5002
        assert candle['high'] == 5002
        assert candle['low'] == 5000
        assert candle['volume'] == 60
    
    def test_large_price_movements(self, pipeline):
        """Testa movimentos extremos de preço"""
        trades = [
            {'timestamp': datetime.now(), 'price': 5000, 'volume': 10, 'trade_type': 2},
            {'timestamp': datetime.now() + timedelta(seconds=1), 'price': 10000, 'volume': 10, 'trade_type': 2},
            {'timestamp': datetime.now() + timedelta(seconds=2), 'price': 2500, 'volume': 10, 'trade_type': 3},
        ]
        
        result = pipeline.process_historical_trades(trades)
        candle = result['candles'].iloc[0]
        
        assert candle['high'] == 10000
        assert candle['low'] == 2500
        assert candle['open'] == 5000
        assert candle['close'] == 2500


if __name__ == '__main__':
    pytest.main([__file__, '-v'])