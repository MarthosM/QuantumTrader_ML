"""
Testes para CircularBuffer e suas especializações
Valida funcionalidade, thread-safety e performance
"""

import pytest
import numpy as np
import pandas as pd
import threading
import time
from datetime import datetime, timedelta
from collections import deque
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.buffers.circular_buffer import CircularBuffer, CandleBuffer, BookBuffer, TradeBuffer


class TestCircularBuffer:
    """Testes para CircularBuffer base"""
    
    def test_init(self):
        """Testa inicialização do buffer"""
        buffer = CircularBuffer(max_size=100, name="test")
        
        assert buffer.max_size == 100
        assert buffer.name == "test"
        assert buffer.size() == 0
        assert buffer.is_empty()
        assert not buffer.is_full()
    
    def test_add_items(self):
        """Testa adição de itens"""
        buffer = CircularBuffer(max_size=5)
        
        # Adicionar itens individuais
        for i in range(3):
            assert buffer.add(i)
        
        assert buffer.size() == 3
        assert list(buffer.data) == [0, 1, 2]
    
    def test_max_size_limit(self):
        """Testa limite de tamanho máximo"""
        buffer = CircularBuffer(max_size=3)
        
        # Adicionar mais itens que o limite
        for i in range(5):
            buffer.add(i)
        
        # Deve manter apenas os últimos 3
        assert buffer.size() == 3
        assert list(buffer.data) == [2, 3, 4]
        assert buffer.is_full()
        
        # Verificar estatísticas
        stats = buffer.get_stats()
        assert stats["total_added"] == 5
        assert stats["total_dropped"] == 2
    
    def test_add_batch(self):
        """Testa adição em lote"""
        buffer = CircularBuffer(max_size=10)
        
        items = [1, 2, 3, 4, 5]
        added = buffer.add_batch(items)
        
        assert added == 5
        assert buffer.size() == 5
        assert list(buffer.data) == items
    
    def test_get_last_n(self):
        """Testa recuperação dos últimos N itens"""
        buffer = CircularBuffer(max_size=10)
        buffer.add_batch(list(range(10)))
        
        # Pegar últimos 3
        last_3 = buffer.get_last_n(3)
        assert last_3 == [7, 8, 9]
        
        # Pegar mais que o tamanho
        all_items = buffer.get_last_n(20)
        assert len(all_items) == 10
        assert all_items == list(range(10))
    
    def test_get_dataframe(self):
        """Testa conversão para DataFrame"""
        buffer = CircularBuffer(max_size=5)
        
        # Adicionar dicts
        for i in range(3):
            buffer.add({"id": i, "value": i * 10})
        
        df = buffer.get_dataframe()
        assert len(df) == 3
        assert list(df.columns) == ["id", "value"]
        assert df["value"].tolist() == [0, 10, 20]
        
        # Testar com colunas específicas
        df = buffer.get_dataframe(columns=["value"])
        assert list(df.columns) == ["value"]
    
    def test_dataframe_with_lists(self):
        """Testa DataFrame com listas/tuplas"""
        buffer = CircularBuffer(max_size=5)
        
        # Adicionar listas
        buffer.add_batch([[1, 2], [3, 4], [5, 6]])
        
        df = buffer.get_dataframe(columns=["a", "b"])
        assert len(df) == 3
        assert df["a"].tolist() == [1, 3, 5]
        assert df["b"].tolist() == [2, 4, 6]
    
    def test_clear(self):
        """Testa limpeza do buffer"""
        buffer = CircularBuffer(max_size=5)
        buffer.add_batch([1, 2, 3])
        
        assert buffer.size() == 3
        buffer.clear()
        assert buffer.size() == 0
        assert buffer.is_empty()
    
    def test_thread_safety(self):
        """Testa thread-safety do buffer"""
        buffer = CircularBuffer(max_size=1000)
        errors = []
        
        def add_items(start, count):
            try:
                for i in range(start, start + count):
                    buffer.add(i)
            except Exception as e:
                errors.append(e)
        
        # Criar múltiplas threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=add_items, args=(i * 100, 100))
            threads.append(t)
            t.start()
        
        # Aguardar conclusão
        for t in threads:
            t.join()
        
        # Verificar se não houve erros
        assert len(errors) == 0
        assert buffer.size() <= buffer.max_size
    
    def test_concurrent_read_write(self):
        """Testa leitura e escrita concorrentes"""
        buffer = CircularBuffer(max_size=100)
        stop_flag = threading.Event()
        errors = []
        
        def writer():
            i = 0
            while not stop_flag.is_set():
                try:
                    buffer.add(i)
                    i += 1
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(e)
        
        def reader():
            while not stop_flag.is_set():
                try:
                    _ = buffer.get_last_n(10)
                    _ = buffer.get_dataframe()
                    _ = buffer.size()
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(e)
        
        # Iniciar threads
        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader) for _ in range(3)]
        
        writer_thread.start()
        for t in reader_threads:
            t.start()
        
        # Executar por 0.5 segundos
        time.sleep(0.5)
        stop_flag.set()
        
        # Aguardar conclusão
        writer_thread.join()
        for t in reader_threads:
            t.join()
        
        # Verificar se não houve erros
        assert len(errors) == 0


class TestCandleBuffer:
    """Testes para CandleBuffer especializado"""
    
    def test_add_candle(self):
        """Testa adição de candle"""
        buffer = CandleBuffer(max_size=10)
        
        timestamp = datetime.now()
        success = buffer.add_candle(
            timestamp=timestamp,
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000.0
        )
        
        assert success
        assert buffer.size() == 1
        
        # Verificar dados
        df = buffer.get_dataframe()
        assert len(df) == 1
        assert df["close"].iloc[0] == 103.0
    
    def test_calculate_returns(self):
        """Testa cálculo de retornos"""
        buffer = CandleBuffer(max_size=20)
        
        # Adicionar candles com preços crescentes
        base_time = datetime.now()
        prices = [100, 102, 101, 103, 105, 104, 106]
        
        for i, price in enumerate(prices):
            buffer.add_candle(
                timestamp=base_time + timedelta(minutes=i),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000
            )
        
        # Calcular retornos de 1 período
        returns = buffer.calculate_returns(periods=1)
        assert len(returns) == 6  # 7 candles - 1 período
        
        # Verificar primeiro retorno: (102 - 100) / 100 = 0.02
        assert abs(returns[0] - 0.02) < 0.0001
    
    def test_calculate_volatility(self):
        """Testa cálculo de volatilidade"""
        buffer = CandleBuffer(max_size=50)
        
        # Adicionar candles com preços variáveis
        base_time = datetime.now()
        np.random.seed(42)
        
        for i in range(30):
            price = 100 + np.random.randn() * 5
            buffer.add_candle(
                timestamp=base_time + timedelta(minutes=i),
                open=price,
                high=price + 2,
                low=price - 2,
                close=price,
                volume=1000
            )
        
        # Calcular volatilidade
        vol = buffer.calculate_volatility(periods=20)
        assert vol > 0
        assert vol < 1  # Volatilidade razoável
    
    def test_get_ohlc_stats(self):
        """Testa estatísticas OHLC"""
        buffer = CandleBuffer(max_size=10)
        
        # Adicionar alguns candles
        base_time = datetime.now()
        for i in range(5):
            buffer.add_candle(
                timestamp=base_time + timedelta(minutes=i),
                open=100 + i,
                high=105 + i,
                low=95 + i,
                close=102 + i,
                volume=1000 * (i + 1)
            )
        
        stats = buffer.get_ohlc_stats()
        
        assert stats["last_close"] == 106  # 102 + 4
        assert stats["high_period"] == 109  # 105 + 4
        assert stats["low_period"] == 95   # Mínimo
        assert stats["avg_volume"] == 3000  # (1000 + 2000 + 3000 + 4000 + 5000) / 5


class TestBookBuffer:
    """Testes para BookBuffer especializado"""
    
    def test_add_snapshot(self):
        """Testa adição de snapshot do book"""
        buffer = BookBuffer(max_size=10, levels=5)
        
        timestamp = datetime.now()
        success = buffer.add_snapshot(
            timestamp=timestamp,
            bid_prices=[100, 99.5, 99, 98.5, 98],
            bid_volumes=[10, 20, 30, 40, 50],
            ask_prices=[101, 101.5, 102, 102.5, 103],
            ask_volumes=[15, 25, 35, 45, 55]
        )
        
        assert success
        assert buffer.size() == 1
    
    def test_calculate_spread(self):
        """Testa cálculo de spread"""
        buffer = BookBuffer(max_size=10)
        
        buffer.add_snapshot(
            timestamp=datetime.now(),
            bid_prices=[100, 99.5],
            bid_volumes=[10, 20],
            ask_prices=[101, 101.5],
            ask_volumes=[15, 25]
        )
        
        spread = buffer.calculate_spread()
        assert spread == 1.0  # 101 - 100
    
    def test_calculate_imbalance(self):
        """Testa cálculo de order flow imbalance"""
        buffer = BookBuffer(max_size=10)
        
        # Adicionar snapshots com volumes desbalanceados
        for i in range(5):
            buffer.add_snapshot(
                timestamp=datetime.now() + timedelta(seconds=i),
                bid_prices=[100, 99.5],
                bid_volumes=[100, 50],  # Total: 150
                ask_prices=[101, 101.5],
                ask_volumes=[50, 25]    # Total: 75
            )
        
        imbalance = buffer.calculate_imbalance(periods=5)
        # (750 - 375) / (750 + 375) = 375 / 1125 ≈ 0.333
        assert abs(imbalance - 0.333) < 0.01
    
    def test_get_book_depth(self):
        """Testa profundidade do book"""
        buffer = BookBuffer(max_size=10)
        
        buffer.add_snapshot(
            timestamp=datetime.now(),
            bid_prices=[100, 99.5, 99],
            bid_volumes=[10, 20, 30],
            ask_prices=[101, 101.5, 102],
            ask_volumes=[15, 25, 35]
        )
        
        depth = buffer.get_book_depth()
        
        assert depth["bid_depth"] == 60   # 10 + 20 + 30
        assert depth["ask_depth"] == 75   # 15 + 25 + 35
        assert depth["total_depth"] == 135
        assert depth["spread"] == 1.0
        assert depth["mid_price"] == 100.5


class TestTradeBuffer:
    """Testes para TradeBuffer especializado"""
    
    def test_add_trade(self):
        """Testa adição de trade"""
        buffer = TradeBuffer(max_size=100)
        
        success = buffer.add_trade(
            timestamp=datetime.now(),
            price=100.5,
            volume=10,
            side="buy",
            aggressor="buyer",
            trader_id="trader_1"
        )
        
        assert success
        assert buffer.size() == 1
    
    def test_calculate_vwap(self):
        """Testa cálculo de VWAP"""
        buffer = TradeBuffer(max_size=100)
        
        # Adicionar trades
        trades = [
            (100, 10),  # price * volume = 1000
            (101, 20),  # 2020
            (102, 30),  # 3060
            (99, 40),   # 3960
        ]
        
        for price, volume in trades:
            buffer.add_trade(
                timestamp=datetime.now(),
                price=price,
                volume=volume,
                side="buy",
                aggressor="buyer"
            )
        
        vwap = buffer.calculate_vwap(periods=4)
        # (1000 + 2020 + 3060 + 3960) / (10 + 20 + 30 + 40)
        # = 10040 / 100 = 100.4
        assert abs(vwap - 100.4) < 0.01
    
    def test_calculate_trade_intensity(self):
        """Testa cálculo de intensidade de negociação"""
        buffer = TradeBuffer(max_size=100)
        
        # Adicionar trades em intervalos de 1 segundo
        base_time = datetime.now()
        for i in range(10):
            buffer.add_trade(
                timestamp=base_time + timedelta(seconds=i),
                price=100,
                volume=10,
                side="buy",
                aggressor="buyer"
            )
        
        # Intensidade em janela de 60 segundos
        intensity = buffer.calculate_trade_intensity(time_window_seconds=60)
        
        # 10 trades em 9 segundos ≈ 1.11 trades/segundo
        assert intensity > 1.0
        assert intensity < 2.0
    
    def test_get_aggressor_ratio(self):
        """Testa proporção de agressores"""
        buffer = TradeBuffer(max_size=100)
        
        # Adicionar trades com diferentes agressores
        for i in range(10):
            aggressor = "buyer" if i < 7 else "seller"
            buffer.add_trade(
                timestamp=datetime.now(),
                price=100,
                volume=10,
                side="buy",
                aggressor=aggressor
            )
        
        ratio = buffer.get_aggressor_ratio()
        
        assert ratio["buyer_aggressor_count"] == 7
        assert ratio["seller_aggressor_count"] == 3
        assert abs(ratio["buyer_aggressor_ratio"] - 0.7) < 0.01
        assert abs(ratio["seller_aggressor_ratio"] - 0.3) < 0.01


class TestPerformance:
    """Testes de performance dos buffers"""
    
    def test_large_buffer_performance(self):
        """Testa performance com buffer grande"""
        buffer = CircularBuffer(max_size=10000)
        
        # Medir tempo de adição
        start = time.time()
        for i in range(10000):
            buffer.add({"id": i, "value": i * 2})
        add_time = time.time() - start
        
        # Deve adicionar 10k itens em menos de 1 segundo
        assert add_time < 1.0
        
        # Medir tempo de conversão para DataFrame
        start = time.time()
        df = buffer.get_dataframe()
        df_time = time.time() - start
        
        # Deve converter em menos de 0.5 segundos
        assert df_time < 0.5
        assert len(df) == 10000
    
    def test_memory_efficiency(self):
        """Testa eficiência de memória"""
        import gc
        import sys
        
        buffer = CircularBuffer(max_size=1000)
        
        # Adicionar muitos itens além do limite
        for i in range(5000):
            buffer.add({"id": i, "data": "x" * 100})
        
        # Forçar coleta de lixo
        gc.collect()
        
        # Buffer deve manter apenas 1000 itens
        assert buffer.size() == 1000
        
        # Verificar que itens antigos foram liberados
        stats = buffer.get_stats()
        assert stats["total_dropped"] == 4000


if __name__ == "__main__":
    # Executar testes
    pytest.main([__file__, "-v", "--tb=short"])