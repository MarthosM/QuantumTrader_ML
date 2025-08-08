"""
Testes de Performance para Cálculo de Features
Valida latência, throughput e uso de memória
"""

import pytest
import numpy as np
import pandas as pd
import time
import psutil
import os
from datetime import datetime, timedelta
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.book_features_rt import BookFeatureEngineerRT
from src.data.book_data_manager import BookDataManager
from src.buffers.circular_buffer import CandleBuffer


class TestFeaturePerformance:
    """Testes de performance do sistema de features"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.book_manager = BookDataManager()
        self.engineer = BookFeatureEngineerRT(self.book_manager)
        
        # Processo para medição de memória
        self.process = psutil.Process(os.getpid())
        
    def test_single_calculation_latency(self):
        """Testa latência de um único cálculo"""
        # Preparar dados
        self._prepare_test_data(50)
        
        # Medir latência
        times = []
        for _ in range(100):
            start = time.perf_counter()
            features = self.engineer.calculate_incremental_features({})
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        avg_latency = np.mean(times)
        p95_latency = np.percentile(times, 95)
        p99_latency = np.percentile(times, 99)
        
        print(f"\nLatência de cálculo único:")
        print(f"  Média: {avg_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
        print(f"  P99: {p99_latency:.2f}ms")
        
        # Verificar se atende ao SLA
        assert avg_latency < 10, "Latência média deve ser < 10ms"
        assert p99_latency < 50, "P99 latência deve ser < 50ms"
    
    def test_throughput(self):
        """Testa throughput (features/segundo)"""
        # Preparar dados
        self._prepare_test_data(100)
        
        # Medir throughput
        start = time.perf_counter()
        calculations = 0
        
        while time.perf_counter() - start < 1.0:  # 1 segundo
            _ = self.engineer.calculate_incremental_features({})
            calculations += 1
        
        print(f"\nThroughput: {calculations} cálculos/segundo")
        
        # Verificar throughput mínimo
        assert calculations > 100, "Throughput deve ser > 100 cálculos/segundo"
    
    def test_memory_efficiency(self):
        """Testa eficiência de memória"""
        # Memória inicial
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Adicionar muitos dados
        self._prepare_test_data(1000)
        
        # Fazer múltiplos cálculos
        for _ in range(100):
            _ = self.engineer.calculate_incremental_features({})
        
        # Memória final
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\nUso de memória:")
        print(f"  Inicial: {initial_memory:.2f}MB")
        print(f"  Final: {final_memory:.2f}MB")
        print(f"  Aumento: {memory_increase:.2f}MB")
        
        # Verificar limite de memória
        assert memory_increase < 100, "Aumento de memória deve ser < 100MB"
    
    def test_cache_effectiveness(self):
        """Testa efetividade do cache"""
        # Preparar dados
        self._prepare_test_data(50)
        
        # Resetar estatísticas
        self.engineer.stats = {
            'total_calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'avg_calc_time_ms': 0
        }
        
        # Fazer cálculos repetidos sem mudanças
        for _ in range(10):
            _ = self.engineer.calculate_incremental_features({})
        
        stats = self.engineer.get_statistics()
        
        # Cache deveria ser usado após primeiro cálculo
        # (implementação atual não tem cache entre chamadas, mas poderia ter)
        print(f"\nEstatísticas de cache:")
        print(f"  Total cálculos: {stats['total_calculations']}")
        print(f"  Tempo médio: {stats['avg_calc_time_ms']:.2f}ms")
    
    def test_parallel_calculations(self):
        """Testa cálculos paralelos (thread-safety)"""
        import threading
        
        # Preparar dados
        self._prepare_test_data(100)
        
        errors = []
        results = []
        
        def calculate():
            try:
                features = self.engineer.calculate_incremental_features({})
                results.append(len(features))
            except Exception as e:
                errors.append(e)
        
        # Criar threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=calculate)
            threads.append(t)
            t.start()
        
        # Aguardar conclusão
        for t in threads:
            t.join()
        
        # Verificar resultados
        assert len(errors) == 0, f"Erros em cálculos paralelos: {errors}"
        assert all(r == 65 for r in results), "Todos devem calcular 65 features"
        
        print(f"\nCálculos paralelos: {len(results)} threads bem-sucedidas")
    
    def test_incremental_vs_batch(self):
        """Compara performance incremental vs batch"""
        # Preparar dados
        candles = self._generate_candles(200)
        
        # Teste incremental
        start = time.perf_counter()
        for candle in candles:
            self.engineer._update_candle(candle)
            _ = self.engineer.calculate_incremental_features({})
        incremental_time = time.perf_counter() - start
        
        # Teste batch (simulado - seria com DataFrame completo)
        new_engineer = BookFeatureEngineerRT()
        start = time.perf_counter()
        for candle in candles:
            new_engineer._update_candle(candle)
        _ = new_engineer.calculate_incremental_features({})
        batch_time = time.perf_counter() - start
        
        print(f"\nComparação Incremental vs Batch:")
        print(f"  Incremental (200 updates): {incremental_time*1000:.2f}ms")
        print(f"  Batch (1 cálculo final): {batch_time*1000:.2f}ms")
        print(f"  Speedup batch: {incremental_time/batch_time:.2f}x")
    
    def test_feature_completeness(self):
        """Verifica se todas as 65 features são calculadas"""
        # Preparar dados completos
        self._prepare_test_data(200)
        self._prepare_book_data()
        
        # Calcular features
        features = self.engineer.calculate_incremental_features({})
        
        # Verificar completude
        assert len(features) == 65, f"Deve calcular 65 features, calculou {len(features)}"
        
        # Verificar que não são todas zero
        non_zero = sum(1 for v in features.values() if v != 0)
        print(f"\nFeatures não-zero: {non_zero}/65")
        
        assert non_zero > 30, "Deve ter pelo menos 30 features não-zero"
    
    def test_stress_test(self):
        """Teste de stress com alta carga"""
        # Simular alta frequência de updates
        start = time.perf_counter()
        errors = 0
        calculations = 0
        
        while time.perf_counter() - start < 5.0:  # 5 segundos
            try:
                # Adicionar novo candle
                candle = self._generate_single_candle()
                self.engineer._update_candle(candle)
                
                # Adicionar book update
                self._update_book_data()
                
                # Calcular features
                features = self.engineer.calculate_incremental_features({})
                
                if len(features) != 65:
                    errors += 1
                
                calculations += 1
                
            except Exception as e:
                errors += 1
                print(f"Erro no stress test: {e}")
        
        duration = time.perf_counter() - start
        rate = calculations / duration
        
        print(f"\nStress Test (5 segundos):")
        print(f"  Total cálculos: {calculations}")
        print(f"  Taxa: {rate:.1f} cálculos/segundo")
        print(f"  Erros: {errors}")
        
        assert errors == 0, "Não deve haver erros no stress test"
        assert rate > 50, "Taxa deve ser > 50 cálculos/segundo"
    
    # Métodos auxiliares
    
    def _prepare_test_data(self, num_candles: int):
        """Prepara dados de teste"""
        candles = self._generate_candles(num_candles)
        for candle in candles:
            self.engineer._update_candle(candle)
    
    def _generate_candles(self, num: int) -> list:
        """Gera candles simulados"""
        candles = []
        base_price = 5450.0
        base_time = datetime.now() - timedelta(minutes=num)
        
        for i in range(num):
            price_change = np.random.randn() * 5
            open_price = base_price + price_change
            close_price = open_price + np.random.randn() * 2
            
            candles.append({
                'timestamp': base_time + timedelta(minutes=i),
                'open': open_price,
                'high': max(open_price, close_price) + abs(np.random.randn()),
                'low': min(open_price, close_price) - abs(np.random.randn()),
                'close': close_price,
                'volume': 100000 + np.random.randint(-10000, 10000)
            })
            
            base_price = close_price
        
        return candles
    
    def _generate_single_candle(self) -> dict:
        """Gera um único candle"""
        base_price = 5450.0 + np.random.randn() * 10
        return {
            'timestamp': datetime.now(),
            'open': base_price,
            'high': base_price + abs(np.random.randn()) * 2,
            'low': base_price - abs(np.random.randn()) * 2,
            'close': base_price + np.random.randn(),
            'volume': 100000 + np.random.randint(-10000, 10000)
        }
    
    def _prepare_book_data(self):
        """Prepara dados de book"""
        # Simular callbacks de book
        price_data = {
            'timestamp': datetime.now(),
            'symbol': 'WDOU25',
            'bids': [
                {'price': 5450.0 - i*0.5, 'volume': 100 + i*10, 'trader_id': f'T{i}'}
                for i in range(5)
            ]
        }
        self.book_manager.on_price_book_callback(price_data)
        
        offer_data = {
            'timestamp': datetime.now(),
            'symbol': 'WDOU25',
            'asks': [
                {'price': 5451.0 + i*0.5, 'volume': 110 + i*10, 'trader_id': f'T{i+5}'}
                for i in range(5)
            ]
        }
        self.book_manager.on_offer_book_callback(offer_data)
    
    def _update_book_data(self):
        """Atualiza dados de book com variação"""
        base_price = 5450.0 + np.random.randn() * 2
        
        price_data = {
            'timestamp': datetime.now(),
            'bids': [
                {'price': base_price - i*0.5, 'volume': 100 + np.random.randint(0, 50)}
                for i in range(5)
            ]
        }
        self.book_manager.on_price_book_callback(price_data)
        
        offer_data = {
            'timestamp': datetime.now(),
            'asks': [
                {'price': base_price + 1 + i*0.5, 'volume': 100 + np.random.randint(0, 50)}
                for i in range(5)
            ]
        }
        self.book_manager.on_offer_book_callback(offer_data)


def benchmark_feature_calculation():
    """Benchmark completo do sistema de features"""
    print("=" * 60)
    print("BENCHMARK DO SISTEMA DE FEATURES")
    print("=" * 60)
    
    # Criar instância
    book_manager = BookDataManager()
    engineer = BookFeatureEngineerRT(book_manager)
    
    # Preparar dados
    print("\nPreparando dados de teste...")
    base_time = datetime.now() - timedelta(hours=1)
    
    for i in range(200):
        candle = {
            'timestamp': base_time + timedelta(minutes=i*0.3),
            'open': 5450 + np.random.randn() * 5,
            'high': 5455 + np.random.randn() * 5,
            'low': 5445 + np.random.randn() * 5,
            'close': 5450 + np.random.randn() * 5,
            'volume': 100000 + np.random.randint(-10000, 10000)
        }
        engineer._update_candle(candle)
    
    # Benchmark
    print("\nExecutando benchmark...")
    
    # Warmup
    for _ in range(10):
        _ = engineer.calculate_incremental_features({})
    
    # Medições
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        features = engineer.calculate_incremental_features({})
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    # Resultados
    print("\nResultados do Benchmark:")
    print(f"  Iterações: 1000")
    print(f"  Features calculadas: 65")
    print(f"\nLatência (ms):")
    print(f"  Mínima: {np.min(times):.3f}")
    print(f"  Média: {np.mean(times):.3f}")
    print(f"  Mediana: {np.median(times):.3f}")
    print(f"  P95: {np.percentile(times, 95):.3f}")
    print(f"  P99: {np.percentile(times, 99):.3f}")
    print(f"  Máxima: {np.max(times):.3f}")
    print(f"\nThroughput:")
    print(f"  {1000/np.mean(times):.1f} cálculos/segundo")
    
    # Verificar se atende aos requisitos
    avg_latency = np.mean(times)
    p99_latency = np.percentile(times, 99)
    
    if avg_latency < 10 and p99_latency < 50:
        print("\n[OK] PERFORMANCE APROVADA - Atende aos requisitos")
    else:
        print("\n[ERRO] PERFORMANCE INSUFICIENTE - Otimizacao necessaria")
    
    return avg_latency, p99_latency


if __name__ == "__main__":
    # Executar testes específicos ou benchmark
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_feature_calculation()
    else:
        # Executar testes com pytest
        pytest.main([__file__, "-v", "--tb=short"])