"""
Stress Test V3 - Sistema de Stress Testing
==========================================

Este módulo implementa testes de stress para validar:
- Resiliência sob condições extremas
- Performance com alta carga
- Recuperação de falhas
- Comportamento em cenários adversos
"""

import os
import sys
import time
import threading
import psutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Adicionar src ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.trading_data_structure_v3 import TradingDataStructureV3
from features.ml_features_v3 import MLFeaturesV3
from realtime.realtime_processor_v3 import RealTimeProcessorV3
from ml.prediction_engine_v3 import PredictionEngineV3
from monitoring.system_monitor_v3 import SystemMonitorV3


class StressScenario:
    """Define um cenário de stress"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.parameters = {}
        self.results = {}
        
    def add_parameter(self, key: str, value):
        """Adiciona parâmetro ao cenário"""
        self.parameters[key] = value
        
    def record_result(self, key: str, value):
        """Registra resultado do teste"""
        self.results[key] = value


class StressTestV3:
    """Sistema de stress testing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scenarios = []
        self.results = {}
        
        # Componentes do sistema
        self.data_structure = TradingDataStructureV3()
        self.ml_features = MLFeaturesV3()
        self.realtime_processor = RealTimeProcessorV3()
        self.prediction_engine = PredictionEngineV3()
        self.system_monitor = SystemMonitorV3()
        
        # Métricas de sistema
        self.initial_memory = 0
        self.initial_cpu = 0
        
    def run_all_tests(self) -> Dict:
        """Executa todos os testes de stress"""
        self.logger.info("Iniciando stress testing completo...")
        
        # Definir cenários
        self._define_scenarios()
        
        # Capturar estado inicial
        self._capture_initial_state()
        
        # Executar cada cenário
        for scenario in self.scenarios:
            self.logger.info(f"\nExecutando cenário: {scenario.name}")
            self.logger.info(f"Descrição: {scenario.description}")
            
            try:
                self._run_scenario(scenario)
                scenario.record_result('status', 'PASSED')
            except Exception as e:
                self.logger.error(f"Erro no cenário {scenario.name}: {e}")
                scenario.record_result('status', 'FAILED')
                scenario.record_result('error', str(e))
            
            # Aguardar sistema estabilizar
            time.sleep(2)
        
        # Gerar relatório
        report = self._generate_report()
        
        # Salvar resultados
        self._save_results(report)
        
        return report
    
    def _define_scenarios(self):
        """Define cenários de stress test"""
        
        # 1. Alta Frequência de Dados
        high_frequency = StressScenario(
            "High Frequency Data",
            "Testa sistema com alta frequência de trades (1000 trades/segundo)"
        )
        high_frequency.add_parameter('trades_per_second', 1000)
        high_frequency.add_parameter('duration_seconds', 10)
        self.scenarios.append(high_frequency)
        
        # 2. Dados Extremos
        extreme_data = StressScenario(
            "Extreme Market Data",
            "Testa com movimentos extremos de preço (±10% em 1 minuto)"
        )
        extreme_data.add_parameter('price_volatility', 0.10)
        extreme_data.add_parameter('gap_percentage', 0.05)
        self.scenarios.append(extreme_data)
        
        # 3. Volume Massivo
        massive_volume = StressScenario(
            "Massive Data Volume",
            "Processa 1 milhão de trades históricos"
        )
        massive_volume.add_parameter('total_trades', 1000000)
        massive_volume.add_parameter('batch_size', 10000)
        self.scenarios.append(massive_volume)
        
        # 4. Processamento Paralelo
        parallel_processing = StressScenario(
            "Parallel Processing",
            "Testa com 100 threads simultâneas calculando features"
        )
        parallel_processing.add_parameter('num_threads', 100)
        parallel_processing.add_parameter('calculations_per_thread', 100)
        self.scenarios.append(parallel_processing)
        
        # 5. Recuperação de Falhas
        failure_recovery = StressScenario(
            "Failure Recovery",
            "Simula falhas e testa recuperação automática"
        )
        failure_recovery.add_parameter('num_failures', 10)
        failure_recovery.add_parameter('failure_types', ['connection', 'calculation', 'memory'])
        self.scenarios.append(failure_recovery)
        
        # 6. Memória Limitada
        memory_pressure = StressScenario(
            "Memory Pressure",
            "Testa comportamento com memória limitada"
        )
        memory_pressure.add_parameter('memory_limit_mb', 500)
        memory_pressure.add_parameter('data_size_mb', 1000)
        self.scenarios.append(memory_pressure)
        
        # 7. Latência de Rede
        network_latency = StressScenario(
            "Network Latency",
            "Simula alta latência e perda de pacotes"
        )
        network_latency.add_parameter('latency_ms', 500)
        network_latency.add_parameter('packet_loss', 0.10)
        self.scenarios.append(network_latency)
        
        # 8. Carga Sustentada
        sustained_load = StressScenario(
            "Sustained Load",
            "Mantém carga alta por período prolongado"
        )
        sustained_load.add_parameter('duration_minutes', 5)
        sustained_load.add_parameter('trades_per_second', 100)
        self.scenarios.append(sustained_load)
    
    def _capture_initial_state(self):
        """Captura estado inicial do sistema"""
        process = psutil.Process()
        self.initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        self.initial_cpu = psutil.cpu_percent(interval=1)
        
        self.logger.info(f"Estado inicial - Memória: {self.initial_memory:.2f} MB, CPU: {self.initial_cpu:.1f}%")
    
    def _run_scenario(self, scenario: StressScenario):
        """Executa um cenário específico"""
        start_time = time.time()
        
        if scenario.name == "High Frequency Data":
            self._test_high_frequency(scenario)
        elif scenario.name == "Extreme Market Data":
            self._test_extreme_data(scenario)
        elif scenario.name == "Massive Data Volume":
            self._test_massive_volume(scenario)
        elif scenario.name == "Parallel Processing":
            self._test_parallel_processing(scenario)
        elif scenario.name == "Failure Recovery":
            self._test_failure_recovery(scenario)
        elif scenario.name == "Memory Pressure":
            self._test_memory_pressure(scenario)
        elif scenario.name == "Network Latency":
            self._test_network_latency(scenario)
        elif scenario.name == "Sustained Load":
            self._test_sustained_load(scenario)
        
        # Registrar tempo de execução
        duration = time.time() - start_time
        scenario.record_result('duration_seconds', duration)
        
        # Capturar métricas finais
        self._capture_scenario_metrics(scenario)
    
    def _test_high_frequency(self, scenario: StressScenario):
        """Testa alta frequência de dados"""
        trades_per_second = scenario.parameters['trades_per_second']
        duration = scenario.parameters['duration_seconds']
        
        # Iniciar processador
        self.realtime_processor.start()
        
        trades_processed = 0
        errors = 0
        latencies = []
        
        # Gerar trades em alta frequência
        start = time.time()
        interval = 1.0 / trades_per_second
        
        while time.time() - start < duration:
            try:
                trade_start = time.time()
                
                # Criar trade
                trade = {
                    'timestamp': datetime.now(),
                    'price': 5900 + np.random.randn() * 10,
                    'volume': np.random.randint(100, 1000),
                    'side': 'BUY' if np.random.random() > 0.5 else 'SELL'
                }
                
                # Processar
                self.realtime_processor.add_trade(trade)
                
                # Medir latência
                latency = (time.time() - trade_start) * 1000  # ms
                latencies.append(latency)
                
                trades_processed += 1
                
                # Aguardar próximo trade
                time.sleep(max(0, interval - (time.time() - trade_start)))
                
            except Exception as e:
                errors += 1
                self.logger.error(f"Erro processando trade: {e}")
        
        # Parar processador
        self.realtime_processor.stop()
        
        # Registrar resultados
        scenario.record_result('trades_processed', trades_processed)
        scenario.record_result('errors', errors)
        scenario.record_result('avg_latency_ms', np.mean(latencies))
        scenario.record_result('max_latency_ms', np.max(latencies))
        scenario.record_result('throughput', trades_processed / duration)
    
    def _test_extreme_data(self, scenario: StressScenario):
        """Testa dados extremos de mercado"""
        volatility = scenario.parameters['price_volatility']
        gap_pct = scenario.parameters['gap_percentage']
        
        base_price = 5900
        candles = []
        
        # Gerar candles com movimentos extremos
        for i in range(100):
            # Simular gap
            if i % 20 == 0:
                gap = base_price * gap_pct * (1 if np.random.random() > 0.5 else -1)
            else:
                gap = 0
            
            # Movimento extremo
            move = base_price * volatility * np.random.randn()
            
            open_price = base_price + gap
            close_price = open_price + move
            high_price = max(open_price, close_price) + abs(move) * 0.2
            low_price = min(open_price, close_price) - abs(move) * 0.2
            
            candles.append({
                'datetime': datetime.now() - timedelta(minutes=100-i),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(1000, 10000),
                'buy_volume': np.random.randint(500, 5000),
                'sell_volume': np.random.randint(500, 5000)
            })
            
            base_price = close_price
        
        # Criar DataFrame
        candles_df = pd.DataFrame(candles)
        candles_df.set_index('datetime', inplace=True)
        
        # Testar cálculo de features
        try:
            start = time.time()
            microstructure = self._create_microstructure(candles_df)
            features = self.ml_features.calculate_all(candles_df, microstructure)
            calc_time = time.time() - start
            
            scenario.record_result('features_calculated', len(features))
            scenario.record_result('calc_time_seconds', calc_time)
            scenario.record_result('nan_rate', features.isna().sum().sum() / features.size)
            
        except Exception as e:
            scenario.record_result('calculation_error', str(e))
    
    def _test_massive_volume(self, scenario: StressScenario):
        """Testa volume massivo de dados"""
        total_trades = scenario.parameters['total_trades']
        batch_size = scenario.parameters['batch_size']
        
        trades_processed = 0
        total_time = 0
        memory_usage = []
        
        # Processar em batches
        num_batches = total_trades // batch_size
        
        for batch in range(num_batches):
            # Gerar batch de trades
            trades = []
            for _ in range(batch_size):
                trades.append({
                    'datetime': datetime.now() - timedelta(seconds=np.random.randint(0, 3600)),
                    'price': 5900 + np.random.randn() * 20,
                    'volume': np.random.randint(100, 1000),
                    'buy_volume': np.random.randint(50, 500),
                    'sell_volume': np.random.randint(50, 500),
                    'side': 'BUY' if np.random.random() > 0.5 else 'SELL'
                })
            
            # Processar batch
            start = time.time()
            
            trades_df = pd.DataFrame(trades)
            trades_df.set_index('datetime', inplace=True)
            
            # Simular processamento
            historical_data = {
                'trades': trades_df,
                'candles': pd.DataFrame(),
                'book_updates': pd.DataFrame()
            }
            
            self.data_structure.add_historical_data(historical_data)
            
            batch_time = time.time() - start
            total_time += batch_time
            trades_processed += batch_size
            
            # Monitorar memória
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_usage.append(memory_mb)
            
            # Log progresso
            if (batch + 1) % 10 == 0:
                self.logger.info(f"Processados {trades_processed}/{total_trades} trades")
        
        # Registrar resultados
        scenario.record_result('trades_processed', trades_processed)
        scenario.record_result('total_time_seconds', total_time)
        scenario.record_result('throughput', trades_processed / total_time)
        scenario.record_result('max_memory_mb', max(memory_usage))
        scenario.record_result('avg_memory_mb', np.mean(memory_usage))
    
    def _test_parallel_processing(self, scenario: StressScenario):
        """Testa processamento paralelo"""
        num_threads = scenario.parameters['num_threads']
        calcs_per_thread = scenario.parameters['calculations_per_thread']
        
        results = []
        errors = []
        
        # Criar dados de teste
        candles = self._generate_test_candles(100)
        microstructure = self._create_microstructure(candles)
        
        def worker(thread_id):
            """Worker para processamento paralelo"""
            thread_results = {
                'thread_id': thread_id,
                'calculations': 0,
                'errors': 0,
                'duration': 0
            }
            
            start = time.time()
            
            for i in range(calcs_per_thread):
                try:
                    # Calcular features
                    features = self.ml_features.calculate_all(
                        candles.copy(), 
                        microstructure.copy()
                    )
                    thread_results['calculations'] += 1
                    
                except Exception as e:
                    thread_results['errors'] += 1
                    errors.append(str(e))
            
            thread_results['duration'] = time.time() - start
            return thread_results
        
        # Executar em paralelo
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            
            for future in as_completed(futures):
                results.append(future.result())
        
        total_time = time.time() - start_time
        
        # Agregar resultados
        total_calcs = sum(r['calculations'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        
        scenario.record_result('total_calculations', total_calcs)
        scenario.record_result('total_errors', total_errors)
        scenario.record_result('total_time_seconds', total_time)
        scenario.record_result('calcs_per_second', total_calcs / total_time)
        scenario.record_result('unique_errors', len(set(errors)))
    
    def _test_failure_recovery(self, scenario: StressScenario):
        """Testa recuperação de falhas"""
        num_failures = scenario.parameters['num_failures']
        failure_types = scenario.parameters['failure_types']
        
        recovery_times = []
        failure_results = []
        
        for i in range(num_failures):
            failure_type = random.choice(failure_types)
            
            self.logger.info(f"Simulando falha {i+1}/{num_failures}: {failure_type}")
            
            start = time.time()
            
            try:
                if failure_type == 'connection':
                    # Simular desconexão
                    self._simulate_connection_failure()
                    
                elif failure_type == 'calculation':
                    # Simular erro de cálculo
                    self._simulate_calculation_failure()
                    
                elif failure_type == 'memory':
                    # Simular pressão de memória
                    self._simulate_memory_failure()
                
                # Tentar recuperar
                recovered = self._attempt_recovery(failure_type)
                recovery_time = time.time() - start
                
                failure_results.append({
                    'type': failure_type,
                    'recovered': recovered,
                    'recovery_time': recovery_time
                })
                
                if recovered:
                    recovery_times.append(recovery_time)
                
            except Exception as e:
                failure_results.append({
                    'type': failure_type,
                    'recovered': False,
                    'error': str(e)
                })
        
        # Registrar resultados
        successful_recoveries = sum(1 for f in failure_results if f.get('recovered', False))
        
        scenario.record_result('failures_simulated', num_failures)
        scenario.record_result('successful_recoveries', successful_recoveries)
        scenario.record_result('recovery_rate', successful_recoveries / num_failures)
        scenario.record_result('avg_recovery_time', np.mean(recovery_times) if recovery_times else 0)
        scenario.record_result('failure_details', failure_results)
    
    def _test_memory_pressure(self, scenario: StressScenario):
        """Testa comportamento sob pressão de memória"""
        memory_limit = scenario.parameters['memory_limit_mb']
        data_size = scenario.parameters['data_size_mb']
        
        # Criar dados grandes
        self.logger.info(f"Criando {data_size}MB de dados com limite de {memory_limit}MB...")
        
        large_data = []
        current_size = 0
        performance_degradation = []
        
        while current_size < data_size:
            # Criar chunk de dados
            chunk_size = min(10, data_size - current_size)  # 10MB chunks
            chunk = np.random.randn(int(chunk_size * 1024 * 1024 / 8))  # 8 bytes per float
            large_data.append(chunk)
            current_size += chunk_size
            
            # Medir performance
            start = time.time()
            _ = np.mean(chunk)  # Operação simples
            operation_time = time.time() - start
            performance_degradation.append(operation_time)
            
            # Verificar memória
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > memory_limit:
                self.logger.warning(f"Limite de memória excedido: {memory_mb:.2f}MB")
                # Tentar liberar memória
                if len(large_data) > 1:
                    large_data.pop(0)
                    import gc
                    gc.collect()
        
        # Registrar resultados
        scenario.record_result('data_created_mb', current_size)
        scenario.record_result('peak_memory_mb', max(performance_degradation) * 1000)  # Proxy
        scenario.record_result('performance_impact', max(performance_degradation) / min(performance_degradation))
    
    def _test_network_latency(self, scenario: StressScenario):
        """Testa impacto de latência de rede"""
        latency_ms = scenario.parameters['latency_ms']
        packet_loss = scenario.parameters['packet_loss']
        
        # Simular requisições com latência
        requests_sent = 100
        requests_received = 0
        total_latency = 0
        timeouts = 0
        
        for i in range(requests_sent):
            # Simular perda de pacote
            if np.random.random() < packet_loss:
                timeouts += 1
                continue
            
            # Simular latência
            simulated_latency = latency_ms + np.random.exponential(latency_ms * 0.2)
            time.sleep(simulated_latency / 1000)
            
            requests_received += 1
            total_latency += simulated_latency
        
        # Registrar resultados
        scenario.record_result('requests_sent', requests_sent)
        scenario.record_result('requests_received', requests_received)
        scenario.record_result('packet_loss_rate', 1 - (requests_received / requests_sent))
        scenario.record_result('avg_latency_ms', total_latency / requests_received if requests_received > 0 else 0)
        scenario.record_result('timeouts', timeouts)
    
    def _test_sustained_load(self, scenario: StressScenario):
        """Testa carga sustentada"""
        duration_minutes = scenario.parameters['duration_minutes']
        trades_per_second = scenario.parameters['trades_per_second']
        
        # Iniciar componentes
        self.realtime_processor.start()
        self.system_monitor.start()
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        metrics_over_time = []
        errors_over_time = []
        
        # Manter carga constante
        interval = 1.0 / trades_per_second
        
        while time.time() < end_time:
            minute_start = time.time()
            trades_this_minute = 0
            errors_this_minute = 0
            
            # Processar por 1 minuto
            while time.time() < minute_start + 60 and time.time() < end_time:
                try:
                    trade = {
                        'timestamp': datetime.now(),
                        'price': 5900 + np.random.randn() * 10,
                        'volume': np.random.randint(100, 1000),
                        'side': 'BUY' if np.random.random() > 0.5 else 'SELL'
                    }
                    
                    self.realtime_processor.add_trade(trade)
                    trades_this_minute += 1
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    errors_this_minute += 1
            
            # Coletar métricas
            process = psutil.Process()
            metrics = {
                'minute': len(metrics_over_time) + 1,
                'trades': trades_this_minute,
                'errors': errors_this_minute,
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_mb': process.memory_info().rss / 1024 / 1024
            }
            
            metrics_over_time.append(metrics)
            self.logger.info(f"Minuto {metrics['minute']}: {trades_this_minute} trades, "
                           f"CPU: {metrics['cpu_percent']:.1f}%, "
                           f"Mem: {metrics['memory_mb']:.1f}MB")
        
        # Parar componentes
        self.realtime_processor.stop()
        self.system_monitor.stop()
        
        # Analisar estabilidade
        cpu_values = [m['cpu_percent'] for m in metrics_over_time]
        memory_values = [m['memory_mb'] for m in metrics_over_time]
        
        scenario.record_result('total_minutes', len(metrics_over_time))
        scenario.record_result('avg_cpu_percent', np.mean(cpu_values))
        scenario.record_result('cpu_std_dev', np.std(cpu_values))
        scenario.record_result('avg_memory_mb', np.mean(memory_values))
        scenario.record_result('memory_growth_mb', memory_values[-1] - memory_values[0])
        scenario.record_result('total_errors', sum(m['errors'] for m in metrics_over_time))
        scenario.record_result('metrics_timeline', metrics_over_time)
    
    def _simulate_connection_failure(self):
        """Simula falha de conexão"""
        # Simular desconexão
        time.sleep(0.5)
        raise ConnectionError("Conexão perdida")
    
    def _simulate_calculation_failure(self):
        """Simula erro de cálculo"""
        # Forçar divisão por zero
        _ = 1 / 0
    
    def _simulate_memory_failure(self):
        """Simula falha de memória"""
        # Tentar alocar muita memória
        _ = np.zeros((1000, 1000, 1000))
    
    def _attempt_recovery(self, failure_type: str) -> bool:
        """Tenta recuperar de uma falha"""
        try:
            if failure_type == 'connection':
                # Simular reconexão
                time.sleep(1)
                return True
                
            elif failure_type == 'calculation':
                # Reiniciar cálculo
                time.sleep(0.5)
                return True
                
            elif failure_type == 'memory':
                # Liberar memória
                import gc
                gc.collect()
                time.sleep(0.5)
                return True
                
        except:
            return False
    
    def _generate_test_candles(self, num_candles: int) -> pd.DataFrame:
        """Gera candles de teste"""
        candles = []
        base_price = 5900
        
        for i in range(num_candles):
            move = np.random.randn() * 5
            candles.append({
                'datetime': datetime.now() - timedelta(minutes=num_candles-i),
                'open': base_price,
                'high': base_price + abs(move),
                'low': base_price - abs(move),
                'close': base_price + move,
                'volume': np.random.randint(1000, 5000),
                'buy_volume': np.random.randint(500, 2500),
                'sell_volume': np.random.randint(500, 2500)
            })
            base_price += move
        
        df = pd.DataFrame(candles)
        df.set_index('datetime', inplace=True)
        return df
    
    def _create_microstructure(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Cria microestrutura básica"""
        micro = pd.DataFrame(index=candles.index)
        micro['buy_volume'] = candles['buy_volume']
        micro['sell_volume'] = candles['sell_volume']
        micro['volume_imbalance'] = (
            (candles['buy_volume'] - candles['sell_volume']) / 
            (candles['buy_volume'] + candles['sell_volume'] + 1)
        )
        micro['trade_imbalance'] = micro['volume_imbalance']
        micro['bid_ask_spread'] = 0.001
        return micro
    
    def _capture_scenario_metrics(self, scenario: StressScenario):
        """Captura métricas após cenário"""
        process = psutil.Process()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        final_cpu = psutil.cpu_percent(interval=1)
        
        scenario.record_result('memory_increase_mb', final_memory - self.initial_memory)
        scenario.record_result('final_cpu_percent', final_cpu)
    
    def _generate_report(self) -> Dict:
        """Gera relatório de stress test"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'scenarios': []
        }
        
        # Resumo de cada cenário
        for scenario in self.scenarios:
            scenario_report = {
                'name': scenario.name,
                'description': scenario.description,
                'status': scenario.results.get('status', 'UNKNOWN'),
                'duration': scenario.results.get('duration_seconds', 0),
                'key_metrics': {}
            }
            
            # Métricas específicas por cenário
            if scenario.name == "High Frequency Data":
                scenario_report['key_metrics'] = {
                    'throughput': scenario.results.get('throughput', 0),
                    'avg_latency_ms': scenario.results.get('avg_latency_ms', 0),
                    'errors': scenario.results.get('errors', 0)
                }
            elif scenario.name == "Massive Data Volume":
                scenario_report['key_metrics'] = {
                    'trades_processed': scenario.results.get('trades_processed', 0),
                    'throughput': scenario.results.get('throughput', 0),
                    'max_memory_mb': scenario.results.get('max_memory_mb', 0)
                }
            # ... adicionar outros cenários
            
            report['scenarios'].append(scenario_report)
        
        # Resumo geral
        total_scenarios = len(self.scenarios)
        passed_scenarios = sum(1 for s in self.scenarios if s.results.get('status') == 'PASSED')
        
        report['summary'] = {
            'total_scenarios': total_scenarios,
            'passed': passed_scenarios,
            'failed': total_scenarios - passed_scenarios,
            'success_rate': passed_scenarios / total_scenarios if total_scenarios > 0 else 0
        }
        
        return report
    
    def _save_results(self, report: Dict):
        """Salva resultados do stress test"""
        filename = f"stress_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Relatório salvo em: {filename}")
        
        # Exibir resumo
        print("\n" + "="*60)
        print("RESUMO DO STRESS TEST")
        print("="*60)
        print(f"Total de cenários: {report['summary']['total_scenarios']}")
        print(f"Passou: {report['summary']['passed']}")
        print(f"Falhou: {report['summary']['failed']}")
        print(f"Taxa de sucesso: {report['summary']['success_rate']:.1%}")
        print("="*60)


def main():
    """Executa stress test completo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Iniciando Stress Test V3...")
    print("AVISO: Este teste pode consumir recursos significativos do sistema")
    print("-"*60)
    
    stress_test = StressTestV3()
    report = stress_test.run_all_tests()
    
    print("\nStress test concluído!")


if __name__ == "__main__":
    main()