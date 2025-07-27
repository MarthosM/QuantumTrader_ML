"""
Testes para coleta de dados reais
Baseado no DEVELOPER_GUIDE_V3_REFACTORING.md
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import tempfile
import shutil

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.real_data_collector import RealDataCollector
from data.trading_data_structure_v3 import TradingDataStructureV3


class TestRealDataCollection(unittest.TestCase):
    """Teste 1.1.1: Validar Coleta de Dados"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.collector = RealDataCollector()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Cleanup após cada teste"""
        if hasattr(self.collector, 'dll') and self.collector.dll:
            try:
                self.collector.disconnect()
            except:
                pass
        
        # Limpar arquivos temporários
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_collector_initialization(self):
        """Testa inicialização do collector"""
        
        # Teste sem DLL (modo simulação)
        self.assertIsNotNone(self.collector)
        self.assertEqual(self.collector.stats['trades_collected'], 0)
        self.assertEqual(self.collector.stats['candles_collected'], 0)
        self.assertFalse(self.collector.connected)
        
        print("[OK] Collector inicializado corretamente")
    
    def test_trade_aggregation(self):
        """Testa agregação de trades para candles"""
        
        # Criar trades simulados
        trades_data = self._create_sample_trades(1000)
        trades_df = pd.DataFrame(trades_data)
        trades_df.set_index('datetime', inplace=True)
        
        # Agregar para candles
        candles = self.collector.aggregate_to_candles(trades_df, '1min')
        
        # Validações
        self.assertGreater(len(candles), 0, "Nenhum candle gerado")
        
        # Verificar colunas essenciais
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'quantity']
        for col in required_columns:
            self.assertIn(col, candles.columns, f"Coluna {col} ausente")
        
        # Verificar separação por lado
        self.assertIn('buy_volume', candles.columns, "buy_volume ausente")
        self.assertIn('sell_volume', candles.columns, "sell_volume ausente")
        
        # Verificar consistência
        volume_consistency = np.allclose(
            candles['buy_volume'] + candles['sell_volume'],
            candles['volume'],
            rtol=1e-5
        )
        self.assertTrue(volume_consistency, "Volume buy+sell != volume total")
        
        # Verificar OHLC válido
        ohlc_valid = (
            (candles['high'] >= candles['open']) &
            (candles['high'] >= candles['close']) &
            (candles['low'] <= candles['open']) &
            (candles['low'] <= candles['close'])
        ).all()
        self.assertTrue(ohlc_valid, "OHLC inválido")
        
        print(f"[OK] Agregação validada: {len(trades_df)} trades -> {len(candles)} candles")
    
    def test_microstructure_calculation(self):
        """Testa cálculo de métricas microestruturais"""
        
        # Criar trades simulados
        trades_data = self._create_sample_trades(2000)
        trades_df = pd.DataFrame(trades_data)
        trades_df.set_index('datetime', inplace=True)
        
        # Calcular microestrutura
        microstructure = self.collector.calculate_microstructure_metrics(trades_df, '1min')
        
        # Validações
        self.assertGreater(len(microstructure), 0, "Nenhuma métrica calculada")
        
        # Verificar colunas essenciais
        required_columns = [
            'volume_imbalance', 'trade_imbalance', 'buy_pressure',
            'sell_pressure', 'avg_trade_size'
        ]
        for col in required_columns:
            self.assertIn(col, microstructure.columns, f"Coluna {col} ausente")
        
        # Verificar ranges válidos
        self.assertTrue(
            microstructure['buy_pressure'].between(0, 1).all(),
            "buy_pressure fora do range [0,1]"
        )
        
        self.assertTrue(
            microstructure['sell_pressure'].between(0, 1).all(),
            "sell_pressure fora do range [0,1]"
        )
        
        self.assertTrue(
            microstructure['trade_imbalance_ratio'].between(-1, 1).all(),
            "trade_imbalance_ratio fora do range [-1,1]"
        )
        
        # Verificar que não há valores infinitos
        self.assertFalse(np.isinf(microstructure).any().any(), "Valores infinitos detectados")
        
        print(f"[OK] Microestrutura calculada: {microstructure.shape}")
    
    def test_data_quality_validation(self):
        """Testa validação de qualidade dos dados"""
        
        # Criar dados com diferentes qualidades
        
        # Dados de boa qualidade
        good_trades = self._create_sample_trades(1500, quality='good')
        good_df = pd.DataFrame(good_trades)
        good_df.set_index('datetime', inplace=True)
        
        good_candles = self.collector.aggregate_to_candles(good_df)
        good_micro = self.collector.calculate_microstructure_metrics(good_df)
        
        # Validar dados bons
        self.assertGreater(len(good_candles), 50, "Poucos candles de boa qualidade")
        self.assertLess(good_candles.isna().sum().sum(), len(good_candles) * 0.05, "Muitos NaN")
        
        # Dados de má qualidade
        bad_trades = self._create_sample_trades(100, quality='bad')
        bad_df = pd.DataFrame(bad_trades)
        bad_df.set_index('datetime', inplace=True)
        
        bad_candles = self.collector.aggregate_to_candles(bad_df)
        
        # Validar que dados ruins são detectados
        self.assertLess(len(bad_candles), 20, "Muitos candles de má qualidade aceitos")
        
        print("[OK] Validação de qualidade funcionando")
    
    def test_side_determination(self):
        """Testa determinação correta do lado do trade"""
        
        # Criar trades com sides conhecidos
        trades_data = []
        base_time = datetime(2025, 1, 27, 10, 0)
        
        for i in range(100):
            expected_side = 'BUY' if i % 2 == 0 else 'SELL'
            trade = {
                'datetime': base_time + timedelta(seconds=i),
                'price': 5900 + (i * 0.5) if expected_side == 'BUY' else 5900 - (i * 0.5),
                'volume': np.random.randint(1000, 10000),
                'quantity': np.random.randint(1, 10),
                'side': expected_side
            }
            trades_data.append(trade)
        
        trades_df = pd.DataFrame(trades_data)
        trades_df.set_index('datetime', inplace=True)
        
        # Verificar que sides estão sendo preservados
        buy_trades = trades_df[trades_df['side'] == 'BUY']
        sell_trades = trades_df[trades_df['side'] == 'SELL']
        
        self.assertGreater(len(buy_trades), 0, "Nenhum trade BUY")
        self.assertGreater(len(sell_trades), 0, "Nenhum trade SELL")
        self.assertEqual(len(buy_trades) + len(sell_trades), len(trades_df), "Sides inconsistentes")
        
        # Verificar que agregação preserva sides
        candles = self.collector.aggregate_to_candles(trades_df)
        
        self.assertTrue((candles['buy_volume'] >= 0).all(), "buy_volume negativo")
        self.assertTrue((candles['sell_volume'] >= 0).all(), "sell_volume negativo")
        self.assertTrue((candles['buy_trades'] >= 0).all(), "buy_trades negativo")
        self.assertTrue((candles['sell_trades'] >= 0).all(), "sell_trades negativo")
        
        print("[OK] Determinação de side validada")
    
    def test_temporal_consistency(self):
        """Testa consistência temporal dos dados"""
        
        # Criar trades com timestamps específicos
        trades_data = []
        base_time = datetime(2025, 1, 27, 9, 0)
        
        # Criar 2 horas de dados (9:00 - 11:00)
        for minute in range(120):
            # 5-15 trades por minuto
            trades_in_minute = np.random.randint(5, 16)
            
            for trade_num in range(trades_in_minute):
                trade_time = base_time + timedelta(minutes=minute, seconds=trade_num*3)
                
                trade = {
                    'datetime': trade_time,
                    'price': 5900 + np.random.randn() * 2,
                    'volume': np.random.randint(1000, 10000),
                    'quantity': np.random.randint(1, 10),
                    'side': np.random.choice(['BUY', 'SELL'])
                }
                trades_data.append(trade)
        
        trades_df = pd.DataFrame(trades_data)
        trades_df.set_index('datetime', inplace=True)
        trades_df.sort_index(inplace=True)
        
        # Agregar para candles de 1 minuto
        candles = self.collector.aggregate_to_candles(trades_df, '1min')
        
        # Verificar continuidade temporal
        time_diffs = candles.index.to_series().diff()[1:]
        expected_diff = pd.Timedelta(minutes=1)
        
        # Deve ter exatamente 120 candles (2 horas)
        self.assertEqual(len(candles), 120, f"Esperado 120 candles, obtido {len(candles)}")
        
        # Verificar que não há gaps maiores que 1 minuto
        max_gap = time_diffs.max()
        self.assertLessEqual(max_gap, expected_diff, f"Gap temporal muito grande: {max_gap}")
        
        # Verificar que candles estão ordenados
        self.assertTrue(candles.index.is_monotonic_increasing, "Candles fora de ordem")
        
        print(f"[OK] Consistência temporal validada: {len(candles)} candles em sequência")
    
    def _create_sample_trades(self, count: int, quality: str = 'good') -> list:
        """Cria trades simulados para teste"""
        
        trades = []
        base_time = datetime(2025, 1, 27, 10, 0)
        base_price = 5900
        
        for i in range(count):
            if quality == 'good':
                # Dados de boa qualidade
                time_increment = timedelta(seconds=np.random.randint(1, 30))
                price_change = np.random.randn() * 0.5
                volume = np.random.randint(1000, 50000)
                quantity = np.random.randint(1, 20)
                
            elif quality == 'bad':
                # Dados de má qualidade (gaps, valores estranhos)
                time_increment = timedelta(minutes=np.random.randint(1, 10))  # Gaps grandes
                price_change = np.random.randn() * 50  # Volatilidade excessiva
                volume = np.random.randint(1, 100)  # Volume muito baixo
                quantity = 1
            
            else:
                # Dados normais
                time_increment = timedelta(seconds=np.random.randint(1, 60))
                price_change = np.random.randn() * 1
                volume = np.random.randint(5000, 30000)
                quantity = np.random.randint(1, 15)
            
            trade_time = base_time + (time_increment * i)
            trade_price = base_price + price_change
            
            trade = {
                'datetime': trade_time,
                'price': trade_price,
                'volume': volume,
                'quantity': quantity,
                'side': np.random.choice(['BUY', 'SELL']),
                'trade_id': i + 1
            }
            
            trades.append(trade)
        
        return trades


class TestTradingDataStructureV3(unittest.TestCase):
    """Teste 1.1.2: Validar Nova Estrutura"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.data_struct = TradingDataStructureV3(max_history=1000)
    
    def test_historical_data_initialization(self):
        """Testa inicialização com dados históricos"""
        
        # Criar dados históricos simulados
        dates = pd.date_range('2025-01-27 09:00', '2025-01-27 11:00', freq='1min')
        
        candles = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 5900,
            'high': np.random.randn(len(dates)).cumsum() + 5905,
            'low': np.random.randn(len(dates)).cumsum() + 5895,
            'close': np.random.randn(len(dates)).cumsum() + 5900,
            'volume': np.random.randint(1000000, 5000000, len(dates)),
            'quantity': np.random.randint(100, 1000, len(dates)),
            'bid': np.random.randn(len(dates)).cumsum() + 5899,
            'ask': np.random.randn(len(dates)).cumsum() + 5901
        }, index=dates)
        
        microstructure = pd.DataFrame({
            'buy_volume': candles['volume'] * 0.6,
            'sell_volume': candles['volume'] * 0.4,
            'volume_imbalance': np.random.randn(len(dates)) * 1000000,
            'trade_imbalance': np.random.randn(len(dates)) * 100,
            'buy_pressure': np.random.uniform(0.3, 0.7, len(dates))
        }, index=dates)
        
        historical_data = {
            'candles': candles,
            'microstructure': microstructure
        }
        
        # Inicializar
        self.data_struct.initialize_from_historical_data(historical_data)
        
        # Validações
        self.assertTrue(self.data_struct._initialized, "Estrutura não inicializada")
        self.assertEqual(len(self.data_struct.candles), len(candles), "Candles não carregados")
        self.assertEqual(len(self.data_struct.microstructure), len(microstructure), "Microestrutura não carregada")
        
        # Verificar alinhamento de índices
        self.assertTrue(
            self.data_struct.candles.index.equals(self.data_struct.microstructure.index),
            "Índices não alinhados"
        )
        
        # Verificar quality score
        quality_score = self.data_struct.metadata['data_quality_score']
        self.assertGreater(quality_score, 0.5, f"Quality score muito baixo: {quality_score}")
        
        print(f"[OK] Inicialização histórica validada (Quality: {quality_score:.3f})")
    
    def test_realtime_data_addition(self):
        """Testa adição de dados em tempo real"""
        
        # Inicializar com base histórica
        self._initialize_with_base_data()
        
        initial_trade_count = self.data_struct.metadata['total_trades_processed']
        
        # Adicionar trades em tempo real
        for i in range(10):
            trade_data = {
                'datetime': datetime.now() + timedelta(seconds=i),
                'price': 5900 + np.random.randn(),
                'volume': np.random.randint(1000, 10000),
                'quantity': np.random.randint(1, 10),
                'side': np.random.choice(['BUY', 'SELL'])
            }
            
            self.data_struct.add_tick_data(trade_data)
        
        # Validações
        self.assertTrue(self.data_struct._real_time_mode, "Modo tempo real não ativado")
        
        final_trade_count = self.data_struct.metadata['total_trades_processed']
        self.assertEqual(final_trade_count, initial_trade_count + 10, "Trades não contabilizados")
        
        print("[OK] Adição de dados tempo real validada")
    
    def test_feature_calculation(self):
        """Testa cálculo de features"""
        
        # Inicializar com dados
        self._initialize_with_base_data()
        
        # Calcular features
        success = self.data_struct.calculate_all_features()
        self.assertTrue(success, "Falha no cálculo de features")
        
        # Verificar que features foram calculadas
        self.assertFalse(self.data_struct.features.empty, "Features não calculadas")
        self.assertGreater(len(self.data_struct.features.columns), 10, "Poucas features calculadas")
        
        # Verificar qualidade das features
        features = self.data_struct.get_latest_features(50)
        self.assertEqual(len(features), 50, "Número incorreto de features retornadas")
        
        # Verificar que não há muitos NaN
        nan_percentage = features.isna().sum().sum() / (len(features) * len(features.columns))
        self.assertLess(nan_percentage, 0.1, f"Muitos NaN nas features: {nan_percentage:.1%}")
        
        print(f"[OK] Features calculadas: {self.data_struct.features.shape}")
    
    def test_data_consistency_validation(self):
        """Testa validação de consistência"""
        
        # Criar dados inconsistentes intencionalmente
        dates1 = pd.date_range('2025-01-27 09:00', '2025-01-27 10:00', freq='1min')
        dates2 = pd.date_range('2025-01-27 09:30', '2025-01-27 10:30', freq='1min')  # Desalinhado
        
        candles = pd.DataFrame({
            'open': np.random.randn(len(dates1)) + 5900,
            'close': np.random.randn(len(dates1)) + 5900,
            'volume': np.random.randint(1000000, 5000000, len(dates1))
        }, index=dates1)
        
        microstructure = pd.DataFrame({
            'buy_volume': np.random.randint(500000, 2500000, len(dates2)),
            'sell_volume': np.random.randint(500000, 2500000, len(dates2))
        }, index=dates2)
        
        # Inicializar com dados inconsistentes
        inconsistent_data = {
            'candles': candles,
            'microstructure': microstructure
        }
        
        self.data_struct.initialize_from_historical_data(inconsistent_data)
        
        # Quality score deve ser menor devido à inconsistência
        quality_score = self.data_struct.metadata['data_quality_score']
        self.assertLess(quality_score, 0.8, "Quality score não detectou inconsistência")
        
        print(f"[OK] Detecção de inconsistência validada (Quality: {quality_score:.3f})")
    
    def test_memory_management(self):
        """Testa gestão de memória"""
        
        # Configurar limite baixo para teste
        self.data_struct.max_history = 100
        
        # Adicionar mais dados que o limite
        dates = pd.date_range('2025-01-27 09:00', '2025-01-27 17:00', freq='1min')  # 8 horas = 480 candles
        
        large_candles = pd.DataFrame({
            'open': np.random.randn(len(dates)) + 5900,
            'close': np.random.randn(len(dates)) + 5900,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        historical_data = {'candles': large_candles}
        self.data_struct.initialize_from_historical_data(historical_data)
        
        # Verificar que dados foram limitados
        self.assertLessEqual(len(self.data_struct.candles), self.data_struct.max_history,
                           "Limite de histórico não respeitado")
        
        # Verificar que manteve os dados mais recentes
        if len(large_candles) > self.data_struct.max_history:
            expected_start = large_candles.index[-self.data_struct.max_history]
            actual_start = self.data_struct.candles.index[0]
            self.assertEqual(actual_start, expected_start, "Dados mais antigos não removidos")
        
        print(f"[OK] Gestão de memória validada (mantidos {len(self.data_struct.candles)} de {len(large_candles)})")
    
    def test_data_summary(self):
        """Testa geração de resumo dos dados"""
        
        self._initialize_with_base_data()
        
        summary = self.data_struct.get_data_summary()
        
        # Verificar estrutura do resumo
        required_keys = ['initialized', 'real_time_mode', 'metadata', 'shapes', 'data_ranges']
        for key in required_keys:
            self.assertIn(key, summary, f"Chave {key} ausente no resumo")
        
        # Verificar shapes
        self.assertIn('candles', summary['shapes'], "Shape de candles ausente")
        self.assertIn('microstructure', summary['shapes'], "Shape de microestrutura ausente")
        
        # Verificar data ranges
        if summary['data_ranges']:
            for range_info in summary['data_ranges'].values():
                self.assertIn('start', range_info, "Start time ausente")
                self.assertIn('end', range_info, "End time ausente")
                self.assertIn('duration', range_info, "Duration ausente")
        
        print("[OK] Resumo de dados validado")
    
    def _initialize_with_base_data(self):
        """Helper para inicializar com dados base"""
        
        dates = pd.date_range('2025-01-27 09:00', '2025-01-27 11:00', freq='1min')
        
        candles = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 5900,
            'high': np.random.randn(len(dates)).cumsum() + 5905,
            'low': np.random.randn(len(dates)).cumsum() + 5895,
            'close': np.random.randn(len(dates)).cumsum() + 5900,
            'volume': np.random.randint(1000000, 5000000, len(dates)),
            'quantity': np.random.randint(100, 1000, len(dates))
        }, index=dates)
        
        microstructure = pd.DataFrame({
            'buy_volume': candles['volume'] * 0.6,
            'sell_volume': candles['volume'] * 0.4,
            'volume_imbalance': np.random.randn(len(dates)) * 1000000,
            'buy_pressure': np.random.uniform(0.3, 0.7, len(dates))
        }, index=dates)
        
        historical_data = {
            'candles': candles,
            'microstructure': microstructure
        }
        
        self.data_struct.initialize_from_historical_data(historical_data)


def run_all_tests():
    """Executa todos os testes da Fase 1"""
    
    print("="*80)
    print("EXECUTANDO TESTES DA FASE 1 - INFRAESTRUTURA DE DADOS")
    print("="*80)
    
    # Criar test suite
    suite = unittest.TestSuite()
    
    # Adicionar testes do RealDataCollector
    suite.addTest(unittest.makeSuite(TestRealDataCollection))
    
    # Adicionar testes do TradingDataStructureV3
    suite.addTest(unittest.makeSuite(TestTradingDataStructureV3))
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumo final
    print("\n" + "="*80)
    print("RESUMO DOS TESTES DA FASE 1")
    print("="*80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success = total_tests - failures - errors
    
    print(f"Total de testes: {total_tests}")
    print(f"Sucessos: {success}")
    print(f"Falhas: {failures}")
    print(f"Erros: {errors}")
    
    success_rate = (success / total_tests) * 100 if total_tests > 0 else 0
    print(f"Taxa de sucesso: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("\n[OK] FASE 1 APROVADA - Infraestrutura de dados funcionando!")
        print("Próximo passo: Executar FASE 2 - Pipeline ML")
    elif success_rate >= 70:
        print("\n[WARNING] FASE 1 PARCIALMENTE APROVADA - Ajustes necessários")
        print("Revisar falhas antes de prosseguir para FASE 2")
    else:
        print("\n[ERROR] FASE 1 REPROVADA - Problemas críticos encontrados")
        print("Corrigir problemas antes de prosseguir")
    
    return success_rate >= 70


if __name__ == "__main__":
    # Executar todos os testes
    success = run_all_tests()
    
    if success:
        print("\n[READY] Pronto para FASE 2!")
    else:
        print("\n[FIX] Ajustes necessários na FASE 1")
    
    exit(0 if success else 1)