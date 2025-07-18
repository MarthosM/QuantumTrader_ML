"""
Testes para Etapa 5 - Geração de Sinais e Estratégia
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from signal_generator import SignalGenerator
from risk_manager import RiskManager
from strategy_engine import StrategyEngine


class MockTradingDataStructure:
    """Mock da estrutura de dados para testes"""
    def __init__(self, n_candles=100):
        dates = pd.date_range(end=datetime.now(), periods=n_candles, freq='1min')
        
        # Criar dados OHLCV realistas
        base_price = 5000
        prices = base_price + np.random.randn(n_candles).cumsum() * 2
        
        self.candles = pd.DataFrame({
            'open': prices + np.random.randn(n_candles) * 0.5,
            'high': prices + abs(np.random.randn(n_candles)) * 2,
            'low': prices - abs(np.random.randn(n_candles)) * 2,
            'close': prices,
            'volume': np.random.randint(50, 200, n_candles)
        }, index=dates)
        
        # Ajustar OHLC para ser consistente
        self.candles['high'] = self.candles[['open', 'high', 'close']].max(axis=1)
        self.candles['low'] = self.candles[['open', 'low', 'close']].min(axis=1)
        
        # Indicadores básicos
        self.indicators = pd.DataFrame({
            'atr': self.candles['high'].rolling(14).mean() - self.candles['low'].rolling(14).mean()
        }, index=dates)
        
        self.features = pd.DataFrame(index=dates)


class TestSignalGenerator(unittest.TestCase):
    """Testes para o SignalGenerator"""
    
    def setUp(self):
        """Configuração inicial"""
        self.config = {
            'direction_threshold': 0.3,
            'magnitude_threshold': 0.0001,
            'confidence_threshold': 0.6,
            'risk_per_trade': 0.02,
            'point_value': 0.5,
            'min_stop_points': 5,
            'default_risk_reward': 2.0
        }
        self.generator = SignalGenerator(self.config)
        
    def test_initialization(self):
        """Testa inicialização"""
        self.assertIsNotNone(self.generator)
        self.assertEqual(self.generator.direction_threshold, 0.3)
        self.assertEqual(self.generator.point_value, 0.5)
        
    def test_generate_buy_signal(self):
        """Testa geração de sinal de compra"""
        # Predição de alta
        prediction = {
            'direction': 0.8,
            'magnitude': 0.002,
            'confidence': 0.85,
            'regime': 'trend_up',
            'regime_confidence': 0.9
        }
        
        # Dados de mercado
        market_data = MockTradingDataStructure()
        
        # Gerar sinal
        signal = self.generator.generate_signal(prediction, market_data)
        
        # Verificar
        self.assertEqual(signal['action'], 'buy')
        self.assertIsNotNone(signal['entry_price'])
        self.assertIsNotNone(signal['stop_loss'])
        self.assertIsNotNone(signal['take_profit'])
        self.assertGreater(signal['confidence'], 0)
        self.assertGreater(signal['risk_reward'], 0)
        
        # Verificar que stop < entry < take para compra
        self.assertLess(signal['stop_loss'], signal['entry_price'])
        self.assertGreater(signal['take_profit'], signal['entry_price'])
        
    def test_generate_sell_signal(self):
        """Testa geração de sinal de venda"""
        # Predição de baixa
        prediction = {
            'direction': -0.7,
            'magnitude': 0.0015,
            'confidence': 0.75,
            'regime': 'trend_down',
            'regime_confidence': 0.85
        }
        
        market_data = MockTradingDataStructure()
        signal = self.generator.generate_signal(prediction, market_data)
        
        # Verificar
        self.assertEqual(signal['action'], 'sell')
        
        # Verificar que stop > entry > take para venda
        self.assertGreater(signal['stop_loss'], signal['entry_price'])
        self.assertLess(signal['take_profit'], signal['entry_price'])
        
    def test_threshold_validation(self):
        """Testa validação de thresholds"""
        market_data = MockTradingDataStructure()
        
        # Teste 1: Direção baixa
        prediction = {
            'direction': 0.1,  # Abaixo do threshold
            'magnitude': 0.002,
            'confidence': 0.85
        }
        signal = self.generator.generate_signal(prediction, market_data)
        self.assertEqual(signal['action'], 'none')
        self.assertEqual(signal['reason'], 'direction_below_threshold')
        
        # Teste 2: Magnitude baixa
        prediction = {
            'direction': 0.8,
            'magnitude': 0.00005,  # Abaixo do threshold
            'confidence': 0.85
        }
        signal = self.generator.generate_signal(prediction, market_data)
        self.assertEqual(signal['action'], 'none')
        self.assertEqual(signal['reason'], 'magnitude_below_threshold')
        
        # Teste 3: Confiança baixa
        prediction = {
            'direction': 0.8,
            'magnitude': 0.002,
            'confidence': 0.4  # Abaixo do threshold
        }
        signal = self.generator.generate_signal(prediction, market_data)
        self.assertEqual(signal['action'], 'none')
        self.assertEqual(signal['reason'], 'confidence_below_threshold')
        
    def test_stop_loss_calculation(self):
        """Testa cálculo de stop loss"""
        prediction = {
            'direction': 0.8,
            'magnitude': 0.002,
            'confidence': 0.85
        }
        
        market_data = MockTradingDataStructure()
        signal = self.generator.generate_signal(prediction, market_data)
        
        # Verificar stop mínimo
        stop_points = signal['stop_points']
        self.assertGreaterEqual(stop_points, self.config['min_stop_points'])
        
    def test_risk_reward_calculation(self):
        """Testa cálculo de risk/reward"""
        prediction = {
            'direction': 0.8,
            'magnitude': 0.003,
            'confidence': 0.85
        }
        
        market_data = MockTradingDataStructure()
        signal = self.generator.generate_signal(prediction, market_data)
        
        # Verificar risk/reward
        self.assertGreaterEqual(signal['risk_reward'], 1.0)
        

class TestRiskManager(unittest.TestCase):
    """Testes para o RiskManager"""
    
    def setUp(self):
        """Configuração inicial"""
        self.config = {
            'max_daily_loss': 0.05,
            'max_positions': 2,
            'max_risk_per_trade': 0.02,
            'min_risk_reward': 1.5
        }
        self.risk_mgr = RiskManager(self.config)
        
    def test_initialization(self):
        """Testa inicialização"""
        self.assertIsNotNone(self.risk_mgr)
        self.assertEqual(self.risk_mgr.max_positions, 2)
        self.assertEqual(len(self.risk_mgr.open_positions), 0)
        
    def test_validate_signal_approved(self):
        """Testa aprovação de sinal válido"""
        signal = {
            'action': 'buy',
            'entry_price': 5000,
            'stop_loss': 4990,
            'take_profit': 5020,
            'risk_reward': 3.0,
            'position_size': 1
        }
        
        # Mock do horário de trading para sempre retornar True
        original_check = self.risk_mgr._check_trading_hours
        self.risk_mgr._check_trading_hours = lambda: True
        
        try:
            is_valid, reason = self.risk_mgr.validate_signal(signal, 100000)
            
            self.assertTrue(is_valid)
            self.assertEqual(reason, 'approved')
        finally:
            # Restaurar método original
            self.risk_mgr._check_trading_hours = original_check
        
    def test_validate_signal_max_positions(self):
        """Testa rejeição por limite de posições"""
        # Adicionar posições existentes
        self.risk_mgr.open_positions = [
            {'id': '1', 'action': 'buy'},
            {'id': '2', 'action': 'sell'}
        ]
        
        signal = {
            'action': 'buy',
            'entry_price': 5000,
            'stop_loss': 4990,
            'risk_reward': 2.0,
            'position_size': 1
        }
        
        is_valid, reason = self.risk_mgr.validate_signal(signal, 100000)
        
        self.assertFalse(is_valid)
        self.assertEqual(reason, 'max_positions_reached')
        
    def test_validate_signal_low_risk_reward(self):
        """Testa rejeição por risk/reward baixo"""
        signal = {
            'action': 'buy',
            'entry_price': 5000,
            'stop_loss': 4990,
            'take_profit': 5010,
            'risk_reward': 1.0,  # Abaixo do mínimo
            'position_size': 1
        }
        
        is_valid, reason = self.risk_mgr.validate_signal(signal, 100000)
        
        self.assertFalse(is_valid)
        self.assertEqual(reason, 'risk_reward_too_low')
        
    def test_register_and_close_position(self):
        """Testa registro e fechamento de posição"""
        # Registrar posição
        position = {
            'id': 'TEST001',
            'action': 'buy',
            'entry_price': 5000,
            'stop_loss': 4990,
            'take_profit': 5020,
            'position_size': 1
        }
        
        self.risk_mgr.register_position(position)
        
        self.assertEqual(len(self.risk_mgr.open_positions), 1)
        self.assertEqual(self.risk_mgr.daily_trades, 1)
        
        # Fechar posição com lucro
        closed = self.risk_mgr.close_position('0', 5015, 'take_profit')
        
        self.assertIsNotNone(closed)
        self.assertEqual(len(self.risk_mgr.open_positions), 0)
        self.assertEqual(len(self.risk_mgr.closed_positions), 1)
        self.assertGreater(closed['pnl'], 0)
        
    def test_daily_loss_limit(self):
        """Testa limite de perda diária"""
        # Simular perda
        self.risk_mgr.daily_pnl = -6000  # 6% de perda em conta de 100k
        
        signal = {
            'action': 'buy',
            'entry_price': 5000,
            'stop_loss': 4990,
            'risk_reward': 2.0,
            'position_size': 1
        }
        
        is_valid, reason = self.risk_mgr.validate_signal(signal, 100000)
        
        self.assertFalse(is_valid)
        self.assertEqual(reason, 'daily_loss_limit_reached')
        
    def test_risk_metrics(self):
        """Testa cálculo de métricas"""
        # Adicionar algumas posições fechadas
        self.risk_mgr.closed_positions = [
            {'pnl': 100, 'pnl_pct': 0.02},
            {'pnl': -50, 'pnl_pct': -0.01},
            {'pnl': 75, 'pnl_pct': 0.015}
        ]
        
        metrics = self.risk_mgr.get_risk_metrics()
        
        self.assertEqual(metrics['total_trades'], 3)
        self.assertAlmostEqual(metrics['win_rate'], 2/3, places=2)
        self.assertGreater(metrics['profit_factor'], 1.0)
        

class TestStrategyEngine(unittest.TestCase):
    """Testes para o StrategyEngine"""
    
    def setUp(self):
        """Configuração inicial"""
        signal_config = {
            'direction_threshold': 0.3,
            'magnitude_threshold': 0.0001,
            'confidence_threshold': 0.6
        }
        
        risk_config = {
            'max_daily_loss': 0.05,
            'max_positions': 2
        }
        
        self.signal_gen = SignalGenerator(signal_config)
        self.risk_mgr = RiskManager(risk_config)
        self.engine = StrategyEngine(self.signal_gen, self.risk_mgr)
        
    def test_initialization(self):
        """Testa inicialização"""
        self.assertIsNotNone(self.engine)
        self.assertEqual(len(self.engine.active_positions), 0)
        
    def test_process_prediction(self):
        """Testa processamento completo de predição"""
        # Mock do horário de trading para sempre retornar True  
        original_check = self.risk_mgr._check_trading_hours
        self.risk_mgr._check_trading_hours = lambda: True
        
        try:
            # Predição válida
            prediction = {
                'direction': 0.8,
                'magnitude': 0.002,
                'confidence': 0.85
            }
            
            market_data = MockTradingDataStructure()
            account_info = {'balance': 100000}
            
            # Processar
            signal = self.engine.process_prediction(prediction, market_data, account_info)
            
            # Verificar
            self.assertIsNotNone(signal)
            if signal is not None:
                self.assertEqual(signal['action'], 'buy')
                self.assertTrue(signal.get('approved', False))
                self.assertIn('id', signal)
                
                # Verificar que foi armazenado
                self.assertEqual(self.engine.current_signal, signal)
                self.assertEqual(len(self.engine.signal_history), 1)
        finally:
            # Restaurar método original
            self.risk_mgr._check_trading_hours = original_check
        
    def test_execute_signal(self):
        """Testa execução de sinal"""
        signal = {
            'id': 'TEST001',
            'action': 'buy',
            'entry_price': 5000,
            'stop_loss': 4990,
            'take_profit': 5020,
            'position_size': 1,
            'confidence': 0.85,
            'reason': 'test'
        }
        
        # Executar
        success = self.engine.execute_signal(signal)
        
        self.assertTrue(success)
        self.assertEqual(len(self.engine.active_positions), 1)
        self.assertEqual(len(self.risk_mgr.open_positions), 1)
        
    def test_check_positions(self):
        """Testa verificação de posições"""
        # Adicionar posição ativa
        position = {
            'id': 'TEST001',
            'action': 'buy',
            'entry_price': 5000,
            'stop_loss': 4990,
            'take_profit': 5020,
            'position_size': 1
        }
        
        self.engine.active_positions.append(position)
        self.risk_mgr.open_positions.append(position)
        
        # Criar dados com preço atingindo take profit
        market_data = MockTradingDataStructure(10)
        # Usar .loc para evitar warnings do pandas
        market_data.candles.loc[market_data.candles.index[-1], 'high'] = 5025
        market_data.candles.loc[market_data.candles.index[-1], 'close'] = 5022
        
        # Verificar posições
        closed_positions = self.engine.check_positions(market_data)
        
        self.assertEqual(len(closed_positions), 1)
        self.assertEqual(closed_positions[0]['exit_reason'], 'take_profit')
        self.assertEqual(len(self.engine.active_positions), 0)
        
    def test_strategy_stats(self):
        """Testa estatísticas da estratégia"""
        # Adicionar alguns sinais ao histórico
        self.engine.signal_history = [
            {'action': 'buy', 'approved': True},
            {'action': 'sell', 'approved': True},
            {'action': 'buy', 'rejected': True},
        ]
        
        stats = self.engine.get_strategy_stats()
        
        self.assertEqual(stats['total_signals'], 3)
        self.assertEqual(stats['signals_approved'], 2)
        self.assertEqual(stats['signals_rejected'], 1)
        self.assertAlmostEqual(stats['approval_rate'], 2/3, places=2)
        

if __name__ == '__main__':
    unittest.main()