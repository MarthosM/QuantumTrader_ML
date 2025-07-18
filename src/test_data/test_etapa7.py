"""
Testes para Etapa 7 - Monitoramento e Métricas
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime
import time
import sys
import os

# Adicionar o diretório src ao path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics_collector import MetricsCollector
from trading_monitor import TradingMonitor


class TestMetricsCollector(unittest.TestCase):
    """Testes do coletor de métricas"""
    
    def setUp(self):
        self.metrics = MetricsCollector()
        
    def test_basic_metrics(self):
        """Testa métricas básicas"""
        # Registrar alguns eventos
        self.metrics.record_trade()
        self.metrics.record_trade()
        self.metrics.record_trade()
        
        self.metrics.record_prediction({
            'direction': 0.8,
            'magnitude': 0.002,
            'confidence': 0.85
        })
        
        self.metrics.record_signal({
            'action': 'buy',
            'price': 5000
        })
        
        # Verificar contadores
        self.assertEqual(self.metrics.metrics['trades_processed'], 3)
        self.assertEqual(self.metrics.metrics['predictions_made'], 1)
        self.assertEqual(self.metrics.metrics['signals_generated'], 1)
        
    def test_price_tracking(self):
        """Testa rastreamento de preços"""
        # Registrar preços
        prices = [5000, 5010, 4995, 5020, 5005]
        
        for price in prices:
            self.metrics.record_price(price)
            
        # Verificar
        self.assertEqual(self.metrics.current_price, 5005)
        self.assertEqual(self.metrics.session_high, 5020)
        self.assertEqual(self.metrics.session_low, 4995)
        self.assertEqual(len(self.metrics.price_history), 5)
        
    def test_trading_metrics(self):
        """Testa métricas de trading"""
        # Simular trade vencedor
        position = {
            'entry_price': 5000,
            'side': 'buy',
            'size': 1
        }
        
        self.metrics.record_position_closed(position, 5050)
        
        # Verificar
        self.assertEqual(self.metrics.trading_metrics['winning_trades'], 1)
        self.assertEqual(self.metrics.trading_metrics['total_pnl'], 50)
        
        # Simular trade perdedor
        position2 = {
            'entry_price': 5050,
            'side': 'sell',
            'size': 1
        }
        
        self.metrics.record_position_closed(position2, 5070)
        
        # Verificar
        self.assertEqual(self.metrics.trading_metrics['losing_trades'], 1)
        self.assertEqual(self.metrics.trading_metrics['total_pnl'], 30)  # 50 - 20
        
    def test_error_tracking(self):
        """Testa rastreamento de erros"""
        # Registrar erros
        self.metrics.record_error("Connection lost", "network")
        self.metrics.record_error("Model prediction failed", "ml")
        
        # Verificar
        self.assertEqual(len(self.metrics.metrics['errors']), 2)
        self.assertEqual(self.metrics.metrics['errors'][0]['type'], 'network')
        
    def test_processing_time(self):
        """Testa registro de tempo de processamento"""
        # Registrar tempos
        self.metrics.record_processing_time('ml', 0.1)
        self.metrics.record_processing_time('ml', 0.2)
        self.metrics.record_processing_time('features', 0.05)
        
        # Obter resumo
        summary = self.metrics.get_summary()
        
        # Verificar médias
        self.assertAlmostEqual(summary['avg_ml_time_ms'], 150, places=1)
        self.assertAlmostEqual(summary['avg_feature_time_ms'], 50, places=1)
        
    def test_summary_generation(self):
        """Testa geração de resumo"""
        # Adicionar dados diversos
        self.metrics.record_trade()
        self.metrics.record_prediction({'direction': 0.5})
        self.metrics.record_signal({'action': 'buy'})
        self.metrics.record_execution({})
        self.metrics.record_price(5000)
        
        # Obter resumo
        summary = self.metrics.get_summary()
        
        # Verificar campos essenciais
        self.assertIn('uptime', summary)
        self.assertIn('trades_processed', summary)
        self.assertIn('predictions_made', summary)
        self.assertIn('signals_generated', summary)
        self.assertIn('current_price', summary)
        self.assertIn('win_rate', summary)
        self.assertIn('total_pnl', summary)
        
    def test_recent_data_retrieval(self):
        """Testa recuperação de dados recentes"""
        # Adicionar várias predições
        for i in range(10):
            self.metrics.record_prediction({
                'direction': i * 0.1,
                'timestamp': datetime.now()
            })
            
        # Obter recentes
        recent = self.metrics.get_recent_predictions(5)
        
        # Verificar
        self.assertEqual(len(recent), 5)
        self.assertEqual(recent[-1]['direction'], 0.9)
        
    def test_uptime_calculation(self):
        """Testa cálculo de uptime"""
        # Simular tempo passado
        self.metrics.start_time = time.time() - 3661  # 1 hora e 1 segundo
        
        summary = self.metrics.get_summary()
        
        # Verificar formato
        self.assertEqual(summary['uptime'], "01:01:01")
        
    def test_win_rate_calculation(self):
        """Testa cálculo de win rate"""
        # Adicionar trades
        for i in range(7):
            position = {
                'entry_price': 5000,
                'side': 'buy',
                'size': 1
            }
            # 5 vencedores, 2 perdedores
            exit_price = 5010 if i < 5 else 4990
            self.metrics.record_position_closed(position, exit_price)
            
        summary = self.metrics.get_summary()
        
        # Verificar win rate (5/7 = 71.4%)
        self.assertAlmostEqual(summary['win_rate'], 71.4, places=1)
        
    def test_drawdown_calculation(self):
        """Testa cálculo de drawdown"""
        initial_balance = 100000
        
        # Simular ganhos primeiro
        position1 = {'entry_price': 5000, 'side': 'buy', 'size': 1}
        self.metrics.record_position_closed(position1, 5100)  # +100
        
        # Peak deve ser 100100
        self.assertEqual(self.metrics.trading_metrics['peak_balance'], 100100)
        
        # Simular perdas
        position2 = {'entry_price': 5100, 'side': 'buy', 'size': 1}
        self.metrics.record_position_closed(position2, 5050)  # -50
        
        # Drawdown deve ser (100100 - 100050) / 100100 = 0.05%
        summary = self.metrics.get_summary()
        self.assertAlmostEqual(summary['current_drawdown'], 0.05, places=2)


class TestTradingMonitor(unittest.TestCase):
    """Testes do monitor de trading"""
    
    def setUp(self):
        # Mock do sistema
        self.mock_system = Mock()
        self.mock_system.get_status.return_value = {
            'running': True,
            'ticker': 'WDOQ25',
            'active_positions': {}
        }
        self.mock_system.metrics = MetricsCollector()
        
    @patch('tkinter.Tk')
    def test_monitor_creation(self, mock_tk):
        """Testa criação do monitor"""
        monitor = TradingMonitor(self.mock_system)
        
        # Verificar estado inicial
        self.assertFalse(monitor.is_running)
        self.assertIsNone(monitor.root)
        self.assertEqual(monitor.system, self.mock_system)
        
    @patch('tkinter.ttk.Style')
    @patch('tkinter.Tk')
    def test_interface_creation(self, mock_tk, mock_style):
        """Testa criação da interface"""
        # Criar mock root
        mock_root = Mock()
        mock_tk.return_value = mock_root
        
        # Mock do style
        mock_style_instance = Mock()
        mock_style.return_value = mock_style_instance
        
        monitor = TradingMonitor(self.mock_system)
        monitor.root = mock_root
        
        # Criar interface
        monitor._setup_style()
        
        # Verificar que style foi criado
        mock_style.assert_called_once()
        mock_style_instance.theme_use.assert_called_once_with('clam')
        
    def test_update_processing(self):
        """Testa processamento de atualizações"""
        monitor = TradingMonitor(self.mock_system)
        
        # Criar widgets mock
        monitor.widgets = {
            'status_label': Mock(),
            'contract_label': Mock(),
            'uptime_label': Mock(),
            'current_price': Mock(),
            'session_high': Mock(),
            'session_low': Mock(),
            'position_info': Mock(),
            'position_pnl': Mock(),
            'metric_trades_count': Mock(),
            'metric_predictions_count': Mock(),
            'metric_signals_count': Mock(),
            'metric_executions_count': Mock(),
            'metric_win_rate': Mock(),
            'metric_total_pnl': Mock(),
            'metric_drawdown': Mock(),
            'metric_errors': Mock()
        }
        
        # Simular atualização
        status = {
            'running': True,
            'ticker': 'WDOQ25',
            'metrics': {
                'uptime': '01:23:45',
                'current_price': 5000,
                'session_high': 5050,
                'session_low': 4950,
                'trades_processed': 100,
                'predictions_made': 20,
                'signals_generated': 5,
                'signals_executed': 3,
                'win_rate': 66.7,
                'total_pnl': 150.50,
                'current_drawdown': 2.5,
                'errors_count': 0
            }
        }
        
        # Aplicar atualizações
        monitor._apply_updates(status)
        
        # Verificar que widgets foram atualizados
        monitor.widgets['current_price'].config.assert_called_with(text='5000.00')
        monitor.widgets['metric_trades_count'].config.assert_called_with(text='100')
        monitor.widgets['metric_win_rate'].config.assert_called_with(text='66.7%')
        
    def test_position_display(self):
        """Testa exibição de posições"""
        monitor = TradingMonitor(self.mock_system)
        
        # Criar widgets mock
        monitor.widgets = {
            'position_info': Mock(),
            'position_pnl': Mock()
        }
        
        # Simular posição ativa
        status = {
            'active_positions': {
                'WDOQ25': {
                    'side': 'buy',
                    'entry_price': 5000,
                    'size': 2
                }
            },
            'metrics': {
                'current_price': 5050
            }
        }
        
        # Aplicar
        monitor._apply_updates(status)
        
        # Verificar exibição
        monitor.widgets['position_info'].config.assert_called()
        monitor.widgets['position_pnl'].config.assert_called()
        
    def test_prediction_display(self):
        """Testa exibição de predições"""
        monitor = TradingMonitor(self.mock_system)
        
        # Criar widgets mock
        monitor.widgets = {
            'pred_direction': Mock(),
            'pred_magnitude': Mock(),
            'pred_confidence': Mock(),
            'pred_time': Mock()
        }
        
        # Simular predição
        status = {
            'last_prediction': {
                'direction': 0.8,
                'magnitude': 0.002,
                'confidence': 0.85,
                'timestamp': datetime.now()
            }
        }
        
        # Aplicar
        monitor._apply_updates(status)
        
        # Verificar que predição foi exibida
        monitor.widgets['pred_direction'].config.assert_called()
        monitor.widgets['pred_confidence'].config.assert_called()


if __name__ == '__main__':
    unittest.main()