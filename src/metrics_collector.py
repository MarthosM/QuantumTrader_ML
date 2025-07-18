"""
Coletor de métricas do sistema de trading
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque
import numpy as np


class MetricsCollector:
    """Coleta e gerencia métricas do sistema"""
    
    def __init__(self):
        """Inicializa o coletor de métricas"""
        self.start_time = time.time()
        
        # Contadores básicos
        self.metrics = {
            'trades_processed': 0,
            'predictions_made': 0,
            'signals_generated': 0,
            'signals_executed': 0,
            'errors': deque(maxlen=100),  # Últimos 100 erros
            'warnings': deque(maxlen=100)
        }
        
        # Performance trading
        self.trading_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'peak_balance': 100000.0,
            'trades_history': deque(maxlen=100)
        }
        
        # Histórico de predições
        self.prediction_history = deque(maxlen=1000)
        self.signal_history = deque(maxlen=100)
        
        # Performance do sistema
        self.system_metrics = {
            'last_update': datetime.now(),
            'updates_per_second': 0,
            'ml_processing_time': deque(maxlen=100),
            'feature_calc_time': deque(maxlen=100)
        }
        
        # Estatísticas de preço
        self.price_history = deque(maxlen=1000)
        self.current_price = 0
        self.session_high = 0
        self.session_low = float('inf')
        
    def record_trade(self):
        """Registra um trade processado"""
        self.metrics['trades_processed'] += 1
        self.system_metrics['last_update'] = datetime.now()
        
    def record_price(self, price: float):
        """Registra preço atual"""
        self.current_price = price
        self.price_history.append({
            'price': price,
            'timestamp': datetime.now()
        })
        
        # Atualizar máxima e mínima
        if price > self.session_high:
            self.session_high = price
        if price < self.session_low:
            self.session_low = price
            
    def record_prediction(self, prediction: Dict):
        """Registra uma predição ML"""
        self.metrics['predictions_made'] += 1
        
        # Adicionar timestamp se não tiver
        if 'timestamp' not in prediction:
            prediction['timestamp'] = datetime.now()
            
        self.prediction_history.append(prediction)
        
    def record_signal(self, signal: Dict):
        """Registra um sinal gerado"""
        self.metrics['signals_generated'] += 1
        self.signal_history.append({
            **signal,
            'timestamp': datetime.now()
        })
        
    def record_execution(self, execution: Dict):
        """Registra execução de ordem"""
        self.metrics['signals_executed'] += 1
        self.trading_metrics['total_trades'] += 1
        
    def record_position_closed(self, position: Dict, closing_price: float):
        """Registra fechamento de posição"""
        entry_price = position.get('entry_price', 0)
        side = position.get('side', 'buy')
        size = position.get('size', 1)
        
        # Calcular P&L
        if side == 'buy':
            pnl = (closing_price - entry_price) * size
        else:
            pnl = (entry_price - closing_price) * size
            
        # Atualizar métricas
        self.trading_metrics['total_pnl'] += pnl
        
        if pnl > 0:
            self.trading_metrics['winning_trades'] += 1
        else:
            self.trading_metrics['losing_trades'] += 1
            
        # Adicionar ao histórico
        self.trading_metrics['trades_history'].append({
            'entry_price': entry_price,
            'exit_price': closing_price,
            'side': side,
            'pnl': pnl,
            'timestamp': datetime.now()
        })
        
        # Atualizar drawdown
        current_balance = 100000 + self.trading_metrics['total_pnl']
        if current_balance > self.trading_metrics['peak_balance']:
            self.trading_metrics['peak_balance'] = current_balance
            self.trading_metrics['current_drawdown'] = 0
        else:
            drawdown = (self.trading_metrics['peak_balance'] - current_balance) / self.trading_metrics['peak_balance']
            self.trading_metrics['current_drawdown'] = drawdown
            if drawdown > self.trading_metrics['max_drawdown']:
                self.trading_metrics['max_drawdown'] = drawdown
                
    def record_error(self, error: str, error_type: str = 'general'):
        """Registra um erro"""
        self.metrics['errors'].append({
            'timestamp': datetime.now(),
            'type': error_type,
            'message': error
        })
        
    def record_warning(self, warning: str):
        """Registra um aviso"""
        self.metrics['warnings'].append({
            'timestamp': datetime.now(),
            'message': warning
        })
        
    def record_processing_time(self, process_type: str, duration: float):
        """Registra tempo de processamento"""
        if process_type == 'ml':
            self.system_metrics['ml_processing_time'].append(duration)
        elif process_type == 'features':
            self.system_metrics['feature_calc_time'].append(duration)
            
    def get_summary(self) -> Dict:
        """Retorna resumo das métricas"""
        uptime_seconds = max(time.time() - self.start_time, 0.001)  # Evitar divisão por zero
        
        # Calcular taxa de atualização
        if self.metrics['trades_processed'] > 0:
            updates_per_second = self.metrics['trades_processed'] / uptime_seconds
        else:
            updates_per_second = 0
            
        # Calcular win rate
        total_closed = self.trading_metrics['winning_trades'] + self.trading_metrics['losing_trades']
        if total_closed > 0:
            win_rate = self.trading_metrics['winning_trades'] / total_closed
        else:
            win_rate = 0
            
        # Calcular médias de processamento
        avg_ml_time = np.mean(self.system_metrics['ml_processing_time']) if self.system_metrics['ml_processing_time'] else 0
        avg_feature_time = np.mean(self.system_metrics['feature_calc_time']) if self.system_metrics['feature_calc_time'] else 0
        
        return {
            'uptime': self._format_uptime(uptime_seconds),
            'uptime_seconds': uptime_seconds,
            'trades_processed': self.metrics['trades_processed'],
            'trades_per_second': round(updates_per_second, 2),
            'predictions_made': self.metrics['predictions_made'],
            'signals_generated': self.metrics['signals_generated'],
            'signals_executed': self.metrics['signals_executed'],
            'current_price': self.current_price,
            'session_high': self.session_high,
            'session_low': self.session_low if self.session_low != float('inf') else 0,
            'total_trades': self.trading_metrics['total_trades'],
            'winning_trades': self.trading_metrics['winning_trades'],
            'losing_trades': self.trading_metrics['losing_trades'],
            'win_rate': round(win_rate * 100, 2),
            'total_pnl': round(self.trading_metrics['total_pnl'], 2),
            'current_drawdown': round(self.trading_metrics['current_drawdown'] * 100, 2),
            'max_drawdown': round(self.trading_metrics['max_drawdown'] * 100, 2),
            'avg_ml_time_ms': round(avg_ml_time * 1000, 2),
            'avg_feature_time_ms': round(avg_feature_time * 1000, 2),
            'errors_count': len(self.metrics['errors']),
            'warnings_count': len(self.metrics['warnings'])
        }
        
    def get_recent_predictions(self, count: int = 5) -> List[Dict]:
        """Retorna predições recentes"""
        return list(self.prediction_history)[-count:]
        
    def get_recent_signals(self, count: int = 5) -> List[Dict]:
        """Retorna sinais recentes"""
        return list(self.signal_history)[-count:]
        
    def get_recent_trades(self, count: int = 5) -> List[Dict]:
        """Retorna trades recentes"""
        return list(self.trading_metrics['trades_history'])[-count:]
        
    def _format_uptime(self, seconds: float) -> str:
        """Formata tempo de uptime"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
    def _calculate_uptime(self) -> float:
        """Calcula uptime em segundos"""
        return time.time() - self.start_time
        
    def _calculate_trade_rate(self) -> float:
        """Calcula taxa de trades por minuto"""
        uptime_minutes = self._calculate_uptime() / 60
        if uptime_minutes > 0:
            return self.metrics['trades_processed'] / uptime_minutes
        return 0
        
    def _calculate_accuracy(self) -> float:
        """Calcula acurácia das predições (placeholder)"""
        # Implementar lógica real de acurácia quando tivermos feedback
        return 0.0