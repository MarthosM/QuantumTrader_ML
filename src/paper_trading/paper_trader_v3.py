"""
Paper Trader V3 - Sistema de Paper Trading em Tempo Real
========================================================

Este módulo implementa um sistema de paper trading que:
- Simula trading em tempo real sem risco financeiro
- Integra com ConnectionManagerV3 para dados reais
- Executa estratégias ML em ambiente simulado
- Gera métricas e relatórios de performance
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
import logging
import json
import threading
import time
from dataclasses import dataclass, asdict
from queue import Queue

# Adicionar src ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from connection.connection_manager_v3 import ConnectionManagerV3
from realtime.realtime_processor_v3 import RealTimeProcessorV3
from ml.prediction_engine_v3 import PredictionEngineV3
from monitoring.system_monitor_v3 import SystemMonitorV3
from backtesting.backtester_v3 import Trade


@dataclass
class PaperOrder:
    """Representa uma ordem no paper trading"""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' ou 'SELL'
    order_type: str  # 'MARKET', 'LIMIT', 'STOP'
    quantity: int
    price: float
    status: str  # 'PENDING', 'FILLED', 'CANCELLED'
    filled_price: Optional[float] = None
    filled_timestamp: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0


class PaperTradingAccount:
    """Conta simulada para paper trading"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, int] = {}  # symbol -> quantity
        self.orders: List[PaperOrder] = []
        self.trades: List[Trade] = []
        self.open_pnl = 0.0
        self.realized_pnl = 0.0
        self._lock = threading.RLock()
        
    def get_position(self, symbol: str) -> int:
        """Retorna posição atual do símbolo"""
        with self._lock:
            return self.positions.get(symbol, 0)
    
    def update_position(self, symbol: str, quantity_change: int):
        """Atualiza posição"""
        with self._lock:
            current = self.positions.get(symbol, 0)
            new_position = current + quantity_change
            
            if new_position == 0:
                self.positions.pop(symbol, None)
            else:
                self.positions[symbol] = new_position
    
    def add_order(self, order: PaperOrder):
        """Adiciona ordem ao histórico"""
        with self._lock:
            self.orders.append(order)
    
    def add_trade(self, trade: Trade):
        """Adiciona trade executado"""
        with self._lock:
            self.trades.append(trade)
            self.realized_pnl += trade.pnl
            self.current_capital += trade.pnl
    
    def get_account_summary(self) -> Dict:
        """Retorna resumo da conta"""
        with self._lock:
            total_pnl = self.realized_pnl + self.open_pnl
            return {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'realized_pnl': self.realized_pnl,
                'open_pnl': self.open_pnl,
                'total_pnl': total_pnl,
                'return_pct': (total_pnl / self.initial_capital) * 100,
                'positions': dict(self.positions),
                'total_trades': len(self.trades),
                'open_orders': sum(1 for o in self.orders if o.status == 'PENDING')
            }


class PaperTraderV3:
    """Sistema de paper trading em tempo real"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o paper trader
        
        Args:
            config: Configurações do paper trading
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Componentes do sistema
        self.connection_manager = ConnectionManagerV3()
        self.realtime_processor = RealTimeProcessorV3()
        self.prediction_engine = PredictionEngineV3()
        self.system_monitor = SystemMonitorV3()
        
        # Conta de paper trading
        self.account = PaperTradingAccount(self.config['initial_capital'])
        
        # Estado do sistema
        self.is_running = False
        self.last_signal_time = {}
        self.current_trades: Dict[str, Trade] = {}  # symbol -> Trade
        
        # Filas de processamento
        self.signal_queue = Queue(maxsize=1000)
        
        # Threads
        self.signal_thread = None
        self.monitor_thread = None
        
        # Callbacks
        self._setup_callbacks()
        
    def _get_default_config(self) -> Dict:
        """Retorna configuração padrão"""
        return {
            'initial_capital': 100000.0,
            'position_size': 1,
            'commission_per_side': 5.0,
            'slippage_ticks': 1,
            'tick_value': 0.5,
            'stop_loss_ticks': 20,
            'take_profit_ticks': 40,
            'max_positions': 1,
            'min_time_between_trades': 60,  # segundos
            'confidence_threshold': 0.65,
            'probability_threshold': 0.60,
            'symbols': ['WDO'],
            'paper_trading_hours': {
                'start': 9,
                'end': 17
            }
        }
    
    def _setup_callbacks(self):
        """Configura callbacks para processar dados em tempo real"""
        # Callback será configurado quando conectar
        pass
    
    def start(self):
        """Inicia o paper trading"""
        self.logger.info("Iniciando Paper Trading V3...")
        
        try:
            # Iniciar componentes
            self.is_running = True
            
            # Iniciar monitoramento
            self.system_monitor.start()
            
            # Iniciar processamento em tempo real
            self.realtime_processor.start()
            
            # Configurar callback para processar features
            self._setup_feature_callback()
            
            # Iniciar threads
            self.signal_thread = threading.Thread(
                target=self._signal_processing_loop,
                name="PaperTradingSignalThread"
            )
            self.signal_thread.start()
            
            self.monitor_thread = threading.Thread(
                target=self._monitor_positions_loop,
                name="PaperTradingMonitorThread"
            )
            self.monitor_thread.start()
            
            # Conectar ao ProfitDLL (simulado para paper trading)
            self._connect_to_market()
            
            self.logger.info("Paper Trading iniciado com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro iniciando paper trading: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Para o paper trading"""
        self.logger.info("Parando Paper Trading...")
        
        self.is_running = False
        
        # Fechar posições abertas
        self._close_all_positions("system_shutdown")
        
        # Parar componentes
        if self.realtime_processor:
            self.realtime_processor.stop()
        
        if self.system_monitor:
            self.system_monitor.stop()
        
        # Aguardar threads
        if self.signal_thread and self.signal_thread.is_alive():
            self.signal_thread.join(timeout=5)
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        # Gerar relatório final
        self._generate_final_report()
        
        self.logger.info("Paper Trading parado")
    
    def _setup_feature_callback(self):
        """Configura callback para quando features são calculadas"""
        def on_features_ready(features: pd.DataFrame):
            """Callback quando features estão prontas"""
            if not self.is_running:
                return
            
            try:
                # Gerar predição
                prediction = self.prediction_engine.predict(features)
                
                if prediction:
                    # Adicionar à fila de sinais
                    self.signal_queue.put({
                        'timestamp': datetime.now(),
                        'features': features,
                        'prediction': prediction
                    })
                    
            except Exception as e:
                self.logger.error(f"Erro no callback de features: {e}")
        
        # Registrar callback (implementação depende do realtime_processor)
        # self.realtime_processor.on_features_ready = on_features_ready
    
    def _connect_to_market(self):
        """Conecta ao mercado (simulado para paper trading)"""
        # Em paper trading, simular conexão ou usar dados históricos
        self.logger.info("Conectando ao mercado simulado...")
        
        # Opção 1: Usar replay de dados históricos
        # Opção 2: Conectar ao ProfitDLL real mas não enviar ordens
        # Por agora, vamos simular com dados sintéticos
        
        self._simulate_market_data()
    
    def _simulate_market_data(self):
        """Simula dados de mercado para teste"""
        def market_simulator():
            """Thread que simula dados de mercado"""
            base_price = 5900.0
            
            while self.is_running:
                # Simular tick
                price_change = np.random.randn() * 2
                current_price = base_price + price_change
                volume = np.random.randint(100, 1000)
                
                trade_data = {
                    'timestamp': datetime.now(),
                    'price': current_price,
                    'volume': volume,
                    'side': 'BUY' if np.random.random() > 0.5 else 'SELL'
                }
                
                # Processar trade
                self.realtime_processor.add_trade(trade_data)
                
                # Aguardar próximo tick (simular frequência real)
                time.sleep(0.1)  # 10 ticks por segundo
        
        simulator_thread = threading.Thread(
            target=market_simulator,
            name="MarketSimulator"
        )
        simulator_thread.daemon = True
        simulator_thread.start()
    
    def _signal_processing_loop(self):
        """Loop de processamento de sinais"""
        self.logger.info("Loop de sinais iniciado")
        
        while self.is_running:
            try:
                # Pegar sinal da fila (timeout para permitir saída)
                signal_data = self.signal_queue.get(timeout=1.0)
                
                # Processar sinal
                self._process_signal(signal_data)
                
            except:
                # Timeout normal, continuar
                pass
    
    def _process_signal(self, signal_data: Dict):
        """Processa um sinal de trading"""
        timestamp = signal_data['timestamp']
        prediction = signal_data['prediction']
        
        # Verificar se está no horário de trading
        if not self._is_trading_hours(timestamp):
            return
        
        # Verificar critérios de trading
        if not self._should_trade(prediction):
            return
        
        # Verificar tempo mínimo entre trades
        symbol = self.config['symbols'][0]  # Por agora, apenas um símbolo
        if symbol in self.last_signal_time:
            time_diff = (timestamp - self.last_signal_time[symbol]).seconds
            if time_diff < self.config['min_time_between_trades']:
                return
        
        # Executar trade
        self._execute_trade(symbol, prediction, timestamp)
        
        # Atualizar último tempo de sinal
        self.last_signal_time[symbol] = timestamp
    
    def _should_trade(self, prediction: Dict) -> bool:
        """Verifica se deve executar o trade"""
        return (prediction.get('confidence', 0) >= self.config['confidence_threshold'] and
                prediction.get('probability', 0) >= self.config['probability_threshold'])
    
    def _is_trading_hours(self, timestamp: datetime) -> bool:
        """Verifica se está no horário de trading"""
        hour = timestamp.hour
        return (self.config['paper_trading_hours']['start'] <= hour < 
                self.config['paper_trading_hours']['end'])
    
    def _execute_trade(self, symbol: str, prediction: Dict, timestamp: datetime):
        """Executa um trade baseado na predição"""
        current_position = self.account.get_position(symbol)
        direction = prediction.get('direction', 0)
        
        # Se já tem posição
        if symbol in self.current_trades:
            current_trade = self.current_trades[symbol]
            
            # Verificar se é sinal contrário
            current_side = 1 if current_trade.side == 'BUY' else -1
            if direction != current_side:
                # Fechar posição atual
                self._close_position(symbol, timestamp, 'signal')
                # Abrir nova posição
                self._open_position(symbol, prediction, timestamp)
        else:
            # Sem posição, abrir nova
            if len(self.current_trades) < self.config['max_positions']:
                self._open_position(symbol, prediction, timestamp)
    
    def _open_position(self, symbol: str, prediction: Dict, timestamp: datetime):
        """Abre uma nova posição"""
        side = 'BUY' if prediction['direction'] > 0 else 'SELL'
        
        # Obter preço atual (simulado)
        current_price = self._get_current_price(symbol)
        
        # Aplicar slippage
        slippage = self.config['slippage_ticks'] * self.config['tick_value']
        if side == 'BUY':
            entry_price = current_price + slippage
            stop_loss = entry_price - (self.config['stop_loss_ticks'] * self.config['tick_value'])
            take_profit = entry_price + (self.config['take_profit_ticks'] * self.config['tick_value'])
        else:
            entry_price = current_price - slippage
            stop_loss = entry_price + (self.config['stop_loss_ticks'] * self.config['tick_value'])
            take_profit = entry_price - (self.config['take_profit_ticks'] * self.config['tick_value'])
        
        # Criar ordem
        order = PaperOrder(
            order_id=f"PO_{timestamp.strftime('%Y%m%d%H%M%S')}_{symbol}",
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            order_type='MARKET',
            quantity=self.config['position_size'],
            price=entry_price,
            status='FILLED',
            filled_price=entry_price,
            filled_timestamp=timestamp,
            commission=self.config['commission_per_side'],
            slippage=slippage
        )
        
        # Adicionar ordem
        self.account.add_order(order)
        
        # Criar trade
        trade = Trade(
            timestamp=timestamp,
            side=side,
            entry_price=entry_price,
            quantity=self.config['position_size'],
            commission=self.config['commission_per_side'],
            slippage=slippage,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Atualizar posição
        quantity_change = self.config['position_size'] if side == 'BUY' else -self.config['position_size']
        self.account.update_position(symbol, quantity_change)
        
        # Armazenar trade aberto
        self.current_trades[symbol] = trade
        
        # Log
        self.logger.info(f"Posição aberta: {symbol} {side} @ {entry_price:.2f}")
        
        # Métricas
        self.system_monitor.record_prediction(prediction)
    
    def _close_position(self, symbol: str, timestamp: datetime, reason: str):
        """Fecha uma posição aberta"""
        if symbol not in self.current_trades:
            return
        
        trade = self.current_trades[symbol]
        
        # Obter preço atual
        current_price = self._get_current_price(symbol)
        
        # Aplicar slippage
        slippage = self.config['slippage_ticks'] * self.config['tick_value']
        if trade.side == 'BUY':
            exit_price = current_price - slippage
            pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            exit_price = current_price + slippage
            pnl = (trade.entry_price - exit_price) * trade.quantity
        
        # Atualizar trade
        trade.exit_price = exit_price
        trade.exit_timestamp = timestamp
        trade.exit_reason = reason
        trade.pnl = pnl - (2 * trade.commission)
        trade.pnl_percent = (trade.pnl / (trade.entry_price * trade.quantity)) * 100
        
        # Criar ordem de saída
        exit_side = 'SELL' if trade.side == 'BUY' else 'BUY'
        order = PaperOrder(
            order_id=f"PO_{timestamp.strftime('%Y%m%d%H%M%S')}_{symbol}_EXIT",
            timestamp=timestamp,
            symbol=symbol,
            side=exit_side,
            order_type='MARKET',
            quantity=trade.quantity,
            price=exit_price,
            status='FILLED',
            filled_price=exit_price,
            filled_timestamp=timestamp,
            commission=self.config['commission_per_side'],
            slippage=slippage
        )
        
        # Adicionar ordem e trade
        self.account.add_order(order)
        self.account.add_trade(trade)
        
        # Atualizar posição
        quantity_change = -trade.quantity if trade.side == 'BUY' else trade.quantity
        self.account.update_position(symbol, quantity_change)
        
        # Remover trade aberto
        del self.current_trades[symbol]
        
        # Log
        self.logger.info(
            f"Posição fechada: {symbol} {reason} @ {exit_price:.2f}, "
            f"PnL: R$ {trade.pnl:.2f} ({trade.pnl_percent:.2f}%)"
        )
    
    def _monitor_positions_loop(self):
        """Loop de monitoramento de posições"""
        self.logger.info("Loop de monitoramento iniciado")
        
        while self.is_running:
            try:
                # Verificar stops para cada posição aberta
                for symbol, trade in list(self.current_trades.items()):
                    current_price = self._get_current_price(symbol)
                    
                    if trade.side == 'BUY':
                        if current_price <= trade.stop_loss:
                            self._close_position(symbol, datetime.now(), 'stop_loss')
                        elif current_price >= trade.take_profit:
                            self._close_position(symbol, datetime.now(), 'take_profit')
                    else:
                        if current_price >= trade.stop_loss:
                            self._close_position(symbol, datetime.now(), 'stop_loss')
                        elif current_price <= trade.take_profit:
                            self._close_position(symbol, datetime.now(), 'take_profit')
                
                # Atualizar PnL aberto
                self._update_open_pnl()
                
                # Aguardar próxima verificação
                time.sleep(1)  # Verificar a cada segundo
                
            except Exception as e:
                self.logger.error(f"Erro no monitoramento: {e}")
    
    def _update_open_pnl(self):
        """Atualiza PnL das posições abertas"""
        total_open_pnl = 0.0
        
        for symbol, trade in self.current_trades.items():
            current_price = self._get_current_price(symbol)
            
            if trade.side == 'BUY':
                open_pnl = (current_price - trade.entry_price) * trade.quantity
            else:
                open_pnl = (trade.entry_price - current_price) * trade.quantity
            
            total_open_pnl += open_pnl
        
        self.account.open_pnl = total_open_pnl
    
    def _get_current_price(self, symbol: str) -> float:
        """Obtém preço atual do símbolo"""
        # Em produção, pegar do realtime_processor
        # Por agora, simular
        base_price = 5900.0
        return base_price + np.random.randn() * 5
    
    def _close_all_positions(self, reason: str):
        """Fecha todas as posições abertas"""
        for symbol in list(self.current_trades.keys()):
            self._close_position(symbol, datetime.now(), reason)
    
    def _generate_final_report(self):
        """Gera relatório final do paper trading"""
        report = {
            'summary': self.account.get_account_summary(),
            'trades': [asdict(t) for t in self.account.trades],
            'orders': [asdict(o) for o in self.account.orders],
            'performance_metrics': self._calculate_performance_metrics(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Salvar relatório
        filename = f"paper_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Relatório salvo em: {filename}")
        
        # Exibir resumo
        summary = report['summary']
        metrics = report['performance_metrics']
        
        print("\n" + "="*60)
        print("RESUMO DO PAPER TRADING")
        print("="*60)
        print(f"Capital Inicial: R$ {summary['initial_capital']:,.2f}")
        print(f"Capital Final: R$ {summary['current_capital']:,.2f}")
        print(f"Retorno: {summary['return_pct']:.2f}%")
        print(f"Total de Trades: {summary['total_trades']}")
        print(f"Taxa de Acerto: {metrics.get('win_rate', 0):.2f}%")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        print("="*60)
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calcula métricas de performance"""
        trades = self.account.trades
        
        if not trades:
            return {}
        
        # Calcular métricas básicas
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl < 0)
        
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        # Calcular Sharpe Ratio simplificado
        returns = [t.pnl_percent for t in trades]
        if len(returns) > 1:
            sharpe = np.sqrt(252) * (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'avg_win': gross_profit / winning_trades if winning_trades > 0 else 0,
            'avg_loss': gross_loss / losing_trades if losing_trades > 0 else 0,
            'max_drawdown': 0  # Simplificado por agora
        }


def main():
    """Função principal para executar paper trading"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Criar paper trader
    paper_trader = PaperTraderV3()
    
    try:
        # Iniciar paper trading
        paper_trader.start()
        
        # Executar por tempo determinado (ex: 1 hora)
        print("Paper Trading iniciado. Pressione Ctrl+C para parar...")
        
        while True:
            time.sleep(10)
            # Exibir status periodicamente
            summary = paper_trader.account.get_account_summary()
            print(f"\rPosições: {len(summary['positions'])}, "
                  f"PnL: R$ {summary['total_pnl']:.2f} "
                  f"({summary['return_pct']:.2f}%)", end='')
            
    except KeyboardInterrupt:
        print("\n\nParando Paper Trading...")
    finally:
        paper_trader.stop()


if __name__ == "__main__":
    main()