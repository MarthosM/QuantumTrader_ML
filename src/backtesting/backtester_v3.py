"""
Backtester V3 - Sistema de Backtesting com Dados Reais
======================================================

Este módulo implementa um sistema completo de backtesting que:
- Usa dados históricos reais
- Simula execução de trades com slippage e custos
- Calcula métricas de performance detalhadas
- Gera relatórios e visualizações
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json

# Adicionar src ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.trading_data_structure_v3 import TradingDataStructureV3
from features.ml_features_v3 import MLFeaturesV3
from ml.prediction_engine_v3 import PredictionEngineV3


@dataclass
class Trade:
    """Representa um trade executado"""
    timestamp: datetime
    side: str  # 'BUY' ou 'SELL'
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    quantity: int = 1
    pnl: float = 0.0
    pnl_percent: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_reason: Optional[str] = None  # 'stop_loss', 'take_profit', 'signal', 'end_of_day'


class BacktesterV3:
    """Sistema de backtesting com dados reais"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o backtester
        
        Args:
            config: Configurações do backtest
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Componentes
        self.data_structure = TradingDataStructureV3()
        self.ml_features = MLFeaturesV3()
        self.prediction_engine = PredictionEngineV3()
        
        # Estado do backtest
        self.trades: List[Trade] = []
        self.open_position: Optional[Trade] = None
        self.equity_curve: List[float] = []
        self.current_equity: float = self.config['initial_capital']
        self.peak_equity: float = self.config['initial_capital']
        
        # Métricas
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'total_commission': 0.0,
            'total_slippage': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_duration': 0,
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_trade': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'avg_trade_duration': timedelta(0),
            'total_time_in_market': timedelta(0)
        }
        
    def _get_default_config(self) -> Dict:
        """Retorna configuração padrão"""
        return {
            'initial_capital': 100000.0,
            'position_size': 1,  # Número de contratos
            'commission_per_side': 5.0,  # R$ por contrato
            'slippage_ticks': 1,  # Ticks de slippage
            'tick_value': 0.5,  # Valor do tick em R$
            'stop_loss_ticks': 20,  # Stop loss em ticks
            'take_profit_ticks': 40,  # Take profit em ticks
            'max_daily_trades': 10,
            'trade_hours': {
                'start': 9,
                'end': 17
            },
            'confidence_threshold': 0.6,
            'probability_threshold': 0.55
        }
    
    def run_backtest(self, 
                     start_date: datetime,
                     end_date: datetime,
                     data_path: Optional[str] = None) -> Dict:
        """
        Executa o backtest completo
        
        Args:
            start_date: Data inicial
            end_date: Data final
            data_path: Caminho para dados históricos
            
        Returns:
            Resultados do backtest
        """
        self.logger.info(f"Iniciando backtest de {start_date} a {end_date}")
        
        # 1. Carregar dados históricos
        if not self._load_historical_data(start_date, end_date, data_path):
            self.logger.error("Falha ao carregar dados históricos")
            return {}
        
        # 2. Preparar features
        # Pegar TODOS os candles, não apenas 100
        candles = self.data_structure.candles  # Acesso direto ao DataFrame completo
        if candles.empty:
            self.logger.error("Nenhum candle disponível")
            return {}
            
        self.logger.info(f"Processando {len(candles)} candles")
        
        microstructure = self._create_microstructure(candles)
        features = self.ml_features.calculate_all(candles, microstructure)
        
        if features.empty:
            self.logger.error("Nenhuma feature calculada")
            return {}
        
        # 3. Executar simulação
        self._run_simulation(candles, features)
        
        # 4. Calcular métricas finais
        self._calculate_final_metrics()
        
        # 5. Gerar relatório
        report = self._generate_report()
        
        self.logger.info("Backtest concluído")
        return report
    
    def _load_historical_data(self, 
                            start_date: datetime,
                            end_date: datetime,
                            data_path: Optional[str]) -> bool:
        """Carrega dados históricos"""
        try:
            # Usar CSV existente se disponível
            if data_path and os.path.exists(data_path):
                df = pd.read_csv(data_path)
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Filtrar período
                mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
                df = df[mask]
                
                if df.empty:
                    self.logger.error("Nenhum dado no período especificado")
                    return False
                
                # Converter para formato esperado
                trades_df = pd.DataFrame()
                trades_df['datetime'] = df['Date']
                trades_df['price'] = df['preco']
                trades_df['volume'] = df['quantidade']
                trades_df['buy_volume'] = df['buy_volume']
                trades_df['sell_volume'] = df['sell_volume']
                trades_df.set_index('datetime', inplace=True)
                
                candles_df = pd.DataFrame()
                candles_df['datetime'] = df['Date']
                candles_df['open'] = df['open']
                candles_df['high'] = df['high']
                candles_df['low'] = df['low']
                candles_df['close'] = df['close']
                candles_df['volume'] = df['volume']
                candles_df['buy_volume'] = df['buy_volume']
                candles_df['sell_volume'] = df['sell_volume']
                candles_df.set_index('datetime', inplace=True)
                
                # Adicionar aos dados
                historical_data = {
                    'trades': trades_df,
                    'candles': candles_df,
                    'book_updates': pd.DataFrame()
                }
                
                self.data_structure.add_historical_data(historical_data)
                
                self.logger.info(f"Dados carregados: {len(candles_df)} candles")
                return True
                
            else:
                self.logger.error("Caminho de dados não especificado")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro carregando dados: {e}")
            return False
    
    def _create_microstructure(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Cria microestrutura a partir dos candles"""
        microstructure = pd.DataFrame(index=candles.index)
        microstructure['buy_volume'] = candles['buy_volume']
        microstructure['sell_volume'] = candles['sell_volume']
        microstructure['volume_imbalance'] = (
            (candles['buy_volume'] - candles['sell_volume']) / 
            (candles['buy_volume'] + candles['sell_volume'] + 1)
        )
        microstructure['trade_imbalance'] = microstructure['volume_imbalance']
        microstructure['bid_ask_spread'] = 0.001  # Default
        
        return microstructure
    
    def _run_simulation(self, candles: pd.DataFrame, features: pd.DataFrame):
        """Executa a simulação de trading"""
        self.logger.info(f"Iniciando simulação de trading com {len(candles)} candles e {len(features)} features...")
        
        signals_generated = 0
        trades_executed = 0
        daily_trades = 0
        current_day = None
        
        # Iterar sobre cada candle
        for i in range(len(candles)):
            timestamp = candles.index[i]
            candle = candles.iloc[i]
            
            # Resetar contador diário
            if current_day != timestamp.date():
                current_day = timestamp.date()
                daily_trades = 0
            
            # Verificar horário de trading
            if not self._is_trading_hours(timestamp):
                # Fechar posição no final do dia
                if self.open_position:
                    self._close_position(candle, timestamp, 'end_of_day')
                continue
            
            # Verificar stops se houver posição aberta
            if self.open_position:
                self._check_stops(candle, timestamp)
            
            # Pular se já atingiu limite diário
            if daily_trades >= self.config['max_daily_trades']:
                continue
            
            # Gerar predição
            if i < len(features):
                feature_row = features.iloc[i:i+1]
                prediction = self._generate_prediction(feature_row)
                
                if prediction:
                    signals_generated += 1
                    # Executar sinal
                    if self._should_trade(prediction):
                        trades_executed += 1
                        if self.open_position:
                            # Fechar posição se sinal contrário
                            if self._is_opposite_signal(prediction):
                                self._close_position(candle, timestamp, 'signal')
                                self._open_position(prediction, candle, timestamp)
                                daily_trades += 1
                        else:
                            # Abrir nova posição
                            self._open_position(prediction, candle, timestamp)
                            daily_trades += 1
            
            # Atualizar equity
            self._update_equity(candle)
        
        # Fechar posição final se houver
        if self.open_position:
            last_candle = candles.iloc[-1]
            self._close_position(last_candle, candles.index[-1], 'end_of_backtest')
        
        self.logger.info(f"Simulação concluída: {signals_generated} sinais gerados, {trades_executed} trades executados")
    
    def _is_trading_hours(self, timestamp: datetime) -> bool:
        """Verifica se está no horário de trading"""
        hour = timestamp.hour
        return (self.config['trade_hours']['start'] <= hour < 
                self.config['trade_hours']['end'])
    
    def _generate_prediction(self, features: pd.DataFrame) -> Optional[Dict]:
        """Gera predição usando o prediction engine"""
        try:
            # Simular predição se não houver modelos
            # Em produção, usar self.prediction_engine.predict(features)
            
            # Para teste, gerar sinal aleatório baseado em momentum
            # As features V3 têm prefixo "v3_"
            if 'v3_momentum_pct_1' in features.columns:
                momentum = features['v3_momentum_pct_1'].iloc[0]
                
                # Lógica simples de exemplo - thresholds baixos para gerar sinais
                if momentum > 0.01:  # 1% de momentum positivo
                    return {
                        'direction': 1,
                        'confidence': 0.65,
                        'probability': 0.60,
                        'regime': 'trend_up'
                    }
                elif momentum < -0.01:  # 1% de momentum negativo
                    return {
                        'direction': -1,
                        'confidence': 0.65,
                        'probability': 0.60,
                        'regime': 'trend_down'
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro gerando predição: {e}")
            return None
    
    def _should_trade(self, prediction: Dict) -> bool:
        """Verifica se deve executar o trade"""
        return (prediction['confidence'] >= self.config['confidence_threshold'] and
                prediction['probability'] >= self.config['probability_threshold'])
    
    def _is_opposite_signal(self, prediction: Dict) -> bool:
        """Verifica se o sinal é oposto à posição atual"""
        if not self.open_position:
            return False
        
        current_side = 1 if self.open_position.side == 'BUY' else -1
        return prediction['direction'] != current_side
    
    def _open_position(self, prediction: Dict, candle: pd.Series, timestamp: datetime):
        """Abre uma nova posição"""
        side = 'BUY' if prediction['direction'] > 0 else 'SELL'
        
        # Calcular preço de entrada com slippage
        slippage = self.config['slippage_ticks'] * self.config['tick_value']
        if side == 'BUY':
            entry_price = candle['close'] + slippage
            stop_loss = entry_price - (self.config['stop_loss_ticks'] * self.config['tick_value'])
            take_profit = entry_price + (self.config['take_profit_ticks'] * self.config['tick_value'])
        else:
            entry_price = candle['close'] - slippage
            stop_loss = entry_price + (self.config['stop_loss_ticks'] * self.config['tick_value'])
            take_profit = entry_price - (self.config['take_profit_ticks'] * self.config['tick_value'])
        
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
        
        self.open_position = trade
        self.current_equity -= trade.commission
        
        self.logger.debug(f"Posição aberta: {side} @ {entry_price:.2f}")
    
    def _close_position(self, candle: pd.Series, timestamp: datetime, reason: str):
        """Fecha a posição atual"""
        if not self.open_position:
            return
        
        trade = self.open_position
        
        # Calcular preço de saída com slippage
        slippage = self.config['slippage_ticks'] * self.config['tick_value']
        if trade.side == 'BUY':
            exit_price = candle['close'] - slippage
            pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            exit_price = candle['close'] + slippage
            pnl = (trade.entry_price - exit_price) * trade.quantity
        
        # Atualizar trade
        trade.exit_price = exit_price
        trade.exit_timestamp = timestamp
        trade.exit_reason = reason
        trade.pnl = pnl - (2 * trade.commission)  # Comissão de entrada e saída
        trade.pnl_percent = (trade.pnl / (trade.entry_price * trade.quantity)) * 100
        
        # Adicionar à lista de trades
        self.trades.append(trade)
        self.open_position = None
        
        # Atualizar equity
        self.current_equity += trade.pnl
        
        # Atualizar métricas
        self.metrics['total_trades'] += 1
        if trade.pnl > 0:
            self.metrics['winning_trades'] += 1
            self.metrics['gross_profit'] += trade.pnl
        else:
            self.metrics['losing_trades'] += 1
            self.metrics['gross_loss'] += abs(trade.pnl)
        
        self.metrics['total_commission'] += (2 * trade.commission)
        self.metrics['total_slippage'] += (2 * slippage)
        
        self.logger.debug(f"Posição fechada: {reason} @ {exit_price:.2f}, PnL: {trade.pnl:.2f}")
    
    def _check_stops(self, candle: pd.Series, timestamp: datetime):
        """Verifica stop loss e take profit"""
        if not self.open_position:
            return
        
        trade = self.open_position
        
        if trade.side == 'BUY':
            # Check stop loss
            if candle['low'] <= trade.stop_loss:
                self._close_position(candle, timestamp, 'stop_loss')
            # Check take profit
            elif candle['high'] >= trade.take_profit:
                self._close_position(candle, timestamp, 'take_profit')
        else:
            # Check stop loss
            if candle['high'] >= trade.stop_loss:
                self._close_position(candle, timestamp, 'stop_loss')
            # Check take profit
            elif candle['low'] <= trade.take_profit:
                self._close_position(candle, timestamp, 'take_profit')
    
    def _update_equity(self, candle: pd.Series):
        """Atualiza a curva de equity"""
        # Calcular equity atual incluindo posição aberta
        equity = self.current_equity
        
        if self.open_position:
            trade = self.open_position
            if trade.side == 'BUY':
                unrealized_pnl = (candle['close'] - trade.entry_price) * trade.quantity
            else:
                unrealized_pnl = (trade.entry_price - candle['close']) * trade.quantity
            equity += unrealized_pnl
        
        self.equity_curve.append(equity)
        
        # Atualizar drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        else:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], drawdown)
    
    def _calculate_final_metrics(self):
        """Calcula métricas finais do backtest"""
        if not self.trades:
            return
        
        # Taxa de acerto
        self.metrics['win_rate'] = (
            self.metrics['winning_trades'] / self.metrics['total_trades'] 
            if self.metrics['total_trades'] > 0 else 0
        )
        
        # Profit factor
        self.metrics['profit_factor'] = (
            self.metrics['gross_profit'] / abs(self.metrics['gross_loss'])
            if self.metrics['gross_loss'] != 0 else float('inf')
        )
        
        # Médias
        pnls = [t.pnl for t in self.trades]
        self.metrics['avg_trade'] = np.mean(pnls) if pnls else 0
        
        winning_pnls = [t.pnl for t in self.trades if t.pnl > 0]
        self.metrics['avg_win'] = np.mean(winning_pnls) if winning_pnls else 0
        
        losing_pnls = [t.pnl for t in self.trades if t.pnl < 0]
        self.metrics['avg_loss'] = np.mean(losing_pnls) if losing_pnls else 0
        
        # Melhor e pior trade
        self.metrics['best_trade'] = max(pnls) if pnls else 0
        self.metrics['worst_trade'] = min(pnls) if pnls else 0
        
        # Duração média dos trades
        durations = []
        for t in self.trades:
            if t.exit_timestamp:
                durations.append(t.exit_timestamp - t.timestamp)
        
        if durations:
            self.metrics['avg_trade_duration'] = sum(durations, timedelta()) / len(durations)
            self.metrics['total_time_in_market'] = sum(durations, timedelta())
        
        # Sharpe Ratio (anualizado)
        if len(self.equity_curve) > 1:
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            if len(returns) > 0:
                sharpe = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() > 0 else 0
                self.metrics['sharpe_ratio'] = sharpe
    
    def _generate_report(self) -> Dict:
        """Gera relatório completo do backtest"""
        return {
            'summary': {
                'initial_capital': self.config['initial_capital'],
                'final_capital': self.current_equity,
                'total_return': ((self.current_equity - self.config['initial_capital']) / 
                               self.config['initial_capital']) * 100,
                'total_trades': self.metrics['total_trades'],
                'win_rate': self.metrics['win_rate'] * 100,
                'profit_factor': self.metrics['profit_factor'],
                'sharpe_ratio': self.metrics['sharpe_ratio'],
                'max_drawdown': self.metrics['max_drawdown'] * 100
            },
            'metrics': self.metrics,
            'trades': [self._trade_to_dict(t) for t in self.trades],
            'equity_curve': self.equity_curve,
            'config': self.config
        }
    
    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Converte trade para dicionário"""
        return {
            'timestamp': trade.timestamp.isoformat(),
            'side': trade.side,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'exit_timestamp': trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
            'quantity': trade.quantity,
            'pnl': trade.pnl,
            'pnl_percent': trade.pnl_percent,
            'exit_reason': trade.exit_reason
        }
    
    def save_report(self, report: Dict, filename: str):
        """Salva relatório em arquivo JSON"""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        self.logger.info(f"Relatório salvo em: {filename}")


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Criar backtester
    backtester = BacktesterV3()
    
    # Executar backtest
    results = backtester.run_backtest(
        start_date=datetime(2025, 2, 3),
        end_date=datetime(2025, 2, 10),
        data_path='wdo_data_20_06_2025.csv'
    )
    
    # Salvar resultados
    if results:
        backtester.save_report(results, 'backtest_results.json')
        print(f"\nResultados do Backtest:")
        print(f"Capital inicial: R$ {results['summary']['initial_capital']:,.2f}")
        print(f"Capital final: R$ {results['summary']['final_capital']:,.2f}")
        print(f"Retorno total: {results['summary']['total_return']:.2f}%")
        print(f"Total de trades: {results['summary']['total_trades']}")
        print(f"Taxa de acerto: {results['summary']['win_rate']:.2f}%")
        print(f"Profit Factor: {results['summary']['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {results['summary']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['summary']['max_drawdown']:.2f}%")