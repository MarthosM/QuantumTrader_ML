"""
Backtesting system para HybridStrategy
Testa a estratégia híbrida com dados históricos
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import json

from ..strategies.hybrid_strategy import HybridStrategy
from ..features.ml_features_v3 import MLFeaturesV3 as MLFeatures
from ..technical_indicators import TechnicalIndicators

class HybridBacktest:
    """
    Sistema de backtesting para estratégia híbrida
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Estratégia
        self.strategy = HybridStrategy(config)
        
        # Feature calculators
        self.ml_features = MLFeatures()
        self.tech_indicators = TechnicalIndicators()
        
        # Dados
        self.tick_data = None
        self.book_data = None
        
        # Resultados
        self.trades = []
        self.equity_curve = []
        self.signals = []
        
        # Parâmetros de backtest
        self.initial_capital = config.get('initial_capital', 100000)
        self.commission = config.get('commission', 5.0)  # Por contrato
        self.slippage = config.get('slippage', 0.0001)  # 0.01%
        
        # Estado
        self.current_position = 0
        self.current_capital = self.initial_capital
        self.entry_price = 0
        self.entry_time = None
        
    def load_data(self, start_date: datetime, end_date: datetime,
                  tick_file: Optional[str] = None,
                  book_dir: Optional[str] = None):
        """Carrega dados para backtest"""
        
        self.logger.info("="*80)
        self.logger.info("CARREGANDO DADOS PARA BACKTEST")
        self.logger.info("="*80)
        
        # 1. Carregar dados tick
        if tick_file:
            self.logger.info(f"\nCarregando tick data: {tick_file}")
            self.tick_data = self._load_tick_data(tick_file, start_date, end_date)
        else:
            # Usar dados CSV padrão
            csv_path = Path(r"C:\Users\marth\Downloads\WDO_FUT\WDOFUT_BMF_T.csv")
            self.logger.info(f"\nCarregando tick data padrão: {csv_path}")
            self.tick_data = self._load_tick_data(str(csv_path), start_date, end_date)
        
        # 2. Carregar dados book (se disponível)
        if book_dir:
            self.logger.info(f"\nCarregando book data: {book_dir}")
            self.book_data = self._load_book_data(book_dir, start_date, end_date)
        else:
            self.logger.warning("Book data não especificado - usando apenas tick data")
            self.book_data = None
        
        self.logger.info(f"\n[OK] Dados carregados:")
        self.logger.info(f"     Tick records: {len(self.tick_data):,}")
        if self.book_data is not None:
            self.logger.info(f"     Book records: {len(self.book_data):,}")
        
    def _load_tick_data(self, file_path: str, start_date: datetime, 
                       end_date: datetime) -> pd.DataFrame:
        """Carrega e prepara dados tick"""
        
        # Carregar CSV
        dtypes = {
            '<date>': 'uint32',
            '<time>': 'uint32',
            '<price>': 'float32',
            '<qty>': 'uint16',
            '<vol>': 'float32',
            '<buy_agent>': 'category',
            '<sell_agent>': 'category',
            '<trade_type>': 'category'
        }
        
        # Carregar em chunks para economizar memória
        chunks = []
        chunk_size = 100000
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, dtype=dtypes):
            # Criar timestamp
            chunk['timestamp'] = pd.to_datetime(
                chunk['<date>'].astype(str) + ' ' + chunk['<time>'].astype(str).str.zfill(6),
                format='%Y%m%d %H%M%S'
            )
            
            # Filtrar período
            mask = (chunk['timestamp'] >= start_date) & (chunk['timestamp'] <= end_date)
            filtered = chunk[mask]
            
            if len(filtered) > 0:
                chunks.append(filtered)
            
            # Parar se já temos dados suficientes
            if len(chunks) > 0 and sum(len(c) for c in chunks) > 1000000:
                break
        
        if not chunks:
            raise ValueError(f"Nenhum dado encontrado no período {start_date} - {end_date}")
        
        # Combinar chunks
        data = pd.concat(chunks, ignore_index=True)
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        return data
    
    def _load_book_data(self, book_dir: str, start_date: datetime,
                       end_date: datetime) -> Optional[pd.DataFrame]:
        """Carrega dados de book"""
        
        book_path = Path(book_dir)
        if not book_path.exists():
            return None
        
        # Procurar arquivos de book no período
        all_data = []
        
        for date_dir in book_path.iterdir():
            if date_dir.is_dir():
                # Verificar se está no período
                try:
                    dir_date = datetime.strptime(date_dir.name, "%Y%m%d")
                    if start_date.date() <= dir_date.date() <= end_date.date():
                        # Procurar arquivo parquet
                        parquet_files = list(date_dir.glob("**/*.parquet"))
                        for pf in parquet_files:
                            try:
                                data = pd.read_parquet(pf)
                                if 'timestamp' in data.columns:
                                    # Filtrar período
                                    mask = (data['timestamp'] >= start_date) & \
                                          (data['timestamp'] <= end_date)
                                    filtered = data[mask]
                                    if len(filtered) > 0:
                                        all_data.append(filtered)
                            except Exception as e:
                                self.logger.warning(f"Erro ao ler {pf}: {e}")
                except:
                    continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=True).sort_values('timestamp')
        
        return None
    
    def run_backtest(self, lookback_candles: int = 100,
                    candle_timeframe: str = '5min') -> Dict:
        """Executa backtest da estratégia"""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("EXECUTANDO BACKTEST")
        self.logger.info("="*80)
        
        # Inicializar estratégia
        self.strategy.load_models()
        
        # Converter tick data para candles
        candles = self._create_candles(self.tick_data, candle_timeframe)
        
        self.logger.info(f"\nCandles criados: {len(candles)}")
        self.logger.info(f"Período: {candles.index[0]} até {candles.index[-1]}")
        
        # Reset estado
        self.trades = []
        self.equity_curve = []
        self.signals = []
        self.current_capital = self.initial_capital
        self.current_position = 0
        
        # Loop principal do backtest
        for i in tqdm(range(lookback_candles, len(candles)), desc="Backtesting"):
            # Candles históricos para cálculo de features
            historical_candles = candles.iloc[i-lookback_candles:i+1].copy()
            current_candle = candles.iloc[i]
            current_time = candles.index[i]
            
            # 1. Calcular features tick
            tick_features = self._calculate_tick_features(historical_candles)
            
            # 2. Obter features book (se disponível)
            if self.book_data is not None:
                book_features = self._get_book_features_at_time(current_time)
            else:
                book_features = self._create_dummy_book_features(current_time)
            
            # 3. Obter sinal híbrido
            signal_info = self.strategy.get_hybrid_signal(
                tick_features,
                book_features
            )
            
            # Registrar sinal
            self.signals.append({
                'timestamp': current_time,
                'signal': signal_info['signal'],
                'confidence': signal_info['confidence'],
                'regime': signal_info['regime'],
                'price': current_candle['close']
            })
            
            # 4. Executar trade logic
            self._execute_trade_logic(
                signal_info,
                current_candle,
                current_time
            )
            
            # 5. Atualizar equity curve
            self.equity_curve.append({
                'timestamp': current_time,
                'capital': self.current_capital,
                'position': self.current_position,
                'equity': self._calculate_equity(current_candle['close'])
            })
        
        # Fechar posição aberta se houver
        if self.current_position != 0:
            self._close_position(
                candles.iloc[-1]['close'],
                candles.index[-1],
                "End of backtest"
            )
        
        # Calcular métricas
        results = self._calculate_metrics()
        
        return results
    
    def _create_candles(self, tick_data: pd.DataFrame, 
                       timeframe: str) -> pd.DataFrame:
        """Converte tick data em candles"""
        
        # Resample para timeframe desejado
        tick_data.set_index('timestamp', inplace=True)
        
        candles = tick_data['<price>'].resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        
        # Volume
        candles['volume'] = tick_data['<qty>'].resample(timeframe).sum()
        
        # Remover candles vazios
        candles = candles.dropna()
        
        return candles
    
    def _calculate_tick_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Calcula features para modelo tick"""
        
        # Calcular indicadores técnicos
        for indicator in ['RSI', 'MACD', 'BB', 'ATR', 'EMA']:
            self.tech_indicators.calculate(candles, indicator)
        
        # Calcular features ML
        features = self.ml_features.calculate_all_features(candles)
        
        # Retornar última linha
        return features.tail(1)
    
    def _get_book_features_at_time(self, timestamp: pd.Timestamp) -> pd.DataFrame:
        """Obtém features do book no momento específico"""
        
        # Encontrar book data mais próximo
        if self.book_data is None or len(self.book_data) == 0:
            return self._create_dummy_book_features(timestamp)
        
        # Buscar dados próximos ao timestamp
        time_window = timedelta(seconds=30)
        mask = (self.book_data['timestamp'] >= timestamp - time_window) & \
               (self.book_data['timestamp'] <= timestamp)
        
        relevant_book = self.book_data[mask]
        
        if len(relevant_book) == 0:
            return self._create_dummy_book_features(timestamp)
        
        # Calcular features do book
        # TODO: Implementar cálculo real de features do book
        
        return self._create_dummy_book_features(timestamp)
    
    def _create_dummy_book_features(self, timestamp: pd.Timestamp) -> pd.DataFrame:
        """Cria features book dummy"""
        
        # Features esperadas pelo modelo book
        dummy_features = {
            'return_1': 0.0,
            'log_return_1': 0.0,
            'price_ma_5': 0.0,
            'price_ma_10': 0.0,
            'price_ma_20': 0.0,
            'volatility_10': 0.01,
            'volatility_30': 0.015,
            'position': 10.0,
            'position_inverse': 0.1,
            'is_top_5': 0.0,
            'is_top_10': 0.0,
            'quantity_log': 0.0,
            'volume_ratio': 1.0,
            'is_large_volume': 0.0,
            'ofi': 0.0,
            'is_bid': 0.5,
            'is_ask': 0.5,
            'momentum_5': 0.0,
            'momentum_10': 0.0,
            'momentum_20': 0.0,
            'hour': timestamp.hour,
            'minute': timestamp.minute,
            'time_normalized': timestamp.hour / 24.0,
            'is_morning': 1.0 if timestamp.hour < 12 else 0.0,
            'is_afternoon': 1.0 if timestamp.hour >= 12 else 0.0
        }
        
        return pd.DataFrame([dummy_features])
    
    def _execute_trade_logic(self, signal_info: Dict,
                           current_candle: pd.Series,
                           timestamp: pd.Timestamp):
        """Executa lógica de trading"""
        
        signal = signal_info['signal']
        confidence = signal_info['confidence']
        current_price = current_candle['close']
        
        # Se temos posição aberta
        if self.current_position != 0:
            # Verificar stops
            if self.current_position > 0:  # Long
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                
                if pnl_pct <= -self.strategy.stop_loss:
                    self._close_position(current_price, timestamp, "Stop Loss")
                elif pnl_pct >= self.strategy.take_profit:
                    self._close_position(current_price, timestamp, "Take Profit")
                elif signal == -1 and confidence > 0.65:  # Sinal de reversão
                    self._close_position(current_price, timestamp, "Signal Reversal")
                    
            else:  # Short
                pnl_pct = (self.entry_price - current_price) / self.entry_price
                
                if pnl_pct <= -self.strategy.stop_loss:
                    self._close_position(current_price, timestamp, "Stop Loss")
                elif pnl_pct >= self.strategy.take_profit:
                    self._close_position(current_price, timestamp, "Take Profit")
                elif signal == 1 and confidence > 0.65:  # Sinal de reversão
                    self._close_position(current_price, timestamp, "Signal Reversal")
        
        # Se não temos posição e temos sinal
        elif signal != 0 and confidence > 0.6:
            self._open_position(signal, current_price, timestamp, signal_info)
    
    def _open_position(self, signal: int, price: float,
                      timestamp: pd.Timestamp, signal_info: Dict):
        """Abre nova posição"""
        
        # Calcular tamanho da posição
        position_size = self.strategy.calculate_position_size(
            signal_info,
            price,
            self.current_capital
        )
        
        if position_size == 0:
            return
        
        # Aplicar slippage
        if signal == 1:  # Buy
            entry_price = price * (1 + self.slippage)
        else:  # Sell
            entry_price = price * (1 - self.slippage)
        
        # Custo da operação
        trade_cost = position_size * self.commission
        
        # Atualizar estado
        self.current_position = signal * position_size
        self.entry_price = entry_price
        self.entry_time = timestamp
        self.current_capital -= trade_cost
        
        # Registrar trade
        trade = {
            'timestamp': timestamp,
            'action': 'BUY' if signal == 1 else 'SELL',
            'price': entry_price,
            'quantity': position_size,
            'commission': trade_cost,
            'signal_info': signal_info
        }
        
        self.trades.append(trade)
        
        self.logger.debug(f"Posição aberta: {trade['action']} {position_size} @ {entry_price:.2f}")
    
    def _close_position(self, price: float, timestamp: pd.Timestamp,
                       reason: str):
        """Fecha posição atual"""
        
        if self.current_position == 0:
            return
        
        # Aplicar slippage
        if self.current_position > 0:  # Fechando long
            exit_price = price * (1 - self.slippage)
            action = 'SELL'
        else:  # Fechando short
            exit_price = price * (1 + self.slippage)
            action = 'BUY'
        
        # Calcular P&L
        quantity = abs(self.current_position)
        if self.current_position > 0:
            gross_pnl = (exit_price - self.entry_price) * quantity
        else:
            gross_pnl = (self.entry_price - exit_price) * quantity
        
        # Custos
        trade_cost = quantity * self.commission
        net_pnl = gross_pnl - trade_cost
        
        # Atualizar capital
        self.current_capital += net_pnl
        
        # Registrar trade
        trade = {
            'timestamp': timestamp,
            'action': action,
            'price': exit_price,
            'quantity': quantity,
            'commission': trade_cost,
            'pnl': net_pnl,
            'pnl_pct': net_pnl / (self.entry_price * quantity) * 100,
            'reason': reason,
            'holding_time': timestamp - self.entry_time if self.entry_time else None
        }
        
        self.trades.append(trade)
        
        # Reset posição
        self.current_position = 0
        self.entry_price = 0
        self.entry_time = None
        
        self.logger.debug(f"Posição fechada: {action} @ {exit_price:.2f} - "
                         f"P&L: ${net_pnl:.2f} ({trade['pnl_pct']:.2f}%)")
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calcula equity total (capital + posição aberta)"""
        
        if self.current_position == 0:
            return self.current_capital
        
        # Calcular P&L não realizado
        quantity = abs(self.current_position)
        if self.current_position > 0:
            unrealized_pnl = (current_price - self.entry_price) * quantity
        else:
            unrealized_pnl = (self.entry_price - current_price) * quantity
        
        return self.current_capital + unrealized_pnl
    
    def _calculate_metrics(self) -> Dict:
        """Calcula métricas de performance"""
        
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0
            }
        
        # Converter para DataFrame
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Filtrar apenas trades fechados
        closed_trades = trades_df[trades_df['pnl'].notna()]
        
        if closed_trades.empty:
            return {
                'total_trades': len(trades_df),
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0
            }
        
        # Métricas básicas
        total_trades = len(closed_trades)
        winning_trades = closed_trades[closed_trades['pnl'] > 0]
        losing_trades = closed_trades[closed_trades['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Profit factor
        total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        
        # Sharpe ratio
        returns = equity_df['equity'].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum drawdown
        equity = equity_df['equity'].values
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Total return
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        
        # Estatísticas adicionais
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Duração média dos trades
        if 'holding_time' in closed_trades.columns:
            avg_holding = closed_trades['holding_time'].mean()
        else:
            avg_holding = None
        
        results = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'total_pnl': closed_trades['pnl'].sum(),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_pnl': closed_trades['pnl'].mean(),
            'avg_holding_time': avg_holding,
            'final_capital': self.current_capital,
            'trades': trades_df.to_dict('records'),
            'equity_curve': equity_df.to_dict('records')
        }
        
        self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict):
        """Imprime resumo dos resultados"""
        
        print("\n" + "="*80)
        print("RESULTADOS DO BACKTEST")
        print("="*80)
        
        print(f"\nCapital inicial: ${self.initial_capital:,.2f}")
        print(f"Capital final: ${results['final_capital']:,.2f}")
        print(f"Retorno total: {results['total_return']:.2%}")
        
        print(f"\nTotal de trades: {results['total_trades']}")
        print(f"Taxa de acerto: {results['win_rate']:.2%}")
        print(f"Profit factor: {results['profit_factor']:.2f}")
        print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max drawdown: {results['max_drawdown']:.2%}")
        
        print(f"\nP&L total: ${results['total_pnl']:,.2f}")
        print(f"P&L médio: ${results['avg_pnl']:,.2f}")
        print(f"Ganho médio: ${results['avg_win']:,.2f}")
        print(f"Perda média: ${results['avg_loss']:,.2f}")
        
        if results['avg_holding_time']:
            print(f"\nTempo médio de posição: {results['avg_holding_time']}")
    
    def save_results(self, output_dir: str = "backtest_results"):
        """Salva resultados do backtest"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salvar trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(output_path / f"trades_{timestamp}.csv", index=False)
        
        # Salvar equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.to_csv(output_path / f"equity_{timestamp}.csv", index=False)
        
        # Salvar sinais
        if self.signals:
            signals_df = pd.DataFrame(self.signals)
            signals_df.to_csv(output_path / f"signals_{timestamp}.csv", index=False)
        
        self.logger.info(f"\nResultados salvos em: {output_path}")