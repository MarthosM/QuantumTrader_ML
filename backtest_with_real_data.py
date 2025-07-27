"""
Backtest com Dados Reais do WDO
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trading_system import TradingSystem
from data_structure import TradingDataStructure
from model_manager import ModelManager
from feature_engine import FeatureEngine
from prediction_engine import PredictionEngine
from ml_coordinator import MLCoordinator
from signal_generator import SignalGenerator
from risk_manager import RiskManager
from features.feature_debugger import FeatureDebugger
from adaptive_threshold_manager import AdaptiveThresholdManager
from training.regime_analyzer import RegimeAnalyzer


class RealDataBacktester:
    """Backtester usando dados reais do WDO"""
    
    def __init__(self, data_file: str, initial_capital: float = 100000):
        self.data_file = data_file
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        
        # Componentes de análise
        self.feature_debugger = FeatureDebugger(self.logger)
        self.threshold_manager = AdaptiveThresholdManager()
        
        # Estatísticas
        self.trades = []
        self.signals_analyzed = 0
        self.signals_generated = 0
        self.trades_executed = 0
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Carrega e prepara dados do CSV"""
        self.logger.info(f"Carregando dados de {self.data_file}")
        
        # Carregar dados
        df = pd.read_csv(self.data_file)
        
        # Converter Date para datetime e definir como índice
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Ordenar por data
        df.sort_index(inplace=True)
        
        self.logger.info(f"Dados carregados: {len(df)} candles")
        self.logger.info(f"Período: {df.index[0]} até {df.index[-1]}")
        self.logger.info(f"Contrato: {df['contract'].iloc[0]}")
        
        return df
    
    def run_backtest(self):
        """Executa backtest com dados reais"""
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.logger.info("="*80)
        self.logger.info("INICIANDO BACKTEST COM DADOS REAIS DO WDO")
        self.logger.info("="*80)
        
        # Carregar dados
        data = self.load_and_prepare_data()
        
        # Configurar sistema
        config = self._create_optimized_config()
        
        # Criar componentes
        self.logger.info("\nInicializando componentes do sistema...")
        
        # Data structure
        trading_data = TradingDataStructure()
        
        # Model manager
        model_manager = ModelManager({'model_path': 'models/'})
        model_manager.load_models()
        
        if not model_manager.models:
            self.logger.warning("AVISO: Nenhum modelo carregado - sistema em modo análise")
        
        # Feature engine
        feature_engine = FeatureEngine(model_manager)
        
        # Prediction engine
        prediction_engine = PredictionEngine(model_manager)
        
        # Regime analyzer
        regime_analyzer = RegimeAnalyzer(self.logger)
        
        # ML Coordinator
        ml_coordinator = MLCoordinator(
            model_manager, 
            feature_engine, 
            prediction_engine,
            regime_analyzer
        )
        
        # Signal generator com thresholds otimizados
        signal_generator = SignalGenerator(config['signal_generator'])
        
        # Risk manager
        risk_manager = RiskManager(config['risk_manager'])
        
        # Processar dados em janelas deslizantes
        window_size = 100  # Últimos 100 candles
        step_size = 1      # Avançar 1 candle por vez
        
        self.logger.info(f"\nProcessando {len(data)} candles...")
        self.logger.info(f"Window size: {window_size}, Step: {step_size}")
        
        # Variáveis de controle
        position = None
        equity_curve = [self.initial_capital]
        current_capital = self.initial_capital
        
        # Estatísticas detalhadas
        stats = {
            'signals_by_regime': {'trend_up': 0, 'trend_down': 0, 'range': 0, 'undefined': 0},
            'trades_by_regime': {'trend_up': 0, 'trend_down': 0, 'range': 0, 'undefined': 0},
            'rejections': {},
            'feature_quality_scores': []
        }
        
        # Processar cada janela
        for i in range(window_size, len(data), step_size):
            # Janela atual
            window_data = data.iloc[i-window_size:i]
            current_candle = data.iloc[i-1]
            
            # Atualizar data structure
            trading_data.candles = window_data
            
            # Calcular indicadores técnicos básicos
            self._calculate_basic_indicators(trading_data)
            
            # ML Prediction
            self.logger.debug(f"\n[{i}/{len(data)}] Processando {window_data.index[-1]}")
            
            try:
                # Processar predição
                ml_result = ml_coordinator.process_prediction_request(trading_data)
                
                if ml_result:
                    self.signals_analyzed += 1
                    
                    # Registrar regime
                    regime = ml_result.get('regime', 'undefined')
                    stats['signals_by_regime'][regime] = stats['signals_by_regime'].get(regime, 0) + 1
                    
                    # Debug a cada 50 sinais
                    if self.signals_analyzed % 50 == 0:
                        self._log_progress(stats, current_capital)
                    
                    # Verificar se pode operar
                    if ml_result.get('can_trade', False):
                        # Gerar sinal
                        signal = signal_generator.generate_signal(ml_result, trading_data)
                        
                        if signal and signal.get('action') in ['buy', 'sell']:
                            self.signals_generated += 1
                            stats['trades_by_regime'][regime] = stats['trades_by_regime'].get(regime, 0) + 1
                            
                            # Simular execução do trade
                            if not position:  # Só abrir se não tiver posição
                                trade = self._execute_simulated_trade(
                                    signal, 
                                    window_data, 
                                    data[i:min(i+20, len(data))],  # Próximos 20 candles
                                    current_capital
                                )
                                
                                if trade:
                                    self.trades.append(trade)
                                    self.trades_executed += 1
                                    current_capital += trade['profit']
                                    
                                    # Atualizar threshold manager
                                    self.threshold_manager.update_trade_result({
                                        'regime': regime,
                                        'win': trade['profit'] > 0,
                                        'return': trade['profit'] / self.initial_capital,
                                        'confidence': ml_result.get('confidence', 0),
                                        'direction': ml_result.get('direction', 0)
                                    })
                                    
                                    # Log do trade
                                    self._log_trade(trade)
                    
                    else:
                        # Registrar rejeição
                        reason = ml_result.get('reason', 'unknown')
                        stats['rejections'][reason] = stats['rejections'].get(reason, 0) + 1
                
                # Analisar qualidade das features periodicamente
                if i % 100 == 0 and not trading_data.features.empty:
                    analysis = self.feature_debugger.analyze_features(trading_data.features)
                    stats['feature_quality_scores'].append(analysis['quality_score'])
                
                # Adaptar thresholds
                if len(self.trades) > 0 and len(self.trades) % 10 == 0:
                    self.threshold_manager.adapt_thresholds()
                    
            except Exception as e:
                self.logger.error(f"Erro processando candle {i}: {e}")
                continue
            
            # Atualizar equity curve
            equity_curve.append(current_capital)
        
        # Gerar relatório final
        self._generate_final_report(stats, equity_curve, current_capital)
        
    def _calculate_basic_indicators(self, data: TradingDataStructure):
        """Calcula indicadores técnicos básicos"""
        df = data.candles
        
        # EMAs
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ADX (simplificado)
        df['adx'] = 25.0  # Placeholder - implementar cálculo completo se necessário
        
        # Volume metrics
        df['volume_ratio'] = df['buy_volume'] / (df['sell_volume'] + 1)
        df['volume_ma'] = df['volume'].rolling(20).mean()
        
        # Atualizar indicators
        data.indicators = df[['ema_9', 'ema_20', 'ema_50', 'atr', 'rsi', 'adx', 
                             'volume_ratio', 'volume_ma']].copy()
        
    def _execute_simulated_trade(self, signal: Dict, current_data: pd.DataFrame, 
                                future_data: pd.DataFrame, capital: float) -> Dict:
        """Simula execução de trade com dados futuros"""
        
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        action = signal['action']
        
        # Simular evolução do preço
        for i, (timestamp, candle) in enumerate(future_data.iterrows()):
            # Verificar stop loss
            if action == 'buy':
                if candle['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                    break
                elif candle['high'] >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    break
            else:  # sell
                if candle['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                    break
                elif candle['low'] <= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    break
        else:
            # Se não atingiu nenhum alvo, sair no último candle
            exit_price = future_data.iloc[-1]['close']
            exit_reason = 'time_exit'
        
        # Calcular lucro/prejuízo
        if action == 'buy':
            profit_points = exit_price - entry_price
        else:
            profit_points = entry_price - exit_price
            
        profit = profit_points * signal.get('position_size', 1) * 0.5  # R$0.50 por ponto
        
        trade = {
            'timestamp': current_data.index[-1],
            'action': action,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'profit': profit,
            'profit_points': profit_points,
            'regime': signal['metadata'].get('regime', 'unknown'),
            'confidence': signal.get('confidence', 0),
            'duration_candles': i + 1 if i < len(future_data) - 1 else len(future_data)
        }
        
        return trade
    
    def _log_trade(self, trade: Dict):
        """Log detalhado do trade"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"TRADE #{self.trades_executed}")
        self.logger.info(f"Timestamp: {trade['timestamp']}")
        self.logger.info(f"Ação: {trade['action'].upper()}")
        self.logger.info(f"Entrada: {trade['entry_price']:.1f}")
        self.logger.info(f"Saída: {trade['exit_price']:.1f} ({trade['exit_reason']})")
        self.logger.info(f"Pontos: {trade['profit_points']:.1f}")
        self.logger.info(f"Lucro: R$ {trade['profit']:.2f}")
        self.logger.info(f"Regime: {trade['regime']}")
        self.logger.info(f"Duração: {trade['duration_candles']} candles")
        self.logger.info(f"{'='*60}")
    
    def _log_progress(self, stats: Dict, capital: float):
        """Log de progresso durante backtest"""
        self.logger.info(f"\n[PROGRESSO] Sinais analisados: {self.signals_analyzed}")
        self.logger.info(f"  Sinais gerados: {self.signals_generated}")
        self.logger.info(f"  Trades executados: {self.trades_executed}")
        self.logger.info(f"  Capital atual: R$ {capital:,.2f}")
        
        # Win rate se houver trades
        if self.trades:
            wins = sum(1 for t in self.trades if t['profit'] > 0)
            win_rate = wins / len(self.trades) * 100
            self.logger.info(f"  Win Rate: {win_rate:.1f}%")
    
    def _generate_final_report(self, stats: Dict, equity_curve: list, final_capital: float):
        """Gera relatório final detalhado"""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("RELATÓRIO FINAL DO BACKTEST")
        self.logger.info("="*80)
        
        # Resultados financeiros
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        
        self.logger.info(f"\nRESULTADOS FINANCEIROS:")
        self.logger.info(f"Capital Inicial: R$ {self.initial_capital:,.2f}")
        self.logger.info(f"Capital Final: R$ {final_capital:,.2f}")
        self.logger.info(f"Lucro/Prejuízo: R$ {final_capital - self.initial_capital:,.2f}")
        self.logger.info(f"Retorno: {total_return:.2f}%")
        
        # Estatísticas de trading
        self.logger.info(f"\nESTATÍSTICAS DE TRADING:")
        self.logger.info(f"Sinais analisados: {self.signals_analyzed}")
        self.logger.info(f"Sinais gerados: {self.signals_generated}")
        self.logger.info(f"Taxa de geração: {self.signals_generated/self.signals_analyzed*100:.1f}%" if self.signals_analyzed > 0 else "N/A")
        self.logger.info(f"Trades executados: {self.trades_executed}")
        
        if self.trades:
            wins = [t for t in self.trades if t['profit'] > 0]
            losses = [t for t in self.trades if t['profit'] <= 0]
            
            self.logger.info(f"Win Rate: {len(wins)/len(self.trades)*100:.1f}%")
            self.logger.info(f"Trades vencedores: {len(wins)}")
            self.logger.info(f"Trades perdedores: {len(losses)}")
            
            if wins:
                avg_win = sum(t['profit'] for t in wins) / len(wins)
                self.logger.info(f"Ganho médio: R$ {avg_win:.2f}")
            
            if losses:
                avg_loss = sum(t['profit'] for t in losses) / len(losses)
                self.logger.info(f"Perda média: R$ {avg_loss:.2f}")
            
            # Profit factor
            if losses:
                total_wins = sum(t['profit'] for t in wins) if wins else 0
                total_losses = abs(sum(t['profit'] for t in losses))
                profit_factor = total_wins / total_losses if total_losses > 0 else 0
                self.logger.info(f"Profit Factor: {profit_factor:.2f}")
        
        # Distribuição por regime
        self.logger.info(f"\nDISTRIBUIÇÃO POR REGIME:")
        self.logger.info("Sinais analisados:")
        for regime, count in stats['signals_by_regime'].items():
            pct = count / self.signals_analyzed * 100 if self.signals_analyzed > 0 else 0
            self.logger.info(f"  {regime}: {count} ({pct:.1f}%)")
        
        self.logger.info("\nTrades executados:")
        for regime, count in stats['trades_by_regime'].items():
            self.logger.info(f"  {regime}: {count}")
        
        # Top razões de rejeição
        if stats['rejections']:
            self.logger.info(f"\nTOP RAZÕES DE REJEIÇÃO:")
            sorted_rejections = sorted(stats['rejections'].items(), key=lambda x: x[1], reverse=True)[:5]
            for reason, count in sorted_rejections:
                pct = count / (self.signals_analyzed - self.signals_generated) * 100 if self.signals_analyzed > self.signals_generated else 0
                self.logger.info(f"  {reason}: {count} ({pct:.1f}%)")
        
        # Feature quality
        if stats['feature_quality_scores']:
            avg_quality = np.mean(stats['feature_quality_scores'])
            self.logger.info(f"\nQUALIDADE DAS FEATURES:")
            self.logger.info(f"Score médio: {avg_quality:.3f}")
            
        # Sugestões de otimização
        self.logger.info(f"\nSUGESTÕES DE OTIMIZAÇÃO:")
        suggestions = self.threshold_manager.suggest_threshold_adjustments()
        for suggestion in suggestions:
            self.logger.info(f"  - {suggestion}")
        
        # Salvar relatório
        report_data = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'signals_analyzed': self.signals_analyzed,
                'signals_generated': self.signals_generated,
                'trades_executed': self.trades_executed,
                'win_rate': len([t for t in self.trades if t['profit'] > 0]) / len(self.trades) * 100 if self.trades else 0
            },
            'trades': self.trades,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        }
        
        report_file = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"\nRelatório salvo em: {report_file}")
        self.logger.info("="*80)
    
    def _create_optimized_config(self) -> Dict:
        """Cria configuração otimizada"""
        
        # Tentar carregar configuração otimizada
        try:
            with open('config/improved_thresholds.json', 'r') as f:
                thresholds = json.load(f)
        except:
            # Fallback para configuração padrão
            thresholds = {
                "signal_generator": {
                    "direction_threshold": 0.25,
                    "magnitude_threshold": 0.0005,
                    "confidence_threshold": 0.50,
                    "risk_per_trade": 0.015
                }
            }
        
        config = {
            'signal_generator': {
                'direction_threshold': thresholds['signal_generator']['direction_threshold'],
                'magnitude_threshold': thresholds['signal_generator']['magnitude_threshold'],
                'confidence_threshold': thresholds['signal_generator']['confidence_threshold'],
                'risk_per_trade': thresholds['signal_generator']['risk_per_trade'],
                'point_value': 0.5,
                'min_stop_points': 5,
                'default_risk_reward': 2.0
            },
            'risk_manager': {
                'max_risk_per_trade': 0.02,
                'max_daily_loss': 0.05,
                'max_positions': 1,
                'point_value': 0.5
            }
        }
        
        return config


if __name__ == "__main__":
    # Executar backtest
    backtester = RealDataBacktester(
        data_file='wdo_data_20_06_2025.csv',
        initial_capital=100000
    )
    
    backtester.run_backtest()