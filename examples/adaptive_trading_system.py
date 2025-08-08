"""
Sistema de Trading Adaptativo com Online Learning
Exemplo completo de uso da estratégia híbrida adaptativa
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
import json
import threading

from src.strategies.adaptive_hybrid_strategy import AdaptiveHybridStrategy
from src.features.ml_features_v3 import MLFeaturesV3
from src.technical_indicators import TechnicalIndicators

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class AdaptiveTradingSystem:
    """
    Sistema de trading completo com aprendizado contínuo
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Estratégia adaptativa
        self.strategy = AdaptiveHybridStrategy(config)
        
        # Feature calculators
        self.ml_features = MLFeaturesV3()
        self.tech_indicators = TechnicalIndicators()
        
        # Estado
        self.is_running = False
        self.current_position = 0
        self.entry_price = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        
        # Simulação de dados (em produção seria real-time)
        self.tick_data_generator = None
        self.book_data_generator = None
        
    def start(self):
        """Inicia o sistema de trading adaptativo"""
        
        self.logger.info("="*80)
        self.logger.info("INICIANDO SISTEMA DE TRADING ADAPTATIVO")
        self.logger.info("="*80)
        
        self.is_running = True
        
        # Iniciar estratégia adaptativa
        self.strategy.start_learning()
        
        # Iniciar threads de simulação
        self.start_data_simulation()
        
        # Thread principal de trading
        self.trading_thread = threading.Thread(
            target=self._trading_loop,
            name="TradingLoop"
        )
        self.trading_thread.start()
        
        self.logger.info("[OK] Sistema de trading iniciado")
        
    def stop(self):
        """Para o sistema de trading"""
        
        self.logger.info("\nParando sistema de trading...")
        self.is_running = False
        
        # Parar estratégia
        self.strategy.stop_learning()
        
        # Aguardar threads
        if hasattr(self, 'trading_thread'):
            self.trading_thread.join(timeout=5)
        
        # Imprimir resultados finais
        self._print_final_results()
        
    def start_data_simulation(self):
        """Inicia simulação de dados de mercado"""
        
        # Thread para gerar tick data
        def generate_tick_data():
            base_price = 5000
            while self.is_running:
                # Simular movimento de preço
                change = np.random.normal(0, 0.001)  # 0.1% volatilidade
                base_price *= (1 + change)
                
                # Criar tick data
                tick_data = pd.DataFrame([{
                    'timestamp': datetime.now(),
                    'price': base_price,
                    'volume': np.random.randint(1, 100),
                    'buy_volume': np.random.randint(0, 50),
                    'sell_volume': np.random.randint(0, 50),
                    'trades': np.random.randint(1, 10)
                }])
                
                # Adicionar ao sistema
                self.latest_tick_data = tick_data
                
                time.sleep(1)  # 1 tick por segundo
        
        # Thread para gerar book data
        def generate_book_data():
            while self.is_running:
                # Simular book data
                book_data = pd.DataFrame([{
                    'timestamp': datetime.now(),
                    'type': 'offer_book',
                    'ticker': 'WDOU25',
                    'position': np.random.randint(1, 20),
                    'price': self.latest_tick_data['price'].iloc[0] + np.random.uniform(-5, 5),
                    'quantity': np.random.randint(10, 1000),
                    'side': np.random.choice(['bid', 'ask']),
                    'hour': datetime.now().hour,
                    'minute': datetime.now().minute
                }])
                
                # Adicionar ao sistema
                self.latest_book_data = book_data
                
                time.sleep(2)  # Book update a cada 2 segundos
        
        # Iniciar threads
        self.tick_thread = threading.Thread(target=generate_tick_data, daemon=True)
        self.book_thread = threading.Thread(target=generate_book_data, daemon=True)
        
        self.tick_thread.start()
        self.book_thread.start()
        
        # Aguardar dados iniciais
        time.sleep(3)
        
    def _trading_loop(self):
        """Loop principal de trading"""
        
        self.logger.info("Trading loop iniciado")
        
        candle_buffer = []
        last_signal_time = datetime.now()
        
        while self.is_running:
            try:
                # Coletar dados recentes
                if hasattr(self, 'latest_tick_data'):
                    # Acumular para formar candles
                    candle_buffer.append(self.latest_tick_data)
                    
                    # Formar candle de 1 minuto
                    if len(candle_buffer) >= 60:  # 60 ticks = 1 minuto
                        candles = self._create_candles(candle_buffer)
                        candle_buffer = []
                        
                        # Calcular features
                        tick_features = self._calculate_features(candles)
                        
                        # Obter book features
                        book_features = self.latest_book_data if hasattr(self, 'latest_book_data') else None
                        
                        # Processar com estratégia adaptativa
                        signal_info = self.strategy.process_market_data(
                            tick_features,
                            book_features
                        )
                        
                        # Executar trade se necessário
                        self._execute_trade(signal_info, candles.iloc[-1])
                        
                        # Log periódico
                        if (datetime.now() - last_signal_time).total_seconds() > 30:
                            self._log_trading_status(signal_info)
                            last_signal_time = datetime.now()
                
                time.sleep(0.1)  # Loop rápido
                
            except Exception as e:
                self.logger.error(f"Erro no trading loop: {e}")
                time.sleep(1)
                
    def _create_candles(self, tick_buffer: list) -> pd.DataFrame:
        """Cria candles a partir de ticks"""
        
        # Combinar ticks
        ticks = pd.concat(tick_buffer, ignore_index=True)
        
        # Criar OHLCV
        candle = pd.DataFrame([{
            'timestamp': ticks['timestamp'].iloc[-1],
            'open': ticks['price'].iloc[0],
            'high': ticks['price'].max(),
            'low': ticks['price'].min(),
            'close': ticks['price'].iloc[-1],
            'volume': ticks['volume'].sum()
        }])
        
        return candle
        
    def _calculate_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Calcula features para o modelo"""
        
        # Simular features (em produção seria cálculo real)
        features = pd.DataFrame([{
            'returns_1': np.random.normal(0, 0.001),
            'returns_5': np.random.normal(0, 0.002),
            'returns_10': np.random.normal(0, 0.003),
            'returns_20': np.random.normal(0, 0.004),
            'returns_50': np.random.normal(0, 0.005),
            'volume_ma_10': candles['volume'].iloc[-1] / 100,
            'volume_ma_20': candles['volume'].iloc[-1] / 100,
            'volume_ma_50': candles['volume'].iloc[-1] / 100,
            'hour': datetime.now().hour,
            'minute': datetime.now().minute
        }])
        
        return features
        
    def _execute_trade(self, signal_info: Dict, current_candle: pd.Series):
        """Executa trades baseado no sinal"""
        
        signal = signal_info['signal']
        confidence = signal_info['confidence']
        current_price = current_candle['close']
        
        # Verificar posição atual
        if self.current_position != 0:
            # Calcular P&L
            if self.current_position > 0:  # Long
                pnl = (current_price - self.entry_price) * abs(self.current_position)
                pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:  # Short
                pnl = (self.entry_price - current_price) * abs(self.current_position)
                pnl_pct = (self.entry_price - current_price) / self.entry_price
            
            # Verificar saída
            should_exit = False
            exit_reason = ""
            
            # Stop loss
            if pnl_pct <= -0.02:
                should_exit = True
                exit_reason = "Stop Loss"
            # Take profit
            elif pnl_pct >= 0.03:
                should_exit = True
                exit_reason = "Take Profit"
            # Sinal reverso
            elif signal != 0 and signal != np.sign(self.current_position) and confidence > 0.65:
                should_exit = True
                exit_reason = "Signal Reversal"
            
            if should_exit:
                # Fechar posição
                self._close_position(current_price, pnl, exit_reason, signal_info)
        
        # Abrir nova posição se não temos posição e sinal forte
        if self.current_position == 0 and signal != 0 and confidence > 0.6:
            self._open_position(signal, current_price, signal_info)
            
    def _open_position(self, signal: int, price: float, signal_info: Dict):
        """Abre nova posição"""
        
        position_size = 1  # Simplificado
        
        self.current_position = signal * position_size
        self.entry_price = price
        self.total_trades += 1
        
        action = "BUY" if signal == 1 else "SELL"
        self.logger.info(f"\n[TRADE] {action} {position_size} @ ${price:.2f} "
                        f"(conf: {signal_info['confidence']:.2%}, "
                        f"regime: {signal_info['regime']})")
        
    def _close_position(self, price: float, pnl: float, reason: str, signal_info: Dict):
        """Fecha posição atual"""
        
        # Atualizar estatísticas
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
        
        # Informar estratégia do resultado
        trade_result = {
            'signal': np.sign(self.current_position),
            'confidence': signal_info['confidence'],
            'pnl': pnl,
            'model_type': signal_info.get('model_type', 'current')
        }
        self.strategy.update_trade_result(trade_result)
        
        # Log
        action = "SELL" if self.current_position > 0 else "BUY"
        self.logger.info(f"\n[CLOSE] {action} @ ${price:.2f} - {reason}")
        self.logger.info(f"         P&L: ${pnl:.2f} - Total P&L: ${self.total_pnl:.2f}")
        
        # Reset posição
        self.current_position = 0
        self.entry_price = 0
        
    def _log_trading_status(self, signal_info: Dict):
        """Log status periódico"""
        
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        self.logger.info("\n" + "-"*60)
        self.logger.info(f"Trades: {self.total_trades} | "
                        f"Win Rate: {win_rate:.2%} | "
                        f"P&L: ${self.total_pnl:.2f}")
        self.logger.info(f"Signal: {signal_info['signal']} | "
                        f"Confidence: {signal_info['confidence']:.2%} | "
                        f"Regime: {signal_info['regime']}")
        
        # Métricas adaptativas
        adaptive_info = signal_info.get('adaptive_info', {})
        if adaptive_info:
            self.logger.info(f"Adaptive - Accuracy: {adaptive_info.get('recent_accuracy', 0):.2%} | "
                            f"Models: Tick v{adaptive_info.get('model_versions', {}).get('tick', 0)}, "
                            f"Book v{adaptive_info.get('model_versions', {}).get('book', 0)}")
        
    def _print_final_results(self):
        """Imprime resultados finais"""
        
        print("\n" + "="*80)
        print("RESULTADOS FINAIS")
        print("="*80)
        
        if self.total_trades > 0:
            win_rate = self.winning_trades / self.total_trades
            avg_pnl = self.total_pnl / self.total_trades
            
            print(f"\nTotal de trades: {self.total_trades}")
            print(f"Taxa de acerto: {win_rate:.2%}")
            print(f"P&L total: ${self.total_pnl:.2f}")
            print(f"P&L médio: ${avg_pnl:.2f}")
        
        # Métricas adaptativas
        adaptive_metrics = self.strategy.get_adaptive_metrics()
        
        print("\n" + "-"*60)
        print("MÉTRICAS ADAPTATIVAS")
        print("-"*60)
        print(f"Predições totais: {adaptive_metrics['total_predictions']}")
        print(f"Accuracy recente: {adaptive_metrics['recent_accuracy']:.2%}")
        print(f"Thresholds adaptativos:")
        print(f"  Regime: {adaptive_metrics['adaptive_thresholds']['regime']:.2f}")
        print(f"  Confiança: {adaptive_metrics['adaptive_thresholds']['confidence']:.2f}")
        
        # A/B Testing
        ab_results = adaptive_metrics['ab_test_results']
        print("\n" + "-"*60)
        print("RESULTADOS A/B TESTING")
        print("-"*60)
        for model_type, results in ab_results.items():
            if results['trades'] > 0:
                wr = results['wins'] / results['trades']
                print(f"{model_type}: {results['trades']} trades, "
                      f"WR: {wr:.2%}, P&L: ${results['pnl']:.2f}")
        
        # Status do online learning
        ol_status = adaptive_metrics['online_learning_status']
        print("\n" + "-"*60)
        print("ONLINE LEARNING STATUS")
        print("-"*60)
        print(f"Buffer sizes - Tick: {ol_status['buffer_sizes']['tick']}, "
              f"Book: {ol_status['buffer_sizes']['book']}")
        print(f"Model versions - Tick: v{ol_status['model_versions']['tick']}, "
              f"Book: v{ol_status['model_versions']['book']}")

def main():
    """Executa sistema de trading adaptativo"""
    
    print("SISTEMA DE TRADING ADAPTATIVO COM ONLINE LEARNING")
    print("Demonstração de aprendizado contínuo em tempo real\n")
    
    # Configuração
    config = {
        'models_path': 'models',
        'regime_threshold': 0.6,
        'tick_weight': 0.4,
        'book_weight': 0.6,
        'max_position': 2,
        'stop_loss': 0.02,
        'take_profit': 0.03,
        # Online learning
        'online_buffer_size': 10000,
        'retrain_interval': 300,  # 5 minutos para demo
        'min_samples_retrain': 1000,
        'validation_window': 100,
        'performance_threshold': 0.55,
        # A/B testing
        'ab_testing_enabled': True,
        'ab_test_ratio': 0.2,
        # Adaptação
        'adaptation_rate': 0.1,
        'performance_window': 50
    }
    
    # Criar sistema
    system = AdaptiveTradingSystem(config)
    
    try:
        # Iniciar sistema
        system.start()
        
        print("\nSistema rodando... (Pressione Ctrl+C para parar)")
        print("O sistema irá:")
        print("1. Executar trades com modelos atuais")
        print("2. Coletar dados em tempo real")
        print("3. Retreinar modelos periodicamente")
        print("4. Fazer A/B testing entre modelos")
        print("5. Adaptar parâmetros automaticamente")
        
        # Rodar por tempo determinado (para demo)
        time.sleep(120)  # 2 minutos
        
    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário")
    finally:
        # Parar sistema
        system.stop()
        
        print("\n[OK] Sistema finalizado")

if __name__ == "__main__":
    main()