"""
Integração da HybridStrategy com o sistema de trading
Conecta a estratégia híbrida ao TradingSystem principal
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime
import logging
from pathlib import Path

from .hybrid_strategy import HybridStrategy
from ..features.ml_features import MLFeatures
from ..features.book_features import BookFeatureEngineer

class HybridStrategyIntegration:
    """
    Integra a HybridStrategy ao sistema de trading principal
    Gerencia fluxo de dados e execução de trades
    """
    
    def __init__(self, trading_system: Any):
        self.trading_system = trading_system
        self.logger = logging.getLogger(__name__)
        
        # Configuração da estratégia
        strategy_config = {
            'models_path': 'models',
            'regime_threshold': 0.6,
            'tick_weight': 0.4,
            'book_weight': 0.6,
            'max_position': trading_system.config.get('max_position', 2),
            'stop_loss': trading_system.config.get('stop_loss', 0.02),
            'take_profit': trading_system.config.get('take_profit', 0.03)
        }
        
        # Inicializar estratégia
        self.strategy = HybridStrategy(strategy_config)
        
        # Feature engineers
        self.ml_features = MLFeatures()
        self.book_features = BookFeatureEngineer() if self._has_book_data() else None
        
        # Estado
        self.last_signal = None
        self.current_position = 0
        self.entry_price = 0
        self.last_update = None
        
        # Métricas
        self.signal_history = []
        self.trade_history = []
        
    def initialize(self):
        """Inicializa a estratégia híbrida"""
        
        self.logger.info("="*80)
        self.logger.info("INICIALIZANDO ESTRATÉGIA HÍBRIDA")
        self.logger.info("="*80)
        
        # Carregar modelos
        try:
            self.strategy.load_models()
            self.logger.info("[OK] Modelos carregados com sucesso")
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelos: {e}")
            raise
        
        # Verificar disponibilidade de dados
        if self.book_features:
            self.logger.info("[OK] Book features disponíveis")
        else:
            self.logger.warning("[AVISO] Book features não disponíveis - usando apenas tick data")
        
        self.logger.info("[OK] Estratégia híbrida inicializada")
        
    def process_data(self, market_data: Dict) -> Optional[Dict]:
        """
        Processa dados de mercado e gera sinais
        
        Args:
            market_data: Dados atuais do mercado
            
        Returns:
            Sinal de trading ou None
        """
        
        try:
            # 1. Extrair dados relevantes
            candles_df = market_data.get('candles')
            book_data = market_data.get('book_data')
            current_price = market_data.get('current_price')
            
            if candles_df is None or candles_df.empty:
                return None
            
            # 2. Calcular features tick
            tick_features = self._calculate_tick_features(candles_df)
            if tick_features is None:
                return None
            
            # 3. Calcular features book (se disponível)
            if self.book_features and book_data is not None:
                book_features_df = self._calculate_book_features(book_data)
            else:
                # Criar features book dummy se não disponível
                book_features_df = self._create_dummy_book_features()
            
            # 4. Obter sinal híbrido
            signal_info = self.strategy.get_hybrid_signal(
                tick_features, 
                book_features_df
            )
            
            # 5. Verificar se deve executar trade
            trade_decision = self._evaluate_trade_decision(
                signal_info, 
                current_price
            )
            
            if trade_decision:
                # Adicionar informações adicionais
                trade_decision['strategy'] = 'hybrid'
                trade_decision['features'] = {
                    'tick_features': tick_features.to_dict(orient='records')[0],
                    'book_features': book_features_df.to_dict(orient='records')[0]
                }
                
                # Registrar sinal
                self._record_signal(signal_info)
                
                return trade_decision
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro ao processar dados: {e}")
            return None
    
    def _calculate_tick_features(self, candles_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calcula features para modelo tick"""
        
        try:
            # Usar MLFeatures existente
            features = self.ml_features.calculate_all_features(candles_df)
            
            # Retornar última linha como DataFrame
            if isinstance(features, pd.DataFrame) and not features.empty:
                return features.tail(1)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular tick features: {e}")
            return None
    
    def _calculate_book_features(self, book_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calcula features para modelo book"""
        
        try:
            if self.book_features is None:
                return self._create_dummy_book_features()
            
            # Calcular features do book
            features = self.book_features.calculate_features(book_data)
            
            # Retornar última linha
            if isinstance(features, pd.DataFrame) and not features.empty:
                return features.tail(1)
            
            return self._create_dummy_book_features()
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular book features: {e}")
            return self._create_dummy_book_features()
    
    def _create_dummy_book_features(self) -> pd.DataFrame:
        """Cria features book dummy quando não há dados reais"""
        
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
            'hour': datetime.now().hour,
            'minute': datetime.now().minute,
            'time_normalized': datetime.now().hour / 24.0,
            'is_morning': 1.0 if datetime.now().hour < 12 else 0.0,
            'is_afternoon': 1.0 if datetime.now().hour >= 12 else 0.0
        }
        
        return pd.DataFrame([dummy_features])
    
    def _evaluate_trade_decision(self, signal_info: Dict, 
                                current_price: float) -> Optional[Dict]:
        """Avalia se deve executar trade baseado no sinal"""
        
        signal = signal_info['signal']
        confidence = signal_info['confidence']
        
        # Verificar se temos sinal e confiança suficiente
        if signal == 0 or confidence < 0.55:
            return None
        
        # Verificar se já temos posição
        if self.current_position != 0:
            # Verificar se é sinal de reversão
            if signal * self.current_position < 0:
                # Sinal oposto à posição atual
                if confidence > 0.65:  # Precisa de mais confiança para reverter
                    return self._create_close_position_order(current_price, "Reversão")
            else:
                # Mesmo direção - verificar stops
                return self._check_exit_conditions(current_price)
        else:
            # Sem posição - avaliar entrada
            if confidence > 0.6:
                return self._create_entry_order(signal_info, current_price)
        
        return None
    
    def _create_entry_order(self, signal_info: Dict, 
                           current_price: float) -> Dict:
        """Cria ordem de entrada"""
        
        signal = signal_info['signal']
        
        # Calcular tamanho da posição
        account_balance = self.trading_system.get_account_balance()
        position_size = self.strategy.calculate_position_size(
            signal_info, current_price, account_balance
        )
        
        if position_size == 0:
            return None
        
        # Calcular stops
        stop_loss = self.strategy.get_stop_loss(current_price, signal)
        take_profit = self.strategy.get_take_profit(current_price, signal)
        
        order = {
            'action': 'BUY' if signal == 1 else 'SELL',
            'type': 'MARKET',
            'quantity': position_size,
            'price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'signal_info': signal_info,
            'timestamp': datetime.now()
        }
        
        # Atualizar estado
        self.current_position = signal * position_size
        self.entry_price = current_price
        self.last_signal = signal_info
        
        self.logger.info(f"Ordem de entrada criada: {order['action']} "
                        f"{position_size} @ ${current_price:.2f}")
        
        return order
    
    def _create_close_position_order(self, current_price: float, 
                                    reason: str) -> Dict:
        """Cria ordem para fechar posição"""
        
        action = 'SELL' if self.current_position > 0 else 'BUY'
        quantity = abs(self.current_position)
        
        # Calcular P&L
        if self.current_position > 0:
            pnl = (current_price - self.entry_price) * quantity
        else:
            pnl = (self.entry_price - current_price) * quantity
        
        pnl_pct = (pnl / (self.entry_price * quantity)) * 100
        
        order = {
            'action': action,
            'type': 'MARKET',
            'quantity': quantity,
            'price': current_price,
            'reason': reason,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'timestamp': datetime.now()
        }
        
        # Registrar trade
        self._record_trade(order)
        
        # Resetar estado
        self.current_position = 0
        self.entry_price = 0
        self.last_signal = None
        
        self.logger.info(f"Ordem de saída criada: {action} {quantity} @ "
                        f"${current_price:.2f} (P&L: ${pnl:.2f} / {pnl_pct:.2f}%)")
        
        return order
    
    def _check_exit_conditions(self, current_price: float) -> Optional[Dict]:
        """Verifica condições de saída"""
        
        if self.current_position == 0 or self.entry_price == 0:
            return None
        
        # Calcular P&L atual
        if self.current_position > 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        # Verificar stop loss
        if pnl_pct <= -self.strategy.stop_loss:
            return self._create_close_position_order(current_price, "Stop Loss")
        
        # Verificar take profit
        if pnl_pct >= self.strategy.take_profit:
            return self._create_close_position_order(current_price, "Take Profit")
        
        # Verificar trailing stop (se implementado)
        # TODO: Implementar trailing stop
        
        return None
    
    def _has_book_data(self) -> bool:
        """Verifica se temos dados de book disponíveis"""
        
        # Verificar se há book collector configurado
        if hasattr(self.trading_system, 'book_collector'):
            return True
        
        # Verificar se há dados de book salvos
        book_path = Path('data/realtime/book')
        if book_path.exists() and any(book_path.iterdir()):
            return True
        
        return False
    
    def _record_signal(self, signal_info: Dict):
        """Registra sinal para análise posterior"""
        
        record = {
            'timestamp': signal_info['timestamp'],
            'signal': signal_info['signal'],
            'confidence': signal_info['confidence'],
            'regime': signal_info['regime'],
            'regime_confidence': signal_info['regime_confidence'],
            'tick_signal': signal_info['components']['tick']['signal'],
            'tick_confidence': signal_info['components']['tick']['confidence'],
            'book_signal': signal_info['components']['book']['signal'],
            'book_confidence': signal_info['components']['book']['confidence']
        }
        
        self.signal_history.append(record)
        
        # Manter apenas últimos 1000 sinais
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
    
    def _record_trade(self, trade_info: Dict):
        """Registra trade para análise posterior"""
        
        self.trade_history.append(trade_info)
        
        # Salvar histórico se tiver muitos trades
        if len(self.trade_history) >= 100:
            self._save_trade_history()
    
    def _save_trade_history(self):
        """Salva histórico de trades"""
        
        try:
            df = pd.DataFrame(self.trade_history)
            
            # Criar diretório se não existir
            output_dir = Path('results/hybrid_trades')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Salvar com timestamp
            filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(output_dir / filename, index=False)
            
            self.logger.info(f"Histórico de trades salvo: {filename}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar histórico: {e}")
    
    def get_performance_metrics(self) -> Dict:
        """Retorna métricas de performance da estratégia"""
        
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'total_pnl': 0.0,
                'sharpe_ratio': 0.0
            }
        
        trades_df = pd.DataFrame(self.trade_history)
        
        # Filtrar apenas trades fechados
        closed_trades = trades_df[trades_df['pnl'].notna()]
        
        if closed_trades.empty:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'total_pnl': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # Calcular métricas
        total_trades = len(closed_trades)
        winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
        win_rate = winning_trades / total_trades
        
        avg_pnl = closed_trades['pnl'].mean()
        total_pnl = closed_trades['pnl'].sum()
        
        # Sharpe ratio simplificado
        returns = closed_trades['pnl_pct'] / 100
        sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe,
            'avg_win': closed_trades[closed_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0,
            'avg_loss': closed_trades[closed_trades['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
        }