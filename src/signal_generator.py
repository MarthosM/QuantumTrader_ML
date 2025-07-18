"""
Gerador de Sinais - Sistema v2.0
Gera sinais de trading baseados em predições ML
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging


class SignalGenerator:
    """Gera sinais de trading baseados em predições"""
    
    def __init__(self, config: Dict):
        """
        Inicializa o gerador de sinais
        
        Args:
            config: Configuração da estratégia
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Thresholds padrão
        self.direction_threshold = config.get('direction_threshold', 0.3)
        self.magnitude_threshold = config.get('magnitude_threshold', 0.0001)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        
        # Parâmetros de risco
        self.risk_per_trade = config.get('risk_per_trade', 0.02)
        self.max_positions = config.get('max_positions', 1)
        
        # Parâmetros WDO
        self.point_value = config.get('point_value', 0.5)
        self.min_stop_points = config.get('min_stop_points', 5)
        self.default_risk_reward = config.get('default_risk_reward', 2.0)
        
        # Estado
        self.last_signal = None
        self.signal_count = 0
        
    def generate_signal(self, prediction: Dict, market_data) -> Dict:
        """
        Gera sinal de trading baseado em predição
        
        Args:
            prediction: Resultado da predição ML
            market_data: TradingDataStructure com dados do mercado
            
        Returns:
            Dict com sinal de trading
        """
        self.logger.info("="*50)
        self.logger.info("[SIGNAL_GEN] Iniciando geração de sinal")
        
        # Validar entrada
        if not prediction:
            return self._create_empty_signal('no_prediction')
        
        if market_data.candles.empty:
            return self._create_empty_signal('no_market_data')
        
        # Extrair informações da predição
        direction = prediction.get('direction', 0)
        magnitude = prediction.get('magnitude', 0)
        confidence = prediction.get('confidence', 0)
        
        # Log dos valores
        self.logger.info(f"[SIGNAL_GEN] Predição recebida:")
        self.logger.info(f"  Direção: {direction:.3f}")
        self.logger.info(f"  Magnitude: {magnitude:.6f}")
        self.logger.info(f"  Confiança: {confidence:.3f}")
        
        # Verificar regime se disponível
        regime = prediction.get('regime', 'unknown')
        regime_confidence = prediction.get('regime_confidence', 0)
        
        if regime != 'unknown':
            self.logger.info(f"  Regime: {regime} (confiança: {regime_confidence:.2%})")
        
        # Validar thresholds
        validation_result = self._validate_thresholds(direction, magnitude, confidence)
        if validation_result is not None:
            return validation_result
        
        # Determinar ação baseada na direção
        if direction > self.direction_threshold:
            action = 'buy'
        elif direction < -self.direction_threshold:
            action = 'sell'
        else:
            return self._create_empty_signal('neutral_direction')
        
        # Obter preço atual
        current_price = float(market_data.candles['close'].iloc[-1])
        
        # Obter ATR para cálculo dinâmico
        atr = self._get_atr(market_data)
        
        # Calcular níveis de stop e take profit
        stop_loss, take_profit = self._calculate_levels(
            action, current_price, atr, magnitude, regime
        )
        
        # Calcular tamanho da posição
        position_size = self._calculate_position_size(
            current_price, stop_loss, market_data
        )
        
        # Criar sinal completo
        signal = {
            'timestamp': datetime.now(),
            'action': action,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': confidence,
            'position_size': position_size,
            'reason': self._get_signal_reason(direction, magnitude, confidence, regime),
            'metadata': {
                'prediction': prediction,
                'atr': atr,
                'regime': regime,
                'regime_confidence': regime_confidence,
                'direction_value': direction,
                'magnitude_value': magnitude
            }
        }
        
        # Calcular métricas adicionais
        signal['risk_reward'] = self._calculate_risk_reward(signal)
        signal['stop_points'] = abs(current_price - stop_loss) / self.point_value
        signal['take_points'] = abs(take_profit - current_price) / self.point_value
        
        # Validar sinal final
        if not self._validate_signal(signal):
            return self._create_empty_signal('invalid_signal_parameters')
        
        # Atualizar estado
        self.last_signal = signal
        self.signal_count += 1
        
        # Log do sinal gerado
        self._log_signal(signal)
        
        return signal
    
    def _validate_thresholds(self, direction: float, magnitude: float, 
                           confidence: float) -> Optional[Dict]:
        """Valida se os valores atendem aos thresholds mínimos"""
        
        if abs(direction) < self.direction_threshold:
            self.logger.info(
                f"[SIGNAL_GEN] Direção abaixo do threshold: "
                f"{abs(direction):.3f} < {self.direction_threshold}"
            )
            return self._create_empty_signal('direction_below_threshold')
        
        if magnitude < self.magnitude_threshold:
            self.logger.info(
                f"[SIGNAL_GEN] Magnitude abaixo do threshold: "
                f"{magnitude:.6f} < {self.magnitude_threshold}"
            )
            return self._create_empty_signal('magnitude_below_threshold')
        
        if confidence < self.confidence_threshold:
            self.logger.info(
                f"[SIGNAL_GEN] Confiança abaixo do threshold: "
                f"{confidence:.3f} < {self.confidence_threshold}"
            )
            return self._create_empty_signal('confidence_below_threshold')
        
        return None
    
    def _get_atr(self, market_data) -> float:
        """Obtém ATR dos indicadores ou calcula valor padrão"""
        if not market_data.indicators.empty and 'atr' in market_data.indicators.columns:
            atr_series = market_data.indicators['atr']
            if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]):
                return float(atr_series.iloc[-1])
        
        # Calcular ATR simples se não disponível
        if len(market_data.candles) >= 14:
            high = market_data.candles['high'].tail(14)
            low = market_data.candles['low'].tail(14)
            close = market_data.candles['close'].tail(14)
            
            tr = pd.DataFrame({
                'hl': high - low,
                'hc': abs(high - close.shift(1)),
                'lc': abs(low - close.shift(1))
            }).max(axis=1)
            
            return float(tr.mean())
        
        # Valor padrão baseado no preço
        current_price = float(market_data.candles['close'].iloc[-1])
        return current_price * 0.001  # 0.1% do preço
    
    def _calculate_levels(self, action: str, current_price: float, atr: float,
                         magnitude: float, regime: str) -> Tuple[float, float]:
        """Calcula níveis de stop loss e take profit"""
        
        # Stop baseado em ATR ou pontos mínimos
        atr_stop = atr * 1.5
        min_stop = self.min_stop_points * self.point_value
        stop_distance = max(atr_stop, min_stop)
        
        # Ajustar stop baseado no regime
        if regime == 'trend_up' or regime == 'trend_down':
            # Em tendência, usar stop mais apertado
            stop_distance *= 0.8
        elif regime == 'range':
            # Em lateralização, stop mais largo
            stop_distance *= 1.2
        
        # Take profit baseado em magnitude esperada e risk/reward
        expected_move = max(magnitude * current_price, stop_distance)
        take_distance = max(expected_move, stop_distance * self.default_risk_reward)
        
        # Calcular níveis finais
        if action == 'buy':
            stop_loss = current_price - stop_distance
            take_profit = current_price + take_distance
        else:  # sell
            stop_loss = current_price + stop_distance
            take_profit = current_price - take_distance
        
        # Arredondar para tick size (0.5 para WDO)
        stop_loss = round(stop_loss / self.point_value) * self.point_value
        take_profit = round(take_profit / self.point_value) * self.point_value
        
        return stop_loss, take_profit
    
    def _calculate_position_size(self, current_price: float, stop_loss: float,
                               market_data) -> int:
        """Calcula tamanho da posição baseado em risco"""
        # Por enquanto, retornar tamanho fixo
        # TODO: Implementar cálculo baseado em capital e risco
        return 1
    
    def _calculate_risk_reward(self, signal: Dict) -> float:
        """Calcula relação risco/retorno"""
        entry = signal['entry_price']
        stop = signal['stop_loss']
        take = signal['take_profit']
        
        risk = abs(entry - stop)
        reward = abs(take - entry)
        
        if risk > 0:
            return reward / risk
        return 0.0
    
    def _get_signal_reason(self, direction: float, magnitude: float, 
                          confidence: float, regime: str) -> str:
        """Gera descrição da razão do sinal"""
        reasons = []
        
        # Força da direção
        if abs(direction) > 0.7:
            reasons.append("strong_direction")
        elif abs(direction) > 0.5:
            reasons.append("moderate_direction")
        
        # Magnitude
        if magnitude > 0.002:
            reasons.append("high_magnitude")
        
        # Confiança
        if confidence > 0.8:
            reasons.append("high_confidence")
        
        # Regime
        if regime in ['trend_up', 'trend_down']:
            reasons.append(f"trending_{regime.split('_')[1]}")
        elif regime == 'range':
            reasons.append("range_trading")
        
        return "_".join(reasons) if reasons else "ml_prediction"
    
    def _validate_signal(self, signal: Dict) -> bool:
        """Valida parâmetros do sinal"""
        # Verificar stop mínimo
        stop_points = signal['stop_points']
        if stop_points < self.min_stop_points:
            self.logger.warning(f"Stop muito pequeno: {stop_points} pontos")
            return False
        
        # Verificar risk/reward
        if signal['risk_reward'] < 1.0:
            self.logger.warning(f"Risk/reward muito baixo: {signal['risk_reward']:.2f}")
            return False
        
        return True
    
    def _create_empty_signal(self, reason: str) -> Dict:
        """Cria sinal vazio (sem ação)"""
        return {
            'timestamp': datetime.now(),
            'action': 'none',
            'reason': reason,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'confidence': 0.0,
            'position_size': 0
        }
    
    def _log_signal(self, signal: Dict):
        """Log detalhado do sinal gerado"""
        self.logger.info("="*50)
        self.logger.info("[SIGNAL_GEN] SINAL GERADO:")
        self.logger.info(f"  Ação: {signal['action'].upper()}")
        self.logger.info(f"  Entrada: {signal['entry_price']:.2f}")
        self.logger.info(f"  Stop Loss: {signal['stop_loss']:.2f} ({signal['stop_points']:.1f} pts)")
        self.logger.info(f"  Take Profit: {signal['take_profit']:.2f} ({signal['take_points']:.1f} pts)")
        self.logger.info(f"  Risk/Reward: 1:{signal['risk_reward']:.1f}")
        self.logger.info(f"  Confiança: {signal['confidence']:.2%}")
        self.logger.info(f"  Razão: {signal['reason']}")
        self.logger.info("="*50)