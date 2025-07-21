import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Optional
from datetime import datetime

class StopStrategy(ABC):
    """Classe base para estratégias de stop loss"""
    
    @abstractmethod
    def calculate_stop(self, position: Dict, 
                      market_data: pd.DataFrame, 
                      market_regime: str) -> float:
        pass


class ATRAdaptiveStop(StopStrategy):
    """Stop loss adaptativo baseado em ATR"""
    
    def calculate_stop(self, position: Dict, 
                      market_data: pd.DataFrame, 
                      market_regime: str) -> float:
        
        # Calcular ATR
        atr = self._calculate_atr(market_data, period=14)
        current_atr = atr.iloc[-1]
        
        # Multiplicador baseado no regime
        multipliers = {
            'high_volatility': 3.0,
            'trending': 2.5,
            'ranging': 2.0,
            'low_volatility': 1.5
        }
        
        multiplier = multipliers.get(market_regime, 2.0)
        
        # Calcular stop
        if position['side'] == 'long':
            stop_price = position['current_price'] - (current_atr * multiplier)
        else:
            stop_price = position['current_price'] + (current_atr * multiplier)
            
        return stop_price
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calcula Average True Range"""
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr


class SupportResistanceStop(StopStrategy):
    """Stop loss baseado em suporte e resistência"""
    
    def calculate_stop(self, position: Dict, 
                      market_data: pd.DataFrame, 
                      market_regime: str) -> float:
        
        # Identificar níveis de suporte/resistência
        levels = self._identify_sr_levels(market_data)
        
        current_price = position['current_price']
        
        if position['side'] == 'long':
            # Encontrar suporte mais próximo abaixo do preço
            supports = [l for l in levels if l < current_price]
            if supports:
                stop_price = max(supports) * 0.995  # 0.5% abaixo do suporte
            else:
                stop_price = current_price * 0.98  # 2% default
        else:
            # Encontrar resistência mais próxima acima do preço
            resistances = [l for l in levels if l > current_price]
            if resistances:
                stop_price = min(resistances) * 1.005  # 0.5% acima da resistência
            else:
                stop_price = current_price * 1.02  # 2% default
                
        return stop_price
    
    def _identify_sr_levels(self, data: pd.DataFrame, window: int = 20) -> list:
        """Identifica níveis de suporte e resistência"""
        
        # Método simplificado - pivots
        highs = data['high'].rolling(window=window, center=True).max()
        lows = data['low'].rolling(window=window, center=True).min()
        
        # Pontos onde o preço é igual ao máximo/mínimo local
        resistance_mask = data['high'] == highs
        support_mask = data['low'] == lows
        
        resistance_levels = data.loc[resistance_mask, 'high'].unique()
        support_levels = data.loc[support_mask, 'low'].unique()
        
        all_levels = np.concatenate([resistance_levels, support_levels])
        
        # Filtrar níveis muito próximos
        filtered_levels = []
        for level in sorted(all_levels):
            if not filtered_levels or level > filtered_levels[-1] * 1.005:
                filtered_levels.append(level)
                
        return filtered_levels


class VolatilityBasedStop(StopStrategy):
    """Stop loss baseado em volatilidade"""
    
    def calculate_stop(self, position: Dict, 
                      market_data: pd.DataFrame, 
                      market_regime: str) -> float:
        
        # Calcular volatilidade realizada
        returns = market_data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        current_vol = volatility.iloc[-1]
        
        # Ajustar para horizonte de tempo (assumindo dados de 1 minuto)
        # Volatilidade para próximos 30 minutos
        adjusted_vol = current_vol * np.sqrt(30)
        
        # Multiplicador baseado em confiança
        confidence_multiplier = 2.0  # 2 desvios padrão (~95% confiança)
        
        # Calcular stop
        stop_distance = position['current_price'] * adjusted_vol * confidence_multiplier
        
        if position['side'] == 'long':
            stop_price = position['current_price'] - stop_distance
        else:
            stop_price = position['current_price'] + stop_distance
            
        return stop_price


class StopLossFeatureCalculator:
    """Calculadora de features para otimização de stop loss por ML"""
    
    def __init__(self):
        self.feature_names = [
            'atr_normalized', 'volatility_realized', 'price_position',
            'volume_profile', 'momentum_short', 'momentum_long',
            'support_distance', 'resistance_distance', 'regime_score',
            'time_since_entry', 'unrealized_pnl_pct', 'market_stress'
        ]
    
    def calculate_features(self, position: Dict, 
                          market_data: pd.DataFrame, 
                          market_regime: str) -> list:
        """
        Calcula features para ML de stop loss
        
        Args:
            position: Dados da posição atual
            market_data: DataFrame com dados de mercado
            market_regime: Regime de mercado atual
            
        Returns:
            list: Lista de features calculadas
        """
        try:
            features = []
            
            # 1. ATR normalizado pelo preço
            atr = self._calculate_atr(market_data, 14)
            current_atr = atr.iloc[-1] if not atr.empty else 0
            atr_normalized = current_atr / position['current_price']
            features.append(atr_normalized)
            
            # 2. Volatilidade realizada (20 períodos)
            returns = market_data['close'].pct_change()
            volatility = returns.rolling(20).std().iloc[-1]
            volatility = volatility if not pd.isna(volatility) else 0
            features.append(volatility)
            
            # 3. Posição do preço no range (0-1)
            high_20 = market_data['high'].rolling(20).max().iloc[-1]
            low_20 = market_data['low'].rolling(20).min().iloc[-1]
            if high_20 != low_20:
                price_position = (position['current_price'] - low_20) / (high_20 - low_20)
            else:
                price_position = 0.5
            features.append(price_position)
            
            # 4. Profile de volume (volume atual vs média)
            avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
            current_volume = market_data['volume'].iloc[-1]
            volume_profile = current_volume / avg_volume if avg_volume > 0 else 1.0
            features.append(min(volume_profile, 5.0))  # Cap em 5x
            
            # 5. Momentum de curto prazo (5 períodos)
            momentum_short = (position['current_price'] / market_data['close'].iloc[-6] - 1) if len(market_data) >= 6 else 0
            features.append(momentum_short)
            
            # 6. Momentum de longo prazo (20 períodos)
            momentum_long = (position['current_price'] / market_data['close'].iloc[-21] - 1) if len(market_data) >= 21 else 0
            features.append(momentum_long)
            
            # 7. Distância do suporte mais próximo
            support_distance = self._calculate_support_distance(market_data, position['current_price'])
            features.append(support_distance)
            
            # 8. Distância da resistência mais próxima
            resistance_distance = self._calculate_resistance_distance(market_data, position['current_price'])
            features.append(resistance_distance)
            
            # 9. Score do regime de mercado
            regime_score = self._encode_market_regime(market_regime)
            features.append(regime_score)
            
            # 10. Tempo desde entrada (em minutos, normalizado)
            time_since_entry = self._calculate_time_since_entry(position)
            features.append(min(time_since_entry / 60.0, 1.0))  # Normalizar para 1 hora
            
            # 11. PnL não realizado percentual
            if 'entry_price' in position:
                if position['side'] == 'long':
                    unrealized_pnl = (position['current_price'] - position['entry_price']) / position['entry_price']
                else:
                    unrealized_pnl = (position['entry_price'] - position['current_price']) / position['entry_price']
            else:
                unrealized_pnl = 0
            features.append(unrealized_pnl)
            
            # 12. Indicador de stress de mercado
            market_stress = self._calculate_market_stress(market_data)
            features.append(market_stress)
            
            return features
            
        except Exception as e:
            # Retornar features padrão em caso de erro
            return [0.01, 0.02, 0.5, 1.0, 0.0, 0.0, 0.02, 0.02, 0.5, 0.0, 0.0, 0.5]
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calcula Average True Range"""
        if len(data) < period:
            return pd.Series([data['high'].iloc[-1] - data['low'].iloc[-1]] * len(data))
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.fillna(tr.mean())
    
    def _calculate_support_distance(self, data: pd.DataFrame, current_price: float) -> float:
        """Calcula distância normalizada do suporte mais próximo"""
        try:
            # Encontrar mínimos locais como suportes
            window = min(10, len(data) // 2)
            if window < 3:
                return 0.02  # Default 2%
                
            low_min = data['low'].rolling(window=window, center=True).min()
            supports = data.loc[data['low'] == low_min, 'low'].unique()
            
            # Filtrar suportes abaixo do preço atual
            valid_supports = [s for s in supports if s < current_price and s > 0]
            
            if valid_supports:
                nearest_support = max(valid_supports)
                distance = (current_price - nearest_support) / current_price
                return min(distance, 0.1)  # Cap em 10%
            else:
                return 0.02  # Default 2%
                
        except Exception:
            return 0.02
    
    def _calculate_resistance_distance(self, data: pd.DataFrame, current_price: float) -> float:
        """Calcula distância normalizada da resistência mais próxima"""
        try:
            # Encontrar máximos locais como resistências
            window = min(10, len(data) // 2)
            if window < 3:
                return 0.02  # Default 2%
                
            high_max = data['high'].rolling(window=window, center=True).max()
            resistances = data.loc[data['high'] == high_max, 'high'].unique()
            
            # Filtrar resistências acima do preço atual
            valid_resistances = [r for r in resistances if r > current_price]
            
            if valid_resistances:
                nearest_resistance = min(valid_resistances)
                distance = (nearest_resistance - current_price) / current_price
                return min(distance, 0.1)  # Cap em 10%
            else:
                return 0.02  # Default 2%
                
        except Exception:
            return 0.02
    
    def _encode_market_regime(self, regime: str) -> float:
        """Codifica regime de mercado em valor numérico"""
        regime_mapping = {
            'trend_up': 0.8,
            'trend_down': 0.2,
            'ranging': 0.5,
            'high_volatility': 0.9,
            'low_volatility': 0.1,
            'normal': 0.5,
            'undefined': 0.5
        }
        return regime_mapping.get(regime, 0.5)
    
    def _calculate_time_since_entry(self, position: Dict) -> float:
        """Calcula tempo desde entrada em minutos"""
        try:
            from datetime import datetime
            
            if 'entry_time' in position:
                if isinstance(position['entry_time'], str):
                    entry_time = pd.to_datetime(position['entry_time'])
                else:
                    entry_time = position['entry_time']
                
                current_time = datetime.now()
                time_diff = (current_time - entry_time).total_seconds() / 60.0
                return max(0, time_diff)
            else:
                return 5.0  # Default 5 minutos
                
        except Exception:
            return 5.0
    
    def _calculate_market_stress(self, data: pd.DataFrame) -> float:
        """Calcula indicador de stress de mercado (0-1)"""
        try:
            if len(data) < 10:
                return 0.5
            
            # Combinar múltiplos indicadores de stress
            
            # 1. Volatilidade vs histórica
            recent_vol = data['close'].pct_change().rolling(5).std().iloc[-1]
            historical_vol = data['close'].pct_change().rolling(20).std().iloc[-1]
            vol_stress = min(recent_vol / historical_vol, 2.0) / 2.0 if historical_vol > 0 else 0.5
            
            # 2. Volume anormal
            recent_volume = data['volume'].rolling(5).mean().iloc[-1]
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            volume_stress = min(recent_volume / avg_volume, 3.0) / 3.0 if avg_volume > 0 else 0.5
            
            # 3. Range expansion
            recent_range = (data['high'] - data['low']).rolling(5).mean().iloc[-1]
            avg_range = (data['high'] - data['low']).rolling(20).mean().iloc[-1]
            range_stress = min(recent_range / avg_range, 2.0) / 2.0 if avg_range > 0 else 0.5
            
            # Combinar com pesos
            market_stress = (vol_stress * 0.4 + volume_stress * 0.3 + range_stress * 0.3)
            return max(0.0, min(market_stress, 1.0))
            
        except Exception:
            return 0.5
    
    def get_feature_names(self) -> list:
        """Retorna nomes das features"""
        return self.feature_names.copy()


class MLOptimizedStop(StopStrategy):
    """Stop loss otimizado por ML"""
    
    def __init__(self):
        # Em produção, carregar modelo treinado
        self.model = None
        self.feature_calculator = StopLossFeatureCalculator()
        
    def calculate_stop(self, position: Dict, 
                      market_data: pd.DataFrame, 
                      market_regime: str) -> float:
        
        # Calcular features
        features = self.feature_calculator.calculate_features(
            position, market_data, market_regime
        )
        
        # Se modelo não disponível, usar fallback
        if self.model is None:
            return self._fallback_calculation(position, market_data)
            
        # Predizer stop ótimo
        stop_distance_pct = self.model.predict([features])[0]
        
        # Aplicar limites de segurança
        stop_distance_pct = max(0.005, min(stop_distance_pct, 0.05))  # 0.5% a 5%
        
        # Calcular stop
        if position['side'] == 'long':
            stop_price = position['current_price'] * (1 - stop_distance_pct)
        else:
            stop_price = position['current_price'] * (1 + stop_distance_pct)
            
        return stop_price
    
    def _fallback_calculation(self, position: Dict, 
                            market_data: pd.DataFrame) -> float:
        """Cálculo fallback quando ML não disponível"""
        
        # Usar ATR como fallback
        atr_stop = ATRAdaptiveStop()
        return atr_stop.calculate_stop(position, market_data, 'normal')