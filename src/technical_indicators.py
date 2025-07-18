import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging


class TechnicalIndicators:
    """Calcula indicadores técnicos"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.indicators = {}
    
    def calculate_all(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos os indicadores técnicos"""
        if candles.empty or len(candles) < 2:
            self.logger.warning("Dados insuficientes para calcular indicadores")
            return pd.DataFrame(index=candles.index)
        
        # DataFrame para armazenar indicadores
        indicators = pd.DataFrame(index=candles.index)
        
        # Médias móveis
        self._calculate_moving_averages(candles, indicators)
        
        # RSI
        self._calculate_rsi(candles, indicators)
        
        # MACD
        self._calculate_macd(candles, indicators)
        
        # Bollinger Bands
        self._calculate_bollinger_bands(candles, indicators)
        
        # Stochastic
        self._calculate_stochastic(candles, indicators)
        
        # ATR
        self._calculate_atr(candles, indicators)
        
        # ADX
        self._calculate_adx(candles, indicators)
        
        self.logger.info(f"Indicadores calculados: {len(indicators.columns)} colunas")
        
        return indicators
    
    def _calculate_moving_averages(self, candles: pd.DataFrame, indicators: pd.DataFrame):
        """Calcula médias móveis exponenciais e simples"""
        try:
            # EMAs principais
            ema_periods = [9, 20, 50, 200]
            for period in ema_periods:
                if len(candles) >= period:
                    indicators[f'ema_{period}'] = candles['close'].ewm(
                        span=period, adjust=False
                    ).mean()
            
            # SMAs
            sma_periods = [10, 20, 50]
            for period in sma_periods:
                if len(candles) >= period:
                    indicators[f'sma_{period}'] = candles['close'].rolling(
                        window=period
                    ).mean()
            
            # EMAs especiais para estratégias
            if len(candles) >= 8:
                indicators['ema_fast'] = candles['close'].ewm(span=8, adjust=False).mean()
            if len(candles) >= 26:
                indicators['ema_med'] = candles['close'].ewm(span=26, adjust=False).mean()
            if len(candles) >= 50:
                indicators['ema_long'] = candles['close'].ewm(span=50, adjust=False).mean()
            if len(candles) >= 200:
                indicators['ema_ultra'] = candles['close'].ewm(span=200, adjust=False).mean()
                
        except Exception as e:
            self.logger.error(f"Erro calculando médias móveis: {e}")
    
    def _calculate_rsi(self, candles: pd.DataFrame, indicators: pd.DataFrame, period: int = 14):
        """Calcula Relative Strength Index"""
        try:
            if len(candles) < period + 1:
                return
            
            delta = pd.to_numeric(candles['close'].diff(), errors='coerce')
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # Evitar divisão por zero
            rs = gain / loss.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            indicators['rsi'] = rsi
            indicators['rsi_14'] = rsi  # Alias para compatibilidade
            
            # RSI levels
            indicators['rsi_oversold'] = (rsi < 30).astype(int)
            indicators['rsi_overbought'] = (rsi > 70).astype(int)
            
        except Exception as e:
            self.logger.error(f"Erro calculando RSI: {e}")
    
    def _calculate_macd(self, candles: pd.DataFrame, indicators: pd.DataFrame,
                       fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """Calcula MACD"""
        try:
            if len(candles) < slow_period:
                return
            
            # MACD line
            ema_fast = candles['close'].ewm(span=fast_period, adjust=False).mean()
            ema_slow = candles['close'].ewm(span=slow_period, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            
            # Signal line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # Histogram
            histogram = macd_line - signal_line
            
            indicators['macd'] = macd_line
            indicators['macd_signal'] = signal_line
            indicators['macd_hist'] = histogram
            
            # MACD crossovers
            indicators['macd_bullish'] = (
                (macd_line > signal_line) & 
                (macd_line.shift(1) <= signal_line.shift(1))
            ).astype(int)
            
            indicators['macd_bearish'] = (
                (macd_line < signal_line) & 
                (macd_line.shift(1) >= signal_line.shift(1))
            ).astype(int)
            
        except Exception as e:
            self.logger.error(f"Erro calculando MACD: {e}")
    
    def _calculate_bollinger_bands(self, candles: pd.DataFrame, indicators: pd.DataFrame):
        """Calcula Bollinger Bands"""
        try:
            periods = [20, 50]
            
            for period in periods:
                if len(candles) < period:
                    continue
                
                # Média móvel (banda do meio)
                middle = candles['close'].rolling(window=period).mean()
                std = candles['close'].rolling(window=period).std()
                
                # Bandas superior e inferior
                upper = middle + (2 * std)
                lower = middle - (2 * std)
                
                indicators[f'bb_upper_{period}'] = upper
                indicators[f'bb_middle_{period}'] = middle
                indicators[f'bb_lower_{period}'] = lower
                indicators[f'bb_width_{period}'] = upper - lower
                
                # Posição do preço em relação às bandas (0-1)
                bb_position = (candles['close'] - lower) / (upper - lower).replace(0, 1)
                indicators[f'bb_position_{period}'] = bb_position.clip(0, 1)
                
                # Sinais
                indicators[f'bb_squeeze_{period}'] = (
                    indicators[f'bb_width_{period}'] < 
                    indicators[f'bb_width_{period}'].rolling(50).mean()
                ).astype(int) if len(candles) >= 50 else 0
                
        except Exception as e:
            self.logger.error(f"Erro calculando Bollinger Bands: {e}")
    
    def _calculate_stochastic(self, candles: pd.DataFrame, indicators: pd.DataFrame,
                            k_period: int = 14, d_period: int = 3):
        """Calcula Stochastic Oscillator"""
        try:
            if len(candles) < k_period:
                return
            
            # %K
            low_min = candles['low'].rolling(window=k_period).min()
            high_max = candles['high'].rolling(window=k_period).max()
            
            k_percent = 100 * ((candles['close'] - low_min) / 
                              (high_max - low_min).replace(0, 1))
            
            # %D
            d_percent = k_percent.rolling(window=d_period).mean()
            
            indicators['stoch_k'] = k_percent
            indicators['stoch_d'] = d_percent
            
            # Slow Stochastic
            indicators['slow_k'] = d_percent
            indicators['slow_d'] = d_percent.rolling(window=d_period).mean()
            
            # Sinais
            indicators['stoch_oversold'] = (k_percent < 20).astype(int)
            indicators['stoch_overbought'] = (k_percent > 80).astype(int)
            
        except Exception as e:
            self.logger.error(f"Erro calculando Stochastic: {e}")
    
    def _calculate_atr(self, candles: pd.DataFrame, indicators: pd.DataFrame, period: int = 14):
        """Calcula Average True Range"""
        try:
            if len(candles) < period:
                return
            
            # True Range
            high_low = candles['high'] - candles['low']
            high_close = abs(candles['high'] - candles['close'].shift(1))
            low_close = abs(candles['low'] - candles['close'].shift(1))
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # ATR
            atr = true_range.ewm(span=period, adjust=False).mean()
            indicators['atr'] = atr
            
            # ATR percentage (normalizado pelo preço)
            indicators['atr_pct'] = (atr / candles['close']) * 100
            
        except Exception as e:
            self.logger.error(f"Erro calculando ATR: {e}")
    
    def _calculate_adx(self, candles: pd.DataFrame, indicators: pd.DataFrame, period: int = 14):
        """Calcula Average Directional Index"""
        try:
            if len(candles) < period * 2:
                return
            
            # +DM e -DM
            plus_dm = pd.to_numeric(candles['high'].diff(), errors='coerce')
            minus_dm = pd.to_numeric(-candles['low'].diff(), errors='coerce')
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            # Quando ambos são positivos, manter apenas o maior
            mask = (plus_dm > 0) & (minus_dm > 0)
            plus_dm[mask & (plus_dm < minus_dm)] = 0
            minus_dm[mask & (minus_dm < plus_dm)] = 0
            
            # True Range (já calculado se ATR foi calculado)
            if 'atr' not in indicators:
                self._calculate_atr(candles, indicators, period)
            
            # Smooth DM
            plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / 
                           indicators['atr'].replace(0, 1))
            minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / 
                            indicators['atr'].replace(0, 1))
            
            # DX
            dx = 100 * (abs(plus_di - minus_di) / 
                       (plus_di + minus_di).replace(0, 1))
            
            # ADX
            adx = dx.ewm(span=period, adjust=False).mean()
            
            indicators['adx'] = adx
            indicators['plus_di'] = plus_di
            indicators['minus_di'] = minus_di
            
            # Força da tendência
            indicators['trend_strength'] = pd.cut(
                adx, bins=[0, 25, 50, 75, 100],
                labels=['weak', 'moderate', 'strong', 'very_strong']
            ).cat.codes
            
        except Exception as e:
            self.logger.error(f"Erro calculando ADX: {e}")
    
    def calculate_specific(self, candles: pd.DataFrame, 
                         indicator_names: List[str]) -> pd.DataFrame:
        """Calcula apenas indicadores específicos"""
        indicators = pd.DataFrame(index=candles.index)
        
        # Mapeamento de nomes para métodos
        indicator_methods = {
            'ema': self._calculate_moving_averages,
            'sma': self._calculate_moving_averages,
            'rsi': self._calculate_rsi,
            'macd': self._calculate_macd,
            'bb': self._calculate_bollinger_bands,
            'stoch': self._calculate_stochastic,
            'atr': self._calculate_atr,
            'adx': self._calculate_adx
        }
        
        # Calcular apenas os solicitados
        calculated = set()
        for name in indicator_names:
            for key, method in indicator_methods.items():
                if key in name and key not in calculated:
                    method(candles, indicators)
                    calculated.add(key)
                    break
        
        # Filtrar apenas as colunas solicitadas
        available_cols = [col for col in indicator_names if col in indicators.columns]
        return indicators[available_cols] if available_cols else indicators