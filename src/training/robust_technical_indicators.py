"""
Sistema Robusto de Indicadores Técnicos
Substitui TA-Lib com implementações precisas e otimizadas
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
import warnings

logger = logging.getLogger(__name__)

class RobustTechnicalIndicators:
    """
    Indicadores técnicos implementados de forma robusta
    Substitui TA-Lib com precisão matemática equivalente
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos os indicadores técnicos de forma robusta"""
        indicators = pd.DataFrame(index=data.index)
        
        try:
            # EMAs com diferentes períodos
            for period in [9, 20, 50, 200]:
                indicators[f'ema_{period}'] = self.calculate_ema(data['close'], period)
            
            # RSI com diferentes períodos
            for period in [9, 14, 25]:
                col_name = 'rsi' if period == 14 else f'rsi_{period}'
                indicators[col_name] = self.calculate_rsi(data['close'], period)
            
            # MACD completo
            macd_data = self.calculate_macd(data['close'])
            indicators['macd'] = macd_data['macd']
            indicators['macd_signal'] = macd_data['signal']
            indicators['macd_hist'] = macd_data['histogram']
            
            # Bollinger Bands
            for period in [20, 50]:
                bb_data = self.calculate_bollinger_bands(data['close'], period)
                indicators[f'bb_upper_{period}'] = bb_data['upper']
                indicators[f'bb_middle_{period}'] = bb_data['middle']
                indicators[f'bb_lower_{period}'] = bb_data['lower']
                indicators[f'bb_width_{period}'] = bb_data['width']
                indicators[f'bb_position_{period}'] = bb_data['position']
            
            # ATR
            atr_data = self.calculate_atr(data)
            indicators['atr'] = atr_data['atr_14']
            indicators['atr_20'] = atr_data['atr_20']
            indicators['true_range'] = atr_data['true_range']
            
            # ADX
            indicators['adx'] = self.calculate_adx(data)
            
            self.logger.info(f"Calculados {len(indicators.columns)} indicadores técnicos")
            
        except Exception as e:
            self.logger.error(f"Erro calculando indicadores: {e}")
            
        return indicators
    
    def calculate_ema(self, series: pd.Series, period: int, 
                     adjust: bool = False) -> pd.Series:
        """
        EMA robusto - Exponential Moving Average
        Implementação matematicamente precisa
        """
        if len(series) < period:
            return pd.Series(np.nan, index=series.index)
        
        # Usar implementação do pandas que é robusta
        return series.ewm(span=period, adjust=adjust).mean()
    
    def calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average robusto"""
        return series.rolling(window=period, min_periods=1).mean()
    
    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI robusto - Relative Strength Index
        Implementação precisa do algoritmo Wilder
        """
        if len(series) < period + 1:
            return pd.Series(np.nan, index=series.index)
        
        delta = series.diff()
        
        # Ensure delta is numeric and handle type issues
        delta = pd.to_numeric(delta, errors='coerce')
        
        # Separar ganhos e perdas
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        # Método Wilder para suavização (similar ao TA-Lib)
        alpha = 1.0 / period
        
        # Primeira média simples
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Aplicar suavização exponencial Wilder
        for i in range(period, len(series)):
            avg_gain.iloc[i] = alpha * gains.iloc[i] + (1 - alpha) * avg_gain.iloc[i-1]
            avg_loss.iloc[i] = alpha * losses.iloc[i] + (1 - alpha) * avg_loss.iloc[i-1]
        
        # Calcular RS e RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, series: pd.Series, 
                      fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        MACD completo - Moving Average Convergence Divergence
        Retorna MACD line, Signal line e Histogram
        """
        ema_fast = self.calculate_ema(series, fast)
        ema_slow = self.calculate_ema(series, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, series: pd.Series, period: int = 20, 
                                std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Bollinger Bands completas
        Retorna upper, middle, lower, width e position
        """
        middle = self.calculate_sma(series, period)
        std = series.rolling(window=period).std()
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        width = upper - lower
        
        # Posição normalizada (0-1)
        position = (series - lower) / width
        position = position.clip(0, 1)  # Manter entre 0 e 1
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width,
            'position': position
        }
    
    def calculate_atr(self, data: pd.DataFrame, periods: Optional[list] = None) -> Dict[str, pd.Series]:
        """
        ATR robusto - Average True Range
        Calcula True Range e ATR para múltiplos períodos
        """
        if periods is None:
            periods = [14, 20]
        
        # True Range calculation
        high_low = data['high'] - data['low']
        high_close_prev = (data['high'] - data['close'].shift(1)).abs()
        low_close_prev = (data['low'] - data['close'].shift(1)).abs()
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        result = {'true_range': true_range}
        
        for period in periods:
            # Usar método Wilder para ATR (como TA-Lib)
            atr = true_range.rolling(window=period).mean()
            
            # Aplicar suavização Wilder
            alpha = 1.0 / period
            for i in range(period, len(true_range)):
                atr.iloc[i] = alpha * true_range.iloc[i] + (1 - alpha) * atr.iloc[i-1]
            
            result[f'atr_{period}'] = atr
        
        return result
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        ADX robusto - Average Directional Index
        Implementação completa do algoritmo Wilder
        """
        # True Range já calculado
        tr = self.calculate_atr(data, [period])[f'atr_{period}']
        
        # Directional Movement
        high_diff = pd.to_numeric(data['high'].diff(), errors='coerce')
        low_diff = pd.to_numeric(data['low'].diff(), errors='coerce')
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        plus_dm_series = pd.Series(plus_dm, index=data.index)
        minus_dm_series = pd.Series(minus_dm, index=data.index)
        
        # Smooth DM usando método Wilder
        plus_dm_smooth = plus_dm_series.rolling(window=period).mean()
        minus_dm_smooth = minus_dm_series.rolling(window=period).mean()
        tr_smooth = tr
        
        # Aplicar suavização Wilder
        alpha = 1.0 / period
        for i in range(period, len(data)):
            plus_dm_smooth.iloc[i] = alpha * plus_dm_series.iloc[i] + (1 - alpha) * plus_dm_smooth.iloc[i-1]
            minus_dm_smooth.iloc[i] = alpha * minus_dm_series.iloc[i] + (1 - alpha) * minus_dm_smooth.iloc[i-1]
        
        # Directional Indicators
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # DX - manter como pandas Series
        dx_numerator = (plus_di - minus_di).abs()
        dx_denominator = plus_di + minus_di
        
        # Evitar divisão por zero
        dx_denominator = dx_denominator.replace(0, np.nan)
        dx = 100 * (dx_numerator / dx_denominator)
        
        # ADX (smoothed DX) - inicializar como Series
        adx = dx.rolling(window=period, min_periods=1).mean()
        
        # Aplicar suavização final Wilder para ADX
        for i in range(period * 2, len(data)):
            if not pd.isna(dx.iloc[i]) and not pd.isna(adx.iloc[i-1]):
                adx.iloc[i] = alpha * dx.iloc[i] + (1 - alpha) * adx.iloc[i-1]
        
        return adx
    
    def calculate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de momentum robustas"""
        momentum_features = pd.DataFrame(index=data.index)
        
        # Momentum básico para diferentes períodos
        for period in [1, 3, 5, 10, 15, 20]:
            momentum_features[f'momentum_{period}'] = data['close'].diff(periods=period)
            momentum_features[f'momentum_pct_{period}'] = data['close'].pct_change(periods=period)
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            momentum_features[f'roc_{period}'] = ((data['close'] - data['close'].shift(period)) / data['close'].shift(period)) * 100
        
        # Returns
        for period in [5, 10, 20, 50]:
            momentum_features[f'return_{period}'] = data['close'].pct_change(periods=period)
        
        # Price acceleration
        momentum_features['price_acceleration'] = data['close'].diff().diff()
        
        return momentum_features
    
    def calculate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de volatilidade robustas"""
        vol_features = pd.DataFrame(index=data.index)
        
        # Volatilidade padrão
        for period in [5, 10, 20, 50]:
            returns = data['close'].pct_change()
            vol_features[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
        
        # Parkinson Volatility (usa High-Low)
        for period in [10, 20]:
            ln_hl = (data['high'] / data['low']).apply(np.log)  # Manter como Series
            parkinson = np.sqrt((1/(4*np.log(2))) * (ln_hl**2).rolling(window=period).mean()) * np.sqrt(252)
            vol_features[f'parkinson_vol_{period}'] = parkinson
        
        # Garman-Klass Volatility
        for period in [10, 20]:
            ln_ho = (data['high'] / data['open']).apply(np.log)  # Manter como Series
            ln_lo = (data['low'] / data['open']).apply(np.log)   # Manter como Series
            ln_co = (data['close'] / data['open']).apply(np.log) # Manter como Series
            
            gk = ln_ho * (ln_ho - ln_co) + ln_lo * (ln_lo - ln_co)
            vol_features[f'gk_vol_{period}'] = (gk.rolling(window=period).mean().apply(np.sqrt)) * np.sqrt(252)
        
        # High-Low range features
        for period in [5, 10, 20]:
            vol_features[f'high_low_range_{period}'] = (
                data['high'].rolling(window=period).max() - 
                data['low'].rolling(window=period).min()
            ) / data['close']
        
        # Volatility ratio
        vol_features['volatility_ratio'] = vol_features['volatility_20'] / vol_features['volatility_50']
        
        return vol_features
    
    def calculate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de volume robustas"""
        volume_features = pd.DataFrame(index=data.index)
        
        # Volume momentum
        for period in [5, 10, 20]:
            volume_features[f'volume_momentum_{period}'] = data['volume'].pct_change(periods=period)
        
        # Volume SMA e ratios
        for period in [5, 10, 20]:
            vol_sma = data['volume'].rolling(window=period).mean()
            volume_features[f'volume_sma_{period}'] = vol_sma
            volume_features[f'volume_ratio_{period}'] = data['volume'] / vol_sma
        
        # Price Volume Trend
        pvt = ((data['close'] - data['close'].shift()) / data['close'].shift()) * data['volume']
        volume_features['price_volume_trend'] = pvt.cumsum()
        
        # VWAP
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        volume_features['vwap'] = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        
        return volume_features
    
    def calculate_lag_features(self, base_features: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de lag robustas"""
        lag_features = pd.DataFrame(index=base_features.index)
        
        # Lags para RSI
        if 'rsi' in base_features.columns:
            for lag in [1, 5, 10, 20]:
                lag_features[f'rsi_lag_{lag}'] = base_features['rsi'].shift(lag)
        
        # Lags para MACD
        if 'macd' in base_features.columns:
            for lag in [1, 5, 10, 20]:
                lag_features[f'macd_lag_{lag}'] = base_features['macd'].shift(lag)
        
        # Lags para volatilidade
        if 'volatility_20' in base_features.columns:
            for lag in [1, 5, 10, 20]:
                lag_features[f'volatility_20_lag_{lag}'] = base_features['volatility_20'].shift(lag)
        
        # Lags para momentum
        if 'momentum_5' in base_features.columns:
            for lag in [1, 5, 10, 20]:
                lag_features[f'momentum_5_lag_{lag}'] = base_features['momentum_5'].shift(lag)
        
        return lag_features
    
    def calculate_cross_features(self, base_features: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de cruzamento"""
        cross_features = pd.DataFrame(index=base_features.index)
        
        # MACD cross change
        if 'macd' in base_features.columns and 'macd_signal' in base_features.columns:
            macd_cross = (base_features['macd'] > base_features['macd_signal']).astype(int)
            cross_features['macd_cross_change'] = macd_cross.diff().fillna(0)
        
        return cross_features
