import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Set
import logging


class MLFeatures:
    """Calcula features específicas para ML"""
    
    def __init__(self, required_features: Optional[List[str]] = None):
        self.required_features = required_features or []
        self.logger = logging.getLogger(__name__)
        
        # Mapeamento de features por categoria
        self.feature_map = {
            'momentum': [],
            'volatility': [],
            'microstructure': [],
            'composite': []
        }
        
        # Features padrão se não especificadas
        self.default_features = {
            'momentum_periods': [1, 3, 5, 10, 15, 20],
            'volatility_periods': [5, 10, 20, 50],
            'return_periods': [5, 10, 20, 50],
            'volume_periods': [5, 10, 20]
        }
    
    def calculate_all(self, 
                     candles: pd.DataFrame,
                     microstructure: Optional[pd.DataFrame] = None,
                     indicators: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calcula todas as features ML"""
        
        if candles.empty:
            self.logger.warning("Candles vazios, retornando DataFrame vazio")
            return pd.DataFrame()
        
        features = pd.DataFrame(index=candles.index)
        
        # Features de momentum
        self._calculate_momentum_features(candles, features)
        
        # Features de volatilidade
        self._calculate_volatility_features(candles, features)
        
        # Features de volume
        self._calculate_volume_features(candles, features)
        
        # Features de microestrutura
        if microstructure is not None and not microstructure.empty:
            self._calculate_microstructure_features(microstructure, features)
        
        # Features compostas
        if indicators is not None and not indicators.empty:
            self._calculate_composite_features(candles, indicators, features, microstructure)
        
        # Features de padrões
        self._calculate_pattern_features(candles, features)
        
        # Filtrar apenas features requeridas se especificadas
        if self.required_features:
            available_features = [f for f in self.required_features if f in features.columns]
            features = features[available_features]
        
        self.logger.info(f"Features ML calculadas: {len(features.columns)} colunas")
        
        return features
    
    def _calculate_momentum_features(self, candles: pd.DataFrame, features: pd.DataFrame):
        """Calcula features de momentum"""
        try:
            for period in self.default_features['momentum_periods']:
                if len(candles) > period:
                    # Momentum absoluto
                    features[f'momentum_{period}'] = candles['close'] - candles['close'].shift(period)
                    
                    # Momentum percentual
                    features[f'momentum_pct_{period}'] = candles['close'].pct_change(period) * 100
                    
                    # Rate of change
                    features[f'roc_{period}'] = (
                        (candles['close'] - candles['close'].shift(period)) / 
                        candles['close'].shift(period).replace(0, 1)
                    ) * 100
            
            # Momentum aceleração
            if len(candles) > 10:
                mom_5 = candles['close'] - candles['close'].shift(5)
                features['momentum_acc'] = mom_5 - mom_5.shift(5)
            
            # Trend strength baseado em momentum
            if 'momentum_5' in features and 'momentum_20' in features:
                features['trend_consistency'] = (
                    (features['momentum_5'] > 0).astype(int) == 
                    (features['momentum_20'] > 0).astype(int)
                ).astype(int)
                
        except Exception as e:
            self.logger.error(f"Erro calculando momentum: {e}")
    
    def _calculate_volatility_features(self, candles: pd.DataFrame, features: pd.DataFrame):
        """Calcula features de volatilidade"""
        try:
            for period in self.default_features['volatility_periods']:
                if len(candles) > period:
                    # Desvio padrão dos retornos
                    returns = candles['close'].pct_change()
                    features[f'volatility_{period}'] = returns.rolling(period).std() * 100
                    
                    # High-Low range
                    features[f'high_low_range_{period}'] = (
                        candles['high'].rolling(period).max() - 
                        candles['low'].rolling(period).min()
                    )
                    
                    # Range percentual
                    features[f'range_pct_{period}'] = (
                        features[f'high_low_range_{period}'] / 
                        candles['close'].rolling(period).mean()
                    ) * 100
            
            # Volatilidade realizada
            if len(candles) > 20:
                returns = candles['close'].pct_change()
                features['realized_vol_5'] = returns.rolling(5).std() * np.sqrt(252) * 100
                features['realized_vol_20'] = returns.rolling(20).std() * np.sqrt(252) * 100
            
            # Volatility ratio
            if 'volatility_5' in features and 'volatility_20' in features:
                features['volatility_ratio'] = (
                    features['volatility_5'] / features['volatility_20'].replace(0, 1)
                )
                
        except Exception as e:
            self.logger.error(f"Erro calculando volatilidade: {e}")
    
    def _calculate_volume_features(self, candles: pd.DataFrame, features: pd.DataFrame):
        """Calcula features de volume"""
        try:
            for period in self.default_features['volume_periods']:
                if len(candles) > period:
                    # Volume médio
                    features[f'volume_ma_{period}'] = candles['volume'].rolling(period).mean()
                    
                    # Volume ratio
                    features[f'volume_ratio_{period}'] = (
                        candles['volume'] / features[f'volume_ma_{period}'].replace(0, 1)
                    )
            
            # VWAP (Volume Weighted Average Price)
            if len(candles) > 20:
                typical_price = (candles['high'] + candles['low'] + candles['close']) / 3
                features['vwap'] = (
                    (typical_price * candles['volume']).rolling(20).sum() / 
                    candles['volume'].rolling(20).sum().replace(0, 1)
                )
                features['price_to_vwap'] = (candles['close'] / features['vwap'] - 1) * 100
            
            # On-Balance Volume (OBV)
            if len(candles) > 1:
                obv = (candles['volume'] * np.sign(candles['close'].diff())).cumsum()
                features['obv'] = obv
                features['obv_ma_20'] = obv.rolling(20).mean()
                
        except Exception as e:
            self.logger.error(f"Erro calculando features de volume: {e}")
    
    def _calculate_microstructure_features(self, microstructure: pd.DataFrame, 
                                         features: pd.DataFrame):
        """Calcula features de microestrutura"""
        try:
            # Garantir alinhamento de índices
            common_index = features.index.intersection(microstructure.index)
            
            if len(common_index) > 0:
                # Buy pressure
                total_volume = (microstructure['buy_volume'] + microstructure['sell_volume']).replace(0, 1)
                features.loc[common_index, 'buy_pressure'] = (
                    microstructure.loc[common_index, 'buy_volume'] / total_volume.loc[common_index]
                )
                
                # Volume imbalance
                features.loc[common_index, 'volume_imbalance'] = microstructure.loc[common_index, 'imbalance']
                
                # Trade imbalance
                total_trades = (microstructure['buy_trades'] + microstructure['sell_trades']).replace(0, 1)
                features.loc[common_index, 'trade_imbalance'] = (
                    (microstructure.loc[common_index, 'buy_trades'] - 
                     microstructure.loc[common_index, 'sell_trades']) / 
                    total_trades.loc[common_index]
                )
                
                # Average trade size
                if 'avg_trade_size' in microstructure.columns:
                    features.loc[common_index, 'avg_trade_size'] = microstructure.loc[common_index, 'avg_trade_size']
                
                # Médias móveis de microestrutura
                for period in [5, 10, 20]:
                    if len(microstructure) > period:
                        features[f'buy_pressure_ma_{period}'] = features['buy_pressure'].rolling(period).mean()
                        features[f'volume_imbalance_ma_{period}'] = features['volume_imbalance'].rolling(period).mean()
                        
        except Exception as e:
            self.logger.error(f"Erro calculando microestrutura: {e}")
    
    def _calculate_composite_features(self, candles: pd.DataFrame, 
                                    indicators: pd.DataFrame,
                                    features: pd.DataFrame,
                                    microstructure: Optional[pd.DataFrame] = None):
        """Calcula features compostas usando indicadores"""
        try:
            # Price to EMAs
            for ema_period in [20, 50, 200]:
                if f'ema_{ema_period}' in indicators.columns:
                    features[f'price_to_ema{ema_period}'] = (
                        (candles['close'] / indicators[f'ema_{ema_period}']) - 1
                    ) * 100
            
            # EMA slopes
            for period in [20, 50]:
                if f'ema_{period}' in indicators.columns:
                    features[f'ema{period}_slope'] = (
                        indicators[f'ema_{period}'].pct_change(5) * 100
                    )
            
            # RSI features
            if 'rsi' in indicators.columns:
                features['rsi_ma_5'] = indicators['rsi'].rolling(5).mean()
                features['rsi_distance_50'] = indicators['rsi'] - 50
                features['rsi_extreme'] = (
                    (indicators['rsi'] < 30) | (indicators['rsi'] > 70)
                ).astype(int)
            
            # MACD features
            if all(col in indicators.columns for col in ['macd', 'macd_signal']):
                features['macd_divergence'] = indicators['macd'] - indicators['macd_signal']
                features['macd_cross'] = (
                    np.sign(features['macd_divergence']) != 
                    np.sign(features['macd_divergence'].shift(1))
                ).astype(int)
            
            # Bollinger Band features
            if all(col in indicators.columns for col in ['bb_upper_20', 'bb_lower_20']):
                bb_width = indicators['bb_upper_20'] - indicators['bb_lower_20']
                features['bb_squeeze'] = (
                    bb_width < bb_width.rolling(50).mean()
                ).astype(int)
            
            # Regime detection
            if 'volatility_20' in features and 'momentum_20' in features:
                # Volatility regime
                vol_median = features['volatility_20'].rolling(100).median()
                features['high_vol_regime'] = (
                    features['volatility_20'] > vol_median * 1.5
                ).astype(int)
                
                # Trend regime
                features['trend_regime'] = pd.cut(
                    features['momentum_20'],
                    bins=[-np.inf, -0.5, 0.5, np.inf],
                    labels=[-1, 0, 1]
                ).cat.codes - 1
                
        except Exception as e:
            self.logger.error(f"Erro calculando features compostas: {e}")
    
    def _calculate_pattern_features(self, candles: pd.DataFrame, features: pd.DataFrame):
        """Calcula features de padrões de preço"""
        try:
            if len(candles) < 5:
                return
            
            # Candle patterns
            body = candles['close'] - candles['open']
            upper_shadow = candles['high'] - candles[['open', 'close']].max(axis=1)
            lower_shadow = candles[['open', 'close']].min(axis=1) - candles['low']
            
            # Doji
            features['doji'] = (
                abs(body) < (candles['high'] - candles['low']) * 0.1
            ).astype(int)
            
            # Hammer/Hanging man
            features['hammer'] = (
                (lower_shadow > abs(body) * 2) & 
                (upper_shadow < abs(body) * 0.5)
            ).astype(int)
            
            # Engulfing
            prev_body = body.shift(1)
            features['bullish_engulfing'] = (
                (body > 0) & (prev_body < 0) & 
                (candles['close'] > candles['open'].shift(1)) &
                (candles['open'] < candles['close'].shift(1))
            ).astype(int)
            
            features['bearish_engulfing'] = (
                (body < 0) & (prev_body > 0) & 
                (candles['close'] < candles['open'].shift(1)) &
                (candles['open'] > candles['close'].shift(1))
            ).astype(int)
            
            # Support/Resistance levels
            if len(candles) > 20:
                # Pivot points
                pivot = (candles['high'] + candles['low'] + candles['close']) / 3
                features['pivot_point'] = pivot
                features['resistance_1'] = 2 * pivot - candles['low']
                features['support_1'] = 2 * pivot - candles['high']
                
                # Distance to recent high/low
                rolling_high = candles['high'].rolling(20).max()
                rolling_low = candles['low'].rolling(20).min()
                
                features['distance_to_high'] = (
                    (candles['close'] - rolling_high) / candles['close']
                ) * 100
                features['distance_to_low'] = (
                    (candles['close'] - rolling_low) / candles['close']
                ) * 100
                
        except Exception as e:
            self.logger.error(f"Erro calculando padrões: {e}")
    
    def get_feature_importance(self) -> Dict[str, List[str]]:
        """Retorna features agrupadas por importância/categoria"""
        return {
            'critical': [
                'momentum_5', 'momentum_10', 'volatility_20',
                'volume_ratio_20', 'buy_pressure', 'rsi'
            ],
            'important': [
                'price_to_ema20', 'macd_divergence', 'volume_imbalance',
                'trend_regime', 'high_vol_regime'
            ],
            'supplementary': [
                'distance_to_high', 'distance_to_low', 'pivot_point',
                'bullish_engulfing', 'bearish_engulfing'
            ]
        }