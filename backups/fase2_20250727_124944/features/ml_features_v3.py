"""
MLFeaturesV3 - Cálculo de features com dados reais microestruturais
Baseado no DEVELOPER_GUIDE_V3_REFACTORING.md
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class MLFeaturesV3:
    """
    Calcula features ML avançadas usando dados reais de microestrutura
    
    Features Categories:
    1. Momentum (price-based and volume-weighted)
    2. Volatility (with microstructure adjustments)
    3. Volume and Microstructure
    4. Technical Indicators Enhanced
    5. Market Regime Features
    6. Order Flow Imbalance
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Inicializa o calculador de features V3
        
        Args:
            logger: Logger opcional
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuração de períodos padrão
        self.momentum_periods = [1, 5, 10, 20, 60]
        self.volatility_periods = [5, 10, 20, 60]
        self.volume_periods = [5, 10, 20, 60]
        self.ema_periods = [9, 20, 50]
        
        # Features mínimas requeridas
        self.required_candle_columns = ['open', 'high', 'low', 'close', 'volume']
        self.required_micro_columns = ['buy_volume', 'sell_volume', 'volume_imbalance']
        
        # Cache para otimização
        self._cache = {}
        
    def calculate_all(self, candles: pd.DataFrame, 
                     microstructure: pd.DataFrame,
                     indicators: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calcula todas as features ML
        
        Args:
            candles: DataFrame com OHLCV
            microstructure: DataFrame com métricas microestruturais
            indicators: DataFrame com indicadores técnicos (opcional)
            
        Returns:
            DataFrame com todas as features calculadas
        """
        self.logger.info("Iniciando cálculo de features V3...")
        
        # Validar dados de entrada
        if not self._validate_input_data(candles, microstructure):
            self.logger.error("Dados de entrada inválidos")
            return pd.DataFrame()
        
        # Alinhar índices temporais
        candles, microstructure = self._align_dataframes(candles, microstructure)
        
        # Container para todas as features
        all_features = pd.DataFrame(index=candles.index)
        
        # 1. Features de Momentum
        momentum_features = self._calculate_momentum_features(candles, microstructure)
        all_features = pd.concat([all_features, momentum_features], axis=1)
        
        # 2. Features de Volatilidade
        volatility_features = self._calculate_volatility_features(candles, microstructure)
        all_features = pd.concat([all_features, volatility_features], axis=1)
        
        # 3. Features de Volume e Microestrutura
        volume_features = self._calculate_volume_features(candles, microstructure)
        all_features = pd.concat([all_features, volume_features], axis=1)
        
        # 4. Features de Order Flow
        orderflow_features = self._calculate_orderflow_features(microstructure)
        all_features = pd.concat([all_features, orderflow_features], axis=1)
        
        # 5. Features Técnicas Enhanced
        technical_features = self._calculate_technical_features(candles, indicators)
        all_features = pd.concat([all_features, technical_features], axis=1)
        
        # 6. Features de Regime de Mercado
        regime_features = self._calculate_regime_features(candles, all_features)
        all_features = pd.concat([all_features, regime_features], axis=1)
        
        # 7. Features de Interação
        interaction_features = self._calculate_interaction_features(all_features)
        all_features = pd.concat([all_features, interaction_features], axis=1)
        
        # Limpeza final
        all_features = self._clean_features(all_features)
        
        self.logger.info(f"Features V3 calculadas: {all_features.shape}")
        self.logger.info(f"NaN rate: {all_features.isna().sum().sum() / (all_features.shape[0] * all_features.shape[1]):.2%}")
        
        return all_features
    
    def _validate_input_data(self, candles: pd.DataFrame, 
                           microstructure: pd.DataFrame) -> bool:
        """Valida dados de entrada"""
        
        # Verificar se DataFrames não estão vazios
        if candles.empty or microstructure.empty:
            self.logger.error("DataFrames vazios fornecidos")
            return False
        
        # Verificar colunas requeridas
        missing_candle_cols = set(self.required_candle_columns) - set(candles.columns)
        if missing_candle_cols:
            self.logger.error(f"Colunas faltando em candles: {missing_candle_cols}")
            return False
        
        missing_micro_cols = set(self.required_micro_columns) - set(microstructure.columns)
        if missing_micro_cols:
            self.logger.error(f"Colunas faltando em microstructure: {missing_micro_cols}")
            return False
        
        # Verificar índices datetime
        if not isinstance(candles.index, pd.DatetimeIndex):
            self.logger.error("Candles deve ter DatetimeIndex")
            return False
        
        if not isinstance(microstructure.index, pd.DatetimeIndex):
            self.logger.error("Microstructure deve ter DatetimeIndex")
            return False
        
        return True
    
    def _align_dataframes(self, candles: pd.DataFrame, 
                         microstructure: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Alinha DataFrames pelo índice temporal"""
        
        # Encontrar índice comum
        common_index = candles.index.intersection(microstructure.index)
        
        if len(common_index) == 0:
            self.logger.warning("Nenhum índice comum encontrado, usando união")
            common_index = candles.index.union(microstructure.index)
            
            # Reindexar e preencher
            candles = candles.reindex(common_index).ffill()
            microstructure = microstructure.reindex(common_index).ffill()
        else:
            # Usar apenas índices comuns
            candles = candles.loc[common_index]
            microstructure = microstructure.loc[common_index]
        
        return candles, microstructure
    
    def _calculate_momentum_features(self, candles: pd.DataFrame, 
                                   microstructure: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de momentum avançadas"""
        
        features = pd.DataFrame(index=candles.index)
        close = candles['close']
        volume = candles['volume']
        
        # Momentum básico
        for period in self.momentum_periods:
            # Price momentum
            features[f'momentum_{period}'] = close - close.shift(period)
            features[f'momentum_pct_{period}'] = close.pct_change(period) * 100
            
            # Volume-weighted momentum
            if 'buy_volume' in microstructure.columns:
                buy_vol = microstructure['buy_volume']
                sell_vol = microstructure['sell_volume']
                
                # Volume-weighted price momentum
                vwap_buy = (close * buy_vol).rolling(period).sum() / buy_vol.rolling(period).sum()
                vwap_sell = (close * sell_vol).rolling(period).sum() / sell_vol.rolling(period).sum()
                
                features[f'momentum_vw_buy_{period}'] = close - vwap_buy
                features[f'momentum_vw_sell_{period}'] = close - vwap_sell
                features[f'momentum_vw_diff_{period}'] = vwap_buy - vwap_sell
        
        # Rate of change (ROC)
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = ((close - close.shift(period)) / close.shift(period)) * 100
        
        # Momentum acceleration
        for period in [5, 10]:
            momentum = close - close.shift(period)
            features[f'momentum_accel_{period}'] = momentum - momentum.shift(period)
        
        return features
    
    def _calculate_volatility_features(self, candles: pd.DataFrame,
                                     microstructure: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de volatilidade com ajustes microestruturais"""
        
        features = pd.DataFrame(index=candles.index)
        
        # Volatilidade básica (Parkinson)
        for period in self.volatility_periods:
            # High-Low volatility (Parkinson)
            hl_ratio = np.log(candles['high'] / candles['low'])
            features[f'volatility_hl_{period}'] = hl_ratio.rolling(period).std() * np.sqrt(252)
            
            # Close-to-close volatility
            returns = candles['close'].pct_change()
            features[f'volatility_cc_{period}'] = returns.rolling(period).std() * np.sqrt(252)
            
            # Garman-Klass volatility
            cc = np.log(candles['close'] / candles['close'].shift(1))
            ho = np.log(candles['high'] / candles['open'])
            lo = np.log(candles['low'] / candles['open'])
            co = np.log(candles['close'] / candles['open'])
            
            gk = 0.5 * (ho - lo)**2 - (2 * np.log(2) - 1) * co**2
            features[f'volatility_gk_{period}'] = np.sqrt(gk.rolling(period).mean() * 252)
        
        # Volatilidade ajustada por volume imbalance
        if 'volume_imbalance' in microstructure.columns:
            vol_imb = microstructure['volume_imbalance'].abs()
            for period in [5, 10, 20]:
                # Volatilidade ponderada por imbalance
                weight = vol_imb / (vol_imb.rolling(period).mean() + 1e-10)
                weighted_vol = returns.rolling(period).std() * weight
                features[f'volatility_imb_adj_{period}'] = weighted_vol * np.sqrt(252)
        
        # Volatilidade assimétrica
        for period in [10, 20]:
            pos_returns = returns.where(returns > 0, 0)
            neg_returns = returns.where(returns < 0, 0)
            
            features[f'volatility_up_{period}'] = pos_returns.rolling(period).std() * np.sqrt(252)
            features[f'volatility_down_{period}'] = neg_returns.rolling(period).std() * np.sqrt(252)
            features[f'volatility_ratio_{period}'] = (
                features[f'volatility_up_{period}'] / 
                (features[f'volatility_down_{period}'] + 1e-10)
            )
        
        return features
    
    def _calculate_volume_features(self, candles: pd.DataFrame,
                                 microstructure: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de volume e microestrutura"""
        
        features = pd.DataFrame(index=candles.index)
        volume = candles['volume']
        
        # Volume básico
        for period in self.volume_periods:
            # Volume médio e ratio
            vol_ma = volume.rolling(period).mean()
            features[f'volume_ma_{period}'] = vol_ma
            features[f'volume_ratio_{period}'] = volume / (vol_ma + 1e-10)
            
            # Volume momentum
            features[f'volume_momentum_{period}'] = volume - volume.shift(period)
        
        # Microestrutura features
        if 'buy_volume' in microstructure.columns:
            buy_vol = microstructure['buy_volume']
            sell_vol = microstructure['sell_volume']
            
            # Buy/Sell pressure
            total_vol = buy_vol + sell_vol + 1e-10
            features['buy_pressure'] = buy_vol / total_vol
            features['sell_pressure'] = sell_vol / total_vol
            
            # Moving averages of pressure
            for period in [5, 10, 20]:
                features[f'buy_pressure_ma_{period}'] = features['buy_pressure'].rolling(period).mean()
                features[f'sell_pressure_ma_{period}'] = features['sell_pressure'].rolling(period).mean()
                
                # Pressure momentum
                features[f'buy_pressure_momentum_{period}'] = (
                    features['buy_pressure'] - features[f'buy_pressure_ma_{period}']
                )
        
        # Trade size features
        if 'avg_trade_size' in microstructure.columns:
            avg_size = microstructure['avg_trade_size']
            for period in [5, 10, 20]:
                features[f'avg_trade_size_ma_{period}'] = avg_size.rolling(period).mean()
                features[f'avg_trade_size_ratio_{period}'] = (
                    avg_size / (features[f'avg_trade_size_ma_{period}'] + 1e-10)
                )
        
        # Volume concentration
        for period in [10, 20]:
            # Herfindahl index proxy
            vol_sum = volume.rolling(period).sum()
            vol_sum_sq = (volume**2).rolling(period).sum()
            features[f'volume_concentration_{period}'] = vol_sum_sq / (vol_sum**2 + 1e-10)
        
        return features
    
    def _calculate_orderflow_features(self, microstructure: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de order flow"""
        
        features = pd.DataFrame(index=microstructure.index)
        
        if 'volume_imbalance' in microstructure.columns:
            vol_imb = microstructure['volume_imbalance']
            
            # Imbalance básico
            features['volume_imbalance'] = vol_imb
            features['volume_imbalance_abs'] = vol_imb.abs()
            
            # Imbalance cumulativo
            features['volume_imbalance_cum'] = vol_imb.cumsum()
            
            # Moving averages
            for period in [5, 10, 20]:
                features[f'volume_imbalance_ma_{period}'] = vol_imb.rolling(period).mean()
                features[f'volume_imbalance_std_{period}'] = vol_imb.rolling(period).std()
                
                # Z-score do imbalance
                mean = features[f'volume_imbalance_ma_{period}']
                std = features[f'volume_imbalance_std_{period}']
                features[f'volume_imbalance_zscore_{period}'] = (vol_imb - mean) / (std + 1e-10)
        
        if 'trade_imbalance' in microstructure.columns:
            trade_imb = microstructure['trade_imbalance']
            
            # Trade imbalance
            features['trade_imbalance'] = trade_imb
            features['trade_imbalance_abs'] = trade_imb.abs()
            
            # Moving averages
            for period in [5, 10, 20]:
                features[f'trade_imbalance_ma_{period}'] = trade_imb.rolling(period).mean()
        
        # Order flow persistence
        if 'buy_trades' in microstructure.columns and 'sell_trades' in microstructure.columns:
            buy_trades = microstructure['buy_trades']
            sell_trades = microstructure['sell_trades']
            
            # Runs de compra/venda
            buy_dominant = (buy_trades > sell_trades).astype(int)
            features['buy_run_length'] = self._calculate_run_length(buy_dominant)
            features['sell_run_length'] = self._calculate_run_length(1 - buy_dominant)
        
        return features
    
    def _calculate_technical_features(self, candles: pd.DataFrame,
                                    indicators: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Calcula features técnicas enhanced"""
        
        features = pd.DataFrame(index=candles.index)
        
        # Se indicadores foram fornecidos, usar
        if indicators is not None and not indicators.empty:
            # Alinhar índices
            indicators = indicators.reindex(candles.index, method='ffill')
            
            # Adicionar indicadores como features
            for col in indicators.columns:
                features[f'ind_{col}'] = indicators[col]
        
        # Calcular indicadores básicos se não fornecidos
        close = candles['close']
        high = candles['high']
        low = candles['low']
        
        # Bollinger Bands
        for period in [20]:
            sma = close.rolling(period).mean()
            std = close.rolling(period).std()
            
            features[f'bb_upper_{period}'] = sma + 2 * std
            features[f'bb_lower_{period}'] = sma - 2 * std
            features[f'bb_width_{period}'] = 4 * std / sma
            features[f'bb_position_{period}'] = (close - features[f'bb_lower_{period}']) / (4 * std + 1e-10)
        
        # ATR (Average True Range)
        for period in [14]:
            high_low = high - low
            high_close = (high - close.shift(1)).abs()
            low_close = (low - close.shift(1)).abs()
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features[f'atr_{period}'] = true_range.rolling(period).mean()
            features[f'atr_ratio_{period}'] = features[f'atr_{period}'] / close
        
        # Price position
        for period in [20, 50]:
            highest = high.rolling(period).max()
            lowest = low.rolling(period).min()
            
            features[f'price_position_{period}'] = (close - lowest) / (highest - lowest + 1e-10)
        
        return features
    
    def _calculate_regime_features(self, candles: pd.DataFrame,
                                 existing_features: pd.DataFrame) -> pd.DataFrame:
        """Calcula features específicas de regime de mercado"""
        
        features = pd.DataFrame(index=candles.index)
        close = candles['close']
        
        # EMAs para regime
        for period in self.ema_periods:
            features[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
        
        # ADX components (simplificado)
        period = 14
        high = candles['high']
        low = candles['low']
        
        # +DM e -DM
        high_diff = high.diff()
        low_diff = -low.diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        # Smoothed values
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di
        features['di_diff'] = plus_di - minus_di
        
        # DX e ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        features['adx'] = dx.rolling(period).mean()
        
        # Regime classification features
        features['trend_strength'] = features['adx'] / 100
        features['trend_direction'] = np.sign(features['di_diff'])
        
        # EMA alignment
        if all(f'ema_{p}' in features.columns for p in [9, 20, 50]):
            features['ema_alignment_score'] = (
                (features['ema_9'] > features['ema_20']).astype(int) +
                (features['ema_20'] > features['ema_50']).astype(int) +
                (close > features['ema_9']).astype(int)
            ) / 3.0
        
        return features
    
    def _calculate_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de interação entre features existentes"""
        
        interaction_features = pd.DataFrame(index=features.index)
        
        # Interações momentum-volatilidade
        if 'momentum_5' in features.columns and 'volatility_cc_5' in features.columns:
            interaction_features['momentum_vol_ratio_5'] = (
                features['momentum_5'] / (features['volatility_cc_5'] + 1e-10)
            )
        
        # Interações volume-volatilidade
        if 'volume_ratio_5' in features.columns and 'volatility_cc_5' in features.columns:
            interaction_features['volume_volatility_5'] = (
                features['volume_ratio_5'] * features['volatility_cc_5']
            )
        
        # Interações buy pressure-momentum
        if 'buy_pressure' in features.columns and 'momentum_pct_5' in features.columns:
            interaction_features['buy_pressure_momentum'] = (
                features['buy_pressure'] * features['momentum_pct_5']
            )
        
        # Interações regime-volume
        if 'trend_strength' in features.columns and 'volume_imbalance' in features.columns:
            interaction_features['trend_volume_interaction'] = (
                features['trend_strength'] * features['volume_imbalance']
            )
        
        return interaction_features
    
    def _calculate_run_length(self, series: pd.Series) -> pd.Series:
        """Calcula comprimento de runs consecutivos"""
        
        # Identificar mudanças
        changes = series.diff().fillna(1) != 0
        
        # Grupos de runs
        groups = changes.cumsum()
        
        # Contar tamanho de cada run
        run_lengths = series.groupby(groups).cumcount() + 1
        
        return run_lengths
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Limpa e finaliza features"""
        
        # Remover colunas com todos NaN
        features = features.dropna(axis=1, how='all')
        
        # Remover colunas com variância zero
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        non_zero_var = features[numeric_cols].var() > 1e-10
        features = features[features.columns[features.columns.isin(non_zero_var[non_zero_var].index)]]
        
        # Substituir infinitos
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill para preencher NaN iniciais
        features = features.ffill().fillna(0)
        
        # Adicionar prefixo para identificar features V3
        features.columns = [f'v3_{col}' for col in features.columns]
        
        return features
    
    def get_feature_importance_hints(self) -> Dict[str, List[str]]:
        """Retorna dicas sobre importância esperada das features"""
        
        return {
            'high_importance': [
                'v3_momentum_5', 'v3_momentum_10',
                'v3_volume_imbalance', 'v3_buy_pressure',
                'v3_volatility_cc_10', 'v3_trend_strength'
            ],
            'medium_importance': [
                'v3_volume_ratio_5', 'v3_momentum_vw_diff_5',
                'v3_bb_position_20', 'v3_adx'
            ],
            'interaction_features': [
                'v3_momentum_vol_ratio_5', 'v3_buy_pressure_momentum',
                'v3_trend_volume_interaction'
            ]
        }


def main():
    """Teste do MLFeaturesV3"""
    
    print("="*60)
    print("TESTE DO ML FEATURES V3")
    print("="*60)
    
    # Criar dados de exemplo
    dates = pd.date_range('2025-01-27 09:00', '2025-01-27 17:00', freq='1min')
    
    # Candles simulados
    candles = pd.DataFrame({
        'open': 5900 + np.random.randn(len(dates)).cumsum(),
        'high': 5905 + np.random.randn(len(dates)).cumsum(),
        'low': 5895 + np.random.randn(len(dates)).cumsum(),
        'close': 5900 + np.random.randn(len(dates)).cumsum(),
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Ajustar high/low
    candles['high'] = candles[['open', 'close', 'high']].max(axis=1)
    candles['low'] = candles[['open', 'close', 'low']].min(axis=1)
    
    # Microestrutura simulada
    microstructure = pd.DataFrame({
        'buy_volume': candles['volume'] * np.random.uniform(0.3, 0.7, len(dates)),
        'sell_volume': candles['volume'] * np.random.uniform(0.3, 0.7, len(dates)),
        'volume_imbalance': np.random.randn(len(dates)) * 1000000,
        'trade_imbalance': np.random.randn(len(dates)) * 100,
        'buy_pressure': np.random.uniform(0.3, 0.7, len(dates)),
        'avg_trade_size': np.random.randint(10000, 50000, len(dates)),
        'buy_trades': np.random.randint(50, 200, len(dates)),
        'sell_trades': np.random.randint(50, 200, len(dates))
    }, index=dates)
    
    # Calcular features
    calculator = MLFeaturesV3()
    features = calculator.calculate_all(candles, microstructure)
    
    print(f"\nFeatures calculadas: {features.shape}")
    print(f"Colunas: {features.shape[1]}")
    print(f"NaN rate: {features.isna().sum().sum() / (features.shape[0] * features.shape[1]):.2%}")
    
    # Mostrar algumas features
    print("\nPrimeiras 10 features:")
    for col in features.columns[:10]:
        print(f"  - {col}")
    
    # Estatísticas
    print("\nEstatísticas das features:")
    print(features.describe().iloc[:, :5])
    
    # Hints de importância
    hints = calculator.get_feature_importance_hints()
    print("\nFeatures de alta importância esperada:")
    for feat in hints['high_importance']:
        if feat in features.columns:
            print(f"  - {feat}: presente")
        else:
            print(f"  - {feat}: AUSENTE")


if __name__ == "__main__":
    main()