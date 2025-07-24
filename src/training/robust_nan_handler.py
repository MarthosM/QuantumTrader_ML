"""
Sistema Robusto de Tratamento de NaN para Features
Resolve valores ausentes de forma inteligente sem introduzir viés
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

class NaNHandlingStrategy(Enum):
    """Estratégias para tratar NaN de acordo com o tipo de feature"""
    FORWARD_FILL = "ffill"        # Para indicadores que podem usar último valor
    BACKWARD_FILL = "bfill"       # Para features temporais que permitem
    LINEAR_INTERPOLATION = "linear"  # Para series temporais suaves
    DROP_ROWS = "drop"           # Remove linhas com NaN críticos
    CALCULATE_PROPER = "calculate"  # Recalcula com janela adequada
    MARK_INVALID = "invalid"     # Marca como inválido para remoção posterior

class RobustNaNHandler:
    """
    Handler inteligente para valores NaN em features financeiras
    Evita introduzir viés mantendo a integridade dos dados
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Mapeamento de features para estratégias apropriadas
        self.feature_strategies = {
            # Indicadores que podem usar forward fill por pouco tempo
            'rsi': NaNHandlingStrategy.CALCULATE_PROPER,
            'rsi_9': NaNHandlingStrategy.CALCULATE_PROPER,
            'rsi_25': NaNHandlingStrategy.CALCULATE_PROPER,
            'macd': NaNHandlingStrategy.CALCULATE_PROPER,
            'macd_signal': NaNHandlingStrategy.CALCULATE_PROPER,
            'macd_hist': NaNHandlingStrategy.CALCULATE_PROPER,
            
            # Bollinger Bands - recalcular adequadamente
            'bb_upper_20': NaNHandlingStrategy.CALCULATE_PROPER,
            'bb_middle_20': NaNHandlingStrategy.CALCULATE_PROPER,
            'bb_lower_20': NaNHandlingStrategy.CALCULATE_PROPER,
            'bb_width_20': NaNHandlingStrategy.CALCULATE_PROPER,
            'bb_position_20': NaNHandlingStrategy.CALCULATE_PROPER,
            'bb_upper_50': NaNHandlingStrategy.CALCULATE_PROPER,
            'bb_middle_50': NaNHandlingStrategy.CALCULATE_PROPER,
            'bb_lower_50': NaNHandlingStrategy.CALCULATE_PROPER,
            'bb_width_50': NaNHandlingStrategy.CALCULATE_PROPER,
            'bb_position_50': NaNHandlingStrategy.CALCULATE_PROPER,
            
            # ATR - valores de volatilidade
            'atr': NaNHandlingStrategy.CALCULATE_PROPER,
            'atr_20': NaNHandlingStrategy.CALCULATE_PROPER,
            'adx': NaNHandlingStrategy.CALCULATE_PROPER,
            
            # Momentum - podem usar interpolação linear
            'momentum_1': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            'momentum_3': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            'momentum_5': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            'momentum_10': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            'momentum_15': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            'momentum_20': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            
            # Percentual momentum
            'momentum_pct_1': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            'momentum_pct_3': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            'momentum_pct_5': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            'momentum_pct_10': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            'momentum_pct_15': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            'momentum_pct_20': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            
            # ROC e Returns
            'roc_5': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            'roc_10': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            'roc_20': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            'return_5': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            'return_10': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            'return_20': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            'return_50': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            
            # Volume features
            'volume_momentum_5': NaNHandlingStrategy.CALCULATE_PROPER,
            'volume_momentum_10': NaNHandlingStrategy.CALCULATE_PROPER,
            'volume_momentum_20': NaNHandlingStrategy.CALCULATE_PROPER,
            'volume_sma_5': NaNHandlingStrategy.CALCULATE_PROPER,
            'volume_ratio_5': NaNHandlingStrategy.CALCULATE_PROPER,
            'volume_sma_10': NaNHandlingStrategy.CALCULATE_PROPER,
            'volume_ratio_10': NaNHandlingStrategy.CALCULATE_PROPER,
            'volume_sma_20': NaNHandlingStrategy.CALCULATE_PROPER,
            'volume_ratio_20': NaNHandlingStrategy.CALCULATE_PROPER,
            
            # Volatilidade
            'volatility_5': NaNHandlingStrategy.CALCULATE_PROPER,
            'volatility_10': NaNHandlingStrategy.CALCULATE_PROPER,
            'volatility_20': NaNHandlingStrategy.CALCULATE_PROPER,
            'volatility_50': NaNHandlingStrategy.CALCULATE_PROPER,
            'volatility_ratio': NaNHandlingStrategy.CALCULATE_PROPER,
            
            # Volatilidade avançada
            'parkinson_vol_10': NaNHandlingStrategy.CALCULATE_PROPER,
            'parkinson_vol_20': NaNHandlingStrategy.CALCULATE_PROPER,
            'gk_vol_10': NaNHandlingStrategy.CALCULATE_PROPER,
            'gk_vol_20': NaNHandlingStrategy.CALCULATE_PROPER,
            
            # Ranges
            'true_range': NaNHandlingStrategy.CALCULATE_PROPER,
            'high_low_range_5': NaNHandlingStrategy.CALCULATE_PROPER,
            'high_low_range_10': NaNHandlingStrategy.CALCULATE_PROPER,
            'high_low_range_20': NaNHandlingStrategy.CALCULATE_PROPER,
            
            # Features especiais
            'price_acceleration': NaNHandlingStrategy.LINEAR_INTERPOLATION,
            'price_volume_trend': NaNHandlingStrategy.CALCULATE_PROPER,
            'vwap': NaNHandlingStrategy.CALCULATE_PROPER,
            
            # Cross changes (podem ser 0 em início)
            'macd_cross_change': NaNHandlingStrategy.FORWARD_FILL,
            
            # Lags - forward fill apropriado
            'rsi_lag_1': NaNHandlingStrategy.FORWARD_FILL,
            'rsi_lag_5': NaNHandlingStrategy.FORWARD_FILL,
            'rsi_lag_10': NaNHandlingStrategy.FORWARD_FILL,
            'rsi_lag_20': NaNHandlingStrategy.FORWARD_FILL,
            'macd_lag_1': NaNHandlingStrategy.FORWARD_FILL,
            'macd_lag_5': NaNHandlingStrategy.FORWARD_FILL,
            'macd_lag_10': NaNHandlingStrategy.FORWARD_FILL,
            'macd_lag_20': NaNHandlingStrategy.FORWARD_FILL,
            'volatility_20_lag_1': NaNHandlingStrategy.FORWARD_FILL,
            'volatility_20_lag_5': NaNHandlingStrategy.FORWARD_FILL,
            'volatility_20_lag_10': NaNHandlingStrategy.FORWARD_FILL,
            'volatility_20_lag_20': NaNHandlingStrategy.FORWARD_FILL,
            'momentum_5_lag_1': NaNHandlingStrategy.FORWARD_FILL,
            'momentum_5_lag_5': NaNHandlingStrategy.FORWARD_FILL,
            'momentum_5_lag_10': NaNHandlingStrategy.FORWARD_FILL,
            'momentum_5_lag_20': NaNHandlingStrategy.FORWARD_FILL,
        }
    
    def handle_nans(self, df: pd.DataFrame, 
                   raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Trata NaNs de forma inteligente baseado na estratégia por feature
        
        Args:
            df: DataFrame com features que podem ter NaN
            raw_data: Dados originais OHLCV para recálculos
            
        Returns:
            Tuple[DataFrame limpo, Dict com estatísticas de NaN]
        """
        self.logger.info("Iniciando tratamento inteligente de NaN")
        
        nan_stats_before = {}
        nan_stats_after = {}
        
        # Estatísticas iniciais
        for col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                nan_stats_before[col] = nan_count
        
        self.logger.info(f"Features com NaN: {len(nan_stats_before)}")
        
        # Tratar cada feature de acordo com sua estratégia
        for col in df.columns:
            if col in nan_stats_before:
                strategy = self.feature_strategies.get(col, NaNHandlingStrategy.DROP_ROWS)
                
                self.logger.debug(f"Tratando {col} com estratégia {strategy.value}")
                
                if strategy == NaNHandlingStrategy.CALCULATE_PROPER:
                    df[col] = self._recalculate_feature(col, raw_data, df)
                    
                elif strategy == NaNHandlingStrategy.LINEAR_INTERPOLATION:
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
                    
                elif strategy == NaNHandlingStrategy.FORWARD_FILL:
                    df[col] = df[col].ffill()
                    
                elif strategy == NaNHandlingStrategy.BACKWARD_FILL:
                    df[col] = df[col].bfill()
                    
                # Ainda restam NaNs? Remove as linhas
                if df[col].isna().any():
                    initial_rows = len(df)
                    df = df.dropna(subset=[col])
                    dropped = initial_rows - len(df)
                    if dropped > 0:
                        self.logger.warning(f"Removidas {dropped} linhas por NaN em {col}")
        
        # Estatísticas finais
        for col in df.columns:
            nan_count = df[col].isna().sum()
            nan_stats_after[col] = nan_count
        
        # Remover colunas que ainda têm muitos NaN (>50% dos dados)
        high_nan_cols = []
        for col in df.columns:
            nan_pct = df[col].isna().sum() / len(df)
            if nan_pct > 0.5:
                high_nan_cols.append(col)
                self.logger.warning(f"Removendo feature {col} - {nan_pct:.1%} NaN")
        
        df = df.drop(columns=high_nan_cols)
        
        # Remover linhas com qualquer NaN restante
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        if initial_rows != final_rows:
            self.logger.info(f"Dataset final: {final_rows}/{initial_rows} linhas ({final_rows/initial_rows:.1%})")
        
        return df, {
            'before': nan_stats_before,
            'after': nan_stats_after,
            'removed_features': high_nan_cols,
            'final_rows': final_rows,
            'initial_rows': initial_rows
        }
    
    def _recalculate_feature(self, feature_name: str, 
                           raw_data: pd.DataFrame, 
                           current_df: pd.DataFrame) -> pd.Series:
        """Recalcula feature específica com parâmetros corretos"""
        
        try:
            if feature_name.startswith('rsi'):
                return self._calculate_rsi_proper(raw_data, feature_name)
            elif feature_name.startswith('bb_'):
                return self._calculate_bb_proper(raw_data, feature_name)
            elif feature_name.startswith('atr'):
                return self._calculate_atr_proper(raw_data, feature_name)
            elif feature_name == 'adx':
                return self._calculate_adx_proper(raw_data)
            elif 'volume' in feature_name:
                return self._calculate_volume_proper(raw_data, feature_name)
            elif 'volatility' in feature_name:
                return self._calculate_volatility_proper(raw_data, feature_name)
            elif feature_name in ['vwap']:
                return self._calculate_vwap_proper(raw_data)
            else:
                # Para features não mapeadas, usar valor atual
                return current_df[feature_name]
                
        except Exception as e:
            self.logger.warning(f"Erro recalculando {feature_name}: {e}")
            return current_df[feature_name]
    
    def _calculate_rsi_proper(self, data: pd.DataFrame, feature_name: str) -> pd.Series:
        """Cálculo robusto do RSI"""
        # Extrair período do nome
        if feature_name == 'rsi':
            period = 14
        else:
            period = int(feature_name.split('_')[1])
        
        delta = data['close'].astype(float).diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Usar EWM para suavização mais robusta
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bb_proper(self, data: pd.DataFrame, feature_name: str) -> pd.Series:
        """Cálculo robusto de Bollinger Bands"""
        parts = feature_name.split('_')
        period = int(parts[2])  # bb_upper_20 -> 20
        
        sma = data['close'].rolling(window=period, min_periods=period//2).mean()
        std = data['close'].rolling(window=period, min_periods=period//2).std()
        
        if 'upper' in feature_name:
            return sma + (2 * std)
        elif 'lower' in feature_name:
            return sma - (2 * std)
        elif 'middle' in feature_name:
            return sma
        elif 'width' in feature_name:
            return (sma + 2*std) - (sma - 2*std)
        elif 'position' in feature_name:
            upper = sma + (2 * std)
            lower = sma - (2 * std)
            width = upper - lower
            return (data['close'] - lower) / width
        
        return sma
    
    def _calculate_atr_proper(self, data: pd.DataFrame, feature_name: str) -> pd.Series:
        """Cálculo robusto do ATR"""
        period = 14 if feature_name == 'atr' else int(feature_name.split('_')[1])
        
        high_low = data['high'] - data['low']
        high_close = pd.Series(np.abs(data['high'] - data['close'].shift()), index=data.index)
        low_close = pd.Series(np.abs(data['low'] - data['close'].shift()), index=data.index)
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period//2).mean()
        
        return atr
    
    def _calculate_adx_proper(self, data: pd.DataFrame) -> pd.Series:
        """Cálculo simplificado mas robusto do ADX"""
        # Implementação simplificada baseada em movimento direcional
        high_diff = data['high'].astype(float).diff()
        low_diff = data['low'].astype(float).diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        plus_dm_series = pd.Series(plus_dm, index=data.index)
        minus_dm_series = pd.Series(minus_dm, index=data.index)
        
        tr = self._calculate_atr_proper(data, 'atr')
        
        plus_di = 100 * (plus_dm_series.rolling(window=14).mean() / tr)
        minus_di = 100 * (minus_dm_series.rolling(window=14).mean() / tr)
        
        # Calcular DX mantendo como Series - evitar operações numpy diretas
        dx_numerator = (plus_di - minus_di).abs()
        dx_denominator = plus_di + minus_di
        
        # Evitar divisão por zero
        dx_denominator = dx_denominator.replace(0, np.nan)
        dx = 100 * (dx_numerator / dx_denominator)
        
        # ADX como média móvel do DX
        adx = dx.rolling(window=14, min_periods=7).mean()
        
        return adx
    
    def _calculate_volume_proper(self, data: pd.DataFrame, feature_name: str) -> pd.Series:
        """Cálculo robusto de features de volume"""
        if 'momentum' in feature_name:
            period = int(feature_name.split('_')[2])
            return data['volume'].pct_change(periods=period)
        elif 'sma' in feature_name:
            period = int(feature_name.split('_')[2])
            return data['volume'].rolling(window=period, min_periods=period//2).mean()
        elif 'ratio' in feature_name:
            period = int(feature_name.split('_')[2])
            sma = data['volume'].rolling(window=period, min_periods=period//2).mean()
            return data['volume'] / sma
        
        return data['volume']
    
    def _calculate_volatility_proper(self, data: pd.DataFrame, feature_name: str) -> pd.Series:
        """Cálculo robusto de volatilidade"""
        if 'parkinson' in feature_name:
            period = int(feature_name.split('_')[2])
            ln_hl = (data['high'] / data['low']).apply(np.log)  # Manter como Series
            return ((ln_hl ** 2).rolling(window=period).mean()).apply(np.sqrt)
        elif 'gk' in feature_name:  # Garman-Klass
            period = int(feature_name.split('_')[2])
            ln_ho = (data['high'] / data['open']).apply(np.log)  # Manter como Series
            ln_lo = (data['low'] / data['open']).apply(np.log)   # Manter como Series
            ln_co = (data['close'] / data['open']).apply(np.log) # Manter como Series
            gk = ln_ho * (ln_ho - ln_co) + ln_lo * (ln_lo - ln_co)
            return (gk.rolling(window=period).mean()).apply(np.sqrt)
        else:
            # Volatilidade padrão
            period = int(feature_name.split('_')[1])
            returns = data['close'].pct_change()
            return returns.rolling(window=period, min_periods=period//2).std()
    
    def _calculate_vwap_proper(self, data: pd.DataFrame) -> pd.Series:
        """VWAP robusto"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        return (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
    
    def validate_nan_handling(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida a qualidade do tratamento de NaN
        
        Returns:
            Dict com métricas de qualidade
        """
        total_features = len(df.columns)
        features_with_nan = df.isnull().sum()
        features_with_nan_count = (features_with_nan > 0).sum()
        
        # Calcular distribuição de NaN por feature
        nan_distribution = {}
        for col in df.columns:
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                nan_distribution[col] = {
                    'count': nan_count,
                    'percentage': (nan_count / len(df)) * 100
                }
        
        # Qualidade geral
        quality_score = 1 - (features_with_nan_count / total_features)
        
        validation_result = {
            'total_features': total_features,
            'features_with_nan': features_with_nan_count,
            'quality_score': quality_score,
            'nan_distribution': nan_distribution,
            'data_rows': len(df),
            'is_clean': features_with_nan_count == 0,
            'recommendations': self._generate_recommendations(nan_distribution, quality_score)
        }
        
        self.logger.info(f"Validação NaN: {features_with_nan_count}/{total_features} features com NaN")
        self.logger.info(f"Score de qualidade: {quality_score:.3f}")
        
        return validation_result
    
    def _generate_recommendations(self, nan_distribution: Dict, 
                                quality_score: float) -> List[str]:
        """Gera recomendações baseadas na análise de NaN"""
        recommendations = []
        
        if quality_score < 0.9:
            recommendations.append("Sistema possui muitas features com NaN - revisar pipeline de features")
        
        if quality_score < 0.7:
            recommendations.append("CRÍTICO: Qualidade muito baixa - verificar dados de origem")
        
        # Verificar features problemáticas
        high_nan_features = [
            feat for feat, info in nan_distribution.items() 
            if info['percentage'] > 10
        ]
        
        if high_nan_features:
            recommendations.append(f"Features com >10% NaN: {', '.join(high_nan_features[:5])}")
        
        if not nan_distribution:
            recommendations.append("✅ Dataset limpo - pronto para treinamento ML")
        
        return recommendations
    
    def create_nan_handling_report(self, stats: Dict, 
                                 validation: Dict) -> str:
        """Cria relatório detalhado do tratamento de NaN"""
        
        report_lines = [
            "=" * 60,
            "📊 RELATÓRIO DE TRATAMENTO DE NaN",
            "=" * 60,
            "",
            f"📈 DATASET INICIAL:",
            f"  • Linhas: {stats['initial_rows']:,}",
            f"  • Features com NaN: {len(stats['before'])}",
            "",
            f"🔧 TRATAMENTO APLICADO:",
        ]
        
        # Estratégias utilizadas
        strategy_counts = {}
        for feature in stats['before'].keys():
            strategy = self.feature_strategies.get(feature, NaNHandlingStrategy.DROP_ROWS)
            strategy_name = strategy.value
            strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
        
        for strategy, count in strategy_counts.items():
            report_lines.append(f"  • {strategy}: {count} features")
        
        report_lines.extend([
            "",
            f"📉 RESULTADO FINAL:",
            f"  • Linhas finais: {stats['final_rows']:,}",
            f"  • Retenção: {stats['final_rows']/stats['initial_rows']:.1%}",
            f"  • Features removidas: {len(stats['removed_features'])}",
            f"  • Score qualidade: {validation['quality_score']:.3f}",
            ""
        ])
        
        if stats['removed_features']:
            report_lines.extend([
                f"🗑️  FEATURES REMOVIDAS ({len(stats['removed_features'])}):",
                *[f"  • {feat}" for feat in stats['removed_features']],
                ""
            ])
        
        if validation['recommendations']:
            report_lines.extend([
                "💡 RECOMENDAÇÕES:",
                *[f"  • {rec}" for rec in validation['recommendations']],
                ""
            ])
        
        report_lines.extend([
            "=" * 60,
            f"Status: {'✅ LIMPO' if validation['is_clean'] else '⚠️  COM PROBLEMAS'}",
            "=" * 60
        ])
        
        return "\n".join(report_lines)
