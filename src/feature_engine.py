import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_selection import mutual_info_regression, f_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import ta library for technical indicators
import ta

from data_structure import TradingDataStructure
from technical_indicators import TechnicalIndicators
from ml_features import MLFeatures


class ProductionDataValidator:
    """Validador rigoroso para dados de produção em trading real"""
    
    def __init__(self, logger, allow_historical_data=False):
        self.logger = logger
        self.min_price_change = 0.00001  # Mudança mínima aceitável de preço
        self.max_price_change = 0.20    # 20% máximo em 1 minuto
        self.min_volume = 1              # Volume mínimo por candle
        self.allow_historical_data = allow_historical_data  # Flag para permitir dados históricos
        
    def validate_real_data(self, data: pd.DataFrame, source: str) -> bool:
        """
        Valida se dados são reais e não sintéticos
        
        CRÍTICO: Bloqueia operação se detectar dados dummy (apenas para trading em tempo real)
        """
        # Para backtest/dados históricos, pular validações rigorosas
        if self.allow_historical_data:
            self.logger.debug(f"Validação relaxada para dados históricos de {source}")
            return True
        
        try:
            # 1. Verificar se DataFrame não está vazio
            if data.empty:
                raise ValueError(f"DataFrame vazio recebido de {source}")
            
            # 2. Verificar colunas obrigatórias
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self.logger.warning(f"Colunas faltantes em {source}: {missing_cols}")
            
            # 3. Detectar padrões sintéticos
            if self._detect_synthetic_patterns(data):
                raise ValueError(
                    f"DADOS SINTÉTICOS DETECTADOS EM {source}! "
                    "OPERAÇÃO BLOQUEADA PARA SEGURANÇA"
                )
            
            # 4. Validar timestamps
            if not self._validate_timestamps(data):
                raise ValueError(
                    f"TIMESTAMPS INVÁLIDOS EM {source}! "
                    "DADOS PODEM NÃO SER REAIS"
                )
            
            # 5. Validar integridade de preços
            if not self._validate_price_integrity(data):
                raise ValueError(
                    f"INTEGRIDADE DE PREÇOS COMPROMETIDA EM {source}! "
                    "POSSÍVEL USO DE DADOS SIMULADOS"
                )
            
            # 6. Validar padrões de volume
            if not self._validate_volume_patterns(data):
                raise ValueError(
                    f"PADRÕES DE VOLUME SUSPEITOS EM {source}! "
                    "CARACTERÍSTICAS DE DADOS DUMMY DETECTADAS"
                )
            
            self.logger.info(f"Dados validados com sucesso de {source}")
            return True
            
        except ValueError as e:
            self.logger.error(f"VALIDAÇÃO FALHOU: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Erro inesperado na validação: {str(e)}")
            raise ValueError(f"Falha na validação de dados de {source}")
    
    def _detect_synthetic_patterns(self, data: pd.DataFrame) -> bool:
        """Detecta padrões típicos de dados sintéticos"""
        
        # Verificar se há uso de random seed fixo
        if 'volume' in data.columns and len(data) > 10:
            volumes = data['volume'].dropna()
            if len(volumes) > 0:
                # Calcular autocorrelação do volume
                if len(volumes) > 20:
                    autocorr = volumes.autocorr(lag=1)
                    # Dados sintéticos geralmente têm baixa autocorrelação
                    if pd.notna(autocorr) and abs(autocorr) < 0.05:
                        self.logger.warning("Baixa autocorrelação no volume detectada")
                        return True
        
        # Verificar spreads muito uniformes
        if all(col in data.columns for col in ['high', 'low', 'close']):
            spreads = (data['high'] - data['low']) / data['close']
            spread_std = spreads.std()
            
            # Spread muito uniforme = possível dado sintético
            if spread_std < 0.00001:
                self.logger.warning("Spreads muito uniformes detectados")
                return True
        
        # Verificar mudanças de preço muito uniformes
        if 'close' in data.columns and len(data) > 10:
            price_changes = data['close'].pct_change().dropna()
            if len(price_changes) > 0:
                # Verificar se mudanças seguem padrão muito regular
                change_std = price_changes.std()
                if change_std > 0:
                    # Coeficiente de variação muito baixo indica possível padrão sintético
                    cv = change_std / abs(price_changes.mean()) if price_changes.mean() != 0 else float('inf')
                    if cv < 0.1:
                        self.logger.warning("Mudanças de preço muito regulares detectadas")
                        return True
        
        return False
    
    def _validate_timestamps(self, data: pd.DataFrame) -> bool:
        """Valida se timestamps são reais e recentes"""
        
        # Verificar se índice é datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Índice não é DatetimeIndex")
            return False
        
        # Verificar se dados são muito antigos (apenas para trading em tempo real)
        if len(data) > 0 and not self.allow_historical_data:
            latest_time = data.index.max()
            current_time = pd.Timestamp.now()
            
            # Se dados são mais antigos que 1 dia, suspeito para trading real
            time_diff = current_time - latest_time
            if time_diff > pd.Timedelta(days=1):
                self.logger.warning(f"Dados muito antigos: {time_diff}")
                return False
        
        # Verificar se há gaps suspeitos (apenas para trading em tempo real)
        if len(data) > 1 and not self.allow_historical_data:
            time_diffs = pd.Series(data.index[1:]) - pd.Series(data.index[:-1])
            
            # Em trading real, espera-se alguns gaps (finais de semana, feriados)
            # Mas não deve haver periodicidade perfeita
            unique_diffs = time_diffs.value_counts()
            if len(unique_diffs) == 1:
                self.logger.warning("Timestamps com periodicidade perfeita detectada")
                return False
        
        return True
    
    def _validate_price_integrity(self, data: pd.DataFrame) -> bool:
        """Valida integridade dos preços"""
        
        price_cols = ['open', 'high', 'low', 'close']
        existing_price_cols = [col for col in price_cols if col in data.columns]
        
        if not existing_price_cols:
            return True  # Sem dados de preço para validar
        
        for col in existing_price_cols:
            prices = data[col].dropna()
            
            if len(prices) == 0:
                continue
                
            # Verificar preços negativos ou zero
            if (prices <= 0).any():
                self.logger.error(f"Preços inválidos (<=0) em {col}")
                return False
            
            # Verificar mudanças impossíveis
            if len(prices) > 1:
                price_changes = prices.pct_change().dropna()
                
                # Mudança maior que 20% em 1 período é suspeita
                if (price_changes.abs() > self.max_price_change).any():
                    self.logger.error(f"Mudanças de preço impossíveis em {col}")
                    return False
        
        # Validar relação OHLC
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # High deve ser >= todos os outros
            invalid_high = (data['high'] < data['open']) | (data['high'] < data['close']) | (data['high'] < data['low'])
            if invalid_high.any():
                self.logger.error("Relação OHLC inválida: High menor que outros preços")
                return False
            
            # Low deve ser <= todos os outros
            invalid_low = (data['low'] > data['open']) | (data['low'] > data['close']) | (data['low'] > data['high'])
            if invalid_low.any():
                self.logger.error("Relação OHLC inválida: Low maior que outros preços")
                return False
        
        return True
    
    def _validate_volume_patterns(self, data: pd.DataFrame) -> bool:
        """Valida padrões de volume real"""
        
        if 'volume' not in data.columns:
            return True
        
        volumes = data['volume'].dropna()
        
        if len(volumes) == 0:
            return True
        
        # Volume não pode ser negativo
        if (volumes < 0).any():
            self.logger.error("Volume negativo detectado")
            return False
        
        # Verificar se há muitos volumes zero
        zero_ratio = (volumes == 0).sum() / len(volumes)
        if zero_ratio > 0.5:  # Mais de 50% zeros é suspeito
            self.logger.warning(f"Muitos volumes zero: {zero_ratio:.1%}")
            return False
        
        # Verificar distribuição de volume
        if len(volumes) > 100:
            # Volume real geralmente segue distribuição log-normal
            # Verificar se há muitos valores idênticos
            value_counts = volumes.value_counts()
            most_common_ratio = value_counts.iloc[0] / len(volumes) if len(value_counts) > 0 else 0
            
            if most_common_ratio > 0.2:  # Mais de 20% com mesmo valor
                self.logger.warning("Distribuição de volume muito uniforme")
                return False
        
        return True


class SmartFillStrategy:
    """Estratégia inteligente de preenchimento para dados de trading"""
    
    def __init__(self, logger):
        self.logger = logger
        
    def fill_missing_values(self, df: pd.DataFrame, feature_type: Optional[str] = None) -> pd.DataFrame:
        """
        Preenche valores faltantes de forma inteligente baseado no tipo de dado
        
        NUNCA usa fillna(0) indiscriminadamente
        """
        df_filled = df.copy()
        
        for col in df_filled.columns:
            if df_filled[col].isna().any():
                # Determinar tipo de feature
                col_type = self._determine_feature_type(col) if feature_type is None else feature_type
                
                # Aplicar estratégia específica
                if col_type == 'price':
                    df_filled[col] = self._fill_price_data(df_filled[col])
                elif col_type == 'volume':
                    df_filled[col] = self._fill_volume_data(df_filled[col])
                elif col_type == 'indicator':
                    df_filled[col] = self._fill_indicator_data(df_filled[col], col)
                elif col_type == 'ratio':
                    df_filled[col] = self._fill_ratio_data(df_filled[col])
                else:
                    # Default: forward fill, depois backward fill
                    df_filled[col] = df_filled[col].ffill().bfill()
                
                # Log se ainda houver NaN
                remaining_nan = df_filled[col].isna().sum()
                if remaining_nan > 0:
                    self.logger.warning(
                        f"Feature '{col}' ainda tem {remaining_nan} NaN após preenchimento"
                    )
        
        return df_filled
    
    def _determine_feature_type(self, column_name: str) -> str:
        """Determina tipo de feature baseado no nome"""
        col_lower = column_name.lower()
        
        if any(x in col_lower for x in ['open', 'high', 'low', 'close', 'price']):
            return 'price'
        elif 'volume' in col_lower:
            return 'volume'
        elif any(x in col_lower for x in ['rsi', 'macd', 'ema', 'sma', 'bb_', 'atr']):
            return 'indicator'
        elif any(x in col_lower for x in ['ratio', 'pct', 'percent', 'imbalance']):
            return 'ratio'
        else:
            return 'other'
    
    def _fill_price_data(self, series: pd.Series) -> pd.Series:
        """Preenche dados de preço - NUNCA com zero"""
        # Forward fill primeiro (último preço conhecido)
        filled = series.ffill()
        
        # Se ainda houver NaN no início, backward fill
        filled = filled.bfill()
        
        # Se ainda houver NaN, usar média dos valores válidos
        if filled.isna().any() and filled.notna().any():
            mean_price = filled.mean()
            if pd.notna(mean_price) and mean_price > 0:
                filled = filled.fillna(mean_price)
        
        return filled
    
    def _fill_volume_data(self, series: pd.Series) -> pd.Series:
        """Preenche dados de volume - com cuidado"""
        # Forward fill
        filled = series.ffill()
        
        # Para volume, podemos usar zero apenas se não houver dados históricos
        if filled.isna().all():
            # Sem dados de volume - usar 1 como mínimo
            filled = filled.fillna(1)
        else:
            # Usar mediana dos volumes não-zero
            non_zero_median = filled[filled > 0].median()
            if pd.notna(non_zero_median):
                filled = filled.fillna(non_zero_median)
        
        return filled
    
    def _fill_indicator_data(self, series: pd.Series, indicator_name: str) -> pd.Series:
        """Preenche indicadores técnicos com valores apropriados"""
        filled = series.copy()
        
        # RSI: valor neutro é 50, não 0
        if 'rsi' in indicator_name.lower():
            filled = filled.ffill()
            # Apenas no início, antes de ter dados suficientes
            if filled.isna().any() and len(filled) > 14:
                filled = filled.fillna(50)
        
        # Moving averages: usar preço se disponível
        elif any(x in indicator_name.lower() for x in ['ema', 'sma', 'ma_']):
            filled = filled.ffill().bfill()
        
        # MACD: forward fill apenas
        elif 'macd' in indicator_name.lower():
            filled = filled.ffill()
        
        # ATR/Volatilidade: nunca zero
        elif any(x in indicator_name.lower() for x in ['atr', 'volatility']):
            filled = filled.ffill()
            if filled.isna().any() and filled.notna().any():
                # Usar média dos valores válidos
                mean_val = filled[filled > 0].mean()
                if pd.notna(mean_val):
                    filled = filled.fillna(mean_val)
        
        # Outros indicadores: forward fill
        else:
            filled = filled.ffill()
        
        return filled
    
    def _fill_ratio_data(self, series: pd.Series) -> pd.Series:
        """Preenche ratios e percentuais"""
        # Forward fill preferencial
        filled = series.ffill()
        
        # Para ratios no início, podemos usar valores neutros
        if filled.isna().any():
            # Identificar range típico dos valores
            if filled.notna().any():
                min_val, max_val = filled.min(), filled.max()
                
                # Se parecer ser percentual (0-1)
                if 0 <= min_val <= 1 and 0 <= max_val <= 1:
                    filled = filled.fillna(0.5)  # Neutro
                # Se for ratio em torno de 1
                elif 0.5 <= min_val <= 2 and 0.5 <= max_val <= 2:
                    filled = filled.fillna(1.0)  # Neutro
                else:
                    # Usar mediana
                    filled = filled.fillna(filled.median())
        
        return filled


class AdvancedFeatureProcessor:
    """Sistema avançado de feature engineering para day trade com ML - PRODUÇÃO"""
    
    def __init__(self, logger):
        self.logger = logger
        self.validator = ProductionDataValidator(logger)
        self.fill_strategy = SmartFillStrategy(logger)
        self.feature_cache = {}
        self.cache_expiry = 300  # 5 minutos
        self.last_cache_time = {}
        
        # Configuração Hughes Phenomenon
        self.max_features_by_samples = 15  # Conservador para day trade
        self.feature_importance_threshold = 0.05
        
        # Flags de segurança
        self.production_mode = True
        self.require_data_validation = True
        
    def extract_all_features(self, candles_df, microstructure_df=None, 
                           orderbook_df=None, timestamp=None):
        """Extrai todas as categorias de features com validação rigorosa"""
        
        # VALIDAÇÃO CRÍTICA - Bloqueia se dados não forem reais
        if self.require_data_validation:
            self.validator.validate_real_data(candles_df, "candles")
            
            if microstructure_df is not None and not microstructure_df.empty:
                self.validator.validate_real_data(microstructure_df, "microstructure")
            
            if orderbook_df is not None and not orderbook_df.empty:
                self.validator.validate_real_data(orderbook_df, "orderbook")
        
        cache_key = f"{len(candles_df)}_{timestamp}"
        
        # Verifica cache
        if self._is_cache_valid(cache_key):
            return self.feature_cache[cache_key]
        
        features = pd.DataFrame(index=candles_df.index)
        
        try:
            # 1. Features de Microestrutura
            if microstructure_df is not None and not microstructure_df.empty:
                micro_features = self._extract_microstructure_features(
                    candles_df, microstructure_df
                )
                features = pd.concat([features, micro_features], axis=1)
            
            # 2. Features Técnicas Adaptativas
            adaptive_features = self._extract_adaptive_technical_features(candles_df)
            features = pd.concat([features, adaptive_features], axis=1)
            
            # 3. Features de Regime de Mercado
            regime_features = self._extract_regime_features(candles_df)
            features = pd.concat([features, regime_features], axis=1)
            
            # 4. Validar features calculadas
            self._validate_calculated_features(features)
            
            # Cache resultado
            self._update_cache(cache_key, features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Erro crítico ao extrair features: {str(e)}")
            raise
    
    def _extract_microstructure_features(self, candles_df, micro_df):
        """Extrai features de microestrutura com validação"""
        
        features = pd.DataFrame(index=candles_df.index)
        
        try:
            # Order Flow Imbalance
            if 'buy_volume' in micro_df.columns and 'sell_volume' in micro_df.columns:
                # Validar volumes
                if (micro_df['buy_volume'] < 0).any() or (micro_df['sell_volume'] < 0).any():
                    raise ValueError("Volumes negativos detectados em microestrutura")
                
                features['order_flow_imbalance_1m'] = self._calculate_order_flow_imbalance(
                    micro_df, window='1min'
                )
                features['order_flow_imbalance_5m'] = self._calculate_order_flow_imbalance(
                    micro_df, window='5min'
                )
                
                # Buy/Sell Pressure
                total_volume = micro_df['buy_volume'] + micro_df['sell_volume']
                # Evitar divisão por zero
                features['buy_pressure'] = np.where(
                    total_volume > 0,
                    micro_df['buy_volume'] / total_volume,
                    0.5  # Neutro quando não há volume
                )
                
                features['volume_imbalance'] = np.where(
                    total_volume > 0,
                    (micro_df['buy_volume'] - micro_df['sell_volume']) / total_volume,
                    0.0  # Neutro
                )
                
                # Volume Rate of Change - sem fillna(0)
                features['volume_roc'] = total_volume.pct_change(5)
                # Forward fill para ROC
                features['volume_roc'] = features['volume_roc'].ffill()
            
            # Volume Profile
            if 'close' in candles_df.columns and 'volume' in candles_df.columns:
                features['volume_at_price_deviation'] = self._volume_at_price_deviation(
                    candles_df
                )
            
            # Trade Intensity
            if 'buy_trades' in micro_df.columns and 'sell_trades' in micro_df.columns:
                total_trades = micro_df['buy_trades'] + micro_df['sell_trades']
                
                features['trade_count_imbalance'] = np.where(
                    total_trades > 0,
                    (micro_df['buy_trades'] - micro_df['sell_trades']) / total_trades,
                    0.0
                )
                
                # Average Trade Size
                if 'buy_volume' in micro_df.columns:
                    features['avg_buy_trade_size'] = np.where(
                        micro_df['buy_trades'] > 0,
                        micro_df['buy_volume'] / micro_df['buy_trades'],
                        0.0
                    )
            
            # Aplicar preenchimento inteligente
            features = self.fill_strategy.fill_missing_values(features, 'ratio')
            
        except Exception as e:
            self.logger.error(f"Erro em microstructure features: {e}")
            # Em produção, não podemos continuar com features incorretas
            raise
        
        return features
    
    def _extract_adaptive_technical_features(self, candles_df):
        """Indicadores técnicos adaptativos com validação"""
        
        features = pd.DataFrame(index=candles_df.index)
        
        try:
            # Validar dados de entrada
            if 'close' not in candles_df.columns:
                raise ValueError("Coluna 'close' não encontrada nos candles")
            
            # Calcula volatilidade para adaptação
            volatility = candles_df['close'].rolling(20).std()
            
            # RSI Adaptativo
            features['adaptive_rsi'] = self._adaptive_rsi(
                candles_df['close'], volatility
            )
            
            # MACD Dinâmico
            features['dynamic_macd'] = self._dynamic_macd(
                candles_df['close'], candles_df.get('volume')
            )
            
            # Bollinger Bands Adaptativo
            features['adaptive_bb_position'] = self._adaptive_bollinger_position(
                candles_df['close'], volatility
            )
            
            # ATR Adaptativo
            if all(col in candles_df.columns for col in ['high', 'low', 'close']):
                features['adaptive_atr'] = self._adaptive_atr(
                    candles_df[['high', 'low', 'close']], volatility
                )
            
            # Aplicar preenchimento inteligente
            features = self.fill_strategy.fill_missing_values(features, 'indicator')
            
        except Exception as e:
            self.logger.error(f"Erro em adaptive features: {e}")
            raise
        
        return features
    
    def _extract_regime_features(self, candles_df):
        """Features para detecção de regime de mercado"""
        
        features = pd.DataFrame(index=candles_df.index)
        
        try:
            if 'close' not in candles_df.columns:
                raise ValueError("Coluna 'close' necessária para regime features")
            
            # Trend Strength
            features['trend_strength'] = self._calculate_trend_strength(candles_df)
            
            # Volatility Regime
            vol = candles_df['close'].rolling(20).std()
            vol_mean = vol.rolling(50).mean()
            
            # Evitar divisão por zero
            features['volatility_regime'] = np.where(
                vol_mean > 0,
                vol / vol_mean,
                1.0  # Neutro
            )
            
            # Price Position
            sma20 = candles_df['close'].rolling(20).mean()
            sma50 = candles_df['close'].rolling(50).mean()
            
            features['price_position'] = np.where(
                sma50 > 0,
                (candles_df['close'] - sma50) / sma50,
                0.0
            )
            
            # Momentum Regime
            mom = candles_df['close'].pct_change(10)
            features['momentum_regime'] = mom.rolling(20).mean()
            
            # Aplicar preenchimento inteligente
            features = self.fill_strategy.fill_missing_values(features, 'indicator')
            
        except Exception as e:
            self.logger.error(f"Erro em regime features: {e}")
            raise
        
        return features
    
    def _calculate_order_flow_imbalance(self, micro_df, window):
        """Calcula desequilíbrio do order flow - SEM fillna(0)"""
        
        if window == '1min':
            lookback = 1
        elif window == '5min':
            lookback = 5
        else:
            lookback = 1
        
        buy_volume = micro_df['buy_volume'].rolling(lookback).sum()
        sell_volume = micro_df['sell_volume'].rolling(lookback).sum()
        
        total_volume = buy_volume + sell_volume
        
        # Cálculo robusto do imbalance
        imbalance = pd.Series(index=micro_df.index, dtype=float)
        
        # Onde há volume, calcular imbalance
        mask = total_volume > 0
        imbalance[mask] = (buy_volume[mask] - sell_volume[mask]) / total_volume[mask]
        
        # Forward fill para períodos sem volume
        imbalance = imbalance.ffill()
        
        return imbalance
    
    def _volume_at_price_deviation(self, candles_df):
        """Calcula desvio do volume no preço atual"""
        
        try:
            # Verificar se temos dados suficientes
            if len(candles_df) < 10:
                return pd.Series(0, index=candles_df.index)
            
            # Cria perfil de volume simplificado
            price_bins = pd.qcut(candles_df['close'].dropna(), q=10, duplicates='drop')
            
            if len(price_bins.cat.categories) < 2:
                # Não há variação suficiente de preço
                return pd.Series(0, index=candles_df.index)
            
            volume_profile = candles_df.groupby(price_bins, observed=True)['volume'].mean()
            
            # Para cada preço, encontra seu desvio do volume médio
            deviation = pd.Series(index=candles_df.index, dtype=float)
            
            for idx, price in candles_df['close'].items():
                if pd.notna(price) and idx in candles_df.index:
                    # Encontra o bin do preço
                    for bin_range in volume_profile.index:
                        if price >= bin_range.left and price <= bin_range.right:
                            avg_volume = volume_profile[bin_range]
                            current_volume = candles_df.loc[idx, 'volume']
                            
                            if avg_volume > 0:
                                deviation[idx] = (current_volume - avg_volume) / avg_volume
                            else:
                                deviation[idx] = 0.0
                            break
            
            # Forward fill para valores faltantes
            deviation = deviation.ffill()
            
            return deviation
            
        except Exception as e:
            self.logger.error(f"Erro em volume_at_price_deviation: {e}")
            return pd.Series(0, index=candles_df.index)
    
    def _adaptive_rsi(self, prices, volatility):
        """RSI com período adaptativo - SEM fillna(50) arbitrário"""
        
        try:
            # Normaliza volatilidade
            vol_percentile = volatility.rolling(100).rank(pct=True)
            vol_percentile = vol_percentile.ffill()
            
            # Se não há dados de volatilidade, usar 0.5 (médio)
            vol_percentile = vol_percentile.fillna(0.5)
            
            # Período varia de 5 a 25 baseado na volatilidade
            adaptive_period = (5 + (25 - 5) * (1 - vol_percentile)).round()
            
            rsi_values = pd.Series(index=prices.index, dtype=float)
            
            # Calcula RSI adaptativo
            for i in range(len(prices)):
                period = int(adaptive_period.iloc[i]) if not pd.isna(adaptive_period.iloc[i]) else 14
                period = max(5, min(period, 25))  # Limitar entre 5 e 25
                
                if i >= period:
                    price_slice = prices.iloc[max(0, i-period+1):i+1]
                    
                    if len(price_slice) >= period:
                        # Cálculo robusto do RSI
                        delta = price_slice.diff()
                        gain = delta.where(delta > 0, 0).mean()
                        loss = -delta.where(delta < 0, 0).mean()
                        
                        if loss > 0:
                            rs = gain / loss
                            rsi_values.iloc[i] = 100 - (100 / (1 + rs))
                        elif gain > 0:
                            rsi_values.iloc[i] = 100
                        else:
                            rsi_values.iloc[i] = 50
            
            # Forward fill para consistência temporal
            rsi_values = rsi_values.ffill()
            
            return rsi_values
            
        except Exception as e:
            self.logger.error(f"Erro em adaptive RSI: {e}")
            return pd.Series(50, index=prices.index)
    
    def _dynamic_macd(self, prices, volume=None):
        """MACD com parâmetros dinâmicos - SEM fillna(0)"""
        
        try:
            # Parâmetros base
            fast = 12
            slow = 26
            signal = 9
            
            # Ajusta baseado em volume se disponível
            if volume is not None and len(volume) > 50:
                vol_ratio = volume.rolling(20).mean() / volume.rolling(50).mean()
                vol_ratio = vol_ratio.ffill()
                
                # Se não há dados suficientes, ratio neutro
                vol_ratio = vol_ratio.fillna(1.0)
                
                # Alta volume = períodos menores (mais responsivo)
                fast_adj = np.where(vol_ratio > 1.2, 10, fast)
                slow_adj = np.where(vol_ratio > 1.2, 22, slow)
            else:
                fast_adj = fast
                slow_adj = slow
            
            # Calcula MACD
            ema_fast = prices.ewm(span=fast_adj, adjust=False).mean()
            ema_slow = prices.ewm(span=slow_adj, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            
            # NÃO preencher com zero - MACD pode e deve cruzar zero
            return macd_line
            
        except Exception as e:
            self.logger.error(f"Erro em dynamic MACD: {e}")
            return pd.Series(index=prices.index, dtype=float)
    
    def _adaptive_bollinger_position(self, prices, volatility):
        """Posição nas Bollinger Bands adaptativas"""
        
        try:
            # Período base
            period = 20
            
            # Multiplier adaptativo baseado em volatilidade
            vol_percentile = volatility.rolling(100).rank(pct=True)
            vol_percentile = vol_percentile.ffill().fillna(0.5)
            
            # Varia de 1.5 a 2.5 desvios
            multiplier = 1.5 + 1.0 * vol_percentile
            
            # Calcula bandas
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            
            upper = sma + (multiplier * std)
            lower = sma - (multiplier * std)
            
            # Posição relativa (0 = lower band, 1 = upper band)
            band_width = upper - lower
            position = pd.Series(index=prices.index, dtype=float)
            
            # Onde há largura de banda válida
            mask = band_width > 0
            position[mask] = (prices[mask] - lower[mask]) / band_width[mask]
            
            # Forward fill
            position = position.ffill()
            
            return position
            
        except Exception as e:
            self.logger.error(f"Erro em adaptive Bollinger: {e}")
            return pd.Series(0.5, index=prices.index)
    
    def _adaptive_atr(self, ohlc_df, volatility):
        """ATR adaptativo - NUNCA pode ser zero"""
        
        try:
            # ATR tradicional
            high_low = ohlc_df['high'] - ohlc_df['low']
            high_close = np.abs(ohlc_df['high'] - ohlc_df['close'].shift())
            low_close = np.abs(ohlc_df['low'] - ohlc_df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            # Período adaptativo baseado em volatilidade
            vol_percentile = volatility.rolling(100).rank(pct=True)
            vol_percentile = vol_percentile.ffill().fillna(0.5)
            
            # Período entre 7 e 21
            adaptive_period = (7 + 14 * (1 - vol_percentile)).round().astype(int)
            
            # ATR com período fixo como base
            atr = true_range.rolling(14).mean()
            
            # Forward fill - ATR não deve ter gaps
            atr = atr.ffill()
            
            # Se ainda há NaN, usar true range médio
            if atr.isna().any():
                tr_mean = true_range[true_range > 0].mean()
                if pd.notna(tr_mean) and tr_mean > 0:
                    atr = atr.fillna(tr_mean)
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Erro em adaptive ATR: {e}")
            # Em caso de erro, retornar série com valores mínimos
            return pd.Series(0.0001, index=ohlc_df.index)
    
    def _calculate_trend_strength(self, candles_df):
        """Calcula força da tendência"""
        
        try:
            close = candles_df['close']
            
            # Diferença entre médias
            ema9 = close.ewm(span=9, adjust=False).mean()
            ema20 = close.ewm(span=20, adjust=False).mean()
            
            # Normaliza pela volatilidade
            volatility = close.rolling(20).std()
            
            # Cálculo robusto
            trend_strength = pd.Series(index=close.index, dtype=float)
            
            mask = volatility > 0
            trend_strength[mask] = (ema9[mask] - ema20[mask]) / volatility[mask]
            
            # Forward fill
            trend_strength = trend_strength.ffill()
            
            return trend_strength
            
        except Exception as e:
            self.logger.error(f"Erro em trend strength: {e}")
            return pd.Series(0, index=candles_df.index)
    
    def _validate_calculated_features(self, features: pd.DataFrame):
        """Valida features calculadas antes de retornar"""
        
        # Verificar se há muitos zeros suspeitos
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                zero_ratio = (features[col] == 0).sum() / len(features)
                
                # Alertar se mais de 50% zeros (exceto onde esperado)
                if zero_ratio > 0.5 and not any(x in col for x in ['imbalance', 'position']):
                    self.logger.warning(
                        f"Feature '{col}' tem {zero_ratio:.1%} valores zero - verificar cálculo"
                    )
        
        # Verificar NaN excessivos
        nan_ratios = features.isna().sum() / len(features)
        high_nan_features = nan_ratios[nan_ratios > 0.2]
        
        if len(high_nan_features) > 0:
            self.logger.warning(
                f"Features com muitos NaN: {high_nan_features.to_dict()}"
            )
    
    def _is_cache_valid(self, key):
        """Verifica se cache ainda é válido"""
        if key not in self.feature_cache:
            return False
        
        current_time = time.time()
        if current_time - self.last_cache_time.get(key, 0) > self.cache_expiry:
            return False
        
        return True
    
    def _update_cache(self, key, data):
        """Atualiza cache com novos dados"""
        self.feature_cache[key] = data
        self.last_cache_time[key] = time.time()
        
        # Limpa cache antigo
        if len(self.feature_cache) > 100:
            oldest_key = min(self.last_cache_time.keys(), 
                           key=lambda x: self.last_cache_time[x])
            del self.feature_cache[oldest_key]
            del self.last_cache_time[oldest_key]


class IntelligentFeatureSelector:
    """Seleção inteligente de features para produção"""
    
    def __init__(self, logger):
        self.logger = logger
        self.selected_features = None
        self.feature_scores = {}
        self.selection_history = []
        
    def select_optimal_features(self, features_df, target, max_features=15):
        """Seleciona features ótimas com validação rigorosa"""
        
        if features_df.empty or len(target) == 0:
            self.logger.error("Dados insuficientes para seleção de features")
            return features_df
        
        # Remove features com muitos NaN
        nan_threshold = 0.2  # Máximo 20% NaN
        valid_features = features_df.dropna(axis=1, thresh=len(features_df)*(1-nan_threshold))
        
        if valid_features.empty:
            self.logger.error("Nenhuma feature válida após remover NaN")
            return features_df
        
        # Validar variância das features
        low_variance_cols = []
        for col in valid_features.columns:
            if valid_features[col].std() < 1e-10:
                low_variance_cols.append(col)
        
        if low_variance_cols:
            self.logger.warning(f"Removendo features com baixa variância: {low_variance_cols}")
            valid_features = valid_features.drop(columns=low_variance_cols)
        
        # Ensemble de métodos de seleção
        scores = {}
        
        try:
            # 1. Mutual Information
            mi_scores = self._mutual_information_score(valid_features, target)
            scores['mutual_info'] = mi_scores
            
            # 2. F-statistic
            f_scores = self._f_statistic_score(valid_features, target)
            scores['f_stat'] = f_scores
            
            # 3. Random Forest Importance (apenas com dados suficientes)
            if len(valid_features) > 200:  # Mais conservador
                rf_scores = self._random_forest_importance(valid_features, target)
                scores['rf_importance'] = rf_scores
            
            # Combina scores
            combined_scores = self._combine_scores(scores)
            
            # Seleciona top features
            top_features = self._select_top_features(
                combined_scores, min(max_features, len(valid_features.columns))
            )
            
            self.selected_features = top_features
            self.feature_scores = combined_scores
            
            # Log seleção
            self.logger.info(
                f"Features selecionadas: {len(top_features)} de {len(features_df.columns)}"
            )
            
            # Salvar histórico
            self.selection_history.append({
                'timestamp': datetime.now(),
                'selected_features': top_features,
                'scores': combined_scores
            })
            
            return features_df[top_features]
            
        except Exception as e:
            self.logger.error(f"Erro na seleção de features: {e}")
            # Em caso de erro, retornar features originais
            return features_df
    
    def _mutual_information_score(self, features, target):
        """Calcula mutual information com tratamento de erros"""
        mi_scores = {}
        
        for col in features.columns:
            try:
                # Preparar dados
                X = features[[col]].fillna(features[[col]].mean())
                y = target.fillna(target.mean())
                
                # Verificar se há variância
                if X[col].std() > 1e-10:
                    score = mutual_info_regression(X, y, random_state=42)[0]
                    mi_scores[col] = max(0, score)  # Garantir não-negativo
                else:
                    mi_scores[col] = 0
                    
            except Exception as e:
                self.logger.warning(f"Erro em MI para {col}: {e}")
                mi_scores[col] = 0
                
        return mi_scores
    
    def _f_statistic_score(self, features, target):
        """Calcula F-statistic com tratamento robusto"""
        f_scores = {}
        
        for col in features.columns:
            try:
                X = features[[col]].fillna(features[[col]].mean())
                y = target.fillna(target.mean())
                
                if X[col].std() > 1e-10:
                    score, _ = f_regression(X, y)
                    f_scores[col] = max(0, score[0])
                else:
                    f_scores[col] = 0
                    
            except Exception as e:
                self.logger.warning(f"Erro em F-stat para {col}: {e}")
                f_scores[col] = 0
                
        return f_scores
    
    def _random_forest_importance(self, features, target):
        """Random Forest importance com configuração conservadora"""
        try:
            # Preparar dados
            X = features.fillna(features.mean())
            y = target.fillna(target.mean())
            
            rf = RandomForestRegressor(
                n_estimators=30,      # Reduzido para velocidade
                max_depth=5,          # Shallow trees
                min_samples_split=20, # Conservador
                random_state=42,
                n_jobs=-1
            )
            
            rf.fit(X, y)
            
            importance_scores = {}
            for idx, col in enumerate(features.columns):
                importance_scores[col] = max(0, rf.feature_importances_[idx])
                
            return importance_scores
            
        except Exception as e:
            self.logger.error(f"Erro em RF importance: {e}")
            return {col: 0 for col in features.columns}
    
    def _combine_scores(self, scores_dict):
        """Combina scores com ponderação robusta"""
        combined = {}
        
        # Coletar todas as features
        all_features = set()
        for method_scores in scores_dict.values():
            all_features.update(method_scores.keys())
        
        for feature in all_features:
            scores = []
            weights = []
            
            # Ponderar por método
            method_weights = {
                'mutual_info': 0.4,
                'f_stat': 0.4,
                'rf_importance': 0.2
            }
            
            for method, method_scores in scores_dict.items():
                if feature in method_scores:
                    score = method_scores[feature]
                    
                    # Normalizar por método
                    all_method_scores = list(method_scores.values())
                    max_score = max(all_method_scores) if all_method_scores else 1
                    
                    if max_score > 0:
                        normalized = score / max_score
                        scores.append(normalized)
                        weights.append(method_weights.get(method, 0.33))
            
            if scores:
                # Média ponderada
                combined[feature] = np.average(scores, weights=weights)
            else:
                combined[feature] = 0
                
        return combined
    
    def _select_top_features(self, scores, max_features):
        """Seleciona features com validação adicional"""
        
        # Ordenar por score
        sorted_features = sorted(
            scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Filtrar features com score mínimo
        min_score = 0.1
        valid_features = [
            f[0] for f in sorted_features 
            if f[1] >= min_score
        ]
        
        # Selecionar top N
        top_features = valid_features[:max_features]
        
        # Log features selecionadas
        self.logger.info(f"Top 5 features: {sorted_features[:5]}")
        
        return top_features


class FeatureEngine:
    """Motor principal de cálculo de features - VERSÃO PRODUÇÃO"""
    
    def __init__(self, model_features: Optional[List[str]] = None, allow_historical_data: bool = False):
        self.model_features = model_features or []
        self.technical = TechnicalIndicators()
        self.ml_features = MLFeatures(model_features)
        self.logger = logging.getLogger(__name__)
        
        # Flag para permitir dados históricos (backtest)
        self.allow_historical_data = allow_historical_data
        
        # Processadores avançados
        self.advanced_processor = AdvancedFeatureProcessor(self.logger)
        self.feature_selector = IntelligentFeatureSelector(self.logger)
        self.validator = ProductionDataValidator(self.logger, allow_historical_data)
        self.fill_strategy = SmartFillStrategy(self.logger)
        
        # Cache
        self.cache = {
            'indicators': None,
            'features': None,
            'advanced_features': None,
            'last_candle_time': None,
            'last_candle_count': 0,
            'last_selection_time': None
        }
        
        # Configurações
        self.min_candles_required = 50
        self.parallel_processing = True
        self.max_workers = 4
        self.use_advanced_features = True
        self.feature_selection_interval = 3600
        
        # FLAGS DE SEGURANÇA
        self.production_mode = True
        self.require_validation = True
        self.block_on_dummy_data = True
        
        # Mapeamento de dependências
        self.feature_dependencies = self._build_feature_dependencies()
        
        self.logger.info("FeatureEngine inicializado em modo PRODUÇÃO")
    
    def calculate(self, data: TradingDataStructure, 
                 force_recalculate: bool = False,
                 use_advanced: Optional[bool] = None) -> Dict[str, pd.DataFrame]:
        """
        Calcula todas as features com validação rigorosa para produção
        
        BLOQUEIA se detectar dados dummy
        """
        
        # VALIDAÇÃO CRÍTICA 1: Dados não podem estar vazios
        if data.candles.empty:
            error_msg = "ERRO CRÍTICO: Sem dados de candles para calcular features"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # VALIDAÇÃO CRÍTICA 2: Validar dados reais
        if self.require_validation:
            try:
                self.validator.validate_real_data(data.candles, "candles_main")
                
                if data.microstructure is not None and not data.microstructure.empty:
                    self.validator.validate_real_data(data.microstructure, "microstructure_main")
                    
            except ValueError as e:
                if self.block_on_dummy_data:
                    error_msg = f"BLOQUEIO DE SEGURANÇA: {str(e)}"
                    self.logger.error(error_msg)
                    raise
                else:
                    self.logger.warning(f"Aviso de validação: {str(e)}")
        
        # Determinar uso de features avançadas
        if use_advanced is None:
            use_advanced = self.use_advanced_features
        
        # Verificar cache
        if not force_recalculate and self._is_cache_valid(data):
            self.logger.info("Usando cache de features válido")
            return self._get_from_cache(data)
        
        self.logger.info(f"Calculando features para {len(data.candles)} candles")
        
        try:
            # Calcular features
            if self.parallel_processing and len(data.candles) > 100:
                result = self._calculate_parallel(data, use_advanced)
            else:
                result = self._calculate_sequential(data, use_advanced)
            
            # Aplicar seleção de features se necessário
            if use_advanced and self._should_select_features():
                result = self._apply_feature_selection(data, result)
            
            # Validação final
            if 'model_ready' in result:
                self._validate_final_features(result['model_ready'])
            
            # Atualizar cache
            self._update_cache(data, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro crítico no cálculo de features: {str(e)}")
            raise
    
    def adapt_features_dynamically(self, market_regime: str, 
                             performance_metrics: Dict) -> None:
        """Adapta features dinamicamente baseado no regime e performance"""
        
        self.logger.info(f"Adaptando features para regime: {market_regime}")
        
        # Ajustar pesos das features baseado no regime
        if market_regime == 'high_volatility':
            self.feature_weights = {
                'momentum': 0.15,
                'volatility': 0.35,
                'microstructure': 0.30,
                'technical': 0.20
            }
        elif market_regime == 'trending':
            self.feature_weights = {
                'momentum': 0.40,
                'volatility': 0.15,
                'microstructure': 0.20,
                'technical': 0.25
            }
        elif market_regime == 'ranging':
            self.feature_weights = {
                'momentum': 0.10,
                'volatility': 0.20,
                'microstructure': 0.35,
                'technical': 0.35
            }
        else:  # normal
            self.feature_weights = {
                'momentum': 0.25,
                'volatility': 0.25,
                'microstructure': 0.25,
                'technical': 0.25
            }
        
        # Ajustar seleção de features baseado na performance
        if performance_metrics.get('win_rate', 0) < 0.5:
            self.logger.warning("Win rate baixo - priorizando features de risco")
            self.prioritize_risk_features = True
        else:
            self.prioritize_risk_features = False
            
    def _calculate_sequential(self, data: TradingDataStructure, 
                            use_advanced: bool = True) -> Dict[str, pd.DataFrame]:
        """Calcula features sequencialmente com validação"""
        
        # Indicadores técnicos
        data.indicators = self.technical.calculate_all(data.candles)
        
        # Features ML básicas
        data.features = self.ml_features.calculate_all(
            data.candles,
            data.microstructure,
            data.indicators
        )
        
        # Features avançadas se habilitado
        if use_advanced and len(data.candles) > 100:
            try:
                advanced_features = self.advanced_processor.extract_all_features(
                    data.candles,
                    data.microstructure,
                    data.orderbook,
                    timestamp=time.time()
                )
                
                if not advanced_features.empty:
                    # Mesclar features sem duplicatas
                    new_cols = [col for col in advanced_features.columns 
                               if col not in data.features.columns]
                    if new_cols:
                        data.features = pd.concat([
                            data.features, 
                            advanced_features[new_cols]
                        ], axis=1)
                
                self.cache['advanced_features'] = advanced_features
                
            except Exception as e:
                self.logger.error(f"Erro em features avançadas: {e}")
                # Em produção, continuar sem features avançadas
                pass
        
        # Preparar DataFrame final
        model_ready_df = self._prepare_model_data(data)
        
        return {
            'indicators': data.indicators,
            'features': data.features,
            'model_ready': model_ready_df,
            'all': self._merge_all_features(data)
        }
    
    def _calculate_parallel(self, data: TradingDataStructure,
                          use_advanced: bool = True) -> Dict[str, pd.DataFrame]:
        """Calcula features em paralelo com validação"""
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            # Indicadores técnicos
            futures['indicators'] = executor.submit(
                self.technical.calculate_all, data.candles
            )
            
            # Features avançadas em paralelo
            if use_advanced and len(data.candles) > 100:
                futures['advanced'] = executor.submit(
                    self.advanced_processor.extract_all_features,
                    data.candles,
                    data.microstructure,
                    data.orderbook,
                    time.time()
                )
            
            # Aguardar indicadores
            indicators = futures['indicators'].result()
            data.indicators = indicators
            
            # Features ML
            futures['features'] = executor.submit(
                self.ml_features.calculate_all,
                data.candles,
                data.microstructure,
                indicators
            )
            
            # Coletar resultados
            data.features = futures['features'].result()
            
            # Features avançadas
            if 'advanced' in futures:
                try:
                    advanced_features = futures['advanced'].result()
                    if not advanced_features.empty:
                        new_cols = [col for col in advanced_features.columns 
                                   if col not in data.features.columns]
                        if new_cols:
                            data.features = pd.concat([
                                data.features, 
                                advanced_features[new_cols]
                            ], axis=1)
                    
                    self.cache['advanced_features'] = advanced_features
                    
                except Exception as e:
                    self.logger.error(f"Erro em features avançadas paralelas: {e}")
        
        # Preparar DataFrame final
        model_ready_df = self._prepare_model_data(data)
        
        return {
            'indicators': data.indicators,
            'features': data.features,
            'model_ready': model_ready_df,
            'all': self._merge_all_features(data)
        }
    
    def _prepare_model_data(self, data: TradingDataStructure) -> pd.DataFrame:
        """
        Prepara DataFrame final com features do modelo
        
        CRÍTICO: Sem fillna(0) arbitrário
        """
        if not self.model_features:
            return self._merge_all_features(data)
        
        # Mapear features para fontes
        feature_sources = {
            'candles': data.candles,
            'indicators': data.indicators,
            'features': data.features,
            'microstructure': data.microstructure
        }
        
        # Coletar features
        features_by_source = {}
        missing_features = []
        
        for feature in self.model_features:
            found = False
            
            for source_name, source_df in feature_sources.items():
                if source_df is not None and not source_df.empty and feature in source_df.columns:
                    if source_name not in features_by_source:
                        features_by_source[source_name] = []
                    features_by_source[source_name].append(feature)
                    found = True
                    break
            
            if not found:
                missing_features.append(feature)
        
        # Criar DataFrames para concatenar
        dfs_to_concat = []
        
        for source_name, feature_list in features_by_source.items():
            source_df = feature_sources[source_name]
            if source_df is not None and feature_list:
                dfs_to_concat.append(source_df[feature_list])
        
        # Concatenar
        if dfs_to_concat:
            model_data = pd.concat(dfs_to_concat, axis=1)
        else:
            model_data = pd.DataFrame(index=data.candles.index)
        
        # Tratar features ausentes
        if missing_features:
            self.logger.warning(f"Features não encontradas: {missing_features[:5]}...")
            
            # NÃO criar com zeros - deixar sistema decidir
            if not self.production_mode:
                # Apenas em desenvolvimento
                missing_df = pd.DataFrame(
                    0, 
                    index=model_data.index,
                    columns=missing_features
                )
                model_data = pd.concat([model_data, missing_df], axis=1)
        
        # Ordenar features
        if self.model_features:
            available_features = [f for f in self.model_features if f in model_data.columns]
            model_data = model_data[available_features]
        
        # PREENCHIMENTO INTELIGENTE - Não usar fillna(0)
        model_data = self._smart_fill_model_data(model_data)
        
        # Remover período de warm-up
        if len(model_data) > self.min_candles_required:
            model_data = self._remove_warmup_period(model_data)
        
        self.logger.info(f"DataFrame do modelo preparado: {model_data.shape}")
        
        return model_data
    
    def _smart_fill_model_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preenchimento inteligente específico para modelo"""
        
        for col in df.columns:
            if df[col].isna().any():
                col_lower = col.lower()
                
                # RSI - valor neutro é 50
                if 'rsi' in col_lower:
                    df[col] = df[col].ffill()
                    # Apenas preencher com 50 se não houver dados históricos
                    if df[col].isna().all():
                        df[col] = 50
                    else:
                        df[col] = df[col].fillna(df[col].mean())
                
                # EMAs/SMAs - usar preço se disponível
                elif any(x in col_lower for x in ['ema', 'sma', 'ma_']):
                    df[col] = df[col].ffill().bfill()
                    if df[col].isna().any() and 'close' in df.columns:
                        df[col] = df[col].fillna(df['close'])
                
                # Volume - forward fill, nunca zero
                elif 'volume' in col_lower:
                    df[col] = df[col].ffill()
                    if df[col].isna().any():
                        non_zero_mean = df[col][df[col] > 0].mean()
                        if pd.notna(non_zero_mean):
                            df[col] = df[col].fillna(non_zero_mean)
                        else:
                            df[col] = df[col].fillna(1)  # Mínimo 1
                
                # Volatilidade/ATR - nunca zero
                elif any(x in col_lower for x in ['volatility', 'atr', 'std']):
                    df[col] = df[col].ffill()
                    if df[col].isna().any():
                        vol_mean = df[col][df[col] > 0].mean()
                        if pd.notna(vol_mean):
                            df[col] = df[col].fillna(vol_mean)
                
                # Ratios/percentuais - forward fill
                elif any(x in col_lower for x in ['ratio', 'pct', 'percent']):
                    df[col] = df[col].ffill().bfill()
                
                # Default - forward fill primeiro
                else:
                    df[col] = df[col].ffill()
                    # Se ainda há NaN, backward fill
                    if df[col].isna().any():
                        df[col] = df[col].bfill()
        
        return df
    
    def _remove_warmup_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove período de aquecimento com validação"""
        
        # Encontrar primeira linha com dados válidos suficientes
        min_valid_ratio = 0.8  # 80% das features devem ser válidas
        
        for i in range(len(df)):
            row = df.iloc[i]
            valid_ratio = row.notna().sum() / len(row)
            
            if valid_ratio >= min_valid_ratio:
                # Encontrou primeira linha válida
                start_idx = max(0, i - 20)  # Manter 20 candles antes
                return df.iloc[start_idx:]
        
        # Se não encontrou, retornar tudo
        return df
    
    def _validate_final_features(self, features_df: pd.DataFrame):
        """Validação final rigorosa antes de usar em trading"""
        
        validation_errors = []
        
        # 1. Verificar se há features suficientes
        if len(features_df.columns) < 5:
            validation_errors.append("Menos de 5 features disponíveis")
        
        # 2. Verificar NaN excessivos
        nan_ratios = features_df.isna().sum() / len(features_df)
        high_nan = nan_ratios[nan_ratios > 0.1]
        if len(high_nan) > 0:
            validation_errors.append(f"Features com >10% NaN: {high_nan.index.tolist()}")
        
        # 3. Verificar zeros suspeitos
        for col in features_df.columns:
            if features_df[col].dtype in ['float64', 'int64']:
                zero_ratio = (features_df[col] == 0).sum() / len(features_df)
                
                # Algumas features podem ter zeros válidos
                if zero_ratio > 0.8 and not any(x in col for x in ['imbalance', 'position', 'flag']):
                    validation_errors.append(f"Feature '{col}' tem {zero_ratio:.1%} zeros")
        
        # 4. Verificar variância
        low_var_features = []
        for col in features_df.columns:
            if features_df[col].std() < 1e-10:
                low_var_features.append(col)
        
        if low_var_features:
            validation_errors.append(f"Features sem variância: {low_var_features}")
        
        # Em produção, erros são críticos
        if validation_errors and self.production_mode:
            error_msg = f"Validação de features falhou: {'; '.join(validation_errors)}"
            self.logger.error(error_msg)
            
            # Decidir se bloqueia operação
            if len(validation_errors) > 2:
                raise ValueError(error_msg)
        elif validation_errors:
            self.logger.warning(f"Avisos de validação: {validation_errors}")
    
    def _merge_all_features(self, data: TradingDataStructure) -> pd.DataFrame:
        """Mescla todas as features com validação"""
        
        dfs_to_merge = [data.candles.copy()]
        
        # Adicionar outros DataFrames
        if data.indicators is not None and not data.indicators.empty:
            cols_to_add = [col for col in data.indicators.columns 
                          if col not in data.candles.columns]
            if cols_to_add:
                dfs_to_merge.append(data.indicators[cols_to_add])
        
        if data.features is not None and not data.features.empty:
            existing_cols = set(data.candles.columns)
            if data.indicators is not None:
                existing_cols.update(data.indicators.columns)
            cols_to_add = [col for col in data.features.columns 
                          if col not in existing_cols]
            if cols_to_add:
                dfs_to_merge.append(data.features[cols_to_add])
        
        # Concatenar
        all_features = pd.concat(dfs_to_merge, axis=1)
        
        # Microestrutura com alinhamento
        if data.microstructure is not None and not data.microstructure.empty:
            common_index = all_features.index.intersection(data.microstructure.index)
            if len(common_index) > 0:
                micro_aligned = data.microstructure.loc[common_index].copy()
                cols_to_add = [col for col in micro_aligned.columns 
                              if col not in all_features.columns]
                
                if cols_to_add:
                    micro_full = pd.DataFrame(index=all_features.index)
                    for col in cols_to_add:
                        micro_full[col] = pd.Series(
                            micro_aligned[col].values,
                            index=common_index
                        )
                    
                    all_features = pd.concat([all_features, micro_full], axis=1)
        
        return all_features
    
    def _should_select_features(self) -> bool:
        """Determina se deve executar seleção de features"""
        
        if self.cache['last_selection_time'] is None:
            return True
        
        time_since_selection = time.time() - self.cache['last_selection_time']
        return time_since_selection > self.feature_selection_interval
    
    def _apply_feature_selection(self, data: TradingDataStructure, 
                               result: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Aplica seleção de features com validação"""
        
        try:
            all_features = result.get('all', pd.DataFrame())
            
            if all_features.empty or len(all_features.columns) <= 20:
                return result
            
            # Preparar target
            if 'close' in data.candles.columns:
                target = data.candles['close'].pct_change().shift(-1)
                
                # Remover features de preço diretas
                feature_cols = [col for col in all_features.columns 
                              if col not in ['open', 'high', 'low', 'close', 'volume']]
                
                # Selecionar features
                selected_df = self.feature_selector.select_optimal_features(
                    all_features[feature_cols],
                    target,
                    max_features=15
                )
                
                # Atualizar resultado
                if self.feature_selector.selected_features:
                    result['selected_features'] = selected_df
                    
                    # Garantir model features
                    if self.model_features:
                        required_cols = []
                        for mf in self.model_features:
                            if mf in all_features.columns and mf not in selected_df.columns:
                                required_cols.append(mf)
                        
                        if required_cols:
                            result['model_ready'] = pd.concat([
                                selected_df,
                                all_features[required_cols]
                            ], axis=1)
                        else:
                            result['model_ready'] = selected_df
                
                self.cache['last_selection_time'] = time.time()
                
        except Exception as e:
            self.logger.error(f"Erro na seleção de features: {e}")
        
        return result
    
    def _build_feature_dependencies(self) -> Dict[str, Set[str]]:
        """Mapa de dependências entre features"""
        
        dependencies = {
            # Indicadores básicos
            'rsi': {'close'},
            'macd': {'close'},
            'macd_signal': {'macd'},
            'macd_hist': {'macd', 'macd_signal'},
            
            # EMAs
            **{f'ema_{p}': {'close'} for p in [5, 9, 20, 50, 200]},
            
            # Bollinger Bands
            'bb_upper_20': {'close'},
            'bb_lower_20': {'close'},
            'bb_position_20': {'close', 'bb_upper_20', 'bb_lower_20'},
            
            # Features compostas
            'price_to_ema20': {'close', 'ema_20'},
            'macd_divergence': {'macd', 'macd_signal'},
            'trend_regime': {'momentum_20', 'volatility_20'},
            
            # Microestrutura
            'buy_pressure': {'buy_volume', 'sell_volume'},
            'volume_imbalance': {'buy_volume', 'sell_volume'},
            'order_flow_imbalance_1m': {'buy_volume', 'sell_volume'},
            'order_flow_imbalance_5m': {'buy_volume', 'sell_volume'},
            
            # Features avançadas
            'adaptive_rsi': {'close'},
            'dynamic_macd': {'close', 'volume'},
            'adaptive_bb_position': {'close'},
            'adaptive_atr': {'high', 'low', 'close'}
        }
        
        return dependencies
    
    def get_required_features_for_models(self, model_features: List[str]) -> Set[str]:
        """Retorna todas as features necessárias incluindo dependências"""
        
        required = set(model_features)
        
        # Adicionar dependências recursivamente
        added = True
        while added:
            added = False
            for feature in list(required):
                if feature in self.feature_dependencies:
                    deps = self.feature_dependencies[feature]
                    new_deps = deps - required
                    if new_deps:
                        required.update(new_deps)
                        added = True
        
        return required
    
    def _is_cache_valid(self, data: TradingDataStructure) -> bool:
        """Verifica validade do cache"""
        
        if self.cache['indicators'] is None or self.cache['features'] is None:
            return False
        
        if len(data.candles) != self.cache['last_candle_count']:
            return False
        
        if not data.candles.empty:
            last_time = data.candles.index[-1]
            if last_time != self.cache['last_candle_time']:
                return False
        
        return True
    
    def _update_cache(self, data: TradingDataStructure, result: Dict[str, pd.DataFrame]):
        """Atualiza cache"""
        
        self.cache['indicators'] = result.get('indicators')
        self.cache['features'] = result.get('features')
        self.cache['last_candle_count'] = len(data.candles)
        
        if not data.candles.empty:
            self.cache['last_candle_time'] = data.candles.index[-1]
    
    def _get_from_cache(self, data: TradingDataStructure) -> Dict[str, pd.DataFrame]:
        """Retorna dados do cache"""
        
        data.indicators = self.cache['indicators']
        data.features = self.cache['features']
        
        model_ready_df = self._prepare_model_data(data)
        
        return {
            'indicators': data.indicators,
            'features': data.features,
            'model_ready': model_ready_df,
            'all': self._merge_all_features(data)
        }
    
    def validate_feature_quality(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Validação completa de qualidade para produção"""
        
        validation_report = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {},
            'recommendations': []
        }
        
        # Verificar cada feature
        for col in features_df.columns:
            col_stats = {
                'name': col,
                'dtype': str(features_df[col].dtype),
                'nan_count': features_df[col].isna().sum(),
                'nan_ratio': features_df[col].isna().sum() / len(features_df),
                'zero_count': 0,
                'zero_ratio': 0,
                'unique_count': features_df[col].nunique(),
                'std': features_df[col].std() if features_df[col].dtype in ['float64', 'int64'] else None
            }
            
            # Análise de zeros
            if features_df[col].dtype in ['float64', 'int64']:
                col_stats['zero_count'] = (features_df[col] == 0).sum()
                col_stats['zero_ratio'] = col_stats['zero_count'] / len(features_df)
                
                # Verificações
                if col_stats['zero_ratio'] > 0.5:
                    if not any(x in col for x in ['binary', 'flag', 'is_', 'imbalance']):
                        validation_report['warnings'].append(
                            f"Feature '{col}' tem {col_stats['zero_ratio']:.1%} zeros"
                        )
                
                if col_stats['zero_ratio'] > 0.9:
                    validation_report['errors'].append(
                        f"Feature '{col}' tem {col_stats['zero_ratio']:.1%} zeros - suspeito"
                    )
                    validation_report['valid'] = False
                
                # Verificar variância
                if col_stats['std'] is not None and col_stats['std'] < 1e-10:
                    validation_report['warnings'].append(
                        f"Feature '{col}' tem variância muito baixa"
                    )
            
            # Verificar NaN
            if col_stats['nan_ratio'] > 0.1:
                validation_report['warnings'].append(
                    f"Feature '{col}' tem {col_stats['nan_ratio']:.1%} NaN"
                )
            
            if col_stats['nan_ratio'] > 0.5:
                validation_report['errors'].append(
                    f"Feature '{col}' tem muitos NaN: {col_stats['nan_ratio']:.1%}"
                )
                validation_report['valid'] = False
        
        # Estatísticas gerais
        validation_report['statistics'] = {
            'total_features': len(features_df.columns),
            'total_rows': len(features_df),
            'features_with_zeros': sum((features_df == 0).any()),
            'features_with_nan': sum(features_df.isna().any()),
            'average_nan_ratio': features_df.isna().sum().sum() / (len(features_df) * len(features_df.columns))
        }
        
        # Recomendações
        if validation_report['warnings'] or validation_report['errors']:
            validation_report['recommendations'].append(
                "Revisar cálculo de features com muitos zeros"
            )
            validation_report['recommendations'].append(
                "Verificar fonte de dados para features com NaN"
            )
        
        return validation_report
    
    def get_feature_statistics(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Estatísticas detalhadas das features"""
        
        stats = pd.DataFrame()
        
        for col in features_df.columns:
            if features_df[col].dtype in ['float64', 'int64']:
                stats.loc[col, 'mean'] = features_df[col].mean()
                stats.loc[col, 'std'] = features_df[col].std()
                stats.loc[col, 'min'] = features_df[col].min()
                stats.loc[col, 'max'] = features_df[col].max()
                stats.loc[col, 'q25'] = features_df[col].quantile(0.25)
                stats.loc[col, 'median'] = features_df[col].median()
                stats.loc[col, 'q75'] = features_df[col].quantile(0.75)
                stats.loc[col, 'nulls'] = features_df[col].isnull().sum()
                stats.loc[col, 'zeros'] = (features_df[col] == 0).sum()
                stats.loc[col, 'unique'] = features_df[col].nunique()
                
                # Adicionar métricas de qualidade
                mean_val = stats.loc[col, 'mean']
                std_val = stats.loc[col, 'std']
                
                # Verificar se os valores são numéricos antes do cálculo do CV
                if (pd.notna(mean_val) and pd.notna(std_val) and 
                    isinstance(mean_val, (int, float)) and isinstance(std_val, (int, float)) and
                    mean_val != 0):
                    stats.loc[col, 'cv'] = std_val / abs(mean_val)
                else:
                    stats.loc[col, 'cv'] = float('inf')
                stats.loc[col, 'skew'] = features_df[col].skew()
                stats.loc[col, 'kurtosis'] = features_df[col].kurtosis()
        
        return stats
    
    def get_features_for_prediction(self, data: TradingDataStructure,
                                  lookback: int = 100, 
                                  force_recalculate: bool = False) -> pd.DataFrame:
        """Obtém features validadas para predição"""
        
        # Calcular se necessário
        if force_recalculate or not self._is_cache_valid(data):
            self.calculate(data, use_advanced=True)
        
        # Obter features
        if hasattr(self.cache, 'selected_features') and self.cache.get('selected_features') is not None:
            features = self.cache['selected_features'].iloc[-lookback:]
        else:
            model_data = self._prepare_model_data(data)
            features = model_data.iloc[-lookback:]
        
        # Validação final
        if self.production_mode:
            validation = self.validate_feature_quality(features)
            if not validation['valid']:
                raise ValueError(f"Features inválidas para predição: {validation['errors']}")
        
        return features
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Retorna importância das features"""
        
        if not self.feature_selector.feature_scores:
            return {
                'status': 'Nenhuma seleção de features realizada',
                'top_features': [],
                'selected_count': 0,
                'total_features': 0
            }
        
        sorted_importance = sorted(
            self.feature_selector.feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'status': 'Importância de features disponível',
            'top_features': sorted_importance[:10],
            'selected_count': len(self.feature_selector.selected_features or []),
            'total_features': len(self.feature_selector.feature_scores),
            'selection_time': self.cache.get('last_selection_time', 0),
            'selection_history': len(self.feature_selector.selection_history)
        }
    
    def enable_advanced_features(self, enabled: bool = True):
        """Habilita/desabilita features avançadas"""
        
        self.use_advanced_features = enabled
        self.logger.info(f"Features avançadas {'habilitadas' if enabled else 'desabilitadas'}")
    
    def set_feature_selection_interval(self, seconds: int):
        """Define intervalo de reseleção de features"""
        
        self.feature_selection_interval = seconds
        self.logger.info(f"Intervalo de seleção de features: {seconds} segundos")
    def create_features_separated(self, candles_df: pd.DataFrame, 
                                microstructure_df: pd.DataFrame, 
                                indicators_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        '''
        Cria features separadas por tipo de dados
        Compatibilidade com testes de integração
        '''
        try:
            if candles_df.empty:
                self.logger.warning("⚠️ DataFrame de candles vazio")
                return {'features': pd.DataFrame(), 'metadata': {}}
            
            # Usar o método principal de features
            if hasattr(self, 'create_features'):
                features_df = self.create_features(candles_df)
            else:
                # Fallback: criar features básicas
                features_df = self._create_basic_features(candles_df)
            
            return {
                'features': features_df,
                'candles': candles_df,
                'microstructure': microstructure_df,
                'indicators': indicators_df,
                'metadata': {
                    'features_count': len(features_df.columns),
                    'data_points': len(features_df),
                    'created_at': pd.Timestamp.now()
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erro em create_features_separated: {e}")
            return {'features': pd.DataFrame(), 'metadata': {}}
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Cria features básicas para fallback'''
        try:
            features = df.copy()
            
            # Features básicas
            if 'close' in df.columns:
                features['returns'] = df['close'].pct_change()
                features['volatility'] = features['returns'].rolling(20).std()
                
                # EMAs simples
                for period in [9, 20, 50]:
                    features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                
                # RSI aproximado
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                features['rsi_14'] = 100 - (100 / (1 + rs))
                
            # Volume features
            if 'volume' in df.columns:
                features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
                
            # Preencher NaNs
            features = features.fillna(method='bfill').fillna(0)
            
            self.logger.info(f"🔧 Features básicas criadas: {len(features.columns)} colunas")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Erro em _create_basic_features: {e}")
            return pd.DataFrame()
