# -*- coding: utf-8 -*-
"""
Time Travel Feature Engine - Calcula features usando dados históricos completos via Valkey
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import logging

class TimeTravelFeatureEngine:
    """
    Feature Engine com capacidades de Time Travel
    Calcula features usando dados históricos completos via Valkey
    """
    
    def __init__(self, valkey_manager, original_feature_engine=None):
        self.valkey_manager = valkey_manager
        self.original_engine = original_feature_engine
        self.logger = logging.getLogger('TimeTravelFeatures')
        
        # Cache para otimização
        self.cache = {}
        self.cache_ttl = 300  # 5 minutos
        
    def calculate_enhanced_features(self, symbol: str, 
                                  current_time: Optional[datetime] = None,
                                  lookback_minutes: int = 120,
                                  use_cache: bool = True) -> Optional[Dict]:
        """
        Calcula features aprimoradas usando time travel
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Check cache
        cache_key = f"{symbol}_{current_time.strftime('%Y%m%d%H%M')}_{lookback_minutes}"
        if use_cache and cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                self.logger.debug(f"Usando cache para {cache_key}")
                return cached_data
        
        try:
            # 1. Buscar dados históricos via time travel
            start_time = current_time - timedelta(minutes=lookback_minutes)
            
            # Buscar ticks
            ticks = self.valkey_manager.time_travel_query(
                symbol, start_time, current_time, 'ticks'
            )
            
            if not ticks:
                self.logger.warning(f"Sem dados para {symbol} no período {start_time} - {current_time}")
                return None
            
            # 2. Converter para DataFrame
            df = pd.DataFrame(ticks)
            
            # Converter timestamp_ms para datetime
            if 'timestamp_ms' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp_ms'].astype(float), unit='ms')
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            
            # 3. Calcular features básicas usando engine original (se disponível)
            if self.original_engine:
                base_features = self.original_engine.calculate_features(df)
            else:
                base_features = self._calculate_basic_features(df)
            
            # 4. Adicionar features exclusivas de time travel
            enhanced = self._add_time_travel_features(df, base_features, symbol, current_time)
            
            # Cache result
            if use_cache:
                self.cache[cache_key] = (datetime.now(), enhanced)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular features enhanced: {e}")
            return None
    
    def _calculate_basic_features(self, df: pd.DataFrame) -> Dict:
        """Calcula features básicas se não houver engine original"""
        features = {}
        
        if 'price' in df.columns:
            features['last_price'] = df['price'].iloc[-1]
            features['mean_price'] = df['price'].mean()
            features['std_price'] = df['price'].std()
            features['price_change'] = df['price'].iloc[-1] - df['price'].iloc[0]
            features['price_change_pct'] = (df['price'].iloc[-1] / df['price'].iloc[0] - 1) * 100
            
        if 'volume' in df.columns:
            features['total_volume'] = df['volume'].sum()
            features['mean_volume'] = df['volume'].mean()
            features['volume_std'] = df['volume'].std()
            
        return features
    
    def _add_time_travel_features(self, df: pd.DataFrame, 
                                 base_features: Dict, 
                                 symbol: str,
                                 current_time: datetime) -> Dict:
        """Adiciona features que só são possíveis com time travel"""
        
        enhanced = base_features.copy()
        
        try:
            # 1. Padrão de Volume Intraday
            enhanced['volume_pattern_score'] = self._calculate_volume_pattern(df, symbol)
            
            # 2. Momentum Histórico Comparativo
            enhanced['historical_momentum_rank'] = self._calculate_momentum_rank(df, symbol, current_time)
            
            # 3. Microestrutura de Mercado
            enhanced['microstructure_imbalance'] = self._calculate_microstructure(df)
            
            # 4. Regime de Volatilidade Estendido
            enhanced['volatility_regime'] = self._calculate_volatility_regime(df, symbol, current_time)
            
            # 5. Sazonalidade Intraday
            enhanced['intraday_seasonality'] = self._calculate_intraday_seasonality(df, current_time)
            
            # 6. Momentum vs Histórico
            enhanced['momentum_percentile'] = self._calculate_momentum_percentile(df, symbol, current_time)
            
            # 7. Volume Profile Analysis
            enhanced['volume_profile_score'] = self._calculate_volume_profile(df)
            
            # 8. Price Action Quality
            enhanced['price_action_quality'] = self._calculate_price_action_quality(df)
            
            # Metadados
            enhanced['time_travel_used'] = True
            enhanced['lookback_minutes'] = len(df) / 60 if len(df) > 0 else 0
            enhanced['data_points'] = len(df)
            enhanced['data_quality'] = self._calculate_data_quality(df)
            
        except Exception as e:
            self.logger.error(f"Erro em time travel features: {e}")
        
        return enhanced
    
    def _calculate_volume_pattern(self, df: pd.DataFrame, symbol: str) -> float:
        """
        Calcula similaridade do padrão de volume atual com dias anteriores
        """
        try:
            # Agregar volume por minuto
            volume_profile = df['volume'].resample('1min').sum()
            
            if len(volume_profile) < 60:
                return 0.5  # Neutro se não houver dados suficientes
            
            # Buscar padrões históricos dos últimos 7 dias
            current_hour = df.index[-1].hour
            current_minute = df.index[-1].minute
            
            similarities = []
            
            for days_back in range(1, 8):
                hist_end = df.index[-1] - timedelta(days=days_back)
                hist_start = hist_end - timedelta(hours=2)
                
                # Time travel para período histórico
                hist_data = self.valkey_manager.time_travel_query(
                    symbol, hist_start, hist_end, 'ticks'
                )
                
                if hist_data and len(hist_data) > 60:
                    hist_df = pd.DataFrame(hist_data)
                    hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp_ms'].astype(float), unit='ms')
                    hist_df.set_index('timestamp', inplace=True)
                    
                    hist_volume = hist_df['volume'].resample('1min').sum()
                    
                    # Calcular correlação
                    if len(hist_volume) >= 60:
                        current_pattern = volume_profile.values[-60:]
                        hist_pattern = hist_volume.values[-60:]
                        
                        if current_pattern.std() > 0 and hist_pattern.std() > 0:
                            correlation = np.corrcoef(current_pattern, hist_pattern)[0, 1]
                            similarities.append(correlation)
            
            return np.nanmean(similarities) if similarities else 0.5
            
        except Exception as e:
            self.logger.error(f"Erro em volume pattern: {e}")
            return 0.5
    
    def _calculate_momentum_rank(self, df: pd.DataFrame, symbol: str, current_time: datetime) -> float:
        """
        Calcula ranking do momentum atual vs histórico
        """
        try:
            if len(df) < 60:
                return 0.5
            
            # Calcular momentum atual (última hora)
            current_return = (df['price'].iloc[-1] / df['price'].iloc[-60] - 1) * 100
            
            # Buscar retornos históricos de 30 dias
            historical_returns = []
            
            for days_back in range(1, 31):
                hist_time = current_time - timedelta(days=days_back)
                hist_start = hist_time - timedelta(hours=1)
                
                hist_data = self.valkey_manager.time_travel_query(
                    symbol, hist_start, hist_time, 'ticks'
                )
                
                if hist_data and len(hist_data) >= 60:
                    hist_df = pd.DataFrame(hist_data)
                    if 'price' in hist_df.columns:
                        hist_return = (float(hist_df['price'].iloc[-1]) / float(hist_df['price'].iloc[0]) - 1) * 100
                        historical_returns.append(hist_return)
            
            # Calcular percentil do momentum atual
            if historical_returns:
                rank = np.sum(current_return > np.array(historical_returns)) / len(historical_returns)
                return rank
            
        except Exception as e:
            self.logger.error(f"Erro em momentum rank: {e}")
        
        return 0.5
    
    def _calculate_microstructure(self, df: pd.DataFrame) -> float:
        """
        Calcula desequilíbrio na microestrutura do mercado
        """
        try:
            if 'quantity' not in df.columns or 'trade_type' not in df.columns:
                return 0.0
            
            # Analisar tamanho dos trades
            median_size = df['quantity'].median()
            
            if median_size > 0:
                # Separar trades grandes vs pequenos
                large_trades = df[df['quantity'] > median_size * 2]
                
                if len(large_trades) > 0:
                    # Calcular pressão direcional dos trades grandes
                    buy_volume = large_trades[large_trades['trade_type'] == 'BUY']['volume'].sum()
                    sell_volume = large_trades[large_trades['trade_type'] == 'SELL']['volume'].sum()
                    
                    total_volume = buy_volume + sell_volume
                    if total_volume > 0:
                        imbalance = (buy_volume - sell_volume) / total_volume
                        return np.clip(imbalance, -1, 1)
            
        except Exception as e:
            self.logger.error(f"Erro em microstructure: {e}")
        
        return 0.0
    
    def _calculate_volatility_regime(self, df: pd.DataFrame, symbol: str, current_time: datetime) -> Dict:
        """
        Identifica regime de volatilidade baseado em 30 dias
        """
        try:
            # Calcular volatilidade realizada atual
            returns = df['price'].pct_change().dropna()
            current_vol = returns.std() * np.sqrt(252 * 24 * 60)  # Anualizada
            
            # Buscar volatilidades históricas de 30 dias
            historical_vols = []
            
            for days_back in range(1, 31):
                hist_end = current_time - timedelta(days=days_back)
                hist_start = hist_end - timedelta(hours=24)
                
                hist_data = self.valkey_manager.time_travel_query(
                    symbol, hist_start, hist_end, 'ticks'
                )
                
                if hist_data and len(hist_data) > 100:
                    hist_df = pd.DataFrame(hist_data)
                    if 'price' in hist_df.columns:
                        hist_returns = hist_df['price'].pct_change().dropna()
                        hist_vol = hist_returns.std() * np.sqrt(252 * 24 * 60)
                        historical_vols.append(hist_vol)
            
            # Classificar regime
            if historical_vols:
                percentile = np.sum(current_vol > np.array(historical_vols)) / len(historical_vols)
                
                if percentile > 0.8:
                    regime = "HIGH_VOL"
                elif percentile < 0.2:
                    regime = "LOW_VOL"
                else:
                    regime = "NORMAL_VOL"
                
                return {
                    'regime': regime,
                    'percentile': percentile,
                    'current_vol': current_vol,
                    'historical_avg': np.mean(historical_vols)
                }
            
        except Exception as e:
            self.logger.error(f"Erro em volatility regime: {e}")
        
        return {'regime': 'UNKNOWN', 'percentile': 0.5, 'current_vol': 0, 'historical_avg': 0}
    
    def _calculate_intraday_seasonality(self, df: pd.DataFrame, current_time: datetime) -> float:
        """
        Calcula score de sazonalidade intraday
        """
        try:
            current_hour = current_time.hour
            current_minute = current_time.minute
            
            # Horários típicos de alta atividade no WDO
            high_activity_periods = [
                (9, 0, 10, 30),   # Abertura
                (14, 0, 15, 30),  # Tarde
                (16, 30, 17, 30)  # Fechamento
            ]
            
            # Verificar se está em período de alta atividade
            seasonality_score = 0.5  # Neutro
            
            for start_h, start_m, end_h, end_m in high_activity_periods:
                start_minutes = start_h * 60 + start_m
                end_minutes = end_h * 60 + end_m
                current_minutes = current_hour * 60 + current_minute
                
                if start_minutes <= current_minutes <= end_minutes:
                    # Calcular intensidade baseada na proximidade do pico
                    period_middle = (start_minutes + end_minutes) / 2
                    distance_from_middle = abs(current_minutes - period_middle)
                    period_length = end_minutes - start_minutes
                    
                    intensity = 1 - (distance_from_middle / (period_length / 2))
                    seasonality_score = max(0.5, intensity)
                    break
            
            return seasonality_score
            
        except Exception as e:
            self.logger.error(f"Erro em intraday seasonality: {e}")
            return 0.5
    
    def _calculate_momentum_percentile(self, df: pd.DataFrame, symbol: str, current_time: datetime) -> float:
        """
        Calcula percentil do momentum em relação ao histórico
        """
        try:
            # Calcular momentum de diferentes períodos
            momentum_periods = [5, 15, 30, 60]  # minutos
            
            momentum_scores = []
            
            for period in momentum_periods:
                if len(df) >= period:
                    momentum = (df['price'].iloc[-1] / df['price'].iloc[-period] - 1) * 100
                    
                    # Comparar com histórico
                    historical_momentums = []
                    
                    for days_back in range(1, 15):  # Últimas 2 semanas
                        hist_end = current_time - timedelta(days=days_back)
                        hist_start = hist_end - timedelta(minutes=period)
                        
                        hist_data = self.valkey_manager.time_travel_query(
                            symbol, hist_start, hist_end, 'ticks'
                        )
                        
                        if hist_data and len(hist_data) >= period:
                            hist_df = pd.DataFrame(hist_data)
                            if 'price' in hist_df.columns:
                                hist_momentum = (float(hist_df['price'].iloc[-1]) / float(hist_df['price'].iloc[0]) - 1) * 100
                                historical_momentums.append(hist_momentum)
                    
                    if historical_momentums:
                        percentile = np.sum(momentum > np.array(historical_momentums)) / len(historical_momentums)
                        momentum_scores.append(percentile)
            
            return np.mean(momentum_scores) if momentum_scores else 0.5
            
        except Exception as e:
            self.logger.error(f"Erro em momentum percentile: {e}")
            return 0.5
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> float:
        """
        Analisa perfil de volume e retorna score
        """
        try:
            if 'volume' not in df.columns or 'price' not in df.columns:
                return 0.5
            
            # Criar bins de preço
            price_range = df['price'].max() - df['price'].min()
            if price_range == 0:
                return 0.5
            
            n_bins = min(20, len(df) // 10)
            price_bins = pd.cut(df['price'], bins=n_bins)
            
            # Volume por faixa de preço
            volume_profile = df.groupby(price_bins)['volume'].sum()
            
            # Identificar POC (Point of Control)
            poc_idx = volume_profile.idxmax()
            poc_volume = volume_profile.max()
            total_volume = volume_profile.sum()
            
            if total_volume > 0:
                # Concentração de volume no POC
                poc_concentration = poc_volume / total_volume
                
                # Distribuição de volume (quanto mais concentrado, maior o score)
                volume_std = volume_profile.std()
                volume_mean = volume_profile.mean()
                
                if volume_mean > 0:
                    cv = volume_std / volume_mean  # Coeficiente de variação
                    distribution_score = 1 / (1 + np.exp(-cv))  # Sigmoid
                    
                    # Score final combina concentração e distribuição
                    profile_score = (poc_concentration + distribution_score) / 2
                    
                    return np.clip(profile_score, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Erro em volume profile: {e}")
        
        return 0.5
    
    def _calculate_price_action_quality(self, df: pd.DataFrame) -> float:
        """
        Avalia qualidade da ação do preço
        """
        try:
            if 'price' not in df.columns or len(df) < 10:
                return 0.5
            
            # 1. Trend strength (R²)
            x = np.arange(len(df))
            y = df['price'].values
            
            # Regressão linear
            coeffs = np.polyfit(x, y, 1)
            y_pred = np.polyval(coeffs, x)
            
            # R-squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # 2. Volatilidade normalizada
            returns = df['price'].pct_change().dropna()
            volatility = returns.std()
            mean_return = abs(returns.mean())
            
            if mean_return > 0:
                volatility_ratio = volatility / mean_return
                volatility_score = 1 / (1 + volatility_ratio)  # Menor volatilidade = maior score
            else:
                volatility_score = 0.5
            
            # 3. Suavidade do movimento (menos whipsaws)
            direction_changes = np.sum(np.diff(np.sign(returns)) != 0)
            smoothness_score = 1 - (direction_changes / len(returns))
            
            # Score final
            quality_score = (r_squared + volatility_score + smoothness_score) / 3
            
            return np.clip(quality_score, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Erro em price action quality: {e}")
            return 0.5
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """
        Avalia qualidade dos dados
        """
        try:
            quality_factors = []
            
            # 1. Completude dos dados
            expected_fields = ['price', 'volume', 'timestamp']
            completeness = sum(1 for field in expected_fields if field in df.columns) / len(expected_fields)
            quality_factors.append(completeness)
            
            # 2. Densidade temporal (gaps)
            if len(df) > 1:
                time_diffs = df.index.to_series().diff().dropna()
                median_diff = time_diffs.median()
                
                # Contar gaps maiores que 2x a mediana
                gaps = sum(time_diffs > median_diff * 2)
                gap_ratio = 1 - (gaps / len(time_diffs))
                quality_factors.append(gap_ratio)
            
            # 3. Valores válidos
            if 'price' in df.columns:
                valid_prices = df['price'] > 0
                valid_ratio = valid_prices.sum() / len(df)
                quality_factors.append(valid_ratio)
            
            return np.mean(quality_factors) if quality_factors else 0.0
            
        except Exception as e:
            self.logger.error(f"Erro em data quality: {e}")
            return 0.0
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retorna importância estimada das features time travel
        """
        return {
            'volume_pattern_score': 0.85,
            'historical_momentum_rank': 0.80,
            'microstructure_imbalance': 0.75,
            'volatility_regime': 0.90,
            'intraday_seasonality': 0.70,
            'momentum_percentile': 0.85,
            'volume_profile_score': 0.80,
            'price_action_quality': 0.75
        }