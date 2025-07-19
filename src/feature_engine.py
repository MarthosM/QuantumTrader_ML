import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_selection import mutual_info_regression, f_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor
import time

# Use ta library instead (already installed and working)
import ta

from data_structure import TradingDataStructure
from technical_indicators import TechnicalIndicators
from ml_features import MLFeatures


class AdvancedFeatureProcessor:
    """Sistema avançado de feature engineering para day trade com ML"""
    
    def __init__(self, logger):
        self.logger = logger
        self.feature_cache = {}
        self.cache_expiry = 300  # 5 minutos
        self.last_cache_time = {}
        
        # Configuração Hughes Phenomenon
        self.max_features_by_samples = 15  # Conservador para day trade
        self.feature_importance_threshold = 0.05
        
    def extract_all_features(self, candles_df, microstructure_df=None, 
                           orderbook_df=None, timestamp=None):
        """Extrai todas as categorias de features com cache inteligente"""
        
        cache_key = f"{len(candles_df)}_{timestamp}"
        
        # Verifica cache
        if self._is_cache_valid(cache_key):
            return self.feature_cache[cache_key]
        
        features = pd.DataFrame(index=candles_df.index)
        
        # 1. Features de Microestrutura (baixa latência)
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
        
        # Cache resultado
        self._update_cache(cache_key, features)
        
        return features
    
    def _extract_microstructure_features(self, candles_df, micro_df):
        """Extrai features de microestrutura para day trade"""
        
        features = pd.DataFrame(index=candles_df.index)
        
        try:
            # Order Flow Imbalance
            if 'buy_volume' in micro_df.columns and 'sell_volume' in micro_df.columns:
                features['order_flow_imbalance_1m'] = self._calculate_order_flow_imbalance(
                    micro_df, window='1min'
                )
                features['order_flow_imbalance_5m'] = self._calculate_order_flow_imbalance(
                    micro_df, window='5min'
                )
                
                # Buy/Sell Pressure
                total_volume = micro_df['buy_volume'] + micro_df['sell_volume']
                features['buy_pressure'] = micro_df['buy_volume'] / (total_volume + 1e-10)
                features['volume_imbalance'] = (
                    micro_df['buy_volume'] - micro_df['sell_volume']
                ) / (total_volume + 1e-10)
                
                # Volume Rate of Change
                features['volume_roc'] = total_volume.pct_change(5).fillna(0)
            
            # Volume Profile
            if 'close' in candles_df.columns and 'volume' in candles_df.columns:
                features['volume_at_price_deviation'] = self._volume_at_price_deviation(
                    candles_df
                )
            
            # Trade Intensity
            if 'buy_trades' in micro_df.columns and 'sell_trades' in micro_df.columns:
                total_trades = micro_df['buy_trades'] + micro_df['sell_trades']
                features['trade_count_imbalance'] = (
                    micro_df['buy_trades'] - micro_df['sell_trades']
                ) / (total_trades + 1e-10)
                
                # Average Trade Size
                if 'buy_volume' in micro_df.columns:
                    features['avg_buy_trade_size'] = (
                        micro_df['buy_volume'] / (micro_df['buy_trades'] + 1e-10)
                    ).fillna(0)
            
        except Exception as e:
            self.logger.error(f"Erro em microstructure features: {e}")
        
        return features
    
    def _extract_adaptive_technical_features(self, candles_df):
        """Indicadores técnicos que se adaptam às condições de mercado"""
        
        features = pd.DataFrame(index=candles_df.index)
        
        try:
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
            
        except Exception as e:
            self.logger.error(f"Erro em adaptive features: {e}")
        
        return features
    
    def _extract_regime_features(self, candles_df):
        """Features para detecção de regime de mercado"""
        
        features = pd.DataFrame(index=candles_df.index)
        
        try:
            # Trend Strength
            if 'close' in candles_df.columns:
                # ADX simplificado
                features['trend_strength'] = self._calculate_trend_strength(candles_df)
                
                # Volatility Regime
                vol = candles_df['close'].rolling(20).std()
                vol_mean = vol.rolling(50).mean()
                features['volatility_regime'] = (vol / (vol_mean + 1e-10)).fillna(1)
                
                # Price Position
                sma20 = candles_df['close'].rolling(20).mean()
                sma50 = candles_df['close'].rolling(50).mean()
                features['price_position'] = (
                    (candles_df['close'] - sma50) / (sma50 + 1e-10)
                ).fillna(0)
                
                # Momentum Regime
                mom = candles_df['close'].pct_change(10)
                features['momentum_regime'] = mom.rolling(20).mean().fillna(0)
            
        except Exception as e:
            self.logger.error(f"Erro em regime features: {e}")
        
        return features
    
    def _calculate_order_flow_imbalance(self, micro_df, window):
        """Calcula desequilíbrio do order flow"""
        
        if window == '1min':
            lookback = 1
        elif window == '5min':
            lookback = 5
        else:
            lookback = 1
        
        buy_volume = micro_df['buy_volume'].rolling(lookback).sum()
        sell_volume = micro_df['sell_volume'].rolling(lookback).sum()
        
        total_volume = buy_volume + sell_volume
        imbalance = (buy_volume - sell_volume) / (total_volume + 1e-10)
        
        return imbalance.fillna(0)
    
    def _volume_at_price_deviation(self, candles_df):
        """Calcula desvio do volume no preço atual"""
        
        try:
            # Cria perfil de volume simplificado
            price_bins = pd.qcut(candles_df['close'].dropna(), q=10, duplicates='drop')
            volume_profile = candles_df.groupby(price_bins, observed=True)['volume'].mean()
            
            # Para cada preço, encontra seu desvio do volume médio
            deviation = pd.Series(index=candles_df.index, dtype=float)
            
            for idx, price in candles_df['close'].items():
                if pd.notna(price):
                    # Encontra o bin do preço
                    for bin_range in volume_profile.index:
                        if price >= bin_range.left and price <= bin_range.right:
                            avg_volume = volume_profile[bin_range]
                            current_volume = candles_df.loc[idx, 'volume']
                            deviation[idx] = (current_volume - avg_volume) / (avg_volume + 1e-10)
                            break
            
            return deviation.fillna(0)
            
        except Exception:
            return pd.Series(0, index=candles_df.index)
    
    def _adaptive_rsi(self, prices, volatility):
        """RSI com período adaptativo baseado na volatilidade"""
        
        try:
            # Normaliza volatilidade
            vol_percentile = volatility.rolling(100).rank(pct=True).fillna(0.5)
            
            # Período varia de 5 a 25 baseado na volatilidade
            adaptive_period = (5 + (25 - 5) * (1 - vol_percentile)).round()
            
            rsi_values = pd.Series(index=prices.index, dtype=float)
            
            # Calcula RSI adaptativo
            for i in range(len(prices)):
                period = int(adaptive_period.iloc[i]) if not pd.isna(adaptive_period.iloc[i]) else 14
                
                if i >= period:
                    price_slice = prices.iloc[i-period+1:i+1]
                    if len(price_slice) >= 5:
                        try:
                            # Use apenas implementação manual robusta
                            # Calcular RSI usando método simples e confiável
                            delta = price_slice.diff()
                            gain = delta.where(delta > 0, 0)
                            loss = -delta.where(delta < 0, 0)
                            
                            # Média móvel exponencial para suavizar
                            alpha = 2 / (period + 1)
                            avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
                            avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
                            
                            if not avg_loss.empty and avg_loss.iloc[-1] > 0:
                                rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
                                rsi_value = 100 - (100 / (1 + rs))
                                if not pd.isna(rsi_value) and 0 <= rsi_value <= 100:
                                    rsi_values.iloc[i] = rsi_value
                        except:
                            # Em caso de erro, usar valor neutro
                            rsi_values.iloc[i] = 50
            
            return rsi_values.fillna(50)  # Valor neutro como default
            
        except Exception:
            return pd.Series(50, index=prices.index)
    
    def _dynamic_macd(self, prices, volume=None):
        """MACD com parâmetros dinâmicos baseados em volume"""
        
        try:
            # Parâmetros base
            fast = 12
            slow = 26
            signal = 9
            
            # Ajusta baseado em volume se disponível
            if volume is not None and len(volume) > 50:
                vol_ratio = volume.rolling(20).mean() / volume.rolling(50).mean()
                vol_ratio = vol_ratio.fillna(1)
                
                # Alta volume = períodos menores
                fast = np.where(vol_ratio > 1.2, 10, fast)
                slow = np.where(vol_ratio > 1.2, 22, slow)
            
            # Calcula MACD
            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            
            return macd_line.fillna(0)
            
        except Exception:
            return pd.Series(0, index=prices.index)
    
    def _adaptive_bollinger_position(self, prices, volatility):
        """Posição nas Bollinger Bands com largura adaptativa"""
        
        try:
            # Período base
            period = 20
            
            # Multiplier adaptativo baseado em volatilidade
            vol_percentile = volatility.rolling(100).rank(pct=True).fillna(0.5)
            multiplier = 1.5 + 1.0 * vol_percentile  # Varia de 1.5 a 2.5
            
            # Calcula bandas
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            
            upper = sma + (multiplier * std)
            lower = sma - (multiplier * std)
            
            # Posição relativa
            position = (prices - lower) / (upper - lower + 1e-10)
            
            return position.fillna(0.5)
            
        except Exception:
            return pd.Series(0.5, index=prices.index)
    
    def _adaptive_atr(self, ohlc_df, volatility):
        """ATR adaptativo baseado em regime de volatilidade"""
        
        try:
            # ATR tradicional
            high_low = ohlc_df['high'] - ohlc_df['low']
            high_close = np.abs(ohlc_df['high'] - ohlc_df['close'].shift())
            low_close = np.abs(ohlc_df['low'] - ohlc_df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            # Período adaptativo
            vol_percentile = volatility.rolling(100).rank(pct=True).fillna(0.5)
            period = (7 + 14 * (1 - vol_percentile)).round().astype(int)
            
            atr = true_range.rolling(14).mean()  # Fallback para período fixo
            
            return atr.fillna(0)
            
        except Exception:
            return pd.Series(0, index=ohlc_df.index)
    
    def _calculate_trend_strength(self, candles_df):
        """Calcula força da tendência"""
        
        try:
            close = candles_df['close']
            
            # Diferença entre médias
            ema9 = close.ewm(span=9, adjust=False).mean()
            ema20 = close.ewm(span=20, adjust=False).mean()
            
            # Normaliza pela volatilidade
            volatility = close.rolling(20).std()
            trend_strength = (ema9 - ema20) / (volatility + 1e-10)
            
            return trend_strength.fillna(0)
            
        except Exception:
            return pd.Series(0, index=candles_df.index)
    
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
            # Encontrar a chave mais antiga
            if self.last_cache_time:
                oldest_key = min(self.last_cache_time.keys(), key=lambda x: self.last_cache_time[x])
                del self.feature_cache[oldest_key]
                del self.last_cache_time[oldest_key]


class IntelligentFeatureSelector:
    """Seleção inteligente de features respeitando Hughes Phenomenon"""
    
    def __init__(self, logger):
        self.logger = logger
        self.selected_features = None
        self.feature_scores = {}
        self.selection_history = []
        
    def select_optimal_features(self, features_df, target, max_features=15):
        """Seleciona features ótimas usando ensemble de métodos"""
        
        if features_df.empty or len(target) == 0:
            return features_df
        
        # Remove features com muitos NaN
        valid_features = features_df.dropna(axis=1, thresh=len(features_df)*0.8)
        
        if valid_features.empty:
            return features_df
        
        # Ensemble de métodos de seleção
        scores = {}
        
        try:
            # 1. Mutual Information
            mi_scores = self._mutual_information_score(valid_features, target)
            scores['mutual_info'] = mi_scores
            
            # 2. F-statistic
            f_scores = self._f_statistic_score(valid_features, target)
            scores['f_stat'] = f_scores
            
            # 3. Random Forest Importance
            if len(valid_features) > 100:  # Só usa RF se tiver dados suficientes
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
            self.logger.info(f"Features selecionadas: {len(top_features)} de {len(features_df.columns)}")
            
            return features_df[top_features]
            
        except Exception as e:
            self.logger.error(f"Erro na seleção de features: {e}")
            return features_df
    
    def _mutual_information_score(self, features, target):
        """Calcula mutual information score"""
        mi_scores = {}
        
        for col in features.columns:
            try:
                if features[col].nunique() > 1:
                    score = mutual_info_regression(
                        features[[col]].fillna(0), 
                        target.fillna(0), 
                        random_state=42
                    )[0]
                    mi_scores[col] = score
                else:
                    mi_scores[col] = 0
            except Exception:
                mi_scores[col] = 0
                
        return mi_scores
    
    def _f_statistic_score(self, features, target):
        """Calcula F-statistic score"""
        f_scores = {}
        
        for col in features.columns:
            try:
                if features[col].nunique() > 1:
                    score, _ = f_regression(
                        features[[col]].fillna(0), 
                        target.fillna(0)
                    )
                    f_scores[col] = score[0] if len(score) > 0 else 0
                else:
                    f_scores[col] = 0
            except Exception:
                f_scores[col] = 0
                
        return f_scores
    
    def _random_forest_importance(self, features, target):
        """Calcula importância usando Random Forest"""
        try:
            rf = RandomForestRegressor(
                n_estimators=50, 
                max_depth=5, 
                random_state=42,
                n_jobs=-1
            )
            
            rf.fit(features.fillna(0), target.fillna(0))
            
            importance_scores = {}
            for idx, col in enumerate(features.columns):
                importance_scores[col] = rf.feature_importances_[idx]
                
            return importance_scores
            
        except Exception:
            return {col: 0 for col in features.columns}
    
    def _combine_scores(self, scores_dict):
        """Combina scores de diferentes métodos"""
        combined = {}
        
        all_features = set()
        for method_scores in scores_dict.values():
            all_features.update(method_scores.keys())
        
        for feature in all_features:
            scores = []
            for method, method_scores in scores_dict.items():
                if feature in method_scores:
                    # Normaliza score
                    all_scores = list(method_scores.values())
                    max_score = max(all_scores) if all_scores else 1
                    if max_score > 0:
                        normalized = method_scores[feature] / max_score
                        scores.append(normalized)
            
            if scores:
                combined[feature] = np.mean(scores)
            else:
                combined[feature] = 0
                
        return combined
    
    def _select_top_features(self, scores, max_features):
        """Seleciona top features baseado nos scores"""
        sorted_features = sorted(
            scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        top_features = [f[0] for f in sorted_features[:max_features]]
        
        return top_features


class FeatureEngine:
    """Motor principal de cálculo de features com capacidades avançadas"""
    
    def __init__(self, model_features: Optional[List[str]] = None):
        self.model_features = model_features or []
        self.technical = TechnicalIndicators()
        self.ml_features = MLFeatures(model_features)
        self.logger = logging.getLogger(__name__)
        
        # Novo: Processadores avançados
        self.advanced_processor = AdvancedFeatureProcessor(self.logger)
        self.feature_selector = IntelligentFeatureSelector(self.logger)
        
        # Cache para evitar recálculos
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
        self.feature_selection_interval = 3600  # Reselecionar features a cada hora
        
        # Mapeamento de features para seus requisitos
        self.feature_dependencies = self._build_feature_dependencies()
    
    def calculate(self, data: TradingDataStructure, 
                 force_recalculate: bool = False,
                 use_advanced: Optional[bool] = None) -> Dict[str, pd.DataFrame]:
        """Calcula todas as features necessárias com processamento avançado"""
        
        if data.candles.empty:
            self.logger.warning("Sem dados de candles para calcular features")
            return {'model_ready': pd.DataFrame()}
        
        # Determina se usa features avançadas
        if use_advanced is None:
            use_advanced = self.use_advanced_features
        
        # Verificar se precisa recalcular
        if not force_recalculate and self._is_cache_valid(data):
            self.logger.info("Usando cache de features")
            return self._get_from_cache(data)
        
        self.logger.info(f"Calculando features para {len(data.candles)} candles")
        
        # Calcular em paralelo se habilitado
        if self.parallel_processing and len(data.candles) > 100:
            result = self._calculate_parallel(data, use_advanced)
        else:
            result = self._calculate_sequential(data, use_advanced)
        
        # Aplicar seleção de features se necessário
        if use_advanced and self._should_select_features():
            result = self._apply_feature_selection(data, result)
        
        # Atualizar cache
        self._update_cache(data, result)
        
        return result
    
    def _calculate_sequential(self, data: TradingDataStructure, 
                            use_advanced: bool = True) -> Dict[str, pd.DataFrame]:
        """Calcula features sequencialmente com opção de processamento avançado"""
        
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
                
                # Mescla features avançadas
                if not advanced_features.empty:
                    # Remove duplicatas antes de concatenar
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
        
        # Preparar DataFrame final para o modelo
        model_ready_df = self._prepare_model_data(data)
        
        return {
            'indicators': data.indicators,
            'features': data.features,
            'model_ready': model_ready_df,
            'all': self._merge_all_features(data)
        }
    
    def _calculate_parallel(self, data: TradingDataStructure,
                          use_advanced: bool = True) -> Dict[str, pd.DataFrame]:
        """Calcula features em paralelo para melhor performance"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submeter tarefas
            futures = {}
            
            # Indicadores técnicos
            futures['indicators'] = executor.submit(
                self.technical.calculate_all, data.candles
            )
            
            # Features avançadas em paralelo se habilitado
            if use_advanced and len(data.candles) > 100:
                futures['advanced'] = executor.submit(
                    self.advanced_processor.extract_all_features,
                    data.candles,
                    data.microstructure,
                    data.orderbook,
                    time.time()
                )
            
            # Aguardar indicadores antes de calcular features compostas
            indicators = futures['indicators'].result()
            data.indicators = indicators
            
            # Features ML (dependem de indicadores)
            futures['features'] = executor.submit(
                self.ml_features.calculate_all,
                data.candles,
                data.microstructure,
                indicators
            )
            
            # Coletar resultados
            data.features = futures['features'].result()
            
            # Coletar features avançadas se calculadas
            if 'advanced' in futures:
                try:
                    advanced_features = futures['advanced'].result()
                    if not advanced_features.empty:
                        # Mescla com features existentes
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
    
    def _should_select_features(self) -> bool:
        """Determina se deve executar seleção de features"""
        
        if self.cache['last_selection_time'] is None:
            return True
        
        time_since_selection = time.time() - self.cache['last_selection_time']
        return time_since_selection > self.feature_selection_interval
    
    def _apply_feature_selection(self, data: TradingDataStructure, 
                               result: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Aplica seleção inteligente de features"""
        
        try:
            all_features = result.get('all', pd.DataFrame())
            
            if all_features.empty or len(all_features.columns) <= 20:
                return result
            
            # Prepara target (retorno futuro)
            if 'close' in data.candles.columns:
                target = data.candles['close'].pct_change().shift(-1).fillna(0)
                
                # Remove features de preço para evitar vazamento
                feature_cols = [col for col in all_features.columns 
                              if col not in ['open', 'high', 'low', 'close', 'volume']]
                
                # Seleciona features ótimas
                selected_df = self.feature_selector.select_optimal_features(
                    all_features[feature_cols],
                    target,
                    max_features=15
                )
                
                # Atualiza resultado com features selecionadas
                if self.feature_selector.selected_features:
                    result['selected_features'] = selected_df
                    
                    # Se temos model_features específicas, garantir que estão incluídas
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
                self.logger.info(f"Seleção de features concluída: {len(selected_df.columns)} features")
                
        except Exception as e:
            self.logger.error(f"Erro na seleção de features: {e}")
        
        return result
    
    def _prepare_model_data(self, data: TradingDataStructure) -> pd.DataFrame:
        """Prepara DataFrame com apenas as features necessárias para o modelo"""
        if not self.model_features:
            # Se não há features específicas, retornar tudo
            return self._merge_all_features(data)
        
        # Mapear features para suas fontes
        feature_sources = {
            'candles': data.candles,
            'indicators': data.indicators,
            'features': data.features,
            'microstructure': data.microstructure
        }
        
        # Coletar features por fonte
        features_by_source = {}
        missing_features = []
        
        for feature in self.model_features:
            found = False
            
            # Procurar em cada fonte
            for source_name, source_df in feature_sources.items():
                if source_df is not None and not source_df.empty and feature in source_df.columns:
                    if source_name not in features_by_source:
                        features_by_source[source_name] = []
                    features_by_source[source_name].append(feature)
                    found = True
                    break
            
            if not found:
                missing_features.append(feature)
        
        # Criar lista de DataFrames para concatenar
        dfs_to_concat = []
        
        # Adicionar features de cada fonte
        for source_name, feature_list in features_by_source.items():
            source_df = feature_sources[source_name]
            if source_df is not None and feature_list:
                dfs_to_concat.append(source_df[feature_list])
        
        # Concatenar todos os DataFrames
        if dfs_to_concat:
            model_data = pd.concat(dfs_to_concat, axis=1)
        else:
            # Se não há dados, criar DataFrame vazio com índice dos candles
            model_data = pd.DataFrame(index=data.candles.index)
        
        # Adicionar features ausentes com valores default
        if missing_features:
            self.logger.warning(f"Features não encontradas: {missing_features[:5]}...")
            # Criar DataFrame com colunas faltantes
            missing_df = pd.DataFrame(
                0, 
                index=model_data.index,
                columns=missing_features
            )
            model_data = pd.concat([model_data, missing_df], axis=1)
        
        # Garantir ordem correta das features
        if self.model_features:
            available_features = [f for f in self.model_features if f in model_data.columns]
            model_data = model_data[available_features]
        
        # Preencher NaN com forward fill e depois com 0
        model_data = model_data.ffill().fillna(0)
        
        # Remover linhas iniciais com muitos zeros (warm-up period)
        if len(model_data) > self.min_candles_required:
            # Encontrar primeira linha com dados válidos
            valid_mask = (model_data != 0).any(axis=1)
            if valid_mask.any():
                first_valid = valid_mask.idxmax()
                first_valid_pos = model_data.index.get_loc(first_valid)
                # Garantir que first_valid_pos seja inteiro e não slice
                if isinstance(first_valid_pos, (np.ndarray, list)):
                    first_valid_pos = int(first_valid_pos[0])
                elif isinstance(first_valid_pos, slice):
                    # Se for slice, usar o início do slice ou 0
                    first_valid_pos = first_valid_pos.start if first_valid_pos.start is not None else 0
                else:
                    first_valid_pos = int(first_valid_pos)
                start_pos = max(0, first_valid_pos - 20)  # Manter 20 candles antes
                model_data = model_data.iloc[start_pos:]
        
        self.logger.info(f"DataFrame do modelo preparado: {model_data.shape}")
        
        return model_data
    
    def _merge_all_features(self, data: TradingDataStructure) -> pd.DataFrame:
        """Mescla todas as features em um único DataFrame"""
        # Lista de DataFrames para concatenar
        dfs_to_merge = [data.candles.copy()]
        
        # Adicionar indicadores
        if data.indicators is not None and not data.indicators.empty:
            # Remover colunas duplicadas
            cols_to_add = [col for col in data.indicators.columns 
                          if col not in data.candles.columns]
            if cols_to_add:
                dfs_to_merge.append(data.indicators[cols_to_add])
        
        # Adicionar features ML
        if data.features is not None and not data.features.empty:
            # Remover colunas duplicadas
            existing_cols = set(data.candles.columns)
            if data.indicators is not None:
                existing_cols.update(data.indicators.columns)
            cols_to_add = [col for col in data.features.columns 
                          if col not in existing_cols]
            if cols_to_add:
                dfs_to_merge.append(data.features[cols_to_add])
        
        # Concatenar todos os DataFrames de uma vez
        all_features = pd.concat(dfs_to_merge, axis=1)
        
        # Adicionar microestrutura se disponível (requer alinhamento de índice)
        if data.microstructure is not None and not data.microstructure.empty:
            # Alinhar índices
            common_index = all_features.index.intersection(data.microstructure.index)
            if len(common_index) > 0:
                # Preparar DataFrame alinhado de microestrutura
                micro_aligned = data.microstructure.loc[common_index].copy()
                
                # Remover colunas duplicadas
                cols_to_add = [col for col in micro_aligned.columns 
                              if col not in all_features.columns]
                
                if cols_to_add:
                    # Criar DataFrame vazio com mesmo índice que all_features
                    micro_full = pd.DataFrame(index=all_features.index)
                    
                    # Preencher apenas onde há dados
                    for col in cols_to_add:
                        micro_full[col] = pd.Series(
                            micro_aligned[col].values,
                            index=common_index
                        )
                    
                    # Concatenar
                    all_features = pd.concat([all_features, micro_full], axis=1)
        
        return all_features
    
    def _build_feature_dependencies(self) -> Dict[str, Set[str]]:
        """Constrói mapa de dependências entre features"""
        dependencies = {
            # Indicadores básicos
            'rsi': {'close'},
            'macd': {'close'},
            'macd_signal': {'macd'},
            'macd_hist': {'macd', 'macd_signal'},
            
            # EMAs
            **{f'ema_{p}': {'close'} for p in [9, 20, 50, 200]},
            
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
            
            # Novas features avançadas
            'order_flow_imbalance_1m': {'buy_volume', 'sell_volume'},
            'order_flow_imbalance_5m': {'buy_volume', 'sell_volume'},
            'adaptive_rsi': {'close'},
            'dynamic_macd': {'close', 'volume'},
            'adaptive_bb_position': {'close'}
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
        """Verifica se o cache ainda é válido"""
        if self.cache['indicators'] is None or self.cache['features'] is None:
            return False
        
        # Verificar se os candles mudaram
        if len(data.candles) != self.cache['last_candle_count']:
            return False
        
        if not data.candles.empty:
            last_time = data.candles.index[-1]
            if last_time != self.cache['last_candle_time']:
                return False
        
        return True
    
    def _update_cache(self, data: TradingDataStructure, result: Dict[str, pd.DataFrame]):
        """Atualiza o cache com os resultados calculados"""
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
    
    def calculate_specific_features(self, 
                                  data: TradingDataStructure,
                                  feature_list: List[str]) -> pd.DataFrame:
        """Calcula apenas features específicas para otimização"""
        # Identificar dependências
        required_features = self.get_required_features_for_models(feature_list)
        
        # Separar por tipo
        indicator_features = [f for f in required_features if any(
            ind in f for ind in ['ema', 'sma', 'rsi', 'macd', 'bb_', 'stoch', 'atr', 'adx']
        )]
        
        # Calcular apenas o necessário
        if indicator_features:
            data.indicators = self.technical.calculate_specific(data.candles, indicator_features)
        
        # ML features sempre precisam ser calculadas se solicitadas
        ml_feature_list = [f for f in required_features if f not in indicator_features]
        if ml_feature_list:
            temp_ml_features = MLFeatures(ml_feature_list)
            data.features = temp_ml_features.calculate_all(
                data.candles, data.microstructure, data.indicators
            )
        
        # Retornar apenas as solicitadas
        return self._prepare_model_data(data)[feature_list]
    
    def get_feature_statistics(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Calcula estatísticas das features para análise"""
        stats = pd.DataFrame()
        
        for col in features_df.columns:
            stats.loc[col, 'mean'] = features_df[col].mean()
            stats.loc[col, 'std'] = features_df[col].std()
            stats.loc[col, 'min'] = features_df[col].min()
            stats.loc[col, 'max'] = features_df[col].max()
            stats.loc[col, 'nulls'] = features_df[col].isnull().sum()
            stats.loc[col, 'zeros'] = (features_df[col] == 0).sum()
            stats.loc[col, 'unique'] = features_df[col].nunique()
        
        return stats
    
    # Novos métodos para funcionalidades avançadas
    
    def get_features_for_prediction(self, data: TradingDataStructure,
                                  lookback: int = 100, 
                                  force_recalculate: bool = False) -> pd.DataFrame:
        """Obtém features otimizadas para predição com cache inteligente"""
        
        # Calcula se necessário
        if force_recalculate or not self._is_cache_valid(data):
            self.calculate(data, use_advanced=True)
        
        # Retorna features selecionadas ou todas
        if hasattr(self.cache, 'selected_features') and self.cache.get('selected_features') is not None:
            return self.cache['selected_features'].iloc[-lookback:]
        else:
            model_data = self._prepare_model_data(data)
            return model_data.iloc[-lookback:]
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Retorna importância das features para análise"""
        
        if not self.feature_selector.feature_scores:
            return {
                'status': 'No feature selection performed yet',
                'top_features': [],
                'selected_count': 0,
                'total_features': 0
            }
        
        # Ordena por importância
        sorted_importance = sorted(
            self.feature_selector.feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'status': 'Feature importance available',
            'top_features': sorted_importance[:10],
            'selected_count': len(self.feature_selector.selected_features or []),
            'total_features': len(self.feature_selector.feature_scores),
            'selection_time': self.cache.get('last_selection_time', 0)
        }
    
    def enable_advanced_features(self, enabled: bool = True):
        """Habilita ou desabilita processamento avançado de features"""
        self.use_advanced_features = enabled
        self.logger.info(f"Features avançadas {'habilitadas' if enabled else 'desabilitadas'}")
    
    def set_feature_selection_interval(self, seconds: int):
        """Define intervalo para reseleção automática de features"""
        self.feature_selection_interval = seconds
        self.logger.info(f"Intervalo de seleção de features: {seconds} segundos")