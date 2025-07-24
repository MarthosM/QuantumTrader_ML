# src/training/feature_pipeline.py
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor

import warnings

warnings.filterwarnings('ignore', message='.*TA-Lib.*')

# Importar sistemas robustos
from .robust_technical_indicators import RobustTechnicalIndicators
from .robust_nan_handler import RobustNaNHandler

# Check TA-Lib availability
try:
    import talib
    HAS_TALIB = True
    logger = logging.getLogger(__name__)
    logger.info("📊 TA-Lib disponível - usando implementações nativas")
except ImportError:
    HAS_TALIB = False
    logger = logging.getLogger(__name__)
    logger.info("🔧 Sistema Robusto: Indicadores Técnicos + Tratamento NaN ativado")

class FeatureEngineeringPipeline:
    """Pipeline otimizado de feature engineering para treinamento"""
    
    def __init__(self, n_jobs: int = -1):
        self.logger = logging.getLogger(__name__)
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        
        # Inicializar sistemas robustos
        self.robust_indicators = RobustTechnicalIndicators()
        self.nan_handler = RobustNaNHandler()
        
        self.logger.info("🚀 Pipeline Robusto de Features inicializado")
        
        # Reutilizar componentes existentes com importações robustas
        try:
            from src.feature_engine import FeatureEngine
            from src.technical_indicators import TechnicalIndicators
            from src.ml_features import MLFeatures
        except ImportError:
            try:
                from feature_engine import FeatureEngine
                from technical_indicators import TechnicalIndicators
                from ml_features import MLFeatures
            except ImportError:
                # Mock classes para desenvolvimento
                self.logger.warning("Usando classes mock para componentes de features")
                
                class MockFeatureEngine:
                    def __init__(self): pass
                    def calculate(self, *args, **kwargs): return {}
                    
                class MockTechnicalIndicators:
                    def __init__(self): pass
                    def calculate_all(self, *args, **kwargs): return {}
                    
                class MockMLFeatures:
                    def __init__(self): pass
                    def calculate_all(self, *args, **kwargs): return {}
                
                FeatureEngine = MockFeatureEngine
                TechnicalIndicators = MockTechnicalIndicators
                MLFeatures = MockMLFeatures
        
        self.feature_engine = FeatureEngine()
        self.technical_indicators = TechnicalIndicators()
        self.ml_features = MLFeatures()
        
        # Cache de features calculadas
        self.feature_cache = {}
        
    def create_training_features(self, data: pd.DataFrame,
                               feature_groups: Optional[List[str]] = None,
                               parallel: bool = True) -> pd.DataFrame:
        """
        🚀 VERSÃO ROBUSTA - Cria features usando sistema robusto
        Resolve todos os problemas de NaN e TA-Lib identificados
        """
        self.logger.info("� Sistema Robusto: Indicadores Técnicos + Tratamento NaN ativado")
        self.logger.info("�🚀 Usando Pipeline Robusto de Features v2.0")
        
        try:
            # Importar e usar pipeline robusto
            from .robust_feature_pipeline import RobustFeaturePipeline
            
            robust_pipeline = RobustFeaturePipeline(n_jobs=self.n_jobs or os.cpu_count() or 1)
            features = robust_pipeline.create_features(data, feature_groups)
            
            self.logger.info(f"✅ Pipeline robusto concluído: {len(features.columns)} features")
            
            return features
            
        except Exception as e:
            # Log detalhado do erro para debug
            self.logger.error(f"❌ Erro no pipeline robusto: {type(e).__name__}: {e}")
            self.logger.warning(f"⚠️ Pipeline robusto falhou ({e}), usando implementação básica")
            
            return self._create_features_fallback(data, feature_groups or ['technical', 'momentum', 'volatility', 'microstructure', 'patterns', 'entry_exit'], parallel)
    
    def _create_features_fallback(self, data: pd.DataFrame,
                                feature_groups: Optional[List[str]] = None,
                                parallel: bool = True) -> pd.DataFrame:
        """
        Implementação fallback com correções básicas para NaN e TA-Lib
        """
        if feature_groups is None:
            feature_groups = ['technical', 'momentum', 'volatility', 'microstructure', 'patterns', 'entry_exit']
            
        self.logger.info(f"Criando features (fallback) para {len(data)} amostras")
        
        # DataFrame para armazenar features - usar as colunas disponíveis
        base_columns = ['open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in base_columns if col in data.columns]
        
        if not available_columns:
            raise ValueError(f"Nenhuma coluna base encontrada. Colunas disponíveis: {data.columns.tolist()}")
        
        self.logger.info(f"Usando colunas base: {available_columns}")
        features_df = data[available_columns].copy()
        
        # Se não temos 'close', mas temos 'preco' ou similar, mapear
        if 'close' not in features_df.columns:
            if 'preco' in data.columns:
                features_df['close'] = data['preco']
                self.logger.info("Mapeando coluna 'preco' para 'close'")
            elif 'last' in data.columns:
                features_df['close'] = data['last']
                self.logger.info("Mapeando coluna 'last' para 'close'")
        
        # Calcular features em paralelo ou sequencial
        if parallel and len(data) > 10000:
            features_df = self._parallel_feature_calculation(features_df, feature_groups)
        else:
            features_df = self._sequential_feature_calculation(features_df, feature_groups)
        
        # CORREÇÃO: Tratar NaN de forma básica
        self.logger.info("Aplicando tratamento básico de NaN")
        
        # Contar NaN antes
        nan_before = features_df.isna().sum().sum()
        
        # Aplicar forward fill limitado e depois backward fill
        features_df = features_df.ffill(limit=5)
        features_df = features_df.bfill(limit=5)
        
        # Remover features com muitos NaN (>50%)
        nan_threshold = len(features_df) * 0.5
        high_nan_cols = []
        for col in features_df.columns:
            if features_df[col].isna().sum() > nan_threshold:
                high_nan_cols.append(col)
        
        if high_nan_cols:
            features_df = features_df.drop(columns=high_nan_cols)
            self.logger.warning(f"Removidas {len(high_nan_cols)} features com >50% NaN")
        
        # Remover linhas restantes com NaN
        initial_rows = len(features_df)
        features_df = features_df.dropna()
        final_rows = len(features_df)
        
        if initial_rows != final_rows:
            self.logger.info(f"Linhas finais após limpeza: {final_rows}/{initial_rows}")
        
        nan_after = features_df.isna().sum().sum()
        self.logger.info(f"NaN: {nan_before} → {nan_after}")
        
        return features_df
        
        # DataFrame para armazenar features - usar as colunas disponíveis
        base_columns = ['open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in base_columns if col in data.columns]
        
        if not available_columns:
            raise ValueError(f"Nenhuma coluna base encontrada. Colunas disponíveis: {data.columns.tolist()}")
        
        self.logger.info(f"Usando colunas base: {available_columns}")
        features_df = data[available_columns].copy()
        
        # Se não temos 'close', mas temos 'preco' ou similar, mapear
        if 'close' not in features_df.columns:
            if 'preco' in data.columns:
                features_df['close'] = data['preco']
                self.logger.info("Mapeando coluna 'preco' para 'close'")
            elif 'last' in data.columns:
                features_df['close'] = data['last']
                self.logger.info("Mapeando coluna 'last' para 'close'")
        
        # Calcular features em paralelo ou sequencial
        if parallel and len(data) > 10000:
            features_df = self._parallel_feature_calculation(features_df, feature_groups)
        else:
            features_df = self._sequential_feature_calculation(features_df, feature_groups)
        
        # Adicionar features compostas
        features_df = self._add_composite_features(features_df)
        
        # Adicionar lags importantes
        features_df = self._add_lag_features(features_df, lags=[1, 5, 10, 20])
        
        # Adicionar features de regime
        features_df = self._add_regime_features(features_df)
        
        # Validar features
        self._validate_features(features_df)
        
        self.logger.info(f"Criadas {len(features_df.columns)} features")
        
        return features_df
    
    def _sequential_feature_calculation(self, data: pd.DataFrame, 
                                      feature_groups: List[str]) -> pd.DataFrame:
        """Cálculo sequencial de features"""
        features_df = data.copy()
        
        for group in feature_groups:
            self.logger.info(f"Calculando grupo de features: {group}")
            
            if group == 'technical':
                # Indicadores técnicos
                tech_features = self._calculate_technical_features(data)
                features_df = pd.concat([features_df, tech_features], axis=1)
                
            elif group == 'momentum':
                # Features de momentum
                momentum_features = self._calculate_momentum_features(data)
                features_df = pd.concat([features_df, momentum_features], axis=1)
                
            elif group == 'volatility':
                # Features de volatilidade
                vol_features = self._calculate_volatility_features(data)
                features_df = pd.concat([features_df, vol_features], axis=1)
                
            elif group == 'microstructure':
                # Features de microestrutura
                try:
                    micro_features = self._calculate_microstructure_features(data)
                    features_df = pd.concat([features_df, micro_features], axis=1)
                except Exception as e:
                    self.logger.warning(f"Erro calculando features de microestrutura: {e}")
                    
            elif group == 'patterns':
                # Padrões de preço
                try:
                    pattern_features = self._calculate_pattern_features(data)
                    features_df = pd.concat([features_df, pattern_features], axis=1)
                except Exception as e:
                    self.logger.warning(f"Erro calculando features de padrões: {e}")

            elif group == 'entry_exit':
                # Features de entrada/saída
                try:
                    entry_exit_features = self._calculate_entry_exit_features(data)
                    features_df = pd.concat([features_df, entry_exit_features], axis=1)
                except Exception as e:
                    self.logger.warning(f"Erro calculando features de entrada/saída: {e}")
                
        return features_df
    
    def _parallel_feature_calculation(self, data: pd.DataFrame,
                                    feature_groups: List[str]) -> pd.DataFrame:
        """Cálculo paralelo de features para grandes datasets"""
        # Dividir dados em chunks - garantir que n_chunks seja um inteiro válido
        n_chunks = self.n_jobs
        if n_chunks is None or n_chunks <= 0:
            n_chunks = 1
        
        chunk_size = len(data) // n_chunks
        chunks = []
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < n_chunks - 1 else len(data)
            
            # Adicionar overlap para indicadores que precisam de lookback
            lookback = 200  # Máximo lookback necessário
            chunk_start = max(0, start_idx - lookback)
            chunk = data.iloc[chunk_start:end_idx].copy()
            chunks.append((chunk, start_idx, end_idx))
        
        # Processar em paralelo
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for chunk_data, start, end in chunks:
                future = executor.submit(
                    self._process_chunk, chunk_data, feature_groups, start, end
                )
                futures.append(future)
            
            # Coletar resultados
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5 minutos timeout
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Erro processando chunk: {e}")
            
        # Concatenar resultados
        if results:
            final_features = pd.concat(results, axis=0)
        else:
            self.logger.warning("Nenhum resultado de chunks, usando processamento sequencial")
            final_features = self._process_chunk(data, feature_groups, 0, len(data))
            
        return final_features

    def _process_chunk(self, chunk_data: pd.DataFrame, feature_groups: List[str], 
                       start_idx: int, end_idx: int) -> pd.DataFrame:
        """Processa um chunk de dados para feature engineering"""
        chunk_features = pd.DataFrame(index=chunk_data.index)
        
        # Processar cada grupo de features
        for group in feature_groups:
            try:
                if group == 'technical':
                    tech_features = self._calculate_technical_features(chunk_data)
                    chunk_features = pd.concat([chunk_features, tech_features], axis=1)
                    
                elif group == 'momentum':
                    mom_features = self._calculate_momentum_features(chunk_data)
                    chunk_features = pd.concat([chunk_features, mom_features], axis=1)
                    
                elif group == 'volatility':
                    vol_features = self._calculate_volatility_features(chunk_data)
                    chunk_features = pd.concat([chunk_features, vol_features], axis=1)
                    
                elif group == 'microstructure':
                    micro_features = self._calculate_microstructure_features(chunk_data)
                    chunk_features = pd.concat([chunk_features, micro_features], axis=1)
                    
            except Exception as e:
                self.logger.error(f"Erro calculando features {group} no chunk {start_idx}-{end_idx}: {e}")
                
        return chunk_features
    
    def _calculate_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos com correções para evitar NaN"""
        tech_features = pd.DataFrame(index=data.index)
        
        try:
            self.logger.info("🔧 Calculando indicadores técnicos com correções anti-NaN")
            
            # EMAs com min_periods para evitar NaN no início
            for period in [9, 20, 50, 200]:
                min_periods = max(1, period // 2)  # Usar metade do período como mínimo
                tech_features[f'ema_{period}'] = data['close'].ewm(
                    span=period, min_periods=min_periods
                ).mean()
            
            # RSI com cálculo robusto
            delta = data['close'].diff()
            # Garantir que delta seja numérico para comparações
            delta = pd.to_numeric(delta, errors='coerce')
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            for period in [9, 14, 25]:
                min_periods = max(1, period // 2)
                avg_gain = gain.ewm(span=period, min_periods=min_periods).mean()
                avg_loss = loss.ewm(span=period, min_periods=min_periods).mean()
                
                # Evitar divisão por zero
                rs = avg_gain / (avg_loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                
                col_name = 'rsi' if period == 14 else f'rsi_{period}'
                tech_features[col_name] = rsi
            
            # MACD simplificado com min_periods
            ema_12 = data['close'].ewm(span=12, min_periods=6).mean()
            ema_26 = data['close'].ewm(span=26, min_periods=13).mean()
            tech_features['macd'] = ema_12 - ema_26
            tech_features['macd_signal'] = tech_features['macd'].ewm(span=9, min_periods=5).mean()
            tech_features['macd_hist'] = tech_features['macd'] - tech_features['macd_signal']
            
            # Bollinger Bands com min_periods
            for period in [20, 50]:
                min_periods = max(1, period // 2)
                sma = data['close'].rolling(window=period, min_periods=min_periods).mean()
                std = data['close'].rolling(window=period, min_periods=min_periods).std()
                
                tech_features[f'bb_upper_{period}'] = sma + (2 * std)
                tech_features[f'bb_middle_{period}'] = sma
                tech_features[f'bb_lower_{period}'] = sma - (2 * std)
                tech_features[f'bb_width_{period}'] = 4 * std  # Simplificado
                
                # Position com proteção contra divisão por zero
                width = tech_features[f'bb_width_{period}']
                position = (data['close'] - tech_features[f'bb_lower_{period}']) / (width + 1e-10)
                tech_features[f'bb_position_{period}'] = position.clip(0, 1)
            
            # ATR com min_periods
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift()).abs()
            low_close = (data['low'] - data['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            tech_features['atr'] = tr.rolling(window=14, min_periods=7).mean()
            tech_features['atr_20'] = tr.rolling(window=20, min_periods=10).mean()
            
            # ADX simplificado
            tech_features['adx'] = abs(data['close'] - data['close'].shift()).rolling(
                window=14, min_periods=7
            ).mean()
            
            self.logger.info(f"✅ Calculados {len(tech_features.columns)} indicadores técnicos")
            return tech_features
            
        except Exception as e:
            self.logger.error(f"❌ Erro calculando indicadores técnicos: {e}")
            # Retornar DataFrame vazio com índice correto em caso de erro
            return pd.DataFrame(index=data.index)
    
    def _calculate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de momentum"""
        mom_features = pd.DataFrame(index=data.index)
        
        # ROC (Rate of Change) - calculado manualmente se talib não disponível
        for period in [5, 10, 20]:
            try:
                if HAS_TALIB:
                    import talib as ta # type: ignore
                    mom_features[f'roc_{period}'] = ta.ROC(data['close'], timeperiod=period)
                else:
                    raise ImportError("TA-Lib not available")
            except (ImportError, Exception):
                # ROC manual: ((close - close_n_periods_ago) / close_n_periods_ago) * 100
                mom_features[f'roc_{period}'] = ((data['close'] - data['close'].shift(period)) / (data['close'].shift(period) + 1e-10)) * 100
        
        # Momentum básico
        for period in [5, 10, 20]:
            mom_features[f'momentum_{period}'] = data['close'] - data['close'].shift(period)
        
        # Returns
        for period in [5, 10, 20, 50]:
            mom_features[f'return_{period}'] = data['close'].pct_change(period) * 100
            
        # Volume momentum
        if 'volume' in data.columns:
            for period in [5, 10, 20]:
                mom_features[f'volume_momentum_{period}'] = (
                    data['volume'].rolling(period).mean() / 
                    data['volume'].rolling(period * 2).mean()
                )
        
        # Acceleration
        if 'momentum_5' in mom_features.columns:
            mom_features['price_acceleration'] = mom_features['momentum_5'].diff()
        
        return mom_features
                    
    def _calculate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de volatilidade"""
        vol_features = pd.DataFrame(index=data.index)
        
        # Volatilidade histórica
        for period in [5, 10, 20, 50]:
            returns = data['close'].pct_change()
            vol_features[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252)
        
        # Parkinson volatility
        for period in [10, 20]:
            hl_ratio = np.log(data['high'] / data['low'])
            vol_features[f'parkinson_vol_{period}'] = (
                pd.Series(hl_ratio, index=data.index).rolling(period).apply(lambda x: np.sqrt(np.sum(x**2) / (4 * period * np.log(2))))
            )
        
        # True Range - calculado manualmente se talib não disponível
        try:
            if HAS_TALIB:
                import talib as ta # type: ignore
                vol_features['true_range'] = ta.TRANGE(data['high'], data['low'], data['close'])
            else:
                raise ImportError("TA-Lib not available")
        except (ImportError, Exception):
            # True Range manual
            hl = data['high'] - data['low']
            hc = np.abs(data['high'] - data['close'].shift(1))
            lc = np.abs(data['low'] - data['close'].shift(1))
            vol_features['true_range'] = np.maximum(hl, np.maximum(hc, lc))
        
        # High-Low Range
        for period in [5, 10, 20]:
            vol_features[f'high_low_range_{period}'] = (
                data['high'].rolling(period).max() - 
                data['low'].rolling(period).min()
            ) / data['close'] * 100
        
        # Volatility ratios
        if 'volatility_10' in vol_features.columns and 'volatility_20' in vol_features.columns:
            vol_features['volatility_ratio'] = vol_features['volatility_10'] / (vol_features['volatility_20'] + 1e-10)
        
        return vol_features
    
    def _calculate_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de microestrutura de mercado"""
        micro_features = pd.DataFrame(index=data.index)
        
        # Verifica se as colunas necessárias existem
        if 'buy_volume' not in data.columns or 'sell_volume' not in data.columns:
            # Features alternativas usando dados disponíveis
            if 'volume' in data.columns:
                # Volume momentum
                for period in [5, 10, 20]:
                    micro_features[f'volume_sma_{period}'] = data['volume'].rolling(period).mean()
                    micro_features[f'volume_ratio_{period}'] = (
                        data['volume'] / micro_features[f'volume_sma_{period}']
                    )
                
                # Price-volume features
                micro_features['price_volume_trend'] = (
                    data['close'].pct_change() * data['volume']
                ).rolling(10).sum()
                
                # Volume weighted price
                micro_features['vwap'] = (
                    (data['close'] * data['volume']).rolling(20).sum() / 
                    data['volume'].rolling(20).sum()
                )
            
            return micro_features
        
        # Buy/Sell pressure (original)
        total_volume = data['buy_volume'] + data['sell_volume']
        micro_features['buy_pressure'] = data['buy_volume'] / total_volume
        micro_features['sell_pressure'] = data['sell_volume'] / total_volume
        
        # Volume imbalance
        micro_features['volume_imbalance'] = (
            (data['buy_volume'] - data['sell_volume']) / total_volume
        )
        
        # Flow metrics
        for period in [5, 10, 20]:
            micro_features[f'buy_flow_{period}'] = data['buy_volume'].rolling(period).sum()
            micro_features[f'sell_flow_{period}'] = data['sell_volume'].rolling(period).sum()
            micro_features[f'net_flow_{period}'] = (
                micro_features[f'buy_flow_{period}'] - micro_features[f'sell_flow_{period}']
            )
            micro_features[f'flow_ratio_{period}'] = (
                micro_features[f'buy_flow_{period}'] / micro_features[f'sell_flow_{period}']
            )
        
        # Trade intensity
        if 'trades' in data.columns:
            micro_features['avg_trade_size'] = data['volume'] / data['trades']
            for period in [5, 10]:
                micro_features[f'trade_intensity_{period}'] = (
                    data['trades'].rolling(period).mean()
                )
        
        # VWAP deviation
        if 'vwap' in data.columns:
            micro_features['vwap_deviation'] = (data['close'] - data['vwap']) / data['vwap'] * 100
            micro_features['vwap_position'] = np.where(
                data['close'] > data['vwap'], 1, -1
            )

        # Order flow momentum
        if 'buy_volume' in data.columns and 'sell_volume' in data.columns:
            # Order flow acceleration
            micro_features['order_flow_acceleration'] = micro_features['volume_imbalance'].diff()
            
            # Cumulative volume delta
            micro_features['cvd'] = (data['buy_volume'] - data['sell_volume']).cumsum()
            micro_features['cvd_normalized'] = micro_features['cvd'] / data['volume'].cumsum()
            
            # Volume concentration
            for period in [5, 10]:
                total_vol = data['volume'].rolling(period).sum()
                micro_features[f'volume_concentration_{period}'] = (
                    data['volume'].rolling(period).max() / total_vol
                )

        # Spread estimation (if we have trades data)
        if 'trades' in data.columns and 'volume' in data.columns:
            micro_features['effective_spread_proxy'] = (
                2 * np.sqrt(np.abs(data['close'].pct_change())) * data['close']
            )

        return micro_features

    def _calculate_entry_exit_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features específicas para detectar pontos de entrada e saída"""
        entry_features = pd.DataFrame(index=data.index)
        
        # Swing highs and lows
        for period in [5, 10, 20]:
            entry_features[f'swing_high_{period}'] = (
                data['high'] == data['high'].rolling(period*2+1, center=True).max()
            ).astype(int)
            entry_features[f'swing_low_{period}'] = (
                data['low'] == data['low'].rolling(period*2+1, center=True).min()
            ).astype(int)
        
        # Price breakouts
        for period in [20, 50]:
            if f'resistance_{period}' in data.columns:
                entry_features[f'breakout_up_{period}'] = (
                    (data['close'] > data[f'resistance_{period}']) & 
                    (data['close'].shift(1) <= data[f'resistance_{period}'].shift(1))
                ).astype(int)
            
            if f'support_{period}' in data.columns:
                entry_features[f'breakout_down_{period}'] = (
                    (data['close'] < data[f'support_{period}']) & 
                    (data['close'].shift(1) >= data[f'support_{period}'].shift(1))
                ).astype(int)
        
        # Momentum divergence
        if 'rsi' in data.columns:
            # Price making new high but RSI not
            price_high = data['high'] == data['high'].rolling(14).max()
            rsi_high = data['rsi'] == data['rsi'].rolling(14).max()
            entry_features['bearish_divergence'] = (price_high & ~rsi_high).astype(int)
            
            # Price making new low but RSI not
            price_low = data['low'] == data['low'].rolling(14).min()
            rsi_low = data['rsi'] == data['rsi'].rolling(14).min()
            entry_features['bullish_divergence'] = (price_low & ~rsi_low).astype(int)
        
        # Volume climax
        if 'volume' in data.columns:
            vol_mean = data['volume'].rolling(20).mean()
            vol_std = data['volume'].rolling(20).std()
            entry_features['volume_climax'] = (
                data['volume'] > (vol_mean + 2 * vol_std)
            ).astype(int)
        
        # Price rejection at levels
        upper_wick_calc = (
            (data['high'] - np.maximum(data['open'], data['close'])) / 
            (data['high'] - data['low'])
        )
        entry_features['upper_wick_ratio'] = pd.Series(upper_wick_calc, index=data.index).fillna(0)
        
        lower_wick_calc = (
            (np.minimum(data['open'], data['close']) - data['low']) / 
            (data['high'] - data['low'])
        )
        entry_features['lower_wick_ratio'] = pd.Series(lower_wick_calc, index=data.index).fillna(0)
        
        # Momentum exhaustion
        if all(f'momentum_{p}' in data.columns for p in [5, 10, 20]):
            entry_features['momentum_exhaustion'] = (
                (np.abs(data['momentum_5']) < np.abs(data['momentum_10'])) &
                (np.abs(data['momentum_10']) < np.abs(data['momentum_20']))
            ).astype(int)
        
        return entry_features
    
    def _calculate_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features baseadas em padrões de preço"""
        pattern_features = pd.DataFrame(index=data.index)
        
        # Candlestick patterns - usando fallbacks se TA-Lib não estiver disponível
        try:
            if HAS_TALIB:
                import talib as ta  # type: ignore
                pattern_features['doji'] = ta.CDLDOJI(data['open'], data['high'], data['low'], data['close'])
                pattern_features['hammer'] = ta.CDLHAMMER(data['open'], data['high'], data['low'], data['close'])
                pattern_features['engulfing'] = ta.CDLENGULFING(data['open'], data['high'], data['low'], data['close'])
                pattern_features['harami'] = ta.CDLHARAMI(data['open'], data['high'], data['low'], data['close'])
                pattern_features['morning_star'] = ta.CDLMORNINGSTAR(data['open'], data['high'], data['low'], data['close'])
                pattern_features['evening_star'] = ta.CDLEVENINGSTAR(data['open'], data['high'], data['low'], data['close'])
            else:
                raise ImportError("TA-Lib not available")
        except (ImportError, Exception):
            # Fallbacks simples para padrões
            # Doji - preço de abertura próximo do fechamento
            pattern_features['doji'] = np.where(
                np.abs(data['close'] - data['open']) <= (data['high'] - data['low']) * 0.1, 100, 0
            )
            
            # Hammer simplificado - corpo pequeno na parte inferior
            body_size = np.abs(data['close'] - data['open'])
            lower_shadow = np.minimum(data['open'], data['close']) - data['low']
            pattern_features['hammer'] = np.where(
                (lower_shadow > 2 * body_size) & (body_size > 0), 100, 0
            )
            
            # Outros padrões como 0 por simplicidade
            for pattern in ['engulfing', 'harami', 'morning_star', 'evening_star']:
                pattern_features[pattern] = 0
        
        # Support/Resistance levels
        for period in [20, 50]:
            pattern_features[f'resistance_{period}'] = data['high'].rolling(period).max()
            pattern_features[f'support_{period}'] = data['low'].rolling(period).min()
            pattern_features[f'sr_range_{period}'] = (
                pattern_features[f'resistance_{period}'] - pattern_features[f'support_{period}']
            )
            pattern_features[f'price_to_resistance_{period}'] = (
                pattern_features[f'resistance_{period}'] - data['close']
            ) / data['close'] * 100
            pattern_features[f'price_to_support_{period}'] = (
                data['close'] - pattern_features[f'support_{period}']
            ) / data['close'] * 100
        
        # Pivot points
        pattern_features['pivot'] = (data['high'] + data['low'] + data['close']) / 3
        pattern_features['r1'] = 2 * pattern_features['pivot'] - data['low']
        pattern_features['s1'] = 2 * pattern_features['pivot'] - data['high']
        
        return pattern_features
   
    def _add_composite_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features compostas e interações"""
        # RSI extremes
        if 'rsi' in features_df.columns:
            features_df['rsi_oversold'] = (features_df['rsi'] < 30).astype(int)
            features_df['rsi_overbought'] = (features_df['rsi'] > 70).astype(int)
        
        # MACD cross
        if 'macd' in features_df.columns and 'macd_signal' in features_df.columns:
            features_df['macd_cross'] = np.where(
                features_df['macd'] > features_df['macd_signal'], 1, -1
            )
            features_df['macd_cross_change'] = features_df['macd_cross'].diff()
        
        # EMA alignments
        if all(f'ema_{p}' in features_df.columns for p in [9, 20, 50]) and 'close' in features_df.columns:
            features_df['ema_alignment'] = (
                (features_df['ema_9'] > features_df['ema_20']).astype(int) +
                (features_df['ema_20'] > features_df['ema_50']).astype(int)
            ) - 1  # -1, 0, 1
            
            # Price vs EMAs
            features_df['price_above_emas'] = (
                (features_df['close'] > features_df['ema_9']).astype(int) +
                (features_df['close'] > features_df['ema_20']).astype(int) +
                (features_df['close'] > features_df['ema_50']).astype(int)
            ) / 3
        
        # Volatility regime
        if 'volatility_20' in features_df.columns:
            vol_median = features_df['volatility_20'].rolling(100).median()
            features_df['high_volatility_regime'] = (
                features_df['volatility_20'] > 1.5 * vol_median
            ).astype(int)
        
        # Momentum strength
        if all(f'momentum_{p}' in features_df.columns for p in [5, 10, 20]) and 'close' in features_df.columns:
            features_df['momentum_strength'] = (
                features_df['momentum_5'] / features_df['close'] * 0.5 +
                features_df['momentum_10'] / features_df['close'] * 0.3 +
                features_df['momentum_20'] / features_df['close'] * 0.2
            )
        
        return features_df
    
    def _add_lag_features(self, features_df: pd.DataFrame, 
                         lags: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """Adiciona features com lag temporal"""
        # Selecionar features importantes para lag
        important_features = [
            'close', 'volume', 'rsi', 'macd', 'volatility_20',
            'momentum_5', 'buy_pressure', 'volume_imbalance'
        ]
        
        # Adicionar apenas features que existem
        lag_features = [f for f in important_features if f in features_df.columns]
        
        for feature in lag_features:
            for lag in lags:
                features_df[f'{feature}_lag_{lag}'] = features_df[feature].shift(lag)
        
        return features_df
    
    def _add_regime_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de regime de mercado"""
        # Trend regime
        if all(f'ema_{p}' in features_df.columns for p in [20, 50]) and 'close' in features_df.columns:
            # Uptrend
            features_df['uptrend'] = (
                (features_df['ema_20'] > features_df['ema_50']) &
                (features_df['close'] > features_df['ema_20'])
            ).astype(int)
            
            # Downtrend
            features_df['downtrend'] = (
                (features_df['ema_20'] < features_df['ema_50']) &
                (features_df['close'] < features_df['ema_20'])
            ).astype(int)
            
            # Ranging
            features_df['ranging'] = (
                (~features_df['uptrend'].astype(bool)) & (~features_df['downtrend'].astype(bool))
            ).astype(int)
        
        # Volatility regime
        if 'atr' in features_df.columns:
            atr_ma = features_df['atr'].rolling(50).mean()
            ratio = features_df['atr'] / atr_ma
            
            # Tratar valores infinitos e NaN
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            ratio = ratio.fillna(1.0)  # valor padrão
            
            features_df['volatility_regime'] = pd.cut(
                ratio,
                bins=[0, 0.8, 1.2, np.inf],
                labels=[0, 1, 2],  # Low, Normal, High
                include_lowest=True
            )
            features_df['volatility_regime'] = features_df['volatility_regime'].fillna(1).astype(int)
        
        return features_df
    
    def _validate_features(self, features_df: pd.DataFrame):
        """Valida features calculadas"""
        # Verificar NaN
        nan_counts = features_df.isna().sum()
        if nan_counts.sum() > 0:
            self.logger.warning(f"Features com NaN: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Verificar infinitos
        inf_counts = np.isinf(features_df).sum()
        if inf_counts.sum() > 0:
            self.logger.warning(f"Features com infinitos: {inf_counts[inf_counts > 0].to_dict()}")
        
        # Verificar variância zero
        zero_var = features_df.var() == 0
        if zero_var.any():
            self.logger.warning(f"Features com variância zero: {list(features_df.columns[zero_var])}")

    def clean_features(self, features_df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """Limpa features com valores faltantes"""
        if method == 'forward_fill':
            # Forward fill primeiro, depois backward fill para início
            features_df = features_df.ffill().bfill()
        elif method == 'interpolate':
            # Interpolação para séries temporais
            features_df = features_df.interpolate(method='time', limit_direction='both')
        elif method == 'drop':
            # Remove linhas com NaN
            features_df = features_df.dropna()
        
        # Substituir infinitos por valores extremos
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Para qualquer NaN restante, usar a mediana da coluna
        for col in features_df.columns:
            if features_df[col].isna().any():
                median_val = features_df[col].median()
                if pd.isna(median_val):  # Se a mediana também é NaN
                    features_df[col] = features_df[col].fillna(0)
                else:
                    features_df[col] = features_df[col].fillna(median_val)
        
        return features_df

    def select_top_features(self, features_df: pd.DataFrame, 
                          target: pd.Series,
                          n_features: int = 30,
                          method: str = 'mutual_info') -> List[str]:
        """Seleciona top N features mais importantes"""
        from sklearn.feature_selection import mutual_info_classif, f_classif
        from sklearn.ensemble import RandomForestClassifier
        
        # Alinhar índices primeiro
        common_index = features_df.index.intersection(target.index)
        
        if len(common_index) == 0:
            self.logger.error("Nenhum índice comum entre features e target")
            return list(features_df.columns[:n_features])
        
        self.logger.info(f"Alinhando dados: {len(features_df)} features -> {len(common_index)} amostras")
        
        # Usar índices comuns
        aligned_features = features_df.loc[common_index]
        aligned_target = target.loc[common_index]
        
        # Remover NaN para seleção
        clean_features = aligned_features.dropna()
        clean_target = aligned_target.loc[clean_features.index]
        
        if method == 'mutual_info':
            # Mutual Information
            scores = mutual_info_classif(clean_features, clean_target)
            
        elif method == 'f_score':
            # F-score
            scores, _ = f_classif(clean_features, clean_target)
            
        elif method == 'random_forest':
            # Random Forest importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(clean_features, clean_target)
            scores = rf.feature_importances_
        
        # Ordenar features por score
        feature_scores = pd.Series(scores, index=features_df.columns)
        top_features = feature_scores.nlargest(n_features).index.tolist()
        
        self.logger.info(f"Top {n_features} features selecionadas por {method}")
        
        return top_features