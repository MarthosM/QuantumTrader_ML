import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_structure import TradingDataStructure
from technical_indicators import TechnicalIndicators
from ml_features import MLFeatures


class FeatureEngine:
    """Motor principal de cálculo de features"""
    
    def __init__(self, model_features: Optional[List[str]] = None):
        self.model_features = model_features or []
        self.technical = TechnicalIndicators()
        self.ml_features = MLFeatures(model_features)
        self.logger = logging.getLogger(__name__)
        
        # Cache para evitar recálculos
        self.cache = {
            'indicators': None,
            'features': None,
            'last_candle_time': None,
            'last_candle_count': 0
        }
        
        # Configurações
        self.min_candles_required = 50
        self.parallel_processing = True
        self.max_workers = 4
        
        # Mapeamento de features para seus requisitos
        self.feature_dependencies = self._build_feature_dependencies()
    
    def calculate(self, data: TradingDataStructure, 
                 force_recalculate: bool = False) -> Dict[str, pd.DataFrame]:
        """Calcula todas as features necessárias"""
        
        if data.candles.empty:
            self.logger.warning("Sem dados de candles para calcular features")
            return {'model_ready': pd.DataFrame()}
        
        # Verificar se precisa recalcular
        if not force_recalculate and self._is_cache_valid(data):
            self.logger.info("Usando cache de features")
            return self._get_from_cache(data)
        
        self.logger.info(f"Calculando features para {len(data.candles)} candles")
        
        # Calcular em paralelo se habilitado
        if self.parallel_processing and len(data.candles) > 100:
            result = self._calculate_parallel(data)
        else:
            result = self._calculate_sequential(data)
        
        # Atualizar cache
        self._update_cache(data, result)
        
        return result
    
    def _calculate_sequential(self, data: TradingDataStructure) -> Dict[str, pd.DataFrame]:
        """Calcula features sequencialmente"""
        # Indicadores técnicos
        data.indicators = self.technical.calculate_all(data.candles)
        
        # Features ML
        data.features = self.ml_features.calculate_all(
            data.candles,
            data.microstructure,
            data.indicators
        )
        
        # Preparar DataFrame final para o modelo
        model_ready_df = self._prepare_model_data(data)
        
        return {
            'indicators': data.indicators,
            'features': data.features,
            'model_ready': model_ready_df,
            'all': self._merge_all_features(data)
        }
    
    def _calculate_parallel(self, data: TradingDataStructure) -> Dict[str, pd.DataFrame]:
        """Calcula features em paralelo para melhor performance"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submeter tarefas
            futures = {}
            
            # Indicadores técnicos
            futures['indicators'] = executor.submit(
                self.technical.calculate_all, data.candles
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
        
        # Preparar DataFrame final
        model_ready_df = self._prepare_model_data(data)
        
        return {
            'indicators': data.indicators,
            'features': data.features,
            'model_ready': model_ready_df,
            'all': self._merge_all_features(data)
        }
    
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
            'volume_imbalance': {'buy_volume', 'sell_volume'}
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