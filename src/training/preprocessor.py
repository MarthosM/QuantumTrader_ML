# src/training/preprocessor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

# Importar RobustNaNHandler
try:
    from .robust_nan_handler import RobustNaNHandler
except ImportError:
    from robust_nan_handler import RobustNaNHandler

class DataPreprocessor:
    """Preprocessador de dados para treinamento de modelos ML"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.imputers = {}
        self.feature_stats = {}
        
        # Inicializar handler de NaN robusto
        self.nan_handler = RobustNaNHandler()
        
    def preprocess_training_data(self, data: pd.DataFrame, 
                               target_col: str = 'target',
                               scale_features: bool = True,
                               raw_ohlcv: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocessa dados para treinamento
        
        Args:
            data: DataFrame com features e target
            target_col: Nome da coluna target
            scale_features: Se deve escalar features
            raw_ohlcv: Dados OHLCV originais para recálculos (opcional)
            
        Returns:
            Tupla (features_processadas, targets)
        """
        self.logger.info(f"Preprocessando {len(data)} amostras")
        
        # Separar features e target
        if target_col in data.columns:
            features = data.drop(columns=[target_col])
            target = data[target_col]
        else:
            # Criar target baseado em retornos futuros
            features = data
            target = self._create_trading_target(data)
        
        # Remover colunas não numéricas
        numeric_features = features.select_dtypes(include=[np.number])
        
        # ✅ USAR ROBUST NaN HANDLER se temos dados OHLCV
        if raw_ohlcv is not None:
            self.logger.info("🧹 Usando RobustNaNHandler para tratar valores ausentes...")
            processed_features, nan_stats = self.nan_handler.handle_nans(numeric_features, raw_ohlcv)
            
            # Validar qualidade do tratamento
            validation_result = self.nan_handler.validate_nan_handling(processed_features)
            
            # Log resultados
            self.logger.info(f"NaN Handler - Score qualidade: {validation_result['quality_score']:.3f}")
            if validation_result['quality_score'] < 0.8:
                self.logger.warning("⚠️ Qualidade baixa no tratamento de NaN - revisar dados")
            
            # Criar relatório se necessário
            if self.logger.level <= logging.INFO:
                report = self.nan_handler.create_nan_handling_report(nan_stats, validation_result)
                self.logger.debug("Relatório NaN Handler:\n" + report)
                
        else:
            # Fallback para método tradicional
            self.logger.warning("Usando método tradicional para NaN (sem dados OHLCV)")
            processed_features = self._handle_missing_values(numeric_features)
        
        # Remover outliers extremos
        processed_features = self._remove_outliers(processed_features)
        
        # Escalar features se necessário
        if scale_features:
            processed_features = self._scale_features(processed_features)
        
        # Adicionar features temporais
        processed_features = self._add_temporal_features(processed_features)
        
        # Validar dados finais
        self._validate_processed_data(processed_features, target)
        
        return processed_features, target
    
    def _create_trading_target(self, data: pd.DataFrame, 
                             forward_periods: int = 5,
                             threshold: float = 0.002) -> pd.Series:
        """
        Cria target para trading baseado em retornos futuros
        
        Classes:
        - 0: Venda (retorno < -threshold)
        - 1: Neutro (-threshold <= retorno <= threshold)
        - 2: Compra (retorno > threshold)
        """
        # Verificar se há dados suficientes
        if len(data) < forward_periods + 1:
            self.logger.warning(f"Dados insuficientes para criar target: {len(data)} < {forward_periods + 1}")
            # Retornar target simples baseado apenas no último movimento
            if len(data) < 2:
                # Para 1 amostra, usar classe neutra
                return pd.Series([1], index=data.index[:1])
            
            # Para poucos dados, usar movimento simples
            price_change = data['close'].iloc[-1] / data['close'].iloc[0] - 1
            if price_change > threshold:
                target_value = 2  # Compra
            elif price_change < -threshold:
                target_value = 0  # Venda  
            else:
                target_value = 1  # Neutro
                
            # Criar target para todas as amostras menos uma
            target_length = max(1, len(data) - 1)
            return pd.Series([target_value] * target_length, index=data.index[:target_length])
        
        # Cálculo normal para dados suficientes
        # Calcular retorno forward
        future_return = data['close'].shift(-forward_periods) / data['close'] - 1
        
        # Classificar em 3 classes
        target = pd.Series(1, index=data.index)  # Default: neutro
        target[future_return < -threshold] = 0   # Venda
        target[future_return > threshold] = 2    # Compra
        
        # Remover últimas observações sem target (evitar NaN)
        valid_length = len(data) - forward_periods
        if valid_length <= 0:
            # Fallback para dados muito pequenos
            self.logger.warning("Dados insuficientes para lookforward, usando target simplificado")
            return pd.Series([1] * max(1, len(data) - 1), index=data.index[:max(1, len(data) - 1)])
            
        target = target[:valid_length]
        
        # Log distribuição de classes
        if len(target) > 0:
            class_dist = target.value_counts(normalize=True)
            self.logger.info(f"Distribuição de classes: {class_dist.to_dict()}")
        else:
            self.logger.warning("Target vazio após processamento")
        
        return target
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Trata valores ausentes de forma inteligente"""
        processed = data.copy()
        
        for column in processed.columns:
            missing_pct = processed[column].isna().sum() / len(processed)
            
            if missing_pct > 0.5:
                # Remover coluna se mais de 50% ausente
                self.logger.warning(f"Removendo coluna {column} com {missing_pct:.1%} ausente")
                processed = processed.drop(columns=[column])
                
            elif missing_pct > 0:
                # Estratégia baseada no tipo de feature
                if 'volume' in column or 'trades' in column:
                    # Volume: usar 0 ou mediana
                    processed[column] = processed[column].fillna(0)
                    
                elif 'price' in column or 'close' in column or 'vwap' in column:
                    # Preços: forward/backward fill
                    processed[column] = processed[column].ffill().bfill()
                    
                else:
                    # Outros: mediana
                    median_val = processed[column].median()
                    processed[column] = processed[column].fillna(median_val)
        
        return processed
    
    def _remove_outliers(self, data: pd.DataFrame, 
                        method: str = 'iqr',
                        threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers extremos"""
        processed = data.copy()
        
        if method == 'iqr':
            # Método IQR
            for column in processed.columns:
                Q1 = processed[column].quantile(0.25)
                Q3 = processed[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Clip valores ao invés de remover linhas
                processed[column] = processed[column].clip(lower=lower_bound, upper=upper_bound)
                
        elif method == 'zscore':
            # Método Z-score
            for column in processed.columns:
                mean = processed[column].mean()
                std = processed[column].std()
                
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                
                processed[column] = processed[column].clip(lower=lower_bound, upper=upper_bound)
        
        return processed
    
    def _scale_features(self, data: pd.DataFrame, 
                       method: str = 'robust') -> pd.DataFrame:
        """Escala features para normalização"""
        scaled_data = data.copy()
        
        for column in scaled_data.columns:
            if method == 'robust':
                # RobustScaler é melhor para dados com outliers
                scaler = RobustScaler()
            else:
                # StandardScaler para distribuições normais
                scaler = StandardScaler()
            
            scaled_data[column] = scaler.fit_transform(scaled_data[[column]])
            
            # Salvar scaler para uso futuro
            self.scalers[column] = scaler
            
            # Salvar estatísticas
            self.feature_stats[column] = {
                'mean': data[column].mean(),
                'std': data[column].std(),
                'min': data[column].min(),
                'max': data[column].max()
            }
        
        return scaled_data
    
    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features temporais relevantes para trading"""
        processed = data.copy()
        
        if isinstance(data.index, pd.DatetimeIndex):
            # Features de tempo
            processed['hour'] = data.index.hour
            processed['minute'] = data.index.minute
            processed['day_of_week'] = data.index.dayofweek
            processed['day_of_month'] = data.index.day
            processed['month'] = data.index.month
            
            # Features cíclicas (seno/cosseno para periodicidade)
            processed['hour_sin'] = np.sin(2 * np.pi * processed['hour'] / 24)
            processed['hour_cos'] = np.cos(2 * np.pi * processed['hour'] / 24)
            
            processed['minute_sin'] = np.sin(2 * np.pi * processed['minute'] / 60)
            processed['minute_cos'] = np.cos(2 * np.pi * processed['minute'] / 60)
            
            # Indicadores de sessão
            processed['is_morning'] = (processed['hour'] >= 9) & (processed['hour'] < 12)
            processed['is_afternoon'] = (processed['hour'] >= 12) & (processed['hour'] < 15)
            processed['is_closing'] = (processed['hour'] >= 15) & (processed['hour'] < 18)
            
            # Converter booleanos para int
            bool_cols = processed.select_dtypes(include=['bool']).columns
            processed[bool_cols] = processed[bool_cols].astype(int)
        
        return processed
    
    def _validate_processed_data(self, features: pd.DataFrame, target: pd.Series):
        """Valida dados processados"""
        # Verificar NaN
        nan_features = features.isna().sum().sum()
        nan_target = target.isna().sum()
        
        if nan_features > 0:
            raise ValueError(f"Features contêm {nan_features} valores NaN após processamento")
        
        if nan_target > 0:
            raise ValueError(f"Target contém {nan_target} valores NaN")
        
        # Verificar infinitos
        inf_features = np.isinf(features).sum().sum()
        if inf_features > 0:
            raise ValueError(f"Features contêm {inf_features} valores infinitos")
        
        # Verificar alinhamento e corrigir se necessário
        if len(features) != len(target):
            self.logger.warning(f"Desalinhamento detectado: features ({len(features)}) != target ({len(target)})")
            
            # Ajustar para o menor tamanho (comum devido a lookback nos targets)
            min_size = min(len(features), len(target))
            features = features.iloc[:min_size]
            target = target.iloc[:min_size] if hasattr(target, 'iloc') else target[:min_size]
            
            self.logger.info(f"Alinhamento corrigido para {min_size} amostras")
            
            # Verificar se ainda há problemas
            if len(features) != len(target):
                raise ValueError(f"Não foi possível alinhar: features ({len(features)}) != target ({len(target)})")
        
        # Verificar variância zero
        zero_var_cols = features.columns[features.var() == 0]
        if len(zero_var_cols) > 0:
            self.logger.warning(f"Colunas com variância zero: {list(zero_var_cols)}")
            
    def transform_inference_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforma dados para inferência usando parâmetros salvos"""
        processed = data.copy()
        
        # Aplicar mesmas transformações do treino
        processed = self._handle_missing_values(processed)
        
        # Escalar usando scalers salvos
        for column, scaler in self.scalers.items():
            if column in processed.columns:
                processed[column] = scaler.transform(processed[[column]])
        
        # Adicionar features temporais
        processed = self._add_temporal_features(processed)
        
        return processed