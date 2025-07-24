"""
Pipeline Robusto de Features v2.0
Sistema integrado com indicadores robustos e tratamento inteligente de NaN
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Any
import logging
from concurrent.futures import ProcessPoolExecutor
import warnings

# Importar sistemas robustos
from .robust_technical_indicators import RobustTechnicalIndicators
from .robust_nan_handler import RobustNaNHandler

warnings.filterwarnings('ignore', message='.*TA-Lib.*')

class RobustFeaturePipeline:
    """
    Pipeline robusto de features que resolve todos os problemas detectados:
    1. Indicadores técnicos precisos sem TA-Lib
    2. Tratamento inteligente de NaN sem introduzir viés
    3. Validação robusta de dados
    4. Performance otimizada
    """
    
    def __init__(self, n_jobs: int = -1):
        self.logger = logging.getLogger(__name__)
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        
        # Inicializar sistemas robustos
        self.robust_indicators = RobustTechnicalIndicators()
        self.nan_handler = RobustNaNHandler()
        
        self.logger.info("🚀 Pipeline Robusto de Features v2.0 inicializado")
    
    def create_features(self, data: pd.DataFrame, 
                       feature_groups: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Cria features usando sistema robusto
        Resolve problemas de NaN de forma inteligente
        """
        if feature_groups is None:
            feature_groups = ['basic', 'technical', 'momentum', 'volatility', 'volume', 'advanced']
        
        self.logger.info(f"🔧 Criando features robustas para {len(data)} amostras")
        
        # Validar dados de entrada
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            # Tentar mapear colunas alternativas
            data = self._map_alternative_columns(data)
            
            # Verificar novamente
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Colunas ausentes após mapeamento: {missing_cols}")
        
        # Inicializar DataFrame de features
        features_df = pd.DataFrame(index=data.index)
        
        try:
            # 1. Features básicas OHLCV
            if 'basic' in feature_groups:
                basic_features = self._create_basic_features(data)
                features_df = pd.concat([features_df, basic_features], axis=1)
                self.logger.info(f"✅ Features básicas: {len(basic_features.columns)}")
            
            # 2. Indicadores técnicos robustos
            if 'technical' in feature_groups:
                tech_features = self.robust_indicators.calculate_all_indicators(data)
                features_df = pd.concat([features_df, tech_features], axis=1)
                self.logger.info(f"✅ Indicadores técnicos: {len(tech_features.columns)}")
            
            # 3. Features de momentum
            if 'momentum' in feature_groups:
                momentum_features = self.robust_indicators.calculate_momentum_features(data)
                features_df = pd.concat([features_df, momentum_features], axis=1)
                self.logger.info(f"✅ Features de momentum: {len(momentum_features.columns)}")
            
            # 4. Features de volatilidade
            if 'volatility' in feature_groups:
                volatility_features = self.robust_indicators.calculate_volatility_features(data)
                features_df = pd.concat([features_df, volatility_features], axis=1)
                self.logger.info(f"✅ Features de volatilidade: {len(volatility_features.columns)}")
            
            # 5. Features de volume
            if 'volume' in feature_groups:
                volume_features = self.robust_indicators.calculate_volume_features(data)
                features_df = pd.concat([features_df, volume_features], axis=1)
                self.logger.info(f"✅ Features de volume: {len(volume_features.columns)}")
            
            # 6. Features avançadas (lags, crosses, etc)
            if 'advanced' in feature_groups:
                # Lag features
                lag_features = self.robust_indicators.calculate_lag_features(features_df)
                features_df = pd.concat([features_df, lag_features], axis=1)
                
                # Cross features
                cross_features = self.robust_indicators.calculate_cross_features(features_df)
                features_df = pd.concat([features_df, cross_features], axis=1)
                
                self.logger.info(f"✅ Features avançadas: {len(lag_features.columns) + len(cross_features.columns)}")
            
            # 7. TRATAMENTO ROBUSTO DE NaN
            self.logger.info(f"🔧 Tratando NaN em {len(features_df.columns)} features")
            
            # Contar NaNs antes do tratamento
            nan_counts_before = features_df.isna().sum()
            features_with_nan = nan_counts_before[nan_counts_before > 0]
            
            if len(features_with_nan) > 0:
                self.logger.warning(f"⚠️ Features com NaN detectadas: {len(features_with_nan)}")
                for feature, count in features_with_nan.head(10).items():
                    self.logger.warning(f"   {feature}: {count} NaN")
                
                # Aplicar tratamento robusto
                features_df, nan_stats = self.nan_handler.handle_nans(features_df, data)
                
                self.logger.info(f"✅ NaN tratados: {nan_stats['initial_rows']} → {nan_stats['final_rows']} linhas")
                
                if nan_stats['removed_features']:
                    self.logger.warning(f"⚠️ Features removidas por excesso de NaN: {nan_stats['removed_features']}")
            
            else:
                self.logger.info("✅ Nenhum NaN detectado")
            
            self.logger.info(f"🎯 Pipeline completo: {len(features_df.columns)} features, {len(features_df)} amostras")
            
        except Exception as e:
            self.logger.error(f"❌ Erro no pipeline de features: {e}")
            raise
        
        return features_df
    
    def _map_alternative_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mapeia colunas alternativas para formato padrão"""
        data = data.copy()
        
        # Mapeamentos conhecidos
        column_mappings = {
            'close': ['preco', 'last', 'price', 'Close'],
            'volume': ['vol', 'Volume', 'quantidade'],
            'open': ['Open', 'abertura'],
            'high': ['High', 'maxima', 'max'],
            'low': ['Low', 'minima', 'min']
        }
        
        for standard_col, alternatives in column_mappings.items():
            if standard_col not in data.columns:
                for alt_col in alternatives:
                    if alt_col in data.columns:
                        data[standard_col] = data[alt_col]
                        self.logger.info(f"Mapeando {alt_col} → {standard_col}")
                        break
        
        return data
    
    def _create_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cria features básicas OHLCV"""
        basic_features = pd.DataFrame(index=data.index)
        
        # Copiar features básicas
        basic_features['open'] = data['open']
        basic_features['high'] = data['high']
        basic_features['low'] = data['low']
        basic_features['close'] = data['close']
        basic_features['volume'] = data['volume']
        
        return basic_features
    
    def select_best_features(self, X: pd.DataFrame, y: pd.Series, 
                           n_features: int = 30, 
                           method: str = 'mutual_info') -> List[str]:
        """
        Seleciona as melhores features usando método especificado
        """
        self.logger.info(f"Selecionando top {n_features} features usando {method}")
        
        if method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_classif
            
            # Calcular mutual information
            mi_scores = mutual_info_classif(X, y, random_state=42)
            feature_scores = pd.Series(mi_scores, index=X.columns)
            
        elif method == 'f_classif':
            from sklearn.feature_selection import f_classif
            
            # Calcular F-score
            f_scores, _ = f_classif(X, y)
            feature_scores = pd.Series(f_scores, index=X.columns)
        
        elif method == 'chi2':
            from sklearn.feature_selection import chi2
            from sklearn.preprocessing import MinMaxScaler
            
            # Normalizar para chi2 (valores não negativos)
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            chi2_scores, _ = chi2(X_scaled, y)
            feature_scores = pd.Series(chi2_scores, index=X.columns)
        
        else:
            raise ValueError(f"Método não suportado: {method}")
        
        # Selecionar top features
        top_features = feature_scores.nlargest(n_features).index.tolist()
        
        self.logger.info(f"Top {len(top_features)} features selecionadas por {method}")
        
        return top_features
    
    def align_features_target(self, features: pd.DataFrame, 
                            target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Alinha features com target removendo desalinhamentos
        """
        self.logger.info(f"Alinhando features ({len(features)}) com target ({len(target)})")
        
        # Encontrar índices comuns
        common_indices = features.index.intersection(target.index)
        
        if len(common_indices) == 0:
            raise ValueError("Nenhum índice comum entre features e target")
        
        # Alinhar dados
        features_aligned = features.loc[common_indices]
        target_aligned = target.loc[common_indices]
        
        # Remover linhas com NaN no target
        valid_indices = ~target_aligned.isna()
        features_final = features_aligned[valid_indices]
        target_final = target_aligned[valid_indices]
        
        self.logger.info(f"Dados alinhados: {len(features_final)} amostras finais")
        
        return features_final, target_final
    
    def validate_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida qualidade das features geradas
        """
        validation_report = {
            'total_features': len(features.columns),
            'total_samples': len(features),
            'nan_features': [],
            'infinite_features': [],
            'constant_features': [],
            'low_variance_features': [],
            'correlation_issues': []
        }
        
        # Verificar NaN
        nan_counts = features.isna().sum()
        validation_report['nan_features'] = nan_counts[nan_counts > 0].to_dict()
        
        # Verificar infinitos
        for col in features.columns:
            if np.isinf(features[col]).any():
                validation_report['infinite_features'].append(col)
        
        # Verificar constantes
        for col in features.columns:
            if features[col].nunique() <= 1:
                validation_report['constant_features'].append(col)
        
        # Verificar baixa variância
        numeric_features = features.select_dtypes(include=[np.number])
        low_var_threshold = 0.01
        
        for col in numeric_features.columns:
            try:
                col_var = numeric_features[col].var(skipna=True)
                # Conversão segura para verificação usando numpy
                try:
                    var_value = np.asarray(col_var).item() if col_var is not None else None
                    if var_value is None or pd.isna(var_value) or not np.isfinite(var_value):
                        validation_report['low_variance_features'].append(col)
                    elif var_value < low_var_threshold:
                        validation_report['low_variance_features'].append(col)
                except (TypeError, ValueError, OverflowError):
                    validation_report['low_variance_features'].append(col)
            except (TypeError, ValueError, ZeroDivisionError):
                # Se não conseguir calcular variância, considerar como problema
                validation_report['low_variance_features'].append(col)
        
        # Verificar alta correlação de forma mais robusta
        if len(numeric_features.columns) > 1:
            try:
                # Remover colunas com variância zero ou constantes para correlação
                valid_cols = []
                for col in numeric_features.columns:
                    try:
                        col_var = numeric_features[col].var(skipna=True)
                        var_value = np.asarray(col_var).item() if col_var is not None else 0.0
                        if var_value > 1e-10 and numeric_features[col].nunique(dropna=True) > 1:
                            valid_cols.append(col)
                    except:
                        continue
                
                if len(valid_cols) > 1:
                    # Calcular correlação apenas com colunas válidas
                    valid_features = numeric_features[valid_cols]
                    corr_matrix = valid_features.corr(method='pearson').abs()
                    
                    # Verificar se a matriz de correlação é válida
                    if not corr_matrix.isna().all().all():
                        high_corr_pairs = []
                        
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                try:
                                    corr_val = corr_matrix.iloc[i, j]
                                    corr_value = np.asarray(corr_val).item() if corr_val is not None else 0.0
                                    
                                    # Verificar se a correlação é válida e alta
                                    if (not pd.isna(corr_value) and 
                                        np.isfinite(corr_value) and 
                                        corr_value > 0.95):
                                        high_corr_pairs.append((
                                            corr_matrix.columns[i], 
                                            corr_matrix.columns[j], 
                                            corr_value
                                        ))
                                except (TypeError, ValueError, OverflowError):
                                    # Pular correlações inválidas
                                    continue
                        
                        validation_report['correlation_issues'] = high_corr_pairs
                    else:
                        self.logger.warning("⚠️ Matriz de correlação inválida - pulando verificação")
                        validation_report['correlation_issues'] = []
                else:
                    self.logger.info("Menos de 2 features válidas para correlação")
                    validation_report['correlation_issues'] = []
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Erro ao calcular correlações: {e}")
                validation_report['correlation_issues'] = []
        
        # Log resumo
        issues = sum([
            len(validation_report['nan_features']),
            len(validation_report['infinite_features']),
            len(validation_report['constant_features']),
            len(validation_report['low_variance_features']),
            len(validation_report['correlation_issues'])
        ])
        
        if issues > 0:
            self.logger.warning(f"⚠️ Validação detectou {issues} problemas nas features")
        else:
            self.logger.info("✅ Validação de features passou sem problemas")
        
        return validation_report
