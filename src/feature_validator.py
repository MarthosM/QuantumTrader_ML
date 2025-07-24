"""
Feature Validator - Sistema de validação automática de features
Garante que todas as features obrigatórias estão disponíveis antes de usar modelos ML
"""

import json
import logging
import pandas as pd
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path

class FeatureValidationError(Exception):
    """Exceção para erros de validação de features"""
    pass

class FeatureValidator:
    """
    Validador automático de features para modelos ML
    """
    
    def __init__(self, features_config_path: str = None):
        """
        Inicializa o validador de features
        
        Args:
            features_config_path: Caminho para all_required_features.json
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Carregar configuração de features
        if features_config_path is None:
            # Buscar arquivo na raiz do projeto
            current_dir = Path(__file__).parent.parent
            features_config_path = current_dir / "all_required_features.json"
        
        self.config_path = Path(features_config_path)
        self.features_config = self._load_features_config()
        
    def _load_features_config(self) -> Dict:
        """Carrega configuração de features do arquivo JSON"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Arquivo de configuração não encontrado: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.logger.info(f"Configuração de features carregada: {self.config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Erro carregando configuração de features: {e}")
            # Fallback com features mínimas
            return self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict:
        """Configuração de fallback quando arquivo principal não está disponível"""
        return {
            "feature_sets": {
                "minimal": {
                    "features": ["ema_9", "ema_20", "ema_50", "rsi_14", "volume_ratio", 
                               "returns", "volatility", "high", "low", "close", "volume"]
                }
            },
            "model_requirements": {
                "fallback_model": {"feature_set": "minimal"}
            }
        }
    
    def get_required_features(self, model_name: str) -> List[str]:
        """
        Retorna lista de features obrigatórias para um modelo
        
        Args:
            model_name: Nome do modelo (ensemble_production, fallback_model, etc.)
            
        Returns:
            Lista de features obrigatórias
        """
        try:
            # Buscar configuração do modelo
            model_config = self.features_config.get("model_requirements", {}).get(model_name)
            
            if not model_config:
                self.logger.warning(f"Modelo '{model_name}' não encontrado na configuração")
                # Retornar features mínimas como fallback
                return self.features_config["feature_sets"]["minimal"]["features"]
            
            # Obter conjunto de features
            feature_set_name = model_config.get("feature_set", "minimal")
            feature_set = self.features_config.get("feature_sets", {}).get(feature_set_name)
            
            if not feature_set:
                raise FeatureValidationError(f"Conjunto de features '{feature_set_name}' não encontrado")
            
            return feature_set["features"]
            
        except Exception as e:
            self.logger.error(f"Erro obtendo features para modelo '{model_name}': {e}")
            raise FeatureValidationError(f"Não foi possível obter features para '{model_name}': {e}")
    
    def validate_features_for_model(self, model_name: str, available_features: List[str]) -> Tuple[bool, Dict]:
        """
        Valida se todas as features obrigatórias estão disponíveis para um modelo
        
        Args:
            model_name: Nome do modelo
            available_features: Lista de features disponíveis
            
        Returns:
            Tuple (is_valid, validation_result)
        """
        try:
            required_features = set(self.get_required_features(model_name))
            available_features_set = set(available_features)
            
            # Features faltantes
            missing_features = required_features - available_features_set
            
            # Features extras (não obrigatórias mas disponíveis)
            extra_features = available_features_set - required_features
            
            # Resultado da validação
            is_valid = len(missing_features) == 0
            
            validation_result = {
                "model_name": model_name,
                "is_valid": is_valid,
                "required_count": len(required_features),
                "available_count": len(available_features_set),
                "missing_features": sorted(list(missing_features)),
                "extra_features": sorted(list(extra_features)),
                "coverage_percentage": (len(available_features_set & required_features) / len(required_features)) * 100
            }
            
            return is_valid, validation_result
            
        except Exception as e:
            self.logger.error(f"Erro validando features para '{model_name}': {e}")
            return False, {"error": str(e)}
    
    def validate_dataframe(self, df: pd.DataFrame, model_name: str) -> Tuple[bool, Dict]:
        """
        Valida um DataFrame contra as features obrigatórias de um modelo
        
        Args:
            df: DataFrame com features calculadas
            model_name: Nome do modelo
            
        Returns:
            Tuple (is_valid, validation_result)
        """
        try:
            # Validar estrutura básica do DataFrame
            basic_validation = self._validate_dataframe_structure(df)
            if not basic_validation["is_valid"]:
                return False, basic_validation
            
            # Validar features específicas do modelo
            available_features = list(df.columns)
            is_valid, feature_validation = self.validate_features_for_model(model_name, available_features)
            
            # Validar qualidade dos dados
            quality_validation = self._validate_data_quality(df, model_name)
            
            # Resultado combinado
            combined_result = {
                **feature_validation,
                "dataframe_validation": {
                    "shape": df.shape,
                    "index_type": str(type(df.index)),
                    "has_nan": df.isnull().any().any(),
                    "nan_count": df.isnull().sum().sum(),
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
                },
                "quality_validation": quality_validation
            }
            
            # Validação geral é válida se features e qualidade estão OK
            overall_valid = is_valid and quality_validation["is_valid"]
            combined_result["overall_valid"] = overall_valid
            
            return overall_valid, combined_result
            
        except Exception as e:
            self.logger.error(f"Erro validando DataFrame: {e}")
            return False, {"error": str(e)}
    
    def _validate_dataframe_structure(self, df: pd.DataFrame) -> Dict:
        """Valida estrutura básica do DataFrame"""
        try:
            issues = []
            
            # Verificar se não está vazio
            if df.empty:
                issues.append("DataFrame está vazio")
            
            # Verificar índice
            if not isinstance(df.index, pd.DatetimeIndex):
                issues.append("Índice não é DatetimeIndex")
            
            # Verificar columns básicas
            required_basic = self.features_config.get("validation_rules", {}).get("required_columns", [])
            missing_basic = set(required_basic) - set(df.columns)
            if missing_basic:
                issues.append(f"Colunas básicas faltantes: {missing_basic}")
            
            return {
                "is_valid": len(issues) == 0,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "issues": [f"Erro validando estrutura: {e}"]
            }
    
    def _validate_data_quality(self, df: pd.DataFrame, model_name: str) -> Dict:
        """Valida qualidade dos dados"""
        try:
            issues = []
            warnings = []
            
            # Verificar valores NaN
            nan_counts = df.isnull().sum()
            features_with_nan = nan_counts[nan_counts > 0]
            
            if len(features_with_nan) > 0:
                issues.append(f"Features com NaN: {dict(features_with_nan)}")
            
            # Verificar valores infinitos
            inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
            features_with_inf = inf_counts[inf_counts > 0]
            
            if len(features_with_inf) > 0:
                issues.append(f"Features com valores infinitos: {dict(features_with_inf)}")
            
            # Verificar tipos de dados
            non_numeric = df.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric) > 0:
                warnings.append(f"Features não numéricas: {list(non_numeric)}")
            
            # Verificar volume positivo (se aplicável)
            if 'volume' in df.columns:
                negative_volume = (df['volume'] < 0).sum()
                if negative_volume > 0:
                    issues.append(f"Volume negativo encontrado: {negative_volume} registros")
            
            # Verificar consistência OHLC (se aplicável)
            ohlc_cols = ['open', 'high', 'low', 'close']
            if all(col in df.columns for col in ohlc_cols):
                # High deve ser >= Open, Low, Close
                high_issues = ((df['high'] < df['open']) | 
                              (df['high'] < df['low']) | 
                              (df['high'] < df['close'])).sum()
                
                # Low deve ser <= Open, High, Close
                low_issues = ((df['low'] > df['open']) | 
                             (df['low'] > df['high']) | 
                             (df['low'] > df['close'])).sum()
                
                if high_issues > 0:
                    issues.append(f"Inconsistências em HIGH: {high_issues} registros")
                
                if low_issues > 0:
                    issues.append(f"Inconsistências em LOW: {low_issues} registros")
            
            return {
                "is_valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "issues": [f"Erro validando qualidade: {e}"],
                "warnings": []
            }
    
    def get_feature_dependencies(self, feature_name: str) -> List[str]:
        """
        Retorna dependências de uma feature específica
        
        Args:
            feature_name: Nome da feature
            
        Returns:
            Lista de features das quais a feature depende
        """
        dependencies = self.features_config.get("dependencies", {})
        return dependencies.get(feature_name, [])
    
    def validate_feature_dependencies(self, available_features: List[str]) -> Dict:
        """
        Valida se todas as dependências de features estão satisfeitas
        
        Args:
            available_features: Lista de features disponíveis
            
        Returns:
            Resultado da validação de dependências
        """
        available_set = set(available_features)
        dependencies = self.features_config.get("dependencies", {})
        
        unsatisfied_deps = {}
        
        for feature, deps in dependencies.items():
            if feature in available_set:
                missing_deps = set(deps) - available_set
                if missing_deps:
                    unsatisfied_deps[feature] = list(missing_deps)
        
        return {
            "is_valid": len(unsatisfied_deps) == 0,
            "unsatisfied_dependencies": unsatisfied_deps
        }
    
    def suggest_model_for_features(self, available_features: List[str]) -> Optional[str]:
        """
        Sugere o melhor modelo baseado nas features disponíveis
        
        Args:
            available_features: Lista de features disponíveis
            
        Returns:
            Nome do modelo sugerido ou None
        """
        available_set = set(available_features)
        best_model = None
        best_coverage = 0
        
        for model_name, model_config in self.features_config.get("model_requirements", {}).items():
            try:
                required_features = set(self.get_required_features(model_name))
                coverage = len(available_set & required_features) / len(required_features)
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_model = model_name
                    
            except Exception:
                continue
        
        return best_model if best_coverage >= 0.8 else None  # Mínimo 80% de cobertura
    
    def generate_validation_report(self, df: pd.DataFrame, model_name: str) -> str:
        """
        Gera relatório completo de validação em formato texto
        
        Args:
            df: DataFrame para validar
            model_name: Nome do modelo
            
        Returns:
            Relatório de validação formatado
        """
        is_valid, result = self.validate_dataframe(df, model_name)
        
        report = f"""
=== RELATÓRIO DE VALIDAÇÃO DE FEATURES ===
Modelo: {model_name}
Data/Hora: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

RESULTADO GERAL: {'✅ VÁLIDO' if is_valid else '❌ INVÁLIDO'}

FEATURES:
- Obrigatórias: {result.get('required_count', 0)}
- Disponíveis: {result.get('available_count', 0)}
- Cobertura: {result.get('coverage_percentage', 0):.1f}%

"""
        
        if result.get('missing_features'):
            report += f"❌ FEATURES FALTANTES ({len(result['missing_features'])}):\n"
            for feature in result['missing_features']:
                report += f"   - {feature}\n"
            report += "\n"
        
        if result.get('extra_features'):
            report += f"ℹ️ FEATURES EXTRAS ({len(result['extra_features'])}):\n"
            for feature in sorted(result['extra_features'])[:10]:  # Primeiras 10
                report += f"   - {feature}\n"
            if len(result['extra_features']) > 10:
                report += f"   ... e mais {len(result['extra_features']) - 10} features\n"
            report += "\n"
        
        # Qualidade dos dados
        quality = result.get('quality_validation', {})
        if quality.get('issues'):
            report += "❌ PROBLEMAS DE QUALIDADE:\n"
            for issue in quality['issues']:
                report += f"   - {issue}\n"
            report += "\n"
        
        if quality.get('warnings'):
            report += "⚠️ AVISOS:\n"
            for warning in quality['warnings']:
                report += f"   - {warning}\n"
            report += "\n"
        
        # Informações do DataFrame
        df_info = result.get('dataframe_validation', {})
        report += f"INFORMAÇÕES DO DATAFRAME:\n"
        report += f"- Shape: {df_info.get('shape', 'N/A')}\n"
        report += f"- Tipo do índice: {df_info.get('index_type', 'N/A')}\n"
        report += f"- Valores NaN: {df_info.get('nan_count', 0)}\n"
        report += f"- Uso de memória: {df_info.get('memory_usage_mb', 0):.1f} MB\n"
        
        return report

# Importar numpy para validação
import numpy as np