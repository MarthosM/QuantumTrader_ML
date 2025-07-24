#!/usr/bin/env python3
"""
Validador RIGOROSO de dados de trading
BLOQUEIA dados problemáticos antes que causem danos
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class TradingDataValidator:
    """Validador CRÍTICO para dados de trading"""
    
    def __init__(self, logger):
        self.logger = logger
        
    def validate_data(self, df: pd.DataFrame, data_type: str) -> Tuple[bool, List[str]]:
        """
        Validação RIGOROSA de dados
        
        Returns:
            (is_valid, errors_list)
        """
        errors = []
        
        # 1. Validações básicas
        errors.extend(self._validate_basic_integrity(df))
        
        # 2. Validações específicas por tipo
        if data_type == 'candles':
            errors.extend(self._validate_candles(df))
        elif data_type == 'features':
            errors.extend(self._validate_features(df))
        elif data_type == 'predictions':
            errors.extend(self._validate_predictions(df))
            
        # 3. Validação de preenchimento
        errors.extend(self._validate_fill_quality(df))
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            self.logger.error(f"VALIDAÇÃO FALHOU para {data_type}:")
            for error in errors:
                self.logger.error(f"  ❌ {error}")
        else:
            self.logger.info(f"✅ Validação OK para {data_type}")
            
        return is_valid, errors
    
    def _validate_basic_integrity(self, df: pd.DataFrame) -> List[str]:
        """Validações básicas de integridade"""
        errors = []
        
        # DataFrame vazio
        if df.empty:
            errors.append("DataFrame está vazio")
            return errors
            
        # NaN restantes
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            errors.append(f"Ainda há {nan_count} valores NaN")
            
        # Infinitos
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            errors.append(f"Há {inf_count} valores infinitos")
            
        return errors
    
    def _validate_candles(self, df: pd.DataFrame) -> List[str]:
        """Validação específica para dados de candles"""
        errors = []
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Colunas obrigatórias ausentes: {missing_cols}")
            return errors
            
        # Preços devem ser positivos
        for col in ['open', 'high', 'low', 'close']:
            if (df[col] <= 0).any():
                errors.append(f"Preços inválidos em {col} (<=0)")
                
        # High >= Low sempre
        if (df['high'] < df['low']).any():
            errors.append("High < Low detectado")
            
        # Volume não pode ser negativo
        if (df['volume'] < 0).any():
            errors.append("Volume negativo detectado")
            
        return errors
    
    def _validate_features(self, df: pd.DataFrame) -> List[str]:
        """Validação para features de ML"""
        errors = []
        
        # Verificar excesso de zeros
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                zero_ratio = (df[col] == 0).sum() / len(df)
                
                # Se não é momentum e tem muitos zeros, é suspeito
                if zero_ratio > 0.5 and 'momentum' not in col.lower():
                    errors.append(f"Feature {col} tem {zero_ratio:.1%} zeros (suspeito)")
                    
        return errors
    
    def _validate_fill_quality(self, df: pd.DataFrame) -> List[str]:
        """Validação da qualidade do preenchimento"""
        errors = []
        
        for col in df.columns:
            # Detectar preenchimento suspeito
            if self._detect_suspicious_fill(df[col]):
                errors.append(f"Preenchimento suspeito em {col}")
                
        return errors
    
    def _detect_suspicious_fill(self, series: pd.Series) -> bool:
        """Detecta padrões suspeitos de preenchimento"""
        
        # Muitos valores consecutivos iguais (possível fillna inadequado)
        consecutive_same = series.groupby((series != series.shift()).cumsum()).size()
        max_consecutive = consecutive_same.max() if len(consecutive_same) > 0 else 0
        
        # Se mais de 10% dos dados são valores consecutivos iguais
        if max_consecutive > len(series) * 0.1:
            return True
            
        return False
