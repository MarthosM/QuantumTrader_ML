
import pandas as pd
import numpy as np
from typing import Optional

class EnhancedSmartFillStrategy:
    """Estratégia APRIMORADA de preenchimento para trading"""
    
    def __init__(self, logger):
        self.logger = logger
        self.fill_stats = {}
        
    def fill_missing_values(self, df: pd.DataFrame, context: Optional[str] = None) -> pd.DataFrame:
        """
        Preenche valores faltantes com estratégia INTELIGENTE e VALIDADA
        
        ❌ NUNCA usa fillna(0) sem justificativa
        ✅ Estratégias específicas por tipo de dado
        ✅ Validação posterior obrigatória
        """
        df_filled = df.copy()
        initial_nan_count = df_filled.isnull().sum().sum()
        
        for col in df_filled.columns:
            if df_filled[col].isna().any():
                original_nan = df_filled[col].isna().sum()
                
                # Aplicar estratégia específica
                df_filled[col] = self._apply_smart_strategy(df_filled[col], col, context)
                
                # Validar resultado
                remaining_nan = df_filled[col].isna().sum()
                self._log_fill_operation(col, original_nan, remaining_nan, context)
        
        final_nan_count = df_filled.isnull().sum().sum()
        self.logger.info(f"SmartFill: {initial_nan_count} → {final_nan_count} NaN")
        
        # VALIDAÇÃO CRÍTICA
        self._validate_filled_data(df_filled, context)
        
        return df_filled
    
    def _apply_smart_strategy(self, series: pd.Series, col_name: str, context: str) -> pd.Series:
        """Aplica estratégia específica baseada em tipo e contexto"""
        
        # 1. PREÇOS: Nunca zero
        if self._is_price_feature(col_name):
            return self._fill_price_safe(series)
            
        # 2. VOLUMES: Cuidado especial
        elif self._is_volume_feature(col_name):
            return self._fill_volume_safe(series)
            
        # 3. INDICADORES TÉCNICOS: Valores apropriados
        elif self._is_technical_indicator(col_name):
            return self._fill_indicator_safe(series, col_name)
            
        # 4. RATIOS/PERCENTUAIS: Interpolação
        elif self._is_ratio_feature(col_name):
            return self._fill_ratio_safe(series)
            
        # 5. MOMENTUM: Pode usar zero COM CUIDADO
        elif self._is_momentum_feature(col_name):
            return self._fill_momentum_safe(series)
            
        # 6. DEFAULT: Estratégia conservadora
        else:
            return self._fill_conservative(series)
    
    def _is_price_feature(self, col_name: str) -> bool:
        """Identifica features de preço"""
        price_indicators = ['open', 'high', 'low', 'close', 'price', 'ema', 'sma', 'bb_']
        return any(indicator in col_name.lower() for indicator in price_indicators)
    
    def _fill_price_safe(self, series: pd.Series) -> pd.Series:
        """Preenche preços de forma SEGURA - NUNCA zero"""
        filled = series.ffill()  # Último preço conhecido
        
        if filled.isna().any():
            filled = filled.bfill()  # Próximo preço conhecido
            
        if filled.isna().any() and filled.notna().any():
            # Usar média dos preços válidos
            valid_prices = filled.dropna()
            if len(valid_prices) > 0:
                median_price = valid_prices.median()
                filled = filled.fillna(median_price)
                
        return filled
    
    def _fill_indicator_safe(self, series: pd.Series, indicator_name: str) -> pd.Series:
        """Preenche indicadores com valores apropriados"""
        
        # RSI: valor neutro é 50
        if 'rsi' in indicator_name.lower():
            filled = series.ffill()
            return filled.fillna(50)  # Apenas RSI pode usar valor fixo
            
        # ADX: valor baixo indica lateralização
        elif 'adx' in indicator_name.lower():
            filled = series.ffill()
            return filled.fillna(15)  # ADX baixo = sem tendência
            
        # MACD: forward fill apenas
        elif 'macd' in indicator_name.lower():
            return series.ffill().bfill()
            
        # ATR: usar média dos últimos valores
        elif 'atr' in indicator_name.lower():
            filled = series.ffill()
            if filled.isna().any() and filled.notna().any():
                mean_atr = filled.rolling(20, min_periods=1).mean()
                filled = filled.fillna(mean_atr)
            return filled
            
        # Default: forward/backward fill
        else:
            return series.ffill().bfill()
    
    def _is_volume_feature(self, col_name: str) -> bool:
        """Identifica features de volume"""
        return 'volume' in col_name.lower()
    
    def _is_technical_indicator(self, col_name: str) -> bool:
        """Identifica indicadores técnicos"""
        indicators = ['rsi', 'macd', 'adx', 'atr', 'stoch', 'cci']
        return any(indicator in col_name.lower() for indicator in indicators)
    
    def _is_ratio_feature(self, col_name: str) -> bool:
        """Identifica features de ratio"""
        ratios = ['ratio', 'pct', 'percent', 'imbalance']
        return any(ratio in col_name.lower() for ratio in ratios)
    
    def _is_momentum_feature(self, col_name: str) -> bool:
        """Identifica features de momentum"""
        momentum_indicators = ['momentum', 'return', 'change', 'diff']
        return any(indicator in col_name.lower() for indicator in momentum_indicators)
    
    def _fill_volume_safe(self, series: pd.Series) -> pd.Series:
        """Preenche dados de volume - com cuidado"""
        filled = series.ffill()
        
        if filled.isna().all():
            filled = filled.fillna(1)
        else:
            non_zero_median = filled[filled > 0].median()
            if pd.notna(non_zero_median):
                filled = filled.fillna(non_zero_median)
        
        return filled
    
    def _fill_ratio_safe(self, series: pd.Series) -> pd.Series:
        """Preenche ratios com interpolação"""
        return series.interpolate(method='linear', limit=5).ffill().bfill()
    
    def _fill_conservative(self, series: pd.Series) -> pd.Series:
        """Estratégia conservadora para casos gerais"""
        return series.ffill().bfill()
    
    def _log_fill_operation(self, col_name: str, original_nan: int, remaining_nan: int, context: str):
        """Log da operação de preenchimento"""
        if remaining_nan == 0:
            self.logger.debug(f"✅ {col_name}: {original_nan} NaN preenchidos")
        else:
            self.logger.warning(f"⚠️ {col_name}: {remaining_nan}/{original_nan} NaN restantes")
    
    def _fill_momentum_safe(self, series: pd.Series) -> pd.Series:
        """Preenche momentum - zero É aceitável aqui"""
        filled = series.ffill()
        
        # Para momentum, zero significa "sem movimento"
        # É o ÚNICO caso onde fillna(0) é justificável
        if filled.isna().any():
            filled = filled.fillna(0)  # Justificado: momentum neutro
            
        return filled
    
    def _validate_filled_data(self, df: pd.DataFrame, context: str):
        """VALIDAÇÃO CRÍTICA dos dados preenchidos"""
        
        validation_errors = []
        
        for col in df.columns:
            # 1. Verificar se ainda há NaN
            if df[col].isnull().any():
                validation_errors.append(f"ERRO: {col} ainda tem NaN após preenchimento")
                
            # 2. Verificar valores suspeitos
            if self._is_price_feature(col):
                if (df[col] <= 0).any():
                    validation_errors.append(f"ERRO: {col} tem preços <= 0")
                    
            # 3. Verificar excesso de zeros (possível fillna(0) incorreto)
            zero_ratio = (df[col] == 0).sum() / len(df[col])
            if zero_ratio > 0.3 and not self._is_momentum_feature(col):
                validation_errors.append(f"SUSPEITO: {col} tem {zero_ratio:.1%} zeros")
        
        if validation_errors:
            self.logger.error(f"VALIDAÇÃO FALHOU em {context}:")
            for error in validation_errors:
                self.logger.error(f"  • {error}")
            raise ValueError(f"Dados inválidos após preenchimento: {validation_errors}")
        else:
            self.logger.info(f"✅ Validação OK em {context}")
