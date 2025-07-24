#!/usr/bin/env python3
"""
Testes para validar correções de preenchimento de dados
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhanced_smart_fill import EnhancedSmartFillStrategy
from trading_data_validator import TradingDataValidator
import logging

class TestDataFillCorrections:
    """Testa se as correções de preenchimento funcionam"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.logger = logging.getLogger('test')
        self.smart_fill = EnhancedSmartFillStrategy(self.logger)
        self.validator = TradingDataValidator(self.logger)
        
    def test_price_never_zero(self):
        """Testa que preços nunca são preenchidos com zero"""
        # Criar dados de preço com NaN
        price_data = pd.Series([100.0, 101.0, np.nan, np.nan, 102.0])
        
        filled = self.smart_fill._fill_price_safe(price_data)
        
        # Verificar que nenhum preço é zero
        assert (filled > 0).all(), "Preços foram preenchidos com zero!"
        assert filled.notna().all(), "Ainda há NaN em preços"
        
    def test_rsi_neutral_fill(self):
        """Testa que RSI é preenchido com valor neutro"""
        rsi_data = pd.Series([30.0, np.nan, np.nan, 70.0])
        
        filled = self.smart_fill._fill_indicator_safe(rsi_data, 'rsi_14')
        
        # RSI deve estar entre 0 e 100
        assert (filled >= 0).all() and (filled <= 100).all()
        assert filled.notna().all()
        
    def test_volume_positive_fill(self):
        """Testa que volume nunca é negativo"""
        volume_data = pd.Series([1000, np.nan, np.nan, 2000])
        
        filled = self.smart_fill._fill_volume_safe(volume_data)
        
        assert (filled >= 0).all(), "Volume negativo detectado"
        assert filled.notna().all()
        
    def test_momentum_can_be_zero(self):
        """Testa que momentum pode ser zero (única exceção)"""
        momentum_data = pd.Series([0.1, np.nan, np.nan, -0.1])
        
        filled = self.smart_fill._fill_momentum_safe(momentum_data)
        
        # Momentum pode ter zero
        assert filled.notna().all()
        
    def test_validator_catches_bad_data(self):
        """Testa que validador detecta dados ruins"""
        # Dados com preços zero (inválido)
        bad_candles = pd.DataFrame({
            'open': [100, 0, 102],  # Zero é inválido
            'high': [101, 101, 103],
            'low': [99, 99, 101],
            'close': [100, 100, 102],
            'volume': [1000, 1000, 1000]
        })
        
        is_valid, errors = self.validator.validate_data(bad_candles, 'candles')
        
        assert not is_valid, "Validador deveria ter detectado problema"
        assert any('inválidos' in error for error in errors)
        
    def test_no_fillna_zero_in_prices(self):
        """Testa que nunca usamos fillna(0) em preços"""
        price_df = pd.DataFrame({
            'close': [100, np.nan, np.nan, 103],
            'ema_20': [99, np.nan, np.nan, 102]
        })
        
        filled_df = self.smart_fill.fill_missing_values(price_df, 'prices')
        
        # Nenhum preço deve ser zero
        for col in filled_df.columns:
            assert (filled_df[col] > 0).all(), f"Coluna {col} tem zeros"
            
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
