"""
Regime Analyzer para ML Trading System
Detecta regime de mercado baseado em ADX + EMAs conforme especificação do sistema
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class RegimeAnalyzer:
    """
    Analisador de regime de mercado para sistema de trading ML
    
    Implementa detecção de regime baseada em:
    - ADX para força de tendência (threshold = 25)
    - EMAs (9, 20, 50) para direção da tendência
    - Configurações específicas por regime conforme CLAUDE.md
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Thresholds conforme documentação
        self.adx_trend_threshold = 25  # ADX > 25 = tendência
        self.min_candles_required = 50  # Mínimo para análise confiável
        
        # Configurações por regime (conforme CLAUDE.md)
        self.regime_configs = {
            'trend_up': {
                'thresholds': {
                    'confidence': 0.60,
                    'probability': 0.60,
                    'direction': 0.70,
                    'magnitude': 0.003
                },
                'risk_reward': 2.0,
                'strategy': 'Follow trend'
            },
            'trend_down': {
                'thresholds': {
                    'confidence': 0.60, 
                    'probability': 0.60,
                    'direction': 0.70,
                    'magnitude': 0.003
                },
                'risk_reward': 2.0,
                'strategy': 'Follow trend'
            },
            'range': {
                'thresholds': {
                    'confidence': 0.60,
                    'probability': 0.55,
                    'direction': 0.50,
                    'magnitude': 0.0015
                },
                'risk_reward': 1.5,
                'strategy': 'Trade reversals at boundaries'
            },
            'undefined': {
                'thresholds': {
                    'confidence': 0.80,  # Mais conservador
                    'probability': 0.80,
                    'direction': 0.80,
                    'magnitude': 0.005
                },
                'risk_reward': 1.0,
                'strategy': 'HOLD (no trading)'
            }
        }
        
        self.logger.info("RegimeAnalyzer inicializado com configurações CLAUDE.md")
    
    def analyze_market(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analisa dados de mercado e retorna informações completas de regime
        
        Args:
            data: DataFrame com candles e indicadores técnicos
            
        Returns:
            Dict com informações detalhadas do regime detectado
        """
        try:
            if data.empty or len(data) < self.min_candles_required:
                self.logger.warning(f"Dados insuficientes para análise: {len(data)} candles (mín: {self.min_candles_required})")
                return self._get_default_regime("insufficient_data")
            
            # Extrair indicadores necessários
            indicators = self._extract_indicators(data)
            
            if not self._validate_indicators(indicators):
                self.logger.warning("Indicadores insuficientes ou inválidos")
                return self._get_default_regime("invalid_indicators")
            
            # Detectar regime baseado em ADX + EMAs
            regime_info = self._classify_regime(indicators, data)
            
            # Adicionar metadados
            regime_info.update({
                'analysis_timestamp': pd.Timestamp.now(),
                'data_quality': self._assess_data_quality(data),
                'indicators_used': list(indicators.keys()),
                'total_candles': len(data)
            })
            
            self.logger.info(f"[REGIME] {regime_info['regime'].upper()}: "
                           f"ADX={indicators['adx']:.1f}, "
                           f"Conf={regime_info['confidence']:.2f}, "
                           f"Dir={regime_info['direction']}")
            
            return regime_info
            
        except Exception as e:
            self.logger.error(f"Erro na análise de regime: {e}")
            return self._get_default_regime("analysis_error")
    
    def _extract_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extrai indicadores necessários dos dados"""
        indicators = {}
        
        try:
            # ADX - prioridade para colunas específicas
            adx_value = None
            for col in ['adx', 'adx_14', 'ADX']:
                if col in data.columns:
                    adx_series = data[col].dropna()
                    if not adx_series.empty:
                        adx_value = float(adx_series.iloc[-1])
                        break
            
            # Se não encontrou ADX, calcular básico
            if adx_value is None:
                adx_value = self._calculate_simple_adx(data)
            
            indicators['adx'] = adx_value
            
            # EMAs - buscar em várias nomenclaturas
            for period in [9, 20, 50]:
                ema_value = None
                for col in [f'ema_{period}', f'ema{period}', f'EMA_{period}']:
                    if col in data.columns:
                        ema_series = data[col].dropna()
                        if not ema_series.empty:
                            ema_value = float(ema_series.iloc[-1])
                            break
                
                # Se não encontrou EMA, calcular
                if ema_value is None and 'close' in data.columns:
                    ema_value = float(data['close'].ewm(span=period).mean().iloc[-1])
                
                indicators[f'ema_{period}'] = ema_value
            
            # Preço atual
            if 'close' in data.columns:
                indicators['price'] = float(data['close'].iloc[-1])
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Erro extraindo indicadores: {e}")
            return {}
    
    def _calculate_simple_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calcula ADX simplificado quando não disponível"""
        try:
            if not all(col in data.columns for col in ['high', 'low', 'close']):
                return 20.0  # Default ADX neutro
            
            # Garantir que temos dados suficientes
            if len(data) < period * 2:
                return 20.0
                
            high = data['high'].iloc[-period*3:] if len(data) > period*3 else data['high']
            low = data['low'].iloc[-period*3:] if len(data) > period*3 else data['low']  
            close = data['close'].iloc[-period*3:] if len(data) > period*3 else data['close']
            
            # True Range com verificação de tamanho
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            # Usar apenas os índices válidos
            tr_data = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).fillna(0)
            tr = tr_data.max(axis=1).rolling(period, min_periods=1).mean()
            
            # Directional Movement com proteção
            h_diff = high.diff()
            l_diff = low.diff() * -1  # Inverter para positivo
            
            dm_plus = np.where((h_diff > l_diff) & (h_diff > 0), h_diff, 0)
            dm_minus = np.where((l_diff > h_diff) & (l_diff > 0), l_diff, 0)
            
            # Garantir mesmo tamanho
            dm_plus_series = pd.Series(dm_plus, index=tr.index)
            dm_minus_series = pd.Series(dm_minus, index=tr.index)
            
            dm_plus_smooth = dm_plus_series.rolling(period, min_periods=1).mean()
            dm_minus_smooth = dm_minus_series.rolling(period, min_periods=1).mean()
            
            # DI e ADX com proteção contra divisão por zero
            di_plus = 100 * dm_plus_smooth / (tr + 1e-10)  # Evitar divisão por zero
            di_minus = 100 * dm_minus_smooth / (tr + 1e-10)
            
            # DX com proteção
            di_sum = di_plus + di_minus + 1e-10
            dx = 100 * abs(di_plus - di_minus) / di_sum
            adx = dx.rolling(period, min_periods=1).mean()
            
            # Retornar último valor válido
            final_adx = adx.dropna()
            return float(final_adx.iloc[-1]) if not final_adx.empty else 20.0
            
        except Exception as e:
            self.logger.error(f"Erro calculando ADX: {e}")
            return 20.0
    
    def _validate_indicators(self, indicators: Dict[str, float]) -> bool:
        """Valida se indicadores estão disponíveis e válidos"""
        required = ['adx', 'ema_9', 'ema_20', 'price']
        
        for req in required:
            if req not in indicators or indicators[req] is None:
                self.logger.warning(f"Indicador obrigatório ausente: {req}")
                return False
                
            if not isinstance(indicators[req], (int, float)) or np.isnan(indicators[req]):
                self.logger.warning(f"Indicador inválido: {req} = {indicators[req]}")
                return False
        
        return True
    
    def _classify_regime(self, indicators: Dict[str, float], data: pd.DataFrame) -> Dict[str, Any]:
        """Classifica regime baseado em ADX + EMAs (conforme CLAUDE.md)"""
        
        adx = indicators['adx']
        ema_9 = indicators['ema_9']
        ema_20 = indicators['ema_20']
        ema_50 = indicators.get('ema_50', ema_20)  # Fallback se EMA50 não existir
        price = indicators['price']
        
        # 1. TREND REGIME (ADX > 25, EMAs alinhadas)
        if adx > self.adx_trend_threshold:
            
            # Trend UP: EMA9 > EMA20 > EMA50
            if ema_9 > ema_20 > ema_50:
                confidence = min(0.8, 0.5 + (adx - 25) / 75)  # 0.5-0.8 baseado em ADX
                return {
                    'regime': 'trend_up',
                    'direction': 1,
                    'confidence': confidence,
                    'strength': 'strong' if adx > 35 else 'moderate',
                    'adx_value': adx,
                    'ema_alignment': 'bullish',
                    **self.regime_configs['trend_up']
                }
            
            # Trend DOWN: EMA9 < EMA20 < EMA50  
            elif ema_9 < ema_20 < ema_50:
                confidence = min(0.8, 0.5 + (adx - 25) / 75)
                return {
                    'regime': 'trend_down', 
                    'direction': -1,
                    'confidence': confidence,
                    'strength': 'strong' if adx > 35 else 'moderate',
                    'adx_value': adx,
                    'ema_alignment': 'bearish',
                    **self.regime_configs['trend_down']
                }
        
        # 2. RANGE REGIME (ADX < 25, independente de EMAs)
        if adx < self.adx_trend_threshold:
            confidence = 0.6  # Confiança fixa para range
            return {
                'regime': 'range',
                'direction': 0,
                'confidence': confidence,
                'strength': 'weak',
                'adx_value': adx,
                'ema_alignment': 'neutral',
                **self.regime_configs['range']
            }
        
        # 3. UNDEFINED REGIME (ADX alto mas EMAs não alinhadas)
        return {
            'regime': 'undefined',
            'direction': 0,
            'confidence': 0.3,
            'strength': 'mixed',
            'adx_value': adx,
            'ema_alignment': 'mixed',
            **self.regime_configs['undefined']
        }
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Avalia qualidade dos dados para análise"""
        try:
            total_cells = data.size
            nan_count = data.isna().sum().sum()
            
            return {
                'total_candles': len(data),
                'nan_ratio': nan_count / total_cells if total_cells > 0 else 1.0,
                'completeness': (total_cells - nan_count) / total_cells if total_cells > 0 else 0.0,
                'time_coverage': str(data.index.max() - data.index.min()) if len(data) > 1 else '0',
                'quality_score': min(1.0, len(data) / self.min_candles_required)
            }
        except Exception as e:
            self.logger.error(f"Erro avaliando qualidade: {e}")
            return {'quality_score': 0.0}
    
    def _get_default_regime(self, reason: str = "unknown") -> Dict[str, Any]:
        """Retorna regime padrão quando análise falha"""
        self.logger.warning(f"Usando regime padrão - Motivo: {reason}")
        
        return {
            'regime': 'undefined',
            'direction': 0,
            'confidence': 0.0,
            'strength': 'undefined',
            'adx_value': 20.0,
            'ema_alignment': 'unknown',
            'analysis_timestamp': pd.Timestamp.now(),
            'failure_reason': reason,
            'data_quality': {'quality_score': 0.0},
            **self.regime_configs['undefined']
        }