"""
🛡️ VALIDADOR DE DADOS PARA PRODUÇÃO
Sistema de Trading ML v2.0 - Anti-Dummy Data

CRÍTICO: Este módulo DEVE ser usado em TODOS os pontos de entrada de dados
antes de qualquer processamento ML ou geração de sinais de trading.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import warnings


class ProductionDataError(Exception):
    """Exceção crítica para dados não adequados para produção"""
    pass


class ProductionDataValidator:
    """
    Validador rigoroso para dados de produção
    
    OBJETIVO: Garantir que NENHUM dado sintético, dummy ou mock
    seja usado em operações reais de trading.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('ProductionValidator')
        self.production_mode = os.getenv('TRADING_PRODUCTION_MODE', 'False').lower() == 'true'
        self.strict_mode = os.getenv('STRICT_VALIDATION', 'True').lower() == 'true'
        
        if self.production_mode:
            self.logger.info("🛡️ MODO PRODUÇÃO ATIVADO - VALIDAÇÃO RIGOROSA")
        else:
            self.logger.warning("⚠️ MODO DESENVOLVIMENTO - DADOS PODEM SER SINTÉTICOS")
    
    def validate_trading_data(self, 
                            data: pd.DataFrame, 
                            source: str,
                            data_type: str = 'market') -> bool:
        """
        Validação principal para dados de trading
        
        Args:
            data: DataFrame com dados de mercado
            source: Fonte dos dados ('ProfitDLL', 'WebSocket', 'API', etc.)
            data_type: Tipo de dados ('market', 'historical', 'realtime')
            
        Returns:
            bool: True se dados são seguros para produção
            
        Raises:
            ProductionDataError: Se dados dummy/unsafe detectados
        """
        
        if data.empty:
            raise ProductionDataError(f"DataFrame vazio recebido de {source}")
        
        # 🔍 BATERIA DE TESTES ANTI-DUMMY
        validation_results = {
            'synthetic_patterns': self._detect_synthetic_patterns(data),
            'timestamp_integrity': self._validate_timestamps(data, data_type),
            'price_integrity': self._validate_price_integrity(data),
            'volume_patterns': self._validate_volume_patterns(data),
            'data_source': self._validate_data_source(source),
            'nan_patterns': self._validate_nan_patterns(data)
        }
        
        # 🚨 ANALISAR RESULTADOS
        failed_validations = []
        for test_name, result in validation_results.items():
            if result['failed']:
                failed_validations.append(f"{test_name}: {result['reason']}")
        
        if failed_validations:
            error_msg = (
                f"🚨 DADOS SUSPEITOS DETECTADOS EM {source.upper()}\n"
                f"Tipo: {data_type}\n"
                f"Falhas: {'; '.join(failed_validations)}\n"
                f"⛔ TRADING BLOQUEADO POR SEGURANÇA"
            )
            
            if self.production_mode:
                raise ProductionDataError(error_msg)
            else:
                self.logger.warning(error_msg)
                return False
        
        self.logger.info(f"✅ Dados validados: {source} - {data_type} - {len(data)} registros")
        return True
    
    def _detect_synthetic_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detecta padrões típicos de dados sintéticos"""
        
        try:
            # ❌ TESTE 1: Volume muito uniforme (padrão np.random.uniform)
            if 'volume' in data.columns:
                volume_cv = data['volume'].std() / data['volume'].mean()
                if volume_cv < 0.15:  # Coeficiente de variação muito baixo
                    return {
                        'failed': True, 
                        'reason': f'Volume uniforme demais (CV={volume_cv:.3f})'
                    }
            
            # ❌ TESTE 2: Spreads muito constantes (padrão sintético)
            if all(col in data.columns for col in ['high', 'low', 'close']):
                spreads = (data['high'] - data['low']) / data['close']
                
                # Verificar se há spreads válidos
                valid_spreads = spreads[spreads > 0]
                if len(valid_spreads) == 0:
                    # Se todos os spreads são zero, pode ser legítimo para dados de minuto
                    return {'failed': False, 'reason': 'Spreads zero aceitáveis para dados de alta frequência'}
                
                spread_cv = valid_spreads.std() / valid_spreads.mean()
                
                # Threshold muito mais relaxado para dados reais de alta frequência
                threshold = 0.001 if self.production_mode else 0.00001
                
                if spread_cv < threshold:
                    # Em desenvolvimento, apenas avisar
                    if not self.production_mode:
                        self.logger.warning(f"Spreads uniformes detectados (CV={spread_cv:.6f}) - aceitável em desenvolvimento")
                        return {'failed': False, 'reason': 'OK'}
                    else:
                        return {
                            'failed': True,
                            'reason': f'Spreads uniformes demais (CV={spread_cv:.6f})'
                        }
            
            # ❌ TESTE 3: Detectar sequências suspeitas
            if 'price' in data.columns:
                price_diffs = data['price'].diff()
                # Muito poucos valores únicos = possível simulação
                unique_ratio = price_diffs.nunique() / len(price_diffs)
                if unique_ratio < 0.7:
                    return {
                        'failed': True,
                        'reason': f'Mudanças de preço pouco variadas ({unique_ratio:.2f})'
                    }
            
            # ❌ TESTE 4: Detectar padrões de np.random.seed
            if 'volume' in data.columns and len(data) > 10:
                # Teste de aleatoriedade usando runs test
                median_vol = data['volume'].median()
                runs = self._count_runs_above_below_median(data['volume'], median_vol)
                expected_runs = len(data) / 2
                if abs(runs - expected_runs) / expected_runs > 0.3:
                    return {
                        'failed': True,
                        'reason': 'Padrão de volume não-aleatório detectado'
                    }
            
            return {'failed': False, 'reason': 'OK'}
            
        except Exception as e:
            return {'failed': True, 'reason': f'Erro na detecção sintética: {str(e)}'}
    
    def _validate_timestamps(self, data: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Valida integridade dos timestamps"""
        
        try:
            # Verificar se há coluna de timestamp ou index datetime
            timestamp_col = None
            if hasattr(data.index, 'dtype') and 'datetime' in str(data.index.dtype):
                timestamps = data.index
            elif 'timestamp' in data.columns:
                timestamps = pd.to_datetime(data['timestamp'])
            elif 'datetime' in data.columns:
                timestamps = pd.to_datetime(data['datetime'])
            else:
                return {'failed': True, 'reason': 'Nenhuma coluna de timestamp encontrada'}
            
            # ❌ TESTE 1: Timestamps muito antigos para tempo real
            if data_type == 'realtime':
                latest_time = timestamps.max()
                if pd.isna(latest_time):
                    return {'failed': True, 'reason': 'Timestamp inválido (NaT)'}
                
                # Remover timezone para comparação
                if hasattr(latest_time, 'tz') and latest_time.tz is not None:
                    latest_time = latest_time.replace(tzinfo=None)
                
                time_diff = datetime.now() - latest_time
                if time_diff > timedelta(minutes=10):
                    return {
                        'failed': True,
                        'reason': f'Dados muito antigos ({time_diff}) para tempo real'
                    }
            
            # ❌ TESTE 2: Intervalos muito regulares (suspeito)
            if len(timestamps) > 5:
                try:
                    # Converter para Series para usar diff de forma segura
                    ts_series = pd.Series(pd.to_datetime(timestamps))
                    intervals = ts_series.diff().dropna()
                        
                    if len(intervals) > 0:
                        # Todos os intervalos iguais = possível simulação
                        unique_intervals = intervals.nunique()
                        if unique_intervals == 1 and len(intervals) > 10:
                            return {
                                'failed': True,
                                'reason': 'Intervalos de tempo perfeitamente regulares (suspeito)'
                            }
                except Exception:
                    # Se não conseguir calcular intervalos, pular teste
                    pass
            
            # ❌ TESTE 3: Timestamps duplicados
            if timestamps.duplicated().any():
                return {'failed': True, 'reason': 'Timestamps duplicados detectados'}
            
            return {'failed': False, 'reason': 'OK'}
            
        except Exception as e:
            return {'failed': True, 'reason': f'Erro na validação de timestamp: {str(e)}'}
    
    def _validate_price_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Valida integridade dos dados de preço"""
        
        try:
            price_cols = []
            for col in ['price', 'close', 'open', 'high', 'low']:
                if col in data.columns:
                    price_cols.append(col)
            
            if not price_cols:
                return {'failed': False, 'reason': 'Nenhuma coluna de preço para validar'}
            
            for col in price_cols:
                prices = data[col]
                
                # ❌ TESTE 1: Preços zeros ou negativos
                if (prices <= 0).any():
                    return {
                        'failed': True,
                        'reason': f'Preços inválidos (≤0) encontrados em {col}'
                    }
                
                # ❌ TESTE 2: Mudanças impossíveis (>50% em poucos registros)
                if len(prices) > 1:
                    price_changes = prices.pct_change().abs()
                    extreme_changes = (price_changes > 0.5).sum()
                    if extreme_changes > len(prices) * 0.05:  # Mais de 5% mudanças extremas
                        return {
                            'failed': True,
                            'reason': f'Mudanças de preço extremas em {col} ({extreme_changes} casos)'
                        }
                
                # ❌ TESTE 3: Preços muito constantes
                if prices.nunique() < len(prices) * 0.8 and len(prices) > 10:
                    return {
                        'failed': True,
                        'reason': f'Preços pouco variados em {col} (possível simulação)'
                    }
            
            # ❌ TESTE 4: Validar OHLC se disponível
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # High deve ser >= max(open, close)
                invalid_high = (data['high'] < data[['open', 'close']].max(axis=1)).any()
                # Low deve ser <= min(open, close)  
                invalid_low = (data['low'] > data[['open', 'close']].min(axis=1)).any()
                
                if invalid_high or invalid_low:
                    return {
                        'failed': True,
                        'reason': 'Inconsistência em dados OHLC (high/low inválidos)'
                    }
            
            return {'failed': False, 'reason': 'OK'}
            
        except Exception as e:
            return {'failed': True, 'reason': f'Erro na validação de preços: {str(e)}'}
    
    def _validate_volume_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Valida padrões de volume"""
        
        try:
            volume_cols = [col for col in data.columns if 'volume' in col.lower()]
            
            if not volume_cols:
                return {'failed': False, 'reason': 'Nenhuma coluna de volume para validar'}
            
            for vol_col in volume_cols:
                volume = data[vol_col]
                
                # ❌ TESTE 1: Volume negativo
                if (volume < 0).any():
                    return {
                        'failed': True,
                        'reason': f'Volume negativo encontrado em {vol_col}'
                    }
                
                # ❌ TESTE 2: Muitos zeros (suspeito para dados reais)
                # EXCEÇÃO: buy_volume e sell_volume podem ter muitos zeros legitimamente
                if vol_col not in ['buy_volume', 'sell_volume']:
                    zero_pct = (volume == 0).sum() / len(volume)
                    if zero_pct > 0.3:  # Mais de 30% zeros
                        return {
                            'failed': True,
                            'reason': f'Muitos zeros em {vol_col} ({zero_pct:.1%}) - suspeito'
                        }
                else:
                    # Para buy/sell volume, só falhar se TODOS forem zero
                    if (volume == 0).all():
                        # Se é desenvolvimento, apenas avisar
                        if self.production_mode:
                            return {
                                'failed': True,
                                'reason': f'Todos os valores de {vol_col} são zero'
                            }
                        else:
                            self.logger.warning(f"{vol_col} com todos zeros - aceitável em desenvolvimento")
                
                # ❌ TESTE 3: Volume muito uniforme
                if len(volume) > 10:
                    vol_cv = volume.std() / volume.mean()
                    if vol_cv < 0.2:  # Coeficiente de variação muito baixo
                        return {
                            'failed': True,
                            'reason': f'Volume uniforme demais em {vol_col} (CV={vol_cv:.3f})'
                        }
            
            return {'failed': False, 'reason': 'OK'}
            
        except Exception as e:
            return {'failed': True, 'reason': f'Erro na validação de volume: {str(e)}'}
    
    def _validate_data_source(self, source: str) -> Dict[str, Any]:
        """Valida se a fonte de dados é confiável para produção"""
        
        # Lista de fontes aprovadas para produção
        approved_sources = {
            'ProfitDLL',
            'MetaTrader5',  
            'B3API',
            'WebSocketReal',
            'BrokerAPI',
            'MarketDataFeed'
        }
        
        # Lista de fontes proibidas em produção
        forbidden_sources = {
            'mock', 'fake', 'dummy', 'test', 'simulation', 
            'synthetic', 'random', 'generator'
        }
        
        source_lower = source.lower()
        
        # ❌ Verificar fontes proibidas
        for forbidden in forbidden_sources:
            if forbidden in source_lower:
                return {
                    'failed': True,
                    'reason': f'Fonte proibida detectada: {source} (contém "{forbidden}")'
                }
        
        # ⚠️ Verificar se é fonte aprovada (apenas aviso)
        if self.production_mode and source not in approved_sources:
            self.logger.warning(f"Fonte não reconhecida: {source}")
        
        return {'failed': False, 'reason': 'OK'}
    
    def _validate_nan_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Valida padrões de NaN que podem indicar fillna perigoso"""
        
        try:
            # ❌ TESTE 1: Muitos NaN podem indicar dados incompletos
            total_cells = data.size
            nan_count = data.isnull().sum().sum()
            nan_ratio = nan_count / total_cells
            
            if nan_ratio > 0.1:  # Mais de 10% NaN
                return {
                    'failed': True,
                    'reason': f'Muitos valores NaN ({nan_ratio:.1%}) - dados incompletos'
                }
            
            # ❌ TESTE 2: Colunas com padrões suspeitos de fillna
            for col in data.columns:
                if data[col].dtype in ['float64', 'int64']:
                    # Verificar se há muitos zeros (possível fillna(0))
                    zero_ratio = (data[col] == 0).sum() / len(data)
                    if zero_ratio > 0.5:  # Mais de 50% zeros
                        return {
                            'failed': True,
                            'reason': f'Coluna {col} com muitos zeros ({zero_ratio:.1%}) - possível fillna(0)'
                        }
            
            return {'failed': False, 'reason': 'OK'}
            
        except Exception as e:
            return {'failed': True, 'reason': f'Erro na validação NaN: {str(e)}'}
    
    def _count_runs_above_below_median(self, series: pd.Series, median: float) -> int:
        """Conta runs acima/abaixo da mediana para teste de aleatoriedade"""
        
        above_median = series > median
        runs = 1
        
        for i in range(1, len(above_median)):
            if above_median.iloc[i] != above_median.iloc[i-1]:
                runs += 1
                
        return runs
    
    def validate_feature_data(self, features: pd.DataFrame) -> bool:
        """
        Validação específica para dados de features ML
        
        Args:
            features: DataFrame com features calculadas
            
        Returns:
            bool: True se features são seguras
        """
        
        if features.empty:
            raise ProductionDataError("DataFrame de features vazio")
        
        # ❌ Detectar features com fillna suspeito
        suspicious_cols = []
        
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                # RSI com valor fixo 50 (suspeito)
                if 'rsi' in col.lower():
                    rsi_50_ratio = (features[col] == 50).sum() / len(features)
                    if rsi_50_ratio > 0.3:
                        suspicious_cols.append(f"{col}: muitos valores 50 (fillna suspeito)")
                
                # Features com muitos zeros
                zero_ratio = (features[col] == 0).sum() / len(features)
                if zero_ratio > 0.4:
                    suspicious_cols.append(f"{col}: muitos zeros ({zero_ratio:.1%})")
                
                # Features com valores constantes
                if features[col].nunique() < 3 and len(features) > 10:
                    suspicious_cols.append(f"{col}: valores pouco variados")
        
        if suspicious_cols:
            error_msg = f"Features suspeitas detectadas: {'; '.join(suspicious_cols)}"
            
            if self.production_mode:
                raise ProductionDataError(error_msg)
            else:
                self.logger.warning(error_msg)
                return False
        
        return True


def enforce_production_mode():
    """
    Força modo produção baseado em variável de ambiente
    DEVE ser chamado no início de qualquer sistema de trading
    """
    
    production_mode = os.getenv('TRADING_PRODUCTION_MODE', 'False').lower() == 'true'
    
    if production_mode:
        # Configurações rigorosas para produção
        os.environ['STRICT_VALIDATION'] = 'True'
        warnings.filterwarnings('error')  # Warnings viram erros
        
        # Log de inicialização
        logger = logging.getLogger('ProductionMode')
        logger.critical("🛡️ MODO PRODUÇÃO ATIVADO - VALIDAÇÃO RIGOROSA DE DADOS")
        logger.critical("⚠️ QUALQUER DADO SUSPEITO RESULTARÁ EM BLOQUEIO DO SISTEMA")
        
        return True
    else:
        logger = logging.getLogger('DevelopmentMode')  
        logger.warning("⚠️ MODO DESENVOLVIMENTO - DADOS PODEM NÃO SER REAIS")
        return False


# Instância global do validador
production_validator = ProductionDataValidator()
