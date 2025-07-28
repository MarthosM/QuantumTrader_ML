"""
Data Validator - Validador de Qualidade de Dados
===============================================

Este módulo implementa validações avançadas para garantir
a qualidade dos dados antes do armazenamento e treinamento.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ValidationResult:
    """Resultado da validação"""
    is_valid: bool
    quality_score: float
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, float]
    suggestions: List[str]


@dataclass
class ValidationRule:
    """Regra de validação"""
    name: str
    check_function: callable
    severity: str  # 'error', 'warning'
    auto_fix: bool = False
    fix_function: Optional[callable] = None


class DataValidator:
    """Validador de dados de trading"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa validador
        
        Args:
            config: Configurações de validação
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configurações padrão
        self.thresholds = {
            'min_quality_score': 0.7,
            'max_missing_pct': 0.05,  # 5%
            'max_duplicate_pct': 0.01,  # 1%
            'max_outlier_pct': 0.001,  # 0.1%
            'min_trades_per_hour': 100,
            'max_spread_pct': 0.01,  # 1%
            'max_time_gap_minutes': 5
        }
        
        # Atualizar com config
        if 'thresholds' in self.config:
            self.thresholds.update(self.config['thresholds'])
        
        # Regras de validação
        self.rules = self._init_validation_rules()
        
        # Cache de resultados
        self._cache = {}
    
    def validate_data(self,
                     data: pd.DataFrame,
                     data_type: str,
                     symbol: str,
                     date: Optional[datetime] = None,
                     auto_fix: bool = False) -> ValidationResult:
        """
        Valida dados completos
        
        Args:
            data: DataFrame para validar
            data_type: Tipo de dados (trades, candles, book)
            symbol: Símbolo do ativo
            date: Data dos dados
            auto_fix: Se deve tentar corrigir automaticamente
            
        Returns:
            ValidationResult com detalhes
        """
        self.logger.info(f"Validando {len(data)} registros de {data_type} para {symbol}")
        
        # Inicializar resultado
        errors = []
        warnings = []
        metrics = {}
        suggestions = []
        
        # Validações básicas
        if data.empty:
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                errors=["DataFrame vazio"],
                warnings=[],
                metrics={},
                suggestions=["Verificar fonte de dados"]
            )
        
        # Executar regras específicas por tipo
        if data_type == 'trades':
            rules = self._get_trade_rules()
        elif data_type == 'candles':
            rules = self._get_candle_rules()
        elif data_type == 'book':
            rules = self._get_book_rules()
        else:
            rules = self._get_generic_rules()
        
        # Executar cada regra
        fixed_data = data.copy() if auto_fix else data
        quality_scores = []
        
        for rule in rules:
            try:
                # Executar validação
                result = rule.check_function(fixed_data, self.thresholds)
                
                if not result['passed']:
                    if rule.severity == 'error':
                        errors.append(f"{rule.name}: {result['message']}")
                        quality_scores.append(0.0)
                    else:
                        warnings.append(f"{rule.name}: {result['message']}")
                        quality_scores.append(0.5)
                    
                    # Tentar corrigir se habilitado
                    if auto_fix and rule.auto_fix and rule.fix_function:
                        fixed_data = rule.fix_function(fixed_data)
                        suggestions.append(f"Aplicada correção automática: {rule.name}")
                else:
                    quality_scores.append(1.0)
                
                # Adicionar métricas
                if 'metrics' in result:
                    metrics.update(result['metrics'])
                    
            except Exception as e:
                self.logger.error(f"Erro na regra {rule.name}: {e}")
                warnings.append(f"Falha ao executar {rule.name}")
                quality_scores.append(0.5)
        
        # Calcular score final
        quality_score = np.mean(quality_scores) if quality_scores else 0.0
        
        # Análise adicional
        analysis = self._analyze_data_patterns(fixed_data, data_type)
        metrics.update(analysis['metrics'])
        suggestions.extend(analysis['suggestions'])
        
        # Determinar validade
        is_valid = (
            len(errors) == 0 and 
            quality_score >= self.thresholds['min_quality_score']
        )
        
        # Se corrigido, retornar dados corrigidos
        if auto_fix and fixed_data is not data:
            metrics['data_corrected'] = True
            metrics['corrections_applied'] = len(suggestions)
        
        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            suggestions=suggestions
        )
    
    def validate_consistency(self,
                           trades_df: pd.DataFrame,
                           candles_df: pd.DataFrame,
                           tolerance: float = 0.01) -> ValidationResult:
        """
        Valida consistência entre trades e candles
        
        Args:
            trades_df: DataFrame de trades
            candles_df: DataFrame de candles
            tolerance: Tolerância para diferenças (1%)
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Agrupar trades por minuto
            trades_df['minute'] = pd.to_datetime(trades_df['datetime']).dt.floor('min')
            
            for _, candle in candles_df.iterrows():
                candle_time = candle['datetime']
                
                # Buscar trades do minuto
                minute_trades = trades_df[trades_df['minute'] == candle_time]
                
                if minute_trades.empty:
                    warnings.append(f"Sem trades para candle {candle_time}")
                    continue
                
                # Validar OHLC
                trade_high = minute_trades['price'].max()
                trade_low = minute_trades['price'].min()
                trade_open = minute_trades.iloc[0]['price']
                trade_close = minute_trades.iloc[-1]['price']
                trade_volume = minute_trades['volume'].sum()
                
                # Comparar com candle
                if abs(candle['high'] - trade_high) / trade_high > tolerance:
                    errors.append(f"High inconsistente em {candle_time}: "
                                f"candle={candle['high']:.2f}, trades={trade_high:.2f}")
                
                if abs(candle['low'] - trade_low) / trade_low > tolerance:
                    errors.append(f"Low inconsistente em {candle_time}: "
                                f"candle={candle['low']:.2f}, trades={trade_low:.2f}")
                
                if abs(candle['volume'] - trade_volume) / trade_volume > tolerance * 10:
                    warnings.append(f"Volume inconsistente em {candle_time}: "
                                  f"candle={candle['volume']}, trades={trade_volume}")
            
            # Calcular métricas
            metrics['consistency_score'] = 1.0 - (len(errors) / len(candles_df))
            metrics['candles_validated'] = len(candles_df)
            metrics['inconsistencies'] = len(errors)
            
        except Exception as e:
            errors.append(f"Erro na validação de consistência: {e}")
            metrics['consistency_score'] = 0.0
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=metrics.get('consistency_score', 0.0),
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            suggestions=[]
        )
    
    def _init_validation_rules(self) -> List[ValidationRule]:
        """Inicializa regras de validação"""
        return [
            # Estrutura básica
            ValidationRule(
                name="Colunas obrigatórias",
                check_function=self._check_required_columns,
                severity="error"
            ),
            
            # Tipos de dados
            ValidationRule(
                name="Tipos de dados",
                check_function=self._check_data_types,
                severity="error",
                auto_fix=True,
                fix_function=self._fix_data_types
            ),
            
            # Dados faltantes
            ValidationRule(
                name="Dados faltantes",
                check_function=self._check_missing_data,
                severity="warning",
                auto_fix=True,
                fix_function=self._fix_missing_data
            ),
            
            # Duplicatas
            ValidationRule(
                name="Duplicatas",
                check_function=self._check_duplicates,
                severity="error",
                auto_fix=True,
                fix_function=self._fix_duplicates
            ),
            
            # Outliers
            ValidationRule(
                name="Outliers",
                check_function=self._check_outliers,
                severity="warning"
            ),
            
            # Sequência temporal
            ValidationRule(
                name="Sequência temporal",
                check_function=self._check_time_sequence,
                severity="error",
                auto_fix=True,
                fix_function=self._fix_time_sequence
            ),
            
            # Horário de mercado
            ValidationRule(
                name="Horário de mercado",
                check_function=self._check_market_hours,
                severity="warning"
            )
        ]
    
    def _get_trade_rules(self) -> List[ValidationRule]:
        """Regras específicas para trades"""
        return self.rules + [
            ValidationRule(
                name="Volume de trades",
                check_function=self._check_trade_volume,
                severity="warning"
            ),
            
            ValidationRule(
                name="Preços negativos",
                check_function=self._check_negative_prices,
                severity="error"
            ),
            
            ValidationRule(
                name="Spread bid-ask",
                check_function=self._check_spread,
                severity="warning"
            )
        ]
    
    def _get_candle_rules(self) -> List[ValidationRule]:
        """Regras específicas para candles"""
        return self.rules + [
            ValidationRule(
                name="Consistência OHLC",
                check_function=self._check_ohlc_consistency,
                severity="error",
                auto_fix=True,
                fix_function=self._fix_ohlc_consistency
            ),
            
            ValidationRule(
                name="Gaps de preço",
                check_function=self._check_price_gaps,
                severity="warning"
            )
        ]
    
    def _get_book_rules(self) -> List[ValidationRule]:
        """Regras específicas para book"""
        return self.rules + [
            ValidationRule(
                name="Profundidade do book",
                check_function=self._check_book_depth,
                severity="warning"
            ),
            
            ValidationRule(
                name="Spread do book",
                check_function=self._check_book_spread,
                severity="warning"
            )
        ]
    
    def _get_generic_rules(self) -> List[ValidationRule]:
        """Regras genéricas"""
        return self.rules
    
    # Funções de validação
    def _check_required_columns(self, df: pd.DataFrame, thresholds: Dict) -> Dict:
        """Verifica colunas obrigatórias"""
        # Definir colunas por tipo (simplificado - expandir conforme necessário)
        required = {
            'trades': ['datetime', 'price', 'volume'],
            'candles': ['datetime', 'open', 'high', 'low', 'close', 'volume'],
            'book': ['datetime', 'bid_price', 'ask_price', 'bid_volume', 'ask_volume']
        }
        
        # Verificar qualquer conjunto básico
        basic_cols = {'datetime', 'price', 'volume'}
        missing = basic_cols - set(df.columns)
        
        return {
            'passed': len(missing) == 0,
            'message': f"Colunas faltantes: {missing}" if missing else "OK",
            'metrics': {'missing_columns': len(missing)}
        }
    
    def _check_data_types(self, df: pd.DataFrame, thresholds: Dict) -> Dict:
        """Verifica tipos de dados"""
        issues = []
        
        # Verificar datetime
        if 'datetime' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                issues.append("Coluna datetime não é timestamp")
        
        # Verificar numéricos
        numeric_cols = ['price', 'volume', 'open', 'high', 'low', 'close']
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append(f"Coluna {col} não é numérica")
        
        return {
            'passed': len(issues) == 0,
            'message': "; ".join(issues) if issues else "OK",
            'metrics': {'type_issues': len(issues)}
        }
    
    def _check_missing_data(self, df: pd.DataFrame, thresholds: Dict) -> Dict:
        """Verifica dados faltantes"""
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        
        return {
            'passed': missing_pct <= thresholds['max_missing_pct'],
            'message': f"Dados faltantes: {missing_pct:.2%}",
            'metrics': {'missing_pct': missing_pct}
        }
    
    def _check_duplicates(self, df: pd.DataFrame, thresholds: Dict) -> Dict:
        """Verifica duplicatas"""
        if 'datetime' in df.columns:
            duplicates = df.duplicated(subset=['datetime']).sum()
            dup_pct = duplicates / len(df) if len(df) > 0 else 0
            
            return {
                'passed': dup_pct <= thresholds['max_duplicate_pct'],
                'message': f"Duplicatas: {duplicates} ({dup_pct:.2%})",
                'metrics': {'duplicate_pct': dup_pct, 'duplicates': duplicates}
            }
        
        return {'passed': True, 'message': "OK", 'metrics': {}}
    
    def _check_outliers(self, df: pd.DataFrame, thresholds: Dict) -> Dict:
        """Verifica outliers usando IQR"""
        outliers = 0
        total_points = 0
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['price', 'volume', 'open', 'high', 'low', 'close']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Outliers extremos (3x IQR)
                outlier_mask = (df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)
                outliers += outlier_mask.sum()
                total_points += len(df)
        
        outlier_pct = outliers / total_points if total_points > 0 else 0
        
        return {
            'passed': outlier_pct <= thresholds['max_outlier_pct'],
            'message': f"Outliers: {outliers} ({outlier_pct:.4%})",
            'metrics': {'outlier_pct': outlier_pct, 'outliers': outliers}
        }
    
    def _check_time_sequence(self, df: pd.DataFrame, thresholds: Dict) -> Dict:
        """Verifica sequência temporal"""
        if 'datetime' not in df.columns:
            return {'passed': True, 'message': "Sem coluna datetime", 'metrics': {}}
        
        # Verificar se está ordenado
        is_sorted = df['datetime'].is_monotonic_increasing
        
        # Verificar gaps
        time_diffs = df['datetime'].diff().dropna()
        max_gap = time_diffs.max()
        
        issues = []
        if not is_sorted:
            issues.append("Dados não ordenados")
        
        if max_gap > pd.Timedelta(minutes=thresholds['max_time_gap_minutes']):
            issues.append(f"Gap máximo: {max_gap}")
        
        return {
            'passed': len(issues) == 0,
            'message': "; ".join(issues) if issues else "OK",
            'metrics': {
                'is_sorted': is_sorted,
                'max_gap_minutes': max_gap.total_seconds() / 60 if pd.notna(max_gap) else 0
            }
        }
    
    def _check_market_hours(self, df: pd.DataFrame, thresholds: Dict) -> Dict:
        """Verifica se dados estão no horário de mercado"""
        if 'datetime' not in df.columns:
            return {'passed': True, 'message': "Sem coluna datetime", 'metrics': {}}
        
        # Horário do mercado brasileiro (9h às 18h)
        market_open = time(9, 0)
        market_close = time(18, 0)
        
        # Verificar dados fora do horário
        df['time'] = pd.to_datetime(df['datetime']).dt.time
        outside_hours = df[(df['time'] < market_open) | (df['time'] > market_close)]
        
        outside_pct = len(outside_hours) / len(df) if len(df) > 0 else 0
        
        return {
            'passed': outside_pct < 0.01,  # Permitir até 1% fora
            'message': f"Dados fora do horário: {len(outside_hours)} ({outside_pct:.2%})",
            'metrics': {'outside_hours_pct': outside_pct}
        }
    
    def _check_trade_volume(self, df: pd.DataFrame, thresholds: Dict) -> Dict:
        """Verifica volume de trades"""
        if 'datetime' not in df.columns:
            return {'passed': True, 'message': "Sem análise de volume", 'metrics': {}}
        
        # Calcular trades por hora
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        trades_per_hour = df.groupby('hour').size().mean()
        
        return {
            'passed': trades_per_hour >= thresholds['min_trades_per_hour'],
            'message': f"Média de {trades_per_hour:.0f} trades/hora",
            'metrics': {'avg_trades_per_hour': trades_per_hour}
        }
    
    def _check_negative_prices(self, df: pd.DataFrame, thresholds: Dict) -> Dict:
        """Verifica preços negativos"""
        price_cols = ['price', 'open', 'high', 'low', 'close']
        negative_count = 0
        
        for col in price_cols:
            if col in df.columns:
                negative_count += (df[col] <= 0).sum()
        
        return {
            'passed': negative_count == 0,
            'message': f"Preços negativos ou zero: {negative_count}",
            'metrics': {'negative_prices': negative_count}
        }
    
    def _check_spread(self, df: pd.DataFrame, thresholds: Dict) -> Dict:
        """Verifica spread bid-ask"""
        if 'bid_price' in df.columns and 'ask_price' in df.columns:
            df['spread_pct'] = (df['ask_price'] - df['bid_price']) / df['bid_price']
            high_spread = (df['spread_pct'] > thresholds['max_spread_pct']).sum()
            avg_spread = df['spread_pct'].mean()
            
            return {
                'passed': high_spread < len(df) * 0.01,  # Menos de 1% com spread alto
                'message': f"Spread médio: {avg_spread:.4%}, Alto: {high_spread}",
                'metrics': {'avg_spread_pct': avg_spread, 'high_spread_count': high_spread}
            }
        
        return {'passed': True, 'message': "Sem dados de spread", 'metrics': {}}
    
    def _check_ohlc_consistency(self, df: pd.DataFrame, thresholds: Dict) -> Dict:
        """Verifica consistência OHLC"""
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # High deve ser >= todos
            high_issues = ((df['high'] < df['open']) | 
                          (df['high'] < df['close']) | 
                          (df['high'] < df['low'])).sum()
            
            # Low deve ser <= todos
            low_issues = ((df['low'] > df['open']) | 
                         (df['low'] > df['close']) | 
                         (df['low'] > df['high'])).sum()
            
            total_issues = high_issues + low_issues
            
            return {
                'passed': total_issues == 0,
                'message': f"Inconsistências OHLC: {total_issues}",
                'metrics': {'ohlc_issues': total_issues}
            }
        
        return {'passed': True, 'message': "Sem dados OHLC", 'metrics': {}}
    
    def _check_price_gaps(self, df: pd.DataFrame, thresholds: Dict) -> Dict:
        """Verifica gaps de preço entre candles"""
        if 'close' in df.columns and 'open' in df.columns:
            df['gap_pct'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            large_gaps = (df['gap_pct'] > 0.02).sum()  # Gaps > 2%
            
            return {
                'passed': large_gaps < len(df) * 0.01,
                'message': f"Gaps grandes: {large_gaps}",
                'metrics': {'large_gaps': large_gaps, 'max_gap_pct': df['gap_pct'].max()}
            }
        
        return {'passed': True, 'message': "Sem análise de gaps", 'metrics': {}}
    
    def _check_book_depth(self, df: pd.DataFrame, thresholds: Dict) -> Dict:
        """Verifica profundidade do book"""
        # Implementar conforme estrutura específica do book
        return {'passed': True, 'message': "OK", 'metrics': {}}
    
    def _check_book_spread(self, df: pd.DataFrame, thresholds: Dict) -> Dict:
        """Verifica spread do book"""
        return self._check_spread(df, thresholds)
    
    # Funções de correção
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrige tipos de dados"""
        # Converter datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Converter numéricos
        numeric_cols = ['price', 'volume', 'open', 'high', 'low', 'close',
                       'bid_price', 'ask_price', 'bid_volume', 'ask_volume']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _fix_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrige dados faltantes"""
        # Forward fill para séries temporais
        if 'datetime' in df.columns:
            df = df.sort_values('datetime')
            df = df.ffill().bfill()
        
        # Preencher zeros onde apropriado
        volume_cols = ['volume', 'bid_volume', 'ask_volume']
        for col in volume_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        return df
    
    def _fix_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicatas"""
        if 'datetime' in df.columns:
            # Manter última ocorrência
            df = df.drop_duplicates(subset=['datetime'], keep='last')
        
        return df
    
    def _fix_time_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrige sequência temporal"""
        if 'datetime' in df.columns:
            df = df.sort_values('datetime')
            df = df.reset_index(drop=True)
        
        return df
    
    def _fix_ohlc_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrige inconsistências OHLC"""
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Garantir high >= todos
            df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
            
            # Garantir low <= todos
            df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        return df
    
    def _analyze_data_patterns(self, df: pd.DataFrame, data_type: str) -> Dict:
        """Análise adicional de padrões nos dados"""
        metrics = {}
        suggestions = []
        
        try:
            if data_type == 'trades' and 'volume' in df.columns:
                # Análise de distribuição de volume
                volume_skew = stats.skew(df['volume'])
                if abs(volume_skew) > 2:
                    suggestions.append(f"Volume com skew alto ({volume_skew:.2f}), "
                                     "considerar transformação log")
                metrics['volume_skew'] = volume_skew
            
            if 'price' in df.columns or 'close' in df.columns:
                price_col = 'close' if 'close' in df.columns else 'price'
                
                # Volatilidade
                returns = df[price_col].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)
                metrics['annualized_volatility'] = volatility
                
                if volatility > 0.5:  # 50% anual
                    suggestions.append(f"Alta volatilidade detectada ({volatility:.2%})")
            
            # Análise de completude temporal
            if 'datetime' in df.columns and len(df) > 1:
                time_range = (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600
                if time_range < 6:  # Menos de 6 horas
                    suggestions.append(f"Período curto de dados ({time_range:.1f} horas)")
                metrics['time_range_hours'] = time_range
            
        except Exception as e:
            self.logger.error(f"Erro na análise de padrões: {e}")
        
        return {'metrics': metrics, 'suggestions': suggestions}


if __name__ == "__main__":
    # Teste do validador
    validator = DataValidator()
    
    # Criar dados de exemplo com alguns problemas
    trades = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01 09:00', periods=100, freq='1min'),
        'price': np.random.normal(5900, 50, 100),
        'volume': np.random.randint(100, 1000, 100),
        'side': np.random.choice(['BUY', 'SELL'], 100)
    })
    
    # Introduzir alguns problemas
    trades.loc[10:15, 'price'] = np.nan  # Dados faltantes
    trades.loc[20, 'price'] = -100  # Preço negativo
    trades.loc[30:32, 'datetime'] = trades.loc[30, 'datetime']  # Duplicatas
    
    # Validar
    result = validator.validate_data(
        data=trades,
        data_type='trades',
        symbol='WDOU25',
        auto_fix=True
    )
    
    print(f"Válido: {result.is_valid}")
    print(f"Score de qualidade: {result.quality_score:.2%}")
    print(f"Erros: {len(result.errors)}")
    print(f"Avisos: {len(result.warnings)}")
    
    if result.errors:
        print("\nErros encontrados:")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print("\nAvisos:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    if result.suggestions:
        print("\nSugestões:")
        for suggestion in result.suggestions:
            print(f"  - {suggestion}")