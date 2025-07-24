# src/backtesting/stress_test_engine.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

class StressTestEngine:
    """Engine para testes de stress e cenários extremos"""
    
    def __init__(self, backtester):
        self.backtester = backtester
        self.logger = logging.getLogger(__name__)
        
        # Cenários pré-definidos
        self.scenarios = {
            'flash_crash': self._create_flash_crash_scenario,
            'high_volatility': self._create_high_volatility_scenario,
            'low_liquidity': self._create_low_liquidity_scenario,
            'trend_reversal': self._create_trend_reversal_scenario,
            'gap_opening': self._create_gap_opening_scenario,
            'black_swan': self._create_black_swan_scenario
        }
    
    def _safe_float_conversion(self, value) -> float:
        """
        Converte qualquer valor para float de forma segura
        
        Args:
            value: Valor a ser convertido (pode ser Scalar, numpy array, etc.)
            
        Returns:
            float: Valor convertido
        """
        try:
            # Se é um array ou serie, pegar o primeiro item
            if hasattr(value, 'item'):
                return float(value.item())
            elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                # Se é iterável mas não string/bytes, pegar primeiro elemento
                return float(next(iter(value)))
            else:
                # Conversão direta
                return float(value)
        except (ValueError, TypeError, StopIteration):
            # Fallback para zero em caso de erro
            self.logger.warning(f"Não foi possível converter {value} para float, usando 0.0")
            return 0.0
    
    def run_stress_tests(self, historical_data: pd.DataFrame,
                        scenarios: Optional[List[str]] = None) -> Dict:
        """
        Executa testes de stress
        
        Args:
            historical_data: Dados históricos base
            scenarios: Lista de cenários para testar
            
        Returns:
            Resultados dos testes de stress
        """
        if scenarios is None:
            scenarios = list(self.scenarios.keys())
        
        results = {}
        
        for scenario_name in scenarios:
            if scenario_name not in self.scenarios:
                self.logger.warning(f"Cenário desconhecido: {scenario_name}")
                continue
            
            self.logger.info(f"Executando teste de stress: {scenario_name}")
            
            # Criar dados de stress
            stress_data = self.scenarios[scenario_name](historical_data)
            
            # Executar backtest com dados de stress
            scenario_results = self.backtester.run_backtest(stress_data)
            
            # Adicionar análise específica do cenário
            scenario_results['scenario_analysis'] = self._analyze_scenario_impact(
                scenario_results, scenario_name
            )
            
            results[scenario_name] = scenario_results
        
        # Análise comparativa
        results['comparative_analysis'] = self._compare_scenarios(results)
        
        return results
    
    def _create_flash_crash_scenario(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cria cenário de flash crash"""
        stress_data = data.copy()
        
        # Selecionar pontos aleatórios para crashes
        n_crashes = max(1, len(data) // 1000)  # 1 crash por 1000 barras
        crash_indices = np.random.choice(
            range(100, len(data) - 100), n_crashes, replace=False
        )
        
        for idx in crash_indices:
            # Crash de 3-5% em 5 minutos
            crash_magnitude = np.random.uniform(0.03, 0.05)
            
            # Aplicar crash
            for i in range(5):
                if idx + i < len(stress_data):
                    stress_data.loc[stress_data.index[idx + i], 'low'] *= (1 - crash_magnitude)
                    stress_data.loc[stress_data.index[idx + i], 'close'] *= (1 - crash_magnitude * 0.8)
                    stress_data.loc[stress_data.index[idx + i], 'volume'] *= 5  # Spike de volume
            
            # Recuperação parcial
            for i in range(5, 20):
                if idx + i < len(stress_data):
                    recovery = crash_magnitude * 0.6 * (i - 5) / 15
                    stress_data.loc[stress_data.index[idx + i], 'close'] *= (1 + recovery)
        
        return stress_data
    
    def _create_high_volatility_scenario(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cria cenário de alta volatilidade"""
        stress_data = data.copy()
        
        # Aumentar volatilidade em 3x
        returns = stress_data['close'].pct_change()
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Aplicar volatilidade aumentada
        stressed_returns = np.random.normal(
            mean_return, std_return * 3, len(stress_data) - 1
        )
        
        # Reconstruir preços
        stress_data['close'].iloc[1:] = stress_data['close'].iloc[0] * (1 + stressed_returns).cumprod()
        
        # Ajustar OHLV
        for i in range(1, len(stress_data)):
            daily_vol = abs(stressed_returns[i-1])
            stress_data.loc[stress_data.index[i], 'high'] = stress_data.loc[stress_data.index[i], 'close'] * (1 + daily_vol/2)
            stress_data.loc[stress_data.index[i], 'low'] = stress_data.loc[stress_data.index[i], 'close'] * (1 - daily_vol/2)
            stress_data.loc[stress_data.index[i], 'volume'] *= (1 + daily_vol * 10)  # Volume proporcional à volatilidade
        
        return stress_data
    
    def _create_low_liquidity_scenario(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cria cenário de baixa liquidez"""
        stress_data = data.copy()
        
        # Reduzir volume em 80%
        stress_data['volume'] *= 0.2
        
        # Aumentar spreads (simulado por maior diferença high-low)
        spread_increase = 0.002  # 0.2%
        stress_data['high'] *= (1 + spread_increase)
        stress_data['low'] *= (1 - spread_increase)
        
        # Adicionar gaps aleatórios
        gap_probability = 0.05
        for i in range(1, len(stress_data)):
            if np.random.random() < gap_probability:
                gap_size = np.random.uniform(-0.005, 0.005)
                # Conversão segura para float
                prev_close = stress_data.loc[stress_data.index[i-1], 'close']
                prev_close_val = self._safe_float_conversion(prev_close)
                stress_data.loc[stress_data.index[i], 'open'] = prev_close_val * (1 + gap_size)
        
        return stress_data
    
    def _create_trend_reversal_scenario(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cria cenário de reversão de tendência abrupta"""
        stress_data = data.copy()
        
        # Identificar tendência atual
        sma_20 = stress_data['close'].rolling(20).mean()
        sma_50 = stress_data['close'].rolling(50).mean()
        
        # Pontos de reversão
        n_reversals = max(1, len(data) // 500)
        reversal_points = np.random.choice(
            range(100, len(data) - 100), n_reversals, replace=False
        )
        
        for point in reversal_points:
            # Determinar direção da reversão
            current_trend = 1 if sma_20.iloc[point] > sma_50.iloc[point] else -1
            reversal_direction = -current_trend
            
            # Aplicar reversão gradual
            reversal_magnitude = np.random.uniform(0.05, 0.10)
            
            for i in range(50):  # Reversão ao longo de 50 períodos
                if point + i < len(stress_data):
                    adjustment = reversal_direction * reversal_magnitude * (i / 50)
                    stress_data.loc[stress_data.index[point + i], 'close'] *= (1 + adjustment)
                    stress_data.loc[stress_data.index[point + i], 'high'] *= (1 + adjustment * 1.2)
                    stress_data.loc[stress_data.index[point + i], 'low'] *= (1 + adjustment * 0.8)
        
        return stress_data
    
    def _create_gap_opening_scenario(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cria cenário com gaps de abertura frequentes"""
        stress_data = data.copy()
        
        # Adicionar gaps em 10% dos dias
        gap_probability = 0.10
        
        for i in range(1, len(stress_data)):
            if np.random.random() < gap_probability:
                # Gap de -2% a +2%
                gap_size = np.random.uniform(-0.02, 0.02)
                
                prev_close = stress_data.loc[stress_data.index[i-1], 'close']
                prev_close_val = self._safe_float_conversion(prev_close)
                gap_open = prev_close_val * (1 + gap_size)
                
                stress_data.loc[stress_data.index[i], 'open'] = gap_open
                
                # Ajustar high/low se necessário
                current_high = self._safe_float_conversion(stress_data.loc[stress_data.index[i], 'high'])
                current_low = self._safe_float_conversion(stress_data.loc[stress_data.index[i], 'low'])
                
                if gap_open > current_high:
                    stress_data.loc[stress_data.index[i], 'high'] = gap_open * 1.001
                if gap_open < current_low:
                    stress_data.loc[stress_data.index[i], 'low'] = gap_open * 0.999
        
        return stress_data
    
    def _create_black_swan_scenario(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cria cenário de evento cisne negro"""
        stress_data = data.copy()
        
        # Evento único catastrófico no meio do período
        event_index = len(data) // 2
        
        # Queda de 15-20% em um dia
        crash_magnitude = np.random.uniform(0.15, 0.20)
        
        # Aplicar crash
        current_open = self._safe_float_conversion(stress_data.loc[stress_data.index[event_index], 'open'])
        current_low = self._safe_float_conversion(stress_data.loc[stress_data.index[event_index], 'low'])
        current_close = self._safe_float_conversion(stress_data.loc[stress_data.index[event_index], 'close'])
        current_volume = self._safe_float_conversion(stress_data.loc[stress_data.index[event_index], 'volume'])
        
        stress_data.loc[stress_data.index[event_index], 'open'] = current_open * (1 - crash_magnitude * 0.5)
        stress_data.loc[stress_data.index[event_index], 'low'] = current_low * (1 - crash_magnitude)
        stress_data.loc[stress_data.index[event_index], 'close'] = current_close * (1 - crash_magnitude * 0.8)
        stress_data.loc[stress_data.index[event_index], 'volume'] = current_volume * 10
        
        # Volatilidade elevada por 30 dias
        for i in range(1, 31):
            if event_index + i < len(stress_data):
                daily_swing = np.random.uniform(-0.05, 0.05)
                
                current_close_i = self._safe_float_conversion(stress_data.loc[stress_data.index[event_index + i], 'close'])
                current_high_i = self._safe_float_conversion(stress_data.loc[stress_data.index[event_index + i], 'high'])
                current_low_i = self._safe_float_conversion(stress_data.loc[stress_data.index[event_index + i], 'low'])
                current_volume_i = self._safe_float_conversion(stress_data.loc[stress_data.index[event_index + i], 'volume'])
                
                stress_data.loc[stress_data.index[event_index + i], 'close'] = current_close_i * (1 + daily_swing)
                stress_data.loc[stress_data.index[event_index + i], 'high'] = current_high_i * (1 + abs(daily_swing))
                stress_data.loc[stress_data.index[event_index + i], 'low'] = current_low_i * (1 - abs(daily_swing))
                stress_data.loc[stress_data.index[event_index + i], 'volume'] = current_volume_i * 3
        
        return stress_data
    
    def _analyze_scenario_impact(self, results: Dict, scenario_name: str) -> Dict:
        """Analisa impacto específico do cenário"""
        baseline_metrics = self.backtester.config.initial_capital
        
        analysis = {
            'scenario': scenario_name,
            'survival_rate': 1.0 if results['final_equity'] > 0 else 0.0,
            'capital_preservation': results['final_equity'] / baseline_metrics,
            'max_drawdown_scenario': results.get('max_drawdown', 0),
            'recovery_ability': self._calculate_recovery_ability(results),
            'strategy_robustness': self._calculate_robustness_score(results)
        }
        
        # Análise específica por cenário
        if scenario_name == 'flash_crash':
            analysis['flash_crash_trades'] = self._count_trades_during_crashes(results)
        elif scenario_name == 'high_volatility':
            analysis['volatility_adaptation'] = self._analyze_volatility_adaptation(results)
        
        return analysis
    
    def _calculate_recovery_ability(self, results: Dict) -> float:
        """Calcula capacidade de recuperação após drawdowns"""
        if 'drawdown_analysis' not in results:
            return 0.0
        
        drawdown_data = results['drawdown_analysis']
        if drawdown_data.get('no_drawdowns', False):
            return 1.0
        
        # Quanto menor o tempo médio de recuperação, melhor
        avg_recovery = drawdown_data.get('avg_recovery_time', float('inf'))
        if avg_recovery == float('inf'):
            return 0.0
        
        # Score baseado em recuperação rápida (normalizado)
        return min(1.0, 30 / (avg_recovery + 1))  # 30 períodos como referência
    
    def _calculate_robustness_score(self, results: Dict) -> float:
        """Calcula score de robustez da estratégia"""
        factors = []
        
        # Fator 1: Preservação de capital
        capital_preservation = results['final_equity'] / self.backtester.config.initial_capital
        factors.append(min(1.0, capital_preservation))
        
        # Fator 2: Consistência (baixo desvio nos retornos)
        if results.get('total_trades', 0) > 0:
            win_rate = results.get('win_rate', 0)
            consistency = 1.0 - abs(win_rate - 0.5) * 2  # Melhor próximo a 50%
            factors.append(consistency)
        
        # Fator 3: Drawdown controlado
        max_dd = abs(results.get('max_drawdown', 0))
        dd_score = max(0, 1.0 - max_dd * 2)  # Penaliza drawdowns > 50%
        factors.append(dd_score)
        
        return float(np.mean(factors)) if factors else 0.0
    
    def _count_trades_during_crashes(self, results: Dict) -> int:
        """Conta trades executados durante crashes"""
        # Implementação simplificada
        # Em produção, analisar timestamps dos trades vs momentos de crash
        return len([t for t in results.get('trades', []) if t.get('mae', 0) < -0.02])
    
    def _analyze_volatility_adaptation(self, results: Dict) -> float:
        """Analisa adaptação a diferentes níveis de volatilidade"""
        # Analisar distribuição de retornos vs volatilidade
        # Implementação simplificada
        return results.get('sharpe_ratio', 0) / 2.0  # Normalizado
    
    def _compare_scenarios(self, all_results: Dict) -> Dict:
        """Compara resultados entre cenários"""
        comparison = {
            'best_scenario': None,
            'worst_scenario': None,
            'most_robust_metrics': {},
            'vulnerability_analysis': {}
        }
        
        # Encontrar melhor e pior cenário
        scenario_scores = {}
        for scenario, results in all_results.items():
            if scenario == 'comparative_analysis':
                continue
            
            score = results.get('final_equity', 0) / self.backtester.config.initial_capital
            scenario_scores[scenario] = score
        
        if scenario_scores:
            comparison['best_scenario'] = max(scenario_scores.keys(), key=lambda k: scenario_scores[k])
            comparison['worst_scenario'] = min(scenario_scores.keys(), key=lambda k: scenario_scores[k])
        
        # Identificar métricas mais robustas
        metric_stability = {}
        metrics_to_analyze = ['win_rate', 'sharpe_ratio', 'max_drawdown']
        
        for metric in metrics_to_analyze:
            values = [results.get(metric, 0) for results in all_results.values() 
                     if 'comparative_analysis' not in results]
            if values:
                metric_stability[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else float('inf')
                }
        
        comparison['most_robust_metrics'] = metric_stability
        
        # Análise de vulnerabilidades
        vulnerabilities = []
        for scenario, results in all_results.items():
            if scenario == 'comparative_analysis':
                continue
            
            if results.get('final_equity', self.backtester.config.initial_capital) < self.backtester.config.initial_capital * 0.8:
                vulnerabilities.append({
                    'scenario': scenario,
                    'capital_loss': 1 - results['final_equity'] / self.backtester.config.initial_capital,
                    'max_drawdown': results.get('max_drawdown', 0)
                })
        
        comparison['vulnerability_analysis'] = vulnerabilities
        
        return comparison