"""
Enhanced ML Backtester com Debug Detalhado
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
from typing import Dict, List, Optional

from trading_system import TradingSystem
from features.feature_debugger import FeatureDebugger
from adaptive_threshold_manager import AdaptiveThresholdManager


class EnhancedMLBacktester:
    """Backtester com análise detalhada e otimização"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.logger = logging.getLogger(__name__)
        
        # Componentes de análise
        self.feature_debugger = FeatureDebugger(self.logger)
        self.threshold_manager = AdaptiveThresholdManager()
        
        # Estatísticas detalhadas
        self.stats = {
            'signals_analyzed': 0,
            'signals_rejected': 0,
            'rejection_reasons': {},
            'regime_distribution': {},
            'feature_quality_scores': [],
            'threshold_violations': {
                'confidence': 0,
                'direction': 0,
                'magnitude': 0,
                'regime_confidence': 0
            }
        }
        
    def run_backtest_with_analysis(self, system: TradingSystem, 
                                  start_date: datetime, 
                                  end_date: datetime,
                                  debug_mode: bool = True) -> Dict:
        """Executa backtest com análise detalhada"""
        
        self.logger.info("="*80)
        self.logger.info("INICIANDO BACKTEST ENHANCED COM ANÁLISE")
        self.logger.info("="*80)
        
        # Preparar sistema
        data = system.data_structure
        trades = []
        equity_curve = [self.initial_capital]
        
        # Log configuração inicial
        self.logger.info(f"Capital inicial: R$ {self.initial_capital:,.2f}")
        self.logger.info(f"Período: {start_date} até {end_date}")
        
        # Simular período
        current_date = start_date
        position = None
        
        while current_date <= end_date:
            # Simular novo candle (em produção virá do callback)
            # Aqui você precisa carregar dados reais
            
            # Processar predição
            ml_result = system.ml_coordinator.process_prediction_request(data)
            
            if ml_result:
                self.stats['signals_analyzed'] += 1
                
                # Debug detalhado
                if debug_mode:
                    self._debug_ml_result(ml_result, data)
                
                # Verificar se pode operar
                if ml_result.get('can_trade', False):
                    # Gerar sinal
                    signal = system.signal_generator.generate_signal(ml_result, data)
                    
                    if signal and signal.get('action') != 'hold':
                        # Executar trade
                        trade = self._execute_trade(signal, position)
                        if trade:
                            trades.append(trade)
                            
                            # Atualizar threshold manager
                            trade_result = {
                                'regime': ml_result.get('regime'),
                                'win': trade['profit'] > 0,
                                'return': trade['profit'] / self.initial_capital,
                                'confidence': ml_result.get('confidence'),
                                'direction': ml_result.get('direction'),
                                'magnitude': ml_result.get('magnitude')
                            }
                            self.threshold_manager.update_trade_result(trade_result)
                else:
                    # Registrar rejeição
                    reason = ml_result.get('reason', 'unknown')
                    self.stats['signals_rejected'] += 1
                    self.stats['rejection_reasons'][reason] = \
                        self.stats['rejection_reasons'].get(reason, 0) + 1
                    
                    # Analisar threshold violations
                    self._analyze_threshold_violations(ml_result)
                
                # Registrar regime
                regime = ml_result.get('regime', 'unknown')
                self.stats['regime_distribution'][regime] = \
                    self.stats['regime_distribution'].get(regime, 0) + 1
                
                # Analisar qualidade das features periodicamente
                if self.stats['signals_analyzed'] % 50 == 0:
                    self._analyze_feature_quality(data)
            
            # Atualizar equity
            current_equity = self._calculate_equity(trades)
            equity_curve.append(current_equity)
            
            # Adaptar thresholds periodicamente
            if len(trades) > 0 and len(trades) % 20 == 0:
                self.threshold_manager.adapt_thresholds()
            
            # Avançar tempo (simplificado)
            current_date = current_date + pd.Timedelta(minutes=5)
        
        # Gerar relatório final
        results = self._generate_detailed_report(trades, equity_curve)
        
        # Sugestões de otimização
        results['optimization_suggestions'] = self._generate_optimization_suggestions()
        
        return results
    
    def _debug_ml_result(self, ml_result: Dict, data):
        """Debug detalhado do resultado ML"""
        self.logger.debug("="*50)
        self.logger.debug("[DEBUG] ML Result Analysis:")
        self.logger.debug(f"  Regime: {ml_result.get('regime')}")
        self.logger.debug(f"  Direction: {ml_result.get('direction', 0):.4f}")
        self.logger.debug(f"  Magnitude: {ml_result.get('magnitude', 0):.6f}")
        self.logger.debug(f"  Confidence: {ml_result.get('confidence', 0):.4f}")
        self.logger.debug(f"  Can Trade: {ml_result.get('can_trade', False)}")
        self.logger.debug(f"  Decision: {ml_result.get('trade_decision', 'HOLD')}")
        
        if not ml_result.get('can_trade', False):
            self.logger.debug(f"  Rejection Reason: {ml_result.get('reason', 'unknown')}")
    
    def _analyze_threshold_violations(self, ml_result: Dict):
        """Analisa quais thresholds estão bloqueando trades"""
        regime = ml_result.get('regime', 'unknown')
        thresholds = self.threshold_manager.get_thresholds_for_regime(regime)
        
        # Verificar cada threshold
        if ml_result.get('confidence', 0) < thresholds.get('confidence', 0.5):
            self.stats['threshold_violations']['confidence'] += 1
            
        if abs(ml_result.get('direction', 0)) < thresholds.get('direction', 0.3):
            self.stats['threshold_violations']['direction'] += 1
            
        if ml_result.get('magnitude', 0) < thresholds.get('magnitude', 0.001):
            self.stats['threshold_violations']['magnitude'] += 1
    
    def _analyze_feature_quality(self, data):
        """Analisa qualidade das features"""
        if not data.features.empty:
            analysis = self.feature_debugger.analyze_features(data.features)
            quality_score = analysis.get('quality_score', 0)
            self.stats['feature_quality_scores'].append(quality_score)
            
            if quality_score < 0.5:
                self.logger.warning(f"[FEATURE QUALITY] Score baixo: {quality_score:.3f}")
    
    def _generate_optimization_suggestions(self) -> List[str]:
        """Gera sugestões específicas de otimização"""
        suggestions = []
        
        # 1. Análise de rejeições
        total_signals = self.stats['signals_analyzed']
        if total_signals > 0:
            rejection_rate = self.stats['signals_rejected'] / total_signals
            
            if rejection_rate > 0.9:
                suggestions.append(f"Taxa de rejeição muito alta ({rejection_rate:.1%}) - revisar thresholds")
                
                # Top razões de rejeição
                top_reasons = sorted(self.stats['rejection_reasons'].items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
                for reason, count in top_reasons:
                    suggestions.append(f"  - {reason}: {count} vezes ({count/total_signals:.1%})")
        
        # 2. Análise de threshold violations
        total_violations = sum(self.stats['threshold_violations'].values())
        if total_violations > 0:
            suggestions.append("\nThresholds mais restritivos:")
            for threshold, count in self.stats['threshold_violations'].items():
                if count > 0:
                    pct = count / total_violations * 100
                    suggestions.append(f"  - {threshold}: {count} violações ({pct:.1f}%)")
        
        # 3. Qualidade das features
        if self.stats['feature_quality_scores']:
            avg_quality = np.mean(self.stats['feature_quality_scores'])
            if avg_quality < 0.6:
                suggestions.append(f"\nQualidade média das features baixa: {avg_quality:.3f}")
                suggestions.append("  - Revisar cálculo de features e tratamento de NaN")
        
        # 4. Distribuição de regimes
        if self.stats['regime_distribution']:
            suggestions.append("\nDistribuição de regimes:")
            for regime, count in self.stats['regime_distribution'].items():
                pct = count / total_signals * 100 if total_signals > 0 else 0
                suggestions.append(f"  - {regime}: {pct:.1f}%")
        
        # 5. Sugestões do threshold manager
        threshold_suggestions = self.threshold_manager.suggest_threshold_adjustments()
        if threshold_suggestions:
            suggestions.append("\nAjustes de threshold sugeridos:")
            suggestions.extend([f"  - {s}" for s in threshold_suggestions])
        
        # 6. Sugestões específicas baseadas em padrões
        if rejection_rate > 0.95 and 'low_confidence' in self.stats['rejection_reasons']:
            suggestions.append("\nAção recomendada: Retreinar modelos com dados mais recentes")
            
        return suggestions
    
    def _generate_detailed_report(self, trades: List[Dict], 
                                equity_curve: List[float]) -> Dict:
        """Gera relatório detalhado do backtest"""
        
        final_capital = equity_curve[-1] if equity_curve else self.initial_capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # Métricas básicas
        win_trades = [t for t in trades if t['profit'] > 0]
        loss_trades = [t for t in trades if t['profit'] <= 0]
        
        report = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'total_trades': len(trades),
                'win_rate': len(win_trades) / len(trades) if trades else 0
            },
            'signal_analysis': {
                'total_signals': self.stats['signals_analyzed'],
                'rejected_signals': self.stats['signals_rejected'],
                'rejection_rate': self.stats['signals_rejected'] / self.stats['signals_analyzed'] 
                                if self.stats['signals_analyzed'] > 0 else 0,
                'rejection_reasons': self.stats['rejection_reasons']
            },
            'threshold_analysis': {
                'violations': self.stats['threshold_violations'],
                'current_thresholds': self.threshold_manager.thresholds
            },
            'regime_analysis': self.stats['regime_distribution'],
            'feature_quality': {
                'scores': self.stats['feature_quality_scores'],
                'avg_score': np.mean(self.stats['feature_quality_scores']) 
                           if self.stats['feature_quality_scores'] else 0
            }
        }
        
        # Estatísticas avançadas se houver trades
        if trades:
            profits = [t['profit'] for t in trades]
            report['advanced_metrics'] = {
                'sharpe_ratio': self._calculate_sharpe_ratio(profits),
                'max_drawdown': self._calculate_max_drawdown(equity_curve),
                'profit_factor': sum(t['profit'] for t in win_trades) / 
                               abs(sum(t['profit'] for t in loss_trades)) 
                               if loss_trades else float('inf')
            }
        
        return report
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calcula Sharpe Ratio"""
        if len(returns) < 2:
            return 0
        returns_array = np.array(returns)
        return np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calcula drawdown máximo"""
        if len(equity_curve) < 2:
            return 0
        peak = equity_curve[0]
        max_dd = 0
        for value in equity_curve[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        return max_dd
    
    def _calculate_equity(self, trades: List[Dict]) -> float:
        """Calcula equity atual"""
        return self.initial_capital + sum(t.get('profit', 0) for t in trades)
    
    def _execute_trade(self, signal: Dict, current_position) -> Optional[Dict]:
        """Simula execução de trade"""
        # Simulação simplificada
        trade = {
            'timestamp': datetime.now(),
            'action': signal['action'],
            'entry_price': signal['entry_price'],
            'size': signal.get('position_size', 1),
            'profit': 0  # Será calculado quando fechar
        }
        
        # Simular resultado (simplificado)
        # Em produção, isso seria baseado em dados reais
        import random
        if random.random() > 0.5:
            trade['profit'] = self.initial_capital * 0.001  # 0.1% profit
        else:
            trade['profit'] = -self.initial_capital * 0.0005  # 0.05% loss
            
        return trade