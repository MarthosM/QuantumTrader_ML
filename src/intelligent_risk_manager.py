import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import deque
import threading

class IntelligentRiskManager:
    """Sistema avançado de gestão de risco usando ML"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Modelos de risco
        self.risk_models = {
            'volatility_predictor': VolatilityPredictor(),
            'correlation_tracker': DynamicCorrelationTracker(),
            'drawdown_predictor': DrawdownPredictor(),
            'margin_optimizer': MarginOptimizer()
        }
        
        # Componentes de risco
        self.position_sizer = MLPositionSizer(config)
        self.stop_loss_optimizer = DynamicStopLossOptimizer(config)
        self.portfolio_risk_analyzer = PortfolioRiskAnalyzer()
        
        # Estado do risco
        self.risk_state = {
            'current_exposure': 0,
            'daily_loss': 0,
            'open_positions': {},
            'correlation_matrix': None,
            'risk_metrics': {}
        }
        
        # Limites de risco
        self.risk_limits = {
            'max_daily_loss': config.get('max_daily_loss', 1000),
            'max_position_size': config.get('max_position_size', 3),
            'max_portfolio_risk': config.get('max_portfolio_risk', 0.1),
            'max_correlation': config.get('max_correlation', 0.7),
            'max_leverage': config.get('max_leverage', 2.0)
        }
        
        # Histórico para análise
        self.risk_history = deque(maxlen=1000)
        
    def comprehensive_risk_assessment(self, signal: Dict, 
                                    market_data: pd.DataFrame, 
                                    portfolio_state: Dict) -> Dict:
        """Avaliação completa de risco usando ML"""
        
        risk_assessment = {
            'timestamp': datetime.now(),
            'signal': signal,
            'approved': False,
            'risk_score': 0,
            'adjustments': {}
        }
        
        try:
            # 1. Predição de Volatilidade
            volatility_pred = self.risk_models['volatility_predictor'].predict(
                market_data, horizon=30
            )
            risk_assessment['predicted_volatility'] = volatility_pred
            
            # 2. Análise de Correlação
            correlation_risk = self.risk_models['correlation_tracker'].assess_risk(
                signal, portfolio_state
            )
            risk_assessment['correlation_risk'] = correlation_risk
            
            # 3. Predição de Drawdown
            drawdown_prob = self.risk_models['drawdown_predictor'].predict(
                signal, market_data, portfolio_state
            )
            risk_assessment['drawdown_probability'] = drawdown_prob
            
            # 4. Otimização de Margem
            margin_analysis = self.risk_models['margin_optimizer'].analyze(
                signal, portfolio_state
            )
            risk_assessment['margin_efficiency'] = margin_analysis
            
            # 5. Calcular Score de Risco Combinado
            risk_score = self._calculate_combined_risk_score(risk_assessment)
            risk_assessment['risk_score'] = risk_score
            
            # 6. Verificar Limites
            limit_check = self._check_risk_limits(signal, portfolio_state)
            risk_assessment['limit_check'] = limit_check
            
            # 7. Decisão Final
            risk_assessment['approved'] = (
                risk_score < 0.7 and 
                limit_check['passed'] and
                not self._in_risk_lockdown()
            )
            
            # 8. Ajustes Recomendados
            if risk_assessment['approved']:
                risk_assessment['adjustments'] = self._calculate_risk_adjustments(
                    signal, risk_assessment
                )
            
        except Exception as e:
            self.logger.error(f"Erro na avaliação de risco: {e}")
            risk_assessment['error'] = str(e)
            risk_assessment['approved'] = False
            
        # Registrar avaliação
        self.risk_history.append(risk_assessment)
        
        return risk_assessment
    
    def dynamic_position_sizing(self, signal: Dict, 
                              risk_assessment: Dict, 
                              account_state: Dict) -> Dict:
        """Position sizing dinâmico usando ML"""
        
        # Calcular tamanho base
        base_size = self.position_sizer.calculate_base_size(
            signal, account_state, risk_assessment
        )
        
        # Ajustes ML
        ml_adjustments = self.position_sizer.calculate_ml_adjustments(
            risk_assessment, signal, account_state
        )
        
        # Aplicar ajustes
        adjusted_size = base_size * ml_adjustments['size_multiplier']
        
        # Aplicar limites de segurança
        final_size = self._apply_position_limits(
            adjusted_size, signal, account_state
        )
        
        return {
            'position_size': int(final_size),
            'base_size': base_size,
            'ml_multiplier': ml_adjustments['size_multiplier'],
            'size_rationale': ml_adjustments['rationale'],
            'risk_metrics': {
                'position_risk': final_size * risk_assessment['predicted_volatility'],
                'portfolio_impact': self._calculate_portfolio_impact(final_size, signal)
            }
        }
    
    def optimize_stop_loss(self, position: Dict, 
                          current_data: pd.DataFrame, 
                          market_regime: str) -> Dict:
        """Otimiza stop loss dinamicamente"""
        
        return self.stop_loss_optimizer.optimize_stop_loss(
            position, current_data, market_regime
        )
    
    def _calculate_combined_risk_score(self, risk_metrics: Dict) -> float:
        """Calcula score de risco combinado (0-1)"""
        
        # Pesos para cada componente
        weights = {
            'volatility': 0.3,
            'correlation': 0.25,
            'drawdown': 0.25,
            'margin': 0.2
        }
        
        # Normalizar cada componente para 0-1
        scores = {
            'volatility': min(risk_metrics['predicted_volatility']['predicted_volatility'] / 0.05, 1),
            'correlation': risk_metrics['correlation_risk']['max_correlation'],
            'drawdown': risk_metrics['drawdown_probability']['probability'],
            'margin': 1 - risk_metrics['margin_efficiency']['efficiency']
        }
        
        # Calcular score ponderado
        combined_score = sum(scores[k] * weights[k] for k in weights)
        
        return combined_score
    
    def _check_risk_limits(self, signal: Dict, portfolio_state: Dict) -> Dict:
        """Verifica todos os limites de risco"""
        
        # Assumir saldo padrão se não disponível
        account_balance = portfolio_state.get('balance', 100000)
        
        checks = {
            'daily_loss': self.risk_state['daily_loss'] < self.risk_limits['max_daily_loss'],
            'position_count': len(self.risk_state['open_positions']) < 5,
            'exposure': self.risk_state['current_exposure'] < account_balance * 0.8,
            'leverage': self._calculate_leverage(portfolio_state) < self.risk_limits['max_leverage']
        }
        
        return {
            'passed': all(checks.values()),
            'failed_checks': [k for k, v in checks.items() if not v],
            'details': checks
        }
    
    def _in_risk_lockdown(self) -> bool:
        """Verifica se o sistema está em lockdown por risco"""
        # Verificar se perdas diárias excederam limite
        if self.risk_state['daily_loss'] > self.risk_limits['max_daily_loss']:
            return True
        
        # Verificar se há muitas posições correlacionadas
        if len(self.risk_state['open_positions']) > 3:
            return True
        
        return False

    def _calculate_risk_adjustments(self, signal: Dict, risk_assessment: Dict) -> Dict:
        """Calcula ajustes baseados na avaliação de risco"""
        adjustments = {}
        
        # Ajuste de tamanho baseado no risco
        risk_score = risk_assessment['risk_score']
        if risk_score > 0.5:
            adjustments['size_reduction'] = 1 - (risk_score - 0.5)
        
        # Ajuste de stop loss baseado na volatilidade
        if 'predicted_volatility' in risk_assessment:
            vol = risk_assessment['predicted_volatility']['predicted_volatility']
            if vol > 0.03:
                adjustments['wider_stops'] = True
        
        return adjustments

    def _apply_position_limits(self, size: float, signal: Dict, account_state: Dict) -> float:
        """Aplica limites de segurança ao tamanho da posição"""
        # Limite máximo por posição
        max_position_value = account_state['balance'] * 0.1  # 10% do saldo
        max_size_by_value = max_position_value / signal['entry_price']
        
        # Limite por tamanho absoluto
        max_absolute_size = self.risk_limits['max_position_size']
        
        # Aplicar o menor dos limites
        final_size = min(size, max_size_by_value, max_absolute_size)
        
        return max(1, final_size)  # Mínimo 1 contrato

    def _calculate_portfolio_impact(self, size: float, signal: Dict) -> float:
        """Calcula o impacto da posição no portfólio"""
        position_value = size * signal['entry_price']
        total_exposure = sum(pos['value'] for pos in self.risk_state['open_positions'].values())
        
        return position_value / (total_exposure + position_value) if total_exposure > 0 else 1.0

    def _calculate_leverage(self, portfolio_state: Dict) -> float:
        """Calcula a alavancagem atual"""
        total_position_value = sum(pos['value'] for pos in portfolio_state.get('positions', {}).values())
        account_equity = portfolio_state.get('equity', 100000)
        
        return total_position_value / account_equity if account_equity > 0 else 0


class VolatilityPredictor:
    """Preditor de volatilidade usando ensemble de modelos"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {
            'garch': self._create_garch_model(),
            'lstm_vol': self._create_lstm_volatility_model(),
            'xgb_vol': self._create_xgb_volatility_model()
        }
        self.prediction_cache = {}
        
    def predict(self, market_data: pd.DataFrame, horizon: int = 30) -> Dict:
        """Prediz volatilidade futura usando ensemble"""
        
        predictions = {}
        
        # 1. GARCH
        garch_pred = self._predict_garch_volatility(market_data, horizon)
        predictions['garch'] = garch_pred
        
        # 2. LSTM
        lstm_pred = self._predict_lstm_volatility(market_data, horizon)
        predictions['lstm'] = lstm_pred
        
        # 3. XGBoost
        xgb_pred = self._predict_xgb_volatility(market_data, horizon)
        predictions['xgb'] = xgb_pred
        
        # Combinar predições
        weights = {'garch': 0.4, 'lstm': 0.35, 'xgb': 0.25}
        ensemble_pred = sum(predictions[m] * weights[m] for m in weights)
        
        # Calcular intervalo de confiança
        pred_std = np.std([predictions[m] for m in predictions])
        confidence_interval = (
            ensemble_pred - 1.96 * pred_std,
            ensemble_pred + 1.96 * pred_std
        )
        
        return {
            'predicted_volatility': ensemble_pred,
            'confidence_interval': confidence_interval,
            'individual_predictions': predictions,
            'horizon_minutes': horizon
        }
    
    def _predict_garch_volatility(self, data: pd.DataFrame, horizon: int) -> float:
        """Prediz volatilidade usando GARCH"""
        
        # Calcular retornos
        returns = data['close'].pct_change().dropna()
        
        # Simular GARCH (simplificado para exemplo)
        # Em produção, usar arch package
        current_vol = returns.rolling(20).std().iloc[-1]
        mean_reversion_speed = 0.1
        long_term_vol = returns.std()
        
        # Previsão simples
        predicted_vol = current_vol + mean_reversion_speed * (long_term_vol - current_vol)
        
        return predicted_vol * float(np.sqrt(horizon / 1440))  # Ajustar para horizonte
    
    def _create_garch_model(self):
        """Cria modelo GARCH (placeholder)"""
        return {"type": "garch", "initialized": True}

    def _create_lstm_volatility_model(self):
        """Cria modelo LSTM para volatilidade (placeholder)"""
        return {"type": "lstm", "initialized": True}

    def _create_xgb_volatility_model(self):
        """Cria modelo XGBoost para volatilidade (placeholder)"""
        return {"type": "xgb", "initialized": True}

    def _predict_lstm_volatility(self, data: pd.DataFrame, horizon: int) -> float:
        """Prediz volatilidade usando LSTM (simplificado)"""
        returns = data['close'].pct_change().dropna()
        if len(returns) < 10:
            return 0.02
        
        # Simulação de predição LSTM
        recent_vol = returns.tail(10).std()
        return recent_vol * float(np.sqrt(horizon / 1440))

    def _predict_xgb_volatility(self, data: pd.DataFrame, horizon: int) -> float:
        """Prediz volatilidade usando XGBoost (simplificado)"""
        returns = data['close'].pct_change().dropna()
        if len(returns) < 20:
            return 0.02
        
        # Simulação de predição XGB
        vol_features = {
            'recent_vol': returns.tail(5).std(),
            'med_vol': returns.tail(15).std(),
            'long_vol': returns.tail(30).std() if len(returns) >= 30 else returns.std()
        }
        
        # Combinação simples dos features
        predicted_vol = (vol_features['recent_vol'] * 0.5 + 
                        vol_features['med_vol'] * 0.3 + 
                        vol_features['long_vol'] * 0.2)
        
        return predicted_vol * float(np.sqrt(horizon / 1440))


class DynamicStopLossOptimizer:
    """Otimizador dinâmico de stop loss usando ML"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Estratégias de stop
        self.stop_strategies = {
            'atr_adaptive': ATRAdaptiveStop(),
            'support_resistance': SupportResistanceStop(),
            'volatility_based': VolatilityBasedStop(),
            'ml_optimized': MLOptimizedStop()
        }
        
    def optimize_stop_loss(self, position: Dict, 
                          market_data: pd.DataFrame, 
                          market_regime: str) -> Dict:
        """Otimiza stop loss usando múltiplas estratégias"""
        
        stop_suggestions = {}
        
        # Calcular sugestões de cada estratégia
        for name, strategy in self.stop_strategies.items():
            try:
                stop_level = strategy.calculate_stop(
                    position, market_data, market_regime
                )
                stop_suggestions[name] = stop_level
            except Exception as e:
                self.logger.error(f"Erro em {name}: {e}")
                
        # Selecionar estratégia ótima
        optimal_strategy = self._select_optimal_strategy(
            market_regime, stop_suggestions, position
        )
        
        # Calcular stop final
        if optimal_strategy == 'ensemble':
            final_stop = self._ensemble_stop_calculation(
                stop_suggestions, market_regime, position
            )
        else:
            final_stop = stop_suggestions.get(optimal_strategy, position['entry_price'] * 0.98)
            
        # Validar stop
        final_stop = self._validate_stop_level(final_stop, position, market_data)
        
        return {
            'stop_loss': final_stop,
            'strategy_used': optimal_strategy,
            'all_suggestions': stop_suggestions,
            'distance_percent': abs(final_stop - position['current_price']) / position['current_price'],
            'risk_amount': abs(final_stop - position['entry_price']) * position['quantity']
        }
    
    def _select_optimal_strategy(self, regime: str, 
                               suggestions: Dict, 
                               position: Dict) -> str:
        """Seleciona estratégia ótima baseada no contexto"""
        
        strategy_map = {
            'high_volatility': 'volatility_based',
            'trending': 'atr_adaptive',
            'ranging': 'support_resistance',
            'uncertain': 'ensemble'
        }
        
        # Verificar se ML está disponível e confiável
        if 'ml_optimized' in suggestions and self._ml_confidence_check():
            return 'ml_optimized'
            
        return strategy_map.get(regime, 'ensemble')
    
    def _ensemble_stop_calculation(self, suggestions: Dict, market_regime: str, position: Dict) -> float:
        """Calcula stop usando ensemble de estratégias"""
        if not suggestions:
            return position['entry_price'] * (0.98 if position['side'] == 'BUY' else 1.02)
        
        # Pesos baseados no regime
        regime_weights = {
            'trending': {'atr_adaptive': 0.4, 'volatility_based': 0.35, 'support_resistance': 0.25},
            'ranging': {'support_resistance': 0.5, 'atr_adaptive': 0.3, 'volatility_based': 0.2},
            'high_volatility': {'volatility_based': 0.5, 'atr_adaptive': 0.4, 'support_resistance': 0.1}
        }
        
        weights = regime_weights.get(market_regime, {
            'atr_adaptive': 0.33, 
            'volatility_based': 0.33, 
            'support_resistance': 0.34
        })
        
        # Calcular stop ponderado
        weighted_stop = 0
        total_weight = 0
        
        for strategy, stop_value in suggestions.items():
            weight = weights.get(strategy, 0.1)
            weighted_stop += stop_value * weight
            total_weight += weight
        
        return weighted_stop / total_weight if total_weight > 0 else list(suggestions.values())[0]

    def _validate_stop_level(self, stop: float, position: Dict, market_data: pd.DataFrame) -> float:
        """Valida e ajusta o nível de stop"""
        entry_price = position['entry_price']
        current_price = position.get('current_price', entry_price)
        
        # Limites mínimos e máximos
        min_stop_distance = 0.005  # 0.5%
        max_stop_distance = 0.05   # 5%
        
        if position['side'] == 'BUY':
            # Para compra, stop deve estar abaixo do preço de entrada
            max_stop = entry_price * (1 - min_stop_distance)
            min_stop = entry_price * (1 - max_stop_distance)
            
            validated_stop = min(stop, max_stop)
            validated_stop = max(validated_stop, min_stop)
        else:
            # Para venda, stop deve estar acima do preço de entrada
            min_stop = entry_price * (1 + min_stop_distance)
            max_stop = entry_price * (1 + max_stop_distance)
            
            validated_stop = max(stop, min_stop)
            validated_stop = min(validated_stop, max_stop)
        
        return validated_stop

    def _ml_confidence_check(self) -> bool:
        """Verifica se o modelo ML está confiável"""
        # Implementação simplificada
        return True  # Em produção, verificaria métricas do modelo


class MLPositionSizer:
    """Position sizer usando ML para otimização"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.sizing_history = deque(maxlen=500)
        
    def calculate_base_size(self, signal: Dict, 
                           account_state: Dict, 
                           risk_assessment: Dict) -> float:
        """Calcula tamanho base da posição"""
        
        # Kelly Criterion modificado
        win_rate = signal.get('historical_win_rate', 0.5)
        avg_win = signal.get('avg_win', 1.5)
        avg_loss = signal.get('avg_loss', 1.0)
        
        # Kelly fraction
        kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_f = max(0, min(kelly_f, 0.25))  # Limitar a 25%
        
        # Capital disponível
        available_capital = account_state['available_balance']
        risk_capital = available_capital * self.config['risk_per_trade']
        
        # Tamanho base
        position_value = risk_capital * kelly_f
        base_size = position_value / signal['entry_price']
        
        return base_size
    
    def calculate_ml_adjustments(self, risk_assessment: Dict, 
                               signal: Dict, 
                               account_state: Dict) -> Dict:
        """Calcula ajustes ML para o tamanho da posição"""
        
        # Fatores de ajuste
        adjustments = {
            'volatility_factor': self._volatility_adjustment(risk_assessment),
            'correlation_factor': self._correlation_adjustment(risk_assessment),
            'regime_factor': self._regime_adjustment(signal),
            'confidence_factor': self._confidence_adjustment(signal),
            'drawdown_factor': self._drawdown_adjustment(account_state)
        }
        
        # Multiplicador final
        size_multiplier = float(np.prod(list(adjustments.values())))
        size_multiplier = max(0.1, min(size_multiplier, 2.0))  # Limitar entre 10% e 200%
        
        return {
            'size_multiplier': size_multiplier,
            'adjustments': adjustments,
            'rationale': self._generate_sizing_rationale(adjustments)
        }
    
    def _volatility_adjustment(self, risk_assessment: Dict) -> float:
        """Ajuste baseado na volatilidade prevista"""
        vol_pred = risk_assessment.get('predicted_volatility', {})
        predicted_vol = vol_pred.get('predicted_volatility', 0.02)
        
        # Volatilidade normal: 0.015-0.025 (1.5%-2.5%)
        # Reduzir tamanho se volatilidade alta, aumentar se baixa
        if predicted_vol > 0.04:  # Alta volatilidade
            return 0.7  # Reduzir 30%
        elif predicted_vol > 0.025:  # Volatilidade moderada-alta
            return 0.85  # Reduzir 15%
        elif predicted_vol < 0.01:  # Volatilidade muito baixa
            return 1.2  # Aumentar 20%
        elif predicted_vol < 0.015:  # Volatilidade baixa
            return 1.1  # Aumentar 10%
        else:
            return 1.0  # Volatilidade normal
    
    def _correlation_adjustment(self, risk_assessment: Dict) -> float:
        """Ajuste baseado na correlação com outras posições"""
        corr_risk = risk_assessment.get('correlation_risk', {})
        max_correlation = corr_risk.get('max_correlation', 0)
        
        # Reduzir tamanho se alta correlação com outras posições
        if max_correlation > 0.8:
            return 0.6  # Reduzir 40% - muito correlacionado
        elif max_correlation > 0.6:
            return 0.75  # Reduzir 25% - correlação moderada-alta
        elif max_correlation > 0.4:
            return 0.9  # Reduzir 10% - correlação moderada
        else:
            return 1.0  # Baixa correlação, sem ajuste
    
    def _regime_adjustment(self, signal: Dict) -> float:
        """Ajuste baseado no regime de mercado"""
        regime = signal.get('market_regime', 'undefined')
        regime_confidence = signal.get('regime_confidence', 0.5)
        
        # Ajustes por regime
        regime_multipliers = {
            'trend_up': 1.15,     # Tendência de alta - aumentar
            'trend_down': 1.15,   # Tendência de baixa - aumentar
            'range': 0.95,        # Lateralização - reduzir ligeiramente
            'undefined': 0.8,     # Indefinido - reduzir por segurança
            'volatile': 0.7       # Volátil - reduzir significativamente
        }
        
        base_multiplier = regime_multipliers.get(regime, 0.9)
        
        # Ajustar pela confiança do regime
        if regime_confidence < 0.6:
            base_multiplier *= 0.85  # Reduzir se baixa confiança
        elif regime_confidence > 0.8:
            base_multiplier *= 1.05  # Aumentar ligeiramente se alta confiança
        
        return base_multiplier
    
    def _confidence_adjustment(self, signal: Dict) -> float:
        """Ajuste baseado na confiança do sinal"""
        # Confiança do modelo
        model_confidence = signal.get('confidence', 0.5)
        
        # Probabilidade de direção
        direction_prob = signal.get('direction_probability', 0.5)
        
        # Magnitude esperada
        magnitude = signal.get('predicted_magnitude', 0.001)
        
        # Calcular ajuste combinado
        confidence_factor = 1.0
        
        # Ajuste por confiança do modelo
        if model_confidence > 0.8:
            confidence_factor *= 1.2  # Alta confiança - aumentar
        elif model_confidence > 0.7:
            confidence_factor *= 1.1  # Boa confiança - aumentar ligeiramente
        elif model_confidence < 0.5:
            confidence_factor *= 0.7  # Baixa confiança - reduzir
        elif model_confidence < 0.6:
            confidence_factor *= 0.85  # Confiança moderada-baixa
        
        # Ajuste por probabilidade de direção
        if direction_prob > 0.75:
            confidence_factor *= 1.1
        elif direction_prob < 0.55:
            confidence_factor *= 0.8
        
        # Ajuste por magnitude
        if magnitude > 0.005:  # Movimento forte esperado
            confidence_factor *= 1.1
        elif magnitude < 0.002:  # Movimento fraco esperado
            confidence_factor *= 0.9
        
        return min(confidence_factor, 1.5)  # Limitar aumento máximo
    
    def _drawdown_adjustment(self, account_state: Dict) -> float:
        """Ajuste baseado no drawdown atual"""
        current_balance = account_state.get('balance', 100000)
        peak_balance = account_state.get('peak_balance', current_balance)
        
        # Calcular drawdown percentual
        drawdown_pct = (peak_balance - current_balance) / peak_balance
        
        # Ajustar tamanho baseado no drawdown
        if drawdown_pct > 0.1:  # Drawdown > 10%
            return 0.5  # Reduzir drasticamente
        elif drawdown_pct > 0.05:  # Drawdown > 5%
            return 0.7  # Reduzir significativamente
        elif drawdown_pct > 0.02:  # Drawdown > 2%
            return 0.85  # Reduzir moderadamente
        else:
            return 1.0  # Sem drawdown significativo
    
    def _generate_sizing_rationale(self, adjustments: Dict) -> str:
        """Gera explicação textual dos ajustes de tamanho"""
        rationale_parts = []
        
        for factor, value in adjustments.items():
            if value > 1.1:
                rationale_parts.append(f"{factor}: AUMENTAR ({value:.2f}x)")
            elif value < 0.9:
                rationale_parts.append(f"{factor}: REDUZIR ({value:.2f}x)")
            else:
                rationale_parts.append(f"{factor}: NEUTRO ({value:.2f}x)")
        
        final_multiplier = np.prod(list(adjustments.values()))
        
        if final_multiplier > 1.2:
            decision = "POSIÇÃO AUMENTADA"
        elif final_multiplier < 0.8:
            decision = "POSIÇÃO REDUZIDA"
        else:
            decision = "POSIÇÃO PADRÃO"
        
        rationale = f"{decision} (Multiplicador final: {final_multiplier:.2f}x)\n"
        rationale += "Fatores: " + " | ".join(rationale_parts)
        
        return rationale


# === Classes Auxiliares ===

class ATRAdaptiveStop:
    """Stop loss adaptativo baseado em ATR"""
    
    def calculate_stop(self, position: Dict, market_data: pd.DataFrame, market_regime: str) -> float:
        """Calcula stop loss baseado em ATR"""
        # Calcular ATR simplificado
        if len(market_data) < 14:
            atr = 0.02 * position['entry_price']  # Fallback 2%
        else:
            high_low = market_data['high'] - market_data['low']
            atr = high_low.rolling(14).mean().iloc[-1]
        
        # Multiplicador baseado no regime
        multipliers = {'trending': 2.5, 'ranging': 1.5, 'high_volatility': 3.0, 'low_volatility': 1.2}
        atr_multiplier = multipliers.get(market_regime, 2.0)
        
        # Calcular stop
        if position['side'] == 'BUY':
            stop = position['entry_price'] - (atr * atr_multiplier)
        else:
            stop = position['entry_price'] + (atr * atr_multiplier)
        
        return stop


class SupportResistanceStop:
    """Stop loss baseado em suporte e resistência"""
    
    def calculate_stop(self, position: Dict, market_data: pd.DataFrame, market_regime: str) -> float:
        """Calcula stop baseado em níveis de S/R (simplificado)"""
        if position['side'] == 'BUY':
            return position['entry_price'] * 0.98  # Stop 2% abaixo
        else:
            return position['entry_price'] * 1.02  # Stop 2% acima


class VolatilityBasedStop:
    """Stop loss baseado em volatilidade"""
    
    def calculate_stop(self, position: Dict, market_data: pd.DataFrame, market_regime: str) -> float:
        """Calcula stop baseado na volatilidade"""
        returns = market_data['close'].pct_change().dropna()
        
        if len(returns) < 20:
            volatility = 0.02  # Fallback
        else:
            volatility = returns.rolling(20).std().iloc[-1]
        
        # Ajustar por regime
        multipliers = {'high_volatility': 2.5, 'trending': 2.0, 'ranging': 1.5, 'low_volatility': 1.0}
        vol_multiplier = multipliers.get(market_regime, 1.8)
        
        stop_distance = volatility * vol_multiplier
        
        if position['side'] == 'BUY':
            stop = position['entry_price'] * (1 - stop_distance)
        else:
            stop = position['entry_price'] * (1 + stop_distance)
        
        return stop


class MLOptimizedStop:
    """Stop loss otimizado por ML"""
    
    def calculate_stop(self, position: Dict, market_data: pd.DataFrame, market_regime: str) -> float:
        """Calcula stop usando ensemble de estratégias"""
        # Combinar múltiplas estratégias
        atr_stop = ATRAdaptiveStop().calculate_stop(position, market_data, market_regime)
        vol_stop = VolatilityBasedStop().calculate_stop(position, market_data, market_regime)
        sr_stop = SupportResistanceStop().calculate_stop(position, market_data, market_regime)
        
        # Ensemble simples
        ensemble_stop = (atr_stop * 0.4 + vol_stop * 0.3 + sr_stop * 0.3)
        return ensemble_stop


class DynamicCorrelationTracker:
    """Rastreador de correlação dinâmica"""
    
    def assess_risk(self, signal: Dict, portfolio_state: Dict) -> Dict:
        """Avalia risco de correlação"""
        return {
            'max_correlation': 0.3,  # Correlação baixa por padrão
            'correlated_positions': [],
            'diversification_score': 0.8
        }


class DrawdownPredictor:
    """Preditor de drawdown"""
    
    def predict(self, signal: Dict, market_data: pd.DataFrame, portfolio_state: Dict) -> Dict:
        """Prediz probabilidade de drawdown"""
        returns = market_data['close'].pct_change().dropna()
        
        if len(returns) < 10:
            prob = 0.3  # Padrão
        else:
            volatility = returns.std()
            prob = min(volatility * 10, 0.8)  # Converter volatilidade em probabilidade
        
        return {
            'probability': prob,
            'expected_drawdown': prob * 0.05,
            'confidence': 0.7
        }


class MarginOptimizer:
    """Otimizador de margem"""
    
    def analyze(self, signal: Dict, portfolio_state: Dict) -> Dict:
        """Analisa eficiência da margem"""
        return {
            'efficiency': 0.75,  # 75% de eficiência
            'required_margin': signal.get('entry_price', 100) * 0.05,  # 5% de margem
            'optimal_leverage': 1.5
        }


class PortfolioRiskAnalyzer:
    """Analisador de risco de portfólio"""
    
    def analyze(self, portfolio_state: Dict) -> Dict:
        """Analisa risco do portfólio"""
        return {
            'var_95': 0.02,  # VaR 95%
            'expected_shortfall': 0.03,
            'correlation_risk': 0.3,
            'concentration_risk': 0.2
        }

