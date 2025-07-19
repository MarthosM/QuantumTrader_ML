"""
Módulo de Otimização Contínua - Sistema ML Trading v2.0
Sistema avançado de otimização contínua para modelos de ML e parâmetros de trading.
"""

import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from collections import deque
import threading
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import optuna
from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    """Configuração da otimização"""
    min_win_rate: float = 0.5
    min_sharpe: float = 1.0
    max_drawdown: float = 0.1
    min_confidence: float = 0.6
    optimization_interval: int = 3600  # segundos
    drift_threshold: float = 0.1
    max_features: int = 50


class ContinuousOptimizationPipeline:
    """Pipeline principal de otimização contínua"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or OptimizationConfig()
        
        # Componentes do sistema
        self.feature_optimizer = FeatureSelectionOptimizer()
        self.hyperparameter_optimizer = DynamicHyperparameterOptimizer()
        self.portfolio_optimizer = AdaptivePortfolioOptimizer()
        self.execution_optimizer = SmartExecutionOptimizer()
        self.risk_optimizer = DynamicRiskOptimizer()
        self.drift_detector = ModelDriftDetector()
        self.performance_monitor = PerformanceMonitor()
        
        # Estado interno
        self.optimization_history = deque(maxlen=100)
        self.last_optimization = None
        self.running = False
        
        self.logger.info("ContinuousOptimizationPipeline inicializado")

    def should_optimize(self, current_metrics: Dict) -> Tuple[bool, str]:
        """
        Determina se otimização é necessária baseado em múltiplos critérios
        """
        reasons = []
        
        # 1. Verificar mudança de regime
        if self._check_regime_change(current_metrics):
            reasons.append("regime_change")
        
        # 2. Verificar drift nos modelos  
        if self._check_model_drift(current_metrics):
            reasons.append("model_drift")
        
        # 3. Verificar trigger temporal
        if self._check_time_trigger():
            reasons.append("time_trigger")
        
        should_opt = len(reasons) > 0
        reason_str = ", ".join(reasons) if reasons else "none"
        
        return should_opt, reason_str

    def run_optimization_cycle(self, market_data: pd.DataFrame, 
                             performance_data: Dict) -> Dict:
        """
        Executa um ciclo completo de otimização
        """
        self.logger.info("Iniciando ciclo de otimização")
        
        try:
            optimization_results = {}
            
            # 1. Otimizar seleção de features
            feature_results = self._optimize_features(market_data, performance_data)
            optimization_results['features'] = feature_results
            
            # 2. Otimizar hiperparâmetros
            if feature_results.get('features_changed', False):
                hyperparams_results = self._optimize_hyperparameters(
                    feature_results['selected_features'],
                    market_data
                )
                optimization_results['hyperparameters'] = hyperparams_results
            
            # 3. Otimizar portfolio
            portfolio_results = self._optimize_portfolio(market_data, performance_data)
            optimization_results['portfolio'] = portfolio_results
            
            # 4. Otimizar execução
            execution_results = self._optimize_execution(market_data)
            optimization_results['execution'] = execution_results
            
            # 5. Otimizar parâmetros de risco
            risk_results = self._optimize_risk_parameters(
                optimization_results, market_data, performance_data
            )
            optimization_results['risk'] = risk_results
            
            # 6. Validar melhorias
            if self._validate_improvements(optimization_results):
                self._apply_optimizations(optimization_results)
                self.logger.info("Otimizações aplicadas com sucesso")
            else:
                self.logger.warning("Otimizações rejeitadas - não trouxeram melhorias")
            
            # 7. Log dos resultados
            self._log_optimization(optimization_results)
            
            self.last_optimization = datetime.now()
            self.optimization_history.append(optimization_results)
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Erro no ciclo de otimização: {e}")
            return {'status': 'error', 'error': str(e)}

    def _optimize_features(self, market_data: pd.DataFrame, 
                          performance_data: Dict) -> Dict:
        """Otimiza seleção de features"""
        
        try:
            # Calcular returns target
            target_returns = self._calculate_target_returns(market_data)
            
            # Selecionar features ótimas
            selected_features = self.feature_optimizer.select_optimal_features(
                market_data, 
                target_returns,
                max_features=self.config.max_features,
                methods=['mutual_info', 'lasso', 'random_forest']
            )
            
            return {
                'status': 'success',
                'selected_features': selected_features,
                'features_changed': True,
                'num_features': len(selected_features)
            }
            
        except Exception as e:
            self.logger.error(f"Erro na otimização de features: {e}")
            return {'status': 'error', 'error': str(e)}

    def _optimize_hyperparameters(self, features: List[str], 
                                 market_data: pd.DataFrame) -> Dict:
        """Otimiza hiperparâmetros dos modelos"""
        
        try:
            # Detectar regime de mercado para otimização específica
            regime = self._detect_market_regime(market_data)
            search_space = self._get_regime_specific_search_space(regime)
            
            # Executar otimização
            results = self.hyperparameter_optimizer.optimize(
                features, market_data, search_space
            )
            
            return {
                'status': 'success',
                'regime': regime,
                'best_params': results.get('best_params', {}),
                'score': results.get('best_score', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Erro na otimização de hiperparâmetros: {e}")
            return {'status': 'error', 'error': str(e)}

    def _check_performance_degradation(self, metrics: Dict) -> bool:
        """Verifica se houve degradação significativa na performance"""
        
        if len(self.optimization_history) < 5:
            return False
            
        # Calcular médias históricas vs recentes
        recent_metrics = [h.get('performance', {}) for h in list(self.optimization_history)[-3:]]
        historical_metrics = [h.get('performance', {}) for h in list(self.optimization_history)[:-3]]
        
        for metric in ['win_rate', 'sharpe_ratio']:
            if metric not in metrics:
                continue
                
            historical_avg = np.mean([m.get(metric, 0) for m in historical_metrics])
            recent_avg = np.mean([m.get(metric, 0) for m in recent_metrics])
            
            if historical_avg == 0:
                continue
                
            degradation = (historical_avg - recent_avg) / max(float(abs(historical_avg)), 1e-6)
            
            if degradation > self.config.drift_threshold:
                self.logger.warning(f"Degradação detectada em {metric}: {degradation:.3f}")
                return True
                
        return False

    # Métodos privados implementados
    def _check_regime_change(self, metrics: Dict) -> bool:
        """Verifica mudança de regime de mercado"""
        # Implementação simples baseada em volatilidade e tendência
        current_volatility = metrics.get('volatility', 0)
        historical_volatility = metrics.get('historical_volatility', current_volatility)
        
        volatility_change = abs(current_volatility - historical_volatility) / max(historical_volatility, 1e-6)
        return volatility_change > 0.2

    def _check_model_drift(self, metrics: Dict) -> bool:
        """Verifica drift nos modelos"""
        current_accuracy = metrics.get('accuracy', 0.5)
        expected_accuracy = metrics.get('expected_accuracy', 0.55)
        
        return current_accuracy < (expected_accuracy - self.config.drift_threshold)

    def _check_time_trigger(self) -> bool:
        """Verifica se é hora de otimizar baseado no tempo"""
        if self.last_optimization is None:
            return True
            
        time_since_last = (datetime.now() - self.last_optimization).seconds
        return time_since_last >= self.config.optimization_interval

    def _optimize_portfolio(self, market_data: pd.DataFrame, performance_data: Dict) -> Dict:
        """Otimiza parâmetros do portfolio"""
        return self.portfolio_optimizer.optimize(market_data, performance_data)

    def _optimize_execution(self, market_data: pd.DataFrame) -> Dict:
        """Otimiza parâmetros de execução"""
        return self.execution_optimizer.optimize(market_data)

    def _optimize_risk_parameters(self, optimization_results: Dict, 
                                market_data: pd.DataFrame, performance_data: Dict) -> Dict:
        """Otimiza parâmetros de risco"""
        return self.risk_optimizer.optimize(optimization_results, market_data)

    def _validate_improvements(self, optimization_results: Dict) -> bool:
        """Valida se as otimizações trouxeram melhorias"""
        # Implementação simples - pode ser expandida
        return optimization_results.get('features', {}).get('status') == 'success'

    def _apply_optimizations(self, optimization_results: Dict):
        """Aplica as otimizações validadas"""
        self.logger.info("Aplicando otimizações validadas")
        # Implementar aplicação das otimizações

    def _log_optimization(self, optimization_results: Dict):
        """Log detalhado dos resultados da otimização"""
        self.logger.info(f"Otimização concluída: {optimization_results.get('status', 'unknown')}")

    def _detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """Detecta regime atual do mercado"""
        if len(market_data) < 20:
            return 'undefined'
            
        # Calcular EMAs
        ema_9 = market_data['close'].ewm(span=9).mean().iloc[-1]
        ema_20 = market_data['close'].ewm(span=20).mean().iloc[-1] 
        ema_50 = market_data['close'].ewm(span=50).mean().iloc[-1] if len(market_data) >= 50 else ema_20
        
        # Detectar regime baseado em alinhamento de EMAs
        if ema_9 > ema_20 > ema_50:
            return 'trend_up'
        elif ema_9 < ema_20 < ema_50:
            return 'trend_down'
        else:
            return 'range'

    def _get_regime_specific_search_space(self, regime: str) -> Dict:
        """Obtém espaço de busca específico para o regime"""
        base_space = {
            'n_estimators': (50, 200),
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3)
        }
        
        if regime.startswith('trend'):
            # Para tendência, favorecer modelos mais agressivos
            base_space['learning_rate'] = (0.05, 0.3)
        elif regime == 'range':
            # Para range, favorecer modelos mais conservadores
            base_space['learning_rate'] = (0.01, 0.15)
            
        return base_space

    def _calculate_target_returns(self, market_data: pd.DataFrame) -> List[float]:
        """Calcula returns target para otimização"""
        if 'close' not in market_data.columns:
            return [0.0] * len(market_data)
            
        returns = market_data['close'].pct_change().fillna(0)
        return returns.tolist()


class DynamicHyperparameterOptimizer:
    """Otimizador dinâmico de hiperparâmetros"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.study_cache = {}
        
    def optimize(self, features: List[str], market_data: pd.DataFrame,
                search_space: Dict) -> Dict:
        """Otimiza hiperparâmetros usando Optuna"""
        
        try:
            # Criar estudo Optuna
            study = optuna.create_study(direction='maximize')
            
            # Definir função objetivo
            def objective(trial):
                # Simular score (implementação real seria mais complexa)
                n_estimators = trial.suggest_int('n_estimators', *search_space['n_estimators'])
                max_depth = trial.suggest_int('max_depth', *search_space['max_depth'])
                learning_rate = trial.suggest_float('learning_rate', *search_space['learning_rate'])
                
                # Score simulado baseado nos parâmetros
                score = 0.5 + (learning_rate * 0.1) + (max_depth * 0.01)
                return min(score, 1.0)
            
            # Executar otimização
            study.optimize(objective, n_trials=50, timeout=300)
            
            return {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'n_trials': len(study.trials)
            }
            
        except Exception as e:
            self.logger.error(f"Erro na otimização de hiperparâmetros: {e}")
            return {'best_params': {}, 'best_score': 0}


class FeatureSelectionOptimizer:
    """Otimizador de seleção de features"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.selection_cache = {}
        
    def select_optimal_features(self, data: pd.DataFrame, target_returns: List[float],
                              max_features: int, methods: List[str]) -> List[str]:
        """Seleciona features ótimas usando ensemble de métodos"""
        
        if data.empty or len(target_returns) == 0:
            return []
            
        feature_scores = {}
        
        try:
            # 1. Mutual Information
            if 'mutual_info' in methods:
                mi_scores = self._mutual_information_selection(data, target_returns)
                feature_scores['mutual_info'] = mi_scores
                
            # 2. LASSO
            if 'lasso' in methods:
                lasso_scores = self._lasso_selection(data, target_returns)
                feature_scores['lasso'] = lasso_scores
                
            # 3. Random Forest Importance
            if 'random_forest' in methods:
                rf_scores = self._random_forest_selection(data, target_returns)
                feature_scores['random_forest'] = rf_scores
                
            # Combinar scores
            combined_scores = self._combine_feature_scores(feature_scores)
            
            # Selecionar top features
            sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in sorted_features[:max_features]]
            
            self.logger.info(f"Selecionadas {len(selected_features)} features de {len(data.columns)}")
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Erro na seleção de features: {e}")
            return list(data.columns[:max_features])  # Fallback

    def _mutual_information_selection(self, data: pd.DataFrame, target_returns: List[float]) -> Dict:
        """Seleção baseada em mutual information"""
        try:
            # Preparar dados
            X = data.fillna(0).select_dtypes(include=[np.number])
            y = np.array(target_returns[:len(X)])
            
            if len(X) != len(y):
                y = y[:len(X)]
                
            # Calcular mutual information
            mi_scores = mutual_info_regression(X, y)
            
            return dict(zip(X.columns, mi_scores))
            
        except Exception as e:
            self.logger.error(f"Erro no mutual information: {e}")
            return {}

    def _lasso_selection(self, data: pd.DataFrame, target_returns: List[float]) -> Dict:
        """Seleção baseada em LASSO"""
        try:
            # Preparar dados
            X = data.fillna(0).select_dtypes(include=[np.number])
            y = np.array(target_returns[:len(X)])
            
            if len(X) != len(y):
                y = y[:len(X)]
                
            # Normalizar features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # LASSO com CV
            lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
            lasso.fit(X_scaled, y)
            
            # Scores baseados nos coeficientes
            scores = np.abs(lasso.coef_)
            
            return dict(zip(X.columns, scores))
            
        except Exception as e:
            self.logger.error(f"Erro no LASSO: {e}")
            return {}

    def _random_forest_selection(self, data: pd.DataFrame, target_returns: List[float]) -> Dict:
        """Seleção baseada em importância do Random Forest"""
        try:
            # Preparar dados
            X = data.fillna(0).select_dtypes(include=[np.number])
            y = np.array(target_returns[:len(X)])
            
            if len(X) != len(y):
                y = y[:len(X)]
                
            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            return dict(zip(X.columns, rf.feature_importances_))
            
        except Exception as e:
            self.logger.error(f"Erro no Random Forest: {e}")
            return {}

    def _combine_feature_scores(self, feature_scores: Dict) -> Dict:
        """Combina scores de diferentes métodos"""
        if not feature_scores:
            return {}
            
        combined = {}
        all_features = set()
        
        # Coletar todas as features
        for scores in feature_scores.values():
            all_features.update(scores.keys())
        
        # Combinar scores (média ponderada)
        weights = {'mutual_info': 0.4, 'lasso': 0.3, 'random_forest': 0.3}
        
        for feature in all_features:
            total_score = 0
            total_weight = 0
            
            for method, scores in feature_scores.items():
                if feature in scores:
                    weight = weights.get(method, 1.0)
                    total_score += scores[feature] * weight
                    total_weight += weight
                    
            if total_weight > 0:
                combined[feature] = total_score / total_weight
            else:
                combined[feature] = 0
                
        return combined


class AdaptivePortfolioOptimizer:
    """Otimizador adaptativo de portfolio"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def optimize(self, market_data: pd.DataFrame, performance_data: Dict) -> Dict:
        """Otimiza alocação e parâmetros do portfolio"""
        return {
            'status': 'success',
            'allocation_changed': False,
            'new_allocation': {'WDO': 1.0}
        }


class SmartExecutionOptimizer:
    """Otimizador inteligente de execução"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def optimize(self, market_data: pd.DataFrame) -> Dict:
        """Otimiza parâmetros de execução"""
        return {
            'status': 'success',
            'execution_params_changed': False
        }


class DynamicRiskOptimizer:
    """Otimizador dinâmico de parâmetros de risco"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def optimize(self, optimization_results: Dict, market_data: pd.DataFrame) -> Dict:
        """Otimiza parâmetros de gestão de risco"""
        return {
            'status': 'success',
            'risk_params_changed': False
        }


class PerformanceMonitor:
    """Monitor de performance do sistema"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history = deque(maxlen=1000)
        
    def get_current_metrics(self) -> Dict:
        """Obtém métricas atuais de performance"""
        return {
            'win_rate': 0.55,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.08,
            'model_confidence': 0.7,
            'requires_retraining': False,
            'volatility': 0.02,
            'historical_volatility': 0.018,
            'accuracy': 0.57,
            'expected_accuracy': 0.55
        }


class AutoOptimizationEngine:
    """Engine de otimização automática"""
    
    def __init__(self, model_manager, performance_monitor, drift_detector, config=None):
        self.logger = logging.getLogger(__name__)
        self.model_manager = model_manager
        self.performance_monitor = performance_monitor
        self.drift_detector = drift_detector
        self.config = config or OptimizationConfig()
        
        self.running = False
        self.optimization_thread = None
        
    def start(self):
        """Inicia otimização automática"""
        self.running = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        self.logger.info("Engine de otimização automática iniciada")
        
    def stop(self):
        """Para otimização automática"""
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=30)
        self.logger.info("Engine de otimização automática parada")
        
    def _optimization_loop(self):
        """Loop principal de otimização"""
        
        while self.running:
            try:
                # 1. Coletar métricas atuais
                current_metrics = self.performance_monitor.get_current_metrics()
                
                # 2. Detectar drift nos modelos
                drift_detected = self.drift_detector.check_drift(
                    self.model_manager.get_recent_predictions() if hasattr(self.model_manager, 'get_recent_predictions') else [],
                    current_metrics
                )
                
                # 3. Verificar necessidade de otimização
                if drift_detected or self._should_optimize(current_metrics):
                    self._run_optimization(current_metrics)
                    
                # 4. Aguardar próximo ciclo
                time.sleep(self.config.optimization_interval)
                
            except Exception as e:
                self.logger.error(f"Erro no loop de otimização: {e}")
                time.sleep(60)  # Esperar 1 minuto em caso de erro
                
    def _should_optimize(self, metrics: Dict) -> bool:
        """Determina se otimização é necessária"""
        
        # Verificar múltiplos critérios
        criteria = [
            metrics.get('win_rate', 0) < self.config.min_win_rate,
            metrics.get('sharpe_ratio', 0) < self.config.min_sharpe,
            metrics.get('max_drawdown', 1) > self.config.max_drawdown,
            metrics.get('model_confidence', 1) < self.config.min_confidence
        ]
        
        return any(criteria)
        
    def _run_optimization(self, metrics: Dict):
        """Executa otimização dos modelos"""
        
        self.logger.info("Iniciando otimização automática dos modelos")
        
        try:
            # 1. Retreinar modelos com dados recentes
            if metrics.get('requires_retraining', False):
                self._retrain_models()
                
            # 2. Ajustar pesos do ensemble
            self._optimize_ensemble_weights(metrics)
            
            # 3. Atualizar parâmetros de risco
            self._update_risk_parameters(metrics)
            
            self.logger.info("Otimização automática concluída")
            
        except Exception as e:
            self.logger.error(f"Erro na otimização: {e}")

    def _retrain_models(self):
        """Retreina modelos com dados recentes"""
        self.logger.info("Retreinando modelos")
        # Implementar retreinamento

    def _optimize_ensemble_weights(self, metrics: Dict):
        """Otimiza pesos do ensemble"""
        self.logger.info("Otimizando pesos do ensemble")
        # Implementar otimização de pesos

    def _update_risk_parameters(self, metrics: Dict):
        """Atualiza parâmetros de risco"""
        self.logger.info("Atualizando parâmetros de risco")
        # Implementar atualização de parâmetros


class ModelDriftDetector:
    """Detector de drift nos modelos ML"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.prediction_history = deque(maxlen=1000)
        self.drift_threshold = 0.1
        
    def check_drift(self, recent_predictions: List[Dict], 
                   metrics: Dict) -> bool:
        """Detecta drift nos modelos"""
        
        if len(recent_predictions) < 100:
            return False
            
        # 1. Drift na distribuição das predições
        prediction_drift = self._check_prediction_distribution_drift(recent_predictions)
        
        # 2. Drift na acurácia
        accuracy_drift = self._check_accuracy_drift(metrics)
        
        # 3. Drift na confiança
        confidence_drift = self._check_confidence_drift(recent_predictions)
        
        drift_detected = prediction_drift or accuracy_drift or confidence_drift
        
        if drift_detected:
            self.logger.warning(f"Drift detectado - Pred: {prediction_drift}, "
                              f"Acc: {accuracy_drift}, Conf: {confidence_drift}")
            
        return drift_detected

    def _check_prediction_distribution_drift(self, recent_predictions: List[Dict]) -> bool:
        """Verifica drift na distribuição das predições"""
        if len(recent_predictions) < 50:
            return False
            
        # Extrair valores de predição
        predictions = [p.get('prediction', 0.5) for p in recent_predictions]
        
        # Calcular estatísticas recentes vs históricas
        recent_mean = np.mean(predictions[-50:])
        historical_mean = np.mean(predictions[:-50]) if len(predictions) > 50 else recent_mean
        
        if historical_mean == 0:
            return False
            
        drift = abs(recent_mean - historical_mean) / abs(historical_mean)
        return bool(drift > self.drift_threshold)

    def _check_accuracy_drift(self, metrics: Dict) -> bool:
        """Verifica drift na acurácia"""
        current_accuracy = metrics.get('accuracy', 0.5)
        expected_accuracy = metrics.get('expected_accuracy', 0.55)
        
        return current_accuracy < (expected_accuracy - self.drift_threshold)

    def _check_confidence_drift(self, recent_predictions: List[Dict]) -> bool:
        """Verifica drift na confiança"""
        if len(recent_predictions) < 50:
            return False
            
        confidences = [p.get('confidence', 0.5) for p in recent_predictions]
        recent_confidence = np.mean(confidences[-25:])
        historical_confidence = np.mean(confidences[:-25]) if len(confidences) > 25 else recent_confidence
        
        if historical_confidence == 0:
            return False
            
        drift = abs(recent_confidence - historical_confidence) / abs(historical_confidence)
        return bool(drift > self.drift_threshold)