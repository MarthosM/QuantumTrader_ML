"""
Coordenador ML - Sistema v2.0
Coordena detecção de regime e predição baseada em regime
"""

from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging


class MLCoordinator:
    """Coordena todo o processo de ML com foco em regime"""
    
    def __init__(self, model_manager, feature_engine, prediction_engine, 
                 regime_trainer=None):
        """
        Inicializa o coordenador
        
        Args:
            model_manager: Gerenciador de modelos
            feature_engine: Motor de cálculo de features
            prediction_engine: Motor de predições
            regime_trainer: Treinador de regime (essencial)
        """
        self.model_manager = model_manager
        self.feature_engine = feature_engine
        self.prediction_engine = prediction_engine
        self.regime_trainer = regime_trainer
        
        self.logger = logging.getLogger(__name__)
        
        # Verificar regime trainer
        if regime_trainer is None:
            self.logger.warning("Regime trainer não fornecido - sistema limitado")
        
        # Configurações
        self.min_candles = 50
        self.prediction_interval = 60  # segundos
        
        # Estado
        self.last_prediction_time = None
        self.last_regime_analysis = None
        self.prediction_count = 0
        self.regime_history = []
        
        # Estatísticas
        self.stats = {
            'total_predictions': 0,
            'trend_predictions': 0,
            'range_predictions': 0,
            'no_trade_predictions': 0
        }
        
    def process_prediction_request(self, data) -> Optional[Dict[str, Any]]:
        """
        Processa requisição seguindo o fluxo:
        1. Detectar regime
        2. Fazer predição baseada no regime
        3. Validar se pode operar
        """
        self.logger.info("="*60)
        self.logger.info("[ML_COORD] Iniciando processo de predição baseado em regime")
        
        # Verificar timing
        if not self._should_predict():
            self.logger.debug("Ainda não é hora de nova predição")
            return self.prediction_engine.last_prediction
        
        # Verificar dados
        if len(data.candles) < self.min_candles:
            self.logger.warning(f"Dados insuficientes: {len(data.candles)} candles")
            return None
        
        try:
            # Log estado inicial
            current_price = data.candles['close'].iloc[-1]
            self.logger.info(f"[ML_COORD] Preço atual: {current_price:.2f}")
            self.logger.info(f"[ML_COORD] Candles disponíveis: {len(data.candles)}")
            
            # PASSO 1: DETECTAR REGIME DE MERCADO
            self.logger.info("[ML_COORD] Passo 1: Detectando regime de mercado...")
            
            regime_info = self._detect_market_regime(data)
            
            if not regime_info or regime_info.get('regime') == 'undefined':
                self.logger.warning("[ML_COORD] Falha na detecção de regime")
                return self._create_no_trade_prediction("regime_undefined")
            
            regime = regime_info['regime']
            regime_confidence = regime_info.get('confidence', 0)
            
            self.logger.info(f"[ML_COORD] Regime detectado: {regime} (confiança: {regime_confidence:.2%})")
            
            # Verificar confiança mínima no regime
            if regime_confidence < 0.6:
                self.logger.warning(f"[ML_COORD] Confiança no regime muito baixa: {regime_confidence:.2%}")
                return self._create_no_trade_prediction("low_regime_confidence")
            
            # PASSO 2: CALCULAR FEATURES
            self.logger.info("[ML_COORD] Passo 2: Calculando features...")
            
            feature_result = self.feature_engine.calculate(data)
            
            if not feature_result or 'model_ready' not in feature_result:
                self.logger.error("Falha no cálculo de features")
                return self._create_no_trade_prediction("feature_calculation_failed")
            
            model_features = feature_result['model_ready']
            
            # PASSO 3: PREDIÇÃO BASEADA NO REGIME
            self.logger.info(f"[ML_COORD] Passo 3: Executando predição para regime {regime}...")
            
            prediction = self.prediction_engine.predict_by_regime(
                model_features, 
                regime_info
            )
            
            if prediction is None:
                self.logger.error("Predição falhou")
                return self._create_no_trade_prediction("prediction_failed")
            
            # PASSO 4: VALIDAR SE PODE OPERAR
            self.logger.info("[ML_COORD] Passo 4: Validando condições de trading...")
            
            can_trade = prediction.get('can_trade', False)
            
            if not can_trade:
                reason = self._get_no_trade_reason(prediction, regime_info)
                self.logger.info(f"[ML_COORD] Sem oportunidade de trade: {reason}")
                prediction['trade_decision'] = 'HOLD'
                prediction['reason'] = reason
            else:
                # Determinar ação baseada na direção
                if prediction['direction'] > 0:
                    prediction['trade_decision'] = 'BUY'
                    prediction['reason'] = f"{regime}_buy_signal"
                else:
                    prediction['trade_decision'] = 'SELL'
                    prediction['reason'] = f"{regime}_sell_signal"
                
                # Adicionar parâmetros de risco baseados no regime
                if regime in ['trend_up', 'trend_down']:
                    prediction['risk_reward_target'] = 2.0  # 2:1 para tendência
                else:
                    prediction['risk_reward_target'] = 1.5  # 1.5:1 para range
            
            # Adicionar metadados
            prediction['analysis_timestamp'] = datetime.now()
            prediction['candles_analyzed'] = len(data.candles)
            prediction['current_price'] = current_price
            
            # Atualizar estatísticas
            self._update_statistics(prediction, regime)
            
            # Log resultado final
            self._log_final_decision(prediction)
            
            # Atualizar estado
            self.last_prediction_time = datetime.now()
            self.prediction_count += 1
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"[ML_COORD] Erro no processo: {e}", exc_info=True)
            return self._create_no_trade_prediction("process_error")
    
    def _detect_market_regime(self, data) -> Optional[Dict[str, Any]]:
        """Detecta regime de mercado usando regime trainer"""
        if not self.regime_trainer:
            self.logger.error("Regime trainer não disponível")
            return None
        
        try:
            # Preparar dados para análise
            # O regime trainer espera dados unificados
            unified_data = self._prepare_unified_data(data)
            
            # Analisar mercado
            regime_info = self.regime_trainer.analyze_market(unified_data)
            
            if not regime_info:
                return None
            
            # Enriquecer com análise adicional
            regime = regime_info.get('regime', 'undefined')
            
            # Para range, detectar proximidade de níveis
            if regime in ['range', 'ranging']:
                regime_info.update(self._analyze_support_resistance(data))
            
            # Para trend, confirmar direção
            elif regime in ['trend_up', 'trend_down']:
                regime_info.update(self._confirm_trend_direction(data))
            
            # Armazenar no histórico
            regime_info['detection_time'] = datetime.now()
            self.regime_history.append(regime_info)
            if len(self.regime_history) > 50:
                self.regime_history.pop(0)
            
            self.last_regime_analysis = regime_info
            
            return regime_info
            
        except Exception as e:
            self.logger.error(f"Erro detectando regime: {e}")
            return None
    
    def _prepare_unified_data(self, data) -> pd.DataFrame:
        """Prepara dados unificados para análise de regime"""
        # Começar com candles
        unified = data.candles.copy()
        
        # Adicionar indicadores principais se disponíveis
        if not data.indicators.empty:
            key_indicators = ['ema_20', 'ema_50', 'rsi', 'atr', 'bb_upper_20', 'bb_lower_20']
            for ind in key_indicators:
                if ind in data.indicators.columns:
                    unified[ind] = data.indicators[ind]
        
        # Adicionar features principais se disponíveis
        if not data.features.empty:
            key_features = ['momentum_5', 'momentum_10', 'volatility_20']
            for feat in key_features:
                if feat in data.features.columns:
                    unified[feat] = data.features[feat]
        
        return unified
    
    def _analyze_support_resistance(self, data) -> Dict[str, Any]:
        """Analisa proximidade de suporte/resistência em range"""
        candles = data.candles
        current_price = candles['close'].iloc[-1]
        
        # Calcular níveis usando máximas e mínimas recentes
        lookback = min(50, len(candles))
        recent_data = candles.tail(lookback)
        
        # Identificar pivôs
        highs = recent_data['high'].rolling(5, center=True).max() == recent_data['high']
        lows = recent_data['low'].rolling(5, center=True).min() == recent_data['low']
        
        resistance_levels = recent_data.loc[highs, 'high'].values
        support_levels = recent_data.loc[lows, 'low'].values
        
        # Encontrar níveis mais próximos
        if len(resistance_levels) > 0:
            nearest_resistance = resistance_levels[resistance_levels > current_price]
            nearest_resistance = float(nearest_resistance[0]) if len(nearest_resistance) > 0 else current_price * 1.01
        else:
            nearest_resistance = current_price * 1.01
            
        if len(support_levels) > 0:
            nearest_support = support_levels[support_levels < current_price]
            nearest_support = float(nearest_support[-1]) if len(nearest_support) > 0 else current_price * 0.99
        else:
            nearest_support = current_price * 0.99
        
        # Calcular distâncias percentuais
        dist_to_resistance = (nearest_resistance - current_price) / current_price
        dist_to_support = (current_price - nearest_support) / current_price
        
        # Determinar proximidade
        proximity_threshold = 0.002  # 0.2%
        
        if dist_to_support <= proximity_threshold:
            proximity = 'near_support'
        elif dist_to_resistance <= proximity_threshold:
            proximity = 'near_resistance'
        else:
            proximity = 'neutral'
        
        return {
            'support_level': nearest_support,
            'resistance_level': nearest_resistance,
            'distance_to_support': dist_to_support,
            'distance_to_resistance': dist_to_resistance,
            'support_resistance_proximity': proximity,
            'key_levels': {
                'support': nearest_support,
                'resistance': nearest_resistance
            }
        }
    
    def _confirm_trend_direction(self, data) -> Dict[str, Any]:
        """Confirma direção da tendência"""
        candles = data.candles
        
        # Usar médias móveis se disponíveis
        trend_confirmations = []
        
        if not data.indicators.empty:
            last_ind = data.indicators.iloc[-1]
            
            # EMA alignment
            if all(col in data.indicators.columns for col in ['ema_9', 'ema_20', 'ema_50']):
                if last_ind['ema_9'] > last_ind['ema_20'] > last_ind['ema_50']:
                    trend_confirmations.append(1)
                elif last_ind['ema_9'] < last_ind['ema_20'] < last_ind['ema_50']:
                    trend_confirmations.append(-1)
        
        # Price action
        close_prices = candles['close'].tail(20)
        if len(close_prices) >= 20:
            slope = np.polyfit(range(len(close_prices)), close_prices.values, 1)[0]
            trend_confirmations.append(1 if slope > 0 else -1)
        
        # Consenso
        if trend_confirmations:
            trend_direction = np.sign(np.mean(trend_confirmations))
        else:
            trend_direction = 0
        
        return {
            'trend_direction_confirmed': trend_direction,
            'trend_strength': abs(np.mean(trend_confirmations)) if trend_confirmations else 0
        }
    
    def _get_no_trade_reason(self, prediction: Dict, regime_info: Dict) -> str:
        """Determina razão para não operar"""
        if prediction['confidence'] < 0.5:
            return "low_confidence"
        
        if abs(prediction['direction']) < 0.3:
            return "weak_direction"
        
        regime = regime_info.get('regime')
        if regime in ['range', 'ranging']:
            proximity = regime_info.get('support_resistance_proximity')
            if proximity == 'neutral':
                return "middle_of_range"
        
        return "conditions_not_met"
    
    def _create_no_trade_prediction(self, reason: str) -> Dict[str, Any]:
        """Cria predição indicando não operar"""
        prediction = {
            'timestamp': datetime.now(),
            'direction': 0.0,
            'magnitude': 0.0,
            'confidence': 0.0,
            'can_trade': False,
            'trade_decision': 'HOLD',
            'reason': reason,
            'regime': 'undefined'
        }
        
        # Atualizar estatísticas mesmo para no_trade
        self._update_statistics(prediction, 'undefined')
        
        return prediction
    
    def _update_statistics(self, prediction: Dict, regime: str):
        """Atualiza estatísticas internas"""
        self.stats['total_predictions'] += 1
        
        if prediction.get('can_trade'):
            if regime in ['trend_up', 'trend_down']:
                self.stats['trend_predictions'] += 1
            else:
                self.stats['range_predictions'] += 1
        else:
            self.stats['no_trade_predictions'] += 1
    
    def _log_final_decision(self, prediction: Dict):
        """Log da decisão final"""
        self.logger.info("="*50)
        self.logger.info("[ML_COORD] DECISÃO FINAL:")
        self.logger.info(f"  Regime: {prediction.get('regime', 'N/A')}")
        self.logger.info(f"  Decisão: {prediction.get('trade_decision', 'N/A')}")
        self.logger.info(f"  Direção: {prediction.get('direction', 0):.3f}")
        self.logger.info(f"  Confiança: {prediction.get('confidence', 0):.2%}")
        
        if prediction.get('can_trade'):
            self.logger.info(f"  Risk/Reward alvo: 1:{prediction.get('risk_reward_target', 0)}")
        else:
            self.logger.info(f"  Razão hold: {prediction.get('reason', 'N/A')}")
        
        self.logger.info("="*50)
    
    def get_coordinator_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do coordenador"""
        stats = self.stats.copy()
        
        # Adicionar distribuição de regimes
        if self.regime_history:
            regimes = [r['regime'] for r in self.regime_history[-50:]]
            regime_dist: Dict[str, float] = {}
            for regime in set(regimes):
                regime_dist[regime] = regimes.count(regime) / len(regimes)
            stats['regime_distribution'] = regime_dist  # type: ignore
        
        # Taxa de sinais
        if stats['total_predictions'] > 0:
            stats['trade_rate'] = int((stats['trend_predictions'] + stats['range_predictions']) / stats['total_predictions'])
        
        return stats

    def _should_predict(self) -> bool:
        """
        Verifica se é hora de realizar uma nova predição com base no intervalo definido.
        """
        now = datetime.now()
        if self.last_prediction_time is None:
            return True
        elapsed = (now - self.last_prediction_time).total_seconds()
        return elapsed >= self.prediction_interval
    
    def force_prediction(self, data) -> Optional[Dict[str, Any]]:
        """
        Força uma nova predição ignorando o intervalo de tempo.
        
        Args:
            data: TradingDataStructure com dados do mercado
            
        Returns:
            Dict com resultado da predição ou None se falhar
        """
        self.logger.info("[ML_COORD] Forçando nova predição (ignorando interval)")
        
        # Temporariamente ignorar o intervalo
        original_time = self.last_prediction_time
        self.last_prediction_time = None
        
        try:
            # Processar predição normalmente
            result = self.process_prediction_request(data)
            
            if result is not None:
                # Atualizar timestamp para nova predição
                self.last_prediction_time = datetime.now()
                self.logger.info("[ML_COORD] Predição forçada executada com sucesso")
            else:
                # Restaurar timestamp original se falhou
                self.last_prediction_time = original_time
                
            return result
            
        except Exception as e:
            # Restaurar timestamp original em caso de erro
            self.last_prediction_time = original_time
            self.logger.error(f"[ML_COORD] Erro na predição forçada: {e}")
            return None