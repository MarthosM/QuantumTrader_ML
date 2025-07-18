"""
Motor de predições ML - Sistema v2.0
Executa predições baseadas em regime de mercado
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import logging
from datetime import datetime


class PredictionEngine:
    """Motor de predições ML com suporte a regime"""
    
    def __init__(self, model_manager):
        """
        Inicializa o motor de predições
        
        Args:
            model_manager: Gerenciador de modelos carregados
        """
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        
        # Estado
        self.last_prediction = None
        self.prediction_history = []
        self.max_history = 100
        
        # Configurações por regime
        self.regime_configs = {
            'trend': {
                'min_probability': 0.60,
                'risk_reward': 2.0,
                'models_prefix': 'trend'
            },
            'range': {
                'min_probability': 0.55,
                'risk_reward': 1.5,
                'models_prefix': 'range'
            }
        }
        
    def predict_by_regime(self, features_df: pd.DataFrame, regime_info: Dict) -> Optional[Dict[str, Any]]:
        """
        Executa predição baseada no regime de mercado
        
        Args:
            features_df: DataFrame com features preparadas
            regime_info: Informações do regime detectado
            
        Returns:
            Dict com resultado da predição ou None se falhar
        """
        if features_df.empty or len(features_df) < 50:
            self.logger.warning(f"Dados insuficientes para predição: {len(features_df)} linhas")
            return None
        
        regime = regime_info.get('regime', 'unknown')
        regime_confidence = regime_info.get('confidence', 0)
        
        self.logger.info(f"[PRED_ENGINE] Predição para regime: {regime} (confiança: {regime_confidence:.2%})")
        
        # Mapear regime para tipo
        if regime in ['trend_up', 'trend_down', 'trending']:
            regime_type = 'trend'
            trend_direction = 1 if regime == 'trend_up' else -1 if regime == 'trend_down' else 0
        elif regime in ['range', 'ranging', 'sideways']:
            regime_type = 'range'
            trend_direction = 0
        else:
            self.logger.warning(f"Regime desconhecido: {regime}")
            return None
        
        # Obter modelos específicos do regime
        regime_models = self._get_regime_models(regime_type)
        
        if not regime_models:
            self.logger.error(f"Nenhum modelo encontrado para regime {regime_type}")
            return None
        
        # Executar predições
        predictions = {}
        
        for model_name, model in regime_models.items():
            try:
                # Preparar features
                model_features = self.model_manager.model_features.get(model_name, [])
                X = self._prepare_features(features_df, model_features)
                
                if X is None:
                    continue
                
                # Executar predição
                if regime_type == 'trend':
                    pred = self._predict_trend(model, X, trend_direction, regime_info)
                else:  # range
                    pred = self._predict_range(model, X, regime_info)
                
                if pred is not None:
                    predictions[model_name] = pred
                    
            except Exception as e:
                self.logger.error(f"Erro no modelo {model_name}: {e}")
        
        if not predictions:
            self.logger.error("Nenhuma predição válida gerada")
            return None
        
        # Calcular resultado final
        result = self._calculate_regime_based_result(predictions, regime_type, regime_info)
        
        # Adicionar informações do regime
        result['regime'] = regime
        result['regime_type'] = regime_type
        result['regime_confidence'] = regime_confidence
        result['regime_details'] = regime_info
        
        # Armazenar
        self.last_prediction = result
        self.prediction_history.append(result)
        if len(self.prediction_history) > self.max_history:
            self.prediction_history.pop(0)
        
        return result
    
    def _get_regime_models(self, regime_type: str) -> Dict[str, Any]:
        """Obtém modelos específicos para o regime"""
        regime_models = {}
        prefix = self.regime_configs[regime_type]['models_prefix']
        
        for model_name, model in self.model_manager.models.items():
            if prefix in model_name.lower():
                regime_models[model_name] = model
        
        self.logger.info(f"Encontrados {len(regime_models)} modelos para {regime_type}")
        return regime_models
    
    def _predict_trend(self, model: Any, X: pd.DataFrame, 
                      trend_direction: int, regime_info: Dict) -> Optional[np.ndarray]:
        """
        Predição específica para tendência
        Busca operações a favor da tendência com R:R 2:1
        """
        try:
            # Obter probabilidade do modelo
            probability = self._get_model_probability(model, X)
            
            if probability is None:
                return None
            
            # Em tendência, só opera a favor
            min_prob = self.regime_configs['trend']['min_probability']
            
            # Para tendência de alta
            if trend_direction > 0:
                # Sinal de compra se probabilidade alta
                if probability >= min_prob:
                    direction = 1.0
                    confidence = probability
                else:
                    direction = 0.0
                    confidence = 1 - probability
                    
            # Para tendência de baixa
            elif trend_direction < 0:
                # Sinal de venda se probabilidade baixa (indica queda)
                if probability <= (1 - min_prob):
                    direction = -1.0
                    confidence = 1 - probability
                else:
                    direction = 0.0
                    confidence = probability
            else:
                # Tendência indefinida
                direction = 0.0
                confidence = 0.5
            
            # Magnitude esperada para tendências (maior movimento)
            magnitude = 0.003 if abs(direction) > 0 else 0.0001
            
            self.logger.debug(f"Trend prediction: dir={direction}, prob={probability:.3f}, conf={confidence:.3f}")
            
            return np.array([direction, magnitude, confidence])
            
        except Exception as e:
            self.logger.error(f"Erro em _predict_trend: {e}")
            return None
    
    def _predict_range(self, model: Any, X: pd.DataFrame, 
                      regime_info: Dict) -> Optional[np.ndarray]:
        """
        Predição específica para lateralização
        Busca reversões em suporte/resistência
        """
        try:
            # Obter probabilidade do modelo
            probability = self._get_model_probability(model, X)
            
            if probability is None:
                return None
            
            # Verificar proximidade de níveis
            proximity = regime_info.get('support_resistance_proximity', 'neutral')
            min_prob = self.regime_configs['range']['min_probability']
            
            direction = 0.0
            confidence = 0.5
            
            # Near support - procura compra
            if proximity == 'near_support':
                if probability >= min_prob:
                    direction = 1.0
                    confidence = probability
                else:
                    direction = 0.0
                    confidence = 0.3
                    
            # Near resistance - procura venda
            elif proximity == 'near_resistance':
                if probability <= (1 - min_prob):
                    direction = -1.0
                    confidence = 1 - probability
                else:
                    direction = 0.0
                    confidence = 0.3
            else:
                # Meio do range - sem sinal
                direction = 0.0
                confidence = 0.2
            
            # Magnitude esperada para range (menor movimento)
            magnitude = 0.0015 if abs(direction) > 0 else 0.0001
            
            self.logger.debug(f"Range prediction: dir={direction}, proximity={proximity}, conf={confidence:.3f}")
            
            return np.array([direction, magnitude, confidence])
            
        except Exception as e:
            self.logger.error(f"Erro em _predict_range: {e}")
            return None
    
    def _get_model_probability(self, model: Any, X: pd.DataFrame) -> Optional[float]:
        """Obtém probabilidade do modelo"""
        try:
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X)
                # Assumir classificação binária
                return float(probas[0, 1] if probas.shape[1] > 1 else probas[0, 0])
            elif hasattr(model, 'predict'):
                pred = model.predict(X)[0]
                # Converter predição binária em probabilidade
                return 0.7 if pred == 1 else 0.3
            else:
                return None
        except Exception as e:
            self.logger.error(f"Erro obtendo probabilidade: {e}")
            return None
    
    def _prepare_features(self, features_df: pd.DataFrame, 
                         required_features: List[str]) -> Optional[pd.DataFrame]:
        """Prepara features para o modelo"""
        try:
            # Verificar features faltantes
            missing = set(required_features) - set(features_df.columns)
            if missing:
                self.logger.warning(f"Features faltantes: {missing}")
                # Tentar continuar com features disponíveis
                available = [f for f in required_features if f in features_df.columns]
                if not available:
                    return None
                required_features = available
            
            # Selecionar e limpar
            X = features_df[required_features].copy()
            X = X.ffill().fillna(0)
            
            # Pegar última linha
            return X.iloc[[-1]]
            
        except Exception as e:
            self.logger.error(f"Erro preparando features: {e}")
            return None
    
    def _calculate_regime_based_result(self, predictions: Dict, 
                                     regime_type: str, regime_info: Dict) -> Dict:
        """Calcula resultado final baseado no regime"""
        # Extrair valores das predições
        directions = []
        magnitudes = []
        confidences = []
        
        for pred in predictions.values():
            if isinstance(pred, (list, np.ndarray)) and len(pred) >= 3:
                directions.append(float(pred[0]))
                magnitudes.append(float(pred[1]))
                confidences.append(float(pred[2]))
        
        if not directions:
            return {
                'timestamp': datetime.now(),
                'direction': 0.0,
                'magnitude': 0.0001,
                'confidence': 0.0,
                'can_trade': False
            }
        
        # Calcular consenso
        avg_direction = np.mean(directions)
        avg_magnitude = np.mean(magnitudes)
        avg_confidence = np.mean(confidences)
        
        # Determinar se pode operar
        can_trade = False
        
        if regime_type == 'trend':
            # Em tendência, precisa direção forte e confiança alta
            can_trade = (abs(avg_direction) >= 0.7 and 
                        avg_confidence >= self.regime_configs['trend']['min_probability'])
        else:  # range
            # Em range, precisa estar perto de níveis
            proximity = regime_info.get('support_resistance_proximity', 'neutral')
            can_trade = (abs(avg_direction) >= 0.5 and 
                        avg_confidence >= self.regime_configs['range']['min_probability'] and
                        proximity in ['near_support', 'near_resistance'])
        
        return {
            'timestamp': datetime.now(),
            'direction': avg_direction,
            'magnitude': avg_magnitude,
            'confidence': avg_confidence,
            'can_trade': can_trade,
            'predictions_by_model': predictions,
            'models_used': len(predictions)
        }