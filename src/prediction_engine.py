import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

class PredictionEngine:
    """
    Motor de predições ML integrado com ModelManager
    
    Executa predições reais usando modelos treinados.
    Falha adequadamente quando modelos não estão disponíveis.
    Suporta predições baseadas em regime de mercado.
    """
    
    def __init__(self, model_manager, logger=None):
        """
        Inicializa PredictionEngine com integração real ao ModelManager
        
        Args:
            model_manager: Gerenciador de modelos ML treinados (obrigatório)
            logger: Logger opcional (será criado se não fornecido)
        """
        self.model_manager = model_manager
        self.logger = logger or logging.getLogger(__name__)
        
    def predict(self, features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Gera predição baseada nas features"""
        try:
            if features.empty:
                self.logger.warning("⚠️ Features vazias para predição")
                return None
                
            # Verificar se temos modelos carregados
            if not hasattr(self.model_manager, 'models') or not self.model_manager.models:
                self.logger.error("❌ NENHUM MODELO DISPONÍVEL - Predição impossível")
                return None
                
            # Usar ModelManager para predição real
            try:
                model_result = self.model_manager.predict(features)
                if model_result is not None:
                    # Converter resultado do ModelManager para formato esperado
                    prediction = self._convert_model_result(model_result)
                    self.logger.info(f"🎯 Predição REAL gerada: dir={prediction['direction']:.3f}, conf={prediction['confidence']:.3f}")
                    return prediction
                else:
                    self.logger.error("❌ ModelManager retornou None - Predição falhou")
                    return None
                    
            except Exception as e:
                self.logger.error(f"❌ ERRO CRÍTICO no ModelManager: {e}")
                return None
            
        except Exception as e:
            self.logger.error(f"❌ Erro em predict: {e}")
            return None
    
    def predict_by_regime(self, features: pd.DataFrame, regime_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Gera predição baseada em features e informações de regime
        
        Args:
            features: DataFrame com features calculadas
            regime_info: Informações do regime detectado pelo RegimeAnalyzer
            
        Returns:
            Dict com predição ajustada para o regime ou None se erro
        """
        try:
            if features.empty:
                self.logger.warning("⚠️ Features vazias para predição por regime")
                return None
            
            if not regime_info:
                self.logger.warning("⚠️ Informações de regime ausentes - usando predição padrão")
                return self.predict(features)
            
            # Verificar se temos modelos carregados  
            if not hasattr(self.model_manager, 'models') or not self.model_manager.models:
                self.logger.error("❌ NENHUM MODELO DISPONÍVEL - Predição por regime impossível")
                return None
            
            # Usar ModelManager com informações de regime
            try:
                # Tentar usar ensemble com regime se disponível
                if hasattr(self.model_manager, 'predict_with_ensemble'):
                    model_result = self.model_manager.predict_with_ensemble(
                        features.values, 
                        market_regime=regime_info.get('regime', 'undefined')
                    )
                else:
                    model_result = self.model_manager.predict(features)
                
                if model_result is not None:
                    # Converter resultado e ajustar para regime
                    prediction = self._convert_model_result_with_regime(model_result, regime_info)
                    self.logger.info(f"🎯 Predição REAL por regime {regime_info['regime']}: "
                                   f"dir={prediction['direction']:.3f}, conf={prediction['confidence']:.3f}")
                    return prediction
                else:
                    self.logger.error("❌ ModelManager retornou None para regime - Predição falhou")
                    return None
                    
            except Exception as e:
                self.logger.error(f"❌ ERRO CRÍTICO no ModelManager com regime: {e}")
                return None
            
        except Exception as e:
            self.logger.error(f"❌ Erro em predict_by_regime: {e}")
            return None
            
    
    def _convert_model_result(self, model_result: Any) -> Dict[str, Any]:
        """
        Converte resultado do ModelManager para formato padronizado
        
        Args:
            model_result: Resultado do ModelManager.predict()
            
        Returns:
            Dict padronizado para o sistema
        """
        try:
            # Se model_result já é um dict padronizado
            if isinstance(model_result, dict) and 'direction' in model_result:
                return model_result
            
            # Se é resultado de ensemble
            if isinstance(model_result, dict) and 'ensemble_prediction' in model_result:
                ensemble_pred = model_result['ensemble_prediction']
                return {
                    'direction': float(ensemble_pred.get('direction', 0.0)),
                    'magnitude': float(ensemble_pred.get('magnitude', 0.001)),
                    'confidence': float(ensemble_pred.get('confidence', 0.5)),
                    'regime': 'unknown',
                    'timestamp': datetime.now().isoformat(),
                    'model_used': 'ensemble',
                    'model_details': model_result
                }
            
            # Se é array numpy (predições brutas)
            if hasattr(model_result, 'shape'):
                # Assumir formato [direction, magnitude, confidence]
                if len(model_result) >= 3:
                    return {
                        'direction': float(model_result[0]),
                        'magnitude': float(abs(model_result[1])),
                        'confidence': float(model_result[2]),
                        'regime': 'unknown',
                        'timestamp': datetime.now().isoformat(),
                        'model_used': 'direct_array'
                    }
            
            # Fallback: interpretar como predição simples
            self.logger.warning(f"Formato de resultado não reconhecido: {type(model_result)}")
            
            # Tentar extrair valores
            if hasattr(model_result, '__getitem__'):
                try:
                    return {
                        'direction': float(model_result[0]) if len(model_result) > 0 else 0.0,
                        'magnitude': float(abs(model_result[1])) if len(model_result) > 1 else 0.001,
                        'confidence': float(model_result[2]) if len(model_result) > 2 else 0.5,
                        'regime': 'unknown',
                        'timestamp': datetime.now().isoformat(),
                        'model_used': 'interpreted'
                    }
                except Exception:
                    pass
            
            # Se não conseguiu interpretar, retornar None
            self.logger.error("❌ Não foi possível interpretar resultado do modelo")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Erro convertendo resultado do modelo: {e}")
            return None
    
    def _convert_model_result_with_regime(self, model_result: Any, regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converte resultado do ModelManager e ajusta com informações de regime
        
        Args:
            model_result: Resultado do ModelManager
            regime_info: Informações do regime detectado
            
        Returns:
            Dict padronizado com ajustes de regime
        """
        try:
            # Converter resultado base
            prediction = self._convert_model_result(model_result)
            
            # Ajustar com informações de regime
            prediction.update({
                'regime': regime_info.get('regime', 'unknown'),
                'regime_confidence': regime_info.get('confidence', 0.0),
                'regime_direction': regime_info.get('direction', 0),
                'regime_info': regime_info,
                'model_used': f"{prediction.get('model_used', 'unknown')}_regime_adjusted"
            })
            
            # Aplicar thresholds específicos do regime se necessário
            thresholds = regime_info.get('thresholds', {})
            if thresholds:
                # Ajustar confiança baseado nos thresholds do regime
                regime_confidence_threshold = thresholds.get('confidence', 0.5)
                prediction['confidence'] = max(prediction['confidence'], regime_confidence_threshold * 0.8)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"❌ Erro convertendo resultado com regime: {e}")
            return None
            
    def batch_predict(self, features_list: List[pd.DataFrame]) -> List[Dict[str, Any]]:
        """Predições em lote"""
        results = []
        for features in features_list:
            result = self.predict(features)
            if result:
                results.append(result)
        return results
        
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações dos modelos carregados"""
        if hasattr(self.model_manager, 'models'):
            return {
                'models_count': len(self.model_manager.models),
                'models_loaded': list(self.model_manager.models.keys())
            }
        return {'models_count': 0, 'models_loaded': []}
