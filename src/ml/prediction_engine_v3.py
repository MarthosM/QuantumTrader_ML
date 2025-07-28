"""
PredictionEngineV3 - Motor de predição com modelos V3 e detecção de regime
Baseado no DEVELOPER_GUIDE_V3_REFACTORING.md
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Any
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Imports internos
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Usar implementação simplificada do RegimeAnalyzer para evitar dependências
class RegimeAnalyzer:
    """Analisador de regime simplificado"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_market(self, candles: pd.DataFrame) -> Dict:
        """Analisa regime de mercado"""
        
        if candles.empty or len(candles) < 50:
            return {'regime': 'undefined', 'confidence': 0.5}
        
        # Calcular indicadores básicos
        close = candles['close']
        
        # EMAs
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        
        # ADX simplificado (usar volatilidade como proxy)
        returns = close.pct_change()
        volatility = returns.rolling(14).std()
        adx_proxy = (volatility.iloc[-1] / volatility.mean()) * 25  # Normalizar para escala ADX
        
        # Determinar regime
        last_ema9 = ema9.iloc[-1]
        last_ema20 = ema20.iloc[-1]
        last_ema50 = ema50.iloc[-1]
        
        if adx_proxy > 25:
            if last_ema9 > last_ema20 > last_ema50:
                regime = 'trend_up'
                confidence = min(0.8, 0.5 + (adx_proxy - 25) / 50)
            elif last_ema9 < last_ema20 < last_ema50:
                regime = 'trend_down'
                confidence = min(0.8, 0.5 + (adx_proxy - 25) / 50)
            else:
                regime = 'undefined'
                confidence = 0.5
        else:
            regime = 'range'
            confidence = 0.6
            
        return {
            'regime': regime,
            'confidence': confidence,
            'adx': adx_proxy,
            'ema_alignment': (last_ema9 > last_ema20 > last_ema50)
        }


class PredictionEngineV3:
    """
    Motor de predição ML com modelos V3
    
    Features:
    - Carrega e gerencia modelos treinados por regime
    - Detecta regime atual do mercado
    - Seleciona modelo apropriado por regime
    - Gera predições com confidence scores
    - Valida qualidade das features antes de prever
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o motor de predição
        
        Args:
            config: Configurações opcionais
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Paths
        self.model_path = self.config.get('model_path', '../../models/')
        
        # Modelos carregados
        self.models = {}  # {regime: {algorithm: model}}
        self.model_metadata = {}
        self.feature_scaler = None
        
        # Analisador de regime
        self.regime_analyzer = RegimeAnalyzer()
        
        # Features requeridas
        self.required_features = set()
        
        # Configurações de predição
        self.confidence_thresholds = self.config.get('confidence_thresholds', {
            'trend_up': 0.60,
            'trend_down': 0.60,
            'range': 0.55,
            'undefined': 0.80
        })
        
        # Cache de predições
        self.last_prediction = None
        self.last_prediction_time = None
        
        # Métricas
        self.metrics = {
            'predictions_made': 0,
            'regime_detections': {'trend_up': 0, 'trend_down': 0, 'range': 0, 'undefined': 0},
            'model_hits': {},
            'average_confidence': 0,
            'errors': 0
        }
        
    def load_models(self, model_metadata_path: Optional[str] = None) -> bool:
        """
        Carrega modelos V3 treinados
        
        Args:
            model_metadata_path: Caminho para metadata dos modelos
            
        Returns:
            bool: True se carregou com sucesso
        """
        try:
            # Se não fornecido, procurar mais recente
            if not model_metadata_path:
                metadata_files = [f for f in os.listdir(self.model_path) 
                                if f.startswith('models_metadata_v3') and f.endswith('.json')]
                
                if not metadata_files:
                    self.logger.error("Nenhum metadata de modelos V3 encontrado")
                    return False
                
                # Usar o mais recente
                model_metadata_path = os.path.join(self.model_path, sorted(metadata_files)[-1])
            
            # Carregar metadata
            with open(model_metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
            
            self.logger.info(f"Carregando modelos V3 de {model_metadata_path}")
            
            # Carregar cada modelo
            for regime, regime_models in self.model_metadata['models'].items():
                self.models[regime] = {}
                
                for algorithm, model_file in regime_models.items():
                    model_path = os.path.join(self.model_path, model_file)
                    
                    if os.path.exists(model_path):
                        model = joblib.load(model_path)
                        self.models[regime][algorithm] = model
                        self.logger.info(f"  Carregado: {regime}/{algorithm}")
                        
                        # Inicializar métricas
                        if algorithm not in self.metrics['model_hits']:
                            self.metrics['model_hits'][algorithm] = 0
            
            # Carregar feature scaler
            scaler_path = os.path.join(self.model_path, 'feature_scaler_v3.pkl')
            if os.path.exists(scaler_path):
                self.feature_scaler = joblib.load(scaler_path)
                self.logger.info("Feature scaler carregado")
            
            # Identificar features requeridas
            self._identify_required_features()
            
            self.logger.info(f"Modelos carregados: {len(self.models)} regimes")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro carregando modelos: {e}")
            return False
            
    def _identify_required_features(self):
        """Identifica features requeridas pelos modelos"""
        
        self.required_features = set()
        
        # Analisar cada modelo
        for regime_models in self.models.values():
            for model in regime_models.values():
                # Tentar obter features do modelo
                if hasattr(model, 'feature_names_in_'):
                    self.required_features.update(model.feature_names_in_)
                elif hasattr(model, 'feature_names_'):
                    self.required_features.update(model.feature_names_)
                elif hasattr(model, 'get_booster'):
                    # XGBoost
                    try:
                        booster = model.get_booster()
                        features = booster.feature_names
                        if features:
                            self.required_features.update(features)
                    except:
                        pass
        
        self.logger.info(f"Features requeridas identificadas: {len(self.required_features)}")
        
    def predict(self, features: pd.DataFrame, candles: pd.DataFrame) -> Optional[Dict]:
        """
        Gera predição baseada nas features e regime atual
        
        Args:
            features: DataFrame com features ML
            candles: DataFrame com dados OHLCV para análise de regime
            
        Returns:
            Dict com predição ou None se não puder prever
        """
        try:
            # Validar entrada
            if features.empty or candles.empty:
                self.logger.warning("Dados vazios fornecidos para predição")
                return None
            
            # Detectar regime atual
            regime_info = self.regime_analyzer.analyze_market(candles)
            
            if not regime_info:
                self.logger.error("Falha na detecção de regime")
                return None
            
            current_regime = regime_info['regime']
            regime_confidence = regime_info['confidence']
            
            # Atualizar métricas
            self.metrics['regime_detections'][current_regime] += 1
            
            self.logger.info(f"Regime detectado: {current_regime} (confiança: {regime_confidence:.2f})")
            
            # Verificar se temos modelos para o regime
            if current_regime not in self.models or not self.models[current_regime]:
                self.logger.warning(f"Sem modelos para regime {current_regime}")
                return None
            
            # Preparar features
            prepared_features = self._prepare_features(features)
            
            if prepared_features is None:
                return None
            
            # Gerar predições com cada modelo do regime
            predictions = {}
            confidences = []
            
            for algorithm, model in self.models[current_regime].items():
                try:
                    # Predição
                    pred = model.predict(prepared_features.tail(1))[0]
                    
                    # Probabilidade/confiança
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(prepared_features.tail(1))[0]
                        conf = max(proba)
                    else:
                        conf = 0.5  # Default para modelos sem probabilidade
                    
                    predictions[algorithm] = {
                        'direction': int(pred),
                        'confidence': float(conf)
                    }
                    confidences.append(conf)
                    
                    # Métricas
                    self.metrics['model_hits'][algorithm] += 1
                    
                except Exception as e:
                    self.logger.error(f"Erro na predição com {algorithm}: {e}")
            
            if not predictions:
                self.logger.error("Nenhuma predição gerada")
                return None
            
            # Combinar predições (ensemble voting)
            final_prediction = self._combine_predictions(predictions, current_regime)
            
            # Adicionar informações do regime
            final_prediction['regime'] = current_regime
            final_prediction['regime_confidence'] = regime_confidence
            final_prediction['regime_details'] = regime_info
            
            # Verificar threshold de confiança
            min_confidence = self.confidence_thresholds.get(current_regime, 0.5)
            if final_prediction['confidence'] < min_confidence:
                self.logger.info(f"Confiança abaixo do threshold ({final_prediction['confidence']:.2f} < {min_confidence})")
                final_prediction['action'] = 'hold'
                final_prediction['reason'] = 'low_confidence'
            
            # Cache e métricas
            self.last_prediction = final_prediction
            self.last_prediction_time = datetime.now()
            self.metrics['predictions_made'] += 1
            
            # Atualizar confiança média
            if self.metrics['predictions_made'] > 0:
                current_avg = self.metrics['average_confidence']
                new_avg = (current_avg * (self.metrics['predictions_made'] - 1) + 
                          final_prediction['confidence']) / self.metrics['predictions_made']
                self.metrics['average_confidence'] = new_avg
            
            return final_prediction
            
        except Exception as e:
            self.logger.error(f"Erro gerando predição: {e}")
            self.metrics['errors'] += 1
            return None
            
    def _prepare_features(self, features: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepara features para predição"""
        
        try:
            # Verificar features requeridas
            missing_features = self.required_features - set(features.columns)
            if missing_features:
                self.logger.warning(f"Features faltando: {list(missing_features)[:10]}")
                # Tentar continuar com features disponíveis
            
            # Selecionar apenas features conhecidas
            available_features = list(self.required_features.intersection(features.columns))
            
            if not available_features:
                self.logger.error("Nenhuma feature requerida disponível")
                return None
            
            prepared = features[available_features].copy()
            
            # Aplicar scaler se disponível
            if self.feature_scaler:
                try:
                    # Garantir que temos as mesmas features do scaler
                    scaler_features = self.feature_scaler.feature_names_in_
                    common_features = list(set(scaler_features).intersection(prepared.columns))
                    
                    if common_features:
                        prepared[common_features] = self.feature_scaler.transform(prepared[common_features])
                except Exception as e:
                    self.logger.warning(f"Erro aplicando scaler: {e}")
            
            # Preencher NaN
            prepared = prepared.fillna(0)
            
            # Verificar infinitos
            prepared = prepared.replace([np.inf, -np.inf], 0)
            
            return prepared
            
        except Exception as e:
            self.logger.error(f"Erro preparando features: {e}")
            return None
            
    def _combine_predictions(self, predictions: Dict, regime: str) -> Dict:
        """Combina predições de múltiplos modelos"""
        
        # Coletar votos e confidências
        directions = []
        confidences = []
        
        for algo, pred in predictions.items():
            directions.append(pred['direction'])
            confidences.append(pred['confidence'])
        
        # Votação majoritária ponderada por confiança
        if not directions:
            return {'direction': 0, 'confidence': 0, 'action': 'hold'}
        
        # Calcular direção consensual
        weighted_sum = sum(d * c for d, c in zip(directions, confidences))
        total_confidence = sum(confidences)
        
        if total_confidence > 0:
            consensus_direction = 1 if weighted_sum / total_confidence > 0 else -1
        else:
            consensus_direction = 0
        
        # Confiança média
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Determinar ação
        if consensus_direction > 0:
            action = 'buy'
        elif consensus_direction < 0:
            action = 'sell'
        else:
            action = 'hold'
        
        # Adicionar detalhes do ensemble
        ensemble_details = {
            'algorithms_used': list(predictions.keys()),
            'individual_predictions': predictions,
            'consensus_method': 'weighted_voting'
        }
        
        return {
            'direction': consensus_direction,
            'confidence': float(avg_confidence),
            'action': action,
            'ensemble_details': ensemble_details,
            'timestamp': datetime.now().isoformat()
        }
        
    def get_metrics(self) -> Dict:
        """Retorna métricas do motor de predição"""
        return self.metrics.copy()
        
    def get_required_features(self) -> List[str]:
        """Retorna lista de features requeridas"""
        return sorted(list(self.required_features))


def main():
    """Teste do PredictionEngineV3"""
    
    print("="*60)
    print("TESTE DO PREDICTION ENGINE V3")
    print("="*60)
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Criar engine
    engine = PredictionEngineV3()
    
    # Carregar modelos
    if not engine.load_models():
        print("[ERRO] Falha ao carregar modelos")
        return
    
    print(f"\nModelos carregados para regimes: {list(engine.models.keys())}")
    print(f"Features requeridas: {len(engine.required_features)}")
    
    # Simular dados para teste
    dates = pd.date_range('2025-01-27 16:00', '2025-01-27 17:00', freq='1min')
    
    # Candles simulados
    candles = pd.DataFrame({
        'open': 5900 + np.random.randn(len(dates)).cumsum() * 0.5,
        'high': 5905 + np.random.randn(len(dates)).cumsum() * 0.5,
        'low': 5895 + np.random.randn(len(dates)).cumsum() * 0.5,
        'close': 5900 + np.random.randn(len(dates)).cumsum() * 0.5,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    candles['high'] = candles[['open', 'close', 'high']].max(axis=1)
    candles['low'] = candles[['open', 'close', 'low']].min(axis=1)
    
    # Features simuladas (usar nomes V3)
    features = pd.DataFrame(index=dates)
    
    # Adicionar features básicas
    for i in range(1, 118):  # MLFeaturesV3 gera ~118 features
        features[f'v3_feature_{i}'] = np.random.randn(len(dates))
    
    # Adicionar features conhecidas
    features['v3_momentum_5'] = candles['close'].pct_change(5)
    features['v3_volatility_cc_10'] = candles['close'].pct_change().rolling(10).std()
    features['v3_volume_ratio_5'] = candles['volume'] / candles['volume'].rolling(5).mean()
    features['v3_adx'] = 25 + np.random.randn(len(dates)) * 5  # ADX simulado
    
    print("\nGerando predição...")
    
    # Gerar predição
    prediction = engine.predict(features, candles)
    
    if prediction:
        print(f"\nPredição gerada:")
        print(f"  Regime: {prediction['regime']}")
        print(f"  Direção: {prediction['direction']}")
        print(f"  Confiança: {prediction['confidence']:.2f}")
        print(f"  Ação: {prediction['action']}")
        
        if 'ensemble_details' in prediction:
            print(f"  Algoritmos usados: {prediction['ensemble_details']['algorithms_used']}")
    else:
        print("\n[ERRO] Não foi possível gerar predição")
    
    # Métricas
    metrics = engine.get_metrics()
    print(f"\nMétricas:")
    print(f"  Predições realizadas: {metrics['predictions_made']}")
    print(f"  Confiança média: {metrics['average_confidence']:.2f}")
    print(f"  Detecções por regime: {metrics['regime_detections']}")
    print(f"  Erros: {metrics['errors']}")


if __name__ == "__main__":
    main()