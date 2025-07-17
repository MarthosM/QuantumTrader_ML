"""
Model Manager - Gerencia carregamento e acesso aos modelos ML
Baseado em model_loader.py e ml_integration.py
"""

import logging
import os
from typing import Dict, List, Optional, Any
import joblib
import json
import pandas as pd

class ModelManager:
    """Gerencia carregamento e acesso aos modelos ML"""
    
    def __init__(self, models_dir: str = 'saved_models'):
        self.models_dir = models_dir if models_dir != 'saved_models' else r"C:\Users\marth\OneDrive\Programacao\Python\Projetos\ML_Tradingv2.0\src\models\models_regime3"
        self.models: Dict[str, Any] = {}
        self.model_features: Dict[str, List[str]] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.logger = logging.getLogger('ModelManager')
        
        # Configurações
        self.supported_extensions = ['.pkl', '.joblib']
        
    def load_models(self) -> bool:
        """
        Carrega todos os modelos disponíveis
        Baseado em model_loader.py load_saved_models()
        """
        try:
            if not os.path.exists(self.models_dir):
                self.logger.error(f"Diretório de modelos não encontrado: {self.models_dir}")
                return False
                
            # Listar arquivos de modelo
            model_files = []
            for file in os.listdir(self.models_dir):
                if any(file.endswith(ext) for ext in self.supported_extensions):
                    model_files.append(file)
                    
            if not model_files:
                self.logger.warning(f"Nenhum modelo encontrado em {self.models_dir}")
                return False
                
            self.logger.info(f"Encontrados {len(model_files)} arquivos de modelo")
            
            # Carregar cada modelo
            for model_file in model_files:
                model_name = os.path.splitext(model_file)[0]
                model_path = os.path.join(self.models_dir, model_file)
                
                try:
                    # Carregar modelo
                    self.logger.info(f"Carregando modelo: {model_name}")
                    model = joblib.load(model_path)
                    self.models[model_name] = model
                    
                    # Descobrir features do modelo
                    features = self._extract_features(model, model_name)
                    self.model_features[model_name] = features
                    self.logger.info(f"Modelo {model_name}: {len(features)} features")
                    
                    # Carregar metadados se existirem
                    self._load_model_metadata(model_name)
                    
                except Exception as e:
                    self.logger.error(f"Erro carregando modelo {model_name}: {e}")
                    continue
            
            self.logger.info(f"Carregados {len(self.models)} modelos com sucesso")
            
            # Log resumo dos modelos
            self._log_models_summary()
            
            return len(self.models) > 0
            
        except Exception as e:
            self.logger.error(f"Erro carregando modelos: {e}", exc_info=True)
            return False
    
    def _extract_features(self, model: Any, model_name: str) -> List[str]:
        """
        Extrai lista de features de um modelo
        Baseado em ml_integration.py _discover_model_features()
        """
        features = []
        
        try:
            # Tentar diferentes métodos dependendo do tipo de modelo
            
            # XGBoost
            if hasattr(model, 'get_booster'):
                booster = model.get_booster()
                if hasattr(booster, 'feature_names'):
                    features = booster.feature_names
                elif hasattr(booster, 'get_score'):
                    features = list(booster.get_score().keys())
                    
            # LightGBM
            elif hasattr(model, 'booster_'):
                if hasattr(model.booster_, 'feature_name'):
                    features = model.booster_.feature_name()
                elif hasattr(model, 'feature_name_'):
                    features = model.feature_name_
                    
            # Scikit-learn
            elif hasattr(model, 'feature_names_in_'):
                features = list(model.feature_names_in_)
            elif hasattr(model, 'get_feature_names_out'):
                features = list(model.get_feature_names_out())
                
            # Fallback: tentar ler de arquivo de features
            if not features:
                features_file = os.path.join(self.models_dir, f"{model_name}_features.json")
                if os.path.exists(features_file):
                    with open(features_file, 'r') as f:
                        features = json.load(f)
                        
        except Exception as e:
            self.logger.warning(f"Erro extraindo features do modelo {model_name}: {e}")
            
        # Se ainda não temos features, usar conjunto padrão
        if not features:
            self.logger.warning(f"Usando features padrão para {model_name}")
            features = self._get_default_features()
            
        return features
    
    def _get_default_features(self) -> List[str]:
        """Retorna conjunto padrão de features"""
        return [
            # OHLCV básicos
            'open', 'high', 'low', 'close', 'volume',
            
            # EMAs principais
            'ema_5', 'ema_20', 'ema_50', 'ema_200',
            
            # Indicadores essenciais
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper_20', 'bb_lower_20', 'bb_middle_20',
            'atr', 'adx',
            
            # Features de momentum
            'momentum_5', 'momentum_10', 'momentum_20',
            'momentum_pct_5', 'momentum_pct_10', 'momentum_pct_20',
            
            # Features de volatilidade
            'volatility_5', 'volatility_10', 'volatility_20', 'volatility_50',
            
            # Features de retorno
            'return_5', 'return_10', 'return_20', 'return_50'
        ]
    
    def _load_model_metadata(self, model_name: str):
        """Carrega metadados do modelo se existirem"""
        metadata_file = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    self.model_metadata[model_name] = json.load(f)
                    self.logger.info(f"Metadados carregados para {model_name}")
            except Exception as e:
                self.logger.warning(f"Erro carregando metadados de {model_name}: {e}")
    
    def _log_models_summary(self):
        """Log resumo dos modelos carregados"""
        self.logger.info("=== RESUMO DOS MODELOS ===")
        
        for model_name, model in self.models.items():
            features = self.model_features.get(model_name, [])
            metadata = self.model_metadata.get(model_name, {})
            
            self.logger.info(f"\nModelo: {model_name}")
            self.logger.info(f"  Tipo: {type(model).__name__}")
            self.logger.info(f"  Features: {len(features)}")
            self.logger.info(f"  Sample features: {features[:5]}...")
            
            if metadata:
                self.logger.info(f"  Accuracy: {metadata.get('accuracy', 'N/A')}")
                self.logger.info(f"  Training date: {metadata.get('training_date', 'N/A')}")
                
        self.logger.info("========================")
    
    def get_all_required_features(self) -> List[str]:
        """Retorna todas as features necessárias para todos os modelos"""
        all_features = set()
        
        for features in self.model_features.values():
            all_features.update(features)
            
        return list(all_features)
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Retorna um modelo específico"""
        return self.models.get(model_name)
    
    def get_model_features(self, model_name: str) -> List[str]:
        """Retorna features de um modelo específico"""
        return self.model_features.get(model_name, [])
    
    def predict(self, model_name: str, features_df: pd.DataFrame) -> Optional[Any]:
        """
        Executa predição com um modelo específico
        """
        if model_name not in self.models:
            self.logger.error(f"Modelo {model_name} não encontrado")
            return None
            
        try:
            model = self.models[model_name]
            model_features = self.model_features[model_name]
            
            # Preparar features
            X = features_df[model_features].fillna(0)
            
            # Executar predição
            prediction = model.predict(X)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Erro na predição com {model_name}: {e}")
            return None
    
    def ensemble_predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Executa predição com ensemble de todos os modelos"""
        predictions = {}
        
        for model_name in self.models:
            pred = self.predict(model_name, features_df)
            if pred is not None:
                predictions[model_name] = pred
                
        return predictions