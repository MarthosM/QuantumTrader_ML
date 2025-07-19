"""
Model Manager - Gerencia carregamento e acesso aos modelos ML
Versão 2.0 - Com Ensemble Multi-Modal e Otimização Avançada
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Type checking imports para evitar problemas de runtime
if TYPE_CHECKING:
    try:
        import tensorflow as tf
        from tensorflow import keras  # type: ignore
    except ImportError:
        pass

# Imports para modelos
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR

# Imports para deep learning
try:
    import tensorflow as tf
    
    # Verificar se TensorFlow está funcionando
    tf_version = tf.__version__
    
    # Importações compatíveis com diferentes versões do TensorFlow
    try:
        # TensorFlow 2.x - tentar tf.keras primeiro
        if hasattr(tf, 'keras'):
            keras = tf.keras # type: ignore
        else:
            # Fallback para Keras standalone
            import keras # type: ignore
    except:
        # Último fallback para Keras standalone
        import keras # type: ignore
    
    # Verificar se keras está disponível
    if keras is None:
        raise ImportError("Keras não disponível")
    
    # Atribuições com type hints para evitar erros do VS Code
    Sequential = keras.Sequential # type: ignore
    Model = keras.Model # type: ignore
    Input = keras.Input # type: ignore
    
    # Layers
    LSTM = keras.layers.LSTM # type: ignore
    Dense = keras.layers.Dense # type: ignore
    Dropout = keras.layers.Dropout # type: ignore
    BatchNormalization = keras.layers.BatchNormalization # type: ignore
    MultiHeadAttention = keras.layers.MultiHeadAttention # type: ignore
    GlobalAveragePooling1D = keras.layers.GlobalAveragePooling1D # type: ignore
    LayerNormalization = keras.layers.LayerNormalization # type: ignore
    Lambda = keras.layers.Lambda # type: ignore
    
    # Optimizers e callbacks
    Adam = keras.optimizers.Adam # type: ignore
    EarlyStopping = keras.callbacks.EarlyStopping # type: ignore
    ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau # type: ignore
    
    # Utils
    to_categorical = keras.utils.to_categorical # type: ignore
    
    TF_AVAILABLE = True
    
except (ImportError, AttributeError) as e:
    TF_AVAILABLE = False
    logging.warning(f"TensorFlow/Keras não disponível - modelos deep learning desabilitados: {e}")
    
    # Definir classes dummy para evitar erros de tipo
    class DummyTensorFlowClass:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return self
        def __getattr__(self, name):
            return self
    
    # Usar a classe dummy para todas as variáveis
    keras = DummyTensorFlowClass() # type: ignore
    Sequential = DummyTensorFlowClass # type: ignore
    Model = DummyTensorFlowClass # type: ignore
    Input = DummyTensorFlowClass # type: ignore
    LSTM = DummyTensorFlowClass # type: ignore
    Dense = DummyTensorFlowClass # type: ignore
    Dropout = DummyTensorFlowClass # type: ignore
    BatchNormalization = DummyTensorFlowClass # type: ignore
    MultiHeadAttention = DummyTensorFlowClass # type: ignore
    GlobalAveragePooling1D = DummyTensorFlowClass # type: ignore
    LayerNormalization = DummyTensorFlowClass # type: ignore
    Lambda = DummyTensorFlowClass # type: ignore
    Adam = DummyTensorFlowClass # type: ignore
    EarlyStopping = DummyTensorFlowClass # type: ignore
    ReduceLROnPlateau = DummyTensorFlowClass # type: ignore
    to_categorical = DummyTensorFlowClass() # type: ignore

# Import para otimização
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna não disponível - otimização de hiperparâmetros desabilitada")

# Import para datetime
from datetime import datetime


class ModelRetrainScheduler:
    """Agendador para retreinamento automático de modelos"""
    
    def __init__(self, model_manager, config):
        self.model_manager = model_manager
        self.config = config
        self.logger = logging.getLogger('ModelRetrainScheduler')
        self.is_running = False
        
    def start(self):
        """Inicia o agendador"""
        self.is_running = True
        self.logger.info("Agendador de retreinamento iniciado")
        
    def stop(self):
        """Para o agendador"""
        self.is_running = False
        self.logger.info("Agendador de retreinamento parado")


class MultiModalEnsemble:
    """Ensemble que combina diferentes tipos de modelos ML para day trading"""
    
    def __init__(self, logger):
        self.logger = logger
        self.models = {}
        self.model_weights = {}
        self.regime_weights = self._initialize_regime_weights()
        self.performance_history = {}
        
        # Configurações
        self.confidence_threshold = 0.6
        self.min_agreement_threshold = 0.5
        
        # Cache de predições
        self.prediction_cache = {}
        self.cache_size = 100
        
    def _initialize_regime_weights(self):
        """Inicializa pesos dos modelos por regime de mercado"""
        return {
            'high_volatility': {
                'xgboost_fast': 0.3,
                'lstm_intraday': 0.2,
                'transformer_attention': 0.2,
                'rf_stable': 0.1,
                'lgb_balanced': 0.1,
                'traditional_models': 0.1
            },
            'low_volatility': {
                'xgboost_fast': 0.2,
                'lstm_intraday': 0.25,
                'transformer_attention': 0.25,
                'rf_stable': 0.15,
                'lgb_balanced': 0.1,
                'traditional_models': 0.05
            },
            'trending': {
                'xgboost_fast': 0.2,
                'lstm_intraday': 0.3,
                'transformer_attention': 0.3,
                'rf_stable': 0.1,
                'lgb_balanced': 0.05,
                'traditional_models': 0.05
            },
            'ranging': {
                'xgboost_fast': 0.25,
                'lstm_intraday': 0.15,
                'transformer_attention': 0.15,
                'rf_stable': 0.25,
                'lgb_balanced': 0.15,
                'traditional_models': 0.05
            },
            'undefined': {
                'xgboost_fast': 0.2,
                'lstm_intraday': 0.2,
                'transformer_attention': 0.2,
                'rf_stable': 0.15,
                'lgb_balanced': 0.15,
                'traditional_models': 0.1
            }
        }
    
    def create_fast_xgboost(self, n_features):
        """XGBoost otimizado para velocidade em day trade"""
        
        return xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            
            # Otimizações para alta frequência
            n_jobs=4,
            tree_method='hist',
            predictor='cpu_predictor',
            
            # Regularização
            reg_alpha=0.1,
            reg_lambda=0.1,
            
            # Classes
            objective='multi:softprob',
            num_class=3,
            
            # Early stopping será usado no treino
            eval_metric='mlogloss'
        )
    
    def create_lgb_balanced(self, n_features):
        """LightGBM balanceado para velocidade e accuracy"""
        
        return lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            num_leaves=31,
            
            # Otimizações
            n_jobs=4,
            boosting_type='gbdt',
            
            # Regularização
            reg_alpha=0.1,
            reg_lambda=0.1,
            
            # Classes
            objective='multiclass',
            num_class=3,
            
            # Outros
            random_state=42,
            verbosity=-1
        )
    
    def create_intraday_lstm(self, n_features, sequence_length=60):
        """LSTM especializado para padrões intraday"""
        
        if not TF_AVAILABLE:
            self.logger.warning("TensorFlow não disponível - LSTM não criado")
            return None
        
        # Type: ignore para evitar erros de tipo durante importações dinâmicas
        model = Sequential([ # type: ignore
            LSTM(64, return_sequences=True, dropout=0.2, 
                 input_shape=(sequence_length, n_features)), # type: ignore
            BatchNormalization(), # type: ignore
            
            LSTM(32, return_sequences=False, dropout=0.2), # type: ignore
            BatchNormalization(), # type: ignore
            
            Dense(16, activation='relu'), # type: ignore
            Dropout(0.3), # type: ignore
            Dense(8, activation='relu'), # type: ignore
            Dense(3, activation='softmax') # type: ignore
        ])
        
        optimizer = Adam(learning_rate=0.001) # type: ignore
        
        model.compile( # type: ignore[misc]
            optimizer=optimizer, # type: ignore[arg-type]
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_attention_transformer(self, n_features, sequence_length=60):
        """Transformer com atenção para capturar dependências temporais"""
        
        if not TF_AVAILABLE:
            self.logger.warning("TensorFlow não disponível - Transformer não criado")
            return None
        
        inputs = Input(shape=(sequence_length, n_features)) # type: ignore
        
        # Projection
        x = Dense(64)(inputs) # type: ignore
        
        # Attention blocks
        for _ in range(2):
            attn_output = MultiHeadAttention( # type: ignore
                num_heads=4, 
                key_dim=64,
                dropout=0.1
            )(x, x)
            
            x = LayerNormalization(epsilon=1e-6)(x + attn_output) # type: ignore
            
            ff_output = Sequential([ # type: ignore
                Dense(256, activation='relu'), # type: ignore
                Dropout(0.1), # type: ignore
                Dense(64) # type: ignore
            ])(x)
            
            x = LayerNormalization(epsilon=1e-6)(x + ff_output) # type: ignore
        
        x = GlobalAveragePooling1D()(x) # type: ignore
        x = Dense(32, activation='relu')(x) # type: ignore
        x = Dropout(0.2)(x) # type: ignore
        outputs = Dense(3, activation='softmax')(x) # type: ignore
        
        model = Model(inputs, outputs) # type: ignore
        
        model.compile( # type: ignore[misc]
            optimizer=Adam(learning_rate=0.0005), # type: ignore[arg-type]
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_stable_random_forest(self, n_features):
        """Random Forest para estabilidade em condições voláteis"""
        
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
    
    def initialize_models(self, n_features, sequence_length=60):
        """Inicializa todos os modelos do ensemble"""
        
        self.logger.info("Inicializando ensemble multi-modal...")
        
        try:
            # XGBoost rápido
            self.models['xgboost_fast'] = self.create_fast_xgboost(n_features)
            self.logger.info("✓ XGBoost rápido criado")
            
            # LightGBM balanceado
            self.models['lgb_balanced'] = self.create_lgb_balanced(n_features)
            self.logger.info("✓ LightGBM balanceado criado")
            
            # Random Forest estável
            self.models['rf_stable'] = self.create_stable_random_forest(n_features)
            self.logger.info("✓ Random Forest estável criado")
            
            # Deep Learning se disponível
            if TF_AVAILABLE:
                # LSTM
                lstm_model = self.create_intraday_lstm(n_features, sequence_length)
                if lstm_model:
                    self.models['lstm_intraday'] = lstm_model
                    self.logger.info("✓ LSTM intraday criado")
                
                # Transformer
                transformer_model = self.create_attention_transformer(n_features, sequence_length)
                if transformer_model:
                    self.models['transformer_attention'] = transformer_model
                    self.logger.info("✓ Transformer attention criado")
            
            # Inicializar pesos
            self.model_weights = {
                name: 1.0 / len(self.models) 
                for name in self.models.keys()
            }
            
            self.logger.info(f"Ensemble inicializado com {len(self.models)} modelos")
            
        except Exception as e:
            self.logger.error(f"Erro inicializando modelos: {e}")
            raise
    
    def prepare_sequential_data(self, X, sequence_length=60):
        """Prepara dados para modelos sequenciais"""
        
        if len(X) < sequence_length:
            pad_length = sequence_length - len(X)
            padding = np.zeros((pad_length, X.shape[1]))
            X_padded = np.vstack([padding, X])
            return X_padded.reshape(1, sequence_length, -1)
        
        sequences = []
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i + sequence_length])
        
        return np.array(sequences)
    
    def predict_with_ensemble(self, X, market_regime='undefined', 
                            return_all=False):
        """Predição com ensemble dinâmico"""
        
        cache_key = f"{X.shape}_{market_regime}_{hash(X.tobytes()) % 1000000}"
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        predictions = {}
        confidences = {}
        
        # Preparar dados sequenciais se necessário
        X_seq = None
        if any(name in self.models for name in ['lstm_intraday', 'transformer_attention']):
            X_seq = self.prepare_sequential_data(X)
        
        # Coletar predições
        for name, model in self.models.items():
            try:
                if name in ['lstm_intraday', 'transformer_attention'] and X_seq is not None:
                    pred_proba = model.predict(X_seq, verbose=0)
                    if len(pred_proba.shape) > 2:
                        pred_proba = pred_proba[-1:]
                else:
                    X_flat = X[-1:] if len(X.shape) == 2 else X
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X_flat)
                    else:
                        pred = model.predict(X_flat)
                        pred_proba = np.zeros((1, 3))
                        pred_proba[0, int(pred[0])] = 1.0
                
                predictions[name] = pred_proba[0]
                confidences[name] = np.max(pred_proba[0])
                
            except Exception as e:
                self.logger.error(f"Erro na predição {name}: {e}")
                predictions[name] = np.array([0.33, 0.34, 0.33])
                confidences[name] = 0.0
        
        # Aplicar pesos por regime
        regime_weights = self.regime_weights.get(
            market_regime, 
            self.regime_weights['undefined']
        )
        
        # Combinar predições
        weighted_prediction = np.zeros(3)
        total_weight = 0
        
        for name, pred in predictions.items():
            if confidences[name] >= self.confidence_threshold:
                weight = regime_weights.get(name, 0.1) * confidences[name]
                weighted_prediction += pred * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_prediction /= total_weight
            final_confidence = np.max(weighted_prediction)
        else:
            weighted_prediction = np.array([0.33, 0.34, 0.33])
            final_confidence = 0.0
        
        result = {
            'prediction': weighted_prediction,
            'class': np.argmax(weighted_prediction),
            'confidence': final_confidence,
            'regime': market_regime
        }
        
        if return_all:
            result['individual_predictions'] = predictions
            result['individual_confidences'] = confidences
            result['weights_used'] = regime_weights
        
        # Cache
        self.prediction_cache[cache_key] = result
        if len(self.prediction_cache) > self.cache_size:
            self.prediction_cache.pop(next(iter(self.prediction_cache)))
        
        return result
    
    def get_model_agreement(self, predictions):
        """Calcula concordância entre modelos"""
        
        if not predictions:
            return 0.0
        
        votes = np.zeros(3)
        for pred in predictions.values():
            votes[np.argmax(pred)] += 1
        
        return np.max(votes) / len(predictions)


class HyperparameterOptimizer:
    """Sistema de otimização de hiperparâmetros"""
    
    def __init__(self, logger):
        self.logger = logger
        self.optimization_history = []
        self.best_params = {}
        self.search_spaces = self._define_search_spaces()
        
    def _define_search_spaces(self):
        """Define espaços de busca"""
        
        return {
            'xgboost_fast': {
                'n_estimators': (25, 100),
                'max_depth': (3, 6),
                'learning_rate': (0.05, 0.2),
                'subsample': (0.7, 0.9),
                'colsample_bytree': (0.7, 0.9),
                'reg_alpha': (0, 0.5),
                'reg_lambda': (0, 0.5)
            },
            
            'lgb_balanced': {
                'n_estimators': (50, 200),
                'max_depth': (3, 8),
                'learning_rate': (0.05, 0.2),
                'num_leaves': (20, 50),
                'reg_alpha': (0, 0.5),
                'reg_lambda': (0, 0.5)
            },
            
            'rf_stable': {
                'n_estimators': (50, 200),
                'max_depth': (5, 15),
                'min_samples_split': (2, 10),
                'min_samples_leaf': (1, 4)
            }
        }
    
    def optimize_hyperparameters(self, model_name, X_train, y_train, 
                               X_val, y_val, n_trials=30):
        """Otimização Bayesiana"""
        
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna não disponível - usando parâmetros padrão")
            return {}, 0.0
        
        def objective(trial):
            params = {}
            search_space = self.search_spaces.get(model_name, {})
            
            for param_name, param_range in search_space.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(
                            param_name, param_range[0], param_range[1]
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, param_range[0], param_range[1]
                        )
            
            score = self._train_and_evaluate(
                model_name, params, X_train, y_train, X_val, y_val
            )
            
            return score
        
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        self.best_params[model_name] = study.best_params
        self.optimization_history.append({
            'model': model_name,
            'best_value': study.best_value,
            'best_params': study.best_params
        })
        
        return study.best_params, study.best_value
    
    def _train_and_evaluate(self, model_name, params, X_train, y_train, 
                          X_val, y_val):
        """Treina e avalia modelo"""
        
        try:
            if model_name == 'xgboost_fast':
                model = xgb.XGBClassifier(**params, random_state=42, 
                                        objective='multi:softprob', num_class=3)
                model.fit(X_train, y_train, 
                         eval_set=[(X_val, y_val)],
                         early_stopping_rounds=10,
                         verbose=False)
                
            elif model_name == 'lgb_balanced':
                model = lgb.LGBMClassifier(**params, random_state=42,
                                         objective='multiclass', num_class=3)
                model.fit(X_train, y_train,
                         eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
                
            elif model_name == 'rf_stable':
                model = RandomForestClassifier(**params, random_state=42)
                model.fit(X_train, y_train)
            
            # Calcular score
            y_pred = model.predict(X_val)
            accuracy = np.mean(y_pred == y_val)
            
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Erro no treino: {e}")
            return 0.0


class ModelManager:
    """Gerencia carregamento e acesso aos modelos ML com Ensemble Avançado"""
    
    def __init__(self, models_dir: str = 'models'):
        # Configurar diretório
        if models_dir in ['models', 'saved_models', 'src/models']:
            self.models_dir = r"C:\Users\marth\OneDrive\Programacao\Python\Projetos\ML_Tradingv2.0\src\models\models_regime3"
        else:
            self.models_dir = models_dir
            
        self.models: Dict[str, Any] = {}
        self.model_features: Dict[str, List[str]] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.logger = logging.getLogger('ModelManager')
        
        # Sistema de ensemble
        self.ensemble = MultiModalEnsemble(self.logger)
        self.hyperopt = HyperparameterOptimizer(self.logger)
        
        # Configurações
        self.supported_extensions = ['.pkl', '.joblib', '.h5']
        self.use_ensemble = True
        self.ensemble_initialized = False
        self.ensemble_config = {
            'sequence_length': 60,
            'n_features': None,
            'retrain_interval': 86400,
            'min_train_samples': 5000
        }
        
        # Performance tracking
        self.model_performance = {}
        self.last_optimization = None
        
        # Cache para prepare_features_for_model
        self._feature_prep_cache = {}
        self._cache_size = 100
        
    def load_models(self) -> bool:
        """Carrega todos os modelos disponíveis"""
        try:
            if not os.path.exists(self.models_dir):
                self.logger.error(f"Diretório de modelos não encontrado: {self.models_dir}")
                return False
                
            # Listar arquivos
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
                    self.logger.info(f"Carregando modelo: {model_name}")
                    
                    if model_file.endswith('.h5') and TF_AVAILABLE:
                        # Modelo Keras/TensorFlow
                        model = keras.models.load_model(model_path)
                    else:
                        # Modelo sklearn/xgboost/lightgbm
                        model = joblib.load(model_path)
                    
                    self.models[model_name] = model
                    
                    # Descobrir features
                    features = self._extract_features(model, model_name)
                    self.model_features[model_name] = features
                    self.logger.info(f"Modelo {model_name}: {len(features)} features")
                    
                    # Carregar metadados
                    self._load_model_metadata(model_name)
                    
                except Exception as e:
                    self.logger.error(f"Erro carregando modelo {model_name}: {e}")
                    continue
            
            self.logger.info(f"Carregados {len(self.models)} modelos com sucesso")
            
            # Carregar ensemble se existir
            self._load_ensemble()
            
            # Log resumo
            self._log_models_summary()
            
            return len(self.models) > 0
            
        except Exception as e:
            self.logger.error(f"Erro carregando modelos: {e}", exc_info=True)
            return False
    
    def setup_auto_retraining(self, config: Dict):
        """Configura sistema de retreinamento automático"""
        
        self.auto_retrain_config = config
        self.retrain_scheduler = ModelRetrainScheduler(self, config)
        self.retrain_history = []
        
        # Iniciar scheduler se habilitado
        if config.get('auto_retrain_enabled', False):
            self.retrain_scheduler.start()
            self.logger.info("Sistema de retreinamento automático ativado")

    def retrain_models(self, training_data: pd.DataFrame, 
                    force: bool = False) -> Dict:
        """Retreina modelos com novos dados"""
        
        self.logger.info("Iniciando retreinamento dos modelos")
        
        retrain_results = {}
        
        try:
            # 1. Validar dados de treino
            if not self._validate_training_data(training_data):
                raise ValueError("Dados de treino inválidos")
                
            # 2. Preparar features
            X_train, y_train = self._prepare_training_data(training_data)
            
            # 3. Retreinar cada modelo
            for model_name, model in self.models.items():
                if self._should_retrain_model(model_name, force):
                    self.logger.info(f"Retreinando {model_name}")
                    
                    # Fazer backup do modelo atual
                    self._backup_model(model_name, model)
                    
                    # Retreinar
                    retrained_model = self._retrain_single_model(
                        model, X_train, y_train, model_name
                    )
                    
                    # Validar novo modelo
                    if self._validate_retrained_model(retrained_model, model):
                        self.models[model_name] = retrained_model
                        retrain_results[model_name] = "success"
                    else:
                        self.logger.warning(f"Modelo {model_name} não passou na validação")
                        retrain_results[model_name] = "validation_failed"
                        
            # 4. Salvar modelos retreinados
            self._save_retrained_models()
            
            # 5. Registrar histórico
            self.retrain_history.append({
                'timestamp': datetime.now(),
                'results': retrain_results,
                'data_size': len(training_data)
            })
            
        except Exception as e:
            self.logger.error(f"Erro no retreinamento: {e}")
            retrain_results['error'] = str(e)
            
        return retrain_results

    def _should_retrain_model(self, model_name: str, force: bool) -> bool:
        """Determina se modelo deve ser retreinado"""
        
        if force:
            return True
            
        # Verificar última vez que foi retreinado
        last_retrain = self._get_last_retrain_time(model_name)
        if not last_retrain:
            return True
            
        # Verificar intervalo mínimo
        min_interval = self.auto_retrain_config.get('min_retrain_interval_hours', 24)
        hours_since_retrain = (datetime.now() - last_retrain).total_seconds() / 3600
        
        return hours_since_retrain >= min_interval

    def _extract_features(self, model: Any, model_name: str) -> List[str]:
        """Extrai lista de features de um modelo"""
        features = []
        
        try:
            # XGBoost
            if hasattr(model, 'get_booster'):
                try:
                    booster = model.get_booster()
                    if hasattr(booster, 'feature_names'):
                        features = list(booster.feature_names)
                    elif hasattr(booster, 'get_score'):
                        features = list(booster.get_score(importance_type='weight').keys())
                    
                    if not features and hasattr(model, 'feature_names_in_'):
                        features = list(model.feature_names_in_)
                        
                except Exception as e:
                    self.logger.warning(f"Erro extraindo features XGBoost: {e}")
            
            # LightGBM
            elif hasattr(model, 'booster_'):
                try:
                    booster = model.booster_
                    if hasattr(booster, 'feature_name'):
                        features = list(booster.feature_name())
                    elif hasattr(model, 'feature_name_'):
                        features = list(model.feature_name_)
                    elif hasattr(model, 'feature_names_in_'):
                        features = list(model.feature_names_in_)
                except Exception as e:
                    self.logger.warning(f"Erro extraindo features LightGBM: {e}")
            
            # Keras/TensorFlow
            elif hasattr(model, 'input_shape'):
                # Para modelos deep learning, carregar de arquivo
                features_file = os.path.join(self.models_dir, f"{model_name}_features.json")
                if os.path.exists(features_file):
                    with open(features_file, 'r') as f:
                        features = json.load(f)
            
            # Scikit-learn
            elif hasattr(model, 'feature_names_in_'):
                features = list(model.feature_names_in_)
            elif hasattr(model, 'get_feature_names_out'):
                try:
                    features = list(model.get_feature_names_out())
                except:
                    pass
                    
            # Arquivo de features
            if not features:
                features_file = os.path.join(self.models_dir, f"{model_name}_features.json")
                if os.path.exists(features_file):
                    with open(features_file, 'r') as f:
                        features = json.load(f)
                        self.logger.info(f"Features carregadas de arquivo para {model_name}")
            
            # Validar
            if features:
                seen = set()
                unique_features = []
                for f in features:
                    if f not in seen:
                        seen.add(f)
                        unique_features.append(f)
                features = unique_features
                
                self.logger.info(f"Modelo {model_name}: {len(features)} features extraídas")
            else:
                self.logger.warning(f"Nenhuma feature extraída para {model_name}")
                
        except Exception as e:
            self.logger.error(f"Erro extraindo features do modelo {model_name}: {e}")
            
        return features
    
    def _load_model_metadata(self, model_name: str):
        """Carrega metadados do modelo"""
        metadata_file = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    self.model_metadata[model_name] = json.load(f)
                    self.logger.info(f"Metadados carregados para {model_name}")
            except Exception as e:
                self.logger.warning(f"Erro carregando metadados de {model_name}: {e}")
    
    def _load_ensemble(self):
        """Carrega ensemble salvo se existir"""
        ensemble_dir = os.path.join(self.models_dir, 'ensemble')
        
        if os.path.exists(ensemble_dir):
            try:
                # Carregar pesos
                weights_path = os.path.join(self.models_dir, 'ensemble_weights.json')
                if os.path.exists(weights_path):
                    with open(weights_path, 'r') as f:
                        weights_data = json.load(f)
                        
                    # Determinar número de features
                    n_features = weights_data.get('config', {}).get('n_features')
                    if not n_features and self.model_features:
                        n_features = len(list(self.model_features.values())[0])
                    
                    if n_features:
                        # Inicializar ensemble
                        self.ensemble.initialize_models(
                            n_features=n_features,
                            sequence_length=weights_data.get('config', {}).get('sequence_length', 60)
                        )
                        
                        # Carregar modelos do ensemble
                        for model_name in self.ensemble.models.keys():
                            model_path = os.path.join(ensemble_dir, f"{model_name}.pkl")
                            h5_path = os.path.join(ensemble_dir, f"{model_name}.h5")
                            
                            if os.path.exists(h5_path) and TF_AVAILABLE:
                                self.ensemble.models[model_name] = keras.models.load_model(h5_path)
                            elif os.path.exists(model_path):
                                self.ensemble.models[model_name] = joblib.load(model_path)
                        
                        # Aplicar pesos
                        self.ensemble.model_weights = weights_data.get('model_weights', {})
                        self.ensemble.regime_weights = weights_data.get('regime_weights', 
                                                                      self.ensemble.regime_weights)
                        
                        self.ensemble_initialized = True
                        self.ensemble_config.update(weights_data.get('config', {}))
                        
                        self.logger.info("Ensemble carregado com sucesso")
                        
            except Exception as e:
                self.logger.warning(f"Erro carregando ensemble: {e}")
    
    def _log_models_summary(self):
        """Log resumo dos modelos"""
        self.logger.info("=== RESUMO DOS MODELOS ===")
        
        # Modelos tradicionais
        for model_name, model in self.models.items():
            features = self.model_features.get(model_name, [])
            metadata = self.model_metadata.get(model_name, {})
            
            self.logger.info(f"\nModelo: {model_name}")
            self.logger.info(f"  Tipo: {type(model).__name__}")
            self.logger.info(f"  Features: {len(features)}")
            
            if metadata:
                self.logger.info(f"  Accuracy: {metadata.get('accuracy', 'N/A')}")
                self.logger.info(f"  Training date: {metadata.get('training_date', 'N/A')}")
        
        # Ensemble
        if self.ensemble_initialized:
            self.logger.info("\n=== ENSEMBLE MULTI-MODAL ===")
            self.logger.info(f"Modelos: {list(self.ensemble.models.keys())}")
            self.logger.info(f"Configuração: {self.ensemble_config}")
                
        self.logger.info("========================")
    
    def initialize_ensemble(self, n_features=None, force_retrain=False):
        """Inicializa e treina o ensemble multi-modal"""
        
        try:
            # Determinar número de features
            if n_features is None:
                if self.model_features:
                    n_features = len(list(self.model_features.values())[0])
                else:
                    self.logger.warning("Sem features definidas, usando padrão 32")
                    n_features = 32
            
            self.ensemble_config['n_features'] = n_features
            
            # Inicializar modelos
            self.ensemble.initialize_models(
                n_features=n_features,
                sequence_length=self.ensemble_config['sequence_length']
            )
            
            # Carregar pesos se existirem
            if not force_retrain:
                weights_path = os.path.join(self.models_dir, 'ensemble_weights.json')
                if os.path.exists(weights_path):
                    with open(weights_path, 'r') as f:
                        saved_weights = json.load(f)
                        self.ensemble.model_weights = saved_weights.get('model_weights', {})
                        self.ensemble.regime_weights = saved_weights.get('regime_weights', {})
            
            self.ensemble_initialized = True
            self.logger.info(f"Ensemble inicializado com {n_features} features")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro inicializando ensemble: {e}")
            return False
    
    def train_ensemble(self, X_train, y_train, X_val=None, y_val=None, 
                      optimize_hyperparams=False):
        """Treina o ensemble"""
        
        if not self.ensemble_initialized:
            self.logger.error("Ensemble não inicializado")
            return False
        
        try:
            # Preparar validação
            if X_val is None or y_val is None:
                split_idx = int(len(X_train) * 0.8)
                X_val = X_train[split_idx:]
                y_val = y_train[split_idx:]
                X_train = X_train[:split_idx]
                y_train = y_train[:split_idx]
            
            # Otimizar hiperparâmetros
            if optimize_hyperparams and OPTUNA_AVAILABLE:
                self.logger.info("Otimizando hiperparâmetros...")
                
                for model_name in ['xgboost_fast', 'lgb_balanced', 'rf_stable']:
                    if model_name in self.ensemble.models:
                        best_params, score = self.hyperopt.optimize_hyperparameters(
                            model_name, X_train, y_train, X_val, y_val, n_trials=20
                        )
                        self.logger.info(f"{model_name} otimizado - Score: {score:.4f}")
            
            # Treinar modelos
            for model_name, model in self.ensemble.models.items():
                self.logger.info(f"Treinando {model_name}...")
                
                if model_name in ['lstm_intraday', 'transformer_attention']:
                    # Modelos sequenciais
                    X_train_seq = self.ensemble.prepare_sequential_data(X_train)
                    X_val_seq = self.ensemble.prepare_sequential_data(X_val)
                    
                    # Ajustar y para corresponder ao tamanho das sequências
                    sequence_length = self.ensemble_config.get('sequence_length', 60)
                    
                    # Para X_train_seq: se temos N amostras e sequence_length S, 
                    # teremos N-S+1 sequências, então y deve ter o mesmo tamanho
                    if len(X_train) >= sequence_length:
                        y_train_seq = y_train[sequence_length-1:]  # Últimos N-S+1 elementos
                    else:
                        y_train_seq = y_train  # Se não temos dados suficientes, usar padding
                    
                    if len(X_val) >= sequence_length:
                        y_val_seq = y_val[sequence_length-1:]
                    else:
                        y_val_seq = y_val
                    
                    # Garantir que os tamanhos correspondam
                    if len(X_train_seq) != len(y_train_seq):
                        min_len = min(len(X_train_seq), len(y_train_seq))
                        X_train_seq = X_train_seq[:min_len]
                        y_train_seq = y_train_seq[:min_len]
                    
                    if len(X_val_seq) != len(y_val_seq):
                        min_len = min(len(X_val_seq), len(y_val_seq))
                        X_val_seq = X_val_seq[:min_len]
                        y_val_seq = y_val_seq[:min_len]
                    
                    y_train_cat = to_categorical(y_train_seq, num_classes=3)
                    y_val_cat = to_categorical(y_val_seq, num_classes=3)
                    
                    callbacks = [
                        EarlyStopping(patience=5, restore_best_weights=True),
                        ReduceLROnPlateau(patience=3, factor=0.5)
                    ]
                    
                    history = model.fit(
                        X_train_seq, y_train_cat,
                        validation_data=(X_val_seq, y_val_cat),
                        epochs=30,
                        batch_size=32,
                        callbacks=callbacks,
                        verbose=0
                    )
                    
                else:
                    # Modelos tradicionais
                    if hasattr(model, 'fit'):
                        if model_name == 'xgboost_fast':
                            model.fit(
                                X_train, y_train,
                                eval_set=[(X_val, y_val)],
                                verbose=False
                            )
                        elif model_name == 'lgb_balanced':
                            # LightGBM usa callbacks para controlar verbose
                            model.fit(
                                X_train, y_train,
                                eval_set=[(X_val, y_val)],
                                callbacks=[lgb.log_evaluation(0)]
                            )
                        else:
                            model.fit(X_train, y_train)
            
            # Salvar ensemble
            self._save_ensemble()
            
            self.logger.info("Ensemble treinado com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro treinando ensemble: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _save_ensemble(self):
        """Salva ensemble"""
        
        try:
            ensemble_dir = os.path.join(self.models_dir, 'ensemble')
            os.makedirs(ensemble_dir, exist_ok=True)
            
            # Salvar modelos
            for model_name, model in self.ensemble.models.items():
                if model_name in ['lstm_intraday', 'transformer_attention']:
                    model_path = os.path.join(ensemble_dir, f"{model_name}.h5")
                    model.save(model_path)
                else:
                    model_path = os.path.join(ensemble_dir, f"{model_name}.pkl")
                    joblib.dump(model, model_path)
            
            # Salvar configurações
            weights_data = {
                'model_weights': self.ensemble.model_weights,
                'regime_weights': self.ensemble.regime_weights,
                'config': self.ensemble_config,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            weights_path = os.path.join(self.models_dir, 'ensemble_weights.json')
            with open(weights_path, 'w') as f:
                json.dump(weights_data, f, indent=2)
            
            self.logger.info("Ensemble salvo com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro salvando ensemble: {e}")
    
    def get_all_required_features(self) -> List[str]:
        """Retorna todas as features necessárias"""
        all_features = set()
        
        for features in self.model_features.values():
            all_features.update(features)
            
        return list(all_features)
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Retorna um modelo específico"""
        return self.models.get(model_name)
    
    def get_model_features(self, model_name: str) -> List[str]:
        """Retorna features de um modelo"""
        return self.model_features.get(model_name, [])
    
    def predict(self, features_df: pd.DataFrame, model_name: Optional[str] = None, 
               use_ensemble: Optional[bool] = None) -> Optional[Any]:
        """Executa predição"""
        
        if use_ensemble is None:
            use_ensemble = self.use_ensemble
        
        # Usar ensemble se disponível
        if use_ensemble and self.ensemble_initialized:
            return self._predict_with_ensemble(features_df)
        
        # Modelo específico
        if model_name:
            if model_name not in self.models:
                self.logger.error(f"Modelo {model_name} não encontrado")
                return None
            
            try:
                model = self.models[model_name]
                model_features = self.model_features[model_name]
                
                # Preparar features com fillna inteligente
                X = self.prepare_features_for_model(features_df, model_features)
                
                # Predição
                prediction = model.predict(X)
                
                result = {
                    'model': model_name,
                    'predictions': prediction
                }
                
                if hasattr(model, 'predict_proba'):
                    result['probabilities'] = model.predict_proba(X)
                
                return result
                
            except Exception as e:
                self.logger.error(f"Erro na predição: {e}")
                return None
        
        # Todos os modelos
        return self.ensemble_predict(features_df)
    
    def prepare_features_for_model(self, features_df: pd.DataFrame, model_features: List[str]) -> pd.DataFrame:
        """
        Prepara features com fillna inteligente baseado no tipo de feature
        
        Args:
            features_df: DataFrame com todas as features
            model_features: Lista de features necessárias para o modelo
            
        Returns:
            pd.DataFrame: Features preparadas sem valores nulos
        """
        # Validar entrada
        if features_df.empty:
            self.logger.warning("DataFrame de features vazio")
            return pd.DataFrame()
        
        # Verificar se todas as features existem
        missing_features = [f for f in model_features if f not in features_df.columns]
        if missing_features:
            self.logger.warning(f"Features ausentes: {missing_features[:5]}...")
            # Criar colunas faltantes com valores apropriados
            for feat in missing_features:
                if 'volume' in feat.lower():
                    features_df[feat] = 1.0  # Volume mínimo
                else:
                    features_df[feat] = 0.0
        
        X = features_df[model_features].copy()
        
        for col in X.columns:
            if X[col].isnull().any():
                # Contar NaNs para log
                nan_count = X[col].isnull().sum()
                
                # Estratégia baseada no tipo de feature
                if 'price' in col.lower() or 'ema' in col or 'sma' in col:
                    # Forward fill para preços e médias móveis
                    X[col] = X[col].ffill()
                    if X[col].isnull().any():  # Se ainda houver NaNs no início
                        X[col] = X[col].bfill()
                        
                elif 'volume' in col.lower():
                    # Média móvel para volume (nunca deve ser 0)
                    rolling_mean = X[col].rolling(20, min_periods=1).mean()
                    X[col] = X[col].fillna(rolling_mean)
                    if X[col].isnull().any():  # Fallback
                        mean_val = X[col].mean()
                        X[col] = X[col].fillna(mean_val if mean_val > 0 else 1.0)
                        
                elif 'rsi' in col.lower():
                    # RSI: forward fill (mantém último estado conhecido)
                    X[col] = X[col].ffill()
                    if X[col].isnull().any():  # Se ainda houver NaNs
                        X[col] = X[col].fillna(50)  # RSI neutro como último recurso
                        
                elif 'momentum' in col.lower() or 'return' in col.lower():
                    # Momentum/returns: usar forward fill primeiro
                    X[col] = X[col].ffill()
                    # Se ainda houver NaN, usar 0 apenas para momentum (é aceitável)
                    if X[col].isnull().any():
                        X[col] = X[col].fillna(0)
                    
                elif 'volatility' in col.lower() or 'atr' in col.lower():
                    # Volatilidade: usar média recente
                    rolling_mean = X[col].rolling(20, min_periods=1).mean()
                    X[col] = X[col].fillna(rolling_mean)
                    if X[col].isnull().any():
                        mean_val = X[col].mean()
                        X[col] = X[col].fillna(mean_val if not pd.isna(mean_val) else 0.01)
                        
                elif 'bb_' in col or 'bollinger' in col.lower():
                    # Bollinger Bands: forward fill
                    X[col] = X[col].ffill().bfill()
                    
                elif 'macd' in col.lower():
                    # MACD: forward fill primeiro
                    X[col] = X[col].ffill()
                    if X[col].isnull().any():
                        # Para MACD, usar interpolação antes de 0
                        X[col] = X[col].interpolate(method='linear', limit=3)
                        if X[col].isnull().any():
                            X[col] = X[col].fillna(0)  # Apenas como último recurso
                        
                else:
                    # Default: interpolação linear com limite
                    X[col] = X[col].interpolate(method='linear', limit=5)
                    # Se ainda houver NaNs, forward fill
                    if X[col].isnull().any():
                        X[col] = X[col].ffill().bfill()
                        # Último recurso: usar média da coluna
                        if X[col].isnull().any():
                            col_mean = X[col].mean()
                            X[col] = X[col].fillna(col_mean if not pd.isna(col_mean) else 0)
                
                # Log se muitos valores foram preenchidos
                if nan_count > len(X) * 0.1:  # Mais de 10% de NaNs
                    self.logger.warning(f"Feature '{col}' tinha {nan_count}/{len(X)} NaNs ({nan_count/len(X)*100:.1f}%)")
        
        return X

    def ensemble_predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Predição com todos os modelos tradicionais"""
        predictions = {}
        
        for model_name in self.models:
            pred = self.predict(features_df, model_name=model_name, use_ensemble=False)
            if pred is not None:
                predictions[model_name] = pred
                
        return predictions
    
    def _predict_with_ensemble(self, features_df: pd.DataFrame):
        """Predição usando ensemble multi-modal"""
        
        try:
            # Detectar regime
            market_regime = self._detect_market_regime(features_df)
            
            # Preparar features
            features = list(set().union(*[self.model_features[m] for m in self.models.keys()]))
            X = self.prepare_features_for_model(features_df, features)
            
            feature_array = X.values
            if len(feature_array.shape) == 1:
                feature_array = feature_array.reshape(1, -1)
            
            # Predição
            result = self.ensemble.predict_with_ensemble(
                feature_array,
                market_regime=market_regime,
                return_all=True
            )
            
            # Formatar
            prediction_map = {0: 'sell', 1: 'hold', 2: 'buy'}
            
            return {
                'ensemble': True,
                'prediction': prediction_map[result['class']],
                'probabilities': result['prediction'],
                'confidence': result['confidence'],
                'market_regime': market_regime,
                'individual_predictions': result.get('individual_predictions', {}),
                'model_agreement': self.ensemble.get_model_agreement(
                    result.get('individual_predictions', {})
                )
            }
            
        except Exception as e:
            self.logger.error(f"Erro na predição do ensemble: {e}")
            return None
    
    def _detect_market_regime(self, features_df: pd.DataFrame):
        """Detecta regime de mercado"""
        
        try:
            # Volatilidade
            if 'volatility_20' in features_df.columns:
                vol = features_df['volatility_20'].iloc[-1]
                vol_mean = features_df['volatility_20'].mean()
                
                if vol > vol_mean * 1.5:
                    vol_regime = 'high_volatility'
                elif vol < vol_mean * 0.7:
                    vol_regime = 'low_volatility'
                else:
                    vol_regime = 'normal'
            else:
                vol_regime = 'undefined'
            
            # Tendência
            if all(col in features_df.columns for col in ['ema_9', 'ema_20', 'ema_50']):
                ema9 = features_df['ema_9'].iloc[-1]
                ema20 = features_df['ema_20'].iloc[-1]
                ema50 = features_df['ema_50'].iloc[-1]
                
                if ema9 > ema20 > ema50:
                    trend = 'trending'
                elif ema9 < ema20 < ema50:
                    trend = 'trending'
                else:
                    trend = 'ranging'
            else:
                trend = 'undefined'
            
            # Combinar
            if vol_regime == 'high_volatility':
                return 'high_volatility'
            elif trend == 'trending' and vol_regime == 'normal':
                return 'trending'
            elif trend == 'ranging':
                return 'ranging'
            elif vol_regime == 'low_volatility':
                return 'low_volatility'
            else:
                return 'undefined'
                
        except Exception:
            return 'undefined'
    
    def _validate_training_data(self, training_data: pd.DataFrame) -> bool:
        """Valida dados de treino"""
        try:
            if training_data.empty:
                self.logger.error("Dados de treino vazios")
                return False
            
            min_samples = self.ensemble_config.get('min_train_samples', 5000)
            if len(training_data) < min_samples:
                self.logger.error(f"Dados insuficientes: {len(training_data)} < {min_samples}")
                return False
            
            # Verificar se tem colunas de target
            target_cols = ['target', 'target_class', 'signal']
            if not any(col in training_data.columns for col in target_cols):
                self.logger.error("Coluna de target não encontrada")
                return False
            
            # Verificar valores nulos excessivos
            null_ratio = training_data.isnull().sum().sum() / (len(training_data) * len(training_data.columns))
            if null_ratio > 0.3:
                self.logger.warning(f"Muitos valores nulos: {null_ratio:.2%}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro validando dados: {e}")
            return False
    
    def _prepare_training_data(self, training_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dados para treino"""
        try:
            # Identificar coluna de target
            target_col = None
            for col in ['target', 'target_class', 'signal']:
                if col in training_data.columns:
                    target_col = col
                    break
            
            if target_col is None:
                raise ValueError("Coluna de target não encontrada")
            
            # Separar features e target
            X = training_data.drop(columns=[target_col])
            y = training_data[target_col]
            
            # Preparar features
            all_features = self.get_all_required_features()
            X_prepared = self.prepare_features_for_model(X, all_features)
            
            return X_prepared.values, np.array(y.values)
            
        except Exception as e:
            self.logger.error(f"Erro preparando dados: {e}")
            raise
    
    def _backup_model(self, model_name: str, model: Any):
        """Faz backup do modelo atual"""
        try:
            backup_dir = os.path.join(self.models_dir, 'backups')
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(backup_dir, f"{model_name}_{timestamp}.pkl")
            
            joblib.dump(model, backup_path)
            self.logger.info(f"Backup criado: {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Erro criando backup: {e}")
    
    def _retrain_single_model(self, model: Any, X_train: np.ndarray, 
                            y_train: np.ndarray, model_name: str) -> Any:
        """Retreina um modelo específico"""
        try:
            # Usar parâmetros otimizados se disponível
            if model_name in self.hyperopt.best_params:
                params = self.hyperopt.best_params[model_name]
                self.logger.info(f"Usando parâmetros otimizados para {model_name}")
            else:
                params = {}
            
            # Retreinar baseado no tipo
            if hasattr(model, 'fit'):
                if 'xgb' in str(type(model)).lower():
                    model.fit(X_train, y_train, verbose=False)
                elif 'lgb' in str(type(model)).lower():
                    model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(0)])
                else:
                    model.fit(X_train, y_train)
                    
                return model
            else:
                self.logger.error(f"Modelo {model_name} não suporta retreinamento")
                return model
                
        except Exception as e:
            self.logger.error(f"Erro retreinando {model_name}: {e}")
            return model
    
    def _validate_retrained_model(self, new_model: Any, old_model: Any) -> bool:
        """Valida modelo retreinado"""
        try:
            # Verificações básicas
            if new_model is None:
                return False
            
            # Comparar tipos
            if type(new_model) != type(old_model):
                self.logger.warning("Tipos de modelo diferentes após retreino")
                return False
            
            # Verificar se modelo pode fazer predições
            dummy_input = np.random.random((1, 10))
            try:
                _ = new_model.predict(dummy_input)
                return True
            except:
                return False
                
        except Exception as e:
            self.logger.error(f"Erro validando modelo retreinado: {e}")
            return False
    
    def _save_retrained_models(self):
        """Salva modelos retreinados"""
        try:
            for model_name, model in self.models.items():
                model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
                joblib.dump(model, model_path)
            
            self.logger.info("Modelos retreinados salvos")
            
        except Exception as e:
            self.logger.error(f"Erro salvando modelos: {e}")
    
    def _get_last_retrain_time(self, model_name: str) -> Optional[datetime]:
        """Obtém última vez que modelo foi retreinado"""
        try:
            metadata = self.model_metadata.get(model_name, {})
            retrain_time_str = metadata.get('last_retrain')
            
            if retrain_time_str:
                return datetime.fromisoformat(retrain_time_str)
            
            return None
            
        except Exception:
            return None