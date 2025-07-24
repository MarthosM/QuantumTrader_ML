# src/training/model_trainer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import pickle
import joblib
from pathlib import Path
from datetime import datetime

# ML Libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# TensorFlow imports with proper error handling
try:
    import tensorflow as tf
    # Para TensorFlow 2.19+ com Keras 3.x, usar keras diretamente
    import keras
    from keras import layers, models, callbacks, optimizers
    Adam = optimizers.Adam
    TENSORFLOW_AVAILABLE = True
except (ImportError, AttributeError):
    # TensorFlow/Keras not installed or incompatible
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    models = None
    callbacks = None
    optimizers = None
    Adam = None

class ModelTrainer:
    """Treinador individual de modelos ML para trading"""
    
    def __init__(self, model_save_path: str = 'src/training/models/trained'):
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Configurações de modelos
        self.model_configs = self._get_model_configs()
        
        # Métricas de treinamento
        self.training_history = {}
        
    def train_model(self, model_type: str,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_val: pd.DataFrame,
                   y_val: pd.Series,
                   hyperparams: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """
        Treina um modelo específico
        
        Args:
            model_type: Tipo do modelo ('xgboost', 'lightgbm', etc)
            X_train: Features de treino
            y_train: Targets de treino
            X_val: Features de validação
            y_val: Targets de validação
            hyperparams: Hiperparâmetros customizados
            
        Returns:
            Tupla (modelo_treinado, métricas)
        """
        self.logger.info(f"Treinando modelo {model_type}")
        
        # Obter configuração base
        if model_type not in self.model_configs:
            raise ValueError(f"Tipo de modelo desconhecido: {model_type}")
        
        config = self.model_configs[model_type].copy()
        
        # Atualizar com hiperparâmetros customizados
        if hyperparams:
            config.update(hyperparams)
        
        # Treinar modelo específico
        if model_type == 'xgboost':
            model, metrics = self._train_xgboost(X_train, y_train, X_val, y_val, config)
            
        elif model_type == 'lightgbm':
            model, metrics = self._train_lightgbm(X_train, y_train, X_val, y_val, config)
            
        elif model_type == 'random_forest':
            model, metrics = self._train_random_forest(X_train, y_train, X_val, y_val, config)
            
        elif model_type == 'svm':
            model, metrics = self._train_svm(X_train, y_train, X_val, y_val, config)
            
        elif model_type == 'neural_network':
            model, metrics = self._train_neural_network(X_train, y_train, X_val, y_val, config)
            
        elif model_type == 'lstm':
            model, metrics = self._train_lstm(X_train, y_train, X_val, y_val, config)
            
        elif model_type == 'transformer':
            model, metrics = self._train_transformer(X_train, y_train, X_val, y_val, config)
            
        else:
            raise ValueError(f"Modelo {model_type} não implementado")
        
        # Salvar histórico
        self.training_history[model_type] = {
            'metrics': metrics,
            'config': config,
            'timestamp': datetime.now()
        }
        
        return model, metrics
    
    def _get_model_configs(self) -> Dict:
        """Configurações base otimizadas para day trading"""
        return {
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'multi:softprob',
                'num_class': 3,
                'tree_method': 'hist',
                'random_state': 42,
                'n_jobs': -1,
                'early_stopping_rounds': 10
            },
            
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'multiclass',
                'num_class': 3,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            },
            
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1
            },
            
            'svm': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'random_state': 42
            },
            
            'neural_network': {
                'hidden_layers': [128, 64, 32],
                'activation': 'relu',
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 50
            },
            
            'lstm': {
                'units': [64, 32],
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50,
                'sequence_length': 60
            },
            
            'transformer': {
                'd_model': 64,
                'num_heads': 8,
                'ff_dim': 256,
                'num_blocks': 3,
                'dropout': 0.2,
                'learning_rate': 0.0005,
                'batch_size': 32,
                'epochs': 50,
                'sequence_length': 60
            }
        }
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val, config):
        """Treina modelo XGBoost"""
        # Preparar dados
        eval_set = [(X_val, y_val)]
        
        # Criar e treinar modelo
        model = XGBClassifier(**config)
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Calcular métricas
        metrics = self._calculate_metrics(model, X_train, y_train, X_val, y_val)
        
        # Feature importance
        feature_importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        metrics['feature_importance'] = feature_importance.to_dict()
        
        return model, metrics
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val, config):
        """Treina modelo LightGBM"""
        # Criar e treinar modelo
        model = LGBMClassifier(**config)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(10),
                lgb.log_evaluation(0)
            ]
        )
        
        # Calcular métricas
        metrics = self._calculate_metrics(model, X_train, y_train, X_val, y_val)
        
        # Feature importance
        feature_importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        metrics['feature_importance'] = feature_importance.to_dict()
        
        return model, metrics
    
    def _train_random_forest(self, X_train, y_train, X_val, y_val, config):
        """Treina Random Forest"""
        model = RandomForestClassifier(**config)
        model.fit(X_train, y_train)
        
        metrics = self._calculate_metrics(model, X_train, y_train, X_val, y_val)
        
        # Feature importance
        feature_importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        metrics['feature_importance'] = feature_importance.to_dict()
        
        return model, metrics
    
    def _train_svm(self, X_train, y_train, X_val, y_val, config):
        """Treina Support Vector Machine"""
        model = SVC(**config)
        model.fit(X_train, y_train)
        
        metrics = self._calculate_metrics(model, X_train, y_train, X_val, y_val)
        
        return model, metrics
    
    def _train_neural_network(self, X_train, y_train, X_val, y_val, config):
        """Treina rede neural MLP (Multilayer Perceptron)"""
        from sklearn.preprocessing import StandardScaler
        
        # Padronizar dados para rede neural
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Configurar MLP
        mlp_config = {
            'hidden_layer_sizes': tuple(config['hidden_layers']),
            'activation': config['activation'],
            'learning_rate_init': config['learning_rate'],
            'batch_size': config['batch_size'],
            'max_iter': config['epochs'],
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
            'random_state': 42
        }
        
        model = MLPClassifier(**mlp_config)
        model.fit(X_train_scaled, y_train)
        
        # Para usar o modelo, precisamos salvar o scaler também
        # Criar um wrapper que inclui o scaler
        class MLPWrapper:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
            
            def predict(self, X):
                X_scaled = self.scaler.transform(X)
                return self.model.predict(X_scaled)
            
            def predict_proba(self, X):
                X_scaled = self.scaler.transform(X)
                return self.model.predict_proba(X_scaled)
        
        wrapped_model = MLPWrapper(model, scaler)
        
        # Calcular métricas usando dados escalados
        metrics = self._calculate_metrics(wrapped_model, X_train, y_train, X_val, y_val)
        
        return wrapped_model, metrics
    
    def _train_lstm(self, X_train, y_train, X_val, y_val, config):
        """Treina modelo LSTM para séries temporais"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow não está disponível para modelos LSTM")
        
        # Preparar dados para LSTM (adicionar dimensão temporal)
        sequence_length = config['sequence_length']
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val, sequence_length)
        
        # Criar modelo
        model = self._build_lstm_model(
            input_shape=(sequence_length, X_train.shape[1]),
            config=config
        )
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Treinar
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=[early_stop, reduce_lr]
        )
        
        # Calcular métricas
        metrics = self._calculate_deep_learning_metrics(
            model, X_train_seq, y_train_seq, X_val_seq, y_val_seq
        )
        metrics['training_history'] = history.history
        
        return model, metrics
    
    def _build_lstm_model(self, input_shape, config):
        """Constrói arquitetura LSTM"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow não está disponível para modelos LSTM")
        
        model = models.Sequential()
        
        # LSTM layers
        for i, units in enumerate(config['units']):
            return_sequences = i < len(config['units']) - 1
            
            model.add(layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=config['dropout'],
                recurrent_dropout=config['recurrent_dropout'],
                input_shape=input_shape if i == 0 else None
            ))
        
        # Dense layers
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(3, activation='softmax'))
        
        # Compilar
        if Adam is not None:
            optimizer = Adam(learning_rate=config['learning_rate'])
        else:
            optimizer = 'adam'  # Fallback para string
        
        model.compile(
            optimizer=optimizer,  # type: ignore
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _train_transformer(self, X_train, y_train, X_val, y_val, config):
        """Treina modelo Transformer com atenção"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow não está disponível para modelos Transformer")
        
        # Preparar dados
        sequence_length = config['sequence_length']
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val, sequence_length)
        
        # Criar modelo
        model = self._build_transformer_model(
            input_shape=(sequence_length, X_train.shape[1]),
            config=config
        )
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Treinar
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=[early_stop]
        )
        
        # Métricas
        metrics = self._calculate_deep_learning_metrics(
            model, X_train_seq, y_train_seq, X_val_seq, y_val_seq
        )
        metrics['training_history'] = history.history
        
        return model, metrics
    
    def _build_transformer_model(self, input_shape, config):
        """Constrói arquitetura Transformer"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow não está disponível para modelos Transformer")
        
        inputs = layers.Input(shape=input_shape)
        
        # Positional encoding
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        position_embedding = layers.Embedding(
            input_dim=input_shape[0],
            output_dim=config['d_model']
        )(positions)
        
        # Project input to d_model dimensions
        x = layers.Dense(config['d_model'])(inputs)
        
        # Add positional encoding
        x = x + position_embedding
        
        # Transformer blocks
        for _ in range(config['num_blocks']):
            # Multi-head attention
            attn_output = layers.MultiHeadAttention(
                num_heads=config['num_heads'],
                key_dim=config['d_model']
            )(x, x)
            
            # Skip connection and normalization
            x = layers.LayerNormalization()(x + attn_output)
            
            # Feed forward
            ff_output = layers.Dense(config['ff_dim'], activation='relu')(x)
            ff_output = layers.Dense(config['d_model'])(ff_output)
            ff_output = layers.Dropout(config['dropout'])(ff_output)
            
            # Skip connection and normalization
            x = layers.LayerNormalization()(x + ff_output)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(config['dropout'])(x)
        outputs = layers.Dense(3, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compilar
        if Adam is not None:
            optimizer = Adam(learning_rate=config['learning_rate'])
        else:
            optimizer = 'adam'  # Fallback para string
        
        model.compile(
            optimizer=optimizer,  # type: ignore
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _prepare_sequences(self, X, y, sequence_length):
        """Prepara sequências para modelos temporais"""
        sequences = []
        targets = []
        
        for i in range(len(X) - sequence_length):
            sequences.append(X.iloc[i:i+sequence_length].values)
            targets.append(y.iloc[i+sequence_length])
        
        return np.array(sequences), np.array(targets)
    
    def _calculate_metrics(self, model, X_train, y_train, X_val, y_val):
        """Calcula métricas de performance para modelos sklearn"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Predições
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Probabilidades
        y_train_proba = model.predict_proba(X_train)
        y_val_proba = model.predict_proba(X_val)
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'train_precision': precision_score(y_train, y_train_pred, average='weighted'),
            'val_precision': precision_score(y_val, y_val_pred, average='weighted'),
            'train_recall': recall_score(y_train, y_train_pred, average='weighted'),
            'val_recall': recall_score(y_val, y_val_pred, average='weighted'),
            'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
            'val_f1': f1_score(y_val, y_val_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_val, y_val_pred).tolist(),
            'classification_report': classification_report(y_val, y_val_pred, output_dict=True),
            'max_proba_mean': np.mean(np.max(y_val_proba, axis=1))
        }
        
        # Métricas de trading específicas
        trading_metrics = self._calculate_trading_metrics(y_val, y_val_pred, y_val_proba)
        metrics.update(trading_metrics)
        
        return metrics
    
    def _calculate_deep_learning_metrics(self, model, X_train, y_train, X_val, y_val):
        """Calcula métricas para modelos deep learning"""
        # Predições
        y_train_proba = model.predict(X_train)
        y_val_proba = model.predict(X_val)
        
        y_train_pred = np.argmax(y_train_proba, axis=1)
        y_val_pred = np.argmax(y_val_proba, axis=1)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'train_precision': precision_score(y_train, y_train_pred, average='weighted'),
            'val_precision': precision_score(y_val, y_val_pred, average='weighted'),
            'train_recall': recall_score(y_train, y_train_pred, average='weighted'),
            'val_recall': recall_score(y_val, y_val_pred, average='weighted'),
            'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
            'val_f1': f1_score(y_val, y_val_pred, average='weighted'),
            'max_proba_mean': np.mean(np.max(y_val_proba, axis=1))
        }
        
        # Métricas de trading
        trading_metrics = self._calculate_trading_metrics(y_val, y_val_pred, y_val_proba)
        metrics.update(trading_metrics)
        
        return metrics
    
    def _calculate_trading_metrics(self, y_true, y_pred, y_proba):
        """Calcula métricas específicas para trading"""
        # Simular trades baseado nas predições
        # 0: Sell, 1: Hold, 2: Buy
        trades = []
        
        for i in range(len(y_pred)):
            if y_pred[i] != 1:  # Não é hold
                trades.append({
                    'position': y_pred[i],
                    'actual': y_true.iloc[i] if hasattr(y_true, 'iloc') else y_true[i],
                    'confidence': np.max(y_proba[i])
                })
        
        if not trades:
            return {
                'trade_accuracy': 0,
                'avg_confidence': 0,
                'trades_per_sample': 0
            }
        
        # Calcular acurácia dos trades
        correct_trades = sum(1 for t in trades if t['position'] == t['actual'])
        trade_accuracy = correct_trades / len(trades) if trades else 0
        
        # Confiança média
        avg_confidence = np.mean([t['confidence'] for t in trades])
        
        # Frequência de trades
        trades_per_sample = len(trades) / len(y_pred)
        
        return {
            'trade_accuracy': trade_accuracy,
            'avg_confidence': avg_confidence,
            'trades_per_sample': trades_per_sample,
            'total_trades': len(trades)
        }
    
    def save_model(self, model, model_type: str, metrics: Dict, 
                  feature_names: List[str]):
        """Salva modelo treinado com metadados"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Nome do arquivo
        model_filename = f"{model_type}_{timestamp}.pkl"
        model_path = self.model_save_path / model_filename
        
        # Preparar metadados
        metadata = {
            'model_type': model_type,
            'timestamp': timestamp,
            'metrics': metrics,
            'feature_names': feature_names,
            'feature_count': len(feature_names),
            'training_config': self.model_configs.get(model_type, {})
        }
        
        # Salvar modelo
        if model_type in ['lstm', 'transformer']:
            # Modelos Keras
            model.save(model_path.with_suffix('.h5'))
            metadata['model_file'] = str(model_path.with_suffix('.h5'))
        else:
            # Modelos sklearn/xgboost/lightgbm
            joblib.dump(model, model_path)
            metadata['model_file'] = str(model_path)
        
        # Salvar metadados
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        # Salvar lista de features
        features_path = model_path.with_name(f"{model_type}_{timestamp}_features.json")
        with open(features_path, 'w') as f:
            import json
            json.dump(feature_names, f, indent=2)
        
        self.logger.info(f"Modelo {model_type} salvo em {model_path}")
        
        return {
            'model_path': str(model_path),
            'metadata_path': str(metadata_path),
            'features_path': str(features_path)
        }