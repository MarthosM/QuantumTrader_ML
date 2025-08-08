"""
HybridStrategy - Combina modelos Tick-Only e Book-Only
Implementação da arquitetura HMARL com estratégia híbrida
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import json
import joblib
import lightgbm as lgb
from datetime import datetime
import logging

class HybridStrategy:
    """
    Estratégia híbrida que combina:
    - Tick-Only Model: Para regime detection e sinais de médio prazo
    - Book-Only Model: Para timing preciso e microestrutura
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Modelos
        self.tick_model = None
        self.book_model = None
        
        # Scalers
        self.tick_scaler = None
        self.book_scaler = None
        
        # Features requeridas
        self.tick_features = []
        self.book_features = []
        
        # Estados
        self.current_regime = 'undefined'
        self.last_tick_signal = 0
        self.last_book_signal = 0
        
        # Configurações da estratégia
        self.regime_threshold = config.get('regime_threshold', 0.6)
        self.tick_weight = config.get('tick_weight', 0.4)  # 40% peso para tick
        self.book_weight = config.get('book_weight', 0.6)  # 60% peso para book
        
        # Risk management
        self.max_position = config.get('max_position', 1)
        self.stop_loss = config.get('stop_loss', 0.02)  # 2%
        self.take_profit = config.get('take_profit', 0.03)  # 3%
        
    def load_models(self):
        """Carrega modelos tick-only e book-only"""
        
        self.logger.info("="*80)
        self.logger.info("CARREGANDO MODELOS HÍBRIDOS")
        self.logger.info("="*80)
        
        models_path = Path(self.config.get('models_path', 'models'))
        
        # 1. CARREGAR MODELO TICK-ONLY
        tick_path = models_path / 'csv_5m'
        if tick_path.exists():
            # Procurar modelo tick específico
            tick_model_file = tick_path / 'lightgbm_tick.txt'
            if tick_model_file.exists():
                tick_models = [tick_model_file]
            else:
                # Fallback: encontrar modelo mais recente
                tick_models = list(tick_path.glob('lightgbm_*.txt'))
            
            if tick_models:
                latest_tick = max(tick_models, key=lambda x: x.stat().st_mtime)
                self.tick_model = lgb.Booster(model_file=str(latest_tick))
                self.logger.info(f"[OK] Modelo Tick-Only carregado: {latest_tick.name}")
                
                # Carregar scaler
                tick_scaler_file = tick_path / 'scaler_tick.pkl'
                if tick_scaler_file.exists():
                    tick_scalers = [tick_scaler_file]
                else:
                    tick_scalers = list(tick_path.glob('scaler_*.pkl'))
                
                if tick_scalers:
                    latest_scaler = max(tick_scalers, key=lambda x: x.stat().st_mtime)
                    self.tick_scaler = joblib.load(latest_scaler)
                    self.logger.info(f"[OK] Tick Scaler carregado")
                
                # Carregar features
                feature_file = tick_path / 'features_tick.json'
                if feature_file.exists():
                    feature_files = [feature_file]
                else:
                    feature_files = list(tick_path.glob('features_*.json'))
                
                if feature_files:
                    latest_features = max(feature_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_features, 'r') as f:
                        features_info = json.load(f)
                        self.tick_features = features_info.get('feature_names', [])
                        self.logger.info(f"[OK] {len(self.tick_features)} features tick carregadas")
            else:
                self.logger.warning("[AVISO] Nenhum modelo tick-only encontrado")
        
        # 2. CARREGAR MODELO BOOK-ONLY
        book_path = models_path / 'book_moderate'
        if book_path.exists():
            # Encontrar modelo mais recente
            book_models = list(book_path.glob('lightgbm_*.txt'))
            if book_models:
                latest_book = max(book_models, key=lambda x: x.stat().st_mtime)
                self.book_model = lgb.Booster(model_file=str(latest_book))
                self.logger.info(f"[OK] Modelo Book-Only carregado: {latest_book.name}")
                
                # Carregar scaler
                book_scalers = list(book_path.glob('scaler_*.pkl'))
                if book_scalers:
                    latest_scaler = max(book_scalers, key=lambda x: x.stat().st_mtime)
                    self.book_scaler = joblib.load(latest_scaler)
                    self.logger.info(f"[OK] Book Scaler carregado")
                
                # Carregar features do CSV de importância
                feature_csvs = list(book_path.glob('features_*.csv'))
                if feature_csvs:
                    latest_csv = max(feature_csvs, key=lambda x: x.stat().st_mtime)
                    importance_df = pd.read_csv(latest_csv, index_col=0)
                    self.book_features = importance_df.index.tolist()
                    self.logger.info(f"[OK] {len(self.book_features)} features book carregadas")
            else:
                self.logger.warning("[AVISO] Nenhum modelo book-only encontrado")
        
        # Verificar se temos pelo menos um modelo
        if not self.tick_model and not self.book_model:
            raise ValueError("Nenhum modelo disponível para estratégia híbrida!")
        
        self.logger.info("\n[OK] Modelos carregados com sucesso")
        
    def detect_regime(self, tick_features: pd.DataFrame) -> Tuple[str, float]:
        """
        Detecta regime de mercado usando modelo tick-only
        
        Returns:
            regime: 'trend_up', 'trend_down', 'range', 'undefined'
            confidence: 0.0 a 1.0
        """
        
        if self.tick_model is None:
            return 'undefined', 0.0
        
        try:
            # Preparar features
            X = self._prepare_tick_features(tick_features)
            if X is None:
                return 'undefined', 0.0
            
            # Normalizar
            if self.tick_scaler:
                X_scaled = self.tick_scaler.transform(X)
            else:
                X_scaled = X
            
            # Predição
            pred_proba = self.tick_model.predict(X_scaled)
            
            # Interpretar predição
            # Assumindo classes: -1 (SELL), 0 (HOLD), 1 (BUY)
            if pred_proba.ndim == 2:
                pred_proba = pred_proba[-1]  # Última predição
            
            # Ajustar para 3 classes se necessário
            if len(pred_proba) < 3:
                pred_proba = np.array([pred_proba[0], 1 - pred_proba[0], 0])
            
            sell_prob = pred_proba[0]
            hold_prob = pred_proba[1]
            buy_prob = pred_proba[2]
            
            # Determinar regime
            max_prob = max(sell_prob, hold_prob, buy_prob)
            
            if buy_prob == max_prob and buy_prob > self.regime_threshold:
                regime = 'trend_up'
                confidence = buy_prob
            elif sell_prob == max_prob and sell_prob > self.regime_threshold:
                regime = 'trend_down'
                confidence = sell_prob
            elif hold_prob > 0.5:
                regime = 'range'
                confidence = hold_prob
            else:
                regime = 'undefined'
                confidence = max_prob
            
            self.current_regime = regime
            
            self.logger.debug(f"Regime: {regime} (confidence: {confidence:.2%})")
            
            return regime, confidence
            
        except Exception as e:
            self.logger.error(f"Erro na detecção de regime: {e}")
            return 'undefined', 0.0
    
    def get_tick_signal(self, tick_features: pd.DataFrame) -> Tuple[int, float]:
        """
        Gera sinal baseado em modelo tick-only
        
        Returns:
            signal: -1 (SELL), 0 (HOLD), 1 (BUY)
            confidence: 0.0 a 1.0
        """
        
        if self.tick_model is None:
            return 0, 0.0
        
        try:
            # Preparar features
            X = self._prepare_tick_features(tick_features)
            if X is None:
                return 0, 0.0
            
            # Normalizar
            if self.tick_scaler:
                X_scaled = self.tick_scaler.transform(X)
            else:
                X_scaled = X
            
            # Predição
            pred_proba = self.tick_model.predict(X_scaled)
            
            if pred_proba.ndim == 2:
                pred_proba = pred_proba[-1]
            
            # Ajustar para 3 classes
            if len(pred_proba) < 3:
                pred_proba = np.array([pred_proba[0], 0, 1 - pred_proba[0]])
            
            # Determinar sinal
            pred_class = np.argmax(pred_proba) - 1  # Converter para -1, 0, 1
            confidence = pred_proba[np.argmax(pred_proba)]
            
            self.last_tick_signal = pred_class
            
            return pred_class, confidence
            
        except Exception as e:
            self.logger.error(f"Erro no sinal tick: {e}")
            return 0, 0.0
    
    def get_book_signal(self, book_features: pd.DataFrame) -> Tuple[int, float]:
        """
        Gera sinal baseado em modelo book-only
        
        Returns:
            signal: -1 (SELL), 0 (HOLD), 1 (BUY)
            confidence: 0.0 a 1.0
        """
        
        if self.book_model is None:
            return 0, 0.0
        
        try:
            # Preparar features
            X = self._prepare_book_features(book_features)
            if X is None:
                return 0, 0.0
            
            # Normalizar
            if self.book_scaler:
                X_scaled = self.book_scaler.transform(X)
            else:
                X_scaled = X
            
            # Predição
            pred_proba = self.book_model.predict(X_scaled)
            
            if pred_proba.ndim == 2:
                pred_proba = pred_proba[-1]
            
            # Determinar sinal
            pred_class = np.argmax(pred_proba) - 1  # Converter para -1, 0, 1
            confidence = pred_proba[np.argmax(pred_proba)]
            
            self.last_book_signal = pred_class
            
            return pred_class, confidence
            
        except Exception as e:
            self.logger.error(f"Erro no sinal book: {e}")
            return 0, 0.0
    
    def get_hybrid_signal(self, tick_features: pd.DataFrame, 
                         book_features: pd.DataFrame) -> Dict:
        """
        Combina sinais de tick e book para decisão final
        
        Returns:
            Dict com signal, confidence, regime, components
        """
        
        # 1. Detectar regime (usa tick model)
        regime, regime_confidence = self.detect_regime(tick_features)
        
        # 2. Obter sinais individuais
        tick_signal, tick_confidence = self.get_tick_signal(tick_features)
        book_signal, book_confidence = self.get_book_signal(book_features)
        
        # 3. Combinar sinais baseado no regime
        if regime == 'trend_up' or regime == 'trend_down':
            # Em tendência: dar mais peso ao tick (60/40)
            final_tick_weight = 0.6
            final_book_weight = 0.4
            
            # Se sinais concordam, aumentar confiança
            if tick_signal == book_signal and tick_signal != 0:
                confidence_boost = 1.2
            else:
                confidence_boost = 1.0
                
        elif regime == 'range':
            # Em range: dar mais peso ao book (30/70)
            final_tick_weight = 0.3
            final_book_weight = 0.7
            
            # Book é mais importante em range
            confidence_boost = 1.1 if book_confidence > 0.7 else 1.0
            
        else:  # undefined
            # Regime indefinido: usar pesos padrão
            final_tick_weight = self.tick_weight
            final_book_weight = self.book_weight
            confidence_boost = 0.8  # Reduzir confiança
        
        # 4. Calcular sinal combinado
        combined_value = (tick_signal * tick_confidence * final_tick_weight + 
                         book_signal * book_confidence * final_book_weight)
        
        combined_confidence = ((tick_confidence * final_tick_weight + 
                               book_confidence * final_book_weight) * 
                              confidence_boost)
        
        # 5. Determinar sinal final
        if abs(combined_value) < 0.3:
            final_signal = 0  # HOLD
        elif combined_value > 0:
            final_signal = 1  # BUY
        else:
            final_signal = -1  # SELL
        
        # 6. Aplicar filtros de segurança
        # Se confiança muito baixa, forçar HOLD
        if combined_confidence < 0.5:
            final_signal = 0
        
        # Se sinais opostos com alta confiança, HOLD
        if (tick_signal * book_signal < 0 and 
            tick_confidence > 0.7 and book_confidence > 0.7):
            final_signal = 0
            combined_confidence *= 0.5
        
        # Limitar confiança a 1.0
        combined_confidence = min(combined_confidence, 1.0)
        
        result = {
            'signal': final_signal,
            'confidence': combined_confidence,
            'regime': regime,
            'regime_confidence': regime_confidence,
            'components': {
                'tick': {
                    'signal': tick_signal,
                    'confidence': tick_confidence,
                    'weight': final_tick_weight
                },
                'book': {
                    'signal': book_signal,
                    'confidence': book_confidence,
                    'weight': final_book_weight
                }
            },
            'timestamp': datetime.now()
        }
        
        # Log decisão
        self.logger.info(f"Hybrid Signal: {final_signal} "
                        f"(conf: {combined_confidence:.2%}, regime: {regime})")
        self.logger.debug(f"Components - Tick: {tick_signal} ({tick_confidence:.2%}), "
                         f"Book: {book_signal} ({book_confidence:.2%})")
        
        return result
    
    def _prepare_tick_features(self, features_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepara features para modelo tick"""
        
        if self.tick_features:
            # Usar apenas features esperadas
            missing = set(self.tick_features) - set(features_df.columns)
            if missing:
                self.logger.warning(f"Features tick faltando: {missing}")
                # Adicionar colunas faltantes com zeros
                for col in missing:
                    features_df[col] = 0
            
            return features_df[self.tick_features].values
        else:
            # Usar todas as features disponíveis
            return features_df.values
    
    def _prepare_book_features(self, features_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepara features para modelo book"""
        
        if self.book_features:
            # Usar apenas features esperadas
            missing = set(self.book_features) - set(features_df.columns)
            if missing:
                self.logger.warning(f"Features book faltando: {missing}")
                # Adicionar colunas faltantes com zeros
                for col in missing:
                    features_df[col] = 0
            
            return features_df[self.book_features].values
        else:
            # Usar todas as features disponíveis
            return features_df.values
    
    def calculate_position_size(self, signal_info: Dict, 
                               current_price: float,
                               account_balance: float) -> float:
        """
        Calcula tamanho da posição baseado no sinal e gestão de risco
        """
        
        if signal_info['signal'] == 0:
            return 0.0
        
        # Base position size (Kelly Criterion simplificado)
        confidence = signal_info['confidence']
        base_size = min(confidence * 0.25, 0.1)  # Máximo 10% por trade
        
        # Ajustar por regime
        regime = signal_info['regime']
        if regime == 'trend_up' or regime == 'trend_down':
            # Aumentar size em tendências fortes
            if signal_info['regime_confidence'] > 0.7:
                base_size *= 1.2
        elif regime == 'range':
            # Reduzir size em ranges
            base_size *= 0.8
        else:
            # Regime indefinido: ser conservador
            base_size *= 0.5
        
        # Calcular valor em dinheiro
        position_value = account_balance * base_size
        
        # Calcular número de contratos
        contracts = position_value / current_price
        
        # Aplicar limites
        contracts = min(contracts, self.max_position)
        
        return round(contracts, 2)
    
    def get_stop_loss(self, entry_price: float, signal: int) -> float:
        """Calcula stop loss baseado no sinal"""
        
        if signal == 1:  # BUY
            return entry_price * (1 - self.stop_loss)
        elif signal == -1:  # SELL
            return entry_price * (1 + self.stop_loss)
        else:
            return 0.0
    
    def get_take_profit(self, entry_price: float, signal: int) -> float:
        """Calcula take profit baseado no sinal"""
        
        if signal == 1:  # BUY
            return entry_price * (1 + self.take_profit)
        elif signal == -1:  # SELL
            return entry_price * (1 - self.take_profit)
        else:
            return 0.0