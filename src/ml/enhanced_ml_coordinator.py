# -*- coding: utf-8 -*-
"""
Enhanced ML Coordinator - Coordenador ML com capacidades de time travel
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
import numpy as np
import pandas as pd

class EnhancedMLCoordinator:
    """
    ML Coordinator com capacidades time travel
    Integra com sistema existente mantendo compatibilidade
    """
    
    def __init__(self, original_coordinator, valkey_manager=None, time_travel_engine=None):
        self.original_coordinator = original_coordinator  # Manter original
        self.valkey_manager = valkey_manager
        self.time_travel_engine = time_travel_engine
        self.logger = logging.getLogger('EnhancedMLCoordinator')
        
        # Configurações
        self.fast_mode_hours = {9, 10, 11, 14, 15, 16}  # Horários de alta atividade
        self.time_travel_threshold = 0.1  # 100ms
        self.enhanced_features_cache = {}
        self.cache_ttl = 60  # 1 minuto
        
        # Estatísticas
        self.stats = {
            'fast_mode_calls': 0,
            'time_travel_calls': 0,
            'cache_hits': 0,
            'total_predictions': 0,
            'avg_latency_fast': 0,
            'avg_latency_enhanced': 0
        }
        
    def process_prediction_request(self, symbol: str, 
                                 current_time: Optional[datetime] = None,
                                 force_mode: Optional[str] = None) -> Optional[Dict]:
        """
        Processa predição usando time travel quando apropriado
        
        Args:
            symbol: Símbolo para predição
            current_time: Tempo atual (para backtesting)
            force_mode: 'fast', 'enhanced' ou None (auto)
            
        Returns:
            Dict com predição ou None
        """
        if current_time is None:
            current_time = datetime.now()
            
        start_time = datetime.now()
        
        # Decidir modo
        if force_mode == 'fast':
            use_fast_mode = True
        elif force_mode == 'enhanced':
            use_fast_mode = False
        else:
            use_fast_mode = self._should_use_fast_mode(current_time)
        
        try:
            # MODO 1: Fast mode (sistema original)
            if use_fast_mode:
                self.stats['fast_mode_calls'] += 1
                
                # Usar coordenador original diretamente
                if hasattr(self.original_coordinator, 'process_prediction_request'):
                    prediction = self.original_coordinator.process_prediction_request(symbol)
                else:
                    # Fallback para métodos específicos
                    prediction = self._process_fast_prediction(symbol)
                
                # Adicionar metadados
                if prediction:
                    prediction['mode'] = 'fast'
                    prediction['enhanced_features'] = False
                
            # MODO 2: Enhanced mode (com time travel)
            else:
                self.stats['time_travel_calls'] += 1
                prediction = self._process_enhanced_prediction(symbol, current_time)
                
            # Calcular latência
            latency = (datetime.now() - start_time).total_seconds()
            self._update_latency_stats(use_fast_mode, latency)
            
            # Adicionar metadados gerais
            if prediction:
                prediction['processing_time'] = latency
                prediction['timestamp'] = current_time.isoformat()
                self.stats['total_predictions'] += 1
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Erro ao processar predição: {e}")
            
            # Fallback para modo fast se enhanced falhar
            if not use_fast_mode and hasattr(self.original_coordinator, 'process_prediction_request'):
                self.logger.info("Fallback para modo fast após erro em enhanced")
                return self.original_coordinator.process_prediction_request(symbol)
            
            return None
    
    def _should_use_fast_mode(self, current_time: datetime) -> bool:
        """
        Decide quando usar modo rápido vs time travel
        
        Critérios:
        1. Horário de alta atividade
        2. Latência aceitável
        3. Disponibilidade de recursos
        """
        # Verificar se time travel está disponível
        if not self.valkey_manager or not self.time_travel_engine:
            return True
        
        # Horário de alta atividade
        current_hour = current_time.hour
        if current_hour in self.fast_mode_hours:
            self.logger.debug(f"Fast mode: horário de alta atividade ({current_hour}h)")
            return True
        
        # Verificar latência média do enhanced mode
        if self.stats['time_travel_calls'] > 10:
            if self.stats['avg_latency_enhanced'] > self.time_travel_threshold:
                self.logger.debug(f"Fast mode: latência enhanced muito alta ({self.stats['avg_latency_enhanced']:.3f}s)")
                return True
        
        # Fora do horário de pico e com recursos disponíveis
        return False
    
    def _process_fast_prediction(self, symbol: str) -> Optional[Dict]:
        """
        Processa predição em modo rápido (sistema original)
        """
        try:
            # Obter dados atuais do sistema
            if hasattr(self.original_coordinator, 'data_structure'):
                current_data = self.original_coordinator.data_structure.get_latest_data(symbol)
            else:
                current_data = None
            
            if not current_data:
                self.logger.warning(f"Sem dados atuais para {symbol}")
                return None
            
            # Calcular features usando sistema original
            if hasattr(self.original_coordinator, 'feature_engine'):
                features = self.original_coordinator.feature_engine.calculate_features(current_data)
            else:
                features = current_data
            
            # Gerar predição
            if hasattr(self.original_coordinator, 'model_manager'):
                raw_prediction = self.original_coordinator.model_manager.predict(features)
            else:
                # Fallback
                raw_prediction = {'direction': 0, 'confidence': 0.5}
            
            # Formatar resposta
            return {
                'symbol': symbol,
                'direction': raw_prediction.get('direction', 0),
                'confidence': raw_prediction.get('confidence', 0.5),
                'features_used': len(features) if isinstance(features, dict) else 0,
                'mode': 'fast'
            }
            
        except Exception as e:
            self.logger.error(f"Erro no fast mode: {e}")
            return None
    
    def _process_enhanced_prediction(self, symbol: str, current_time: datetime) -> Optional[Dict]:
        """
        Processa predição com features enhanced via time travel
        """
        # Check cache
        cache_key = f"{symbol}_{current_time.strftime('%Y%m%d%H%M')}"
        
        if cache_key in self.enhanced_features_cache:
            cached_time, cached_features = self.enhanced_features_cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                self.stats['cache_hits'] += 1
                self.logger.debug("Usando features do cache")
                enhanced_features = cached_features
            else:
                enhanced_features = None
        else:
            enhanced_features = None
        
        # Calcular features se não estiver em cache
        if enhanced_features is None:
            # Configurações de lookback adaptativas
            lookback_config = self._get_adaptive_lookback(current_time)
            
            # Calcular features enhanced
            enhanced_features = self.time_travel_engine.calculate_enhanced_features(
                symbol=symbol,
                current_time=current_time,
                lookback_minutes=lookback_config['minutes'],
                use_cache=True
            )
            
            if enhanced_features:
                # Adicionar ao cache
                self.enhanced_features_cache[cache_key] = (datetime.now(), enhanced_features)
                
                # Limpar cache antigo
                self._cleanup_cache()
        
        if not enhanced_features:
            self.logger.warning("Falha ao calcular features enhanced, usando fast mode")
            return self._process_fast_prediction(symbol)
        
        # Processar predição com features enhanced
        try:
            # Preparar features para modelo
            model_features = self._prepare_features_for_model(enhanced_features)
            
            # Gerar predição
            if hasattr(self.original_coordinator, 'model_manager'):
                raw_prediction = self.original_coordinator.model_manager.predict(model_features)
            else:
                # Mock para teste
                raw_prediction = {
                    'direction': np.sign(enhanced_features.get('momentum_percentile', 0.5) - 0.5),
                    'confidence': enhanced_features.get('price_action_quality', 0.5)
                }
            
            # Ajustar confiança baseado em qualidade dos dados
            adjusted_confidence = self._adjust_confidence(
                raw_prediction.get('confidence', 0.5),
                enhanced_features
            )
            
            # Análise de regime (se disponível)
            regime_info = enhanced_features.get('volatility_regime', {})
            
            return {
                'symbol': symbol,
                'direction': raw_prediction.get('direction', 0),
                'confidence': adjusted_confidence,
                'features_used': len(model_features),
                'mode': 'enhanced',
                'enhanced_features': True,
                'time_travel_lookback': enhanced_features.get('lookback_minutes', 0),
                'data_points': enhanced_features.get('data_points', 0),
                'data_quality': enhanced_features.get('data_quality', 0),
                'regime': regime_info.get('regime', 'UNKNOWN'),
                'special_features': {
                    'volume_pattern': enhanced_features.get('volume_pattern_score', 0.5),
                    'momentum_rank': enhanced_features.get('historical_momentum_rank', 0.5),
                    'microstructure': enhanced_features.get('microstructure_imbalance', 0),
                    'seasonality': enhanced_features.get('intraday_seasonality', 0.5)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Erro no enhanced mode: {e}")
            return self._process_fast_prediction(symbol)
    
    def _get_adaptive_lookback(self, current_time: datetime) -> Dict:
        """
        Define lookback adaptativo baseado no contexto
        """
        hour = current_time.hour
        
        # Configurações por período
        if 9 <= hour <= 10:  # Abertura
            return {
                'minutes': 60,
                'reason': 'market_open',
                'weight_recent': 0.7
            }
        elif 14 <= hour <= 16:  # Tarde ativa
            return {
                'minutes': 120,
                'reason': 'afternoon_session',
                'weight_recent': 0.6
            }
        elif hour >= 17:  # Fechamento
            return {
                'minutes': 180,
                'reason': 'market_close',
                'weight_recent': 0.5
            }
        else:  # Outros horários
            return {
                'minutes': 90,
                'reason': 'normal_hours',
                'weight_recent': 0.6
            }
    
    def _prepare_features_for_model(self, enhanced_features: Dict) -> Dict:
        """
        Prepara features para formato esperado pelo modelo
        """
        # Features que o modelo espera (baseado no sistema original)
        model_features = {}
        
        # Mapear features enhanced para features do modelo
        feature_mapping = {
            # Features básicas
            'last_price': 'last_price',
            'price_change_pct': 'price_change_pct',
            'total_volume': 'total_volume',
            
            # Features enhanced para nomes esperados pelo modelo
            'volume_pattern_score': 'volume_pattern',
            'historical_momentum_rank': 'momentum_rank',
            'microstructure_imbalance': 'order_imbalance',
            'momentum_percentile': 'momentum_strength',
            'price_action_quality': 'trend_quality'
        }
        
        for enhanced_name, model_name in feature_mapping.items():
            if enhanced_name in enhanced_features:
                model_features[model_name] = enhanced_features[enhanced_name]
        
        # Adicionar features derivadas
        if 'volatility_regime' in enhanced_features:
            regime = enhanced_features['volatility_regime']
            model_features['volatility_high'] = 1 if regime.get('regime') == 'HIGH_VOL' else 0
            model_features['volatility_low'] = 1 if regime.get('regime') == 'LOW_VOL' else 0
            model_features['volatility_percentile'] = regime.get('percentile', 0.5)
        
        # Garantir que todas features necessárias estejam presentes
        self._fill_missing_features(model_features)
        
        return model_features
    
    def _fill_missing_features(self, features: Dict):
        """
        Preenche features faltantes com valores padrão
        """
        # Lista de features obrigatórias e seus valores padrão
        required_features = {
            'last_price': 0,
            'price_change_pct': 0,
            'total_volume': 0,
            'volume_pattern': 0.5,
            'momentum_rank': 0.5,
            'order_imbalance': 0,
            'momentum_strength': 0.5,
            'trend_quality': 0.5,
            'volatility_high': 0,
            'volatility_low': 0,
            'volatility_percentile': 0.5
        }
        
        for feature, default_value in required_features.items():
            if feature not in features:
                features[feature] = default_value
    
    def _adjust_confidence(self, base_confidence: float, enhanced_features: Dict) -> float:
        """
        Ajusta confiança baseado na qualidade dos dados e contexto
        """
        adjusted = base_confidence
        
        # Ajustar por qualidade dos dados
        data_quality = enhanced_features.get('data_quality', 1.0)
        adjusted *= (0.7 + 0.3 * data_quality)  # Mínimo 70% se qualidade ruim
        
        # Ajustar por quantidade de dados
        data_points = enhanced_features.get('data_points', 0)
        if data_points < 100:
            adjusted *= 0.8  # Reduzir 20% se poucos dados
        elif data_points < 500:
            adjusted *= 0.9  # Reduzir 10% se dados moderados
        
        # Ajustar por regime de volatilidade
        regime = enhanced_features.get('volatility_regime', {})
        if regime.get('regime') == 'HIGH_VOL':
            adjusted *= 0.85  # Reduzir 15% em alta volatilidade
        
        # Ajustar por sazonalidade
        seasonality = enhanced_features.get('intraday_seasonality', 0.5)
        if seasonality < 0.3:  # Período de baixa atividade
            adjusted *= 0.9
        
        return np.clip(adjusted, 0.1, 0.95)
    
    def _update_latency_stats(self, is_fast_mode: bool, latency: float):
        """
        Atualiza estatísticas de latência
        """
        if is_fast_mode:
            # Média móvel exponencial
            alpha = 0.1
            self.stats['avg_latency_fast'] = (
                alpha * latency + 
                (1 - alpha) * self.stats['avg_latency_fast']
            )
        else:
            alpha = 0.1
            self.stats['avg_latency_enhanced'] = (
                alpha * latency + 
                (1 - alpha) * self.stats['avg_latency_enhanced']
            )
    
    def _cleanup_cache(self):
        """
        Remove entries antigas do cache
        """
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, (cached_time, _) in self.enhanced_features_cache.items():
            if (current_time - cached_time).seconds > self.cache_ttl * 2:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.enhanced_features_cache[key]
    
    def get_stats(self) -> Dict:
        """
        Retorna estatísticas do coordenador
        """
        total_calls = self.stats['fast_mode_calls'] + self.stats['time_travel_calls']
        
        return {
            'total_predictions': self.stats['total_predictions'],
            'fast_mode_calls': self.stats['fast_mode_calls'],
            'time_travel_calls': self.stats['time_travel_calls'],
            'fast_mode_percentage': (self.stats['fast_mode_calls'] / total_calls * 100) if total_calls > 0 else 0,
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': (self.stats['cache_hits'] / self.stats['time_travel_calls'] * 100) if self.stats['time_travel_calls'] > 0 else 0,
            'avg_latency_fast_ms': self.stats['avg_latency_fast'] * 1000,
            'avg_latency_enhanced_ms': self.stats['avg_latency_enhanced'] * 1000,
            'latency_improvement': f"{(1 - self.stats['avg_latency_fast'] / self.stats['avg_latency_enhanced']) * 100:.1f}%" if self.stats['avg_latency_enhanced'] > 0 else "N/A"
        }
    
    def reset_stats(self):
        """
        Reseta estatísticas
        """
        self.stats = {
            'fast_mode_calls': 0,
            'time_travel_calls': 0,
            'cache_hits': 0,
            'total_predictions': 0,
            'avg_latency_fast': 0,
            'avg_latency_enhanced': 0
        }