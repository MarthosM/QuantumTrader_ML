#!/usr/bin/env python3
"""
Monitor de Fluxo de Dados ML
Sistema de Trading v2.0

Monitora e mapeia o fluxo completo de dados:
1. Candles recebidos
2. Features calculadas
3. Predições geradas
4. Resultado exibido no GUI

Este módulo garante que o processo de features → predição aconteça corretamente
a cada novo candle e exibe os resultados no monitor GUI.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque

# Adicionar src ao path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)


@dataclass
class DataFlowStep:
    """Representa um passo do fluxo de dados"""
    step_id: str
    name: str
    timestamp: datetime
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    processing_time: float = 0.0
    status: str = "pending"  # pending, processing, completed, error
    data_summary: Optional[Dict] = None
    error_message: Optional[str] = None


@dataclass
class PredictionResult:
    """Resultado de uma predição"""
    timestamp: datetime
    candle_timestamp: datetime
    direction: float
    magnitude: float
    confidence: float
    regime: str
    features_count: int
    model_used: str
    processing_time: float
    raw_features: Optional[Dict] = None


class DataFlowMonitor:
    """
    Monitor completo do fluxo de dados ML
    Rastreia: Candles → Features → Predições → GUI
    """
    
    def __init__(self, trading_system=None):
        self.logger = logging.getLogger('DataFlowMonitor')
        self.trading_system = trading_system
        
        # História do fluxo de dados
        self.flow_history = deque(maxlen=1000)
        self.prediction_history = deque(maxlen=100)
        self.error_history = deque(maxlen=50)
        
        # Estado atual
        self.current_candle = None
        self.current_features = None
        self.current_prediction = None
        
        # Configurações
        self.feature_validation_enabled = True
        self.detailed_logging = True
        
        # Thread de monitoramento
        self.monitoring_thread = None
        self.monitoring_active = False
        
    def start_monitoring(self):
        """Inicia o monitoramento do fluxo de dados"""
        if self.monitoring_active:
            self.logger.warning("Monitoramento já está ativo")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="DataFlowMonitor"
        )
        self.monitoring_thread.start()
        self.logger.info("✓ Monitor de fluxo de dados iniciado")
        
    def stop_monitoring(self):
        """Para o monitoramento"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)
        self.logger.info("Monitor de fluxo de dados parado")
        
    def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        self.logger.info("Loop de monitoramento iniciado")
        
        while self.monitoring_active:
            try:
                # Verificar se há novos dados
                self._check_for_new_data()
                
                # Aguardar próxima verificação
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Erro no loop de monitoramento: {e}")
                time.sleep(5)  # Aguardar mais tempo em caso de erro
                
    def _check_for_new_data(self):
        """Verifica por novos candles e processa se necessário"""
        if not self.trading_system:
            return
            
        try:
            # Obter dados mais recentes do sistema
            data_structure = getattr(self.trading_system, 'data_structure', None)
            if not data_structure:
                return
                
            # Verificar por novo candle
            candles_df = getattr(data_structure, 'candles_df', None)
            if candles_df is not None and not candles_df.empty:
                latest_candle = candles_df.iloc[-1]
                
                # Verificar se é um novo candle
                if self._is_new_candle(latest_candle):
                    self.logger.info(f"🕯️ Novo candle detectado: {latest_candle.name}")
                    self._process_new_candle(latest_candle, data_structure)
                    
        except Exception as e:
            self.logger.error(f"Erro verificando novos dados: {e}")
            
    def _is_new_candle(self, candle) -> bool:
        """Verifica se é um candle novo"""
        if self.current_candle is None:
            return True
            
        # Comparar timestamp ou índice
        if hasattr(candle, 'name') and hasattr(self.current_candle, 'name'):
            return candle.name != self.current_candle.name
            
        return True
        
    def _process_new_candle(self, candle, data_structure):
        """Processa um novo candle e dispara o fluxo de dados completo"""
        flow_id = f"flow_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.logger.info(f"🔄 Iniciando fluxo de dados {flow_id}")
        
        try:
            # 1. Registrar recebimento do candle
            step1 = self._create_flow_step(
                flow_id, "candle_received",
                "Candle Recebido",
                input_shape=(1,),
                data_summary=self._summarize_candle(candle)
            )
            self._update_step_status(step1, "completed")
            
            # 2. Calcular features
            step2 = self._create_flow_step(
                flow_id, "features_calculation",
                "Cálculo de Features"
            )
            
            features_df = self._calculate_features_for_candle(data_structure, step2)
            
            if features_df is not None and not features_df.empty:
                self._update_step_status(
                    step2, "completed",
                    output_shape=features_df.shape,
                    data_summary=self._summarize_features(features_df)
                )
                
                # 3. Executar predição
                step3 = self._create_flow_step(
                    flow_id, "prediction_execution",
                    "Execução da Predição"
                )
                
                prediction = self._execute_prediction(features_df, step3)
                
                if prediction:
                    self._update_step_status(step3, "completed")
                    
                    # 4. Atualizar GUI
                    step4 = self._create_flow_step(
                        flow_id, "gui_update",
                        "Atualização do GUI"
                    )
                    
                    self._update_gui_with_results(prediction, features_df, step4)
                    self._update_step_status(step4, "completed")
                    
                    # Armazenar estado atual
                    self.current_candle = candle
                    self.current_features = features_df
                    self.current_prediction = prediction
                    
                    self.logger.info(f"✅ Fluxo {flow_id} concluído com sucesso")
                    
                else:
                    self._update_step_status(step3, "error", "Falha na predição")
                    
            else:
                self._update_step_status(step2, "error", "Falha no cálculo de features")
                
        except Exception as e:
            self.logger.error(f"Erro processando fluxo {flow_id}: {e}")
            self._record_error(flow_id, str(e))
            
    def _create_flow_step(self, flow_id: str, step_id: str, name: str, **kwargs) -> DataFlowStep:
        """Cria um novo passo do fluxo"""
        step = DataFlowStep(
            step_id=f"{flow_id}_{step_id}",
            name=name,
            timestamp=datetime.now(),
            **kwargs
        )
        step.status = "processing"
        self.flow_history.append(step)
        
        if self.detailed_logging:
            self.logger.debug(f"📋 Iniciando: {name}")
            
        return step
        
    def _update_step_status(self, step: DataFlowStep, status: str, 
                           error_message: Optional[str] = None, **kwargs):
        """Atualiza status de um passo"""
        step.status = status
        step.processing_time = (datetime.now() - step.timestamp).total_seconds()
        
        if error_message:
            step.error_message = error_message
            
        for key, value in kwargs.items():
            setattr(step, key, value)
            
        if self.detailed_logging:
            status_emoji = {"completed": "✅", "error": "❌", "processing": "🔄"}
            emoji = status_emoji.get(status, "📋")
            self.logger.debug(f"{emoji} {step.name}: {status} ({step.processing_time:.3f}s)")
            
    def _calculate_features_for_candle(self, data_structure, step: DataFlowStep) -> Optional[pd.DataFrame]:
        """Calcula features para o candle atual"""
        try:
            # Verificar se temos feature_engine disponível
            feature_engine = getattr(self.trading_system, 'feature_engine', None)
            if not feature_engine:
                step.error_message = "FeatureEngine não disponível"
                return None
                
            # Obter dados necessários
            candles_df = data_structure.candles_df
            microstructure_df = getattr(data_structure, 'microstructure_df', pd.DataFrame())
            orderbook_df = getattr(data_structure, 'orderbook_df', pd.DataFrame())
            
            if candles_df.empty:
                step.error_message = "Candles DataFrame está vazio"
                return None
                
            # Calcular indicadores primeiro
            indicators_result = feature_engine.request_indicator_calculation(
                candles_df.copy()
            )
            
            if not indicators_result['success']:
                step.error_message = f"Falha nos indicadores: {indicators_result.get('error', 'Unknown')}"
                return None
                
            indicators_df = indicators_result['data']
            
            # Calcular features ML
            features_result = feature_engine.create_features_separated(
                candles_df.copy(),
                microstructure_df.copy() if not microstructure_df.empty else pd.DataFrame(),
                indicators_df.copy()
            )
            
            if not features_result['success']:
                step.error_message = f"Falha nas features: {features_result.get('error', 'Unknown')}"
                return None
                
            features_df = features_result['features_df']
            
            # Validar features se habilitado
            if self.feature_validation_enabled:
                if not self._validate_features(features_df):
                    step.error_message = "Validação de features falhou"
                    return None
                    
            self.logger.info(f"✓ Features calculadas: {features_df.shape}")
            return features_df
            
        except Exception as e:
            step.error_message = f"Exceção no cálculo de features: {str(e)}"
            self.logger.error(f"Erro calculando features: {e}")
            return None
            
    def _execute_prediction(self, features_df: pd.DataFrame, step: DataFlowStep) -> Optional[PredictionResult]:
        """Executa predição com as features calculadas"""
        try:
            # Verificar se temos ML disponível
            ml_coordinator = getattr(self.trading_system, 'ml_coordinator', None)
            if not ml_coordinator:
                step.error_message = "MLCoordinator não disponível"
                return None
                
            # Preparar dados para predição
            prediction_data = {
                'features_df': features_df.copy(),
                'timestamp': datetime.now()
            }
            
            # Executar predição
            start_time = time.time()
            prediction_result = ml_coordinator.process_prediction_request(prediction_data)
            processing_time = time.time() - start_time
            
            if not prediction_result or 'trade_decision' not in prediction_result:
                step.error_message = "Resultado de predição inválido"
                return None
                
            # Criar objeto de resultado
            prediction = PredictionResult(
                timestamp=datetime.now(),
                candle_timestamp=features_df.index[-1] if not features_df.empty else datetime.now(),
                direction=prediction_result.get('direction', 0.0),
                magnitude=prediction_result.get('magnitude', 0.0),
                confidence=prediction_result.get('confidence', 0.0),
                regime=prediction_result.get('regime', 'unknown'),
                features_count=len(features_df.columns) if not features_df.empty else 0,
                model_used=prediction_result.get('model_used', 'unknown'),
                processing_time=processing_time,
                raw_features=self._extract_key_features(features_df)
            )
            
            # Armazenar na história
            self.prediction_history.append(prediction)
            
            self.logger.info(f"🎯 Predição executada: {prediction.direction:.3f} "
                           f"(confiança: {prediction.confidence:.3f})")
            
            return prediction
            
        except Exception as e:
            step.error_message = f"Exceção na predição: {str(e)}"
            self.logger.error(f"Erro executando predição: {e}")
            return None
            
    def _update_gui_with_results(self, prediction: PredictionResult, 
                               features_df: pd.DataFrame, step: DataFlowStep):
        """Atualiza GUI com resultados da predição"""
        try:
            # Verificar se temos GUI disponível
            monitor = getattr(self.trading_system, 'monitor', None)
            if not monitor:
                step.error_message = "Monitor GUI não disponível"
                return
                
            # Preparar dados para o GUI
            gui_data = {
                'prediction': {
                    'timestamp': prediction.timestamp.strftime('%H:%M:%S'),
                    'direction': prediction.direction,
                    'magnitude': prediction.magnitude,
                    'confidence': prediction.confidence,
                    'regime': prediction.regime,
                    'model': prediction.model_used,
                    'processing_time': f"{prediction.processing_time:.3f}s"
                },
                'features': {
                    'count': prediction.features_count,
                    'sample': prediction.raw_features or {},
                    'last_values': self._get_latest_feature_values(features_df)
                },
                'status': 'active'
            }
            
            # Atualizar dados no monitor
            if hasattr(monitor, 'update_prediction_data'):
                monitor.update_prediction_data(gui_data)
                
            # Atualizar dados de debug se disponível
            if hasattr(monitor, 'update_debug_data'):
                debug_data = {
                    'features_shape': features_df.shape,
                    'features_columns': list(features_df.columns)[:10],  # Primeiras 10
                    'last_flow_steps': [
                        {'name': step.name, 'status': step.status, 'time': step.processing_time}
                        for step in list(self.flow_history)[-5:]  # Últimos 5 passos
                    ]
                }
                monitor.update_debug_data(debug_data)
                
            self.logger.debug("✓ GUI atualizado com resultados")
            
        except Exception as e:
            step.error_message = f"Erro atualizando GUI: {str(e)}"
            self.logger.error(f"Erro atualizando GUI: {e}")
            
    def _validate_features(self, features_df: pd.DataFrame) -> bool:
        """Valida se as features estão corretas"""
        try:
            if features_df.empty:
                self.logger.warning("Features DataFrame está vazio")
                return False
                
            # Verificar valores inválidos
            inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                self.logger.warning(f"Features contêm {inf_count} valores infinitos")
                return False
                
            # Verificar muitos NaN
            nan_ratio = features_df.isnull().sum().sum() / (features_df.shape[0] * features_df.shape[1])
            if nan_ratio > 0.5:
                self.logger.warning(f"Features contêm {nan_ratio:.1%} de valores NaN")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Erro validando features: {e}")
            return False
            
    def _summarize_candle(self, candle) -> Dict:
        """Cria resumo de um candle"""
        try:
            return {
                'timestamp': str(candle.name) if hasattr(candle, 'name') else 'unknown',
                'open': float(candle.get('open', 0)),
                'high': float(candle.get('high', 0)),
                'low': float(candle.get('low', 0)),
                'close': float(candle.get('close', 0)),
                'volume': float(candle.get('volume', 0))
            }
        except Exception:
            return {'error': 'Failed to summarize candle'}
            
    def _summarize_features(self, features_df: pd.DataFrame) -> Dict:
        """Cria resumo das features"""
        try:
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            return {
                'shape': features_df.shape,
                'columns_count': len(features_df.columns),
                'numeric_columns': len(numeric_cols),
                'nan_count': features_df.isnull().sum().sum(),
                'sample_columns': list(features_df.columns)[:5]
            }
        except Exception:
            return {'error': 'Failed to summarize features'}
            
    def _extract_key_features(self, features_df: pd.DataFrame) -> Dict:
        """Extrai features-chave para exibição"""
        try:
            if features_df.empty:
                return {}
                
            last_row = features_df.iloc[-1]
            
            # Selecionar features importantes para exibição
            key_features = {}
            
            # Preços básicos
            for col in ['close', 'open', 'high', 'low', 'volume']:
                if col in last_row:
                    key_features[col] = float(last_row[col])
                    
            # Indicadores técnicos
            for col in ['ema_9', 'ema_20', 'rsi', 'atr', 'adx']:
                if col in last_row:
                    key_features[col] = float(last_row[col])
                    
            # Features de momentum
            for col in features_df.columns:
                if 'momentum' in col.lower():
                    try:
                        val = features_df[col].iloc[-1]
                        if not pd.isna(val):
                            key_features[col] = float(val)
                            break
                    except Exception:
                        continue
                    
            return key_features
            
        except Exception:
            return {'error': 'Failed to extract key features'}
            
    def _get_latest_feature_values(self, features_df: pd.DataFrame) -> Dict:
        """Obtém valores mais recentes das features"""
        try:
            if features_df.empty:
                return {}
                
            latest = features_df.iloc[-1]
            
            # Retornar amostra das features mais recentes
            result = {}
            for i, (col, val) in enumerate(latest.items()):
                if i >= 10:  # Limitar a 10 features
                    break
                if not pd.isna(val):
                    result[col] = float(val)
                    
            return result
            
        except Exception:
            return {'error': 'Failed to get latest values'}
            
    def _record_error(self, flow_id: str, error_message: str):
        """Registra erro no fluxo"""
        error_record = {
            'timestamp': datetime.now(),
            'flow_id': flow_id,
            'error': error_message
        }
        self.error_history.append(error_record)
        
    def get_flow_summary(self) -> Dict:
        """Retorna resumo do fluxo de dados"""
        try:
            recent_flows = list(self.flow_history)[-20:]  # Últimos 20 passos
            recent_predictions = list(self.prediction_history)[-10:]  # Últimas 10 predições
            recent_errors = list(self.error_history)[-5:]  # Últimos 5 erros
            
            return {
                'total_flows_processed': len(self.flow_history),
                'total_predictions': len(self.prediction_history),
                'total_errors': len(self.error_history),
                'current_status': {
                    'has_candle': self.current_candle is not None,
                    'has_features': self.current_features is not None,
                    'has_prediction': self.current_prediction is not None,
                    'monitoring_active': self.monitoring_active
                },
                'recent_flows': [
                    {
                        'name': step.name,
                        'status': step.status,
                        'processing_time': step.processing_time,
                        'timestamp': step.timestamp.strftime('%H:%M:%S')
                    }
                    for step in recent_flows
                ],
                'recent_predictions': [
                    {
                        'timestamp': pred.timestamp.strftime('%H:%M:%S'),
                        'direction': pred.direction,
                        'confidence': pred.confidence,
                        'regime': pred.regime
                    }
                    for pred in recent_predictions
                ],
                'recent_errors': [
                    {
                        'timestamp': err['timestamp'].strftime('%H:%M:%S'),
                        'flow_id': err['flow_id'],
                        'error': err['error']
                    }
                    for err in recent_errors
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Erro gerando resumo: {e}")
            return {'error': str(e)}
            
    def print_current_status(self):
        """Imprime status atual do monitor"""
        summary = self.get_flow_summary()
        
        print(f"\n{'='*60}")
        print(f"DATA FLOW MONITOR - STATUS ATUAL")
        print(f"{'='*60}")
        print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔄 Monitoramento Ativo: {summary['current_status']['monitoring_active']}")
        print(f"📊 Total de Fluxos: {summary['total_flows_processed']}")
        print(f"🎯 Total de Predições: {summary['total_predictions']}")
        print(f"❌ Total de Erros: {summary['total_errors']}")
        
        print(f"\n📋 Estado Atual:")
        print(f"  • Candle: {'✓' if summary['current_status']['has_candle'] else '✗'}")
        print(f"  • Features: {'✓' if summary['current_status']['has_features'] else '✗'}")
        print(f"  • Predição: {'✓' if summary['current_status']['has_prediction'] else '✗'}")
        
        if summary['recent_predictions']:
            print(f"\n🎯 Última Predição:")
            last_pred = summary['recent_predictions'][-1]
            print(f"  • Timestamp: {last_pred['timestamp']}")
            print(f"  • Direção: {last_pred['direction']:.3f}")
            print(f"  • Confiança: {last_pred['confidence']:.3f}")
            print(f"  • Regime: {last_pred['regime']}")
            
        if summary['recent_errors']:
            print(f"\n❌ Últimos Erros:")
            for err in summary['recent_errors'][-3:]:
                print(f"  • {err['timestamp']}: {err['error']}")
                
        print(f"{'='*60}\n")


def main():
    """Teste do monitor de fluxo de dados"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('DataFlowTest')
    logger.info("Testando Data Flow Monitor...")
    
    # Criar monitor sem sistema (para teste)
    monitor = DataFlowMonitor()
    
    # Testar métodos básicos
    logger.info("Monitor criado com sucesso")
    monitor.print_current_status()
    
    logger.info("✓ Teste do Data Flow Monitor concluído")


if __name__ == "__main__":
    main()
