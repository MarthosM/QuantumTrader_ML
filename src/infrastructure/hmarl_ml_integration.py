"""
Integração HMARL com Sistema ML
Conecta o sistema de ML existente com a infraestrutura HMARL de análise de fluxo
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import pandas as pd
import numpy as np
import zmq
import orjson

from src.infrastructure.zmq_valkey_flow_setup import TradingInfrastructureWithFlow
from src.coordination.flow_aware_coordinator import FlowAwareCoordinator
from src.agents.flow_aware_base_agent import FlowAwareBaseAgent
from src.features.book_features import BookFeatureEngineer
from src.ml_coordinator import MLCoordinator
from src.signal_generator import SignalGenerator


class HMARLMLBridge:
    """
    Ponte entre sistema ML existente e infraestrutura HMARL
    Permite que o sistema atual se beneficie da análise de fluxo
    """
    
    def __init__(self, ml_system, hmarl_config: Dict[str, Any]):
        self.ml_system = ml_system
        self.config = hmarl_config
        self.logger = logging.getLogger('HMARLMLBridge')
        
        # Componentes HMARL
        self.infrastructure = None
        self.flow_coordinator = None
        self.book_engineer = BookFeatureEngineer()
        
        # Estado
        self.is_running = False
        self.flow_features_cache = {}
        self.last_flow_analysis = {}
        
        # Threads
        self.flow_consumer_thread = None
        
        # Callbacks originais do sistema ML
        self.original_callbacks = {}
        
    def initialize(self) -> bool:
        """Inicializa componentes HMARL"""
        try:
            # Criar infraestrutura
            self.infrastructure = TradingInfrastructureWithFlow(self.config)
            self.infrastructure.initialize()
            
            # Criar coordenador
            valkey_config = self.config.get('valkey', {})
            self.flow_coordinator = FlowAwareCoordinator(valkey_config)
            
            # Interceptar callbacks do sistema ML
            self._intercept_ml_callbacks()
            
            # Iniciar thread consumidora
            self.is_running = True
            self.flow_consumer_thread = threading.Thread(
                target=self._flow_consumer_loop,
                daemon=True
            )
            self.flow_consumer_thread.start()
            
            self.logger.info("✅ HMARL-ML Bridge inicializado com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao inicializar HMARL-ML Bridge: {e}")
            return False
            
    def _intercept_ml_callbacks(self):
        """Intercepta callbacks do sistema ML para adicionar análise de fluxo"""
        # Salvar callbacks originais
        if hasattr(self.ml_system, 'connection'):
            conn = self.ml_system.connection
            
            # Interceptar callback de tick
            if hasattr(conn, '_tick_callback'):
                self.original_callbacks['tick'] = conn._tick_callback
                conn._tick_callback = self._enhanced_tick_callback
                
            # Interceptar callback de book
            if hasattr(conn, '_offer_book_callback'):
                self.original_callbacks['offer_book'] = conn._offer_book_callback
                conn._offer_book_callback = self._enhanced_book_callback
                
    def _enhanced_tick_callback(self, tick_data: Dict):
        """Callback de tick aprimorado com análise de fluxo"""
        # Chamar callback original
        if 'tick' in self.original_callbacks:
            self.original_callbacks['tick'](tick_data)
            
        # Publicar para análise de fluxo HMARL
        if self.infrastructure:
            try:
                flow_analysis = self.infrastructure.publish_tick_with_flow(tick_data)
                
                # Cachear análise de fluxo
                symbol = tick_data.get('symbol', 'UNKNOWN')
                self.flow_features_cache[symbol] = flow_analysis
                self.last_flow_analysis[symbol] = time.time()
                
            except Exception as e:
                self.logger.error(f"Erro na análise de fluxo: {e}")
                
    def _enhanced_book_callback(self, book_data: Dict):
        """Callback de book aprimorado"""
        # Chamar callback original
        if 'offer_book' in self.original_callbacks:
            self.original_callbacks['offer_book'](book_data)
            
        # Publicar book para HMARL
        if self.infrastructure:
            try:
                topic = f"book_{book_data.get('symbol', 'UNKNOWN')}"
                self.infrastructure.publish_data(topic, book_data)
                
            except Exception as e:
                self.logger.error(f"Erro ao publicar book: {e}")
                
    def enhance_ml_features(self, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Aprimora features do ML com análise de fluxo HMARL
        
        Args:
            features: DataFrame com features originais
            symbol: Símbolo sendo analisado
            
        Returns:
            DataFrame com features aprimoradas
        """
        enhanced = features.copy()
        
        # Adicionar flow features se disponíveis
        if symbol in self.flow_features_cache:
            flow_data = self.flow_features_cache[symbol]
            
            # Features de OFI
            for window in [1, 5, 15]:
                col_name = f'hmarl_ofi_{window}m'
                enhanced[col_name] = flow_data.get(f'ofi_{window}', 0)
                
            # Features de fluxo
            enhanced['hmarl_volume_imbalance'] = flow_data.get('volume_imbalance', 0)
            enhanced['hmarl_aggression_ratio'] = flow_data.get('aggression_ratio', 0)
            enhanced['hmarl_large_trade_ratio'] = flow_data.get('large_trade_ratio', 0)
            
            # Features de liquidez
            enhanced['hmarl_liquidity_score'] = flow_data.get('liquidity_score', 0)
            enhanced['hmarl_spread_quality'] = flow_data.get('spread_quality', 0)
            
            # Padrões detectados
            if 'patterns' in flow_data:
                patterns = flow_data['patterns']
                enhanced['hmarl_sweep_detected'] = patterns.get('sweep', 0)
                enhanced['hmarl_iceberg_detected'] = patterns.get('iceberg', 0)
                enhanced['hmarl_accumulation'] = patterns.get('accumulation', 0)
                
        return enhanced
        
    def get_flow_consensus(self, symbol: str) -> Optional[Dict]:
        """Obtém consenso de análise de fluxo dos agentes HMARL"""
        if not self.flow_coordinator:
            return None
            
        try:
            # Coletar sinais dos agentes
            signals = self.flow_coordinator.collect_agent_signals(timeout=0.5)
            
            # Filtrar por símbolo
            symbol_signals = [s for s in signals if s.get('symbol') == symbol]
            
            if not symbol_signals:
                return None
                
            # Construir consenso
            consensus = self.flow_coordinator.flow_consensus_builder.build(symbol_signals)
            
            return consensus
            
        except Exception as e:
            self.logger.error(f"Erro ao obter consenso de fluxo: {e}")
            return None
            
    def enhance_trading_signal(self, original_signal: Dict, symbol: str) -> Dict:
        """
        Aprimora sinal de trading com análise de fluxo
        
        Args:
            original_signal: Sinal original do sistema ML
            symbol: Símbolo
            
        Returns:
            Sinal aprimorado com contexto de fluxo
        """
        enhanced_signal = original_signal.copy()
        
        # Obter consenso de fluxo
        flow_consensus = self.get_flow_consensus(symbol)
        
        if flow_consensus:
            # Ajustar confiança baseado em alinhamento com fluxo
            original_action = original_signal.get('action', 'hold')
            flow_direction = flow_consensus.get('direction', 'neutral')
            flow_strength = flow_consensus.get('strength', 0)
            
            # Alinhamento perfeito aumenta confiança
            if (original_action == 'buy' and flow_direction == 'bullish') or \
               (original_action == 'sell' and flow_direction == 'bearish'):
                confidence_boost = min(flow_strength * 0.2, 0.3)
                enhanced_signal['confidence'] = min(
                    enhanced_signal.get('confidence', 0.5) + confidence_boost,
                    1.0
                )
                enhanced_signal['flow_aligned'] = True
                
            # Desalinhamento reduz confiança
            elif (original_action == 'buy' and flow_direction == 'bearish') or \
                 (original_action == 'sell' and flow_direction == 'bullish'):
                confidence_penalty = min(flow_strength * 0.3, 0.4)
                enhanced_signal['confidence'] = max(
                    enhanced_signal.get('confidence', 0.5) - confidence_penalty,
                    0.1
                )
                enhanced_signal['flow_aligned'] = False
                
            # Adicionar contexto de fluxo
            enhanced_signal['flow_analysis'] = {
                'consensus': flow_consensus,
                'timestamp': datetime.now().isoformat()
            }
            
        return enhanced_signal
        
    def _flow_consumer_loop(self):
        """Loop para consumir análises de fluxo"""
        context = zmq.Context()
        
        # Subscribers para diferentes streams
        subscribers = {
            'flow': self._create_subscriber(context, 5557, b'flow_'),
            'footprint': self._create_subscriber(context, 5558, b'footprint_'),
            'liquidity': self._create_subscriber(context, 5559, b'liquidity_')
        }
        
        poller = zmq.Poller()
        for sub in subscribers.values():
            poller.register(sub, zmq.POLLIN)
            
        self.logger.info("Flow consumer loop iniciado")
        
        while self.is_running:
            try:
                socks = dict(poller.poll(100))
                
                for stream_type, sub in subscribers.items():
                    if sub in socks:
                        topic, data = sub.recv_multipart()
                        flow_data = orjson.loads(data)
                        
                        # Processar dados de fluxo
                        self._process_flow_stream(stream_type, flow_data)
                        
            except Exception as e:
                self.logger.error(f"Erro no flow consumer: {e}")
                
        # Cleanup
        for sub in subscribers.values():
            sub.close()
        context.term()
        
    def _create_subscriber(self, context: zmq.Context, port: int, filter: bytes) -> zmq.Socket:
        """Cria subscriber ZMQ"""
        sub = context.socket(zmq.SUB)
        sub.connect(f"tcp://localhost:{port}")
        sub.setsockopt(zmq.SUBSCRIBE, filter)
        return sub
        
    def _process_flow_stream(self, stream_type: str, data: Dict):
        """Processa dados de stream de fluxo"""
        symbol = data.get('symbol', 'UNKNOWN')
        
        # Atualizar cache específico do stream
        if stream_type == 'flow':
            self.flow_features_cache[symbol] = data.get('analysis', {})
        elif stream_type == 'footprint':
            if symbol not in self.flow_features_cache:
                self.flow_features_cache[symbol] = {}
            self.flow_features_cache[symbol]['footprint'] = data
        elif stream_type == 'liquidity':
            if symbol not in self.flow_features_cache:
                self.flow_features_cache[symbol] = {}
            self.flow_features_cache[symbol]['liquidity'] = data
            
    def get_enhanced_market_state(self, symbol: str) -> Dict:
        """Obtém estado de mercado aprimorado com análise de fluxo"""
        state = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'ml_state': {},
            'flow_state': {},
            'microstructure': {}
        }
        
        # Estado do sistema ML
        if hasattr(self.ml_system, 'get_market_state'):
            state['ml_state'] = self.ml_system.get_market_state(symbol)
            
        # Estado de fluxo HMARL
        if symbol in self.flow_features_cache:
            state['flow_state'] = self.flow_features_cache[symbol]
            
        # Microestrutura do book
        if hasattr(self.ml_system, 'data_structure'):
            book_data = self.ml_system.data_structure.get_book_data(symbol)
            if not book_data.empty:
                book_features = self.book_engineer.calculate_spread_features(book_data)
                state['microstructure'] = book_features.iloc[-1].to_dict()
                
        return state
        
    def shutdown(self):
        """Desliga a ponte HMARL-ML"""
        self.logger.info("Desligando HMARL-ML Bridge...")
        
        self.is_running = False
        
        # Restaurar callbacks originais
        if hasattr(self.ml_system, 'connection'):
            conn = self.ml_system.connection
            for callback_type, callback_func in self.original_callbacks.items():
                if callback_type == 'tick':
                    conn._tick_callback = callback_func
                elif callback_type == 'offer_book':
                    conn._offer_book_callback = callback_func
                    
        # Desligar infraestrutura
        if self.infrastructure:
            self.infrastructure.stop()
            
        # Aguardar threads
        if self.flow_consumer_thread:
            self.flow_consumer_thread.join(timeout=5)
            
        self.logger.info("HMARL-ML Bridge desligado")


def integrate_hmarl_with_ml_system(ml_system, hmarl_config: Optional[Dict] = None) -> HMARLMLBridge:
    """
    Função helper para integrar HMARL com sistema ML existente
    
    Args:
        ml_system: Sistema ML existente (TradingSystem)
        hmarl_config: Configuração HMARL (opcional)
        
    Returns:
        HMARLMLBridge configurado
    """
    if not hmarl_config:
        hmarl_config = {
            'symbol': 'WDOU25',
            'zmq': {
                'tick_port': 5555,
                'book_port': 5556,
                'flow_port': 5557,
                'footprint_port': 5558,
                'liquidity_port': 5559,
                'tape_port': 5560
            },
            'valkey': {
                'host': 'localhost',
                'port': 6379,
                'stream_maxlen': 100000,
                'ttl_days': 30
            }
        }
        
    # Criar ponte
    bridge = HMARLMLBridge(ml_system, hmarl_config)
    
    # Inicializar
    if bridge.initialize():
        # Modificar sistema ML para usar features aprimoradas
        if hasattr(ml_system, 'feature_engine'):
            original_calculate = ml_system.feature_engine.calculate_features
            
            def enhanced_calculate(data, symbol=None):
                # Calcular features originais
                features = original_calculate(data)
                
                # Aprimorar com HMARL
                if symbol:
                    features = bridge.enhance_ml_features(features, symbol)
                    
                return features
                
            ml_system.feature_engine.calculate_features = enhanced_calculate
            
        # Modificar gerador de sinais
        if hasattr(ml_system, 'signal_generator'):
            original_generate = ml_system.signal_generator.generate_signal
            
            def enhanced_generate(prediction, symbol=None):
                # Gerar sinal original
                signal = original_generate(prediction)
                
                # Aprimorar com fluxo
                if symbol:
                    signal = bridge.enhance_trading_signal(signal, symbol)
                    
                return signal
                
            ml_system.signal_generator.generate_signal = enhanced_generate
            
    return bridge


# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Simular sistema ML
    class MockMLSystem:
        def __init__(self):
            self.connection = type('obj', (object,), {
                '_tick_callback': lambda x: print(f"Tick: {x}"),
                '_offer_book_callback': lambda x: print(f"Book: {x}")
            })
            
    ml_system = MockMLSystem()
    
    # Integrar HMARL
    bridge = integrate_hmarl_with_ml_system(ml_system)
    
    print("HMARL-ML Bridge ativo. Pressione Ctrl+C para sair...")
    
    try:
        while True:
            time.sleep(1)
            
            # Obter estado aprimorado
            state = bridge.get_enhanced_market_state('WDOU25')
            print(f"\nEstado de mercado: {state}")
            
    except KeyboardInterrupt:
        bridge.shutdown()