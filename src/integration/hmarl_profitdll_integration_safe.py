"""
Integração HMARL com ProfitDLL usando Buffer Thread-Safe
Versão corrigida que previne Segmentation Fault
"""

import logging
import time
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from threading import Thread, Event, Lock
from queue import Queue, Empty

# Importar buffer thread-safe e rate limiter
from .thread_safe_buffer import ThreadSafeBuffer, BufferedData, IsolatedProcessor
from .rate_limiter import AdaptiveRateLimiter

# Importar componentes do HMARL
from src.infrastructure.zmq_valkey_flow_setup import TradingInfrastructureWithFlow
from src.agents.flow_aware_base_agent import FlowAwareBaseAgent
from src.agents.order_flow_specialist import OrderFlowSpecialistAgent
from src.agents.footprint_pattern_agent import FootprintPatternAgent
from src.coordination.flow_aware_coordinator import FlowAwareCoordinator
from src.features.flow_feature_system import FlowFeatureSystem

# Importar componentes do sistema principal
from src.connection_manager_v4 import ConnectionManagerV4
from src.data_structure import TradingDataStructure
from src.feature_engine import FeatureEngine
from src.technical_indicators import TechnicalIndicators
from src.ml_features import MLFeatures


class HMARLProfitDLLIntegrationSafe:
    """
    Versão thread-safe da integração HMARL com ProfitDLL
    Usa buffer isolado para prevenir Segmentation Fault
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('HMARLProfitDLLIntegrationSafe')
        
        # Componentes do sistema principal
        self.connection_manager = None
        self.data_structure = TradingDataStructure()
        self.feature_engine = FeatureEngine(self.data_structure)
        self.tech_indicators = TechnicalIndicators()
        self.ml_features = MLFeatures()
        
        # Buffer thread-safe para isolamento
        self.buffer = ThreadSafeBuffer(max_size=10000)
        self.isolated_processor = None
        
        # Rate limiter adaptativo
        self.rate_limiter = AdaptiveRateLimiter(
            initial_rate=100,  # 100 trades/segundo inicial
            target_buffer_usage=0.7
        )
        
        # Componentes HMARL (inicializados depois)
        self.infrastructure = None
        self.flow_feature_system = None
        self.flow_coordinator = None
        self.agents = {}
        
        # Estado e controle
        self.is_running = False
        self.stop_event = Event()
        
        # Publishers ZMQ (inicializados depois)
        self.tick_publisher = None
        self.flow_publisher = None
        self.footprint_publisher = None
        
        # Métricas
        self.metrics = {
            'trades_processed': 0,
            'features_calculated': 0,
            'agent_signals': 0,
            'errors': 0,
            'buffer_drops': 0
        }
        
        # Lock para métricas
        self.metrics_lock = Lock()
        
        self.logger.info("🚀 HMARL-ProfitDLL Integration (Thread-Safe) inicializada")
    
    def initialize_hmarl(self):
        """
        Inicializa componentes HMARL em contexto isolado
        Deve ser chamado ANTES de conectar ao ProfitDLL
        """
        try:
            self.logger.info("Inicializando componentes HMARL...")
            
            # Infraestrutura HMARL
            self.infrastructure = TradingInfrastructureWithFlow(self.config)
            if not self.infrastructure.initialize():
                raise RuntimeError("Falha ao inicializar infraestrutura HMARL")
            
            # Configuração do Valkey
            valkey_config = self.config.get('valkey', {
                'host': 'localhost',
                'port': 6379
            })
            
            # Componentes de análise
            self.flow_feature_system = FlowFeatureSystem(self.infrastructure.valkey_client)
            self.flow_coordinator = FlowAwareCoordinator(valkey_config)
            
            # Inicializar agentes
            self._initialize_agents()
            
            # Criar processador isolado
            self.isolated_processor = IsolatedProcessor(self)
            
            # Registrar processadores no buffer
            self.buffer.register_processor('trade', self.isolated_processor.process_trade)
            self.buffer.register_processor('order', self.isolated_processor.process_order)
            self.buffer.register_processor('book', self.isolated_processor.process_book)
            
            # Iniciar buffer ANTES de conectar ao ProfitDLL
            self.buffer.start()
            
            self.logger.info("✅ Componentes HMARL inicializados com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro inicializando HMARL: {e}")
            raise
    
    def _initialize_agents(self):
        """Inicializa agentes especializados em fluxo"""
        try:
            # Criar agentes com registry desabilitado para evitar conflitos
            agent_config = {
                'use_registry': False,  # Importante: desabilitar registry
                'min_signal_interval': 1.0
            }
            
            # Order Flow Specialist
            self.agents['order_flow'] = OrderFlowSpecialistAgent({
                **agent_config,
                'ofi_threshold': 0.3,
                'delta_threshold': 1000,
                'aggression_threshold': 0.6
            })
            
            # Footprint Pattern
            self.agents['footprint'] = FootprintPatternAgent({
                **agent_config,
                'pattern_confidence_threshold': 0.7
            })
            
            # Registrar agentes no Valkey (sem usar registry ZMQ)
            for agent_name, agent in self.agents.items():
                self.infrastructure.valkey_client.xadd(
                    f'agent_registry:{agent_name}',
                    {
                        'agent_id': agent.agent_id,
                        'agent_type': agent.agent_type,
                        'status': 'active',
                        'timestamp': str(datetime.now())
                    }
                )
            
            self.logger.info(f"✅ {len(self.agents)} agentes HMARL inicializados")
            
        except Exception as e:
            self.logger.error(f"Erro inicializando agentes: {e}")
            raise
    
    def connect_to_profitdll(self, connection_manager: ConnectionManagerV4):
        """
        Conecta ao ConnectionManager existente com callbacks isolados
        """
        try:
            self.connection_manager = connection_manager
            
            # Registrar callback isolado para trades
            self.connection_manager.register_trade_callback(self._isolated_trade_callback)
            
            self.logger.info("✅ Conectado ao ProfitDLL com callbacks isolados")
            
        except Exception as e:
            self.logger.error(f"Erro conectando ao ProfitDLL: {e}")
            raise
    
    def _isolated_trade_callback(self, trade_data: Dict):
        """
        Callback isolado que apenas adiciona ao buffer
        NÃO processa diretamente para evitar Segmentation Fault
        """
        try:
            # Ignorar eventos especiais
            if trade_data.get('event_type') == 'historical_data_complete':
                self.logger.info("📊 Dados históricos completos")
                return
            
            # Aplicar rate limiting
            if not self.rate_limiter.should_process():
                return  # Silenciosamente descartar para evitar spam
            
            # Ajustar rate limiter baseado no uso do buffer
            buffer_usage = self.buffer.buffer.qsize() / self.buffer.buffer.maxsize
            self.rate_limiter.adjust_rate(buffer_usage)
            
            # Criar dado bufferizado
            buffered = BufferedData(
                timestamp=datetime.now(),
                data_type='trade',
                data=trade_data,
                priority=0
            )
            
            # Adicionar ao buffer (thread-safe)
            if not self.buffer.put(buffered, timeout=0.01):
                with self.metrics_lock:
                    self.metrics['buffer_drops'] += 1
            else:
                with self.metrics_lock:
                    self.metrics['trades_processed'] += 1
                    
        except Exception as e:
            self.logger.error(f"Erro no callback isolado: {e}")
            with self.metrics_lock:
                self.metrics['errors'] += 1
    
    def start(self):
        """Inicia o sistema de processamento HMARL"""
        try:
            self.is_running = True
            self.stop_event.clear()
            
            # Buffer já deve estar rodando (iniciado em initialize_hmarl)
            if not self.buffer.is_running:
                self.logger.warning("Buffer não está rodando - iniciando agora")
                self.buffer.start()
            
            # Thread de análise de fluxo (isolada)
            flow_thread = Thread(
                target=self._safe_flow_analysis_loop, 
                name="HMARL-FlowAnalyzer",
                daemon=True
            )
            flow_thread.start()
            
            # Thread de coordenação (isolada)
            coord_thread = Thread(
                target=self._safe_coordination_loop,
                name="HMARL-Coordinator", 
                daemon=True
            )
            coord_thread.start()
            
            # NÃO iniciar threads dos agentes diretamente
            # Eles serão acionados pelo coordenador quando necessário
            
            self.logger.info("🚀 Sistema HMARL iniciado em modo thread-safe!")
            
        except Exception as e:
            self.logger.error(f"Erro iniciando sistema HMARL: {e}")
            self.stop()
            raise
    
    def _safe_flow_analysis_loop(self):
        """Loop de análise de fluxo com tratamento de erros"""
        self.logger.info("🔄 Loop de análise de fluxo iniciado (thread-safe)")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Analisar fluxo a cada 5 segundos
                time.sleep(5)
                
                # Calcular features de fluxo
                flow_features = self._calculate_flow_features()
                
                if flow_features:
                    # Publicar análise de fluxo
                    self._publish_flow_analysis(flow_features)
                    
            except Exception as e:
                self.logger.error(f"Erro na análise de fluxo: {e}")
                self.logger.error(traceback.format_exc())
                time.sleep(1)
    
    def _safe_coordination_loop(self):
        """Loop de coordenação com tratamento de erros"""
        self.logger.info("🔄 Loop de coordenação iniciado (thread-safe)")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Coordenar decisões a cada 10 segundos
                time.sleep(10)
                
                # Processar sinais dos agentes de forma isolada
                for agent_name, agent in self.agents.items():
                    try:
                        # Gerar sinal se condições adequadas
                        if agent._should_generate_signal():
                            signal = agent.generate_signal_with_flow(
                                agent.state['price_state'],
                                agent.state['flow_state']
                            )
                            
                            if signal and signal.get('confidence', 0) > 0.3:
                                agent._publish_enhanced_signal(signal)
                                with self.metrics_lock:
                                    self.metrics['agent_signals'] += 1
                                    
                    except Exception as e:
                        self.logger.error(f"Erro processando agente {agent_name}: {e}")
                
                # Coordenar decisões
                try:
                    best_strategy = self.flow_coordinator.coordinate_with_flow_analysis()
                    
                    if best_strategy:
                        self.logger.info(f"📊 Estratégia selecionada: {best_strategy}")
                        
                        # Armazenar decisão
                        self.infrastructure.valkey_client.xadd(
                            'agent_decisions:all',
                            {
                                'timestamp': str(datetime.now()),
                                'decision': json.dumps(best_strategy)
                            }
                        )
                        
                except Exception as e:
                    self.logger.error(f"Erro na coordenação: {e}")
                    
            except Exception as e:
                self.logger.error(f"Erro no loop de coordenação: {e}")
                self.logger.error(traceback.format_exc())
                time.sleep(1)
    
    def _process_trade_data(self, trade_data: Dict) -> Optional[Dict]:
        """Processa dados de trade de forma isolada"""
        try:
            ticker = trade_data.get('ticker', '')
            
            # Por enquanto, criar estrutura enriquecida simples
            enriched_data = {
                'timestamp': trade_data.get('timestamp', datetime.now()),
                'ticker': ticker,
                'price': trade_data['price'],
                'volume': trade_data['volume'],
                'quantity': trade_data.get('quantity', 0),
                'trade_type': trade_data.get('trade_type', 0),
                'features': {},
                'raw_trade': trade_data
            }
            
            with self.metrics_lock:
                self.metrics['features_calculated'] += 1
            
            return enriched_data
            
        except Exception as e:
            self.logger.error(f"Erro processando trade data: {e}")
            return None
    
    def _publish_market_data(self, data: Dict):
        """Publica dados via ZMQ de forma thread-safe"""
        try:
            # Publicar apenas se publishers existirem
            if self.infrastructure and self.infrastructure.publishers:
                self.infrastructure.publish_tick_with_flow(data)
                
        except Exception as e:
            self.logger.error(f"Erro publicando dados: {e}")
    
    def _store_in_valkey(self, data: Dict):
        """Armazena dados no Valkey de forma thread-safe"""
        try:
            if self.infrastructure and self.infrastructure.valkey_client:
                # Stream de market data
                self.infrastructure.valkey_client.xadd(
                    f'market_data:{data["ticker"]}',
                    {
                        'timestamp': str(data['timestamp']),
                        'price': data['price'],
                        'volume': data['volume'],
                        'quantity': data.get('quantity', 0)
                    },
                    maxlen=100000
                )
                
        except Exception as e:
            self.logger.error(f"Erro armazenando no Valkey: {e}")
    
    def _calculate_flow_features(self) -> Optional[Dict]:
        """Calcula features de fluxo de forma segura"""
        try:
            ticker = self.config.get('ticker', 'WDOQ25')
            
            if not self.infrastructure or not self.infrastructure.valkey_client:
                return None
            
            # Buscar dados recentes
            recent_trades = self.infrastructure.valkey_client.xrange(
                f'market_data:{ticker}',
                '-',
                '+',
                count=100
            )
            
            if not recent_trades:
                return None
            
            # Calcular métricas básicas
            buy_volume = 0
            sell_volume = 0
            total_volume = 0
            
            for trade_id, fields in recent_trades:
                volume = float(fields.get(b'volume', 0))
                total_volume += volume
                buy_volume += volume * 0.5
                sell_volume += volume * 0.5
            
            return {
                'timestamp': datetime.now(),
                'ticker': ticker,
                'order_flow_imbalance': (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0,
                'total_volume': total_volume,
                'trade_count': len(recent_trades)
            }
            
        except Exception as e:
            self.logger.error(f"Erro calculando flow features: {e}")
            return None
    
    def _publish_flow_analysis(self, flow_features: Dict):
        """Publica análise de fluxo de forma segura"""
        try:
            if self.infrastructure and self.infrastructure.publishers.get('footprint'):
                msg = {
                    'topic': f"footprint_{flow_features['ticker']}",
                    'data': flow_features
                }
                # Usar método thread-safe da infraestrutura
                self.infrastructure._publish_tape_pattern(
                    flow_features['ticker'],
                    {
                        'pattern_type': 'flow_analysis',
                        'confidence': 0.8,
                        'details': flow_features
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Erro publicando análise: {e}")
    
    def get_metrics(self) -> Dict:
        """Retorna métricas do sistema de forma thread-safe"""
        with self.metrics_lock:
            buffer_stats = self.buffer.get_stats()
            
            return {
                **self.metrics,
                'buffer_size': buffer_stats['buffer_size'],
                'buffer_efficiency': buffer_stats['efficiency'],
                'rate_limit': self.rate_limiter.max_per_second,
                'current_rate': self.rate_limiter.get_rate(),
                'is_running': self.is_running,
                'active_agents': len(self.agents),
                'timestamp': datetime.now().isoformat()
            }
    
    def stop(self):
        """Para o sistema de forma segura"""
        try:
            self.logger.info("🛑 Parando sistema HMARL...")
            
            self.is_running = False
            self.stop_event.set()
            
            # Parar buffer primeiro (importante!)
            if self.buffer:
                self.buffer.stop(timeout=3.0)
            
            # Parar agentes
            for agent in self.agents.values():
                agent.is_active = False
            
            # Parar infraestrutura
            if self.infrastructure:
                self.infrastructure.stop()
            
            self.logger.info("✅ Sistema HMARL parado com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro parando sistema: {e}")