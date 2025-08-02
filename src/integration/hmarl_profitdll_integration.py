"""
Integração HMARL com dados reais do ProfitDLL v4
Conecta o sistema multi-agente com fluxo de dados do mercado real
"""

import logging
import time
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from threading import Thread, Event, Lock
from queue import Queue, Empty
import zmq
import numpy as np

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

class HMARLProfitDLLIntegration:
    """
    Integra o sistema HMARL com dados reais do ProfitDLL
    Processa dados de mercado e alimenta agentes de análise de fluxo
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('HMARLProfitDLLIntegration')
        
        # Componentes do sistema principal
        self.connection_manager = None
        self.data_structure = TradingDataStructure()
        self.feature_engine = FeatureEngine(self.data_structure)
        self.tech_indicators = TechnicalIndicators()
        self.ml_features = MLFeatures()
        
        # Componentes HMARL
        self.infrastructure = TradingInfrastructureWithFlow(config)
        # Inicializar infraestrutura para conectar ao Valkey
        if not self.infrastructure.initialize():
            raise RuntimeError("Falha ao inicializar infraestrutura HMARL")
        
        # Configuração do Valkey
        valkey_config = config.get('valkey', {
            'host': 'localhost',
            'port': 6379
        })
        
        self.flow_feature_system = FlowFeatureSystem(self.infrastructure.valkey_client)
        self.flow_coordinator = FlowAwareCoordinator(valkey_config)
        
        # Agentes especializados
        self.agents = {}
        self._initialize_agents()
        
        # Estado e controle
        self.is_running = False
        self.stop_event = Event()
        self.data_queue = Queue(maxsize=1000)
        self.flow_data_lock = Lock()
        
        # Publishers ZMQ
        self.zmq_context = zmq.Context()
        self.tick_publisher = None
        self.flow_publisher = None
        self.footprint_publisher = None
        
        # Métricas
        self.metrics = {
            'trades_processed': 0,
            'features_calculated': 0,
            'agent_signals': 0,
            'errors': 0
        }
        
        self.logger.info("🚀 HMARL-ProfitDLL Integration inicializada")
    
    def _initialize_agents(self):
        """Inicializa agentes especializados em fluxo"""
        try:
            # Criar agentes
            self.agents['order_flow'] = OrderFlowSpecialistAgent({
                'ofi_threshold': 0.3,
                'delta_threshold': 1000,
                'aggression_threshold': 0.6
            })
            
            self.agents['footprint'] = FootprintPatternAgent({
                'pattern_confidence_threshold': 0.7
            })
            
            # Registrar agentes na infraestrutura
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
        Conecta ao ConnectionManager existente e registra callbacks
        """
        try:
            self.connection_manager = connection_manager
            
            # Registrar callback para receber trades
            self.connection_manager.register_trade_callback(self._on_trade_data)
            
            # Inicializar publishers ZMQ
            self._setup_zmq_publishers()
            
            self.logger.info("✅ Conectado ao ProfitDLL - callbacks registrados")
            
        except Exception as e:
            self.logger.error(f"Erro conectando ao ProfitDLL: {e}")
            raise
    
    def _setup_zmq_publishers(self):
        """Configura publishers ZMQ para distribuir dados"""
        try:
            # Tick data publisher
            self.tick_publisher = self.zmq_context.socket(zmq.PUB)
            self.tick_publisher.bind("tcp://*:5555")
            
            # Flow data publisher
            self.flow_publisher = self.zmq_context.socket(zmq.PUB)
            self.flow_publisher.bind("tcp://*:5557")
            
            # Footprint publisher
            self.footprint_publisher = self.zmq_context.socket(zmq.PUB)
            self.footprint_publisher.bind("tcp://*:5558")
            
            self.logger.info("✅ Publishers ZMQ configurados")
            
        except Exception as e:
            self.logger.error(f"Erro configurando ZMQ: {e}")
            raise
    
    def _on_trade_data(self, trade_data: Dict):
        """
        Callback chamado quando novos dados de trade chegam do ProfitDLL
        """
        try:
            # Ignorar eventos especiais
            if trade_data.get('event_type') == 'historical_data_complete':
                self.logger.info("📊 Dados históricos completos - iniciando processamento em tempo real")
                return
            
            # Adicionar à fila para processamento
            if not self.data_queue.full():
                self.data_queue.put(trade_data)
                self.metrics['trades_processed'] += 1
            else:
                self.logger.warning("Fila de dados cheia - descartando trade")
                
        except Exception as e:
            self.logger.error(f"Erro no callback de trade: {e}")
            self.metrics['errors'] += 1
    
    def start(self):
        """Inicia o sistema de processamento HMARL"""
        try:
            self.is_running = True
            self.stop_event.clear()
            
            # Thread de processamento de dados
            data_thread = Thread(target=self._data_processing_loop, name="HMARL-DataProcessor")
            data_thread.daemon = True
            data_thread.start()
            
            # Thread de análise de fluxo
            flow_thread = Thread(target=self._flow_analysis_loop, name="HMARL-FlowAnalyzer")
            flow_thread.daemon = True
            flow_thread.start()
            
            # Thread de coordenação de agentes
            coord_thread = Thread(target=self._agent_coordination_loop, name="HMARL-Coordinator")
            coord_thread.daemon = True
            coord_thread.start()
            
            # Iniciar agentes
            for agent in self.agents.values():
                agent_thread = Thread(
                    target=agent.run_enhanced_agent_loop,
                    name=f"HMARL-Agent-{agent.agent_type}"
                )
                agent_thread.daemon = True
                agent_thread.start()
            
            self.logger.info("🚀 Sistema HMARL iniciado com sucesso!")
            
        except Exception as e:
            self.logger.error(f"Erro iniciando sistema HMARL: {e}")
            self.stop()
            raise
    
    def _data_processing_loop(self):
        """Loop principal de processamento de dados de mercado"""
        self.logger.info("🔄 Loop de processamento de dados iniciado")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Pegar dados da fila (timeout de 1 segundo)
                trade_data = self.data_queue.get(timeout=1.0)
                
                # Processar e enriquecer dados
                enriched_data = self._process_trade_data(trade_data)
                
                if enriched_data:
                    # Publicar via ZMQ
                    self._publish_market_data(enriched_data)
                    
                    # Armazenar no Valkey
                    self._store_in_valkey(enriched_data)
                    
            except Empty:
                # Normal - sem dados novos
                continue
            except Exception as e:
                self.logger.error(f"Erro no loop de processamento: {e}")
                self.metrics['errors'] += 1
                time.sleep(0.1)
    
    def _process_trade_data(self, trade_data: Dict) -> Optional[Dict]:
        """
        Processa dados de trade e calcula features básicas
        """
        try:
            ticker = trade_data.get('ticker', '')
            
            # Atualizar estrutura de dados
            # TODO: Implementar update_from_trade em TradingDataStructure
            # self.data_structure.update_from_trade(trade_data)
            
            # Calcular indicadores técnicos básicos
            ohlc_data = self.data_structure.get_ohlc_data(ticker, timeframe='1min')
            
            if ohlc_data is None or len(ohlc_data) < 20:
                return None
            
            # Calcular features técnicas
            tech_features = {}
            
            # RSI
            rsi = self.tech_indicators.calculate_rsi(ohlc_data['close'])
            if rsi is not None and len(rsi) > 0:
                tech_features['rsi'] = rsi.iloc[-1]
            
            # Volume
            tech_features['volume'] = trade_data.get('volume', 0)
            tech_features['volume_ma'] = ohlc_data['volume'].rolling(10).mean().iloc[-1]
            
            # Criar estrutura enriquecida
            enriched_data = {
                'timestamp': trade_data['timestamp'],
                'ticker': ticker,
                'price': trade_data['price'],
                'volume': trade_data['volume'],
                'quantity': trade_data.get('quantity', 0),
                'trade_type': trade_data.get('trade_type', 0),
                'features': tech_features,
                'raw_trade': trade_data
            }
            
            self.metrics['features_calculated'] += 1
            
            return enriched_data
            
        except Exception as e:
            self.logger.error(f"Erro processando trade data: {e}")
            return None
    
    def _publish_market_data(self, data: Dict):
        """Publica dados de mercado via ZMQ"""
        try:
            # Publicar tick data
            if self.tick_publisher:
                tick_msg = {
                    'topic': f"tick_{data['ticker']}",
                    'data': {
                        'timestamp': data['timestamp'].isoformat(),
                        'ticker': data['ticker'],
                        'price': data['price'],
                        'volume': data['volume']
                    }
                }
                self.tick_publisher.send_json(tick_msg)
            
            # Publicar flow data
            if self.flow_publisher and 'features' in data:
                flow_msg = {
                    'topic': f"flow_{data['ticker']}",
                    'data': {
                        'timestamp': data['timestamp'].isoformat(),
                        'ticker': data['ticker'],
                        'order_flow_imbalance': self._calculate_ofi(data),
                        'aggression_score': self._calculate_aggression(data),
                        'features': data['features']
                    }
                }
                self.flow_publisher.send_json(flow_msg)
                
        except Exception as e:
            self.logger.error(f"Erro publicando dados ZMQ: {e}")
    
    def _store_in_valkey(self, data: Dict):
        """Armazena dados no Valkey para análise histórica"""
        try:
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
            
            # Stream de features
            if 'features' in data:
                self.infrastructure.valkey_client.xadd(
                    f'features:{data["ticker"]}',
                    {
                        'timestamp': str(data['timestamp']),
                        'features': json.dumps(data['features'])
                    },
                    maxlen=10000
                )
                
        except Exception as e:
            self.logger.error(f"Erro armazenando no Valkey: {e}")
    
    def _flow_analysis_loop(self):
        """Loop de análise de fluxo de ordens"""
        self.logger.info("🔄 Loop de análise de fluxo iniciado")
        
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
                time.sleep(1)
    
    def _calculate_flow_features(self) -> Optional[Dict]:
        """Calcula features avançadas de fluxo"""
        try:
            # Por enquanto, implementação simplificada
            # TODO: Integrar com FlowFeatureSystem completo
            
            ticker = self.config.get('ticker', 'WDOQ25')
            
            # Buscar dados recentes do Valkey
            recent_trades = self.infrastructure.valkey_client.xrange(
                f'market_data:{ticker}',
                '-',
                '+',
                count=100
            )
            
            if not recent_trades:
                return None
            
            # Calcular métricas básicas de fluxo
            buy_volume = 0
            sell_volume = 0
            total_volume = 0
            
            for trade_id, fields in recent_trades:
                volume = float(fields.get(b'volume', 0))
                total_volume += volume
                # Simplificado - seria necessário análise mais sofisticada
                buy_volume += volume * 0.5
                sell_volume += volume * 0.5
            
            flow_features = {
                'timestamp': datetime.now(),
                'ticker': ticker,
                'order_flow_imbalance': (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0,
                'total_volume': total_volume,
                'trade_count': len(recent_trades)
            }
            
            return flow_features
            
        except Exception as e:
            self.logger.error(f"Erro calculando flow features: {e}")
            return None
    
    def _publish_flow_analysis(self, flow_features: Dict):
        """Publica análise de fluxo para os agentes"""
        try:
            if self.footprint_publisher:
                msg = {
                    'topic': f"footprint_{flow_features['ticker']}",
                    'data': flow_features
                }
                self.footprint_publisher.send_json(msg)
                
        except Exception as e:
            self.logger.error(f"Erro publicando análise de fluxo: {e}")
    
    def _agent_coordination_loop(self):
        """Loop de coordenação entre agentes"""
        self.logger.info("🔄 Loop de coordenação de agentes iniciado")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Coordenar decisões a cada 10 segundos
                time.sleep(10)
                
                # Coletar sinais dos agentes
                best_strategy = self.flow_coordinator.coordinate_with_flow_analysis()
                
                if best_strategy:
                    self.logger.info(f"📊 Estratégia selecionada: {best_strategy}")
                    self.metrics['agent_signals'] += 1
                    
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
                time.sleep(1)
    
    def _calculate_ofi(self, data: Dict) -> float:
        """Calcula Order Flow Imbalance simplificado"""
        # Implementação simplificada - seria mais complexa em produção
        return 0.0
    
    def _calculate_aggression(self, data: Dict) -> float:
        """Calcula score de agressão"""
        # Implementação simplificada
        return 0.5
    
    def get_metrics(self) -> Dict:
        """Retorna métricas do sistema"""
        return {
            **self.metrics,
            'is_running': self.is_running,
            'queue_size': self.data_queue.qsize(),
            'active_agents': len(self.agents),
            'timestamp': datetime.now().isoformat()
        }
    
    def stop(self):
        """Para o sistema HMARL"""
        try:
            self.logger.info("🛑 Parando sistema HMARL...")
            
            self.is_running = False
            self.stop_event.set()
            
            # Parar agentes
            for agent in self.agents.values():
                agent.is_active = False
            
            # Fechar conexões ZMQ
            if self.tick_publisher:
                self.tick_publisher.close()
            if self.flow_publisher:
                self.flow_publisher.close()
            if self.footprint_publisher:
                self.footprint_publisher.close()
            
            self.zmq_context.term()
            
            self.logger.info("✅ Sistema HMARL parado com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro parando sistema: {e}")