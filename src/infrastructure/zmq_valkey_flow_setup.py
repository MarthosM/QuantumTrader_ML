"""
ZeroMQ + Valkey Infrastructure with Flow Analysis
Task 1.1: Setup Avançado com Streams de Fluxo
"""

import zmq
import redis
import orjson
import logging
import threading
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class FlowDataPoint:
    """Estrutura de dados para análise de fluxo"""
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    trade_type: int  # 2=buy, 3=sell
    aggressor: str  # 'buyer' or 'seller'
    trade_size_category: str  # 'small', 'medium', 'large', 'whale'
    speed_of_tape: float  # trades per second
    
    def to_dict(self) -> Dict:
        """Converte para dicionário serializável"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class TradingInfrastructureWithFlow:
    """
    Infraestrutura completa ZeroMQ + Valkey com análise de fluxo
    Compatível com o sistema atual, adiciona capacidades de flow analysis
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # ZeroMQ Context
        self.context = zmq.Context()
        
        # Publishers para diferentes tipos de dados
        self.publishers = {
            'tick': None,
            'book': None,
            'flow': None,
            'footprint': None,
            'liquidity': None,
            'tape': None
        }
        
        # Valkey connection
        self.valkey_client = None
        
        # Flow analysis components
        self.flow_analyzer = None
        self.tape_reader = None
        self.liquidity_monitor = None
        
        # Threading control
        self.running = False
        self.threads = []
        
        # Performance metrics
        self.metrics = {
            'messages_published': 0,
            'messages_stored': 0,
            'flow_events_detected': 0,
            'latency_ms': []
        }
        
    def initialize(self) -> bool:
        """Inicializa toda a infraestrutura"""
        try:
            # 1. Setup ZeroMQ publishers
            self._setup_zmq_publishers()
            
            # 2. Connect to Valkey
            self._connect_valkey()
            
            # 3. Initialize flow analysis components
            self._initialize_flow_components()
            
            # 4. Create Valkey streams
            self._create_valkey_streams()
            
            self.logger.info("✅ Infraestrutura ZeroMQ + Valkey + Flow inicializada")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro inicializando infraestrutura: {e}")
            return False
    
    def _setup_zmq_publishers(self):
        """Configura publishers ZeroMQ para todos os tipos de dados"""
        
        # Configurações de portas
        ports = {
            'tick': self.config.get('zmq', {}).get('tick_port', 5555),
            'book': self.config.get('zmq', {}).get('book_port', 5556),
            'flow': self.config.get('zmq', {}).get('flow_port', 5557),
            'footprint': self.config.get('zmq', {}).get('footprint_port', 5558),
            'liquidity': self.config.get('zmq', {}).get('liquidity_port', 5559),
            'tape': self.config.get('zmq', {}).get('tape_port', 5560)
        }
        
        for stream_type, port in ports.items():
            try:
                socket = self.context.socket(zmq.PUB)
                socket.setsockopt(zmq.SNDHWM, 0)  # High water mark ilimitado
                socket.setsockopt(zmq.LINGER, 0)  # Não bloquear ao fechar
                socket.bind(f"tcp://*:{port}")
                self.publishers[stream_type] = socket
                self.logger.info(f"📡 ZeroMQ {stream_type} publisher na porta {port}")
            except zmq.error.ZMQError as e:
                self.logger.error(f"Erro ao criar publisher {stream_type} na porta {port}: {e}")
                # Tentar fechar socket se existir
                if stream_type in self.publishers and self.publishers[stream_type]:
                    self.publishers[stream_type].close()
                    self.publishers[stream_type] = None
                raise
    
    def _connect_valkey(self):
        """Conecta ao Valkey com configurações otimizadas"""
        
        valkey_config = self.config.get('valkey', {})
        
        self.valkey_client = redis.Redis(
            host=valkey_config.get('host', 'localhost'),
            port=valkey_config.get('port', 6379),
            decode_responses=True,  # Usar strings para compatibilidade
            socket_keepalive=True,
            socket_keepalive_options={
                1: 120,  # TCP_KEEPIDLE
                2: 3,    # TCP_KEEPINTVL
                3: 5,    # TCP_KEEPCNT
            }
        )
        
        # Testar conexão
        self.valkey_client.ping()
        self.logger.info("✅ Conectado ao Valkey")
    
    def _initialize_flow_components(self):
        """Inicializa componentes de análise de fluxo"""
        
        # Flow Analysis Engine
        self.flow_analyzer = FlowAnalysisEngine(self.config)
        
        # Automated Tape Reader
        self.tape_reader = AutomatedTapeReader(self.config)
        
        # Liquidity Monitor
        self.liquidity_monitor = LiquidityMonitor(self.config)
        
        self.logger.info("🔍 Componentes de análise de fluxo inicializados")
    
    def _create_valkey_streams(self):
        """Cria streams no Valkey para armazenamento eficiente"""
        
        symbol = self.config.get('symbol', 'WDOH25')
        max_len = self.config.get('valkey', {}).get('stream_maxlen', 100000)
        
        streams = [
            f"market_data:{symbol}",
            f"order_flow:{symbol}",
            f"footprint:{symbol}",
            f"tape_reading:{symbol}",
            f"liquidity_profile:{symbol}",
            f"flow_events:{symbol}",
            f"agent_decisions:all",
            f"agent_performance:all",
            f"meta_decisions:global"
        ]
        
        for stream in streams:
            try:
                # Criar stream se não existir
                self.valkey_client.xadd(
                    stream, 
                    {"init": "true", "timestamp": datetime.now().isoformat()}, 
                    maxlen=max_len
                )
                self.logger.info(f"📊 Stream criado/verificado: {stream}")
            except Exception as e:
                self.logger.warning(f"Stream {stream} já existe ou erro: {e}")
    
    def publish_tick_with_flow(self, tick_data: Dict):
        """
        Publica tick com análise de fluxo em tempo real
        Mantém compatibilidade com sistema atual
        """
        start_time = time.time()
        
        try:
            # 1. Publicar tick normal (compatibilidade)
            tick_msg = orjson.dumps(tick_data)
            self.publishers['tick'].send_multipart([
                f"tick_{tick_data['symbol']}".encode(),
                tick_msg
            ])
            
            # 2. Análise de fluxo
            flow_data = self._analyze_tick_flow(tick_data)
            
            # 3. Publicar dados de fluxo
            if flow_data:
                flow_msg = orjson.dumps(flow_data)
                self.publishers['flow'].send_multipart([
                    f"flow_{tick_data['symbol']}".encode(),
                    flow_msg
                ])
                
                # 4. Armazenar no Valkey
                self._store_flow_data(tick_data['symbol'], flow_data)
            
            # 5. Detectar padrões de tape reading
            tape_pattern = self.tape_reader.analyze_tick(tick_data)
            if tape_pattern:
                self._publish_tape_pattern(tick_data['symbol'], tape_pattern)
            
            # Métricas
            latency = (time.time() - start_time) * 1000
            self.metrics['latency_ms'].append(latency)
            self.metrics['messages_published'] += 1
            
        except Exception as e:
            self.logger.error(f"Erro publicando tick com fluxo: {e}")
            import traceback
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
    
    def _analyze_tick_flow(self, tick_data: Dict) -> Optional[Dict]:
        """Analisa características de fluxo do tick"""
        
        try:
            # Criar FlowDataPoint
            flow_point = FlowDataPoint(
                timestamp=datetime.fromisoformat(tick_data['timestamp']),
                symbol=tick_data['symbol'],
                price=tick_data['price'],
                volume=tick_data['volume'],
                trade_type=tick_data['trade_type'],
                aggressor='buyer' if tick_data['trade_type'] == 2 else 'seller',
                trade_size_category=self._categorize_trade_size(tick_data['volume']),
                speed_of_tape=self.tape_reader.get_current_speed()
            )
            
            # Análise de fluxo
            flow_analysis = self.flow_analyzer.analyze(flow_point)
            
            return {
                'timestamp': flow_point.timestamp.isoformat(),
                'symbol': flow_point.symbol,
                'flow_point': flow_point.to_dict(),
                'analysis': flow_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Erro analisando fluxo: {e}")
            return None
    
    def _categorize_trade_size(self, volume: int) -> str:
        """Categoriza tamanho do trade"""
        
        # Configurações por símbolo (exemplo para mini-índice)
        thresholds = self.config.get('flow', {}).get('trade_size_thresholds', {
            'small': 5,
            'medium': 20,
            'large': 50,
            'whale': 100
        })
        
        if volume <= thresholds['small']:
            return 'small'
        elif volume <= thresholds['medium']:
            return 'medium'
        elif volume <= thresholds['large']:
            return 'large'
        else:
            return 'whale'
    
    def _store_flow_data(self, symbol: str, flow_data: Dict):
        """Armazena dados de fluxo no Valkey"""
        
        try:
            # Stream de fluxo
            stream_key = f"order_flow:{symbol}"
            
            # Usar timestamp como ID para ordenação
            timestamp_ms = int(datetime.fromisoformat(
                flow_data['timestamp']
            ).timestamp() * 1000)
            
            # Preparar dados para armazenamento
            store_data = {
                'timestamp': flow_data['timestamp'],
                'price': str(flow_data['flow_point']['price']),
                'volume': str(flow_data['flow_point']['volume']),
                'aggressor': flow_data['flow_point']['aggressor'],
                'category': flow_data['flow_point']['trade_size_category'],
                'speed': str(flow_data['flow_point']['speed_of_tape']),
                'analysis': orjson.dumps(flow_data['analysis']).decode()
            }
            
            # Adicionar ao stream (deixar Redis gerar ID único)
            self.valkey_client.xadd(
                stream_key,
                store_data,
                maxlen=self.config.get('valkey', {}).get('stream_maxlen', 100000)
            )
            
            self.metrics['messages_stored'] += 1
            
        except Exception as e:
            self.logger.error(f"Erro armazenando dados de fluxo: {e}")
    
    def _publish_tape_pattern(self, symbol: str, pattern: Dict):
        """Publica padrão detectado no tape reading"""
        
        try:
            # Publicar via ZeroMQ
            tape_msg = orjson.dumps(pattern)
            self.publishers['tape'].send_multipart([
                f"tape_{symbol}".encode(),
                tape_msg
            ])
            
            # Armazenar evento no Valkey
            event_stream = f"flow_events:{symbol}"
            self.valkey_client.xadd(
                event_stream,
                {
                    'type': 'tape_pattern',
                    'pattern': pattern['pattern_type'],
                    'confidence': str(pattern['confidence']),
                    'details': orjson.dumps(pattern).decode(),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            self.metrics['flow_events_detected'] += 1
            
        except Exception as e:
            self.logger.error(f"Erro publicando padrão de tape: {e}")
    
    def get_flow_history(self, symbol: str, minutes_back: int = 60) -> List[Dict]:
        """Recupera histórico de fluxo via time travel"""
        
        try:
            stream_key = f"order_flow:{symbol}"
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=minutes_back)
            
            # IDs baseados em timestamp
            start_id = f"{int(start_time.timestamp() * 1000)}-0"
            end_id = f"{int(end_time.timestamp() * 1000)}-0"
            
            # Buscar range
            entries = self.valkey_client.xrange(stream_key, start_id, end_id)
            
            # Processar entradas
            flow_history = []
            for entry_id, fields in entries:
                flow_data = fields.copy()
                
                # Parse análise JSON
                if 'analysis' in flow_data:
                    flow_data['analysis'] = orjson.loads(flow_data['analysis'])
                
                flow_data['entry_id'] = entry_id
                flow_history.append(flow_data)
            
            return flow_history
            
        except Exception as e:
            self.logger.error(f"Erro recuperando histórico de fluxo: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict:
        """Retorna métricas de performance da infraestrutura"""
        
        avg_latency = np.mean(self.metrics['latency_ms'][-1000:]) if self.metrics['latency_ms'] else 0
        
        return {
            'messages_published': self.metrics['messages_published'],
            'messages_stored': self.metrics['messages_stored'],
            'flow_events_detected': self.metrics['flow_events_detected'],
            'avg_latency_ms': round(avg_latency, 2),
            'uptime_seconds': time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }
    
    def start(self):
        """Inicia a infraestrutura"""
        self.running = True
        self.start_time = time.time()
        self.logger.info("🚀 Infraestrutura HMARL iniciada")
    
    def stop(self):
        """Para a infraestrutura de forma limpa"""
        self.running = False
        
        # Fechar sockets ZeroMQ
        for socket in self.publishers.values():
            if socket:
                socket.close()
        
        # Fechar contexto
        self.context.term()
        
        # Desconectar Valkey
        if self.valkey_client:
            self.valkey_client.close()
        
        self.logger.info("🛑 Infraestrutura HMARL parada")


class FlowAnalysisEngine:
    """Motor de análise de fluxo de ordens"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Buffers para análise
        self.recent_flows = []
        self.max_buffer_size = 1000
        
        # Configurações de análise
        self.ofi_windows = config.get('flow', {}).get('ofi_windows', [1, 5, 15])
        
    def analyze(self, flow_point: FlowDataPoint) -> Dict:
        """Analisa ponto de fluxo e retorna métricas"""
        
        # Adicionar ao buffer
        self.recent_flows.append(flow_point)
        if len(self.recent_flows) > self.max_buffer_size:
            self.recent_flows.pop(0)
        
        # Calcular métricas
        analysis = {
            'ofi': self._calculate_ofi(),
            'volume_imbalance': self._calculate_volume_imbalance(),
            'aggression_ratio': self._calculate_aggression_ratio(),
            'large_trade_ratio': self._calculate_large_trade_ratio(),
            'flow_momentum': self._calculate_flow_momentum()
        }
        
        return analysis
    
    def _calculate_ofi(self) -> Dict[str, float]:
        """Calcula Order Flow Imbalance para diferentes janelas"""
        ofi_values = {}
        
        for window in self.ofi_windows:
            cutoff_time = datetime.now() - timedelta(minutes=window)
            recent = [f for f in self.recent_flows if f.timestamp > cutoff_time]
            
            if recent:
                buy_volume = sum(f.volume for f in recent if f.trade_type == 2)
                sell_volume = sum(f.volume for f in recent if f.trade_type == 3)
                total_volume = buy_volume + sell_volume
                
                if total_volume > 0:
                    ofi_values[str(window)] = (buy_volume - sell_volume) / total_volume
                else:
                    ofi_values[str(window)] = 0.0
            else:
                ofi_values[str(window)] = 0.0
        
        return ofi_values
    
    def _calculate_volume_imbalance(self) -> float:
        """Calcula desequilíbrio de volume recente"""
        if len(self.recent_flows) < 10:
            return 0.0
        
        recent = self.recent_flows[-100:]
        buy_volume = sum(f.volume for f in recent if f.trade_type == 2)
        sell_volume = sum(f.volume for f in recent if f.trade_type == 3)
        
        total = buy_volume + sell_volume
        if total > 0:
            return (buy_volume - sell_volume) / total
        return 0.0
    
    def _calculate_aggression_ratio(self) -> float:
        """Calcula taxa de agressão (market orders)"""
        if len(self.recent_flows) < 10:
            return 0.5
        
        recent = self.recent_flows[-50:]
        aggressive_buys = sum(1 for f in recent if f.aggressor == 'buyer')
        
        if len(recent) > 0:
            return aggressive_buys / len(recent)
        return 0.5
    
    def _calculate_large_trade_ratio(self) -> float:
        """Calcula proporção de trades grandes"""
        if len(self.recent_flows) < 10:
            return 0.0
        
        recent = self.recent_flows[-100:]
        large_trades = sum(1 for f in recent if f.trade_size_category in ['large', 'whale'])
        
        if len(recent) > 0:
            return large_trades / len(recent)
        return 0.0
    
    def _calculate_flow_momentum(self) -> float:
        """Calcula momentum do fluxo"""
        if len(self.recent_flows) < 20:
            return 0.0
        
        # Comparar fluxo recente com anterior
        mid_point = len(self.recent_flows) // 2
        first_half = self.recent_flows[:mid_point]
        second_half = self.recent_flows[mid_point:]
        
        # Volume médio
        avg_vol_first = np.mean([f.volume for f in first_half])
        avg_vol_second = np.mean([f.volume for f in second_half])
        
        if avg_vol_first > 0:
            return float((avg_vol_second - avg_vol_first) / avg_vol_first)
        return 0.0


class AutomatedTapeReader:
    """Leitor automatizado de tape (time & sales)"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Buffer de trades
        self.trade_buffer = []
        self.max_buffer = 500
        
        # Métricas de velocidade
        self.trade_timestamps = []
        self.speed_window = 10  # segundos
        
    def analyze_tick(self, tick_data: Dict) -> Optional[Dict]:
        """Analisa tick para padrões de tape reading"""
        
        # Adicionar ao buffer
        self.trade_buffer.append(tick_data)
        self.trade_timestamps.append(datetime.now())
        
        # Limitar buffers
        if len(self.trade_buffer) > self.max_buffer:
            self.trade_buffer.pop(0)
        if len(self.trade_timestamps) > self.max_buffer:
            self.trade_timestamps.pop(0)
        
        # Detectar padrões
        pattern = self._detect_patterns()
        
        return pattern
    
    def get_current_speed(self) -> float:
        """Retorna velocidade atual do tape (trades/segundo)"""
        
        if len(self.trade_timestamps) < 2:
            return 0.0
        
        cutoff = datetime.now() - timedelta(seconds=self.speed_window)
        recent = [t for t in self.trade_timestamps if t > cutoff]
        
        if len(recent) >= 2:
            duration = (recent[-1] - recent[0]).total_seconds()
            if duration > 0:
                return len(recent) / duration
        
        return 0.0
    
    def _detect_patterns(self) -> Optional[Dict]:
        """Detecta padrões no tape"""
        
        if len(self.trade_buffer) < 10:
            return None
        
        recent = self.trade_buffer[-20:]
        
        # Detectar sweep (varredura rápida)
        if self._detect_sweep(recent):
            return {
                'pattern_type': 'sweep',
                'confidence': 0.8,
                'trades_analyzed': len(recent),
                'timestamp': datetime.now().isoformat()
            }
        
        # Detectar iceberg (ordens ocultas)
        if self._detect_iceberg(recent):
            return {
                'pattern_type': 'iceberg',
                'confidence': 0.7,
                'trades_analyzed': len(recent),
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    def _detect_sweep(self, trades: List[Dict]) -> bool:
        """Detecta padrão de sweep (varredura de liquidez)"""
        
        if len(trades) < 5:
            return False
        
        # Verificar se trades recentes são na mesma direção e rápidos
        last_5 = trades[-5:]
        
        # Todos na mesma direção?
        trade_types = [t['trade_type'] for t in last_5]
        if len(set(trade_types)) != 1:
            return False
        
        # Volume crescente?
        volumes = [t['volume'] for t in last_5]
        if volumes != sorted(volumes):
            return False
        
        # Velocidade alta?
        if self.get_current_speed() > 5.0:  # 5 trades/segundo
            return True
        
        return False
    
    def _detect_iceberg(self, trades: List[Dict]) -> bool:
        """Detecta padrão de iceberg (ordem oculta)"""
        
        if len(trades) < 10:
            return False
        
        # Procurar por trades repetidos no mesmo preço
        price_counts = {}
        for trade in trades:
            price = trade['price']
            if price not in price_counts:
                price_counts[price] = []
            price_counts[price].append(trade)
        
        # Verificar se algum preço tem muitos trades pequenos
        for price, price_trades in price_counts.items():
            if len(price_trades) >= 5:
                volumes = [t['volume'] for t in price_trades]
                avg_volume = np.mean(volumes)
                
                # Trades pequenos e consistentes indicam iceberg
                if avg_volume < 10 and np.std(volumes) < 2:
                    return True
        
        return False


class LiquidityMonitor:
    """Monitor de liquidez do mercado"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Estado do book
        self.current_book = {
            'bids': {},
            'asks': {}
        }
        
        # Histórico de liquidez
        self.liquidity_history = []
        self.max_history = 100
    
    def update_book(self, book_data: Dict):
        """Atualiza estado do livro de ofertas"""
        
        # Atualizar bids e asks
        if 'bids' in book_data:
            self.current_book['bids'] = book_data['bids']
        if 'asks' in book_data:
            self.current_book['asks'] = book_data['asks']
        
        # Calcular métricas de liquidez
        liquidity_metrics = self._calculate_liquidity_metrics()
        
        # Armazenar histórico
        self.liquidity_history.append({
            'timestamp': datetime.now(),
            'metrics': liquidity_metrics
        })
        
        if len(self.liquidity_history) > self.max_history:
            self.liquidity_history.pop(0)
    
    def _calculate_liquidity_metrics(self) -> Dict:
        """Calcula métricas de liquidez"""
        
        metrics = {
            'bid_depth': self._calculate_depth('bids'),
            'ask_depth': self._calculate_depth('asks'),
            'spread': self._calculate_spread(),
            'imbalance': self._calculate_book_imbalance(),
            'liquidity_score': 0.0
        }
        
        # Score geral de liquidez
        total_depth = metrics['bid_depth'] + metrics['ask_depth']
        if total_depth > 0 and metrics['spread'] > 0:
            metrics['liquidity_score'] = total_depth / metrics['spread']
        
        return metrics
    
    def _calculate_depth(self, side: str) -> float:
        """Calcula profundidade de um lado do book"""
        
        if side not in self.current_book:
            return 0.0
        
        total_volume = 0
        for price, volume in self.current_book[side].items():
            total_volume += volume
        
        return total_volume
    
    def _calculate_spread(self) -> float:
        """Calcula spread bid-ask"""
        
        if not self.current_book['bids'] or not self.current_book['asks']:
            return 0.0
        
        best_bid = max(self.current_book['bids'].keys())
        best_ask = min(self.current_book['asks'].keys())
        
        return best_ask - best_bid
    
    def _calculate_book_imbalance(self) -> float:
        """Calcula desequilíbrio do book"""
        
        bid_depth = self._calculate_depth('bids')
        ask_depth = self._calculate_depth('asks')
        
        total = bid_depth + ask_depth
        if total > 0:
            return (bid_depth - ask_depth) / total
        
        return 0.0


# Exemplo de configuração
HMARL_FLOW_CONFIG = {
    'symbol': 'WDOH25',
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
    },
    'flow': {
        'ofi_windows': [1, 5, 15, 30, 60],
        'trade_size_thresholds': {
            'small': 5,
            'medium': 20,
            'large': 50,
            'whale': 100
        },
        'min_confidence': 0.3,
        'flow_weight': 0.4
    }
}