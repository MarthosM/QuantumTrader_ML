"""
ZeroMQ Consumers - Consumidores de dados para o sistema HMARL
Permite que componentes consumam dados de fluxo sem modificar código existente
"""

import zmq
import orjson
import logging
import threading
import time
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
from collections import deque
from abc import ABC, abstractmethod


class BaseZMQConsumer(ABC):
    """Classe base para consumidores ZMQ"""
    
    def __init__(self, topic: str, port: int, callback: Optional[Callable] = None):
        self.topic = topic
        self.port = port
        self.callback = callback
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # ZeroMQ setup
        self.context = zmq.Context()
        self.subscriber = None
        
        # Control
        self.running = False
        self.consumer_thread = None
        
        # Buffer
        self.buffer = deque(maxlen=1000)
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'errors': 0
        }
    
    def connect(self) -> bool:
        """Conecta ao publisher ZMQ"""
        try:
            self.subscriber = self.context.socket(zmq.SUB)
            self.subscriber.connect(f"tcp://localhost:{self.port}")
            self.subscriber.setsockopt_string(zmq.SUBSCRIBE, self.topic)
            
            # Configurações de socket
            self.subscriber.setsockopt(zmq.RCVHWM, 0)  # High water mark
            self.subscriber.setsockopt(zmq.RCVTIMEO, 100)  # Timeout 100ms
            
            self.logger.info(f"📡 Conectado ao tópico '{self.topic}' na porta {self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro conectando: {e}")
            return False
    
    def start(self):
        """Inicia consumo em thread separada"""
        if self.running:
            self.logger.warning("Consumer já está rodando")
            return
        
        if not self.subscriber:
            if not self.connect():
                return
        
        self.running = True
        self.consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        self.consumer_thread.start()
        
        self.logger.info("▶️ Consumer iniciado")
    
    def stop(self):
        """Para o consumer"""
        self.running = False
        
        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=2)
        
        if self.subscriber:
            self.subscriber.close()
            
        self.context.term()
        self.logger.info("⏹️ Consumer parado")
    
    def _consumer_loop(self):
        """Loop principal do consumer"""
        while self.running:
            try:
                # Receber mensagem
                topic, message = self.subscriber.recv_multipart()
                
                # Decodificar
                data = orjson.loads(message)
                
                self.stats['messages_received'] += 1
                
                # Processar
                processed_data = self.process_message(data)
                
                # Adicionar ao buffer
                self.buffer.append(processed_data)
                
                # Callback se definido
                if self.callback:
                    self.callback(processed_data)
                
                self.stats['messages_processed'] += 1
                
            except zmq.Again:
                # Timeout normal, continuar
                continue
                
            except Exception as e:
                self.stats['errors'] += 1
                if self.running:  # Só logar se ainda estiver rodando
                    self.logger.error(f"Erro no consumer loop: {e}")
    
    @abstractmethod
    def process_message(self, data: Dict) -> Dict:
        """Processa mensagem recebida (deve ser implementado)"""
        pass
    
    def get_latest(self, n: int = 10) -> List[Dict]:
        """Retorna últimas n mensagens do buffer"""
        return list(self.buffer)[-n:]
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do consumer"""
        return self.stats.copy()


class FlowConsumer(BaseZMQConsumer):
    """Consumer especializado para dados de fluxo"""
    
    def __init__(self, symbol: str, callback: Optional[Callable] = None):
        super().__init__(f"flow_{symbol}", 5557, callback)
        self.symbol = symbol
        
        # Agregações de fluxo
        self.flow_aggregations = {
            '1m': deque(maxlen=60),
            '5m': deque(maxlen=300),
            '15m': deque(maxlen=900)
        }
    
    def process_message(self, data: Dict) -> Dict:
        """Processa mensagem de fluxo"""
        
        # Adicionar timestamp de recebimento
        data['received_at'] = datetime.now().isoformat()
        
        # Agregar dados
        self._update_aggregations(data)
        
        # Adicionar agregações ao retorno
        data['aggregations'] = self._get_current_aggregations()
        
        return data
    
    def _update_aggregations(self, data: Dict):
        """Atualiza agregações de fluxo"""
        
        if 'flow_point' in data:
            flow_point = data['flow_point']
            
            # Adicionar a todas as janelas
            for window in self.flow_aggregations.values():
                window.append(flow_point)
    
    def _get_current_aggregations(self) -> Dict:
        """Calcula agregações atuais"""
        
        aggregations = {}
        
        for window_name, window_data in self.flow_aggregations.items():
            if len(window_data) > 0:
                # Calcular métricas agregadas
                buy_volume = sum(fp['volume'] for fp in window_data if fp['trade_type'] == 2)
                sell_volume = sum(fp['volume'] for fp in window_data if fp['trade_type'] == 3)
                total_volume = buy_volume + sell_volume
                
                aggregations[window_name] = {
                    'buy_volume': buy_volume,
                    'sell_volume': sell_volume,
                    'total_volume': total_volume,
                    'imbalance': (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0,
                    'trade_count': len(window_data)
                }
        
        return aggregations
    
    def get_ofi(self, window: str = '5m') -> float:
        """Retorna Order Flow Imbalance para janela específica"""
        
        if window in self.flow_aggregations and len(self.flow_aggregations[window]) > 0:
            agg = self._get_current_aggregations()
            if window in agg:
                return agg[window]['imbalance']
        
        return 0.0


class TapeConsumer(BaseZMQConsumer):
    """Consumer para padrões de tape reading"""
    
    def __init__(self, symbol: str, callback: Optional[Callable] = None):
        super().__init__(f"tape_{symbol}", 5560, callback)
        self.symbol = symbol
        
        # Armazenar padrões detectados
        self.patterns = deque(maxlen=100)
        self.pattern_counts = {}
    
    def process_message(self, data: Dict) -> Dict:
        """Processa padrão de tape"""
        
        # Adicionar ao histórico de padrões
        self.patterns.append(data)
        
        # Contar tipos de padrões
        pattern_type = data.get('pattern_type', 'unknown')
        self.pattern_counts[pattern_type] = self.pattern_counts.get(pattern_type, 0) + 1
        
        # Adicionar contexto
        data['pattern_frequency'] = self._calculate_pattern_frequency(pattern_type)
        
        return data
    
    def _calculate_pattern_frequency(self, pattern_type: str) -> float:
        """Calcula frequência de um tipo de padrão"""
        
        if len(self.patterns) == 0:
            return 0.0
        
        count = sum(1 for p in self.patterns if p.get('pattern_type') == pattern_type)
        return count / len(self.patterns)
    
    def get_recent_patterns(self, pattern_type: Optional[str] = None) -> List[Dict]:
        """Retorna padrões recentes, opcionalmente filtrados por tipo"""
        
        if pattern_type:
            return [p for p in self.patterns if p.get('pattern_type') == pattern_type]
        
        return list(self.patterns)


class LiquidityConsumer(BaseZMQConsumer):
    """Consumer para dados de liquidez"""
    
    def __init__(self, symbol: str, callback: Optional[Callable] = None):
        super().__init__(f"liquidity_{symbol}", 5559, callback)
        self.symbol = symbol
        
        # Histórico de liquidez
        self.liquidity_history = deque(maxlen=500)
        self.liquidity_alerts = []
    
    def process_message(self, data: Dict) -> Dict:
        """Processa dados de liquidez"""
        
        # Adicionar ao histórico
        self.liquidity_history.append(data)
        
        # Detectar mudanças significativas
        alerts = self._detect_liquidity_changes(data)
        if alerts:
            self.liquidity_alerts.extend(alerts)
            data['alerts'] = alerts
        
        return data
    
    def _detect_liquidity_changes(self, current_data: Dict) -> List[Dict]:
        """Detecta mudanças significativas na liquidez"""
        
        alerts = []
        
        if len(self.liquidity_history) < 10:
            return alerts
        
        # Comparar com média recente
        recent = list(self.liquidity_history)[-10:]
        
        if 'liquidity_score' in current_data:
            avg_score = sum(d.get('liquidity_score', 0) for d in recent) / len(recent)
            current_score = current_data['liquidity_score']
            
            # Alerta se liquidez cair mais de 30%
            if current_score < avg_score * 0.7:
                alerts.append({
                    'type': 'low_liquidity',
                    'severity': 'high',
                    'message': f"Liquidez caiu {(1 - current_score/avg_score)*100:.1f}%",
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def get_liquidity_profile(self) -> Dict:
        """Retorna perfil atual de liquidez"""
        
        if len(self.liquidity_history) == 0:
            return {}
        
        recent = list(self.liquidity_history)[-50:]
        
        scores = [d.get('liquidity_score', 0) for d in recent if 'liquidity_score' in d]
        spreads = [d.get('spread', 0) for d in recent if 'spread' in d]
        
        profile = {
            'avg_liquidity_score': sum(scores) / len(scores) if scores else 0,
            'min_liquidity_score': min(scores) if scores else 0,
            'max_liquidity_score': max(scores) if scores else 0,
            'avg_spread': sum(spreads) / len(spreads) if spreads else 0,
            'recent_alerts': self.liquidity_alerts[-10:]
        }
        
        return profile


class MultiStreamConsumer:
    """Consumer que agrega múltiplos streams"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = logging.getLogger(__name__)
        
        # Consumers individuais
        self.consumers = {
            'flow': FlowConsumer(symbol),
            'tape': TapeConsumer(symbol),
            'liquidity': LiquidityConsumer(symbol)
        }
        
        # Dados agregados
        self.aggregated_data = {}
        self.last_update = {}
    
    def start(self):
        """Inicia todos os consumers"""
        
        for name, consumer in self.consumers.items():
            # Definir callback para agregar dados
            consumer.callback = lambda data, n=name: self._update_aggregated_data(n, data)
            consumer.start()
            
        self.logger.info(f"🚀 MultiStreamConsumer iniciado para {self.symbol}")
    
    def stop(self):
        """Para todos os consumers"""
        
        for consumer in self.consumers.values():
            consumer.stop()
            
        self.logger.info("⏹️ MultiStreamConsumer parado")
    
    def _update_aggregated_data(self, stream_name: str, data: Dict):
        """Atualiza dados agregados"""
        
        self.aggregated_data[stream_name] = data
        self.last_update[stream_name] = datetime.now()
    
    def get_unified_view(self) -> Dict:
        """Retorna visão unificada de todos os streams"""
        
        unified = {
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'streams': {}
        }
        
        # Adicionar dados de cada stream
        for stream_name, data in self.aggregated_data.items():
            unified['streams'][stream_name] = {
                'data': data,
                'last_update': self.last_update.get(stream_name, '').isoformat() if stream_name in self.last_update else None
            }
        
        # Adicionar métricas específicas
        if 'flow' in self.consumers:
            unified['ofi_5m'] = self.consumers['flow'].get_ofi('5m')
        
        if 'liquidity' in self.consumers:
            unified['liquidity_profile'] = self.consumers['liquidity'].get_liquidity_profile()
        
        if 'tape' in self.consumers:
            unified['recent_patterns'] = self.consumers['tape'].get_recent_patterns()
        
        return unified
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas de todos os consumers"""
        
        stats = {}
        for name, consumer in self.consumers.items():
            stats[name] = consumer.get_stats()
        
        return stats


# Exemplo de uso com callbacks personalizados
class FlowAlertSystem:
    """Sistema de alertas baseado em fluxo"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = logging.getLogger(__name__)
        
        # Configurar consumers com callbacks
        self.flow_consumer = FlowConsumer(symbol, callback=self.on_flow_update)
        self.tape_consumer = TapeConsumer(symbol, callback=self.on_tape_pattern)
        
        # Estado para alertas
        self.alert_thresholds = {
            'ofi_extreme': 0.7,  # OFI > 70% ou < -70%
            'large_trade_spike': 0.3,  # 30% de trades grandes
            'tape_acceleration': 10.0  # 10 trades/segundo
        }
    
    def start(self):
        """Inicia sistema de alertas"""
        self.flow_consumer.start()
        self.tape_consumer.start()
        self.logger.info("🚨 Sistema de alertas iniciado")
    
    def stop(self):
        """Para sistema de alertas"""
        self.flow_consumer.stop()
        self.tape_consumer.stop()
    
    def on_flow_update(self, flow_data: Dict):
        """Callback para atualizações de fluxo"""
        
        if 'analysis' in flow_data:
            analysis = flow_data['analysis']
            
            # Verificar OFI extremo
            if 'ofi' in analysis:
                ofi_5m = analysis['ofi'].get(5, 0)
                if abs(ofi_5m) > self.alert_thresholds['ofi_extreme']:
                    self._trigger_alert('ofi_extreme', {
                        'value': ofi_5m,
                        'direction': 'bullish' if ofi_5m > 0 else 'bearish',
                        'timestamp': flow_data.get('timestamp')
                    })
            
            # Verificar spike de trades grandes
            if 'large_trade_ratio' in analysis:
                if analysis['large_trade_ratio'] > self.alert_thresholds['large_trade_spike']:
                    self._trigger_alert('large_trade_spike', {
                        'value': analysis['large_trade_ratio'],
                        'timestamp': flow_data.get('timestamp')
                    })
    
    def on_tape_pattern(self, pattern_data: Dict):
        """Callback para padrões de tape"""
        
        pattern_type = pattern_data.get('pattern_type')
        
        if pattern_type == 'sweep':
            self._trigger_alert('sweep_detected', pattern_data)
        
        elif pattern_type == 'iceberg':
            self._trigger_alert('iceberg_detected', pattern_data)
    
    def _trigger_alert(self, alert_type: str, data: Dict):
        """Dispara alerta"""
        
        alert = {
            'type': alert_type,
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol
        }
        
        self.logger.warning(f"🚨 ALERTA: {alert_type} - {data}")
        
        # Aqui você pode adicionar outras ações:
        # - Enviar para sistema de notificações
        # - Ajustar parâmetros de trading
        # - Registrar em banco de dados
        
        return alert


# Exemplo de uso
"""
# Consumer simples de fluxo
flow_consumer = FlowConsumer('WDOH25')
flow_consumer.start()

# Aguardar dados
time.sleep(5)

# Obter últimos dados
latest_flow = flow_consumer.get_latest(10)
current_ofi = flow_consumer.get_ofi('5m')

# Consumer multi-stream
multi_consumer = MultiStreamConsumer('WDOH25')
multi_consumer.start()

# Obter visão unificada
unified = multi_consumer.get_unified_view()

# Sistema de alertas
alert_system = FlowAlertSystem('WDOH25')
alert_system.start()
"""