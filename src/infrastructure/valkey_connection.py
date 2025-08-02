"""
Valkey Connection Manager
Gerenciador de conexão com Valkey para persistência de dados HMARL
"""

import redis
import logging
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import orjson


class ValkeyConnectionManager:
    """Gerenciador de conexão com Valkey"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, password: Optional[str] = None,
                 decode_responses: bool = False):
        self.logger = logging.getLogger(f"{__name__}.ValkeyConnection")
        
        # Configuração de conexão
        self.connection_params = {
            'host': host,
            'port': port,
            'db': db,
            'decode_responses': decode_responses,
            'socket_timeout': 5,
            'socket_connect_timeout': 5,
            'retry_on_timeout': True,
            'health_check_interval': 30
        }
        
        if password:
            self.connection_params['password'] = password
            
        # Cliente Redis/Valkey
        self.client = None
        self.connected = False
        
        # Prefixos para organização
        self.prefixes = {
            'decision': 'hmarl:decision:',
            'feedback': 'hmarl:feedback:',
            'agent': 'hmarl:agent:',
            'performance': 'hmarl:performance:',
            'pattern': 'hmarl:pattern:',
            'flow_state': 'hmarl:flow_state:',
            'signal': 'hmarl:signal:',
            'metrics': 'hmarl:metrics:'
        }
        
        # TTLs padrão (em segundos)
        self.ttls = {
            'decision': 86400,      # 24 horas
            'feedback': 604800,     # 7 dias
            'flow_state': 3600,     # 1 hora
            'signal': 7200,         # 2 horas
            'metrics': 86400        # 24 horas
        }
        
    def connect(self) -> bool:
        """Estabelece conexão com Valkey"""
        try:
            self.client = redis.Redis(**self.connection_params)
            
            # Testar conexão
            self.client.ping()
            self.connected = True
            
            self.logger.info(f"Conectado ao Valkey em {self.connection_params['host']}:{self.connection_params['port']}")
            
            # Criar índices se necessário
            self._setup_indexes()
            
            return True
            
        except redis.ConnectionError as e:
            self.logger.error(f"Erro ao conectar com Valkey: {e}")
            self.connected = False
            return False
        except Exception as e:
            self.logger.error(f"Erro inesperado ao conectar: {e}")
            self.connected = False
            return False
            
    def disconnect(self):
        """Desconecta do Valkey"""
        if self.client:
            self.client.close()
            self.connected = False
            self.logger.info("Desconectado do Valkey")
            
    def is_connected(self) -> bool:
        """Verifica se está conectado"""
        if not self.client or not self.connected:
            return False
            
        try:
            self.client.ping()
            return True
        except:
            self.connected = False
            return False
    
    def ping(self) -> bool:
        """Verifica se o Valkey está respondendo"""
        try:
            if not self.connected:
                self.connect()
            
            if self.client:
                return self.client.ping()
            return False
        except:
            return False
            
    def _setup_indexes(self):
        """Configura índices para busca eficiente"""
        try:
            # Criar sets para indexação
            # Exemplo: agentes por tipo
            self.logger.info("Índices configurados no Valkey")
        except Exception as e:
            self.logger.error(f"Erro configurando índices: {e}")
            
    # ========== Métodos para Decisões ==========
    
    def store_decision(self, decision: Dict) -> bool:
        """Armazena uma decisão"""
        try:
            decision_id = decision.get('decision_id')
            if not decision_id:
                self.logger.error("Decisão sem ID")
                return False
                
            key = f"{self.prefixes['decision']}{decision_id}"
            
            # Adicionar timestamp se não existir
            if 'timestamp' not in decision:
                decision['timestamp'] = time.time()
                
            # Serializar e armazenar
            value = orjson.dumps(decision)
            self.client.setex(key, self.ttls['decision'], value)
            
            # Adicionar ao índice por agente
            agent_id = decision.get('agent_id')
            if agent_id:
                agent_key = f"{self.prefixes['agent']}{agent_id}:decisions"
                self.client.zadd(agent_key, {decision_id: decision['timestamp']})
                
            # Stream para decisões recentes
            stream_key = f"{self.prefixes['decision']}stream"
            self.client.xadd(stream_key, decision, maxlen=10000)
            
            self.logger.debug(f"Decisão {decision_id} armazenada")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro armazenando decisão: {e}")
            return False
            
    def get_decision(self, decision_id: str) -> Optional[Dict]:
        """Recupera uma decisão"""
        try:
            key = f"{self.prefixes['decision']}{decision_id}"
            value = self.client.get(key)
            
            if value:
                return orjson.loads(value)
            return None
            
        except Exception as e:
            self.logger.error(f"Erro recuperando decisão: {e}")
            return None
            
    def get_recent_decisions(self, limit: int = 100) -> List[Dict]:
        """Recupera decisões recentes"""
        try:
            stream_key = f"{self.prefixes['decision']}stream"
            
            # Ler do stream
            messages = self.client.xrevrange(stream_key, count=limit)
            
            decisions = []
            for msg_id, data in messages:
                # Converter bytes para dict
                decision = {k.decode(): v.decode() if isinstance(v, bytes) else v 
                           for k, v in data.items()}
                decision['stream_id'] = msg_id.decode()
                decisions.append(decision)
                
            return decisions
            
        except Exception as e:
            self.logger.error(f"Erro recuperando decisões recentes: {e}")
            return []
            
    # ========== Métodos para Feedback ==========
    
    def store_feedback(self, feedback: Dict) -> bool:
        """Armazena feedback de execução"""
        try:
            decision_id = feedback.get('decision_id')
            if not decision_id:
                self.logger.error("Feedback sem decision_id")
                return False
                
            key = f"{self.prefixes['feedback']}{decision_id}"
            
            # Adicionar timestamp
            if 'timestamp' not in feedback:
                feedback['timestamp'] = time.time()
                
            # Serializar e armazenar
            value = orjson.dumps(feedback)
            self.client.setex(key, self.ttls['feedback'], value)
            
            # Atualizar métricas do agente
            agent_id = feedback.get('agent_id')
            if agent_id:
                self._update_agent_metrics(agent_id, feedback)
                
            # Stream para feedback recente
            stream_key = f"{self.prefixes['feedback']}stream"
            self.client.xadd(stream_key, feedback, maxlen=10000)
            
            self.logger.debug(f"Feedback para decisão {decision_id} armazenado")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro armazenando feedback: {e}")
            return False
            
    def get_feedback(self, decision_id: str) -> Optional[Dict]:
        """Recupera feedback de uma decisão"""
        try:
            key = f"{self.prefixes['feedback']}{decision_id}"
            value = self.client.get(key)
            
            if value:
                return orjson.loads(value)
            return None
            
        except Exception as e:
            self.logger.error(f"Erro recuperando feedback: {e}")
            return None
            
    # ========== Métodos para Performance ==========
    
    def _update_agent_metrics(self, agent_id: str, feedback: Dict):
        """Atualiza métricas de performance do agente"""
        try:
            key = f"{self.prefixes['performance']}{agent_id}"
            
            # Buscar métricas existentes
            existing = self.client.get(key)
            if existing:
                metrics = orjson.loads(existing)
            else:
                metrics = {
                    'total_decisions': 0,
                    'successful_decisions': 0,
                    'total_reward': 0.0,
                    'avg_confidence': 0.0,
                    'last_update': time.time()
                }
                
            # Atualizar métricas
            metrics['total_decisions'] += 1
            
            if feedback.get('profitable', False):
                metrics['successful_decisions'] += 1
                
            reward = feedback.get('reward', 0)
            metrics['total_reward'] += reward
            
            # Calcular taxa de sucesso
            if metrics['total_decisions'] > 0:
                metrics['success_rate'] = metrics['successful_decisions'] / metrics['total_decisions']
                
            metrics['last_update'] = time.time()
            
            # Salvar
            self.client.setex(key, self.ttls['metrics'], orjson.dumps(metrics))
            
        except Exception as e:
            self.logger.error(f"Erro atualizando métricas do agente: {e}")
            
    def get_agent_performance(self, agent_id: str) -> Optional[Dict]:
        """Recupera métricas de performance do agente"""
        try:
            key = f"{self.prefixes['performance']}{agent_id}"
            value = self.client.get(key)
            
            if value:
                return orjson.loads(value)
            return None
            
        except Exception as e:
            self.logger.error(f"Erro recuperando performance: {e}")
            return None
            
    # ========== Métodos para Padrões ==========
    
    def store_pattern(self, pattern: Dict) -> bool:
        """Armazena padrão aprendido"""
        try:
            pattern_id = pattern.get('pattern_id')
            if not pattern_id:
                pattern_id = f"pattern_{int(time.time()*1000)}"
                pattern['pattern_id'] = pattern_id
                
            key = f"{self.prefixes['pattern']}{pattern_id}"
            
            # Serializar e armazenar
            value = orjson.dumps(pattern)
            self.client.set(key, value)  # Padrões não expiram
            
            # Indexar por tipo
            pattern_type = pattern.get('type', 'unknown')
            index_key = f"{self.prefixes['pattern']}index:{pattern_type}"
            self.client.sadd(index_key, pattern_id)
            
            self.logger.debug(f"Padrão {pattern_id} armazenado")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro armazenando padrão: {e}")
            return False
            
    def get_patterns_by_type(self, pattern_type: str) -> List[Dict]:
        """Recupera padrões por tipo"""
        try:
            index_key = f"{self.prefixes['pattern']}index:{pattern_type}"
            pattern_ids = self.client.smembers(index_key)
            
            patterns = []
            for pattern_id in pattern_ids:
                key = f"{self.prefixes['pattern']}{pattern_id.decode()}"
                value = self.client.get(key)
                if value:
                    patterns.append(orjson.loads(value))
                    
            return patterns
            
        except Exception as e:
            self.logger.error(f"Erro recuperando padrões: {e}")
            return []
            
    # ========== Métodos para Flow State ==========
    
    def store_flow_state(self, symbol: str, flow_state: Dict) -> bool:
        """Armazena estado de fluxo"""
        try:
            key = f"{self.prefixes['flow_state']}{symbol}"
            
            # Adicionar timestamp
            flow_state['timestamp'] = time.time()
            
            # Serializar e armazenar
            value = orjson.dumps(flow_state)
            self.client.setex(key, self.ttls['flow_state'], value)
            
            # Histórico em time series
            ts_key = f"{self.prefixes['flow_state']}{symbol}:history"
            self.client.zadd(ts_key, {value: flow_state['timestamp']})
            
            # Limitar histórico
            self.client.zremrangebyrank(ts_key, 0, -1001)  # Manter últimos 1000
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro armazenando flow state: {e}")
            return False
            
    def get_flow_state(self, symbol: str, timestamp: Optional[float] = None) -> Optional[Dict]:
        """Recupera estado de fluxo"""
        try:
            if timestamp:
                # Buscar histórico próximo ao timestamp
                ts_key = f"{self.prefixes['flow_state']}{symbol}:history"
                
                # Buscar por score (timestamp)
                results = self.client.zrangebyscore(
                    ts_key, 
                    timestamp - 60,  # 1 minuto antes
                    timestamp + 60,  # 1 minuto depois
                    withscores=True,
                    num=1
                )
                
                if results:
                    value, score = results[0]
                    return orjson.loads(value)
            else:
                # Buscar estado atual
                key = f"{self.prefixes['flow_state']}{symbol}"
                value = self.client.get(key)
                
                if value:
                    return orjson.loads(value)
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Erro recuperando flow state: {e}")
            return None
            
    # ========== Métodos para Sinais ==========
    
    def publish_signal(self, channel: str, signal: Dict) -> bool:
        """Publica sinal em canal pub/sub"""
        try:
            # Serializar sinal
            message = orjson.dumps(signal)
            
            # Publicar
            subscribers = self.client.publish(channel, message)
            
            self.logger.debug(f"Sinal publicado em {channel} para {subscribers} subscribers")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro publicando sinal: {e}")
            return False
            
    def subscribe_signals(self, channels: List[str]) -> 'redis.client.PubSub':
        """Inscreve em canais de sinais"""
        try:
            pubsub = self.client.pubsub()
            pubsub.subscribe(*channels)
            
            self.logger.info(f"Inscrito em canais: {channels}")
            return pubsub
            
        except Exception as e:
            self.logger.error(f"Erro inscrevendo em canais: {e}")
            return None
            
    # ========== Métodos Utilitários ==========
    
    def health_check(self) -> Dict[str, Any]:
        """Verifica saúde da conexão e estatísticas"""
        try:
            # Ping
            start_time = time.time()
            self.client.ping()
            ping_time = (time.time() - start_time) * 1000  # ms
            
            # Info do servidor
            info = self.client.info()
            
            # Estatísticas de uso
            stats = {
                'connected': True,
                'ping_ms': round(ping_time, 2),
                'used_memory_mb': round(info.get('used_memory', 0) / 1024 / 1024, 2),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace': {}
            }
            
            # Contar chaves por tipo
            for prefix_name, prefix in self.prefixes.items():
                pattern = f"{prefix}*"
                count = len(list(self.client.scan_iter(match=pattern, count=1000)))
                stats['keyspace'][prefix_name] = count
                
            return stats
            
        except Exception as e:
            self.logger.error(f"Erro no health check: {e}")
            return {
                'connected': False,
                'error': str(e)
            }
            
    def cleanup_old_data(self, days: int = 30):
        """Limpa dados antigos"""
        try:
            cutoff_time = time.time() - (days * 86400)
            deleted_count = 0
            
            # Limpar streams antigos
            for prefix in ['decision', 'feedback', 'signal']:
                stream_key = f"{self.prefixes[prefix]}stream"
                
                # Buscar IDs antigos
                old_messages = self.client.xrange(
                    stream_key,
                    min='-',
                    max=f'{int(cutoff_time * 1000)}-0',
                    count=1000
                )
                
                if old_messages:
                    # Deletar mensagens antigas
                    for msg_id, _ in old_messages:
                        self.client.xdel(stream_key, msg_id)
                        deleted_count += 1
                        
            self.logger.info(f"Limpeza concluída: {deleted_count} registros removidos")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Erro na limpeza: {e}")
            return 0


# Singleton para facilitar uso
_valkey_connection = None


def get_valkey_connection() -> ValkeyConnectionManager:
    """Retorna conexão singleton com Valkey"""
    global _valkey_connection
    
    if _valkey_connection is None:
        _valkey_connection = ValkeyConnectionManager()
        _valkey_connection.connect()
        
    return _valkey_connection


if __name__ == "__main__":
    # Teste de conexão
    logging.basicConfig(level=logging.INFO)
    
    print("Testando conexão com Valkey...")
    
    # Criar conexão
    valkey = ValkeyConnectionManager()
    
    if valkey.connect():
        print("✅ Conectado com sucesso!")
        
        # Teste de armazenamento
        test_decision = {
            'decision_id': 'test_001',
            'agent_id': 'test_agent',
            'action': 'buy',
            'confidence': 0.75
        }
        
        if valkey.store_decision(test_decision):
            print("✅ Decisão armazenada")
            
            # Recuperar
            retrieved = valkey.get_decision('test_001')
            print(f"✅ Decisão recuperada: {retrieved}")
            
        # Health check
        health = valkey.health_check()
        print(f"\nHealth Check:")
        print(f"  Conectado: {health['connected']}")
        print(f"  Ping: {health.get('ping_ms', 'N/A')}ms")
        print(f"  Memória: {health.get('used_memory_mb', 'N/A')}MB")
        print(f"  Keyspace: {health.get('keyspace', {})}")
        
        valkey.disconnect()
    else:
        print("❌ Falha na conexão")