# 🚀 Guia de Implantação ZeroMQ + Valkey - Sistema ML Trading

## 📊 Visão Geral da Integração

Este guia apresenta a implantação gradual de ZeroMQ e Valkey no sistema atual de ML Trading, **sem quebrar funcionalidades existentes**.

### 🎯 Objetivos da Integração
- **Reduzir latência**: De ~3s para <100ms no pipeline completo
- **Habilitar Time Travel**: Análise histórica instantânea
- **Escalar processamento**: Múltiplos consumidores paralelos
- **Persistir dados**: Backup automático e recuperação rápida
- **Zero Breaking Changes**: Sistema atual continua funcionando

## 🏗️ Arquitetura de Integração

### Sistema Atual (Preservado)
```
ConnectionManager → TradingDataStructure → FeatureEngine → MLCoordinator
      ↓                     ↓                    ↓              ↓
 ProfitDLL Callbacks    DataFrame Memory    Calculations    Predictions
```

### Nova Camada (Adicionada)
```
ConnectionManager → [ZMQ Publisher] → [Valkey Streams] → [Enhanced Features]
      ↓                    ↓                ↓                    ↓
 Original Flow +      Pub/Sub Layer    Time Series DB      Time Travel ML
```

## 📋 Fases de Implementação

### **FASE 1: Preparação e Setup (Dia 1-2)**

#### 1.1 Instalação de Dependências
```bash
# Adicionar ao requirements.txt
pyzmq>=25.1.0
valkey>=6.0.0
orjson>=3.9.0
```

#### 1.2 Configuração do Valkey
```bash
# Docker Compose para Valkey
# Arquivo: docker-compose.valkey.yml
version: '3.8'
services:
  valkey:
    image: valkey/valkey:latest
    container_name: ml-trading-valkey
    ports:
      - "6379:6379"
    volumes:
      - valkey-data:/data
    command: >
      valkey-server
      --maxmemory 4gb
      --maxmemory-policy allkeys-lru
      --save 60 1000
      --appendonly yes
    restart: unless-stopped

volumes:
  valkey-data:
```

#### 1.3 Arquivo de Configuração
```python
# Arquivo: src/config/zmq_valkey_config.py
import os
from pathlib import Path

class ZMQValkeyConfig:
    """Configuração centralizada para ZMQ e Valkey"""
    
    # ZeroMQ Configuration
    ZMQ_TICK_PORT = int(os.getenv('ZMQ_TICK_PORT', 5555))
    ZMQ_BOOK_PORT = int(os.getenv('ZMQ_BOOK_PORT', 5556))
    ZMQ_HISTORY_PORT = int(os.getenv('ZMQ_HISTORY_PORT', 5557))
    ZMQ_SIGNAL_PORT = int(os.getenv('ZMQ_SIGNAL_PORT', 5558))
    
    # Valkey Configuration
    VALKEY_HOST = os.getenv('VALKEY_HOST', 'localhost')
    VALKEY_PORT = int(os.getenv('VALKEY_PORT', 6379))
    VALKEY_PASSWORD = os.getenv('VALKEY_PASSWORD', None)
    VALKEY_DB = int(os.getenv('VALKEY_DB', 0))
    
    # Stream Configuration
    STREAM_MAX_LEN = 100000  # Máximo de entries por stream
    STREAM_RETENTION_DAYS = 30  # Dias de retenção
    
    # Feature Configuration
    TIME_TRAVEL_ENABLED = os.getenv('TIME_TRAVEL_ENABLED', 'true').lower() == 'true'
    TIME_TRAVEL_LOOKBACK_MINUTES = 120
    FAST_MODE_LATENCY_THRESHOLD = 0.1  # 100ms
    
    @classmethod
    def get_zmq_urls(cls):
        """Retorna URLs ZMQ configuradas"""
        return {
            'tick': f"tcp://localhost:{cls.ZMQ_TICK_PORT}",
            'book': f"tcp://localhost:{cls.ZMQ_BOOK_PORT}",
            'history': f"tcp://localhost:{cls.ZMQ_HISTORY_PORT}",
            'signal': f"tcp://localhost:{cls.ZMQ_SIGNAL_PORT}"
        }
```

### **FASE 2: Camada ZeroMQ (Dia 3-5)**

#### 2.1 ZMQ Publisher Wrapper
```python
# Arquivo: src/integration/zmq_publisher_wrapper.py
import zmq
import orjson
import logging
from typing import Dict, Any, Optional
from datetime import datetime

class ZMQPublisherWrapper:
    """
    Wrapper que adiciona publicação ZMQ aos callbacks existentes
    sem modificar o código original
    """
    
    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
        self.context = zmq.Context()
        self.publishers = {}
        self.logger = logging.getLogger('ZMQPublisher')
        
        # Criar publishers
        self._setup_publishers()
        
        # Interceptar callbacks
        self._setup_callback_interceptors()
        
    def _setup_publishers(self):
        """Cria publishers ZMQ para diferentes tipos de dados"""
        from src.config.zmq_valkey_config import ZMQValkeyConfig
        
        urls = ZMQValkeyConfig.get_zmq_urls()
        
        for data_type, url in urls.items():
            socket = self.context.socket(zmq.PUB)
            socket.bind(url)
            self.publishers[data_type] = socket
            self.logger.info(f"ZMQ Publisher {data_type} iniciado em {url}")
    
    def _setup_callback_interceptors(self):
        """Intercepta callbacks do ConnectionManager"""
        
        # Salvar callbacks originais
        original_callbacks = {}
        if hasattr(self.connection_manager, 'callbacks'):
            original_callbacks = self.connection_manager.callbacks.copy()
        
        # Interceptar newTradeCallback
        if 'newTradeCallback' in original_callbacks:
            original_trade = original_callbacks['newTradeCallback']
            
            def enhanced_trade_callback(assetId, date, tradeNumber, price, vol, qty, buyAgent, sellAgent, tradeType, bIsEdit):
                # Executar callback original
                result = original_trade(assetId, date, tradeNumber, price, vol, qty, buyAgent, sellAgent, tradeType, bIsEdit)
                
                # Publicar via ZMQ
                try:
                    tick_data = {
                        'type': 'tick',
                        'symbol': assetId.ticker if hasattr(assetId, 'ticker') else str(assetId),
                        'timestamp': str(date),
                        'timestamp_ms': int(datetime.now().timestamp() * 1000),
                        'trade_id': tradeNumber,
                        'price': float(price),
                        'volume': float(vol),
                        'quantity': int(qty),
                        'buyer': buyAgent,
                        'seller': sellAgent,
                        'trade_type': tradeType,
                        'is_edit': bIsEdit
                    }
                    
                    topic = f"tick_{tick_data['symbol']}".encode()
                    data = orjson.dumps(tick_data)
                    
                    self.publishers['tick'].send_multipart([topic, data], zmq.NOBLOCK)
                    
                except Exception as e:
                    self.logger.error(f"Erro ao publicar tick ZMQ: {e}")
                
                return result
            
            # Substituir callback
            self.connection_manager.callbacks['newTradeCallback'] = enhanced_trade_callback
    
    def publish_feature_update(self, symbol: str, features: Dict[str, Any]):
        """Publica atualização de features calculadas"""
        try:
            feature_data = {
                'type': 'features',
                'symbol': symbol,
                'timestamp_ms': int(datetime.now().timestamp() * 1000),
                'features': features
            }
            
            topic = f"features_{symbol}".encode()
            data = orjson.dumps(feature_data)
            
            self.publishers['signal'].send_multipart([topic, data], zmq.NOBLOCK)
            
        except Exception as e:
            self.logger.error(f"Erro ao publicar features: {e}")
    
    def close(self):
        """Fecha publishers ZMQ"""
        for socket in self.publishers.values():
            socket.close()
        self.context.term()
```

#### 2.2 ZMQ Consumer para Sistema Atual
```python
# Arquivo: src/integration/zmq_data_consumer.py
import zmq
import orjson
import threading
import logging
import time
from typing import Optional

class ZMQDataConsumer:
    """
    Consome dados ZMQ e alimenta TradingDataStructure existente
    Mantém compatibilidade total com sistema atual
    """
    
    def __init__(self, trading_data_structure):
        self.data_structure = trading_data_structure
        self.context = zmq.Context()
        self.subscribers = {}
        self.running = False
        self.threads = []
        self.logger = logging.getLogger('ZMQConsumer')
        
    def start(self):
        """Inicia consumo de dados ZMQ"""
        from src.config.zmq_valkey_config import ZMQValkeyConfig
        
        self.running = True
        urls = ZMQValkeyConfig.get_zmq_urls()
        
        # Criar subscribers
        for data_type, url in urls.items():
            socket = self.context.socket(zmq.SUB)
            socket.connect(url)
            socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscrever a tudo
            socket.setsockopt(zmq.RCVTIMEO, 1000)  # Timeout 1s
            self.subscribers[data_type] = socket
            
            # Criar thread para cada tipo de dado
            thread = threading.Thread(
                target=self._consume_loop,
                args=(data_type, socket),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
            
        self.logger.info("ZMQ Consumer iniciado")
    
    def _consume_loop(self, data_type: str, socket):
        """Loop de consumo para um tipo de dado"""
        while self.running:
            try:
                topic, data = socket.recv_multipart()
                parsed_data = orjson.loads(data)
                
                # Processar baseado no tipo
                if data_type == 'tick' and parsed_data.get('type') == 'tick':
                    self._process_tick(parsed_data)
                elif data_type == 'book':
                    self._process_book(parsed_data)
                elif data_type == 'history':
                    self._process_history(parsed_data)
                    
            except zmq.Again:
                continue  # Timeout normal
            except Exception as e:
                self.logger.error(f"Erro no consumo {data_type}: {e}")
                time.sleep(0.1)
    
    def _process_tick(self, tick_data: Dict):
        """Processa tick e adiciona ao TradingDataStructure"""
        try:
            # Converter para formato esperado pelo sistema atual
            if hasattr(self.data_structure, 'add_tick'):
                self.data_structure.add_tick({
                    'symbol': tick_data['symbol'],
                    'timestamp': tick_data['timestamp'],
                    'price': tick_data['price'],
                    'volume': tick_data['volume'],
                    'quantity': tick_data['quantity']
                })
        except Exception as e:
            self.logger.error(f"Erro ao processar tick: {e}")
    
    def stop(self):
        """Para consumo de dados"""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=2)
        
        for socket in self.subscribers.values():
            socket.close()
        self.context.term()
```

### **FASE 3: Integração Valkey (Dia 6-8)**

#### 3.1 Valkey Stream Manager
```python
# Arquivo: src/integration/valkey_stream_manager.py
import valkey
import orjson
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class ValkeyStreamManager:
    """
    Gerencia streams Valkey para armazenamento e time travel
    """
    
    def __init__(self):
        from src.config.zmq_valkey_config import ZMQValkeyConfig
        
        self.config = ZMQValkeyConfig
        self.client = None
        self.logger = logging.getLogger('ValkeyManager')
        self._connect()
        
    def _connect(self):
        """Conecta ao Valkey"""
        try:
            self.client = valkey.Valkey(
                host=self.config.VALKEY_HOST,
                port=self.config.VALKEY_PORT,
                password=self.config.VALKEY_PASSWORD,
                db=self.config.VALKEY_DB,
                decode_responses=False
            )
            self.client.ping()
            self.logger.info("Conectado ao Valkey")
        except Exception as e:
            self.logger.error(f"Erro ao conectar Valkey: {e}")
            raise
    
    def create_symbol_streams(self, symbol: str):
        """Cria streams necessários para um símbolo"""
        streams = {
            'ticks': f"stream:ticks:{symbol}",
            'candles_1m': f"stream:candles:1m:{symbol}",
            'features': f"stream:features:{symbol}",
            'signals': f"stream:signals:{symbol}",
            'predictions': f"stream:predictions:{symbol}"
        }
        
        for stream_type, stream_key in streams.items():
            try:
                # Criar stream com entry inicial se não existir
                self.client.xadd(
                    stream_key,
                    {"init": "true", "timestamp": str(datetime.now())},
                    maxlen=self.config.STREAM_MAX_LEN,
                    approximate=True
                )
                self.logger.debug(f"Stream {stream_key} criado/verificado")
            except Exception as e:
                self.logger.error(f"Erro ao criar stream {stream_key}: {e}")
        
        return streams
    
    def add_tick(self, symbol: str, tick_data: Dict):
        """Adiciona tick ao stream"""
        stream_key = f"stream:ticks:{symbol}"
        
        # Usar timestamp como ID para ordenação temporal
        timestamp_ms = tick_data.get('timestamp_ms', int(datetime.now().timestamp() * 1000))
        
        # Converter para bytes
        tick_bytes = {k.encode(): str(v).encode() for k, v in tick_data.items()}
        
        try:
            self.client.xadd(
                stream_key,
                tick_bytes,
                id=f"{timestamp_ms}-*",
                maxlen=self.config.STREAM_MAX_LEN,
                approximate=True
            )
        except Exception as e:
            self.logger.error(f"Erro ao adicionar tick: {e}")
    
    def add_candle(self, symbol: str, timeframe: str, candle_data: Dict):
        """Adiciona candle ao stream"""
        stream_key = f"stream:candles:{timeframe}:{symbol}"
        
        timestamp_ms = candle_data.get('timestamp_ms', int(datetime.now().timestamp() * 1000))
        candle_bytes = {k.encode(): str(v).encode() for k, v in candle_data.items()}
        
        try:
            self.client.xadd(
                stream_key,
                candle_bytes,
                id=f"{timestamp_ms}-*",
                maxlen=self.config.STREAM_MAX_LEN,
                approximate=True
            )
        except Exception as e:
            self.logger.error(f"Erro ao adicionar candle: {e}")
    
    def time_travel_query(self, symbol: str, start_time: datetime, end_time: datetime, 
                         data_type: str = 'ticks') -> List[Dict]:
        """
        Realiza time travel query entre timestamps
        """
        if data_type == 'ticks':
            stream_key = f"stream:ticks:{symbol}"
        elif data_type.startswith('candles'):
            timeframe = data_type.split('_')[1]
            stream_key = f"stream:candles:{timeframe}:{symbol}"
        else:
            stream_key = f"stream:{data_type}:{symbol}"
        
        # Converter para IDs de stream
        start_id = f"{int(start_time.timestamp() * 1000)}-0"
        end_id = f"{int(end_time.timestamp() * 1000)}-0"
        
        try:
            entries = self.client.xrange(stream_key, start_id, end_id)
            
            # Converter de volta para dicts
            results = []
            for entry_id, fields in entries:
                data = {k.decode(): v.decode() for k, v in fields.items()}
                data['stream_id'] = entry_id.decode()
                
                # Converter tipos numéricos
                for key in ['price', 'volume', 'high', 'low', 'open', 'close']:
                    if key in data:
                        try:
                            data[key] = float(data[key])
                        except:
                            pass
                
                results.append(data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erro no time travel query: {e}")
            return []
    
    def get_latest_features(self, symbol: str, count: int = 1) -> Optional[Dict]:
        """Obtém últimas features calculadas"""
        stream_key = f"stream:features:{symbol}"
        
        try:
            entries = self.client.xrevrange(stream_key, count=count)
            if entries:
                _, fields = entries[0]
                features = {k.decode(): v.decode() for k, v in fields.items()}
                
                # Parse features JSON
                if 'features' in features:
                    features['features'] = orjson.loads(features['features'])
                
                return features
            
        except Exception as e:
            self.logger.error(f"Erro ao obter features: {e}")
        
        return None
    
    def cleanup_old_data(self, days_to_keep: int = None):
        """Remove dados antigos dos streams"""
        if days_to_keep is None:
            days_to_keep = self.config.STREAM_RETENTION_DAYS
        
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        cutoff_id = f"{int(cutoff_time.timestamp() * 1000)}-0"
        
        # Buscar todos os streams
        streams = self.client.keys("stream:*")
        
        for stream_key in streams:
            try:
                # Remover entries antigas
                self.client.xtrim(stream_key, minid=cutoff_id, approximate=True)
            except Exception as e:
                self.logger.error(f"Erro ao limpar {stream_key}: {e}")
```

#### 3.2 Bridge ZMQ → Valkey
```python
# Arquivo: src/integration/zmq_valkey_bridge.py
import zmq
import orjson
import threading
import logging
import time
from typing import Dict

class ZMQValkeyBridge:
    """
    Ponte entre ZMQ e Valkey
    Consome dados ZMQ e armazena em Valkey para time travel
    """
    
    def __init__(self, valkey_manager):
        self.valkey_manager = valkey_manager
        self.context = zmq.Context()
        self.running = False
        self.threads = []
        self.logger = logging.getLogger('ZMQValkeyBridge')
        
    def start(self):
        """Inicia ponte ZMQ → Valkey"""
        from src.config.zmq_valkey_config import ZMQValkeyConfig
        
        self.running = True
        urls = ZMQValkeyConfig.get_zmq_urls()
        
        # Criar subscribers para cada tipo
        for data_type, url in urls.items():
            thread = threading.Thread(
                target=self._bridge_loop,
                args=(data_type, url),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        self.logger.info("ZMQ-Valkey Bridge iniciada")
    
    def _bridge_loop(self, data_type: str, url: str):
        """Loop principal da ponte para um tipo de dado"""
        socket = self.context.socket(zmq.SUB)
        socket.connect(url)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.setsockopt(zmq.RCVTIMEO, 1000)
        
        while self.running:
            try:
                topic, data = socket.recv_multipart()
                parsed_data = orjson.loads(data)
                
                # Processar baseado no tipo
                if data_type == 'tick':
                    self._process_tick_to_valkey(parsed_data)
                elif data_type == 'signal':
                    self._process_signal_to_valkey(parsed_data)
                    
            except zmq.Again:
                continue
            except Exception as e:
                self.logger.error(f"Erro na ponte {data_type}: {e}")
                time.sleep(0.1)
        
        socket.close()
    
    def _process_tick_to_valkey(self, tick_data: Dict):
        """Processa tick e armazena no Valkey"""
        try:
            symbol = tick_data.get('symbol')
            if symbol:
                # Adicionar ao stream de ticks
                self.valkey_manager.add_tick(symbol, tick_data)
                
                # Também agregar em candles (se implementado)
                # self._aggregate_to_candles(symbol, tick_data)
                
        except Exception as e:
            self.logger.error(f"Erro ao processar tick para Valkey: {e}")
    
    def _process_signal_to_valkey(self, signal_data: Dict):
        """Processa sinais/features e armazena no Valkey"""
        try:
            symbol = signal_data.get('symbol')
            data_type = signal_data.get('type', 'signal')
            
            if symbol and data_type == 'features':
                # Armazenar features
                stream_key = f"stream:features:{symbol}"
                self.valkey_manager.client.xadd(
                    stream_key,
                    {
                        'timestamp_ms': str(signal_data.get('timestamp_ms')),
                        'features': orjson.dumps(signal_data.get('features', {}))
                    },
                    maxlen=10000,
                    approximate=True
                )
                
        except Exception as e:
            self.logger.error(f"Erro ao processar signal para Valkey: {e}")
    
    def stop(self):
        """Para a ponte"""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=2)
        self.context.term()
```

### **FASE 4: Sistema Enhanced (Dia 9-10)**

#### 4.1 Trading System Enhanced
```python
# Arquivo: src/trading_system_enhanced.py
import logging
from typing import Dict, Optional

class TradingSystemEnhanced:
    """
    Sistema de trading aprimorado com ZMQ + Valkey
    Mantém compatibilidade total com sistema atual
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger('TradingSystemEnhanced')
        
        # Componentes originais (sem modificação)
        from src.trading_system import TradingSystem
        self.original_system = TradingSystem(config)
        
        # Componentes ZMQ + Valkey (opcionais)
        self.zmq_enabled = config.get('zmq_enabled', False)
        self.valkey_enabled = config.get('valkey_enabled', False)
        
        self.zmq_publisher = None
        self.zmq_consumer = None
        self.valkey_manager = None
        self.zmq_valkey_bridge = None
        
        if self.zmq_enabled or self.valkey_enabled:
            self._setup_enhanced_components()
    
    def _setup_enhanced_components(self):
        """Configura componentes aprimorados se habilitados"""
        
        if self.valkey_enabled:
            from src.integration.valkey_stream_manager import ValkeyStreamManager
            self.valkey_manager = ValkeyStreamManager()
            self.logger.info("✅ Valkey habilitado")
        
        if self.zmq_enabled:
            from src.integration.zmq_publisher_wrapper import ZMQPublisherWrapper
            from src.integration.zmq_data_consumer import ZMQDataConsumer
            
            # Wrapper para publicar dados via ZMQ
            self.zmq_publisher = ZMQPublisherWrapper(
                self.original_system.connection_manager
            )
            
            # Consumer opcional (se quiser processar dados ZMQ)
            if self.config.get('zmq_consume_enabled', False):
                self.zmq_consumer = ZMQDataConsumer(
                    self.original_system.data_structure
                )
            
            self.logger.info("✅ ZeroMQ habilitado")
        
        if self.zmq_enabled and self.valkey_enabled:
            from src.integration.zmq_valkey_bridge import ZMQValkeyBridge
            self.zmq_valkey_bridge = ZMQValkeyBridge(self.valkey_manager)
            self.logger.info("✅ Bridge ZMQ-Valkey habilitada")
    
    def start(self):
        """Inicia sistema completo"""
        
        # 1. Iniciar sistema original
        self.original_system.start()
        
        # 2. Iniciar componentes aprimorados
        if self.zmq_consumer:
            self.zmq_consumer.start()
        
        if self.zmq_valkey_bridge:
            self.zmq_valkey_bridge.start()
        
        self.logger.info("🚀 Sistema Enhanced iniciado com sucesso")
        
        # 3. Criar streams Valkey para símbolos configurados
        if self.valkey_manager:
            symbols = self.config.get('symbols', ['WDOQ25'])
            for symbol in symbols:
                self.valkey_manager.create_symbol_streams(symbol)
    
    def stop(self):
        """Para sistema completo"""
        
        # Parar componentes aprimorados
        if self.zmq_valkey_bridge:
            self.zmq_valkey_bridge.stop()
        
        if self.zmq_consumer:
            self.zmq_consumer.stop()
        
        if self.zmq_publisher:
            self.zmq_publisher.close()
        
        # Parar sistema original
        self.original_system.stop()
    
    def get_time_travel_data(self, symbol: str, start_time, end_time, data_type='ticks'):
        """
        Interface para time travel queries
        Retorna None se Valkey não estiver habilitado
        """
        if self.valkey_manager:
            return self.valkey_manager.time_travel_query(
                symbol, start_time, end_time, data_type
            )
        return None
```

#### 4.2 Script de Inicialização
```python
# Arquivo: start_enhanced.py
#!/usr/bin/env python3
"""
Script para iniciar sistema ML Trading com ZMQ + Valkey
"""

import sys
import os
import logging
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.trading_system_enhanced import TradingSystemEnhanced
from src.config.zmq_valkey_config import ZMQValkeyConfig

def setup_logging():
    """Configura logging do sistema"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/enhanced_system.log')
        ]
    )

def main():
    """Função principal"""
    
    # Setup
    setup_logging()
    logger = logging.getLogger('MainEnhanced')
    
    # Configuração do sistema
    config = {
        # Configurações originais
        'dll_path': os.getenv('PROFIT_DLL_PATH'),
        'user': os.getenv('PROFIT_USER'),
        'password': os.getenv('PROFIT_PASSWORD'),
        'symbols': ['WDOQ25'],
        
        # Configurações Enhanced (habilitar conforme necessário)
        'zmq_enabled': True,           # Habilitar ZeroMQ
        'valkey_enabled': True,         # Habilitar Valkey
        'zmq_consume_enabled': False,   # Consumir dados ZMQ (opcional)
        
        # Time Travel Features
        'time_travel_enabled': ZMQValkeyConfig.TIME_TRAVEL_ENABLED,
        'time_travel_lookback': ZMQValkeyConfig.TIME_TRAVEL_LOOKBACK_MINUTES
    }
    
    try:
        # Criar e iniciar sistema
        logger.info("🚀 Iniciando ML Trading System Enhanced...")
        logger.info(f"ZMQ: {config['zmq_enabled']} | Valkey: {config['valkey_enabled']}")
        
        system = TradingSystemEnhanced(config)
        system.start()
        
        logger.info("✅ Sistema iniciado com sucesso!")
        logger.info("💡 Pressione Ctrl+C para parar")
        
        # Manter rodando
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("⏹️ Parando sistema...")
                break
        
        # Parar sistema
        system.stop()
        logger.info("✅ Sistema parado com sucesso")
        
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### **FASE 5: Time Travel Features (Dia 11-12)**

#### 5.1 Enhanced Feature Engine
```python
# Arquivo: src/features/time_travel_feature_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging

class TimeTravelFeatureEngine:
    """
    Feature Engine com capacidades de Time Travel
    Calcula features usando dados históricos completos via Valkey
    """
    
    def __init__(self, valkey_manager, original_feature_engine):
        self.valkey_manager = valkey_manager
        self.original_engine = original_feature_engine
        self.logger = logging.getLogger('TimeTravelFeatures')
        
    def calculate_enhanced_features(self, symbol: str, 
                                  current_time: Optional[datetime] = None,
                                  lookback_minutes: int = 120) -> Optional[Dict]:
        """
        Calcula features aprimoradas usando time travel
        """
        if current_time is None:
            current_time = datetime.now()
        
        try:
            # 1. Buscar dados históricos via time travel
            start_time = current_time - timedelta(minutes=lookback_minutes)
            
            # Buscar ticks
            ticks = self.valkey_manager.time_travel_query(
                symbol, start_time, current_time, 'ticks'
            )
            
            if not ticks:
                self.logger.warning(f"Sem dados para {symbol} no período")
                return None
            
            # 2. Converter para DataFrame
            df = pd.DataFrame(ticks)
            
            # Converter timestamp_ms para datetime
            if 'timestamp_ms' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp_ms'].astype(float), unit='ms')
                df.set_index('timestamp', inplace=True)
            
            # 3. Calcular features básicas usando engine original
            base_features = self.original_engine.calculate_features(df)
            
            # 4. Adicionar features exclusivas de time travel
            enhanced = self._add_time_travel_features(df, base_features, symbol)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular features enhanced: {e}")
            return None
    
    def _add_time_travel_features(self, df: pd.DataFrame, 
                                 base_features: Dict, 
                                 symbol: str) -> Dict:
        """Adiciona features que só são possíveis com time travel"""
        
        enhanced = base_features.copy()
        
        try:
            # 1. Padrão de Volume Intraday
            enhanced['volume_pattern_score'] = self._calculate_volume_pattern(df)
            
            # 2. Momentum Histórico Comparativo
            enhanced['historical_momentum_rank'] = self._calculate_momentum_rank(df, symbol)
            
            # 3. Microestrutura de Mercado
            enhanced['microstructure_imbalance'] = self._calculate_microstructure(df)
            
            # 4. Regime de Volatilidade Estendido
            enhanced['volatility_regime_30d'] = self._calculate_volatility_regime(df, symbol)
            
            # 5. Correlações Dinâmicas (se múltiplos símbolos)
            enhanced['correlation_stability'] = self._calculate_correlation_stability(df, symbol)
            
        except Exception as e:
            self.logger.error(f"Erro em time travel features: {e}")
        
        return enhanced
    
    def _calculate_volume_pattern(self, df: pd.DataFrame) -> float:
        """
        Calcula similaridade do padrão de volume atual com dias anteriores
        """
        try:
            # Agregar volume por minuto
            volume_profile = df['volume'].resample('1min').sum()
            
            # Comparar com mesmo horário em dias anteriores
            current_hour = df.index[-1].hour
            current_minute = df.index[-1].minute
            
            # Buscar dados históricos de 30 dias
            historical_patterns = []
            
            for days_back in range(1, 8):  # Última semana
                hist_start = df.index[-1] - timedelta(days=days_back, hours=2)
                hist_end = df.index[-1] - timedelta(days=days_back)
                
                # Time travel para período histórico
                hist_data = self.valkey_manager.time_travel_query(
                    df.name if hasattr(df, 'name') else 'WDOQ25',
                    hist_start, hist_end, 'ticks'
                )
                
                if hist_data:
                    hist_df = pd.DataFrame(hist_data)
                    hist_volume = hist_df['volume'].values
                    historical_patterns.append(hist_volume)
            
            # Calcular score de similaridade
            if historical_patterns:
                current_pattern = volume_profile.values[-60:]  # Última hora
                
                similarities = []
                for hist_pattern in historical_patterns:
                    if len(hist_pattern) >= 60:
                        correlation = np.corrcoef(current_pattern, hist_pattern[-60:])[0, 1]
                        similarities.append(correlation)
                
                return np.nanmean(similarities) if similarities else 0.5
            
        except Exception as e:
            self.logger.error(f"Erro em volume pattern: {e}")
        
        return 0.5  # Neutro
    
    def _calculate_momentum_rank(self, df: pd.DataFrame, symbol: str) -> float:
        """
        Calcula ranking do momentum atual vs histórico
        """
        try:
            # Calcular momentum atual (última hora)
            current_return = (df['price'].iloc[-1] / df['price'].iloc[-60] - 1) * 100
            
            # Buscar retornos históricos de 30 dias
            historical_returns = []
            
            for days_back in range(1, 31):
                hist_time = df.index[-1] - timedelta(days=days_back)
                hist_start = hist_time - timedelta(hours=1)
                
                hist_data = self.valkey_manager.time_travel_query(
                    symbol, hist_start, hist_time, 'candles_1m'
                )
                
                if hist_data and len(hist_data) >= 60:
                    hist_df = pd.DataFrame(hist_data)
                    hist_return = (hist_df['close'].iloc[-1] / hist_df['close'].iloc[0] - 1) * 100
                    historical_returns.append(hist_return)
            
            # Calcular percentil do momentum atual
            if historical_returns:
                rank = np.sum(current_return > np.array(historical_returns)) / len(historical_returns)
                return rank
            
        except Exception as e:
            self.logger.error(f"Erro em momentum rank: {e}")
        
        return 0.5  # Mediano
    
    def _calculate_microstructure(self, df: pd.DataFrame) -> float:
        """
        Calcula desequilíbrio na microestrutura do mercado
        """
        try:
            # Analisar tamanho dos trades
            if 'quantity' in df.columns:
                # Separar trades grandes vs pequenos
                median_size = df['quantity'].median()
                
                large_trades = df[df['quantity'] > median_size * 2]
                small_trades = df[df['quantity'] <= median_size]
                
                # Calcular pressão direcional
                if len(large_trades) > 0:
                    large_buy_volume = large_trades[large_trades['trade_type'] == 'BUY']['volume'].sum()
                    large_sell_volume = large_trades[large_trades['trade_type'] == 'SELL']['volume'].sum()
                    
                    if (large_buy_volume + large_sell_volume) > 0:
                        imbalance = (large_buy_volume - large_sell_volume) / (large_buy_volume + large_sell_volume)
                        return imbalance
            
        except Exception as e:
            self.logger.error(f"Erro em microstructure: {e}")
        
        return 0.0  # Equilibrado
    
    def _calculate_volatility_regime(self, df: pd.DataFrame, symbol: str) -> str:
        """
        Identifica regime de volatilidade baseado em 30 dias
        """
        try:
            # Calcular volatilidade realizada atual
            returns = df['price'].pct_change().dropna()
            current_vol = returns.std() * np.sqrt(252 * 24 * 60)  # Anualizada
            
            # Buscar volatilidades históricas de 30 dias
            historical_vols = []
            
            for days_back in range(1, 31):
                hist_end = df.index[-1] - timedelta(days=days_back)
                hist_start = hist_end - timedelta(hours=24)
                
                hist_data = self.valkey_manager.time_travel_query(
                    symbol, hist_start, hist_end, 'ticks'
                )
                
                if hist_data and len(hist_data) > 100:
                    hist_df = pd.DataFrame(hist_data)
                    hist_returns = hist_df['price'].pct_change().dropna()
                    hist_vol = hist_returns.std() * np.sqrt(252 * 24 * 60)
                    historical_vols.append(hist_vol)
            
            # Classificar regime
            if historical_vols:
                percentile = np.sum(current_vol > np.array(historical_vols)) / len(historical_vols)
                
                if percentile > 0.8:
                    return "HIGH_VOL"
                elif percentile < 0.2:
                    return "LOW_VOL"
                else:
                    return "NORMAL_VOL"
            
        except Exception as e:
            self.logger.error(f"Erro em volatility regime: {e}")
        
        return "UNKNOWN"
    
    def _calculate_correlation_stability(self, df: pd.DataFrame, symbol: str) -> float:
        """
        Calcula estabilidade das correlações com outros ativos
        """
        # Implementar se houver múltiplos símbolos
        return 1.0  # Placeholder
```

## 📊 Scripts de Monitoramento

### Dashboard em Tempo Real
```python
# Arquivo: scripts/monitor_enhanced.py
#!/usr/bin/env python3
"""
Monitor em tempo real do sistema Enhanced
"""

import time
import os
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.live import Live

from src.integration.valkey_stream_manager import ValkeyStreamManager

console = Console()

def create_dashboard_table(valkey_manager, symbols):
    """Cria tabela do dashboard"""
    
    table = Table(title=f"ML Trading Enhanced Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    table.add_column("Symbol", style="cyan")
    table.add_column("Last Price", style="green")
    table.add_column("Volume 5m", style="yellow")
    table.add_column("Trades 5m", style="blue")
    table.add_column("Change %", style="magenta")
    table.add_column("Features", style="white")
    table.add_column("Signals", style="red")
    
    for symbol in symbols:
        # Buscar dados recentes
        recent_ticks = valkey_manager.get_recent_ticks(symbol, count=300)
        
        if recent_ticks:
            last_price = recent_ticks[0].get('price', 0)
            volume_5m = sum(float(t.get('volume', 0)) for t in recent_ticks)
            trades_5m = len(recent_ticks)
            
            # Calcular mudança percentual
            if len(recent_ticks) > 1:
                first_price = recent_ticks[-1].get('price', last_price)
                change_pct = ((last_price - first_price) / first_price) * 100
            else:
                change_pct = 0
            
            # Buscar última feature
            latest_features = valkey_manager.get_latest_features(symbol)
            feature_count = len(latest_features.get('features', {})) if latest_features else 0
            
            # Adicionar linha
            table.add_row(
                symbol,
                f"{last_price:.2f}",
                f"{volume_5m:,.0f}",
                str(trades_5m),
                f"{change_pct:+.2f}%",
                str(feature_count),
                "ACTIVE" if feature_count > 0 else "WAITING"
            )
        else:
            table.add_row(symbol, "-", "-", "-", "-", "-", "NO DATA")
    
    return table

def main():
    """Monitor principal"""
    
    # Conectar ao Valkey
    valkey_manager = ValkeyStreamManager()
    
    # Símbolos para monitorar
    symbols = os.getenv('MONITOR_SYMBOLS', 'WDOQ25').split(',')
    
    console.print("[bold green]ML Trading Enhanced Monitor[/bold green]")
    console.print(f"Monitorando: {', '.join(symbols)}")
    
    # Live update
    with Live(create_dashboard_table(valkey_manager, symbols), refresh_per_second=1) as live:
        while True:
            time.sleep(1)
            live.update(create_dashboard_table(valkey_manager, symbols))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor parado[/yellow]")
```

## 🚀 Guia de Deploy

### 1. Preparação do Ambiente
```bash
# Clone do repositório
git clone <seu-repo>
cd QuantumTrader_ML

# Criar branch para implementação
git checkout -b feature/zmq-valkey-integration

# Instalar dependências
pip install -r requirements.txt
pip install pyzmq valkey orjson rich

# Criar diretórios necessários
mkdir -p logs
mkdir -p src/integration
mkdir -p src/config
mkdir -p scripts
```

### 2. Deploy do Valkey
```bash
# Iniciar Valkey via Docker
docker-compose -f docker-compose.valkey.yml up -d

# Verificar se está rodando
docker ps | grep valkey

# Testar conexão
python -c "import valkey; c=valkey.Valkey(); print('OK' if c.ping() else 'FAIL')"
```

### 3. Configuração do Sistema
```bash
# Adicionar ao .env
echo "ZMQ_ENABLED=true" >> .env
echo "VALKEY_ENABLED=true" >> .env
echo "TIME_TRAVEL_ENABLED=true" >> .env
```

### 4. Teste Gradual
```bash
# Fase 1: Apenas ZMQ (sem Valkey)
ZMQ_ENABLED=true VALKEY_ENABLED=false python start_enhanced.py

# Fase 2: ZMQ + Valkey (sem consumir)
ZMQ_ENABLED=true VALKEY_ENABLED=true python start_enhanced.py

# Fase 3: Sistema completo
python start_enhanced.py

# Monitor
python scripts/monitor_enhanced.py
```

## 📈 Métricas de Sucesso

### Performance
- [ ] Latência de publicação ZMQ < 1ms
- [ ] Throughput > 10k msgs/segundo
- [ ] Time travel query < 100ms para 1h de dados
- [ ] Zero perda de dados em 24h

### Funcionalidade
- [ ] Sistema original continua funcionando
- [ ] Dados persistidos no Valkey
- [ ] Time travel funcionando
- [ ] Dashboard mostrando dados em tempo real

## 🔧 Troubleshooting

### ZMQ não conecta
```bash
# Verificar portas
netstat -an | grep 5555

# Testar publisher isolado
python -c "import zmq; ctx=zmq.Context(); s=ctx.socket(zmq.PUB); s.bind('tcp://*:5555'); print('OK')"
```

### Valkey sem dados
```bash
# Verificar streams
docker exec ml-trading-valkey valkey-cli XINFO STREAM stream:ticks:WDOQ25

# Debug bridge
export LOG_LEVEL=DEBUG
python start_enhanced.py
```

### Performance degradada
```python
# Adicionar profiling
import cProfile
cProfile.run('system.start()', 'profile_stats')

# Analisar
import pstats
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(20)
```

## 📚 Próximos Passos

1. **Otimização de Queries**: Implementar cache Redis para queries frequentes
2. **Compressão**: Adicionar compressão zstd para reduzir uso de memória
3. **Sharding**: Distribuir dados entre múltiplas instâncias Valkey
4. **ML Pipeline**: Integrar time travel features no treinamento
5. **Alertas**: Sistema de alertas baseado em patterns históricos

---

**Esta implementação permite evolução gradual do sistema sem riscos, habilitando capacidades avançadas conforme necessário!**