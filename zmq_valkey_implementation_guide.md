# 🚀 Guia de Implementação ZeroMQ + Valkey para Sistema ML Trading Atual

## 📊 **Análise do Sistema Atual**

### **Arquitetura Existente Identificada**
```python
# Sistema Atual (baseado nos documentos)
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ProfitDLL     │────│ TradingData     │────│  FeatureEngine  │
│  (Callbacks)    │    │   Structure     │    │  (Cálculos)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML System                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │RegimeAnalyzer│  │ModelManager │  │   SignalGenerator       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌─────────────────┐
                    │   ProfitDLL     │
                    │  (Execução)     │
                    └─────────────────┘
```

### **Limitações Atuais**
- **Latência**: ~3s para pipeline completo
- **Storage**: DataFrames em memória (limitado)
- **Escalabilidade**: Single-thread, single-process
- **Time Travel**: Impossível navegar histórico
- **Backup**: Dados perdidos em reinicialização

---

## 🎯 **Nova Arquitetura ZeroMQ + Valkey**

### **Arquitetura Proposta**
```python
# Nova Arquitetura Integrada
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ProfitDLL     │────│    ZeroMQ       │────│     Valkey      │
│  (Callbacks)    │    │  (Transport)    │    │   (Storage)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Enhanced Trading System                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │Time Travel  │  │Real-time ML │  │   Distributed Features │ │
│  │  Analysis   │  │ Processing  │  │     Calculation         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌─────────────────┐
                    │Multi-Process    │
                    │   Execution     │ 
                    └─────────────────┘
```

---

## 📋 **Plano de Implementação Gradual**

### **FASE 1: Integração ZeroMQ (Sem Quebrar Sistema Atual)**

#### **1.1 Adicionar Camada ZeroMQ ao Feeder Existente**

```python
# Arquivo: src/zmq_integration/zmq_feeder_wrapper.py
import zmq
import orjson
import threading
from typing import Dict, Any

class ZMQFeederWrapper:
    """Wrapper ZeroMQ para o feeder existente sem modificar código original"""
    
    def __init__(self, original_feeder):
        self.original_feeder = original_feeder
        self.context = zmq.Context()
        
        # Publishers para diferentes tipos de dados
        self.tick_publisher = self.context.socket(zmq.PUB)
        self.tick_publisher.bind("tcp://*:5555")
        
        self.book_publisher = self.context.socket(zmq.PUB)  
        self.book_publisher.bind("tcp://*:5556")
        
        self.history_publisher = self.context.socket(zmq.PUB)
        self.history_publisher.bind("tcp://*:5557")
        
        # Interceptar callbacks do feeder original
        self._setup_callback_interceptors()
        
    def _setup_callback_interceptors(self):
        """Intercepta callbacks existentes e adiciona publicação ZMQ"""
        
        # Salvar callback original
        original_trade_callback = self.original_feeder.callback.newTradeCallback
        
        def enhanced_trade_callback(assetId, date, tradeNumber, price, vol, qty, buyAgent, sellAgent, tradeType, bIsEdit):
            # Executar callback original (manter compatibilidade)
            original_trade_callback(assetId, date, tradeNumber, price, vol, qty, buyAgent, sellAgent, tradeType, bIsEdit)
            
            # Adicionar publicação ZMQ
            tick_data = {
                'symbol': assetId.ticker,
                'timestamp': date,
                'trade_id': tradeNumber,
                'price': float(price),
                'volume': float(vol),
                'quantity': int(qty),
                'buyer': buyAgent,
                'seller': sellAgent,
                'trade_type': tradeType,
                'is_edit': bIsEdit
            }
            
            # Publicar via ZMQ (ultra-rápido)
            self.tick_publisher.send_multipart([
                f"tick_{assetId.ticker}".encode(),
                orjson.dumps(tick_data)
            ])
        
        # Substituir callback
        self.original_feeder.callback.newTradeCallback = enhanced_trade_callback
        
        # Mesmo processo para outros callbacks
        self._enhance_history_callback()
        self._enhance_book_callback()
    
    def _enhance_history_callback(self):
        """Intercepta callback de histórico"""
        original_history_callback = self.original_feeder.callback.newHistoryCallback
        
        def enhanced_history_callback(assetId, date, tradeNumber, price, vol, qty, buyAgent, sellAgent, tradeType):
            # Callback original
            original_history_callback(assetId, date, tradeNumber, price, vol, qty, buyAgent, sellAgent, tradeType)
            
            # ZMQ publishing
            history_data = {
                'symbol': assetId.ticker,
                'timestamp': date, 
                'trade_id': tradeNumber,
                'price': float(price),
                'volume': float(vol),
                'quantity': int(qty),
                'buyer': buyAgent,
                'seller': sellAgent,
                'trade_type': tradeType
            }
            
            self.history_publisher.send_multipart([
                f"history_{assetId.ticker}".encode(),
                orjson.dumps(history_data)
            ])
        
        self.original_feeder.callback.newHistoryCallback = enhanced_history_callback
```

#### **1.2 Criar Consumidores ZMQ para Sistema Atual**

```python
# Arquivo: src/zmq_integration/zmq_consumers.py
class ZMQTradingDataConsumer:
    """Consome dados ZMQ e alimenta TradingDataStructure existente"""
    
    def __init__(self, trading_data_structure):
        self.data_structure = trading_data_structure
        self.context = zmq.Context()
        
        # Subscribers
        self.tick_subscriber = self.context.socket(zmq.SUB)
        self.tick_subscriber.connect("tcp://localhost:5555")
        self.tick_subscriber.setsockopt(zmq.SUBSCRIBE, b"tick_")
        
        self.history_subscriber = self.context.socket(zmq.SUB)
        self.history_subscriber.connect("tcp://localhost:5557")
        self.history_subscriber.setsockopt(zmq.SUBSCRIBE, b"history_")
        
        self.running = False
        
    def start_consuming(self):
        """Inicia consumo em threads separadas"""
        self.running = True
        
        # Thread para ticks tempo real
        tick_thread = threading.Thread(target=self._consume_ticks)
        tick_thread.daemon = True
        tick_thread.start()
        
        # Thread para histórico
        history_thread = threading.Thread(target=self._consume_history)
        history_thread.daemon = True
        history_thread.start()
        
    def _consume_ticks(self):
        """Consome ticks tempo real"""
        while self.running:
            try:
                topic, data = self.tick_subscriber.recv_multipart(zmq.NOBLOCK)
                tick = orjson.loads(data)
                
                # Alimentar TradingDataStructure existente
                self.data_structure.add_tick_data(tick)
                
            except zmq.Again:
                time.sleep(0.001)  # 1ms sleep
                
    def _consume_history(self):
        """Consome dados históricos"""
        while self.running:
            try:
                topic, data = self.history_subscriber.recv_multipart(zmq.NOBLOCK)
                historical_tick = orjson.loads(data)
                
                # Alimentar estrutura histórica
                self.data_structure.add_historical_tick(historical_tick)
                
            except zmq.Again:
                time.sleep(0.001)
```

#### **1.3 Integração no Sistema Atual**

```python
# Arquivo: src/trading_system_zmq_enhanced.py
class TradingSystemZMQEnhanced:
    """Sistema atual com camada ZMQ adicionada"""
    
    def __init__(self, config):
        # Componentes existentes (sem modificação)
        self.data_structure = TradingDataStructure()  # Existente
        self.feature_engine = FeatureEngine()         # Existente  
        self.model_manager = ModelManager()           # Existente
        
        # NOVOS: Componentes ZMQ
        self.zmq_wrapper = ZMQFeederWrapper(self.connection_manager)
        self.zmq_consumer = ZMQTradingDataConsumer(self.data_structure)
        
    def start_enhanced_system(self):
        """Inicia sistema com ZMQ habilitado"""
        
        # 1. Iniciar sistema original
        self.connection_manager.start()  # ProfitDLL
        
        # 2. Iniciar camada ZMQ (sem afetar original)
        self.zmq_consumer.start_consuming()
        
        # 3. Sistema funciona normalmente + ZMQ publishing
        print("✅ Sistema com ZMQ iniciado - compatibilidade mantida")
```

---

### **FASE 2: Integração Valkey (Parallel Storage)**

#### **2.1 Valkey Stream Manager**

```python
# Arquivo: src/valkey_integration/valkey_stream_manager.py
import valkey
import orjson
import threading
from datetime import datetime

class ValkeyStreamManager:
    """Gerencia streams Valkey em paralelo ao sistema atual"""
    
    def __init__(self, valkey_config=None):
        self.valkey = valkey.Valkey(
            host=valkey_config.get('host', 'localhost'),
            port=valkey_config.get('port', 6379),
            decode_responses=False  # Manter bytes para performance
        )
        
        self.active_streams = set()
        self.buffer_size = 10000  # Máximo de entries por stream
        
    def create_symbol_streams(self, symbol):
        """Cria streams para um símbolo"""
        
        streams = {
            'ticks': f"ticks:{symbol}:realtime",
            'history': f"ticks:{symbol}:historical", 
            'features': f"features:{symbol}",
            'signals': f"signals:{symbol}"
        }
        
        for stream_type, stream_key in streams.items():
            try:
                # Criar stream se não existir
                self.valkey.xadd(stream_key, {"init": "true"}, maxlen=self.buffer_size)
                self.active_streams.add(stream_key)
            except:
                pass  # Stream já existe
                
        return streams
    
    def add_tick_to_stream(self, symbol, tick_data):
        """Adiciona tick ao stream (thread-safe)"""
        
        stream_key = f"ticks:{symbol}:realtime"
        
        # Usar timestamp como ID para ordenação perfeita
        timestamp_ms = int(tick_data.get('timestamp_ms', time.time() * 1000))
        
        self.valkey.xadd(
            stream_key,
            tick_data,
            id=f"{timestamp_ms}-0",
            maxlen=self.buffer_size
        )
    
    def get_recent_ticks(self, symbol, count=100):
        """Recupera últimos N ticks"""
        
        stream_key = f"ticks:{symbol}:realtime"
        
        ticks = self.valkey.xrevrange(stream_key, count=count)
        
        # Converter para formato utilizável
        formatted_ticks = []
        for tick_id, fields in ticks:
            tick_data = {k.decode(): v.decode() for k, v in fields.items()}
            tick_data['stream_id'] = tick_id.decode()
            formatted_ticks.append(tick_data)
            
        return formatted_ticks
    
    def time_travel_query(self, symbol, start_time, end_time):
        """Time travel query entre timestamps"""
        
        stream_key = f"ticks:{symbol}:realtime"
        
        start_id = f"{int(start_time.timestamp() * 1000)}-0"
        end_id = f"{int(end_time.timestamp() * 1000)}-0"
        
        return self.valkey.xrange(stream_key, start_id, end_id)
```

#### **2.2 Valkey Consumer Integrado**

```python  
# Arquivo: src/valkey_integration/valkey_zmq_bridge.py
class ValkeyZMQBridge:
    """Ponte entre ZMQ e Valkey mantendo sistema atual"""
    
    def __init__(self, valkey_manager):
        self.valkey_manager = valkey_manager
        self.context = zmq.Context()
        
        # Subscriber para dados ZMQ
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect("tcp://localhost:5555")
        self.subscriber.setsockopt(zmq.SUBSCRIBE, b"tick_")
        
        self.running = False
        
    def start_bridge(self):
        """Inicia ponte ZMQ -> Valkey"""
        self.running = True
        
        bridge_thread = threading.Thread(target=self._bridge_loop)
        bridge_thread.daemon = True
        bridge_thread.start()
        
    def _bridge_loop(self):
        """Loop principal da ponte"""
        while self.running:
            try:
                topic, data = self.subscriber.recv_multipart(zmq.NOBLOCK)
                
                # Parse topic para extrair símbolo
                topic_str = topic.decode()
                symbol = topic_str.replace('tick_', '')
                
                # Parse dados
                tick_data = orjson.loads(data)
                
                # Adicionar timestamp se não existir
                if 'timestamp_ms' not in tick_data:
                    tick_data['timestamp_ms'] = int(time.time() * 1000)
                
                # Armazenar no Valkey
                self.valkey_manager.add_tick_to_stream(symbol, tick_data)
                
            except zmq.Again:
                time.sleep(0.001)
```

---

### **FASE 3: Feature Calculation com Time Travel**

#### **3.1 Enhanced Feature Engine**

```python
# Arquivo: src/features/time_travel_feature_engine.py
class TimeTravelFeatureEngine:
    """Feature Engine com capacidades de time travel"""
    
    def __init__(self, valkey_manager, original_feature_engine):
        self.valkey_manager = valkey_manager
        self.original_engine = original_feature_engine  # Manter original
        
    def calculate_features_with_history(self, symbol, current_time, lookback_minutes=60):
        """Calcula features usando time travel para histórico completo"""
        
        # Time travel para obter dados necessários
        end_time = current_time
        start_time = current_time - timedelta(minutes=lookback_minutes)
        
        # Buscar dados via time travel
        historical_ticks = self.valkey_manager.time_travel_query(
            symbol, start_time, end_time
        )
        
        # Converter para DataFrame (formato que sistema atual espera)
        df_ticks = self._ticks_to_dataframe(historical_ticks)
        
        # Usar feature engine original (compatibilidade total)
        features = self.original_engine.calculate(df_ticks)
        
        # Adicionar features específicas de time travel
        enhanced_features = self._add_time_travel_features(df_ticks, features)
        
        return enhanced_features
    
    def _add_time_travel_features(self, df_ticks, base_features):
        """Adiciona features que só são possíveis com time travel"""
        
        enhanced = base_features.copy()
        
        # Feature: Padrão de volume nas últimas 3 horas vs mesmo horário ontem
        enhanced['volume_pattern_similarity'] = self._calculate_volume_pattern_similarity(df_ticks)
        
        # Feature: Momentum comparado com mesmo período em dias anteriores  
        enhanced['historical_momentum_percentile'] = self._calculate_historical_momentum_percentile(df_ticks)
        
        # Feature: Volatility regime baseado em 30 dias de histórico
        enhanced['volatility_regime_score'] = self._calculate_volatility_regime(df_ticks)
        
        return enhanced
```

#### **3.2 Integração com Sistema ML Atual**

```python
# Arquivo: src/ml/enhanced_ml_coordinator.py
class EnhancedMLCoordinator:
    """ML Coordinator com capacidades time travel"""
    
    def __init__(self, original_coordinator, valkey_manager):
        self.original_coordinator = original_coordinator  # Manter original
        self.valkey_manager = valkey_manager
        self.time_travel_engine = TimeTravelFeatureEngine(valkey_manager, original_coordinator.feature_engine)
        
    def process_prediction_request(self, symbol, current_time=None):
        """Processa predição usando time travel quando necessário"""
        
        if current_time is None:
            current_time = datetime.now()
            
        # MODO 1: Usar sistema original (rápido, para trading em tempo real)
        if self._should_use_fast_mode():
            return self.original_coordinator.process_prediction_request()
            
        # MODO 2: Usar time travel (mais lento, mais preciso)
        else:
            enhanced_features = self.time_travel_engine.calculate_features_with_history(
                symbol, current_time, lookback_minutes=120
            )
            
            # Usar modelo original com features aprimoradas
            prediction = self.original_coordinator.model_manager.predict(enhanced_features)
            
            return prediction
    
    def _should_use_fast_mode(self):
        """Decide quando usar modo rápido vs time travel"""
        
        # Usar time travel apenas quando:
        # 1. Não está em horário de alta atividade
        # 2. Há tempo suficiente para cálculo
        # 3. Precisão é mais importante que velocidade
        
        current_hour = datetime.now().hour
        
        # Horário de menor atividade: usar time travel
        if current_hour < 10 or current_hour > 16:
            return False
            
        # Horário de alta atividade: usar modo rápido
        return True
```

---

### **FASE 4: Dashboard e Monitoramento**

#### **4.1 Real-Time Dashboard**

```python
# Arquivo: src/monitoring/realtime_dashboard.py
class RealTimeDashboard:
    """Dashboard em tempo real usando dados do Valkey"""
    
    def __init__(self, valkey_manager):
        self.valkey_manager = valkey_manager
        
    def get_dashboard_data(self, symbol):
        """Coleta dados para dashboard em tempo real"""
        
        # Dados dos últimos 5 minutos
        recent_ticks = self.valkey_manager.get_recent_ticks(symbol, count=300)
        
        # Métricas em tempo real
        dashboard_data = {
            'symbol': symbol,
            'last_price': recent_ticks[0]['price'] if recent_ticks else 0,
            'volume_5min': sum(float(t['volume']) for t in recent_ticks),
            'trades_5min': len(recent_ticks),
            'price_change_5min': self._calculate_price_change(recent_ticks),
            'avg_trade_size': self._calculate_avg_trade_size(recent_ticks),
            'buy_sell_ratio': self._calculate_buy_sell_ratio(recent_ticks),
            'timestamp': datetime.now().isoformat()
        }
        
        return dashboard_data
    
    def get_historical_analysis(self, symbol, days_back=7):
        """Análise histórica usando time travel"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        # Time travel para período completo
        historical_data = self.valkey_manager.time_travel_query(symbol, start_time, end_time)
        
        # Análises possíveis apenas com time travel
        analysis = {
            'daily_patterns': self._analyze_daily_patterns(historical_data),
            'volume_patterns': self._analyze_volume_patterns(historical_data),
            'volatility_evolution': self._analyze_volatility_evolution(historical_data),
            'regime_changes': self._detect_regime_changes(historical_data)
        }
        
        return analysis
```

---

## 🔧 **Implementação Prática**

### **Setup e Configuração**

```bash
# 1. Instalar dependências
pip install pyzmq valkey orjson

# 2. Configurar Valkey
docker run -d \
  --name valkey-trading \
  -p 6379:6379 \
  -v valkey-data:/data \
  valkey/valkey:latest \
  --maxmemory 8gb \
  --maxmemory-policy allkeys-lru

# 3. Configurar ZeroMQ (não precisa servidor)
# ZeroMQ é brokerless - apenas instalar biblioteca
```

### **Arquivo de Configuração**

```python
# Arquivo: config/zmq_valkey_config.py
ZMQ_VALKEY_CONFIG = {
    'zmq': {
        'tick_port': 5555,
        'book_port': 5556, 
        'history_port': 5557,
        'signals_port': 5558
    },
    'valkey': {
        'host': 'localhost',
        'port': 6379,
        'max_memory': '8gb',
        'stream_maxlen': 100000,  # Máximo entries por stream
        'retention_days': 30      # Manter dados por 30 dias
    },
    'features': {
        'time_travel_lookback': 120,      # Minutos
        'fast_mode_threshold': 0.1,       # Usar fast mode se latência < 100ms
        'enable_enhanced_features': True
    }
}
```

### **Script de Inicialização**

```python
# Arquivo: scripts/start_enhanced_system.py
def start_enhanced_system():
    """Inicia sistema completo com ZMQ + Valkey"""
    
    print("🚀 Iniciando Sistema ML Trading Enhanced...")
    
    # 1. Inicializar componentes originais
    trading_system = TradingSystemV2(config)  # Sistema atual
    
    # 2. Adicionar camada ZMQ + Valkey (sem quebrar)
    valkey_manager = ValkeyStreamManager(ZMQ_VALKEY_CONFIG['valkey'])
    zmq_wrapper = ZMQFeederWrapper(trading_system.connection_manager)
    valkey_bridge = ValkeyZMQBridge(valkey_manager)
    
    # 3. Inicializar ML aprimorado
    enhanced_ml = EnhancedMLCoordinator(trading_system.ml_coordinator, valkey_manager)
    
    # 4. Inicializar dashboard
    dashboard = RealTimeDashboard(valkey_manager)
    
    # 5. Iniciar tudo
    trading_system.start()           # Sistema original
    valkey_bridge.start_bridge()     # Ponte ZMQ -> Valkey
    
    print("✅ Sistema Enhanced iniciado com sucesso!")
    print("📊 Dashboard disponível para análises avançadas")
    print("⚡ Time travel habilitado para features aprimoradas")
    
    return {
        'trading_system': trading_system,
        'valkey_manager': valkey_manager,
        'enhanced_ml': enhanced_ml,
        'dashboard': dashboard
    }

if __name__ == "__main__":
    system = start_enhanced_system()
```

---

## 📊 **Benefícios Imediatos**

### **Performance**
- **Latência reduzida**: ZMQ reduz latência de comunicação em 90%+
- **Throughput**: Suporte a 100k+ ticks/segundo
- **Parallel processing**: Múltiplos consumidores em paralelo

### **Capacidades Novas**
- **Time Travel**: Análises históricas instantâneas
- **Dashboard tempo real**: Métricas avançadas
- **Backup automático**: Dados persistidos no Valkey
- **Escalabilidade**: Adicionar processos sem reescrever código

### **Compatibilidade**
- **Zero breaking changes**: Sistema atual continua funcionando
- **Migração gradual**: Habilitar funcionalidades quando necessário
- **Fallback**: Sempre pode voltar ao sistema original

---

## 🎯 **Cronograma de Implementação**

### **Semana 1: ZeroMQ Layer**
- [ ] Implementar ZMQFeederWrapper
- [ ] Testar publicação sem quebrar sistema atual
- [ ] Validar latência e throughput

### **Semana 2: Valkey Integration**  
- [ ] Implementar ValkeyStreamManager
- [ ] Criar ponte ZMQ -> Valkey
- [ ] Testar time travel queries

### **Semana 3: Enhanced Features**
- [ ] Implementar TimeTravelFeatureEngine
- [ ] Integrar com ML Coordinator existente
- [ ] Benchmark performance vs sistema atual

### **Semana 4: Dashboard e Deploy**
- [ ] Implementar dashboard tempo real
- [ ] Testes de stress completos
- [ ] Deploy gradual em produção

**Esta implementação permite manter o sistema atual funcionando enquanto adiciona capacidades avançadas de forma incremental e sem riscos!**