# 📚 Guia de Infraestrutura HMARL - ZeroMQ + Valkey + Flow

## 📋 Índice
1. [Visão Geral](#visão-geral)
2. [Arquitetura](#arquitetura)
3. [Instalação e Setup](#instalação-e-setup)
4. [Componentes Principais](#componentes-principais)
5. [Integração com Sistema Atual](#integração-com-sistema-atual)
6. [Análise de Fluxo](#análise-de-fluxo)
7. [Uso e Exemplos](#uso-e-exemplos)
8. [Monitoramento e Métricas](#monitoramento-e-métricas)
9. [Troubleshooting](#troubleshooting)

## 🎯 Visão Geral

A infraestrutura HMARL (Hierarchical Multi-Agent Reinforcement Learning) adiciona capacidades avançadas de análise de fluxo ao sistema QuantumTrader_ML v2.0, mantendo **compatibilidade total** com o sistema existente.

### Principais Benefícios
- ✅ **Zero Breaking Changes**: Sistema atual continua funcionando
- ⚡ **Ultra-baixa latência**: < 1ms para publicação de dados
- 📊 **Análise de Fluxo**: OFI, tape reading, footprint em tempo real
- ⏰ **Time Travel**: Consulta histórica instantânea via Valkey
- 🔄 **Escalabilidade**: Processamento distribuído com ZeroMQ

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    Sistema Atual (Intacto)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ ProfitDLL   │  │   Trading   │  │    ML Pipeline      │ │
│  │ v4.0        │  │   System    │  │                     │ │
│  └──────┬──────┘  └─────────────┘  └─────────────────────┘ │
└─────────┼───────────────────────────────────────────────────┘
          │
          ▼ Callbacks Interceptados (não-invasivo)
┌─────────────────────────────────────────────────────────────┐
│              HMARL Infrastructure Layer                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   ZeroMQ    │  │   Valkey    │  │  Flow Analysis      │ │
│  │ Publishers  │  │  Streams    │  │    Engine           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
          │
          ▼ Streams de Dados
┌─────────────────────────────────────────────────────────────┐
│                    Data Consumers                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │    Flow     │  │    Tape     │  │    Liquidity        │ │
│  │  Consumer   │  │  Consumer   │  │    Consumer         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Portas ZeroMQ Padrão
- **5555**: Tick data (compatível com sistema atual)
- **5556**: Order book
- **5557**: Flow analysis
- **5558**: Footprint
- **5559**: Liquidity
- **5560**: Tape reading patterns

## 🔧 Instalação e Setup

### 1. Instalar Dependências

```bash
# Dependências Python
pip install pyzmq valkey orjson numpy

# Valkey (Redis fork)
docker run -d \
  --name valkey-trading \
  -p 6379:6379 \
  -v valkey-data:/data \
  valkey/valkey:latest \
  --maxmemory 8gb \
  --maxmemory-policy allkeys-lru
```

### 2. Configuração Básica

```python
HMARL_CONFIG = {
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
        }
    }
}
```

## 📦 Componentes Principais

### 1. TradingInfrastructureWithFlow
Núcleo da infraestrutura que gerencia ZeroMQ e Valkey.

```python
from src.infrastructure.zmq_valkey_flow_setup import TradingInfrastructureWithFlow

# Criar e inicializar
infrastructure = TradingInfrastructureWithFlow(HMARL_CONFIG)
infrastructure.initialize()
infrastructure.start()

# Publicar tick com análise de fluxo
tick_data = {
    'symbol': 'WDOH25',
    'timestamp': datetime.now().isoformat(),
    'price': 5000.0,
    'volume': 10,
    'trade_type': 2  # 2=buy, 3=sell
}
infrastructure.publish_tick_with_flow(tick_data)
```

### 2. HMARLSystemWrapper
Integra HMARL ao sistema existente sem modificações.

```python
from src.infrastructure.system_integration_wrapper import integrate_hmarl_with_system

# Integrar ao sistema existente
hmarl_wrapper = integrate_hmarl_with_system(trading_system, HMARL_CONFIG)

# Sistema continua funcionando normalmente + análise de fluxo
trading_system.start('WDOH25')
```

### 3. Flow Consumers
Consumidores especializados para diferentes tipos de dados.

```python
from src.infrastructure.zmq_consumers import FlowConsumer, MultiStreamConsumer

# Consumer de fluxo
flow_consumer = FlowConsumer('WDOH25')
flow_consumer.start()

# Obter OFI (Order Flow Imbalance)
ofi_5m = flow_consumer.get_ofi('5m')

# Consumer multi-stream
multi_consumer = MultiStreamConsumer('WDOH25')
multi_consumer.start()

# Visão unificada
unified_data = multi_consumer.get_unified_view()
```

## 🔗 Integração com Sistema Atual

### Método 1: Wrapper Automático (Recomendado)

```python
# No seu main.py, após inicializar o sistema
from src.infrastructure.system_integration_wrapper import integrate_hmarl_with_system

# Sistema existente
trading_system = TradingSystem(config)
trading_system.initialize()

# Adicionar HMARL (uma linha!)
hmarl_wrapper = integrate_hmarl_with_system(trading_system)

# Pronto! Sistema funciona com análise de fluxo
trading_system.start('WDOH25')
```

### Método 2: Features Aprimoradas

```python
# Obter features de fluxo para ML
flow_features = hmarl_wrapper.get_flow_enhanced_features('WDOH25')

# Features adicionadas automaticamente:
# - flow_ofi_1m, flow_ofi_5m, flow_ofi_15m
# - flow_volume_imbalance
# - flow_aggression_ratio
# - flow_large_trade_ratio
# - tape_speed
# - liquidity_score
```

### Método 3: Consumers Independentes

```python
# Criar sistema de alertas paralelo
from src.infrastructure.zmq_consumers import FlowAlertSystem

alert_system = FlowAlertSystem('WDOH25')
alert_system.start()

# Sistema detecta automaticamente:
# - OFI extremo (> 70%)
# - Spikes de trades grandes
# - Padrões de sweep/iceberg
```

## 📊 Análise de Fluxo

### Order Flow Imbalance (OFI)
Mede o desequilíbrio entre compras e vendas.

```python
# OFI = (Buy Volume - Sell Volume) / Total Volume
# Valores: -1 (100% vendas) a +1 (100% compras)

ofi_values = {
    1: 0.15,   # 1 minuto: leve pressão compradora
    5: 0.35,   # 5 minutos: pressão compradora moderada
    15: 0.55   # 15 minutos: pressão compradora forte
}
```

### Tape Reading Patterns

#### Sweep Pattern
Varredura rápida de liquidez em uma direção.
```python
# Detectado quando:
# - 5+ trades consecutivos na mesma direção
# - Volume crescente
# - Velocidade > 5 trades/segundo
```

#### Iceberg Pattern
Ordem grande sendo executada em pequenos pedaços.
```python
# Detectado quando:
# - Múltiplos trades pequenos no mesmo preço
# - Volume consistente
# - Baixa variação de tamanho
```

### Categorias de Trade Size
```python
trade_categories = {
    'small': volume <= 5,      # Retail
    'medium': 5 < volume <= 20,   # Pequenos institucionais
    'large': 20 < volume <= 50,   # Institucionais
    'whale': volume > 50          # Grandes players
}
```

## 🎮 Uso e Exemplos

### Exemplo 1: Monitoramento Básico

```python
# Criar consumer de fluxo
flow_consumer = FlowConsumer('WDOH25')
flow_consumer.start()

# Monitorar OFI em tempo real
while True:
    ofi = flow_consumer.get_ofi('5m')
    
    if ofi > 0.7:
        print("🔴 PRESSÃO COMPRADORA EXTREMA!")
    elif ofi < -0.7:
        print("🔵 PRESSÃO VENDEDORA EXTREMA!")
    
    time.sleep(1)
```

### Exemplo 2: Análise Histórica (Time Travel)

```python
# Buscar fluxo dos últimos 30 minutos
history = infrastructure.get_flow_history('WDOH25', minutes_back=30)

# Analisar padrões
buy_pressure_periods = [h for h in history 
                       if h['analysis']['ofi'][5] > 0.5]

print(f"Períodos de forte pressão compradora: {len(buy_pressure_periods)}")
```

### Exemplo 3: Detecção de Eventos

```python
# Callback para eventos de fluxo
def on_flow_event(data):
    if data['analysis']['large_trade_ratio'] > 0.3:
        print("🐋 Atividade de grandes players detectada!")
    
    if abs(data['analysis']['volume_imbalance']) > 0.8:
        print("⚡ Desequilíbrio extremo de volume!")

# Configurar consumer com callback
flow_consumer = FlowConsumer('WDOH25', callback=on_flow_event)
flow_consumer.start()
```

## 📈 Monitoramento e Métricas

### Métricas de Performance

```python
# Obter métricas da infraestrutura
metrics = infrastructure.get_performance_metrics()

print(f"Mensagens publicadas: {metrics['messages_published']}")
print(f"Latência média: {metrics['avg_latency_ms']}ms")
print(f"Eventos de fluxo: {metrics['flow_events_detected']}")
```

### Dashboard de Liquidez

```python
# Monitorar perfil de liquidez
liquidity_consumer = LiquidityConsumer('WDOH25')
liquidity_consumer.start()

profile = liquidity_consumer.get_liquidity_profile()
print(f"Score de liquidez médio: {profile['avg_liquidity_score']}")
print(f"Spread médio: {profile['avg_spread']}")
```

### Estatísticas de Consumers

```python
# Verificar saúde dos consumers
stats = multi_consumer.get_stats()

for stream, stream_stats in stats.items():
    print(f"\nStream: {stream}")
    print(f"  Recebidas: {stream_stats['messages_received']}")
    print(f"  Processadas: {stream_stats['messages_processed']}")
    print(f"  Erros: {stream_stats['errors']}")
```

## 🔧 Troubleshooting

### Problema: "Connection refused" no Valkey
```bash
# Verificar se Valkey está rodando
docker ps | grep valkey

# Se não estiver, iniciar:
docker start valkey-trading

# Testar conexão
redis-cli ping
```

### Problema: Consumers não recebem dados
```python
# Verificar se publishers estão ativos
netstat -an | grep 5555  # Deve mostrar LISTEN

# Verificar configuração de porta
print(flow_consumer.port)  # Deve ser 5557 para flow
```

### Problema: Alta latência
```python
# Verificar métricas
metrics = infrastructure.get_performance_metrics()
if metrics['avg_latency_ms'] > 10:
    # Reduzir buffer size
    infrastructure.flow_analyzer.max_buffer_size = 500
    
    # Verificar CPU/memória
    # Considerar escalar horizontalmente
```

### Problema: Memória crescente no Valkey
```bash
# Verificar uso de memória
redis-cli info memory

# Ajustar maxlen dos streams
redis-cli XLEN order_flow:WDOH25

# Se necessário, trimmar manualmente
redis-cli XTRIM order_flow:WDOH25 MAXLEN 50000
```

## 🚀 Próximos Passos

### Semana 2: Algoritmos de Agentes
- Implementar agentes especialistas em flow
- Q-learning para decisões baseadas em OFI
- PPO para otimização de execução

### Semana 3: Sistema Multi-Agente
- Hierarquia de agentes
- Comunicação inter-agentes
- Consenso para decisões

### Semana 4: Testes e Otimização
- Backtesting com dados históricos
- Otimização de hiperparâmetros
- Validação em ambiente simulado

---

**Versão**: 1.0.0  
**Data**: Agosto 2025  
**Compatível com**: QuantumTrader_ML v2.0+