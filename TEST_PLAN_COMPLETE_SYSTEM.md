# 🧪 Plano de Testes - Sistema Completo

## 📋 Visão Geral

Este documento define o plano completo de testes para validar cada componente do sistema de trading, desde a coleta de dados até a execução de ordens.

## 🎯 Objetivos dos Testes

1. **Validar Funcionalidade**: Cada componente funciona conforme esperado
2. **Verificar Performance**: Latências dentro dos limites aceitáveis
3. **Testar Confiabilidade**: Sistema resiliente a falhas
4. **Confirmar Integração**: Componentes trabalham juntos corretamente

## 📊 Estrutura de Testes

### Nível 1: Testes Unitários (Componentes Isolados)
### Nível 2: Testes de Integração (Componentes Conectados)
### Nível 3: Testes End-to-End (Sistema Completo)
### Nível 4: Testes de Stress (Condições Extremas)

---

## 🔍 NÍVEL 1: Testes Unitários

### 1.1 Data Collection Tests

#### Test: Book Collector
```python
# tests/unit/test_book_collector.py

def test_book_collector_connection():
    """Testa conexão com ProfitDLL"""
    collector = BookCollector(config)
    assert collector.connect() == True
    
def test_book_data_parsing():
    """Testa parsing de dados do book"""
    raw_data = get_mock_book_data()
    parsed = collector.parse_book_data(raw_data)
    assert 'price' in parsed
    assert 'quantity' in parsed
    assert parsed['price'] > 0

def test_book_data_storage():
    """Testa armazenamento em parquet"""
    data = get_sample_book_data()
    path = collector.save_data(data)
    assert Path(path).exists()
    loaded = pd.read_parquet(path)
    assert len(loaded) == len(data)
```

#### Test: Data Synchronizer
```python
# tests/unit/test_data_synchronizer.py

def test_temporal_alignment():
    """Testa alinhamento temporal tick/book"""
    sync = DataSynchronizer(buffer_ms=100)
    
    # Adicionar dados com timestamps diferentes
    tick_time = datetime.now()
    book_time = tick_time + timedelta(milliseconds=50)
    
    sync.add_tick({'timestamp': tick_time, 'price': 5000})
    sync.add_book({'timestamp': book_time, 'bid': 4999})
    
    aligned = sync.get_synchronized_data()
    time_diff = abs((aligned['tick_time'] - aligned['book_time']).total_seconds())
    assert time_diff < 0.1  # < 100ms

def test_buffer_overflow():
    """Testa comportamento com buffer cheio"""
    sync = DataSynchronizer(buffer_size=100)
    
    # Adicionar mais dados que o buffer suporta
    for i in range(200):
        sync.add_tick({'timestamp': datetime.now(), 'price': 5000 + i})
    
    assert len(sync.tick_buffer) <= 100
    assert sync.get_oldest_tick() is not None
```

### 1.2 Feature Engineering Tests

#### Test: ML Features
```python
# tests/unit/test_ml_features.py

def test_feature_calculation_speed():
    """Testa velocidade de cálculo de features"""
    candles = get_sample_candles(1000)
    ml_features = MLFeaturesV3()
    
    start = time.time()
    features = ml_features.calculate_all_features(candles)
    elapsed = time.time() - start
    
    assert elapsed < 0.02  # < 20ms
    assert not features.isnull().any().any()

def test_feature_validity():
    """Testa validade das features calculadas"""
    candles = get_sample_candles(100)
    features = ml_features.calculate_all_features(candles)
    
    # Verificar ranges esperados
    assert features['returns_1'].between(-0.1, 0.1).all()
    assert features['rsi'].between(0, 100).all()
    assert features['volume_ma_10'] > 0
```

### 1.3 Model Tests

#### Test: Model Loading
```python
# tests/unit/test_model_loading.py

def test_tick_model_loading():
    """Testa carregamento do modelo tick"""
    model_path = 'models/csv_5m/lightgbm_tick.txt'
    model = lgb.Booster(model_file=model_path)
    
    assert model is not None
    assert model.num_trees() > 0
    
def test_model_prediction_speed():
    """Testa velocidade de predição"""
    model = load_tick_model()
    features = get_sample_features(1)
    
    start = time.time()
    prediction = model.predict(features)
    elapsed = time.time() - start
    
    assert elapsed < 0.01  # < 10ms
    assert prediction.shape[0] == 1
```

### 1.4 Strategy Tests

#### Test: Hybrid Strategy
```python
# tests/unit/test_hybrid_strategy.py

def test_regime_detection():
    """Testa detecção de regime"""
    strategy = HybridStrategy(config)
    features = create_trend_features()  # Features indicando tendência
    
    regime, confidence = strategy.detect_regime(features)
    assert regime in ['trend', 'range', 'undefined']
    assert 0 <= confidence <= 1

def test_signal_generation():
    """Testa geração de sinais"""
    strategy = HybridStrategy(config)
    tick_features = get_tick_features()
    book_features = get_book_features()
    
    signal_info = strategy.get_hybrid_signal(tick_features, book_features)
    assert signal_info['signal'] in [-1, 0, 1]
    assert 0 <= signal_info['confidence'] <= 1
```

### 1.5 Order Management Tests

#### Test: Order Manager
```python
# tests/unit/test_order_manager.py

def test_order_validation():
    """Testa validação de ordens"""
    order_manager = OrderManager(config)
    
    # Ordem válida
    valid_order = {
        'action': 'BUY',
        'quantity': 1,
        'symbol': 'WDOU25'
    }
    assert order_manager.validate_order(valid_order) == True
    
    # Ordem inválida
    invalid_order = {
        'action': 'BUY',
        'quantity': -1  # Quantidade negativa
    }
    assert order_manager.validate_order(invalid_order) == False

def test_order_state_machine():
    """Testa máquina de estados da ordem"""
    order = Order()
    
    # Transições válidas
    order.submit()
    assert order.state == 'PENDING'
    
    order.fill()
    assert order.state == 'FILLED'
    
    # Transição inválida
    with pytest.raises(InvalidStateTransition):
        order.cancel()  # Não pode cancelar ordem filled
```

---

## 🔗 NÍVEL 2: Testes de Integração

### 2.1 Data Pipeline Integration

```python
# tests/integration/test_data_pipeline.py

def test_tick_book_integration():
    """Testa integração tick + book em tempo real"""
    # Iniciar collectors
    tick_collector = TickCollector(config)
    book_collector = BookCollector(config)
    synchronizer = DataSynchronizer()
    
    # Coletar dados por 10 segundos
    tick_collector.start()
    book_collector.start()
    
    time.sleep(10)
    
    # Verificar sincronização
    synced_data = synchronizer.get_synchronized_data()
    assert len(synced_data) > 0
    assert 'tick_price' in synced_data.columns
    assert 'book_bid' in synced_data.columns
    
    # Verificar alinhamento temporal
    time_diffs = synced_data['tick_time'] - synced_data['book_time']
    assert time_diffs.abs().max() < pd.Timedelta('100ms')
```

### 2.2 Feature to Model Integration

```python
# tests/integration/test_feature_model_pipeline.py

def test_feature_model_pipeline():
    """Testa pipeline features → modelo"""
    # Dados de entrada
    candles = load_test_candles()
    
    # Calcular features
    ml_features = MLFeaturesV3()
    features = ml_features.calculate_all_features(candles)
    
    # Carregar modelo
    model = load_tick_model()
    required_features = load_model_features()
    
    # Verificar compatibilidade
    assert all(f in features.columns for f in required_features)
    
    # Fazer predição
    X = features[required_features]
    predictions = model.predict(X)
    
    assert len(predictions) == len(features)
    assert all(0 <= p <= 1 for p in predictions.max(axis=1))
```

### 2.3 Strategy to Execution Integration

```python
# tests/integration/test_strategy_execution.py

def test_signal_to_order():
    """Testa fluxo sinal → ordem"""
    # Setup
    strategy = HybridStrategy(config)
    order_manager = OrderManager(config)
    risk_manager = RiskManager(config)
    
    # Gerar sinal
    signal_info = {
        'signal': 1,  # BUY
        'confidence': 0.75,
        'price': 5000
    }
    
    # Validar com risk manager
    approved = risk_manager.validate_signal(signal_info)
    assert approved == True
    
    # Criar ordem
    order = order_manager.create_order_from_signal(signal_info)
    assert order['action'] == 'BUY'
    assert order['quantity'] > 0
    
    # Mock de execução
    order_id = order_manager.send_order(order)
    assert order_id is not None
```

---

## 🚀 NÍVEL 3: Testes End-to-End

### 3.1 Full Trading Cycle Test

```python
# tests/e2e/test_full_trading_cycle.py

def test_complete_trading_cycle():
    """Testa ciclo completo de trading"""
    # 1. Inicializar sistema
    system = TradingSystem(config)
    system.initialize()
    
    # 2. Iniciar coleta de dados
    system.start_data_collection()
    
    # 3. Aguardar dados suficientes
    wait_for_data(min_candles=100)
    
    # 4. Verificar geração de sinais
    signals = []
    for _ in range(60):  # 1 minuto
        signal = system.get_current_signal()
        if signal['signal'] != 0:
            signals.append(signal)
        time.sleep(1)
    
    assert len(signals) > 0, "Nenhum sinal gerado"
    
    # 5. Verificar execução de ordem
    if signals:
        order_status = system.get_order_status(signals[0]['order_id'])
        assert order_status in ['FILLED', 'PENDING', 'CANCELLED']
    
    # 6. Verificar position tracking
    position = system.get_current_position()
    assert 'quantity' in position
    assert 'avg_price' in position
    assert 'pnl' in position
```

### 3.2 Paper Trading Test

```python
# tests/e2e/test_paper_trading.py

def test_paper_trading_session():
    """Testa sessão completa de paper trading"""
    # Configurar paper trading
    config['paper_trading'] = True
    system = TradingSystem(config)
    
    # Executar por 1 hora
    system.start()
    start_time = time.time()
    
    metrics = {
        'trades': 0,
        'wins': 0,
        'losses': 0,
        'max_drawdown': 0
    }
    
    while time.time() - start_time < 3600:  # 1 hora
        # Atualizar métricas
        current_stats = system.get_statistics()
        metrics['trades'] = current_stats['total_trades']
        metrics['wins'] = current_stats['winning_trades']
        
        # Verificar saúde do sistema
        assert system.is_healthy()
        
        time.sleep(10)
    
    # Validar resultados
    assert metrics['trades'] > 0
    win_rate = metrics['wins'] / metrics['trades']
    assert 0.4 <= win_rate <= 0.7  # Win rate razoável
```

---

## 💪 NÍVEL 4: Testes de Stress

### 4.1 High Frequency Test

```python
# tests/stress/test_high_frequency.py

def test_high_frequency_data():
    """Testa sistema com alta frequência de dados"""
    system = TradingSystem(config)
    
    # Simular 1000 updates por segundo
    start = time.time()
    updates_processed = 0
    
    for _ in range(10000):  # 10 segundos de teste
        tick_data = generate_random_tick()
        book_data = generate_random_book()
        
        system.process_tick(tick_data)
        system.process_book(book_data)
        updates_processed += 2
        
    elapsed = time.time() - start
    updates_per_second = updates_processed / elapsed
    
    assert updates_per_second > 900  # Processar pelo menos 90%
    assert system.get_dropped_messages() < 100  # < 1% dropped
```

### 4.2 Recovery Test

```python
# tests/stress/test_recovery.py

def test_connection_recovery():
    """Testa recuperação de falhas de conexão"""
    system = TradingSystem(config)
    system.start()
    
    # Simular perda de conexão
    system.connection_manager.disconnect()
    time.sleep(5)
    
    # Verificar tentativa de reconexão
    assert system.connection_manager.is_reconnecting()
    
    # Aguardar reconexão
    wait_for_connection(timeout=30)
    assert system.connection_manager.is_connected()
    
    # Verificar que sistema continua funcionando
    signal = system.get_current_signal()
    assert signal is not None
```

### 4.3 Memory Leak Test

```python
# tests/stress/test_memory_leak.py

def test_memory_stability():
    """Testa estabilidade de memória em execução longa"""
    import psutil
    process = psutil.Process()
    
    system = TradingSystem(config)
    system.start()
    
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_readings = []
    
    # Executar por 1 hora
    for i in range(360):  # 10 segundos * 360 = 1 hora
        time.sleep(10)
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_readings.append(current_memory)
        
        # Verificar crescimento excessivo
        if current_memory > initial_memory * 2:
            pytest.fail(f"Memória dobrou: {initial_memory}MB → {current_memory}MB")
    
    # Verificar tendência
    memory_growth = memory_readings[-1] - memory_readings[0]
    assert memory_growth < 500  # < 500MB de crescimento em 1 hora
```

---

## 📊 Métricas de Sucesso

### Performance
- ✅ Latência dados → ordem: < 100ms
- ✅ Taxa de processamento: > 1000 msg/s
- ✅ Uso de CPU: < 50%
- ✅ Uso de memória: < 4GB

### Confiabilidade
- ✅ Uptime: > 99.9%
- ✅ Recovery time: < 30s
- ✅ Data loss: 0%
- ✅ Order failures: < 0.1%

### Trading
- ✅ Win rate: > 55%
- ✅ Sharpe ratio: > 1.5
- ✅ Max drawdown: < 10%
- ✅ Trades por dia: 5-20

## 🏃 Como Executar os Testes

### Executar todos os testes
```bash
pytest tests/ -v --cov=src --cov-report=html
```

### Executar por nível
```bash
# Unitários
pytest tests/unit/ -v

# Integração
pytest tests/integration/ -v

# End-to-end
pytest tests/e2e/ -v

# Stress
pytest tests/stress/ -v --durations=10
```

### Executar teste específico
```bash
pytest tests/unit/test_hybrid_strategy.py::test_signal_generation -v
```

## 📝 Relatório de Cobertura

Após executar os testes, verificar:
- Coverage report: `htmlcov/index.html`
- Métricas no terminal
- Logs detalhados em `test_logs/`

---

**Meta de Cobertura**: > 80%  
**Tempo Total de Testes**: ~30 minutos  
**Frequência**: A cada commit (CI/CD)