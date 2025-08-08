# 🏗️ Arquitetura do Sistema - Atual vs Objetivo

## 📊 Estado Atual do Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                      CAMADA DE DADOS                             │
├─────────────────────────┬───────────────────────────────────────┤
│   ✅ Book Collector     │   ✅ Historical CSV                   │
│   └─ ProfitDLL         │   └─ 5M records                      │
│   └─ Parquet files     │   └─ Tick data                       │
└─────────────────────────┴───────────────────────────────────────┘
                    ⬇️ (Parcialmente conectado)
┌─────────────────────────────────────────────────────────────────┐
│                    CAMADA DE FEATURES                            │
├─────────────────────────┬───────────────────────────────────────┤
│   ✅ MLFeaturesV3       │   ✅ BookFeatures                    │
│   ✅ TechnicalIndicators│   ✅ Feature Engineering             │
└─────────────────────────┴───────────────────────────────────────┘
                    ⬇️
┌─────────────────────────────────────────────────────────────────┐
│                    CAMADA DE MODELOS                             │
├─────────────────────────┬───────────────────────────────────────┤
│   ✅ Tick Model         │   ✅ Book Model                      │
│   └─ 47% accuracy      │   └─ 69% accuracy                    │
│   ✅ HybridStrategy     │   ✅ AdaptiveStrategy                │
└─────────────────────────┴───────────────────────────────────────┘
                    ⬇️
┌─────────────────────────────────────────────────────────────────┐
│                    CAMADA DE EXECUÇÃO                            │
├─────────────────────────┬───────────────────────────────────────┤
│   ⚠️  OrderManager      │   ❌ PositionTracker                 │
│   ⚠️  RiskManager       │   ❌ P&L Calculator                  │
└─────────────────────────┴───────────────────────────────────────┘
                    ⬇️
┌─────────────────────────────────────────────────────────────────┐
│                    CAMADA DE MONITORAMENTO                       │
├─────────────────────────┬───────────────────────────────────────┤
│   ✅ AdaptiveMonitor    │   ❌ Real-time Dashboard             │
│   ✅ Logging System     │   ❌ Alert System                    │
└─────────────────────────┴───────────────────────────────────────┘

Legenda: ✅ Implementado  ⚠️ Parcial  ❌ Não implementado
```

## 🎯 Sistema Objetivo (Target)

```
┌─────────────────────────────────────────────────────────────────┐
│                 CAMADA DE DADOS UNIFICADA                        │
├─────────────────────────────────────────────────────────────────┤
│                    DataSynchronizer                              │
│  ┌──────────────┐     ┌──────────────┐    ┌──────────────┐    │
│  │ Tick Stream  │ ──> │   Temporal   │ <──│ Book Stream  │    │
│  │ (ProfitDLL)  │     │ Alignment    │    │ (ProfitDLL)  │    │
│  └──────────────┘     └──────────────┘    └──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                    ⬇️ Unified Data Stream
┌─────────────────────────────────────────────────────────────────┐
│                  PIPELINE DE FEATURES                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐     ┌──────────────┐    ┌──────────────┐    │
│  │Feature Cache │ ──> │  Parallel    │ <──│ NaN Handler  │    │
│  │   < 20ms     │     │ Processing   │    │              │    │
│  └──────────────┘     └──────────────┘    └──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                    ⬇️ Feature Matrix
┌─────────────────────────────────────────────────────────────────┐
│              SISTEMA DE PREDIÇÃO HMARL                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐     ┌──────────────┐    ┌──────────────┐    │
│  │ Tick Model   │     │   Hybrid     │    │ Book Model   │    │
│  │              │ ──> │  Strategy    │ <──│              │    │
│  └──────────────┘     │  + Regime    │    └──────────────┘    │
│                       └──────────────┘                          │
│                    ⬇️ Signal + Confidence                       │
└─────────────────────────────────────────────────────────────────┘
                    ⬇️
┌─────────────────────────────────────────────────────────────────┐
│               SISTEMA DE EXECUÇÃO                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐     ┌──────────────┐    ┌──────────────┐    │
│  │Order Manager │     │   Position   │    │Risk Manager  │    │
│  │              │ <-> │   Tracker    │ <->│              │    │
│  │ • Send Order │     │ • Open Pos   │    │ • Stop Loss  │    │
│  │ • Confirm    │     │ • P&L Calc   │    │ • Take Profit│    │
│  │ • Cancel     │     │ • History    │    │ • Max Exposure│   │
│  └──────────────┘     └──────────────┘    └──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                    ⬇️
┌─────────────────────────────────────────────────────────────────┐
│            SISTEMA DE MONITORAMENTO                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐     ┌──────────────┐    ┌──────────────┐    │
│  │  Real-time   │     │   Alert      │    │   Backup &   │    │
│  │  Dashboard   │     │   System     │    │   Recovery   │    │
│  └──────────────┘     └──────────────┘    └──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Fluxo de Dados Completo

```
1. COLETA DE DADOS (Real-time)
   ┌────────┐      ┌────────┐
   │  Tick  │      │  Book  │
   └───┬────┘      └───┬────┘
       └──────┬────────┘
              ⬇️
   
2. SINCRONIZAÇÃO
   ┌─────────────────┐
   │DataSynchronizer │ <- Alinha temporalmente
   └────────┬────────┘    (buffer 100ms)
            ⬇️

3. FEATURES
   ┌─────────────────┐
   │FeaturePipeline │ <- Calcula em paralelo
   └────────┬────────┘    (cache ativo)
            ⬇️

4. PREDIÇÃO HMARL
   ┌─────────────────┐
   │ Tick Model (47%)│ ─┐
   └─────────────────┘  │
                        ├─> HybridStrategy
   ┌─────────────────┐  │   (weighted avg)
   │ Book Model (69%)│ ─┘
   └─────────────────┘
            ⬇️

5. DECISÃO
   ┌─────────────────┐
   │ Signal Generator│ <- Aplica thresholds
   └────────┬────────┘    e regime rules
            ⬇️

6. EXECUÇÃO
   ┌─────────────────┐
   │  Order Manager  │ <- Envia ordem
   └────────┬────────┘    via ProfitDLL
            ⬇️

7. MONITORAMENTO
   ┌─────────────────┐
   │Position Tracker │ <- Atualiza P&L
   └────────┬────────┘    e métricas
            ⬇️

8. FEEDBACK
   ┌─────────────────┐
   │ Online Learning │ <- Atualiza modelos
   └─────────────────┘    continuamente
```

## 📋 Componentes a Implementar

### 1. DataSynchronizer 🔄
```python
class DataSynchronizer:
    """Sincroniza tick e book data em tempo real"""
    
    def __init__(self, buffer_ms=100):
        self.tick_buffer = deque()
        self.book_buffer = deque()
        self.sync_window = buffer_ms
        
    def add_tick(self, tick_data):
        # Adiciona com timestamp preciso
        
    def add_book(self, book_data):
        # Adiciona com timestamp preciso
        
    def get_synchronized_data(self):
        # Retorna dados alinhados temporalmente
```

### 2. OrderManager Completo 📊
```python
class OrderManager:
    """Gerencia ciclo completo de ordens"""
    
    def send_order(self, signal_info):
        # 1. Validar signal
        # 2. Calcular size
        # 3. Enviar via ProfitDLL
        # 4. Aguardar confirmação
        # 5. Atualizar estado
        
    def cancel_order(self, order_id):
        # Cancelar ordem pendente
        
    def get_order_status(self, order_id):
        # Retornar estado atual
```

### 3. PositionTracker 📈
```python
class PositionTracker:
    """Rastreia posições e calcula P&L"""
    
    def update_position(self, order_fill):
        # Atualizar posição atual
        
    def calculate_pnl(self, current_price):
        # P&L realizado e não-realizado
        
    def get_exposure(self):
        # Exposição total ao risco
```

### 4. Real-time Dashboard 📊
```python
class DashboardServer:
    """Servidor web para dashboard"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        
    def emit_update(self, data):
        # Enviar atualização real-time
        
    def get_metrics(self):
        # Retornar métricas atuais
```

## 🧪 Validação por Componente

### Teste DataSync
```bash
python tests/test_data_synchronizer.py
# Valida: Alinhamento < 100ms
```

### Teste OrderManager
```bash
python tests/test_order_manager.py
# Valida: Ciclo completo de ordem
```

### Teste PositionTracker
```bash
python tests/test_position_tracker.py
# Valida: Cálculo correto de P&L
```

### Teste End-to-End
```bash
python tests/test_full_system.py
# Valida: Sistema completo funcionando
```

## 🎯 Critérios de Sucesso

1. **Latência**: Dados → Ordem < 100ms
2. **Confiabilidade**: 99.9% uptime
3. **Accuracy**: Win rate > 55%
4. **Risk**: Max drawdown < 10%
5. **Scalability**: Suporta múltiplos símbolos

---

**Próximo Passo**: Implementar DataSynchronizer  
**Prioridade**: Alta  
**Estimativa**: 3-5 dias