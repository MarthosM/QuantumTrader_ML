# Monitor GUI - Trading System ML v2.0

## 📊 **Monitor Gráfico em Tempo Real**

O **Monitor GUI** é uma interface gráfica completa para monitoramento do Trading System ML v2.0 em tempo real, exibindo:

- 🎯 **Predições ML**: Direção, confiança, magnitude, regime
- 📈 **Dados de Candles**: OHLCV em tempo real 
- 💹 **Métricas Trading**: P&L, win rate, posições
- ⚡ **Sistema**: CPU, memória, uptime
- 🚨 **Alertas**: Notificações críticas em tempo real

---

## 🚀 **Inicialização Automática**

### Integração no Trading System

O monitor GUI é **automaticamente inicializado** quando:

```python
# trading_system.py
config = {
    'use_gui': True,  # 🔑 Chave principal!
    # ... outras configurações
}

system = TradingSystem(config)
system.start()  # GUI abre automaticamente
```

### Execução Standalone (Demonstração)

```bash
# Demo com dados simulados
python run_monitor_gui.py
```

---

## 🖥️ **Interface Gráfica**

### Layout Principal

```
┌─────────────────────────────────────────────────────────────┐
│ Trading System ML v2.0                      Sistema: Running │
├─────────────────────────────────────────────────────────────┤
│ ┌─── Dados de Trading ────┐ ┌─── Monitoramento ────┐        │
│ │                         │ │                      │        │
│ │ 🎯 Última Predição ML   │ │ 📊 Métricas         │        │
│ │ ├ Direção: +0.820      │ │ ├ Trading            │        │
│ │ ├ Confiança: 91.2%     │ │ ├ Sistema            │        │
│ │ ├ Magnitude: 0.0047    │ │ └ Posições           │        │
│ │ ├ Ação: BUY            │ │                      │        │
│ │ └ Regime: Trend Up     │ │ ⚠️ Alertas            │        │
│ │                        │ │ (Lista de alertas)   │        │
│ │ 📈 Último Candle       │ │                      │        │
│ │ ├ Open: 123,456.50     │ │                      │        │
│ │ ├ High: 123,789.25     │ │                      │        │
│ │ ├ Low:  123,234.75     │ │                      │        │
│ │ ├ Close: 123,678.25    │ │                      │        │
│ │ ├ Volume: 15,750       │ │                      │        │
│ │ └ Var: +0.18%          │ │                      │        │
│ └─────────────────────────┘ └────────────────────┘        │
├─────────────────────────────────────────────────────────────┤
│ ▶ Iniciar Monitor  ⏸ Parar Monitor    Última: 14:23:45   │
└─────────────────────────────────────────────────────────────┘
```

### Cores Intuitivas

- 🟢 **Verde**: Lucro, trades positivos, BUY
- 🔴 **Vermelho**: Perda, trades negativos, SELL  
- 🟡 **Amarelo**: Neutro, HOLD, aguardando
- 🔵 **Azul**: Informações, títulos, sistema

---

## 📋 **Dados Exibidos**

### 🎯 Predições ML
```python
{
    'direction': 0.820,      # -1 a +1
    'confidence': 0.912,     # 0 a 1  
    'magnitude': 0.0047,     # Força da predição
    'action': 'BUY',         # BUY/SELL/HOLD
    'regime': 'trend_up',    # regime de mercado
    'timestamp': '14:23:45'  # Hora da predição
}
```

### 📈 Dados de Candle
```python
{
    'open': 123456.50,
    'high': 123789.25,
    'low': 123234.75,
    'close': 123678.25,
    'volume': 15750,
    'variation_pct': 0.18,   # Variação %
    'timestamp': '14:23:00'
}
```

### 💹 Métricas Trading
```python
{
    'daily_pnl': 1250.75,    # P&L do dia
    'trades_today': 8,        # Trades executados
    'win_rate': 0.75,        # Taxa de acerto
    'active_positions': 1,    # Posições abertas
    'balance': 150000.0,     # Saldo total
    'available': 142300.0    # Disponível
}
```

### ⚡ Métricas Sistema
```python
{
    'cpu_percent': 22.5,     # Uso CPU %
    'memory_mb': 387.2,      # Memória MB
    'threads': 12,           # Threads ativas  
    'uptime': '3h:58m:45s',  # Tempo ativo
    'ml_predictions': 156,   # Predições feitas
    'signals_generated': 89  # Sinais gerados
}
```

---

## ⚙️ **Configuração**

### Parâmetros no Config

```python
config = {
    # Monitor GUI
    'use_gui': True,                # Habilitar GUI
    'gui_update_interval': 1,       # Intervalo atualização (seg)
    'gui_auto_start': True,         # Auto-iniciar monitoramento
    
    # Outras configurações do sistema...
}
```

### Personalização Visual

```python
# No código trading_monitor_gui.py
colors = {
    'profit': '#00FF00',     # Verde para lucros
    'loss': '#FF0000',       # Vermelho para perdas  
    'neutral': '#FFFF00',    # Amarelo neutro
    'bg_dark': '#2b2b2b',    # Fundo escuro
    'accent': '#007ACC'      # Azul para títulos
}

fonts = {
    'title': ('Arial', 14, 'bold'),
    'data': ('Courier', 10),        # Fonte monospace para dados
    'status': ('Arial', 12, 'bold')
}
```

---

## 🔄 **Integração com Trading System**

### Coleta de Dados

O monitor acessa diretamente:

```python
# Dados coletados automaticamente
trading_system.last_prediction          # Última predição ML
trading_system.data_structure.candles   # Candles atuais
trading_system.active_positions          # Posições abertas
trading_system.account_info              # Info da conta
trading_system.metrics.metrics           # Métricas ML
```

### Atualização Automática

- ⏰ **Intervalo**: 1 segundo (configurável)
- 🔄 **Threading**: Não bloqueia sistema principal
- 📡 **Real-time**: Dados sempre atualizados
- 🛡️ **Fallbacks**: Continua funcionando mesmo com erros

---

## 🚨 **Sistema de Alertas**

### Alertas Implementados

1. **📉 Drawdown Crítico**
   - Trigger: > 5% (configurável)
   - Nível: `critical`

2. **📊 Win Rate Baixo**
   - Trigger: < 45% (configurável)  
   - Nível: `warning`

3. **💾 Memória Alta**
   - Trigger: > 80% uso
   - Nível: `warning`

4. **⏱️ Latência Alta**
   - Trigger: > 100ms
   - Nível: `warning`

5. **🤖 Model Drift**
   - Trigger: Score > 0.1
   - Nível: `warning`

### Configuração de Alertas

```python
alerts_config = {
    'max_drawdown_alert': 0.05,      # 5%
    'min_win_rate_alert': 0.45,      # 45%
    'max_latency_alert': 100,        # 100ms
    'max_drift_alert': 0.1           # 0.1 score
}
```

---

## 🛠️ **Desenvolvimento e Extensões**

### Adicionar Novos Widgets

```python
def _create_custom_section(self, parent):
    """Adiciona seção personalizada"""
    custom_frame = ttk.LabelFrame(parent, text="📊 Minha Seção")
    custom_frame.pack(fill=tk.X, pady=10)
    
    # Seus widgets aqui...
```

### Adicionar Métricas

```python
def _update_custom_metrics(self):
    """Atualiza métricas customizadas"""
    # Coletar dados do trading_system
    custom_data = self.trading_system.get_custom_data()
    
    # Atualizar displays
    self.custom_labels['metric'].config(text=f"{custom_data:.2f}")
```

### Threading Personalizado

```python
def _custom_update_loop(self):
    """Loop personalizado de atualização"""
    while self.running:
        # Sua lógica personalizada
        time.sleep(self.custom_interval)
```

---

## 📚 **Exemplos de Uso**

### 1. Demo Standalone

```bash
python run_monitor_gui.py
# Escolha opção 1 para demo com dados simulados
```

### 2. Integração Completa

```python
from trading_system import TradingSystem

config = {
    'dll_path': 'caminho/para/ProfitDLL.dll',
    'username': 'usuario',
    'password': 'senha',
    'models_dir': 'models/',
    'use_gui': True,  # 🔑 Importante!
    'ticker': 'WDOQ25'
}

system = TradingSystem(config)

if system.initialize():
    system.start()  # GUI abre automaticamente
```

### 3. Monitor Externo

```python
from trading_monitor_gui import create_monitor_gui

# Assumindo que você tem um sistema rodando
monitor = create_monitor_gui(trading_system)
monitor.run()
```

---

## 🔧 **Troubleshooting**

### Problemas Comuns

**❌ GUI não abre**
- Verificar `use_gui: True` no config
- Instalar tkinter: `sudo apt-get install python3-tk` (Linux)

**❌ Dados não aparecem**
- Sistema precisa estar inicializado
- Verificar se `trading_system.is_running = True`

**❌ Interface trava**  
- Verificar threading
- Logs em caso de exceção

**❌ Métricas incorretas**
- Verificar métodos `_get_*_metrics_safe()`
- Fallbacks implementados para robustez

---

## 🎯 **Funcionalidades Futuras**

- [ ] **Gráficos**: Plots em tempo real com matplotlib
- [ ] **WebSocket**: Interface web complementar
- [ ] **Notificações**: Sistema alerts desktop
- [ ] **Export**: Salvar dados históricos
- [ ] **Themes**: Temas visuais (claro/escuro)
- [ ] **Layouts**: Configurações de layout salvos

---

## ✅ **Status de Implementação**

- [x] ✅ Interface básica tkinter
- [x] ✅ Integração automática com TradingSystem
- [x] ✅ Display predições ML tempo real
- [x] ✅ Display dados candles OHLCV
- [x] ✅ Métricas trading e sistema
- [x] ✅ Sistema de alertas integrado
- [x] ✅ Threading não-bloqueante
- [x] ✅ Fallbacks seguros
- [x] ✅ Demo standalone funcional

**🎉 Monitor GUI 100% Implementado e Funcional!**

---

*Desenvolvido para Trading System ML v2.0 - Julho 2025*
