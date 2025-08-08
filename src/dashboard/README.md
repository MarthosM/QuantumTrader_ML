# 📊 QuantumTrader ML Dashboard

Dashboard em tempo real para monitoramento do sistema de trading, construído com Flask e WebSockets.

## 🚀 Características

### Interface Real-time
- **WebSockets**: Atualizações em tempo real sem refresh
- **Gráficos Interativos**: Curva de equity com Chart.js
- **Responsivo**: Funciona em desktop e mobile

### Métricas Monitoradas
- **Portfolio**: Valor total, P&L realizado/não realizado
- **Posições**: Todas as posições abertas com P&L individual
- **Ordens**: Status de ordens em tempo real
- **Risco**: Nível de risco, drawdown, exposição
- **Alertas**: Stop loss, circuit breakers, eventos importantes

### Visualizações
- Curva de equity em tempo real
- Tabelas de posições e ordens
- Histórico de trades recentes
- Alertas e notificações visuais

## 📦 Instalação

```bash
# Instalar dependências do dashboard
pip install -r requirements_dashboard.txt
```

## 🔧 Uso

### Standalone (para desenvolvimento)
```bash
# Executar dashboard sozinho
python src/dashboard/dashboard_integration.py
```

### Com Sistema Completo
```bash
# Executar sistema com dashboard
python examples/run_system_with_dashboard.py
```

### Configuração
```python
config = {
    'dashboard_host': '127.0.0.1',  # IP do servidor
    'dashboard_port': 5000,         # Porta do servidor
    'dashboard_debug': False        # Modo debug (True para desenvolvimento)
}
```

## 🌐 Acessando o Dashboard

Após iniciar, acesse:
```
http://localhost:5000
```

## 📡 API REST

O dashboard também expõe endpoints REST:

### Portfolio
```
GET /api/portfolio
```

### Posições
```
GET /api/positions
```

### Trades
```
GET /api/trades
```

### Métricas de Risco
```
GET /api/risk_metrics
```

### Ordens
```
GET /api/orders
```

### Curva de Equity
```
GET /api/equity_curve
```

## 🔌 WebSocket Events

### Eventos Emitidos pelo Servidor

#### portfolio_update
```javascript
{
    total_value: 100000.00,
    cash_balance: 50000.00,
    total_pnl: 500.00,
    daily_pnl: 100.00,
    unrealized_pnl: 50.00
}
```

#### positions_update
```javascript
[{
    symbol: "WDOU25",
    quantity: 2,
    entry_price: 5000.00,
    current_price: 5050.00,
    unrealized_pnl: 100.00
}]
```

#### risk_update
```javascript
{
    risk_level: "MEDIUM",
    current_exposure: 10000.00,
    current_drawdown: 0.02,
    is_locked: false
}
```

#### alert
```javascript
{
    type: "stop_loss",
    symbol: "WDOU25",
    details: "Stop loss triggered",
    severity: "warning",
    timestamp: "2024-01-01T10:00:00"
}
```

### Eventos do Cliente

#### request_update
```javascript
socket.emit('request_update', { type: 'all' });
// type: 'all' | 'portfolio' | 'positions' | 'orders' | 'risk'
```

## 🎨 Customização

### Modificar Cores/Tema
Edite o CSS em `templates/dashboard.html`:
```css
body {
    background-color: #1a1a1a;  /* Fundo escuro */
    color: #ffffff;             /* Texto claro */
}

.positive { color: #00ff00; }  /* Verde para lucro */
.negative { color: #ff0000; }  /* Vermelho para perda */
```

### Adicionar Novos Gráficos
```javascript
// Em dashboard.html
const newChart = new Chart(ctx, {
    type: 'bar',
    data: { /* seus dados */ },
    options: { /* suas opções */ }
});
```

### Adicionar Novos Widgets
```html
<!-- Em dashboard.html -->
<div class="card metric-card">
    <div class="metric-label">Nova Métrica</div>
    <div class="metric-value" id="newMetric">0</div>
</div>
```

## 🔒 Segurança

### Produção
- Use HTTPS com certificado SSL
- Configure autenticação (não implementada no demo)
- Limite CORS para domínios específicos
- Use servidor WSGI (Gunicorn, uWSGI)

### Exemplo com Gunicorn
```bash
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 src.dashboard.app:app
```

## 📊 Performance

### Otimizações
- Limita histórico a 500 pontos no gráfico
- Atualiza a cada 1 segundo (configurável)
- WebSockets para reduzir overhead
- Buffers circulares para dados históricos

### Monitoramento
- Verificar uso de memória do navegador
- Monitorar latência WebSocket
- Ajustar frequência de atualização se necessário

## 🐛 Troubleshooting

### Dashboard não carrega
- Verificar se o servidor está rodando
- Verificar porta 5000 não está em uso
- Verificar firewall/antivírus

### Dados não atualizam
- Verificar conexão WebSocket no console do navegador
- Verificar se componentes foram inicializados
- Verificar logs do servidor

### Performance ruim
- Reduzir frequência de atualização
- Limitar número de pontos no gráfico
- Verificar CPU/memória do servidor

## 🚀 Melhorias Futuras

1. **Autenticação**: Sistema de login/senha
2. **Múltiplos Dashboards**: Diferentes views para diferentes usuários
3. **Mobile App**: App nativo para iOS/Android
4. **Mais Gráficos**: Candlestick, volume, indicadores
5. **Controles**: Botões para pausar/iniciar trading
6. **Exportação**: Download de relatórios em PDF/Excel
7. **Alertas**: Notificações push/email/SMS
8. **Dark/Light Mode**: Temas customizáveis

## 📝 Notas de Desenvolvimento

- O dashboard é independente do sistema principal
- Usa callbacks para receber eventos dos componentes
- Thread-safe para múltiplas conexões simultâneas
- Pode ser estendido com novos componentes facilmente

---

Dashboard desenvolvido como parte do QuantumTrader ML v2.0