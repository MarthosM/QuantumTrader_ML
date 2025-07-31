# GUI Extensions - ZMQ/Valkey Monitor Integration

Este diretório contém extensões para o monitor GUI existente, adicionando capacidades de monitoramento do sistema ZMQ + Valkey.

## Componentes

### 1. `zmq_valkey_monitor_extension.py`
Extensão principal que adiciona novas abas ao monitor Tkinter existente:
- **Aba ZMQ/Valkey**: Status e estatísticas em tempo real
- **Aba Time Travel**: Features avançadas quando disponíveis

### 2. `activate_enhanced_monitor.py`
Script exemplo para ativar o monitor com extensões enhanced.

### 3. `monitor_integration_patch.py`
Exemplos de como integrar a extensão ao código existente.

## Como Usar

### Opção 1: Script Standalone
```python
python src/gui_extensions/activate_enhanced_monitor.py
```

### Opção 2: Integração Mínima
No seu código existente, após criar o monitor:

```python
from src.trading_monitor_gui import TradingMonitorGUI
from src.gui_extensions.monitor_integration_patch import setup_enhanced_monitor_if_available

# Criar monitor normalmente
monitor = TradingMonitorGUI(trading_system)

# Adicionar extensões se disponível
setup_enhanced_monitor_if_available(monitor)

# Executar
monitor.run()
```

### Opção 3: Modificação do Monitor Existente
Adicione ao `__init__` do `TradingMonitorGUI`:

```python
# No final do __init__
try:
    from gui_extensions.zmq_valkey_monitor_extension import integrate_zmq_valkey_monitor
    self.zmq_valkey_extension = integrate_zmq_valkey_monitor(self, self.trading_system)
except:
    self.zmq_valkey_extension = None
```

## Features

### Aba ZMQ/Valkey
- **Status Geral**: Estado de cada componente (ON/OFF)
- **ZeroMQ Stats**: Ticks publicados, trades, erros, taxa/seg
- **Valkey Stats**: Streams ativos, entries, memória, latência
- **Bridge Stats**: Dados bridged, taxa de sucesso, último tick

### Aba Time Travel (quando disponível)
- **Enhanced Features**: 8 features exclusivas com valores em tempo real
- **Métricas**: Lookback, data points, qualidade dos dados, cache hits

## Requisitos

- Sistema enhanced ativo (`TradingSystemEnhanced`)
- ZMQ e/ou Valkey habilitados no `.env`
- Monitor GUI Tkinter existente

## Comportamento

1. **Detecção Automática**: Verifica se sistema enhanced está ativo
2. **Zero Breaking Changes**: Não modifica funcionalidade existente
3. **Fallback Gracioso**: Se enhanced não disponível, monitor funciona normalmente
4. **Updates Integrados**: Atualiza junto com o monitor principal

## Configuração

A extensão respeita as configurações do sistema principal:

```env
# .env
ZMQ_ENABLED=true
VALKEY_ENABLED=true
TIME_TRAVEL_ENABLED=true
```

## Troubleshooting

### Extensão não aparece
- Verificar se sistema enhanced está ativo
- Confirmar que ZMQ/Valkey estão habilitados
- Checar logs para erros de importação

### Dados não atualizam
- Verificar conexão com Valkey
- Confirmar que bridge está rodando
- Checar se há dados sendo publicados

### Performance
- A extensão adiciona mínimo overhead
- Updates são otimizados para não impactar o monitor principal
- Cache é usado para reduzir queries