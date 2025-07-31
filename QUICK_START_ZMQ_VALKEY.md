# 🚀 Quick Start - ZMQ + Valkey Integration

## 📋 Pré-requisitos

- Python 3.8+
- Docker Desktop (para Valkey)
- Sistema ML Trading funcionando

## 🎯 Setup Rápido (5 minutos)

### 1. Execute o Setup
```bash
# Windows
setup_zmq_valkey.bat

# Linux/Mac
python setup_zmq_valkey.py
```

### 2. Configure o .env
```bash
# Copie o template
copy .env.zmq_valkey .env

# Edite com suas configurações
notepad .env
```

Configurações mínimas:
```env
# Habilitar gradualmente
ZMQ_ENABLED=true        # Fase 1
VALKEY_ENABLED=false    # Fase 2
TIME_TRAVEL_ENABLED=false # Fase 3
```

### 3. Inicie o Valkey
```bash
docker-compose -f docker-compose.valkey.yml up -d
```

## 🧪 Testes Rápidos

### Teste 1: ZMQ Publicação
```bash
# Terminal 1 - Publisher
python scripts\test_zmq_publisher.py

# Terminal 2 - Monitor
python scripts\monitor_zmq_valkey.py
```

### Teste 2: Valkey Time Travel
```bash
python scripts\test_valkey_time_travel.py
```

## 🔧 Integração com Sistema Atual

### Opção 1: Modificar main.py (Recomendado)
```python
# src/main.py
from trading_system_enhanced import TradingSystemEnhanced

# Substituir TradingSystem por TradingSystemEnhanced
system = TradingSystemEnhanced(config)
```

### Opção 2: Novo Entry Point
```python
# start_enhanced.py
from src.trading_system_enhanced import TradingSystemEnhanced

config = {
    'zmq_enabled': True,
    'valkey_enabled': True,
    # ... outras configs
}

system = TradingSystemEnhanced(config)
system.start()
```

## 📊 Validação da Integração

### 1. Sistema Original Funcionando
```bash
# Desabilitar tudo novo
set ZMQ_ENABLED=false
set VALKEY_ENABLED=false
python src/main.py
# ✅ Deve funcionar normalmente
```

### 2. Apenas ZMQ (Sem Valkey)
```bash
set ZMQ_ENABLED=true
set VALKEY_ENABLED=false
python src/main.py
# ✅ Sistema funciona + publica ZMQ
```

### 3. ZMQ + Valkey
```bash
set ZMQ_ENABLED=true
set VALKEY_ENABLED=true
python src/main.py
# ✅ Sistema completo enhanced
```

## 🎮 Comandos Úteis

### Verificar Valkey
```bash
# Status
docker ps | findstr valkey

# Logs
docker logs ml-trading-valkey

# Console Valkey
docker exec -it ml-trading-valkey valkey-cli

# Dentro do console
> INFO
> XLEN stream:ticks:WDOQ25
> XRANGE stream:ticks:WDOQ25 - + COUNT 10
```

### Debug ZMQ
```python
# test_zmq_consumer.py
import zmq

context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://localhost:5555")
subscriber.setsockopt(zmq.SUBSCRIBE, b"")

while True:
    topic, data = subscriber.recv_multipart()
    print(f"Recebido: {topic.decode()} - {data[:50]}...")
```

## 🚨 Troubleshooting

### Problema: "Valkey connection refused"
```bash
# Verificar se está rodando
docker ps

# Reiniciar
docker-compose -f docker-compose.valkey.yml restart

# Verificar logs
docker logs ml-trading-valkey
```

### Problema: "ZMQ Address already in use"
```bash
# Windows - Encontrar processo usando porta
netstat -ano | findstr :5555

# Matar processo (substitua PID)
taskkill /PID 12345 /F
```

### Problema: "Import error"
```bash
# Reinstalar dependências
pip install --upgrade pyzmq valkey orjson
```

## 📈 Próximos Passos

1. **Fase 1 (Dia 1-2)**: Validar ZMQ publishing
   - Confirmar dados sendo publicados
   - Verificar latência < 1ms
   
2. **Fase 2 (Dia 3-4)**: Ativar Valkey storage
   - Habilitar bridge ZMQ → Valkey
   - Verificar persistência de dados
   
3. **Fase 3 (Dia 5-6)**: Time Travel features
   - Implementar enhanced features
   - Testar queries históricas
   
4. **Fase 4 (Dia 7-8)**: Enhanced ML
   - Ativar predições melhoradas
   - Comparar performance

## 🎯 Checklist de Sucesso

- [ ] Setup executado sem erros
- [ ] Valkey respondendo (docker ps)
- [ ] ZMQ publicando dados
- [ ] Sistema original ainda funciona
- [ ] Monitor mostrando atividade
- [ ] Nenhum breaking change

---

**Dúvidas?** Verifique os logs em `logs/` ou execute o monitor para diagnóstico em tempo real.