#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script para ZMQ + Valkey integration
Configura e valida ambiente para implantação
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import json

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import locale
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

def print_header(text):
    """Imprime cabeçalho formatado"""
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def check_python_version():
    """Verifica versão do Python"""
    print("[Python] Verificando Python...")
    if sys.version_info < (3, 8):
        print("[ERRO] Python 3.8+ necessario")
        return False
    print(f"[OK] Python {sys.version.split()[0]} OK")
    return True

def check_dependencies():
    """Verifica e instala dependências"""
    print_header("Verificando Dependências")
    
    required_packages = {
        'pyzmq': '25.1.0',
        'valkey': '6.0.0', 
        'orjson': '3.9.0',
        'pandas': '2.0.0',
        'numpy': '1.24.0'
    }
    
    missing = []
    
    for package, min_version in required_packages.items():
        try:
            __import__(package)
            print(f"[OK] {package} instalado")
        except ImportError:
            missing.append(f"{package}>={min_version}")
            print(f"[ERRO] {package} nao encontrado")
    
    if missing:
        print("\n[Info] Instalando pacotes faltantes...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("[OK] Dependencias instaladas")
    
    return True

def setup_directories():
    """Cria estrutura de diretórios"""
    print_header("Criando Estrutura de Diretórios")
    
    directories = [
        "src/integration",
        "src/config", 
        "scripts",
        "logs",
        "data/valkey"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"[OK] {dir_path}")
    
    return True

def create_docker_compose():
    """Cria docker-compose para Valkey"""
    print_header("Criando Docker Compose")
    
    docker_compose = """version: '3.8'

services:
  valkey:
    image: valkey/valkey:latest
    container_name: ml-trading-valkey
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - ./data/valkey:/data
    command: >
      valkey-server
      --maxmemory 4gb
      --maxmemory-policy allkeys-lru
      --save 60 1000
      --save 300 10
      --save 3600 1
      --appendonly yes
      --appendfsync everysec
    environment:
      - VALKEY_REPLICATION_MODE=master
    healthcheck:
      test: ["CMD", "valkey-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  valkey-data:
    driver: local
"""
    
    with open("docker-compose.valkey.yml", "w") as f:
        f.write(docker_compose)
    
    print("[OK] docker-compose.valkey.yml criado")
    return True

def create_env_template():
    """Cria template .env"""
    print_header("Criando Template de Configuração")
    
    env_template = """# ZMQ + Valkey Configuration
# Copie este arquivo para .env e ajuste conforme necessário

# Sistema Original
PROFIT_DLL_PATH=C:/Path/To/ProfitDLL.dll
PROFIT_USER=seu_usuario
PROFIT_PASSWORD=sua_senha

# ZeroMQ Configuration
ZMQ_ENABLED=false
ZMQ_TICK_PORT=5555
ZMQ_BOOK_PORT=5556
ZMQ_HISTORY_PORT=5557
ZMQ_SIGNAL_PORT=5558

# Valkey Configuration
VALKEY_ENABLED=false
VALKEY_HOST=localhost
VALKEY_PORT=6379
VALKEY_PASSWORD=
VALKEY_DB=0

# Time Travel Features
TIME_TRAVEL_ENABLED=false
TIME_TRAVEL_LOOKBACK_MINUTES=120
TIME_TRAVEL_PERCENTAGE=1.0

# Enhanced ML
ENHANCED_ML_ENABLED=false
FORCE_FAST_MODE=false
FALLBACK_ON_ERROR=true

# Monitoring
MONITORING_ENABLED=true
ALERT_ON_FALLBACK=false
LOG_LEVEL=INFO

# Performance
MAX_MEMORY_GB=4
STREAM_MAX_LENGTH=100000
STREAM_RETENTION_DAYS=30
"""
    
    if not Path(".env").exists():
        with open(".env.zmq_valkey", "w") as f:
            f.write(env_template)
        print("[OK] .env.zmq_valkey template criado")
        print("   [AVISO] Copie para .env e configure")
    else:
        print("[Info] .env ja existe, template salvo em .env.zmq_valkey")
    
    return True

def test_zmq():
    """Testa funcionamento do ZMQ"""
    print_header("Testando ZeroMQ")
    
    try:
        import zmq
        
        # Teste de pub/sub local
        context = zmq.Context()
        
        # Publisher
        publisher = context.socket(zmq.PUB)
        publisher.bind("tcp://127.0.0.1:15555")
        
        # Subscriber
        subscriber = context.socket(zmq.SUB)
        subscriber.connect("tcp://127.0.0.1:15555")
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        
        # Aguardar conexão
        time.sleep(0.1)
        
        # Enviar mensagem
        test_msg = b"test_message"
        publisher.send(test_msg)
        
        # Receber com timeout
        subscriber.setsockopt(zmq.RCVTIMEO, 1000)
        received = subscriber.recv()
        
        # Cleanup
        publisher.close()
        subscriber.close()
        context.term()
        
        if received == test_msg:
            print("[OK] ZeroMQ funcionando corretamente")
            return True
        else:
            print("[ERRO] ZeroMQ: mensagem nao recebida")
            return False
            
    except Exception as e:
        print(f"[ERRO] Erro ao testar ZeroMQ: {e}")
        return False

def start_valkey():
    """Inicia Valkey via Docker"""
    print_header("Iniciando Valkey")
    
    # Verificar se Docker está instalado
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except:
        print("[ERRO] Docker nao encontrado. Instale Docker primeiro.")
        return False
    
    # Verificar se já está rodando
    result = subprocess.run(
        ["docker", "ps", "-q", "-f", "name=ml-trading-valkey"],
        capture_output=True,
        text=True
    )
    
    if result.stdout.strip():
        print("[Info] Valkey ja esta rodando")
        return True
    
    # Iniciar Valkey
    print("[Info] Iniciando Valkey...")
    try:
        subprocess.run(
            ["docker-compose", "-f", "docker-compose.valkey.yml", "up", "-d"],
            check=True
        )
        
        # Aguardar inicialização
        print("[Info] Aguardando Valkey inicializar...")
        time.sleep(5)
        
        # Verificar saúde
        for i in range(10):
            result = subprocess.run(
                ["docker", "exec", "ml-trading-valkey", "valkey-cli", "ping"],
                capture_output=True,
                text=True
            )
            if "PONG" in result.stdout:
                print("[OK] Valkey iniciado e respondendo")
                return True
            time.sleep(1)
        
        print("[ERRO] Valkey nao respondeu no tempo esperado")
        return False
        
    except Exception as e:
        print(f"[ERRO] Erro ao iniciar Valkey: {e}")
        return False

def test_valkey_connection():
    """Testa conexão com Valkey"""
    print_header("Testando Conexão Valkey")
    
    try:
        import valkey
        
        client = valkey.Valkey(host='localhost', port=6379, decode_responses=True)
        
        # Teste ping
        if client.ping():
            print("[OK] Conexao com Valkey OK")
        
        # Teste escrita/leitura
        test_key = "test:setup"
        test_value = "setup_test_value"
        
        client.set(test_key, test_value)
        retrieved = client.get(test_key)
        
        if retrieved == test_value:
            print("[OK] Leitura/Escrita OK")
        
        # Teste streams
        stream_key = "test:stream"
        client.xadd(stream_key, {"test": "data"})
        entries = client.xrange(stream_key)
        
        if entries:
            print("[OK] Streams funcionando")
        
        # Cleanup
        client.delete(test_key, stream_key)
        
        return True
        
    except Exception as e:
        print(f"[ERRO] Erro ao conectar com Valkey: {e}")
        return False

def create_test_scripts():
    """Cria scripts de teste"""
    print_header("Criando Scripts de Teste")
    
    # Script de teste ZMQ
    zmq_test = '''#!/usr/bin/env python3
"""Teste básico de publicação ZMQ"""

import zmq
import time
import json
from datetime import datetime

def test_zmq_publisher():
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5555")
    
    print("ZMQ Publisher iniciado na porta 5555")
    print("Publicando dados de teste...")
    
    symbols = ["WDOQ25", "WINQ25", "INDQ25"]
    
    try:
        while True:
            for symbol in symbols:
                tick_data = {
                    "symbol": symbol,
                    "price": 5000 + (hash(str(time.time())) % 100),
                    "volume": 100 + (hash(str(time.time())) % 50),
                    "timestamp": datetime.now().isoformat()
                }
                
                topic = f"tick_{symbol}".encode()
                data = json.dumps(tick_data).encode()
                
                publisher.send_multipart([topic, data])
                print(f"Publicado: {symbol} - {tick_data['price']}")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\\nParando publisher...")
    finally:
        publisher.close()
        context.term()

if __name__ == "__main__":
    test_zmq_publisher()
'''
    
    # Script de teste Valkey
    valkey_test = '''#!/usr/bin/env python3
"""Teste de time travel com Valkey"""

import valkey
from datetime import datetime, timedelta
import json

def test_time_travel():
    client = valkey.Valkey(host='localhost', port=6379)
    
    print("Testando Time Travel com Valkey...")
    
    # Criar dados históricos
    stream_key = "stream:ticks:WDOQ25"
    
    # Simular 1 hora de dados
    now = datetime.now()
    
    for minutes_ago in range(60, 0, -1):
        timestamp = now - timedelta(minutes=minutes_ago)
        timestamp_ms = int(timestamp.timestamp() * 1000)
        
        tick_data = {
            "symbol": "WDOQ25",
            "price": str(5000 + minutes_ago),
            "volume": str(100 + minutes_ago % 10),
            "timestamp": timestamp.isoformat()
        }
        
        client.xadd(
            stream_key,
            tick_data,
            id=f"{timestamp_ms}-0"
        )
    
    print(f"[OK] Adicionados 60 minutos de dados históricos")
    
    # Time travel query - últimos 10 minutos
    end_time = now
    start_time = now - timedelta(minutes=10)
    
    start_id = f"{int(start_time.timestamp() * 1000)}-0"
    end_id = f"{int(end_time.timestamp() * 1000)}-0"
    
    entries = client.xrange(stream_key, start_id, end_id)
    
    print(f"\\n[Info] Time Travel Query (últimos 10 minutos):")
    print(f"Encontrados {len(entries)} ticks")
    
    if entries:
        first_tick = {k.decode(): v.decode() for k, v in entries[0][1].items()}
        last_tick = {k.decode(): v.decode() for k, v in entries[-1][1].items()}
        
        print(f"Primeiro tick: {first_tick['timestamp']} - Preço: {first_tick['price']}")
        print(f"Último tick: {last_tick['timestamp']} - Preço: {last_tick['price']}")

if __name__ == "__main__":
    test_time_travel()
'''
    
    # Salvar scripts
    with open("scripts/test_zmq_publisher.py", "w") as f:
        f.write(zmq_test)
    
    with open("scripts/test_valkey_time_travel.py", "w") as f:
        f.write(valkey_test)
    
    # Tornar executáveis (Linux/Mac)
    if os.name != 'nt':
        os.chmod("scripts/test_zmq_publisher.py", 0o755)
        os.chmod("scripts/test_valkey_time_travel.py", 0o755)
    
    print("[OK] Scripts de teste criados em scripts/")
    return True

def create_monitor_script():
    """Cria script de monitoramento"""
    print_header("Criando Script de Monitoramento")
    
    monitor_script = '''#!/usr/bin/env python3
"""Monitor simples para ZMQ + Valkey"""

import zmq
import valkey
import json
import time
from datetime import datetime

def monitor_system():
    # Conectar ZMQ
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://localhost:5555")
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.setsockopt(zmq.RCVTIMEO, 1000)
    
    # Conectar Valkey
    valkey_client = valkey.Valkey(host='localhost', port=6379)
    
    print("[Monitor] ZMQ + Valkey")
    print("="*50)
    
    zmq_count = 0
    last_stats_time = time.time()
    
    try:
        while True:
            # Monitor ZMQ
            try:
                topic, data = subscriber.recv_multipart()
                zmq_count += 1
                
                if zmq_count % 10 == 0:
                    tick = json.loads(data)
                    print(f"ZMQ: {tick['symbol']} - ${tick['price']} - {datetime.now():%H:%M:%S}")
                    
            except zmq.Again:
                pass
            
            # Stats a cada 5 segundos
            if time.time() - last_stats_time > 5:
                # Contar streams no Valkey
                streams = valkey_client.keys("stream:*")
                
                total_entries = 0
                for stream in streams:
                    info = valkey_client.xinfo_stream(stream)
                    total_entries += info['length']
                
                print(f"\\n[Stats] ZMQ msgs: {zmq_count} | Valkey streams: {len(streams)} | Total entries: {total_entries}")
                print("-"*50)
                
                last_stats_time = time.time()
                
    except KeyboardInterrupt:
        print("\\nMonitor parado")
    finally:
        subscriber.close()
        context.term()

if __name__ == "__main__":
    monitor_system()
'''
    
    with open("scripts/monitor_zmq_valkey.py", "w") as f:
        f.write(monitor_script)
    
    if os.name != 'nt':
        os.chmod("scripts/monitor_zmq_valkey.py", 0o755)
    
    print("[OK] Script de monitoramento criado")
    return True

def generate_summary():
    """Gera resumo da instalação"""
    print_header("Resumo da Instalação")
    
    from datetime import datetime
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "directories_created": [
            "src/integration",
            "src/config",
            "scripts",
            "logs",
            "data/valkey"
        ],
        "files_created": [
            "docker-compose.valkey.yml",
            ".env.zmq_valkey",
            "scripts/test_zmq_publisher.py",
            "scripts/test_valkey_time_travel.py",
            "scripts/monitor_zmq_valkey.py"
        ],
        "next_steps": [
            "1. Configure .env com suas credenciais",
            "2. Inicie Valkey: docker-compose -f docker-compose.valkey.yml up -d",
            "3. Teste ZMQ: python scripts/test_zmq_publisher.py",
            "4. Teste Valkey: python scripts/test_valkey_time_travel.py",
            "5. Monitor: python scripts/monitor_zmq_valkey.py"
        ]
    }
    
    with open("setup_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n[Info] Proximos Passos:")
    for step in summary["next_steps"]:
        print(f"   {step}")
    
    print("\n[OK] Setup concluido! Resumo salvo em setup_summary.json")
    return True

def main():
    """Função principal do setup"""
    print("[SETUP] ZMQ + Valkey para ML Trading System")
    print("="*60)
    
    steps = [
        ("Verificando Python", check_python_version),
        ("Verificando Dependências", check_dependencies),
        ("Criando Diretórios", setup_directories),
        ("Criando Docker Compose", create_docker_compose),
        ("Criando Configuração", create_env_template),
        ("Testando ZeroMQ", test_zmq),
        ("Iniciando Valkey", start_valkey),
        ("Testando Valkey", test_valkey_connection),
        ("Criando Scripts de Teste", create_test_scripts),
        ("Criando Monitor", create_monitor_script),
        ("Gerando Resumo", generate_summary)
    ]
    
    failed = False
    
    for step_name, step_func in steps:
        try:
            if not step_func():
                print(f"\n[AVISO] {step_name} falhou, mas continuando...")
                failed = True
        except Exception as e:
            print(f"\n[ERRO] Erro em {step_name}: {e}")
            failed = True
    
    if failed:
        print("\n[AVISO] Setup concluido com avisos. Verifique os erros acima.")
    else:
        print("\n[SUCESSO] Setup concluido com sucesso!")
    
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())