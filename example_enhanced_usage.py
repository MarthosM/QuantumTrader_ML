#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemplo de uso do sistema Enhanced com ZMQ + Valkey
"""

import os
import time
from datetime import datetime, timedelta

def example_basic_usage():
    """Exemplo básico de uso"""
    print("\n=== Exemplo 1: Uso Básico ===")
    
    # 1. Habilitar no .env
    print("1. Configure no .env:")
    print("   ZMQ_ENABLED=true")
    print("   VALKEY_ENABLED=true")
    
    # 2. Importar e usar
    print("\n2. No seu código:")
    print("""
from src.trading_system_enhanced import TradingSystemEnhanced

config = {
    'dll_path': 'path/to/dll',
    'user': 'seu_usuario',
    'password': 'sua_senha',
    'ticker': 'WDOQ25'
}

# Sistema detecta automaticamente se deve usar enhanced
system = TradingSystemEnhanced(config)
system.start()
""")

def example_time_travel():
    """Exemplo de time travel"""
    print("\n=== Exemplo 2: Time Travel ===")
    
    print("1. Habilite time travel no .env:")
    print("   TIME_TRAVEL_ENABLED=true")
    
    print("\n2. Use time travel para análise histórica:")
    print("""
# Buscar dados dos últimos 30 minutos
end_time = datetime.now()
start_time = end_time - timedelta(minutes=30)

# Query time travel
data = system.get_time_travel_data(
    symbol='WDOQ25',
    start_time=start_time,
    end_time=end_time,
    data_type='ticks'  # ou 'candles_1m'
)

# Converter para DataFrame se necessário
df = system.valkey_manager.time_travel_to_dataframe(
    'WDOQ25', start_time, end_time
)

print(f"Encontrados {len(df)} ticks históricos")
print(f"Preço médio: {df['price'].mean():.2f}")
print(f"Volume total: {df['volume'].sum():,.0f}")
""")

def example_enhanced_features():
    """Exemplo de features enhanced"""
    print("\n=== Exemplo 3: Features Enhanced ===")
    
    print("Com time travel habilitado, calcule features avançadas:")
    print("""
# Features normais
features = system.feature_engine.calculate_features(df_atual)

# Features com time travel (lookback maior)
enhanced_features = system.feature_engine.calculate_with_time_travel(
    symbol='WDOQ25',
    lookback_minutes=120  # 2 horas de histórico
)

# Features exclusivas do time travel
print(f"Data points usados: {enhanced_features['data_points']}")
print(f"Time travel usado: {enhanced_features['time_travel_used']}")
""")

def example_monitoring():
    """Exemplo de monitoramento"""
    print("\n=== Exemplo 4: Monitoramento ===")
    
    print("Monitore o sistema enhanced em tempo real:")
    print("""
# Obter status completo
status = system.get_enhanced_status()

# Status dos componentes
print(f"ZMQ ativo: {status['enhanced_features']['zmq_enabled']}")
print(f"Valkey ativo: {status['enhanced_features']['valkey_enabled']}")

# Estatísticas ZMQ
if 'zmq_stats' in status:
    print(f"Ticks publicados: {status['zmq_stats']['ticks_published']}")
    print(f"Erros ZMQ: {status['zmq_stats']['errors']}")

# Estatísticas Valkey
if 'valkey_stats' in status:
    print(f"Streams ativos: {status['valkey_stats']['active_streams']}")
    
# Estatísticas Bridge
if 'bridge_stats' in status:
    print(f"Ticks bridged: {status['bridge_stats']['ticks_bridged']}")
    print(f"Último tick: {status['bridge_stats']['last_tick_time']}")
""")

def example_gradual_adoption():
    """Exemplo de adoção gradual"""
    print("\n=== Exemplo 5: Adoção Gradual ===")
    
    print("Habilite funcionalidades gradualmente:")
    print("""
# Fase 1: Apenas ZMQ (publicação de dados)
ZMQ_ENABLED=true
VALKEY_ENABLED=false
# Sistema publica dados mas não armazena

# Fase 2: ZMQ + Valkey (armazenamento)
ZMQ_ENABLED=true
VALKEY_ENABLED=true
# Dados são publicados e armazenados

# Fase 3: Time Travel (análise histórica)
TIME_TRAVEL_ENABLED=true
# Features enhanced disponíveis

# Fase 4: ML Enhanced (predições melhoradas)
ENHANCED_ML_ENABLED=true
# Sistema completo enhanced
""")

def example_fallback():
    """Exemplo de fallback"""
    print("\n=== Exemplo 6: Fallback Automático ===")
    
    print("Sistema volta ao original se enhanced falhar:")
    print("""
# Configure fallback no .env
FALLBACK_ON_ERROR=true

# Se Valkey cair, sistema continua sem time travel
# Se ZMQ falhar, sistema continua sem publicação
# Sistema NUNCA para por causa de enhanced

# Verificar se enhanced está ativo
if hasattr(system, 'valkey_manager') and system.valkey_manager:
    print("Time travel disponível")
else:
    print("Usando sistema original")
""")

def main():
    """Executa exemplos"""
    print("="*60)
    print("  Exemplos de Uso - Sistema Enhanced")
    print("="*60)
    
    example_basic_usage()
    example_time_travel()
    example_enhanced_features()
    example_monitoring()
    example_gradual_adoption()
    example_fallback()
    
    print("\n" + "="*60)
    print("Para testar agora:")
    print("1. Configure .env com ZMQ_ENABLED=true")
    print("2. Execute: python test_enhanced_integration.py")
    print("3. Ou use: python src/main_enhanced.py")
    print("="*60)

if __name__ == "__main__":
    main()