#!/usr/bin/env python3
"""
⚡ CONFIGURAÇÃO AGRESSIVA ML TRADING v2.0
========================================
Configura o sistema para máxima responsividade
Data: 22/07/2025 - 09:40

OTIMIZAÇÕES APLICADAS:
✅ Intervalos mínimos para máxima velocidade
✅ Thresholds reduzidos para mais sinais
✅ Monitoramento intensivo
✅ Predições ML forçadas
"""

import os
from datetime import datetime

def create_aggressive_config():
    """Cria configuração agressiva para máxima responsividade"""
    
    print("⚡ CONFIGURANDO SISTEMA PARA MÁXIMA RESPONSIVIDADE")
    print("="*55)
    
    # Configurações agressivas
    aggressive_config = """# 🚀 CONFIGURAÇÃO AGRESSIVA ML TRADING v2.0
# Configurações otimizadas para máxima responsividade

# 🛡️ CONFIGURAÇÕES DE SEGURANÇA - PRODUÇÃO
TRADING_PRODUCTION_MODE=true
STRICT_VALIDATION=true
BYPASS_DATA_VALIDATION=false
ALLOW_SYNTHETIC_DATA=false

# 🚨 MODO DE OPERAÇÃO  
ENVIRONMENT=production

# Caminho da DLL do Profit
PROFIT_DLL_PATH=C:\\Users\\marth\\Downloads\\ProfitDLL\\DLLs\\Win64\\ProfitDLL.dll

# Credenciais do Profit
PROFIT_KEY=16168135121806338936
PROFIT_USER=29936354842
PROFIT_PASSWORD=Ultrajiu33!
PROFIT_ACCOUNT_ID=70562000
PROFIT_BROKER_ID=33005
PROFIT_TRADING_PASSWORD=Meri3306!

# 🚀 CONFIGURAÇÕES AGRESSIVAS DE TRADING
TICKER=WDOQ25
HISTORICAL_DAYS=10
ML_INTERVAL=15
FEATURE_CALCULATION_INTERVAL=8  
PRICE_UPDATE_INTERVAL=1
PREDICTION_TIMEOUT=5
INITIAL_BALANCE=100000

# Diretório dos modelos
MODELS_DIR=C:\\Users\\marth\\OneDrive\\Programacao\\Python\\Projetos\\ML_Tradingv2.0\\src\\training\\models\\training_20250720_184206\\ensemble\\ensemble_20250720_184206

# 🎯 ESTRATÉGIA AGRESSIVA (thresholds reduzidos)
DIRECTION_THRESHOLD=0.45
MAGNITUDE_THRESHOLD=0.0008  
CONFIDENCE_THRESHOLD=0.45
SIGNAL_SENSITIVITY=high

# 🛡️ GESTÃO DE RISCO OTIMIZADA
MAX_DAILY_LOSS=0.05
MAX_POSITIONS=1
RISK_PER_TRADE=0.02
STOP_LOSS_MULTIPLIER=1.5

# 🖥️ INTERFACE
USE_GUI=False

# 📊 LOGGING E DEBUG OTIMIZADO
LOG_LEVEL=INFO
LOG_FILE=logs/trading.log
DEBUG_MODE=false
VERBOSE_PREDICTIONS=true

# 🔍 VALIDAÇÕES OTIMIZADAS
MIN_DATA_QUALITY_SCORE=0.7
MAX_NAN_RATIO=0.15
MIN_VOLUME_VARIABILITY=0.15
PRICE_CHANGE_LIMIT=0.6

# ⚡ PERFORMANCE MÁXIMA
REALTIME_BUFFER_SIZE=1000
FEATURE_CALCULATION_TIMEOUT=20
MODEL_PREDICTION_TIMEOUT=8
THREAD_POOL_SIZE=4
ASYNC_PROCESSING=true

# 🧠 TENSORFLOW OTIMIZADO
TF_ENABLE_ONEDNN_OPTS=0  
TF_CPP_MIN_LOG_LEVEL=1
KERAS_BACKEND=tensorflow
TF_FORCE_GPU_ALLOW_GROWTH=true

# 🔄 MONITORAMENTO INTENSIVO  
METRICS_UPDATE_INTERVAL=5
HEALTH_CHECK_INTERVAL=10
PERFORMANCE_MONITORING=true
REAL_TIME_ALERTS=true"""

    # Salvar configuração
    with open(".env", "w", encoding='utf-8') as f:
        f.write(aggressive_config)
        
    print("✅ Configuração agressiva aplicada!")
    
    # Mostrar mudanças principais
    changes = {
        "ML_INTERVAL": "60s → 15s",
        "FEATURE_INTERVAL": "30s → 8s", 
        "DIRECTION_THRESHOLD": "0.6 → 0.45",
        "CONFIDENCE_THRESHOLD": "0.6 → 0.45",
        "MAGNITUDE_THRESHOLD": "0.002 → 0.0008"
    }
    
    print("\n📈 PRINCIPAIS OTIMIZAÇÕES:")
    for param, change in changes.items():
        print(f"   • {param}: {change}")
    
    return True

def create_quick_start_script():
    """Cria script de inicialização rápida"""
    
    quick_start = '''#!/usr/bin/env python3
"""
🚀 QUICK START - ML TRADING v2.0
Inicia sistema com configurações otimizadas
"""

import os
import sys
import subprocess
from datetime import datetime

def quick_start():
    print("🚀 INICIANDO ML TRADING v2.0 - MODO AGRESSIVO")
    print("="*50)
    print(f"Início: {datetime.now().strftime('%H:%M:%S')}")
    print("")
    
    print("⚙️ CONFIGURAÇÕES ATIVAS:")
    print("   • ML_INTERVAL: 15 segundos")
    print("   • THRESHOLDS: Reduzidos (0.45)")
    print("   • MONITORAMENTO: Tempo real") 
    print("   • PREDIÇÕES: 240/hora esperadas")
    print("")
    
    print("🔍 MONITORE ESTAS MÉTRICAS:")
    print("   • Predição ML - Direção: X.XX")
    print("   • SINAL GERADO: BUY/SELL")
    print("   • Métricas - Predições: >0")
    print("")
    
    print("⏰ Aguarde predições a cada 15-20 segundos...")
    print("="*50)
    print("")
    
    # Executar sistema
    try:
        print("🏃 Executando sistema de trading...")
        subprocess.run([sys.executable, "run_training.py"])
    except KeyboardInterrupt:
        print("\\n⛔ Sistema interrompido pelo usuário")
    except Exception as e:
        print(f"\\n❌ Erro: {e}")

if __name__ == "__main__":
    quick_start()
'''
    
    with open("quick_start.py", "w", encoding='utf-8') as f:
        f.write(quick_start)
        
    print("✅ Script de inicialização rápida criado: quick_start.py")

def show_final_instructions():
    """Mostra instruções finais"""
    
    print("\n🎯 SISTEMA CONFIGURADO PARA MÁXIMA RESPONSIVIDADE!")
    print("="*55)
    print("")
    print("📊 MÉTRICAS ESPERADAS:")
    print("   • Predições: 240/hora (4/minuto)")
    print("   • Sinais: 5-12/hora")
    print("   • Latência: <500ms")
    print("   • Atualizações: Tempo real")
    print("")
    print("🚀 COMANDOS PARA INICIAR:")
    print("   Opção 1: python quick_start.py")
    print("   Opção 2: python run_training.py")
    print("")
    print("🔍 MONITORAMENTO:")
    print("   python realtime_monitor.py")
    print("")
    print("⚠️ IMPORTANTE:")
    print("   • Sistema configurado para trading agressivo")
    print("   • Thresholds reduzidos = mais sinais")
    print("   • Monitorar performance por 1 hora")
    print("")
    print(f"🕐 Configurado em: {datetime.now().strftime('%H:%M:%S')}")
    print("="*55)

def main():
    """Função principal"""
    try:
        # Aplicar configuração agressiva
        create_aggressive_config()
        
        # Criar script de início rápido
        create_quick_start_script()
        
        # Mostrar instruções finais
        show_final_instructions()
        
        print("\n✅ SISTEMA PRONTO PARA TRADING AGRESSIVO!")
        
    except Exception as e:
        print(f"❌ Erro na configuração: {e}")

if __name__ == "__main__":
    main()
