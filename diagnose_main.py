#!/usr/bin/env python3
"""
Teste de diagnóstico do main.py
"""

import os
import sys
import logging
import traceback
from dotenv import load_dotenv

print("🔍 DIAGNÓSTICO DO MAIN.PY")
print("=" * 50)

try:
    print("1. Verificando diretório atual...")
    print(f"   Diretório: {os.getcwd()}")
    
    print("2. Verificando sys.path...")
    for i, path in enumerate(sys.path[:5]):
        print(f"   [{i}] {path}")
    
    print("3. Adicionando src ao path...")
    # Adicionar src ao path
    src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_path not in sys.path:
        sys.path.append(src_path)
    print(f"   Adicionado: {src_path}")
    
    print("4. Configurando logging...")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('debug_main.log', encoding='utf-8')
        ]
    )
    logger = logging.getLogger('MainDiagnostic')
    logger.info("Logging configurado")
    
    print("5. Tentando importar TradingSystem...")
    from trading_system import TradingSystem
    print("   ✅ TradingSystem importado com sucesso")
    
    print("6. Carregando configuração .env...")
    # Tentar diferentes caminhos para o .env
    possible_env_paths = [
        '.env',
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'),
        r'C:\Users\marth\OneDrive\Programacao\Python\Projetos\ML_Tradingv2.0\.env'
    ]
    
    env_loaded = False
    for env_path in possible_env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            env_loaded = True
            print(f"   ✅ .env carregado: {env_path}")
            break
    
    if not env_loaded:
        print("   ❌ Nenhum arquivo .env encontrado!")
        print("   Caminhos testados:")
        for path in possible_env_paths:
            print(f"     - {path}")
        
    print("7. Verificando variáveis de ambiente...")
    required_vars = ['PROFIT_KEY', 'PROFIT_USER', 'PROFIT_PASSWORD']
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var}: {'*' * len(value)}")
        else:
            print(f"   ❌ {var}: NÃO DEFINIDA")
    
    print("8. Criando configuração...")
    config = {
        'dll_path': os.getenv("PROFIT_DLL_PATH", r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"),
        'key': os.getenv('PROFIT_KEY'),
        'username': os.getenv('PROFIT_USER'),
        'password': os.getenv('PROFIT_PASSWORD'),
        'account_id': os.getenv('PROFIT_ACCOUNT_ID'),
        'broker_id': os.getenv('PROFIT_BROKER_ID'),
        'trading_password': os.getenv('PROFIT_TRADING_PASSWORD'),
        'models_dir': os.getenv('MODELS_DIR', r'C:\Users\marth\OneDrive\Programacao\Python\Projetos\ML_Tradingv2.0\src\models\models_regime3'),
        'ticker': os.getenv('TICKER'),
        'historical_days': int(os.getenv('HISTORICAL_DAYS', '10')),
        'ml_interval': int(os.getenv('ML_INTERVAL', '60')),
        'initial_balance': float(os.getenv('INITIAL_BALANCE', '100000')),
    }
    
    # Verificar credenciais obrigatórias
    required_fields = ['key', 'username', 'password']
    missing_fields = [field for field in required_fields if not config.get(field)]
    
    if missing_fields:
        print(f"   ❌ Campos obrigatórios não configurados: {', '.join(missing_fields)}")
        print("   Sistema não pode continuar sem credenciais")
        sys.exit(1)
    else:
        print("   ✅ Todas as credenciais obrigatórias estão presentes")
    
    print("9. Tentando criar TradingSystem...")
    system = TradingSystem(config)
    print("   ✅ TradingSystem criado com sucesso")
    
    print("10. Tentando inicializar sistema...")
    if system.initialize():
        print("   ✅ Sistema inicializado com sucesso")
        
        print("11. Tentando iniciar operação...")
        if system.start():
            print("   ✅ Sistema iniciado com sucesso!")
            print("   Sistema está em operação...")
            
            # Deixar rodando por alguns segundos para teste
            import time
            time.sleep(10)
            
            print("   Parando sistema de teste...")
            if hasattr(system, 'stop'):
                system.stop()
            
        else:
            print("   ❌ Falha ao iniciar operação")
            sys.exit(1)
    else:
        print("   ❌ Falha na inicialização do sistema")
        sys.exit(1)
        
    print("\n✅ DIAGNÓSTICO CONCLUÍDO COM SUCESSO!")
    
except Exception as e:
    print(f"\n❌ ERRO DURANTE DIAGNÓSTICO:")
    print(f"   Tipo: {type(e).__name__}")
    print(f"   Mensagem: {str(e)}")
    print(f"\n📋 TRACEBACK COMPLETO:")
    traceback.print_exc()
    sys.exit(1)
