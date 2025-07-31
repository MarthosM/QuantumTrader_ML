#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste da integração Enhanced com sistema atual
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

def test_enhanced_system():
    """Testa sistema enhanced"""
    
    print("="*60)
    print("  Teste Sistema Enhanced")
    print("="*60)
    
    # Configuração de teste
    config = {
        'dll_path': os.getenv('PROFIT_DLL_PATH'),
        'user': os.getenv('PROFIT_USER'),
        'password': os.getenv('PROFIT_PASSWORD'),
        'ticker': 'WDOQ25',
        'symbols': ['WDOQ25'],  # Para criar streams
        'dev_mode': True  # Modo desenvolvimento
    }
    
    # Verificar se enhanced está habilitado
    from src.config.zmq_valkey_config import ZMQValkeyConfig
    
    print("\nConfigurações Enhanced:")
    print(f"ZMQ: {ZMQValkeyConfig.ZMQ_ENABLED}")
    print(f"Valkey: {ZMQValkeyConfig.VALKEY_ENABLED}")
    print(f"Time Travel: {ZMQValkeyConfig.TIME_TRAVEL_ENABLED}")
    
    if not ZMQValkeyConfig.is_enhanced_enabled():
        print("\n[AVISO] Nenhuma funcionalidade enhanced habilitada!")
        print("Configure no .env:")
        print("  ZMQ_ENABLED=true")
        print("  VALKEY_ENABLED=true")
        return
    
    try:
        # Importar sistema enhanced
        from src.trading_system_enhanced import TradingSystemEnhanced
        
        print("\n[INFO] Criando sistema enhanced...")
        system = TradingSystemEnhanced(config)
        
        print("\n[INFO] Iniciando sistema...")
        system.start()
        
        # Aguardar estabilização
        print("\n[INFO] Aguardando sistema estabilizar...")
        time.sleep(5)
        
        # Mostrar status
        status = system.get_enhanced_status()
        print("\n[Status Enhanced]:")
        print(f"Features habilitadas: {status['enhanced_features']}")
        
        if 'zmq_stats' in status:
            print(f"\nZMQ Stats:")
            for k, v in status['zmq_stats'].items():
                print(f"  {k}: {v}")
        
        if 'valkey_stats' in status:
            print(f"\nValkey Stats:")
            print(f"  Conectado: {status['valkey_stats']['connected']}")
            print(f"  Streams ativos: {status['valkey_stats']['active_streams']}")
        
        if 'bridge_stats' in status:
            print(f"\nBridge Stats:")
            for k, v in status['bridge_stats'].items():
                print(f"  {k}: {v}")
        
        # Testar time travel se disponível
        if ZMQValkeyConfig.TIME_TRAVEL_ENABLED and system.valkey_manager:
            print("\n[Time Travel Test]")
            
            # Aguardar alguns dados
            print("Aguardando dados...")
            time.sleep(10)
            
            # Query últimos 5 minutos
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=5)
            
            data = system.get_time_travel_data('WDOQ25', start_time, end_time)
            
            if data:
                print(f"Time travel retornou {len(data)} entries")
                if len(data) > 0:
                    print(f"Primeiro: {data[0].get('timestamp', 'N/A')}")
                    print(f"Último: {data[-1].get('timestamp', 'N/A')}")
            else:
                print("Sem dados time travel ainda")
        
        # Executar por 30 segundos
        print("\n[INFO] Sistema rodando por 30 segundos...")
        start = time.time()
        
        while time.time() - start < 30:
            # Mostrar progresso
            remaining = int(30 - (time.time() - start))
            print(f"\r[Rodando] {remaining}s restantes...", end="")
            time.sleep(1)
        
        print("\n\n[INFO] Parando sistema...")
        system.stop()
        
        # Status final
        final_status = system.get_enhanced_status()
        
        print("\n[Status Final]:")
        if 'zmq_stats' in final_status:
            print(f"Ticks publicados: {final_status['zmq_stats'].get('ticks_published', 0)}")
        if 'bridge_stats' in final_status:
            print(f"Ticks bridged: {final_status['bridge_stats'].get('ticks_bridged', 0)}")
        
        print("\n[SUCESSO] Teste concluído!")
        
    except Exception as e:
        print(f"\n[ERRO] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # Verificar .env
    if not os.path.exists('.env'):
        print("[ERRO] Arquivo .env não encontrado!")
        sys.exit(1)
    
    # Executar teste
    success = test_enhanced_system()
    
    sys.exit(0 if success else 1)