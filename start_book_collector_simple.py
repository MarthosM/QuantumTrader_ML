"""
Script simples para iniciar coleta contínua
Executa diretamente sem subprocess
"""

import sys
import os
from datetime import datetime

print("\n" + "="*70)
print("INICIANDO COLETA CONTÍNUA SIMPLES")
print(f"Horário: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("="*70 + "\n")

# Importar e executar diretamente
try:
    from book_collector_continuous import ContinuousBookCollector, signal_handler
    import signal
    
    # Configurar handler de sinal
    signal.signal(signal.SIGINT, signal_handler)
    
    # Criar e inicializar coletor
    collector = ContinuousBookCollector()
    
    print("Inicializando sistema...")
    if not collector.initialize():
        print("\n[ERRO] Falha na inicialização")
        sys.exit(1)
        
    # Aguardar estabilização
    print("\nAguardando estabilização do sistema...")
    import time
    time.sleep(3)
    
    # Subscrever WDO
    print("\nSubscrevendo WDOU25...")
    collector.subscribe_wdo()
    
    # Aguardar dados começarem
    time.sleep(2)
    
    # Executar coleta contínua
    print("\nIniciando coleta contínua...")
    print("Pressione Ctrl+C para parar\n")
    
    try:
        collector.run_continuous()
    finally:
        # Finalizar
        collector.cleanup()
        
    print("\n[FIM] Coleta finalizada")
    print(f"Dados salvos em: data/realtime/book/{datetime.now().strftime('%Y%m%d')}/")
    
except Exception as e:
    print(f"\n[ERRO] Erro durante execução: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)