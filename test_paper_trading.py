"""
Script de teste para Paper Trading
"""

import sys
import os
import logging
import time
from datetime import datetime
import signal

sys.path.insert(0, os.path.abspath('src'))

from paper_trading.paper_trader_v3 import PaperTraderV3

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Variável global para controlar a execução
running = True

def signal_handler(signum, frame):
    global running
    print("\n\nSinal de interrupção recebido. Parando...")
    running = False

# Registrar handler para Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def main():
    print("="*60)
    print("TESTE DE PAPER TRADING V3")
    print("="*60)
    print(f"Início: {datetime.now()}")
    print("Pressione Ctrl+C para parar")
    print("-"*60)
    
    # Configuração customizada para teste
    config = {
        'initial_capital': 100000.0,
        'position_size': 1,
        'confidence_threshold': 0.60,  # Reduzido para gerar mais trades
        'probability_threshold': 0.55,
        'min_time_between_trades': 30,  # 30 segundos entre trades
        'paper_trading_hours': {
            'start': 0,  # Trading 24h para teste
            'end': 24
        }
    }
    
    # Criar paper trader
    paper_trader = PaperTraderV3(config)
    
    try:
        # Iniciar
        paper_trader.start()
        
        # Loop principal
        last_display = time.time()
        
        while running:
            time.sleep(1)
            
            # Atualizar display a cada 5 segundos
            if time.time() - last_display > 5:
                summary = paper_trader.account.get_account_summary()
                
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Trades: {summary['total_trades']}, "
                      f"Posições: {len(summary['positions'])}, "
                      f"PnL: R$ {summary['total_pnl']:.2f} "
                      f"({summary['return_pct']:.2f}%)", end='', flush=True)
                
                last_display = time.time()
        
    except Exception as e:
        print(f"\n\nErro: {e}")
        
    finally:
        print("\n\nParando paper trading...")
        paper_trader.stop()
        print("Paper trading finalizado.")

if __name__ == "__main__":
    main()