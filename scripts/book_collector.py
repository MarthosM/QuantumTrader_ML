"""
Coletor de Book de Ofertas para Pregão
Este script coleta book de ofertas continuamente durante o pregão
"""

import os
import sys
import time
import subprocess
import signal
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.realtime_book_collector import RealtimeBookCollector

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BookCollector')

# Variável global para controle
collector = None
server_process = None


def signal_handler(signum, frame):
    """Handler para Ctrl+C"""
    logger.info("\n⚠️ Interrupção recebida - finalizando coleta...")
    
    global collector, server_process
    
    if collector:
        collector.stop_collection()
    
    if server_process:
        logger.info("Finalizando servidor...")
        server_process.terminate()
        server_process.wait(timeout=5)
        if server_process.poll() is None:
            server_process.kill()
    
    logger.info("✅ Coleta finalizada com segurança")
    sys.exit(0)


def check_server_running():
    """Verifica se o servidor está rodando"""
    try:
        from multiprocessing.connection import Client
        
        client = Client(('localhost', 6789), authkey=b'profit_dll_secret')
        client.send({'type': 'ping'})
        
        if client.poll(timeout=2):
            response = client.recv()
            if response.get('type') == 'pong':
                client.close()
                return True
        
        client.close()
        return False
        
    except:
        return False


def start_server_if_needed():
    """Inicia o servidor se não estiver rodando"""
    if check_server_running():
        logger.info("✅ Servidor já está rodando")
        return None
    
    logger.info("🚀 Iniciando servidor ProfitDLL...")
    
    server_script = "src/integration/profit_dll_server.py"
    process = subprocess.Popen(
        [sys.executable, server_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    logger.info(f"Servidor iniciado com PID: {process.pid}")
    
    # Aguardar inicialização
    time.sleep(10)
    
    # Verificar se ainda está rodando
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        logger.error("Servidor morreu durante inicialização!")
        logger.error(f"STDOUT: {stdout}")
        logger.error(f"STDERR: {stderr}")
        return None
    
    return process


def get_current_wdo_contract():
    """Detecta o contrato WDO atual"""
    current_date = datetime.now()
    month_codes = {
        1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
        7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
    }
    
    # REGRA WDO: Sempre usa próximo mês
    if current_date.month == 12:
        next_month = 1
        next_year = str(current_date.year + 1)[-2:]
    else:
        next_month = current_date.month + 1
        next_year = str(current_date.year)[-2:]
    
    contract = f"WDO{month_codes[next_month]}{next_year}"
    return contract


def check_market_hours():
    """Verifica se está em horário de pregão"""
    now = datetime.now()
    
    # Verificar dia da semana (0=segunda, 4=sexta)
    if now.weekday() > 4:
        return False, "Fim de semana - mercado fechado"
    
    # Horário do pregão: 9h às 17h45
    # After market: 17h50 às 18h30
    market_open = now.replace(hour=9, minute=0, second=0)
    market_close = now.replace(hour=17, minute=45, second=0)
    after_open = now.replace(hour=17, minute=50, second=0)
    after_close = now.replace(hour=18, minute=30, second=0)
    
    if market_open <= now <= market_close:
        return True, "Pregão regular"
    elif after_open <= now <= after_close:
        return True, "After market"
    else:
        return False, f"Fora do horário de pregão (9h-17h45, 17h50-18h30)"


def run_book_collector():
    """Executa o coletor de book"""
    global collector, server_process
    
    # Registrar handler para Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("="*80)
    logger.info("📊 COLETOR DE BOOK DE OFERTAS - MODO PREGÃO")
    logger.info("="*80)
    
    # Verificar horário
    is_market_open, market_status = check_market_hours()
    logger.info(f"Status do mercado: {market_status}")
    
    if not is_market_open:
        logger.warning("⚠️ AVISO: Mercado fechado!")
        response = input("\nIniciar coleta mesmo assim? (s/n): ")
        if response.lower() != 's':
            return
    
    # Detectar contrato
    ticker = get_current_wdo_contract()
    logger.info(f"Contrato detectado: {ticker}")
    
    # Opções de coleta
    print("\nOpções de coleta:")
    print("1. Book completo (offer + price)")
    print("2. Apenas offer book")
    print("3. Apenas price book")
    
    choice = input("\nEscolha (1/2/3) [padrão: 1]: ").strip() or "1"
    
    book_type_map = {
        "1": "both",
        "2": "offer",
        "3": "price"
    }
    book_type = book_type_map.get(choice, "both")
    
    # Iniciar servidor se necessário
    server_process = start_server_if_needed()
    
    try:
        # Configurar coletor
        config = {
            'data_dir': 'data/realtime/book',
            'server_address': ('localhost', 6789)
        }
        
        collector = RealtimeBookCollector(config)
        
        # Conectar
        logger.info("\n📡 Conectando ao servidor...")
        if not collector.connect():
            logger.error("Falha ao conectar ao servidor")
            return
        
        # Subscrever
        logger.info(f"\n📈 Subscrevendo ao book de {ticker} (tipo: {book_type})...")
        if not collector.subscribe_book(ticker, book_type):
            logger.error("Falha ao subscrever book")
            return
        
        # Informações sobre a coleta
        logger.info("\n" + "="*60)
        logger.info("🟢 COLETA INICIADA")
        logger.info("="*60)
        logger.info(f"Ticker: {ticker}")
        logger.info(f"Tipo: {book_type}")
        logger.info(f"Salvando em: {config['data_dir']}/")
        logger.info(f"Intervalo de salvamento: 60 segundos")
        logger.info("\nPressione Ctrl+C para parar a coleta")
        logger.info("="*60 + "\n")
        
        # Iniciar coleta contínua (duration=0 significa infinito)
        collector.start_collection(duration_minutes=0)
        
        # Loop infinito (será interrompido por Ctrl+C)
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        # Já tratado pelo signal_handler
        pass
        
    except Exception as e:
        logger.error(f"Erro durante coleta: {e}", exc_info=True)
        
    finally:
        # Limpeza final (caso não tenha passado pelo signal_handler)
        if collector and collector.is_running:
            collector.stop_collection()
        
        if server_process and server_process.poll() is None:
            server_process.terminate()
            server_process.wait(timeout=5)
            if server_process.poll() is None:
                server_process.kill()


def main():
    """Função principal"""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║          COLETOR DE BOOK DE OFERTAS - WDO             ║
    ║                                                       ║
    ║  Este script coleta book de ofertas continuamente    ║
    ║  durante o pregão e salva em arquivos Parquet        ║
    ║                                                       ║
    ║  Compatível com ProfitDLL v4.0.0.30                  ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # Mostrar configurações
    print("Configurações:")
    print(f"- Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"- Diretório de dados: data/realtime/book/")
    print(f"- Formato: Parquet com compressão Snappy")
    print(f"- Buffer: Salvamento a cada 60 segundos")
    print("")
    
    # Executar coletor
    run_book_collector()


if __name__ == "__main__":
    main()