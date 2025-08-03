"""
Script de teste para coleta de book de ofertas em tempo real
"""

import os
import sys
import time
import subprocess
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
logger = logging.getLogger('TestBookCollection')


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


def test_book_collection():
    """Testa coleta de book de ofertas"""
    logger.info("="*80)
    logger.info("🧪 TESTE DE COLETA DE BOOK DE OFERTAS")
    logger.info("="*80)
    
    # 1. Verificar/iniciar servidor
    server_process = start_server_if_needed()
    
    try:
        # 2. Configurar coletor
        config = {
            'data_dir': 'data/realtime/book',
            'server_address': ('localhost', 6789)
        }
        
        collector = RealtimeBookCollector(config)
        
        # 3. Conectar ao servidor
        logger.info("\n📡 Conectando ao servidor...")
        if not collector.connect():
            logger.error("Falha ao conectar ao servidor")
            return
        
        # 4. Detectar contrato WDO atual
        from datetime import datetime
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
        
        ticker = f"WDO{month_codes[next_month]}{next_year}"
        logger.info(f"📊 Ticker detectado: {ticker}")
        
        # 5. Subscrever ao book
        logger.info(f"\n📈 Subscrevendo ao book de {ticker}...")
        if not collector.subscribe_book(ticker, 'both'):
            logger.error("Falha ao subscrever book")
            return
        
        # 6. Coletar dados por 2 minutos
        duration = 2
        logger.info(f"\n⏳ Coletando dados por {duration} minutos...")
        logger.info("Pressione Ctrl+C para interromper\n")
        
        # Iniciar coleta
        collector.start_collection(duration_minutes=duration)
        
        # 7. Mostrar resultados
        logger.info("\n📊 RESUMO DA COLETA")
        logger.info("="*60)
        
        # Verificar arquivos salvos
        data_dir = Path(config['data_dir'])
        today_dir = data_dir / datetime.now().strftime('%Y%m%d')
        
        if today_dir.exists():
            offer_files = list(today_dir.glob("offer_book_*.parquet"))
            price_files = list(today_dir.glob("price_book_*.parquet"))
            
            logger.info(f"Arquivos de offer book: {len(offer_files)}")
            logger.info(f"Arquivos de price book: {len(price_files)}")
            
            # Mostrar amostra do último arquivo
            if offer_files:
                latest_offer = collector.get_latest_book_snapshot(ticker, 'offer')
                if latest_offer is not None and not latest_offer.empty:
                    logger.info(f"\n📖 Amostra do Offer Book ({len(latest_offer)} registros):")
                    logger.info(latest_offer.head(10).to_string())
            
            if price_files:
                latest_price = collector.get_latest_book_snapshot(ticker, 'price')
                if latest_price is not None and not latest_price.empty:
                    logger.info(f"\n📖 Amostra do Price Book ({len(latest_price)} registros):")
                    logger.info(latest_price.head(10).to_string())
        else:
            logger.warning("Nenhum arquivo salvo encontrado")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrompido pelo usuário")
        
    except Exception as e:
        logger.error(f"Erro durante teste: {e}", exc_info=True)
        
    finally:
        # Finalizar servidor se foi iniciado por este script
        if server_process:
            logger.info("\n🛑 Finalizando servidor...")
            server_process.terminate()
            server_process.wait(timeout=5)
            
            if server_process.poll() is None:
                server_process.kill()
            
            logger.info("Servidor finalizado")
    
    logger.info("\n✅ TESTE CONCLUÍDO!")


def main():
    """Função principal"""
    logger.info("🔧 Sistema de Teste de Book de Ofertas")
    logger.info("Compatível com ProfitDLL v4.0.0.30")
    logger.info("")
    
    # Verificar se é horário de pregão
    now = datetime.now()
    if now.weekday() > 4:  # Fim de semana
        logger.warning("⚠️ AVISO: Hoje é fim de semana - mercado fechado")
        logger.warning("O book só terá dados durante o pregão (seg-sex)")
        response = input("\nContinuar mesmo assim? (s/n): ")
        if response.lower() != 's':
            return
    
    elif now.hour < 9 or now.hour >= 18:
        logger.warning("⚠️ AVISO: Fora do horário de pregão")
        logger.warning("O book só terá dados entre 9h e 18h")
        response = input("\nContinuar mesmo assim? (s/n): ")
        if response.lower() != 's':
            return
    
    # Executar teste
    test_book_collection()


if __name__ == "__main__":
    main()