"""
Teste GetHistoryTrades com formato completo de data e hora
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestWithTime')


def test_with_datetime():
    """Teste com formato completo DD/MM/YYYY HH:mm:SS"""
    logger.info("="*60)
    logger.info("TESTE GetHistoryTrades COM DATA E HORA")
    logger.info("="*60)
    
    from src.connection_manager_v4 import ConnectionManagerV4
    
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    
    try:
        # Criar connection manager
        conn = ConnectionManagerV4(dll_path)
        
        # Conectar
        if not conn.initialize(
            key=os.getenv("PROFIT_KEY"),
            username=os.getenv("PROFIT_USERNAME"),
            password=os.getenv("PROFIT_PASSWORD")
        ):
            logger.error("Falha ao conectar")
            return
            
        logger.info("✅ Conectado ao ProfitDLL")
        
        # Aguardar conexão completa
        time.sleep(5)
        
        # Configurar callback
        trades_received = []
        
        def on_history_trade(data):
            trades_received.append(data)
            logger.info(f"🎯 Trade recebido: {data}")
        
        conn.register_history_trade_callback(on_history_trade)
        
        # Testar com diferentes períodos e formatos
        tests = [
            # Teste 1: Ontem com horário de pregão
            {
                'name': 'Ontem (01/08/2025) - Pregão completo',
                'ticker': 'WDOU25',
                'start': '01/08/2025 09:00:00',
                'end': '01/08/2025 18:00:00'
            },
            # Teste 2: Ontem - Apenas manhã
            {
                'name': 'Ontem manhã',
                'ticker': 'WDOU25',
                'start': '01/08/2025 09:00:00',
                'end': '01/08/2025 12:00:00'
            },
            # Teste 3: Período menor (1 hora)
            {
                'name': '1 hora específica ontem',
                'ticker': 'WDOU25',
                'start': '01/08/2025 10:00:00',
                'end': '01/08/2025 11:00:00'
            },
            # Teste 4: Julho com horário
            {
                'name': 'Julho 2025 - 1 dia com horário',
                'ticker': 'WDOU25',
                'start': '15/07/2025 09:00:00',
                'end': '15/07/2025 18:00:00'
            },
            # Teste 5: Sem segundos (HH:mm)
            {
                'name': 'Formato sem segundos',
                'ticker': 'WDOU25',
                'start': '01/08/2025 09:00',
                'end': '01/08/2025 18:00'
            },
            # Teste 6: Formato alternativo
            {
                'name': 'Formato com zeros nos segundos',
                'ticker': 'WDOU25',
                'start': '01/08/2025 09:00:00',
                'end': '01/08/2025 09:30:00'
            }
        ]
        
        # Executar testes
        for test in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 TESTE: {test['name']}")
            logger.info(f"Ticker: {test['ticker']}")
            logger.info(f"Período: {test['start']} até {test['end']}")
            logger.info(f"{'='*60}")
            
            trades_received.clear()
            
            # Usar get_history_trades
            result = conn.get_history_trades(
                ticker=test['ticker'],
                exchange='F',
                date_start=test['start'],
                date_end=test['end']
            )
            
            if result:
                logger.info("✅ Solicitação enviada com sucesso!")
                
                # Aguardar dados
                logger.info("Aguardando dados...")
                received_count = 0
                for i in range(20):  # 20 segundos
                    time.sleep(1)
                    
                    if len(trades_received) > received_count:
                        new_trades = len(trades_received) - received_count
                        logger.info(f"📈 +{new_trades} trades (total: {len(trades_received)})")
                        received_count = len(trades_received)
                    
                    # Se recebeu muitos dados, pode parar
                    if len(trades_received) > 100:
                        logger.info("✅ Recebidos dados suficientes!")
                        break
                
                if trades_received:
                    logger.info(f"\n✅ SUCESSO! Total: {len(trades_received)} trades")
                    logger.info("\n📊 Primeiros 5 trades:")
                    for i, trade in enumerate(trades_received[:5]):
                        logger.info(f"  {i+1}. {trade}")
                    
                    logger.info("\n📊 Últimos 5 trades:")
                    for i, trade in enumerate(trades_received[-5:]):
                        logger.info(f"  {i+1}. {trade}")
                else:
                    logger.warning("⚠️ Nenhum dado recebido")
            else:
                logger.error("❌ Falha ao enviar solicitação")
            
            # Pausa entre testes
            time.sleep(3)
        
        # Desconectar
        conn.disconnect()
        logger.info("\n✅ Teste concluído")
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}", exc_info=True)


if __name__ == "__main__":
    test_with_datetime()