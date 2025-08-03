"""
Script simples para testar GetHistoryTrades com datas passadas
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
logger = logging.getLogger('SimpleHistoryTest')


def test_simple_history():
    """Teste simples e direto"""
    logger.info("="*60)
    logger.info("TESTE SIMPLES GetHistoryTrades")
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
            logger.info(f"🎯 Trade histórico recebido: Preço={data.get('price')}, Volume={data.get('volume')}")
        
        conn.register_history_trade_callback(on_history_trade)
        
        # Testar com diferentes períodos e contratos
        tests = [
            # Teste 1: Julho 2024 - dados mais antigos
            {
                'name': 'Julho 2024 - WDOQ24',
                'ticker': 'WDOQ24',  # WDO Julho 2024
                'start': '01/07/2024',
                'end': '05/07/2024'
            },
            # Teste 2: Janeiro 2025
            {
                'name': 'Janeiro 2025 - WDOF25',
                'ticker': 'WDOF25',  # WDO Janeiro 2025
                'start': '06/01/2025',
                'end': '10/01/2025'
            },
            # Teste 3: Maio 2025 (3 meses atrás)
            {
                'name': 'Maio 2025 - WDOM25',
                'ticker': 'WDOM25',  # WDO Junho 2025
                'start': '05/05/2025',
                'end': '09/05/2025'
            },
            # Teste 4: Junho 2025
            {
                'name': 'Junho 2025 - WDOQ25',
                'ticker': 'WDOQ25',  # WDO Julho 2025
                'start': '09/06/2025',
                'end': '13/06/2025'
            },
            # Teste 5: Julho 2025 (mês passado)
            {
                'name': 'Julho 2025 - WDOU25',
                'ticker': 'WDOU25',  # WDO Agosto 2025
                'start': '07/07/2025',
                'end': '11/07/2025'
            }
        ]
        
        # Executar testes
        for test in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"📊 TESTE: {test['name']}")
            logger.info(f"Ticker: {test['ticker']}")
            logger.info(f"Período: {test['start']} até {test['end']}")
            logger.info(f"{'='*50}")
            
            trades_received.clear()
            
            # Usar get_history_trades
            result = conn.get_history_trades(
                ticker=test['ticker'],
                exchange='F',  # Usar F para futuros
                date_start=test['start'],
                date_end=test['end']
            )
            
            if result:
                logger.info("✅ Solicitação enviada com sucesso!")
                
                # Aguardar dados
                logger.info("Aguardando dados...")
                for i in range(15):  # 15 segundos
                    time.sleep(1)
                    if trades_received:
                        logger.info(f"✅ {len(trades_received)} trades recebidos!")
                        break
                    if i % 3 == 0:
                        logger.info(f"Aguardando... {i+1}s")
                
                if trades_received:
                    logger.info(f"\n✅ SUCESSO! Total: {len(trades_received)} trades")
                    logger.info(f"Primeiro trade: {trades_received[0]}")
                    logger.info(f"Último trade: {trades_received[-1]}")
                else:
                    logger.warning("⚠️ Nenhum dado recebido após 15 segundos")
            else:
                logger.error("❌ Falha ao enviar solicitação")
            
            # Pausa entre testes
            time.sleep(3)
        
        # Desconectar
        conn.disconnect()
        logger.info("\n✅ Teste concluído")
        
        # Resumo
        logger.info("\n" + "="*60)
        logger.info("RESUMO DOS TESTES")
        logger.info("="*60)
        logger.info("\nSe nenhum dado foi recebido, possíveis causas:")
        logger.info("1. Mercado fechado - execute durante o pregão")
        logger.info("2. Dados não disponíveis para as datas solicitadas")
        logger.info("3. Contratos incorretos para os períodos")
        logger.info("4. Limitações da conta no ProfitDLL")
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}", exc_info=True)


if __name__ == "__main__":
    test_simple_history()