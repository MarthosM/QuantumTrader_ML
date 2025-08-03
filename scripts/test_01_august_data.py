"""
Teste específico para captar dados do dia 01/08/2025
"""

import os
import sys
import time
import logging
from datetime import datetime
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
logger = logging.getLogger('Test01August')


def test_01_august():
    """Testa coleta de dados do dia 01/08/2025"""
    logger.info("="*80)
    logger.info("📅 TESTE ESPECÍFICO - DADOS DE 01/08/2025")
    logger.info("="*80)
    logger.info("Data atual: 02/08/2025")
    logger.info("Objetivo: Captar dados de ontem (01/08/2025)")
    
    from src.connection_manager_v4 import ConnectionManagerV4
    
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    
    try:
        # Criar connection manager
        conn = ConnectionManagerV4(dll_path)
        
        # Conectar
        logger.info("\n🔌 Conectando ao ProfitDLL...")
        if not conn.initialize(
            key=os.getenv("PROFIT_KEY"),
            username=os.getenv("PROFIT_USERNAME"),
            password=os.getenv("PROFIT_PASSWORD")
        ):
            logger.error("❌ Falha ao conectar")
            return
            
        logger.info("✅ Conectado com sucesso!")
        
        # Aguardar conexão completa
        logger.info("⏳ Aguardando estabilização da conexão...")
        time.sleep(5)
        
        # Configurar callback para receber dados
        trades_received = []
        
        def on_history_trade(data):
            trades_received.append(data)
            # Log detalhado do primeiro trade
            if len(trades_received) == 1:
                logger.info("🎯 PRIMEIRO TRADE RECEBIDO!")
                logger.info(f"   Dados completos: {data}")
            elif len(trades_received) % 100 == 0:
                logger.info(f"📊 {len(trades_received)} trades recebidos...")
        
        conn.register_history_trade_callback(on_history_trade)
        logger.info("✅ Callback registrado")
        
        # Lista de testes para o dia 01/08
        tests = [
            {
                'name': 'Dia completo - Sem horário',
                'ticker': 'WDOU25',
                'exchange': 'F',
                'start': '01/08/2025',
                'end': '01/08/2025'
            },
            {
                'name': 'Dia completo - Com horário de pregão',
                'ticker': 'WDOU25',
                'exchange': 'F',
                'start': '01/08/2025 09:00:00',
                'end': '01/08/2025 18:00:00'
            },
            {
                'name': 'Abertura do pregão (primeira hora)',
                'ticker': 'WDOU25',
                'exchange': 'F',
                'start': '01/08/2025 09:00:00',
                'end': '01/08/2025 10:00:00'
            },
            {
                'name': 'Período da tarde',
                'ticker': 'WDOU25',
                'exchange': 'F',
                'start': '01/08/2025 14:00:00',
                'end': '01/08/2025 16:00:00'
            },
            {
                'name': 'Com exchange vazia',
                'ticker': 'WDOU25',
                'exchange': '',
                'start': '01/08/2025',
                'end': '01/08/2025'
            },
            {
                'name': 'WDO genérico',
                'ticker': 'WDO',
                'exchange': 'F',
                'start': '01/08/2025 09:00:00',
                'end': '01/08/2025 18:00:00'
            }
        ]
        
        # Executar cada teste
        for i, test in enumerate(tests, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"📊 TESTE {i}/{len(tests)}: {test['name']}")
            logger.info(f"Ticker: {test['ticker']}")
            logger.info(f"Exchange: '{test['exchange']}'")
            logger.info(f"Período: {test['start']} até {test['end']}")
            logger.info(f"{'='*70}")
            
            trades_received.clear()
            
            # Fazer a solicitação
            logger.info("📤 Enviando solicitação...")
            
            # Chamar diretamente a DLL
            if hasattr(conn.dll, 'GetHistoryTrades'):
                result = conn.dll.GetHistoryTrades(
                    test['ticker'],
                    test['exchange'],
                    test['start'],
                    test['end']
                )
                
                logger.info(f"📥 Resposta da DLL: {result}")
                
                if result == 0:
                    logger.info("✅ Solicitação aceita!")
                    
                    # Aguardar dados com monitoramento detalhado
                    logger.info("⏳ Aguardando dados...")
                    
                    last_count = 0
                    no_data_counter = 0
                    
                    for second in range(30):  # 30 segundos máximo
                        time.sleep(1)
                        
                        current_count = len(trades_received)
                        
                        if current_count > last_count:
                            new_trades = current_count - last_count
                            logger.info(f"   📈 +{new_trades} trades (total: {current_count})")
                            last_count = current_count
                            no_data_counter = 0
                        else:
                            no_data_counter += 1
                            if second % 5 == 0:
                                logger.info(f"   ⏳ Aguardando... {second}s")
                        
                        # Se recebeu muitos dados ou parou de receber
                        if current_count > 1000 or (current_count > 0 and no_data_counter > 5):
                            break
                    
                    # Resultado do teste
                    if trades_received:
                        logger.info(f"\n✅ SUCESSO! Recebidos {len(trades_received)} trades")
                        
                        # Análise dos dados
                        logger.info("\n📊 ANÁLISE DOS DADOS:")
                        
                        # Primeiro e último trade
                        first_trade = trades_received[0]
                        last_trade = trades_received[-1]
                        
                        logger.info(f"Primeiro trade:")
                        logger.info(f"  Timestamp: {first_trade.get('timestamp')}")
                        logger.info(f"  Preço: {first_trade.get('price')}")
                        logger.info(f"  Volume: {first_trade.get('volume')}")
                        
                        logger.info(f"\nÚltimo trade:")
                        logger.info(f"  Timestamp: {last_trade.get('timestamp')}")
                        logger.info(f"  Preço: {last_trade.get('price')}")
                        logger.info(f"  Volume: {last_trade.get('volume')}")
                        
                        # Estatísticas
                        prices = [t.get('price', 0) for t in trades_received if t.get('price')]
                        volumes = [t.get('volume', 0) for t in trades_received if t.get('volume')]
                        
                        if prices:
                            logger.info(f"\nEstatísticas:")
                            logger.info(f"  Preço mínimo: {min(prices)}")
                            logger.info(f"  Preço máximo: {max(prices)}")
                            logger.info(f"  Preço médio: {sum(prices)/len(prices):.2f}")
                            logger.info(f"  Volume total: {sum(volumes)}")
                        
                        # Salvar amostra
                        logger.info("\n💾 Salvando amostra dos dados...")
                        import json
                        sample_file = f"data/test/sample_01aug_{test['ticker']}_{i}.json"
                        os.makedirs("data/test", exist_ok=True)
                        
                        with open(sample_file, 'w') as f:
                            json.dump(trades_received[:10], f, indent=2, default=str)
                        logger.info(f"   Amostra salva em: {sample_file}")
                        
                        # Este teste funcionou!
                        logger.info("\n🎉 TESTE BEM-SUCEDIDO! Podemos coletar dados históricos!")
                        break  # Parar nos outros testes se este funcionou
                        
                    else:
                        logger.warning("⚠️ Nenhum dado recebido após 30 segundos")
                        
                elif result == -2147483645:
                    logger.error("❌ Erro -2147483645: Parâmetros inválidos ou dados indisponíveis")
                else:
                    logger.error(f"❌ Erro desconhecido: {result}")
            else:
                logger.error("❌ GetHistoryTrades não disponível na DLL")
            
            # Pausa entre testes
            if i < len(tests):
                logger.info("\n⏸️ Pausa de 5 segundos antes do próximo teste...")
                time.sleep(5)
        
        # Desconectar
        logger.info("\n🔌 Desconectando...")
        conn.disconnect()
        
        # Resumo final
        logger.info("\n" + "="*80)
        logger.info("📋 RESUMO DO TESTE")
        logger.info("="*80)
        
        if any(trades_received):
            logger.info("✅ SUCESSO! Conseguimos coletar dados históricos!")
            logger.info("\nPróximos passos:")
            logger.info("1. Usar o coletor automático para períodos maiores")
            logger.info("2. Implementar estratégia de coleta em blocos de 9 dias")
            logger.info("3. Armazenar dados em formato Parquet")
        else:
            logger.info("❌ Não conseguimos coletar dados históricos")
            logger.info("\nPossíveis causas:")
            logger.info("1. Mercado pode estar fechado")
            logger.info("2. Dados de 01/08 podem não estar disponíveis ainda")
            logger.info("3. Limitações da conta ou da API")
            logger.info("4. Tentar durante o pregão para melhores resultados")
        
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}", exc_info=True)


if __name__ == "__main__":
    test_01_august()