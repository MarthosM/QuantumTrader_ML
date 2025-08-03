"""
Testa coleta de dados históricos de 2024
"""

import os
import sys
import time
import logging
import multiprocessing
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.collect_historical_data import AutomatedDataCollector

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Test2024Data')


def test_2024_data():
    """Testa coleta de dados de 2024"""
    logger.info("="*80)
    logger.info("🔍 TESTE DE DADOS HISTÓRICOS 2024")
    logger.info("="*80)
    
    # Verificar credenciais
    required_vars = ['PROFIT_USERNAME', 'PROFIT_PASSWORD', 'PROFIT_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"❌ Variáveis de ambiente faltando: {missing}")
        return
    
    logger.info("✅ Credenciais verificadas")
    
    # Configuração base
    config = {
        "data_types": ["trades"],
        "data_dir": "data/historical",
        "csv_dir": "data/csv", 
        "log_dir": "logs",
        "report_dir": "reports",
        "dll_path": r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll",
        "profitdll_server_port": 6790,
        "username": os.getenv("PROFIT_USERNAME"),
        "password": os.getenv("PROFIT_PASSWORD"),
        "key": os.getenv("PROFIT_KEY")
    }
    
    # Criar coletor
    collector = AutomatedDataCollector.__new__(AutomatedDataCollector)
    collector.config = config
    collector._setup_logging()
    collector.collector = None
    collector.server_process = None
    collector.collection_status = {
        'last_run': None,
        'last_success': None,
        'errors': [],
        'symbols_collected': {}
    }
    
    # Iniciar servidor isolado
    logger.info("\n🚀 Iniciando servidor ProfitDLL isolado...")
    if not collector.start_isolated_server():
        logger.error("❌ Falha ao iniciar servidor")
        return
    
    logger.info("✅ Servidor ProfitDLL rodando em processo isolado")
    time.sleep(2)
    
    # Inicializar coletor
    from src.database.historical_data_collector import HistoricalDataCollector
    collector.collector = HistoricalDataCollector(config)
    
    # Lista de testes - vamos tentar dados de 2024
    tests = []
    
    # Teste 1: Janeiro 2024 (dados antigos)
    tests.append({
        'name': 'Janeiro 2024 - 5 dias',
        'symbol': 'WDOG24',  # WDO Fevereiro 2024
        'start': datetime(2024, 1, 15),
        'end': datetime(2024, 1, 19)
    })
    
    # Teste 2: Maio 2024
    tests.append({
        'name': 'Maio 2024 - 5 dias',
        'symbol': 'WDOM24',  # WDO Junho 2024
        'start': datetime(2024, 5, 6),
        'end': datetime(2024, 5, 10)
    })
    
    # Teste 3: Julho 2024 (90 dias atrás)
    tests.append({
        'name': 'Julho 2024 - 7 dias',
        'symbol': 'WDOU24',  # WDO Agosto 2024
        'start': datetime(2024, 7, 1),
        'end': datetime(2024, 7, 7)
    })
    
    # Teste 4: Agosto 2024
    tests.append({
        'name': 'Agosto 2024 - 5 dias',
        'symbol': 'WDOV24',  # WDO Setembro 2024
        'start': datetime(2024, 8, 5),
        'end': datetime(2024, 8, 9)
    })
    
    # Teste 5: Dezembro 2024
    tests.append({
        'name': 'Dezembro 2024 - 5 dias',
        'symbol': 'WDOF25',  # WDO Janeiro 2025
        'start': datetime(2024, 12, 9),
        'end': datetime(2024, 12, 13)
    })
    
    # Teste 6: WDO genérico com data recente
    tests.append({
        'name': 'WDO genérico - Maio 2024',
        'symbol': 'WDO',
        'start': datetime(2024, 5, 1),
        'end': datetime(2024, 5, 5)
    })
    
    # Teste 7: WDOQ24 (contrato passado)
    tests.append({
        'name': 'WDOQ24 - Junho 2024',
        'symbol': 'WDOQ24',  # WDO Julho 2024
        'start': datetime(2024, 6, 10),
        'end': datetime(2024, 6, 14)
    })
    
    # Executar testes
    results = []
    
    for i, test in enumerate(tests):
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 TESTE {i+1}/{len(tests)}: {test['name']}")
        logger.info(f"Símbolo: {test['symbol']}")
        logger.info(f"Período: {test['start'].strftime('%d/%m/%Y')} até {test['end'].strftime('%d/%m/%Y')}")
        logger.info(f"{'='*60}")
        
        try:
            # Coletar dados
            data = collector.collector.collect_historical_data(
                symbol=test['symbol'],
                start_date=test['start'],
                end_date=test['end'],
                data_types=config['data_types']
            )
            
            # Verificar resultado
            success = False
            trades_count = 0
            
            if 'trades' in data and not data['trades'].empty:
                trades_count = len(data['trades'])
                success = True
                logger.info(f"✅ SUCESSO! {trades_count:,} trades coletados")
                
                # Mostrar amostra
                logger.info("\n📊 Amostra dos dados:")
                logger.info(data['trades'].head(3))
                
                # Estatísticas
                logger.info(f"\n📈 Estatísticas:")
                logger.info(f"  Preço médio: R$ {data['trades']['price'].mean():.2f}")
                logger.info(f"  Volume total: {data['trades']['volume'].sum():,.0f}")
                logger.info(f"  Primeiro trade: {data['trades']['datetime'].min()}")
                logger.info(f"  Último trade: {data['trades']['datetime'].max()}")
            else:
                logger.warning("❌ FALHA - Nenhum dado retornado")
            
            results.append({
                'test': test['name'],
                'symbol': test['symbol'],
                'period': f"{test['start'].strftime('%d/%m/%Y')} - {test['end'].strftime('%d/%m/%Y')}",
                'success': success,
                'trades': trades_count
            })
            
        except Exception as e:
            logger.error(f"❌ ERRO: {e}")
            results.append({
                'test': test['name'],
                'symbol': test['symbol'],
                'period': f"{test['start'].strftime('%d/%m/%Y')} - {test['end'].strftime('%d/%m/%Y')}",
                'success': False,
                'trades': 0,
                'error': str(e)
            })
        
        # Pausa entre testes
        time.sleep(2)
    
    # Resumo dos resultados
    logger.info(f"\n{'='*80}")
    logger.info("📊 RESUMO DOS TESTES DE 2024")
    logger.info(f"{'='*80}")
    
    successful_tests = [r for r in results if r['success']]
    
    if successful_tests:
        logger.info(f"\n✅ TESTES BEM-SUCEDIDOS ({len(successful_tests)}/{len(results)}):")
        for result in successful_tests:
            logger.info(f"  - {result['test']}")
            logger.info(f"    Símbolo: {result['symbol']}")
            logger.info(f"    Período: {result['period']}")
            logger.info(f"    Trades: {result['trades']:,}")
    
    failed_tests = [r for r in results if not r['success']]
    if failed_tests:
        logger.info(f"\n❌ TESTES FALHADOS ({len(failed_tests)}/{len(results)}):")
        for result in failed_tests:
            logger.info(f"  - {result['test']}")
            logger.info(f"    Símbolo: {result['symbol']}")
            logger.info(f"    Período: {result['period']}")
            if 'error' in result:
                logger.info(f"    Erro: {result['error']}")
    
    # Parar servidor
    collector.stop_server()
    
    logger.info("\n✨ Teste de dados de 2024 concluído!")
    
    # Informações importantes
    logger.info("\n📌 INFORMAÇÕES IMPORTANTES:")
    logger.info("1. O ProfitDLL tem limite de 3 meses para dados históricos")
    logger.info("2. Hoje é 02/08/2025, então dados mais antigos que maio/2025 podem não estar disponíveis")
    logger.info("3. Use os contratos corretos para cada mês (WDO + código do mês + ano)")
    logger.info("4. Dados podem estar disponíveis apenas durante horário de pregão")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    try:
        test_2024_data()
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Teste interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}", exc_info=True)