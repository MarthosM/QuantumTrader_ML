"""
Script de diagnóstico para coleta de dados históricos
Testa diferentes períodos e símbolos para identificar o problema
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
logger = logging.getLogger('DiagnoseCollection')


def get_wdo_symbol(date):
    """Retorna o símbolo WDO correto para a data"""
    # Month codes for WDO futures
    month_codes = {
        1: 'F',  # January
        2: 'G',  # February
        3: 'H',  # March
        4: 'J',  # April
        5: 'K',  # May
        6: 'M',  # June
        7: 'Q',  # July
        8: 'U',  # August
        9: 'V',  # September
        10: 'X', # October
        11: 'Z', # November
        12: 'F'  # December (volta para F)
    }
    
    # WDO usa o próximo mês
    next_month = date.month + 1
    year = date.year
    
    if next_month > 12:
        next_month = 1
        year += 1
    
    month_code = month_codes.get(next_month, 'F')
    year_suffix = str(year)[-2:]
    
    return f"WDO{month_code}{year_suffix}"


def test_recent_periods():
    """Testa coleta de períodos recentes (últimos 9 dias)"""
    logger.info("="*80)
    logger.info("🔍 DIAGNÓSTICO DE COLETA HISTÓRICA")
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
    
    # Lista de testes a executar
    tests = []
    
    # Teste 1: Últimos 9 dias com símbolo atual
    end_date = datetime.now()
    start_date = end_date - timedelta(days=9)
    current_symbol = get_wdo_symbol(datetime.now())
    tests.append({
        'name': 'Últimos 9 dias - Símbolo atual',
        'symbol': current_symbol,
        'start': start_date,
        'end': end_date
    })
    
    # Teste 2: Últimos 7 dias (período menor)
    start_date_7d = end_date - timedelta(days=7)
    tests.append({
        'name': 'Últimos 7 dias - Símbolo atual',
        'symbol': current_symbol,
        'start': start_date_7d,
        'end': end_date
    })
    
    # Teste 3: Apenas ontem
    yesterday = end_date - timedelta(days=1)
    tests.append({
        'name': 'Apenas ontem',
        'symbol': current_symbol,
        'start': yesterday,
        'end': yesterday
    })
    
    # Teste 4: Período de 10 a 19 dias atrás
    end_19d = end_date - timedelta(days=10)
    start_19d = end_date - timedelta(days=19)
    tests.append({
        'name': '10 a 19 dias atrás',
        'symbol': current_symbol,
        'start': start_19d,
        'end': end_19d
    })
    
    # Teste 5: Diferentes símbolos WDO
    symbols_to_test = []
    
    # Month codes definido localmente para este bloco
    month_codes_local = {
        1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
        7: 'Q', 8: 'U', 9: 'V', 10: 'X', 11: 'Z', 12: 'F'
    }
    
    # Símbolo do mês atual
    current_month_symbol = f"WDO{month_codes_local.get(datetime.now().month, 'F')}{str(datetime.now().year)[-2:]}"
    symbols_to_test.append(current_month_symbol)
    
    # Próximo mês
    next_month = datetime.now().month + 1
    next_year = datetime.now().year
    if next_month > 12:
        next_month = 1
        next_year += 1
    next_symbol = f"WDO{month_codes_local.get(next_month, 'F')}{str(next_year)[-2:]}"
    symbols_to_test.append(next_symbol)
    
    # WDO genérico
    symbols_to_test.append("WDO")
    
    for symbol in symbols_to_test:
        tests.append({
            'name': f'Últimos 5 dias - {symbol}',
            'symbol': symbol,
            'start': end_date - timedelta(days=5),
            'end': end_date
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
    logger.info("📊 RESUMO DOS TESTES")
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
    
    # Recomendações
    logger.info("\n💡 RECOMENDAÇÕES:")
    
    if successful_tests:
        # Pegar o teste bem-sucedido com mais dados
        best_test = max(successful_tests, key=lambda x: x['trades'])
        logger.info(f"\n✅ Melhor configuração encontrada:")
        logger.info(f"  - Símbolo: {best_test['symbol']}")
        logger.info(f"  - Período: {best_test['period']}")
        logger.info(f"  - Trades coletados: {best_test['trades']:,}")
        
        logger.info("\n📋 Próximos passos:")
        logger.info("1. Use o símbolo e período que funcionou")
        logger.info("2. Colete dados de 9 em 9 dias para períodos maiores")
        logger.info("3. Verifique se o pregão estava aberto nos dias testados")
    else:
        logger.info("\n❌ Nenhum teste foi bem-sucedido. Possíveis causas:")
        logger.info("1. Mercado fechado ou sem dados para os períodos testados")
        logger.info("2. Símbolo incorreto - verifique o contrato ativo")
        logger.info("3. Limitações da conta ou do ProfitDLL")
        logger.info("4. Tente executar durante o horário de pregão")
    
    # Parar servidor
    collector.stop_server()
    
    logger.info("\n✨ Diagnóstico concluído!")


def test_date_formats():
    """Testa diferentes formatos de data"""
    logger.info("\n" + "="*60)
    logger.info("🔍 TESTE DE FORMATOS DE DATA")
    logger.info("="*60)
    
    from src.connection_manager_v4 import ConnectionManagerV4
    
    config = {
        'dll_path': r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll",
        'username': os.getenv("PROFIT_USERNAME"),
        'password': os.getenv("PROFIT_PASSWORD"),
        'key': os.getenv("PROFIT_KEY")
    }
    
    # Formatos de data para testar
    date_formats = [
        ('01/08/2025', '02/08/2025', 'DD/MM/YYYY'),
        ('20250801', '20250802', 'YYYYMMDD'),
        ('01082025', '02082025', 'DDMMYYYY'),
        ('2025-08-01', '2025-08-02', 'YYYY-MM-DD'),
        ('01-08-2025', '02-08-2025', 'DD-MM-YYYY'),
    ]
    
    logger.info("\nTestando diferentes formatos de data com GetHistoryTrades...")
    
    # Esta parte seria executada apenas se necessário
    # Por ora, vamos focar no teste principal
    

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    try:
        # Executar diagnóstico principal
        test_recent_periods()
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Diagnóstico interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}", exc_info=True)