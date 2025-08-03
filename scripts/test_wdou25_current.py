"""
Script para testar coleta de dados do contrato atual WDOU25
Testa diferentes períodos e formatos de data
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
logger = logging.getLogger('TestWDOU25')


def test_current_contract():
    """Testa coleta do contrato atual WDOU25"""
    logger.info("="*80)
    logger.info("🔍 TESTE DO CONTRATO ATUAL - WDOU25")
    logger.info("="*80)
    logger.info("📅 Data atual: 02/08/2025")
    logger.info("📊 Contrato: WDOU25 (Agosto 2025)")
    
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
    
    # Lista de testes para WDOU25
    tests = []
    
    # Teste 1: Apenas ontem (01/08/2025)
    tests.append({
        'name': 'Apenas ontem (01/08/2025)',
        'symbol': 'WDOU25',
        'start': datetime(2025, 8, 1),
        'end': datetime(2025, 8, 1)
    })
    
    # Teste 2: Últimos 3 dias
    tests.append({
        'name': 'Últimos 3 dias',
        'symbol': 'WDOU25',
        'start': datetime(2025, 7, 30),
        'end': datetime(2025, 8, 1)
    })
    
    # Teste 3: Últimos 7 dias
    tests.append({
        'name': 'Últimos 7 dias',
        'symbol': 'WDOU25',
        'start': datetime(2025, 7, 26),
        'end': datetime(2025, 8, 1)
    })
    
    # Teste 4: Últimos 9 dias (limite recomendado)
    tests.append({
        'name': 'Últimos 9 dias',
        'symbol': 'WDOU25',
        'start': datetime(2025, 7, 24),
        'end': datetime(2025, 8, 1)
    })
    
    # Teste 5: 10 dias atrás (período de 5 dias)
    tests.append({
        'name': '10-15 dias atrás',
        'symbol': 'WDOU25',
        'start': datetime(2025, 7, 18),
        'end': datetime(2025, 7, 22)
    })
    
    # Teste 6: Início de julho
    tests.append({
        'name': 'Início de julho (1-5)',
        'symbol': 'WDOU25',
        'start': datetime(2025, 7, 1),
        'end': datetime(2025, 7, 5)
    })
    
    # Teste 7: Meio de junho (contrato anterior)
    tests.append({
        'name': 'Junho - contrato anterior',
        'symbol': 'WDOQ25',  # Julho 2025
        'start': datetime(2025, 6, 16),
        'end': datetime(2025, 6, 20)
    })
    
    # Teste 8: Maio 2025 (limite de 3 meses)
    tests.append({
        'name': 'Maio 2025 (limite 3 meses)',
        'symbol': 'WDOM25',  # Junho 2025
        'start': datetime(2025, 5, 5),
        'end': datetime(2025, 5, 9)
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
                logger.info("\n📊 Primeiros 5 trades:")
                logger.info(data['trades'].head())
                
                # Estatísticas
                logger.info(f"\n📈 Estatísticas:")
                logger.info(f"  Total de trades: {trades_count:,}")
                logger.info(f"  Preço médio: R$ {data['trades']['price'].mean():.2f}")
                logger.info(f"  Volume total: {data['trades']['volume'].sum():,.0f}")
                logger.info(f"  Primeiro trade: {data['trades']['datetime'].min()}")
                logger.info(f"  Último trade: {data['trades']['datetime'].max()}")
                
                # Salvar arquivo para verificação
                if success:
                    filename = f"test_data_{test['symbol']}_{test['start'].strftime('%Y%m%d')}.csv"
                    filepath = os.path.join("data", "test", filename)
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    data['trades'].to_csv(filepath, index=False)
                    logger.info(f"  Dados salvos em: {filepath}")
                    
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
    logger.info("📊 RESUMO DOS TESTES WDOU25")
    logger.info(f"{'='*80}")
    
    successful_tests = [r for r in results if r['success']]
    
    if successful_tests:
        logger.info(f"\n✅ TESTES BEM-SUCEDIDOS ({len(successful_tests)}/{len(results)}):")
        for result in successful_tests:
            logger.info(f"\n  📌 {result['test']}")
            logger.info(f"     Símbolo: {result['symbol']}")
            logger.info(f"     Período: {result['period']}")
            logger.info(f"     Trades: {result['trades']:,}")
        
        # Melhor resultado
        best_test = max(successful_tests, key=lambda x: x['trades'])
        logger.info(f"\n🏆 MELHOR RESULTADO:")
        logger.info(f"   {best_test['test']}")
        logger.info(f"   {best_test['trades']:,} trades coletados")
        
    failed_tests = [r for r in results if not r['success']]
    if failed_tests:
        logger.info(f"\n❌ TESTES FALHADOS ({len(failed_tests)}/{len(results)}):")
        for result in failed_tests:
            logger.info(f"\n  ❌ {result['test']}")
            logger.info(f"     Símbolo: {result['symbol']}")
            logger.info(f"     Período: {result['period']}")
            if 'error' in result:
                logger.info(f"     Erro: {result['error']}")
    
    # Parar servidor
    collector.stop_server()
    
    # Conclusões e recomendações
    logger.info(f"\n{'='*80}")
    logger.info("💡 ANÁLISE E RECOMENDAÇÕES")
    logger.info(f"{'='*80}")
    
    if successful_tests:
        logger.info("\n✅ COLETA FUNCIONANDO!")
        logger.info("\nRecomendações para coleta completa:")
        logger.info("1. Use períodos de 9 dias por vez")
        logger.info("2. Comece com dados mais recentes")
        logger.info("3. Colete de 9 em 9 dias retroativamente")
        logger.info("4. Respeite o limite de 3 meses do ProfitDLL")
        logger.info("5. Execute durante horário de pregão para melhores resultados")
        
        logger.info("\n📋 Estratégia sugerida para WDOU25:")
        logger.info("   - Período 1: 24/07 a 01/08 (últimos 9 dias)")
        logger.info("   - Período 2: 15/07 a 23/07")
        logger.info("   - Período 3: 06/07 a 14/07")
        logger.info("   - Período 4: 27/06 a 05/07")
        logger.info("   - Continue até maio/2025 (limite de 3 meses)")
        
    else:
        logger.info("\n❌ NENHUM TESTE FOI BEM-SUCEDIDO")
        logger.info("\nPossíveis causas:")
        logger.info("1. Mercado fechado (execute durante pregão)")
        logger.info("2. Erro no formato de data")
        logger.info("3. Limitações da conta")
        logger.info("4. Problema com a API do ProfitDLL")
        
        logger.info("\n🔧 Sugestões de debug:")
        logger.info("1. Verifique os logs do servidor ProfitDLL")
        logger.info("2. Teste com o Profit Chart Pro diretamente")
        logger.info("3. Confirme se sua conta tem acesso a dados históricos")
        logger.info("4. Tente durante o horário de pregão (9h-18h)")
    
    logger.info("\n✨ Teste concluído!")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    try:
        test_current_contract()
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Teste interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}", exc_info=True)