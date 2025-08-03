"""
Script para executar coleta completa de dados históricos
Coleta dados dos últimos 3 meses (limite do ProfitDLL)
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
logger = logging.getLogger('FullHistoricalCollection')


def run_full_collection():
    """Executa coleta completa de dados históricos"""
    logger.info("="*80)
    logger.info("🚀 COLETA COMPLETA DE DADOS HISTÓRICOS")
    logger.info("="*80)
    
    # Verificar credenciais
    required_vars = ['PROFIT_USERNAME', 'PROFIT_PASSWORD', 'PROFIT_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"❌ Variáveis de ambiente faltando: {missing}")
        return
    
    logger.info("✅ Credenciais verificadas")
    
    # Configuração para coleta completa
    config = {
        "symbols": ["WDOQ25", "WDOU25"],  # WDO agosto e setembro
        "data_types": ["trades"],  # Começar com trades
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
    
    # Determinar período de coleta
    # ProfitDLL permite até 3 meses, vamos coletar 90 dias
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    logger.info(f"\n📅 Período de coleta: {start_date.date()} até {end_date.date()}")
    logger.info(f"📊 Símbolos: {', '.join(config['symbols'])}")
    logger.info(f"📈 Tipos de dados: {', '.join(config['data_types'])}")
    
    # Avisos importantes
    logger.info("\n⚠️  AVISOS IMPORTANTES:")
    logger.info("1. A coleta pode demorar vários minutos por símbolo")
    logger.info("2. O ProfitDLL coleta em blocos de 9 dias por vez")
    logger.info("3. Não interrompa o processo durante a coleta")
    logger.info("4. Os dados serão salvos em formato Parquet (comprimido)")
    
    # Confirmar início
    logger.info("\n🔔 Iniciando coleta em 5 segundos...")
    time.sleep(5)
    
    try:
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
        logger.info("\n1️⃣ Iniciando servidor ProfitDLL isolado...")
        if not collector.start_isolated_server():
            logger.error("❌ Falha ao iniciar servidor")
            # Tentar parar qualquer servidor existente e reiniciar
            logger.info("Tentando parar servidor existente e reiniciar...")
            collector.stop_server()
            time.sleep(2)
            if not collector.start_isolated_server():
                logger.error("❌ Falha definitiva ao iniciar servidor")
                return
        
        # Servidor está rodando - continuar com a coleta
        logger.info("✅ Servidor ProfitDLL rodando em processo isolado")
        time.sleep(2)  # Dar tempo para o servidor estar pronto
        
        # Inicializar coletor de dados
        from src.database.historical_data_collector import HistoricalDataCollector
        collector.collector = HistoricalDataCollector(config)
        
        # Executar coleta para cada símbolo
        total_start = time.time()
        
        for symbol in config['symbols']:
            logger.info(f"\n{'='*60}")
            logger.info(f"📈 Coletando dados de {symbol}")
            logger.info(f"{'='*60}")
            
            symbol_start = time.time()
            
            try:
                # Dividir em períodos de 9 dias (limite do ProfitDLL)
                current_start = start_date
                total_trades = 0
                
                while current_start < end_date:
                    # Calcular fim do período (máximo 9 dias)
                    current_end = min(current_start + timedelta(days=8), end_date)
                    
                    logger.info(f"\n📅 Coletando período: {current_start.date()} até {current_end.date()}")
                    
                    # Coletar dados do período
                    data = collector.collector.collect_historical_data(
                        symbol=symbol,
                        start_date=current_start,
                        end_date=current_end,
                        data_types=config['data_types']
                    )
                    
                    # Verificar resultado
                    if 'trades' in data and not data['trades'].empty:
                        period_trades = len(data['trades'])
                        total_trades += period_trades
                        logger.info(f"✅ {period_trades} trades coletados neste período")
                    else:
                        logger.warning(f"⚠️ Nenhum dado coletado para este período")
                    
                    # Próximo período
                    current_start = current_end + timedelta(days=1)
                    
                    # Pausa entre requisições
                    time.sleep(2)
                
                # Resumo do símbolo
                symbol_time = time.time() - symbol_start
                logger.info(f"\n✅ {symbol} concluído:")
                logger.info(f"   - Total de trades: {total_trades:,}")
                logger.info(f"   - Tempo: {symbol_time:.1f} segundos")
                
                # Verificar dados salvos
                summary = collector.collector.get_data_summary(symbol)
                if 'dates' in summary:
                    logger.info(f"   - Dias salvos: {len(summary['dates'])}")
                    logger.info(f"   - Tamanho em disco: {summary.get('total_size_mb', 0):.2f} MB")
                
            except Exception as e:
                logger.error(f"❌ Erro coletando {symbol}: {e}")
                continue
        
        # Resumo final
        total_time = time.time() - total_start
        logger.info(f"\n{'='*80}")
        logger.info("📊 RESUMO DA COLETA")
        logger.info(f"{'='*80}")
        
        # Listar todos os dados coletados
        logger.info("\nDados disponíveis:")
        available = collector.collector.get_available_data()
        
        total_files = 0
        total_size = 0
        
        for symbol, dates in available.items():
            logger.info(f"\n{symbol}:")
            logger.info(f"  - Dias coletados: {len(dates)}")
            if dates:
                logger.info(f"  - Período: {dates[0]} até {dates[-1]}")
            
            # Calcular tamanho
            summary = collector.collector.get_data_summary(symbol)
            if 'total_size_mb' in summary:
                size_mb = summary['total_size_mb']
                total_size += size_mb
                logger.info(f"  - Tamanho: {size_mb:.2f} MB")
            
            total_files += len(dates)
        
        logger.info(f"\n📈 Total geral:")
        logger.info(f"  - Arquivos: {total_files}")
        logger.info(f"  - Tamanho total: {total_size:.2f} MB")
        logger.info(f"  - Tempo total: {total_time/60:.1f} minutos")
        
        # Parar servidor
        collector.stop_server()
        
        logger.info("\n✨ Coleta completa finalizada com sucesso!")
        logger.info(f"📁 Dados salvos em: {os.path.abspath(config['data_dir'])}")
        
        # Sugestões de próximos passos
        logger.info("\n📋 Próximos passos sugeridos:")
        logger.info("1. Verificar qualidade dos dados coletados")
        logger.info("2. Executar validação e limpeza dos dados")
        logger.info("3. Preparar dados para treinamento HMARL")
        logger.info("4. Configurar coleta automática diária")
        
    except Exception as e:
        logger.error(f"❌ Erro durante coleta: {e}", exc_info=True)
        if 'collector' in locals() and hasattr(collector, 'server_process'):
            collector.stop_server()


def show_collection_stats():
    """Mostra estatísticas dos dados coletados"""
    from src.database.historical_data_collector import HistoricalDataCollector
    
    logger.info("\n" + "="*60)
    logger.info("📊 ESTATÍSTICAS DOS DADOS HISTÓRICOS")
    logger.info("="*60)
    
    config = {'data_dir': 'data/historical'}
    collector = HistoricalDataCollector(config)
    
    available = collector.get_available_data()
    
    if not available:
        logger.info("Nenhum dado histórico encontrado.")
        return
    
    for symbol, dates in available.items():
        logger.info(f"\n{symbol}:")
        logger.info(f"  Total de dias: {len(dates)}")
        
        if dates:
            # Carregar um dia de exemplo
            try:
                sample_date = datetime.strptime(dates[-1], '%Y%m%d')
                df = collector._load_existing_data(
                    symbol, 
                    sample_date,
                    sample_date + timedelta(days=1),
                    ['trades']
                ).get('trades', pd.DataFrame())
            except:
                df = pd.DataFrame()
            
            if not df.empty:
                logger.info(f"  Trades por dia (média): {len(df):,}")
                logger.info(f"  Preço médio: R$ {df['price'].mean():.2f}")
                logger.info(f"  Volume médio: {df['volume'].mean():.0f}")
                
                # Horário de pregão
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                trades_by_hour = df.groupby('hour').size()
                
                logger.info(f"  Horário mais ativo: {trades_by_hour.idxmax()}h ({trades_by_hour.max()} trades)")


if __name__ == "__main__":
    import pandas as pd
    
    # Necessário para Windows
    multiprocessing.freeze_support()
    
    try:
        # Executar coleta completa
        run_full_collection()
        
        # Mostrar estatísticas
        show_collection_stats()
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Coleta interrompida pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}", exc_info=True)