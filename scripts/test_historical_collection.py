"""
Script de teste para coleta de dados históricos
Testa a coleta usando processo isolado
"""

import os
import sys
import logging
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
logger = logging.getLogger('TestHistoricalCollection')


def test_basic_collection():
    """Testa coleta básica de dados"""
    logger.info("="*60)
    logger.info("🧪 Teste de Coleta de Dados Históricos")
    logger.info("="*60)
    
    # Verificar credenciais
    required_vars = ['PROFIT_USERNAME', 'PROFIT_PASSWORD', 'PROFIT_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"❌ Variáveis de ambiente faltando: {missing}")
        logger.info("Configure as credenciais no arquivo .env")
        return
    
    # Configuração de teste
    config = {
        "symbols": ["WDOQ25"],
        "data_types": ["trades"],
        "data_dir": "data/historical_test",
        "csv_dir": "data/csv",
        "log_dir": "logs",
        "dll_path": r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll",
        "profitdll_server_port": 6789,
        "username": os.getenv("PROFIT_USERNAME"),
        "password": os.getenv("PROFIT_PASSWORD"),
        "key": os.getenv("PROFIT_KEY")
    }
    
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
        
        # Testar servidor isolado
        logger.info("\n1️⃣ Testando servidor isolado...")
        if collector.start_isolated_server():
            logger.info("✅ Servidor iniciado com sucesso")
        else:
            logger.error("❌ Falha ao iniciar servidor")
            return
        
        # Inicializar coletor
        from src.database.historical_data_collector import HistoricalDataCollector
        collector.collector = HistoricalDataCollector(config)
        
        # Testar coleta de período curto
        logger.info("\n2️⃣ Testando coleta de dados...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)  # Apenas 2 dias para teste
        
        logger.info(f"Período: {start_date.date()} até {end_date.date()}")
        
        # Coletar dados
        success = collector._collect_symbol_data(
            symbol="WDOQ25",
            start_date=start_date,
            end_date=end_date,
            data_types=["trades"]
        )
        
        if success:
            logger.info("✅ Coleta bem-sucedida!")
        else:
            logger.warning("⚠️ Coleta parcial ou sem dados")
        
        # Verificar dados coletados
        logger.info("\n3️⃣ Verificando dados salvos...")
        summary = collector.collector.get_data_summary("WDOQ25")
        
        if 'dates' in summary:
            logger.info(f"✅ Dados encontrados:")
            logger.info(f"   - Dias: {len(summary['dates'])}")
            logger.info(f"   - Tipos: {summary.get('data_types', [])}")
            logger.info(f"   - Tamanho: {summary.get('total_size_mb', 0):.2f} MB")
        else:
            logger.info("⚠️ Nenhum dado salvo encontrado")
        
        # Parar servidor
        collector.stop_server()
        logger.info("\n✅ Teste concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"❌ Erro durante teste: {e}", exc_info=True)
        if 'collector' in locals() and hasattr(collector, 'server_process'):
            collector.stop_server()


def test_load_saved_data():
    """Testa carregamento de dados salvos"""
    logger.info("\n"+"="*60)
    logger.info("📖 Teste de Carregamento de Dados")
    logger.info("="*60)
    
    try:
        from src.database.historical_data_collector import HistoricalDataCollector
        
        config = {
            'data_dir': 'data/historical_test'
        }
        
        collector = HistoricalDataCollector(config)
        
        # Listar dados disponíveis
        available = collector.get_available_data()
        
        if available:
            logger.info("Dados disponíveis:")
            for symbol, dates in available.items():
                logger.info(f"\n{symbol}:")
                logger.info(f"  Dias: {len(dates)}")
                if dates:
                    logger.info(f"  Primeiro: {dates[0]}")
                    logger.info(f"  Último: {dates[-1]}")
        else:
            logger.info("Nenhum dado salvo encontrado")
            return
        
        # Carregar dados de exemplo
        if available:
            symbol = list(available.keys())[0]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            df = collector.load_historical_data(symbol, start_date, end_date)
            
            if not df.empty:
                logger.info(f"\n✅ Carregados {len(df)} trades")
                logger.info("\nPrimeiros trades:")
                logger.info(df.head())
                
                # Estatísticas
                logger.info("\nEstatísticas:")
                logger.info(f"  Preço médio: {df['price'].mean():.2f}")
                logger.info(f"  Volume total: {df['volume'].sum():,.0f}")
            else:
                logger.info("⚠️ Nenhum dado no período especificado")
                
    except Exception as e:
        logger.error(f"❌ Erro no teste de carregamento: {e}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Executar testes
    logger.info("🚀 Iniciando testes de coleta histórica")
    
    # Teste 1: Coleta básica
    test_basic_collection()
    
    # Teste 2: Carregamento de dados
    test_load_saved_data()
    
    logger.info("\n✨ Todos os testes concluídos!")