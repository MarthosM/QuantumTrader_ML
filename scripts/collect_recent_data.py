"""
Script para coletar dados históricos recentes (últimos 30 dias)
Usa servidor isolado para evitar Segmentation Fault
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
logger = logging.getLogger('RecentDataCollection')


def collect_recent_data():
    """Coleta dados dos últimos 30 dias"""
    logger.info("="*80)
    logger.info("📊 COLETA DE DADOS RECENTES (ÚLTIMOS 30 DIAS)")
    logger.info("="*80)
    
    # Verificar credenciais
    required_vars = ['PROFIT_USERNAME', 'PROFIT_PASSWORD', 'PROFIT_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"❌ Variáveis de ambiente faltando: {missing}")
        return
    
    logger.info("✅ Credenciais verificadas")
    
    # Configuração para coleta recente
    config = {
        "symbols": ["WDOU25"],  # WDO Setembro - contrato mais recente
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
    
    # Período de coleta - últimos 30 dias úteis
    end_date = datetime.now() - timedelta(days=1)  # Ontem
    start_date = end_date - timedelta(days=30)
    
    logger.info(f"\n📅 Período de coleta: {start_date.date()} até {end_date.date()}")
    logger.info(f"📊 Símbolo: {config['symbols'][0]} (WDO Setembro 2025)")
    
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
        logger.info("\n🚀 Iniciando servidor ProfitDLL isolado...")
        if not collector.start_isolated_server():
            logger.error("❌ Falha ao iniciar servidor")
            return
        
        logger.info("✅ Servidor ProfitDLL rodando em processo isolado")
        
        # Inicializar coletor
        from src.database.historical_data_collector import HistoricalDataCollector
        collector.collector = HistoricalDataCollector(config)
        
        # Executar coleta
        total_start = time.time()
        
        for symbol in config['symbols']:
            logger.info(f"\n{'='*60}")
            logger.info(f"📈 Coletando dados de {symbol}")
            logger.info(f"{'='*60}")
            
            try:
                # Coletar dados do período
                data = collector.collector.collect_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    data_types=config['data_types']
                )
                
                # Verificar resultado
                if 'trades' in data and not data['trades'].empty:
                    trades_count = len(data['trades'])
                    logger.info(f"✅ {trades_count:,} trades coletados")
                    
                    # Mostrar amostra dos dados
                    logger.info("\n📊 Amostra dos dados coletados:")
                    logger.info(data['trades'].head())
                    
                    # Estatísticas
                    logger.info(f"\n📈 Estatísticas:")
                    logger.info(f"  Preço médio: R$ {data['trades']['price'].mean():.2f}")
                    logger.info(f"  Volume total: {data['trades']['volume'].sum():,.0f}")
                    logger.info(f"  Período: {data['trades']['datetime'].min()} até {data['trades']['datetime'].max()}")
                else:
                    logger.warning(f"⚠️ Nenhum dado coletado para {symbol}")
                    
                    # Tentar com datas específicas recentes
                    logger.info("\n🔄 Tentando período mais recente (últimos 7 dias)...")
                    end_date_recent = datetime.now() - timedelta(days=1)
                    start_date_recent = end_date_recent - timedelta(days=7)
                    
                    data_recent = collector.collector.collect_historical_data(
                        symbol=symbol,
                        start_date=start_date_recent,
                        end_date=end_date_recent,
                        data_types=config['data_types']
                    )
                    
                    if 'trades' in data_recent and not data_recent['trades'].empty:
                        logger.info(f"✅ {len(data_recent['trades']):,} trades dos últimos 7 dias")
                    else:
                        logger.warning("⚠️ Ainda sem dados - verifique se o mercado está aberto")
                
            except Exception as e:
                logger.error(f"❌ Erro coletando {symbol}: {e}")
                
        # Tempo total
        total_time = time.time() - total_start
        logger.info(f"\n⏱️ Tempo total: {total_time/60:.1f} minutos")
        
        # Verificar dados salvos
        logger.info("\n📁 Verificando dados salvos...")
        available = collector.collector.get_available_data()
        
        if available:
            for symbol, dates in available.items():
                logger.info(f"\n{symbol}: {len(dates)} arquivos")
                if dates:
                    logger.info(f"  Período: {dates[0]} até {dates[-1]}")
        else:
            logger.info("Nenhum arquivo salvo encontrado")
        
        # Parar servidor
        collector.stop_server()
        
        logger.info("\n✨ Coleta finalizada!")
        
        # Sugestões
        logger.info("\n💡 Dicas:")
        logger.info("1. Se não coletou dados, verifique se o mercado está aberto")
        logger.info("2. WDOU25 é o contrato de setembro (mais líquido agora)")
        logger.info("3. Dados históricos têm limite de 3 meses no ProfitDLL")
        logger.info("4. Execute durante o horário de pregão para melhores resultados")
        
    except Exception as e:
        logger.error(f"❌ Erro durante coleta: {e}", exc_info=True)
        if 'collector' in locals() and hasattr(collector, 'server_process'):
            collector.stop_server()


def test_direct_connection():
    """Testa conexão direta com GetHistoryTrades"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Teste de Conexão Direta")
    logger.info("="*60)
    
    from src.connection_manager_v4 import ConnectionManagerV4
    
    config = {
        'dll_path': r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll",
        'username': os.getenv("PROFIT_USERNAME"),
        'password': os.getenv("PROFIT_PASSWORD"),
        'key': os.getenv("PROFIT_KEY")
    }
    
    try:
        # Criar connection manager
        conn = ConnectionManagerV4(config['dll_path'])
        
        # Conectar
        if conn.initialize(
            key=config['key'],
            username=config['username'],
            password=config['password']
        ):
            logger.info("✅ Conectado ao ProfitDLL")
            
            # Testar GetHistoryTrades com período recente
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=5)
            
            logger.info(f"\n📅 Testando período: {start_date.date()} até {end_date.date()}")
            
            # Registrar callback
            trades_received = []
            
            def on_history_trade(data):
                trades_received.append(data)
                if len(trades_received) <= 5:
                    logger.info(f"Trade recebido: {data.get('ticker')} @ {data.get('price')}")
            
            conn.register_history_trade_callback(on_history_trade)
            
            # Testar diferentes formatos de data
            date_formats = [
                (start_date.strftime('%d/%m/%Y'), end_date.strftime('%d/%m/%Y')),
                (start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')),
                ('01/08/2025', '02/08/2025'),
                ('20250801', '20250802')
            ]
            
            for start_fmt, end_fmt in date_formats:
                logger.info(f"\nTestando formato: {start_fmt} até {end_fmt}")
                
                result = conn.get_history_trades(
                    ticker="WDOU25",
                    exchange="BMF",
                    date_start=start_fmt,
                    date_end=end_fmt
                )
                
                if result:
                    logger.info("✅ Solicitação enviada com sucesso")
                    time.sleep(3)
                    
                    if trades_received:
                        logger.info(f"✅ Recebidos {len(trades_received)} trades!")
                        break
                else:
                    logger.warning("❌ Falha na solicitação")
            
            # Desconectar
            conn.disconnect()
            
        else:
            logger.error("❌ Falha ao conectar")
            
    except Exception as e:
        logger.error(f"❌ Erro no teste: {e}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    try:
        # Executar coleta de dados recentes
        collect_recent_data()
        
        # Opcional: testar conexão direta
        # logger.info("\n" + "="*80)
        # test_direct_connection()
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Coleta interrompida pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}", exc_info=True)