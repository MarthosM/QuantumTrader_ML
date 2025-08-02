"""
Script para testar integração HMARL thread-safe
Usa buffer isolado para prevenir Segmentation Fault
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar componentes
from src.connection_manager_v4 import ConnectionManagerV4
from src.integration.hmarl_profitdll_integration_safe import HMARLProfitDLLIntegrationSafe
from src.infrastructure.valkey_connection import ValkeyConnectionManager as ValkeyConnection

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestHMARLSafe')


def verify_prerequisites():
    """Verifica pré-requisitos antes de iniciar"""
    logger.info("🔍 Verificando pré-requisitos...")
    
    # Verificar Valkey
    try:
        valkey_conn = ValkeyConnection()
        if valkey_conn.ping():
            logger.info("✅ Valkey está rodando")
        else:
            logger.error("❌ Valkey não está respondendo")
            return False
    except Exception as e:
        logger.error(f"❌ Erro conectando ao Valkey: {e}")
        return False
    
    # Verificar credenciais
    required_vars = ['PROFIT_USERNAME', 'PROFIT_PASSWORD', 'PROFIT_KEY']
    missing = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        logger.error(f"❌ Variáveis de ambiente faltando: {missing}")
        return False
    
    logger.info("✅ Todos os pré-requisitos verificados")
    return True


def test_hmarl_safe():
    """Testa integração HMARL thread-safe"""
    
    if not verify_prerequisites():
        return
    
    logger.info("🚀 Iniciando teste HMARL Thread-Safe")
    
    # Configuração
    config = {
        'ticker': 'WDOQ25',
        'max_lookback_days': 3,
        'enable_flow_analysis': True,
        'enable_footprint': True,
        'enable_tape_reading': True,
        'valkey': {
            'host': 'localhost',
            'port': 6379
        }
    }
    
    # Credenciais do ProfitDLL
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    username = os.getenv("PROFIT_USERNAME")
    password = os.getenv("PROFIT_PASSWORD")
    key = os.getenv("PROFIT_KEY")
    
    connection_manager = None
    hmarl_integration = None
    
    try:
        # 1. Criar integração HMARL thread-safe
        logger.info("🤖 Inicializando sistema HMARL thread-safe...")
        hmarl_integration = HMARLProfitDLLIntegrationSafe(config)
        
        # 2. Inicializar HMARL ANTES de conectar ao ProfitDLL
        hmarl_integration.initialize_hmarl()
        logger.info("✅ HMARL inicializado com buffer thread-safe")
        
        # 3. Criar e inicializar ConnectionManager
        logger.info("📡 Conectando ao ProfitDLL...")
        connection_manager = ConnectionManagerV4(dll_path)
        
        if not connection_manager.initialize(
            key=key,
            username=username,
            password=password
        ):
            logger.error("❌ Falha ao conectar ao ProfitDLL")
            return
        
        logger.info("✅ Conectado ao ProfitDLL com sucesso!")
        
        # 4. Conectar HMARL ao ProfitDLL (com callbacks isolados)
        hmarl_integration.connect_to_profitdll(connection_manager)
        
        # 5. Iniciar sistema HMARL
        hmarl_integration.start()
        logger.info("✅ Sistema HMARL iniciado com proteção thread-safe!")
        
        # 6. Solicitar dados históricos para warm-up
        logger.info("📊 Solicitando dados históricos...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        request_id = connection_manager.request_historical_data(
            ticker=config['ticker'],
            start_date=start_date,
            end_date=end_date
        )
        
        if request_id >= 0:
            logger.info(f"✅ Dados históricos solicitados (ID: {request_id})")
            
            # Aguardar dados históricos
            if connection_manager.wait_for_historical_data(timeout_seconds=60):
                logger.info("✅ Dados históricos carregados!")
            else:
                logger.warning("⚠️ Timeout aguardando dados históricos")
        
        # 7. Subscrever para dados em tempo real
        logger.info(f"📈 Subscrevendo para dados em tempo real de {config['ticker']}...")
        if connection_manager.subscribe_ticker(config['ticker']):
            logger.info("✅ Subscrição realizada com sucesso!")
        else:
            logger.warning("⚠️ Falha na subscrição do ticker")
        
        # 8. Monitorar sistema
        logger.info("🔄 Sistema rodando com proteção thread-safe... (Ctrl+C para parar)")
        
        start_time = time.time()
        last_metrics_time = start_time
        metrics_interval = 30  # segundos
        
        while True:
            try:
                current_time = time.time()
                
                # Exibir métricas periodicamente
                if current_time - last_metrics_time >= metrics_interval:
                    metrics = hmarl_integration.get_metrics()
                    
                    logger.info("📊 === MÉTRICAS DO SISTEMA ===")
                    logger.info(f"  Trades processados: {metrics['trades_processed']}")
                    logger.info(f"  Features calculadas: {metrics['features_calculated']}")
                    logger.info(f"  Sinais de agentes: {metrics['agent_signals']}")
                    logger.info(f"  Buffer size: {metrics['buffer_size']}")
                    logger.info(f"  Buffer efficiency: {metrics['buffer_efficiency']:.2%}")
                    logger.info(f"  Buffer drops: {metrics['buffer_drops']}")
                    logger.info(f"  Erros: {metrics['errors']}")
                    logger.info(f"  Tempo rodando: {int(current_time - start_time)}s")
                    logger.info("  " + "="*30)
                    
                    last_metrics_time = current_time
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("\n⏹️ Interrupção do usuário detectada")
                break
        
    except Exception as e:
        logger.error(f"❌ Erro durante teste: {e}", exc_info=True)
        
    finally:
        # Limpar recursos
        logger.info("🧹 Limpando recursos...")
        
        if hmarl_integration:
            try:
                hmarl_integration.stop()
                logger.info("✅ Sistema HMARL parado")
            except Exception as e:
                logger.error(f"Erro parando HMARL: {e}")
        
        if connection_manager:
            try:
                connection_manager.disconnect()
                logger.info("✅ Desconectado do ProfitDLL")
            except Exception as e:
                logger.error(f"Erro desconectando ProfitDLL: {e}")
        
        logger.info("✨ Teste thread-safe finalizado")


def main():
    """Função principal"""
    try:
        # Verificar horário
        now = datetime.now()
        weekday = now.weekday()
        hour = now.hour
        
        if weekday >= 5:
            logger.warning("⚠️ AVISO: Mercado fechado (fim de semana)")
        elif hour < 9 or hour >= 18:
            logger.warning(f"⚠️ AVISO: Fora do horário de pregão ({hour}h)")
        
        logger.info("Este teste usa buffer thread-safe para prevenir Segmentation Fault")
        logger.info("="*60)
        
        # Executar teste
        test_hmarl_safe()
        
    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()