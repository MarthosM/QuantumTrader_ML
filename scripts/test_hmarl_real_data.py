"""
Script para testar integração HMARL com dados reais do ProfitDLL
Conecta ao mercado real e processa dados com sistema multi-agente
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
from src.integration.hmarl_profitdll_integration import HMARLProfitDLLIntegration
from src.infrastructure.valkey_connection import ValkeyConnectionManager as ValkeyConnection

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestHMARLRealData')


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
            logger.info("Execute: docker-compose up -d valkey")
            return False
    except Exception as e:
        logger.error(f"❌ Erro conectando ao Valkey: {e}")
        logger.info("Certifique-se que o Valkey está rodando")
        return False
    
    # Verificar credenciais
    required_vars = ['PROFIT_USERNAME', 'PROFIT_PASSWORD', 'PROFIT_KEY']
    missing = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        logger.error(f"❌ Variáveis de ambiente faltando: {missing}")
        logger.info("Configure as credenciais no arquivo .env")
        return False
    
    logger.info("✅ Todos os pré-requisitos verificados")
    return True


def test_hmarl_integration():
    """Testa integração HMARL com dados reais"""
    
    if not verify_prerequisites():
        return
    
    logger.info("🚀 Iniciando teste de integração HMARL com dados reais")
    
    # Configuração
    config = {
        'ticker': 'WDOQ25',  # Ajustar conforme necessário
        'max_lookback_days': 3,
        'enable_flow_analysis': True,
        'enable_footprint': True,
        'enable_tape_reading': True
    }
    
    # Credenciais do ProfitDLL
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    username = os.getenv("PROFIT_USERNAME")
    password = os.getenv("PROFIT_PASSWORD")
    key = os.getenv("PROFIT_KEY")
    
    connection_manager = None
    hmarl_integration = None
    
    try:
        # 1. Criar e inicializar ConnectionManager
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
        
        # 2. Criar integração HMARL
        logger.info("🤖 Inicializando sistema HMARL...")
        hmarl_integration = HMARLProfitDLLIntegration(config)
        
        # 3. Conectar HMARL ao ProfitDLL
        hmarl_integration.connect_to_profitdll(connection_manager)
        
        # 4. Iniciar sistema HMARL
        hmarl_integration.start()
        logger.info("✅ Sistema HMARL iniciado!")
        
        # 5. Solicitar dados históricos para warm-up
        logger.info("📊 Solicitando dados históricos para warm-up...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)  # Último dia
        
        request_id = connection_manager.request_historical_data(
            ticker=config['ticker'],
            start_date=start_date,
            end_date=end_date
        )
        
        if request_id >= 0:
            logger.info(f"✅ Dados históricos solicitados (ID: {request_id})")
            
            # Aguardar dados históricos
            if connection_manager.wait_for_historical_data(timeout_seconds=120):
                logger.info("✅ Dados históricos carregados com sucesso!")
            else:
                logger.warning("⚠️ Timeout aguardando dados históricos")
        else:
            logger.warning("⚠️ Falha ao solicitar dados históricos")
        
        # 6. Subscrever para dados em tempo real
        logger.info(f"📈 Subscrevendo para dados em tempo real de {config['ticker']}...")
        if connection_manager.subscribe_ticker(config['ticker']):
            logger.info("✅ Subscrição realizada com sucesso!")
        else:
            logger.warning("⚠️ Falha na subscrição do ticker")
        
        # 7. Monitorar sistema por um período
        logger.info("🔄 Sistema rodando... (Pressione Ctrl+C para parar)")
        
        start_time = time.time()
        last_metrics_time = start_time
        
        while True:
            try:
                # Exibir métricas a cada 30 segundos
                current_time = time.time()
                if current_time - last_metrics_time >= 30:
                    metrics = hmarl_integration.get_metrics()
                    logger.info("📊 Métricas do Sistema HMARL:")
                    logger.info(f"  - Trades processados: {metrics['trades_processed']}")
                    logger.info(f"  - Features calculadas: {metrics['features_calculated']}")
                    logger.info(f"  - Sinais de agentes: {metrics['agent_signals']}")
                    logger.info(f"  - Erros: {metrics['errors']}")
                    logger.info(f"  - Fila de dados: {metrics['queue_size']}")
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
            hmarl_integration.stop()
            logger.info("✅ Sistema HMARL parado")
        
        if connection_manager:
            connection_manager.disconnect()
            logger.info("✅ Desconectado do ProfitDLL")
        
        logger.info("✨ Teste finalizado")


def main():
    """Função principal"""
    try:
        # Verificar se é horário de mercado
        now = datetime.now()
        weekday = now.weekday()
        hour = now.hour
        
        # Avisar se fora do horário
        if weekday >= 5:  # Fim de semana
            logger.warning("⚠️ AVISO: Mercado fechado (fim de semana)")
            logger.info("Dados históricos ainda funcionarão, mas não haverá dados em tempo real")
        elif hour < 9 or hour >= 18:  # Fora do pregão
            logger.warning(f"⚠️ AVISO: Fora do horário de pregão ({hour}h)")
            logger.info("Dados históricos ainda funcionarão, mas pode não haver dados em tempo real")
        
        # Executar teste
        test_hmarl_integration()
        
    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()