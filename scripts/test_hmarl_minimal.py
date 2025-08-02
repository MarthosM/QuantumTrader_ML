"""
Teste mínimo HMARL + ProfitDLL
Sem threads adicionais para identificar problema
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestHMARLMinimal')


def test_minimal():
    """Teste mínimo sem threads extras"""
    
    logger.info("🚀 Teste Mínimo HMARL + ProfitDLL")
    
    # 1. Testar infraestrutura HMARL isolada
    logger.info("\n=== PASSO 1: Infraestrutura HMARL ===")
    try:
        from src.infrastructure.zmq_valkey_flow_setup import TradingInfrastructureWithFlow
        
        config = {
            'ticker': 'WDOQ25',
            'valkey': {'host': 'localhost', 'port': 6379}
        }
        
        infrastructure = TradingInfrastructureWithFlow(config)
        if infrastructure.initialize():
            logger.info("✅ Infraestrutura HMARL OK")
            infrastructure.stop()
        else:
            logger.error("❌ Falha na infraestrutura")
            return
    except Exception as e:
        logger.error(f"❌ Erro na infraestrutura: {e}")
        return
    
    # 2. Criar integração HMARL (sem ProfitDLL ainda)
    logger.info("\n=== PASSO 2: Integração HMARL ===")
    try:
        from src.integration.hmarl_profitdll_integration import HMARLProfitDLLIntegration
        
        hmarl = HMARLProfitDLLIntegration(config)
        logger.info("✅ Integração HMARL criada")
        logger.info(f"   Agentes: {len(hmarl.agents)}")
        logger.info(f"   Valkey: {hmarl.infrastructure.valkey_client is not None}")
        
    except Exception as e:
        logger.error(f"❌ Erro criando integração: {e}")
        return
    
    # 3. Conectar ProfitDLL
    logger.info("\n=== PASSO 3: Conexão ProfitDLL ===")
    try:
        from src.connection_manager_v4 import ConnectionManagerV4
        
        dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
        username = os.getenv("PROFIT_USERNAME")
        password = os.getenv("PROFIT_PASSWORD")
        key = os.getenv("PROFIT_KEY")
        
        connection = ConnectionManagerV4(dll_path)
        
        if connection.initialize(key=key, username=username, password=password):
            logger.info("✅ ProfitDLL conectado")
        else:
            logger.error("❌ Falha conectando ProfitDLL")
            return
            
    except Exception as e:
        logger.error(f"❌ Erro com ProfitDLL: {e}")
        return
    
    # 4. Conectar HMARL ao ProfitDLL (sem iniciar threads)
    logger.info("\n=== PASSO 4: Conectar HMARL ao ProfitDLL ===")
    try:
        hmarl.connect_to_profitdll(connection)
        logger.info("✅ HMARL conectado ao ProfitDLL")
        
        # NÃO iniciar threads - apenas verificar conexão
        logger.info("✅ Conexão estabelecida sem threads extras")
        
    except Exception as e:
        logger.error(f"❌ Erro conectando HMARL ao ProfitDLL: {e}")
        return
    
    # 5. Testar processamento de um trade simulado
    logger.info("\n=== PASSO 5: Teste de processamento ===")
    try:
        test_trade = {
            'timestamp': datetime.now(),
            'ticker': 'WDOQ25',
            'price': 5000.0,
            'volume': 100,
            'quantity': 100,
            'trade_type': 2
        }
        
        # Chamar callback diretamente
        hmarl._on_trade_data(test_trade)
        logger.info("✅ Trade processado sem erro")
        
        # Verificar métricas
        metrics = hmarl.get_metrics()
        logger.info(f"   Trades processados: {metrics['trades_processed']}")
        logger.info(f"   Erros: {metrics['errors']}")
        
    except Exception as e:
        logger.error(f"❌ Erro processando trade: {e}")
    
    # 6. Limpar recursos
    logger.info("\n=== PASSO 6: Limpeza ===")
    try:
        hmarl.stop()
        logger.info("✅ HMARL parado")
        
        connection.disconnect()
        logger.info("✅ ProfitDLL desconectado")
        
    except Exception as e:
        logger.error(f"❌ Erro na limpeza: {e}")
    
    logger.info("\n✨ Teste mínimo concluído")


if __name__ == "__main__":
    try:
        test_minimal()
    except KeyboardInterrupt:
        logger.info("\n⏹️ Teste interrompido pelo usuário")
    except Exception as e:
        logger.error(f"\n❌ Erro fatal: {e}", exc_info=True)