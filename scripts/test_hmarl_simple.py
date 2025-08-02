"""
Script de teste simplificado para HMARL
Testa componentes isoladamente para identificar problemas
"""

import os
import sys
import time
import logging

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestHMARLSimple')


def test_infrastructure_only():
    """Testa apenas a infraestrutura HMARL sem ProfitDLL"""
    logger.info("=== TESTE 1: Infraestrutura HMARL ===")
    
    try:
        from src.infrastructure.zmq_valkey_flow_setup import TradingInfrastructureWithFlow
        
        config = {
            'symbol': 'WDOQ25',
            'valkey': {
                'host': 'localhost',
                'port': 6379
            }
        }
        
        # Criar e inicializar infraestrutura
        infrastructure = TradingInfrastructureWithFlow(config)
        
        if infrastructure.initialize():
            logger.info("✅ Infraestrutura inicializada com sucesso")
            
            # Testar Valkey
            if infrastructure.valkey_client:
                infrastructure.valkey_client.ping()
                logger.info("✅ Valkey conectado e respondendo")
            else:
                logger.error("❌ Valkey client é None")
                
            # Limpar
            infrastructure.stop()
            logger.info("✅ Infraestrutura parada com sucesso")
        else:
            logger.error("❌ Falha ao inicializar infraestrutura")
            
    except Exception as e:
        logger.error(f"❌ Erro no teste de infraestrutura: {e}", exc_info=True)


def test_agents_only():
    """Testa agentes HMARL isoladamente"""
    logger.info("\n=== TESTE 2: Agentes HMARL ===")
    
    try:
        from src.agents.order_flow_specialist import OrderFlowSpecialistAgent
        
        # Criar agente simples
        agent = OrderFlowSpecialistAgent({
            'ofi_threshold': 0.3,
            'use_registry': False  # Desabilitar registry por enquanto
        })
        
        logger.info(f"✅ Agente criado: {agent.agent_id}")
        
        # Simular alguns dados
        agent.process_market_data({
            'price': 5000.0,
            'volume': 100,
            'timestamp': time.time()
        })
        
        agent.process_flow_data({
            'order_flow_imbalance': 0.5,
            'aggression_score': 0.7,
            'delta': 150,
            'footprint_pattern': 'accumulation'
        })
        
        # Gerar sinal
        signal = agent.generate_signal_with_flow(
            agent.state['price_state'],
            agent.state['flow_state']
        )
        
        logger.info(f"✅ Sinal gerado: {signal}")
        
        # Limpar
        agent.shutdown()
        logger.info("✅ Agente finalizado com sucesso")
        
    except Exception as e:
        logger.error(f"❌ Erro no teste de agentes: {e}", exc_info=True)


def test_integration_without_profitdll():
    """Testa integração HMARL sem conectar ao ProfitDLL"""
    logger.info("\n=== TESTE 3: Integração HMARL (sem ProfitDLL) ===")
    
    try:
        from src.integration.hmarl_profitdll_integration import HMARLProfitDLLIntegration
        
        config = {
            'ticker': 'WDOQ25',
            'valkey': {
                'host': 'localhost',
                'port': 6379
            }
        }
        
        # Criar integração
        logger.info("Criando integração HMARL...")
        hmarl = HMARLProfitDLLIntegration(config)
        logger.info("✅ Integração criada com sucesso")
        
        # Verificar componentes
        logger.info(f"Infrastructure: {hmarl.infrastructure}")
        logger.info(f"Valkey client: {hmarl.infrastructure.valkey_client}")
        logger.info(f"Agentes criados: {len(hmarl.agents)}")
        
        # Simular processamento de dados
        test_trade = {
            'timestamp': time.time(),
            'ticker': 'WDOQ25',
            'price': 5000.0,
            'volume': 100,
            'quantity': 100,
            'trade_type': 2  # Buy
        }
        
        enriched = hmarl._process_trade_data(test_trade)
        if enriched:
            logger.info("✅ Dados processados com sucesso")
        
        # Obter métricas
        metrics = hmarl.get_metrics()
        logger.info(f"Métricas: {metrics}")
        
        # Limpar
        hmarl.stop()
        logger.info("✅ Integração finalizada com sucesso")
        
    except Exception as e:
        logger.error(f"❌ Erro no teste de integração: {e}", exc_info=True)


def main():
    """Executa testes isolados"""
    logger.info("🚀 Iniciando testes simplificados HMARL\n")
    
    # Teste 1: Infraestrutura
    test_infrastructure_only()
    
    # Teste 2: Agentes
    test_agents_only()
    
    # Teste 3: Integração sem ProfitDLL
    test_integration_without_profitdll()
    
    logger.info("\n✨ Testes simplificados concluídos")


if __name__ == "__main__":
    main()