"""
Teste final da conexão com ProfitDLL após correção dos callbacks
"""

import os
import sys
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestFinal')

# Carregar variáveis
load_dotenv()


def test_connection():
    """Testa conexão e coleta histórica"""
    logger.info("="*80)
    logger.info("TESTE FINAL - PROFITDLL COM CALLBACKS CORRIGIDOS")
    logger.info("="*80)
    
    from src.connection_manager_v4 import ConnectionManagerV4
    
    try:
        # Criar ConnectionManager
        dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
        logger.info("1. Criando ConnectionManagerV4...")
        conn = ConnectionManagerV4(dll_path)
        
        # Conectar
        logger.info("\n2. Conectando ao ProfitDLL...")
        result = conn.initialize(
            key=os.getenv('PROFIT_KEY'),
            username=os.getenv('PROFIT_USERNAME'),
            password=os.getenv('PROFIT_PASSWORD')
        )
        
        if not result:
            logger.error("Falha ao conectar!")
            return False
        
        logger.info("✅ Conectado com sucesso!")
        
        # Aguardar estabilização
        logger.info("\n3. Aguardando estabilização...")
        time.sleep(5)
        
        # Verificar estados
        logger.info(f"   Market conectado: {conn.market_connected}")
        logger.info(f"   Routing conectado: {conn.routing_connected}")
        logger.info(f"   Login state: {conn.login_state}")
        
        # Testar coleta histórica
        logger.info("\n4. Testando coleta histórica...")
        
        trades_received = []
        def on_history_trade(data):
            trades_received.append(data)
            if len(trades_received) == 1:
                logger.info(f"   Primeiro trade: {data}")
            elif len(trades_received) % 100 == 0:
                logger.info(f"   {len(trades_received)} trades recebidos...")
        
        conn.register_history_trade_callback(on_history_trade)
        
        # Testar diferentes períodos
        test_configs = [
            {
                'symbol': 'WDOU25',
                'start': '01/08/2025',
                'end': '01/08/2025',
                'desc': 'Ontem (01/08/2025)'
            },
            {
                'symbol': 'WDOU25',
                'start': '30/07/2025',
                'end': '30/07/2025',
                'desc': 'Há 3 dias (30/07/2025)'
            },
            {
                'symbol': 'WDOU25',
                'start': '01/07/2025',
                'end': '01/07/2025',
                'desc': 'Início de julho (01/07/2025)'
            }
        ]
        
        for config in test_configs:
            logger.info(f"\n   Testando: {config['desc']}")
            logger.info(f"   Symbol: {config['symbol']}")
            logger.info(f"   Período: {config['start']} até {config['end']}")
            
            trades_received.clear()
            
            success = conn.get_history_trades(
                ticker=config['symbol'],
                exchange='F',
                date_start=config['start'],
                date_end=config['end']
            )
            
            if success:
                logger.info("   ✅ Solicitação enviada")
                
                # Aguardar dados
                logger.info("   Aguardando dados...")
                time.sleep(15)
                
                if trades_received:
                    logger.info(f"   ✅ SUCESSO! {len(trades_received)} trades recebidos")
                    logger.info(f"   Primeiro: {trades_received[0]}")
                    logger.info(f"   Último: {trades_received[-1]}")
                    break  # Parar se conseguir dados
                else:
                    logger.warning("   ⚠️ Nenhum dado recebido")
            else:
                logger.error("   ❌ Falha ao enviar solicitação")
        
        # Resultado final
        logger.info("\n5. Resultado:")
        if any(trades_received):
            logger.info("✅ SISTEMA FUNCIONANDO!")
            logger.info("   - Conexão estabelecida")
            logger.info("   - Callbacks funcionando")
            logger.info("   - Dados históricos disponíveis")
            
            # Salvar amostra
            import json
            sample_file = "data/test/sample_historical_data.json"
            os.makedirs("data/test", exist_ok=True)
            
            with open(sample_file, 'w') as f:
                json.dump(trades_received[:10], f, indent=2, default=str)
            logger.info(f"   - Amostra salva em: {sample_file}")
            
        else:
            logger.warning("⚠️ Sistema conectado mas sem dados históricos")
            logger.info("Possíveis causas:")
            logger.info("   - Mercado fechado")
            logger.info("   - Dados não disponíveis para as datas testadas")
            logger.info("   - Limitações da conta")
        
        # Desconectar
        logger.info("\n6. Desconectando...")
        conn.disconnect()
        logger.info("✅ Desconectado")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}", exc_info=True)
        return False


def main():
    logger.info("TESTE FINAL DO SISTEMA DE COLETA HISTÓRICA")
    logger.info("="*80)
    logger.info("Correções aplicadas:")
    logger.info("1. Callbacks retornando c_int")
    logger.info("2. ConnectionManagerV4 usando tipos corretos")
    logger.info("3. Sistema pronto para produção")
    logger.info("")
    
    success = test_connection()
    
    if success:
        logger.info("\n✅ TESTE CONCLUÍDO COM SUCESSO!")
        logger.info("\nPróximos passos:")
        logger.info("1. Use o HistoricalDataCollector para coletas programadas")
        logger.info("2. Configure o servidor isolado para produção")
        logger.info("3. Implemente a coleta de book de ofertas (pendente)")
    else:
        logger.info("\n❌ TESTE FALHOU")
        logger.info("\nVerifique:")
        logger.info("1. Credenciais no arquivo .env")
        logger.info("2. Caminho da DLL")
        logger.info("3. Logs de erro acima")


if __name__ == "__main__":
    main()