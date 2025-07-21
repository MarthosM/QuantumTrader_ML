#!/usr/bin/env python3
"""
Teste específico: Conexão com apenas 5 dias de dados históricos
Objetivo: Verificar se período reduzido resolve erro -2147483645
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connection_manager import ConnectionManager

# Configurar logging detalhado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_5_days.log')
    ]
)

logger = logging.getLogger('Test5Days')

def test_5_days_connection():
    """Testa conexão com apenas 5 dias de dados históricos"""
    
    logger.info("🚀 TESTE: Conexão com 5 dias de dados históricos")
    logger.info("=" * 60)
    
    # Configurações de teste
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    
    # Credenciais
    key = "SUA_CHAVE_AQUI"
    username = "SEU_USUARIO_AQUI" 
    password = "SUA_SENHA_AQUI"
    
    # Verificar se DLL existe
    if not os.path.exists(dll_path):
        logger.error(f"❌ DLL não encontrada: {dll_path}")
        return False
    
    try:
        # Criar connection manager
        conn = ConnectionManager(dll_path)
        
        # Definir período de 5 dias - MUY RECENTE
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        logger.info(f"📅 Período de teste:")
        logger.info(f"   Início: {start_date}")
        logger.info(f"   Fim: {end_date}")
        logger.info(f"   Dias: {(end_date - start_date).days}")
        
        # Callback para dados históricos
        historical_data_received = []
        
        def on_historical_data(trade_data):
            """Callback para capturar dados históricos"""
            if trade_data.get('source') == 'historical':
                historical_data_received.append(trade_data)
                if len(historical_data_received) % 50 == 0:
                    logger.info(f"📈 {len(historical_data_received)} dados históricos recebidos...")
        
        # Registrar callback
        conn.register_trade_callback(on_historical_data)
        
        # Inicializar conexão
        logger.info("🔌 Inicializando conexão...")
        if not conn.initialize(key, username, password):
            logger.error("❌ Falha na inicialização da conexão")
            return False
        
        logger.info("✅ Conexão inicializada com sucesso!")
        
        # Testar diferentes tickers com período reduzido
        tickers_to_test = ["WDOQ25", "WDO", "PETR4", "VALE3"]
        
        for ticker in tickers_to_test:
            logger.info(f"\n🎯 Testando ticker: {ticker}")
            logger.info("-" * 40)
            
            # Limpar contador
            conn._historical_data_count = 0
            historical_data_received.clear()
            
            # Solicitar dados históricos
            result = conn.request_historical_data(ticker, start_date, end_date)
            logger.info(f"📊 Resultado da solicitação: {result}")
            
            if result >= 0:
                logger.info(f"✅ Solicitação aceita para {ticker}!")
                
                # Aguardar dados com timeout menor (30s para 5 dias)
                if conn.wait_for_historical_data(30):
                    final_count = len(historical_data_received)
                    logger.info(f"🎉 SUCESSO! {final_count} dados históricos recebidos para {ticker}")
                    
                    # Se conseguiu dados, não precisa testar outros tickers
                    if final_count > 0:
                        logger.info(f"🏆 TICKER FUNCIONANDO: {ticker}")
                        logger.info(f"📊 Total de dados: {final_count}")
                        logger.info(f"⏱️ Período: {start_date.date()} até {end_date.date()}")
                        
                        # Mostrar alguns exemplos de dados
                        if len(historical_data_received) > 0:
                            logger.info("📈 Exemplos de dados recebidos:")
                            for i, data in enumerate(historical_data_received[:3]):
                                logger.info(f"   {i+1}: {data}")
                        
                        conn.disconnect()
                        return True
                else:
                    logger.warning(f"⚠️ Timeout aguardando dados de {ticker}")
            else:
                logger.error(f"❌ Erro {result} para ticker {ticker}")
                
                # Se erro -2147483645, dar detalhes
                if result == -2147483645:
                    logger.error("🔍 DIAGNÓSTICO DETALHADO:")
                    logger.error("   - Erro -2147483645 = Parâmetros inválidos")
                    logger.error("   - Possíveis causas:")
                    logger.error("     * Ticker inexistente ou inativo")
                    logger.error("     * Exchange incorreta")
                    logger.error("     * Formato de data inválido")
                    logger.error("     * Sem permissão para dados históricos")
                    logger.error("     * Período fora do disponível")
        
        # Se chegou aqui, nenhum ticker funcionou
        logger.error("❌ NENHUM TICKER FUNCIONOU com 5 dias")
        logger.error("💡 Possíveis soluções:")
        logger.error("   1. Verificar credenciais e permissões")
        logger.error("   2. Testar com ticker mais comum (PETR4)")
        logger.error("   3. Contactar suporte da corretora")
        logger.error("   4. Verificar se conta tem acesso a dados históricos")
        
        conn.disconnect()
        return False
        
    except Exception as e:
        logger.error(f"💥 Erro durante o teste: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🧪 INICIANDO TESTE COM 5 DIAS DE DADOS")
    logger.info("Objetivo: Verificar se período reduzido resolve o erro")
    logger.info("=" * 60)
    
    success = test_5_days_connection()
    
    logger.info("=" * 60)
    if success:
        logger.info("🎉 TESTE CONCLUÍDO COM SUCESSO!")
        logger.info("✅ Dados históricos funcionaram com 5 dias")
    else:
        logger.error("❌ TESTE FALHADO")
        logger.error("🔍 Problema persiste mesmo com período reduzido")
    
    logger.info("📋 Verifique o arquivo test_5_days.log para detalhes completos")
