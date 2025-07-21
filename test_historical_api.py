#!/usr/bin/env python3
"""
Teste específico da API de dados históricos
Para diagnosticar e corrigir problemas de carregamento
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from connection_manager import ConnectionManager

def setup_logging():
    """Configura logging detalhado"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_historical_api.log')
        ]
    )

def test_historical_api():
    """Teste isolado da API de dados históricos"""
    
    print("=== TESTE DA API DE DADOS HISTÓRICOS ===")
    print()
    
    # Configurar logging
    setup_logging()
    logger = logging.getLogger('TEST')
    
    try:
        # 1. Carregar configurações
        from dotenv import load_dotenv
        load_dotenv()
        
        key = os.getenv('PROFIT_KEY')
        username = os.getenv('PROFIT_USERNAME') 
        password = os.getenv('PROFIT_PASSWORD')
        account = os.getenv('PROFIT_ACCOUNT')
        broker = os.getenv('PROFIT_BROKER')
        
        if not all([key, username, password]):
            logger.error("❌ Credenciais não encontradas no .env")
            logger.error("Necessário: PROFIT_KEY, PROFIT_USERNAME, PROFIT_PASSWORD")
            return False
            
        logger.info("✅ Credenciais carregadas")
        
        # 2. Inicializar conexão
        logger.info("🔌 Inicializando conexão...")
        
        dll_path = os.getenv('PROFIT_DLL_PATH', r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll")
        connection = ConnectionManager(dll_path)
        
        # 3. Conectar
        logger.info("🔐 Conectando ao Profit...")
        success = connection.initialize(
            key=str(key),
            username=str(username), 
            password=str(password),
            account_id=account,
            broker_id=broker
        )
        
        if not success:
            logger.error("❌ Falha na conexão")
            return False
            
        logger.info("✅ Conexão estabelecida!")
        
        # 4. Aguardar estabilização
        import time
        logger.info("⏳ Aguardando estabilização da conexão...")
        time.sleep(3)
        
        # 5. Testar dados históricos
        logger.info("📊 Testando dados históricos...")
        
        ticker = "WDOQ25"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)  # 5 dias atrás
        
        logger.info(f"Ticker: {ticker}")
        logger.info(f"Período: {start_date.date()} até {end_date.date()}")
        
        # 6. Fazer requisição
        logger.info("📞 Fazendo requisição de dados históricos...")
        result = connection.request_historical_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )
        
        if result >= 0:
            logger.info(f"✅ Requisição aceita! Código: {result}")
            
            # 7. Aguardar dados
            logger.info("⏳ Aguardando dados via callback...")
            success = connection.wait_for_historical_data(timeout_seconds=120)
            
            if success:
                count = getattr(connection, '_historical_data_count', 0)
                logger.info(f"🎉 SUCESSO! {count} dados históricos recebidos")
                return True
            else:
                logger.error("❌ Timeout ou erro ao receber dados")
                
        else:
            logger.error(f"❌ Requisição falhou! Código: {result}")
            
            # Tentar método alternativo
            logger.info("🔄 Tentando método alternativo...")
            result_alt = connection.request_historical_data_alternative(
                ticker=ticker,
                start_date=start_date, 
                end_date=end_date
            )
            
            if result_alt >= 0:
                logger.info(f"✅ Método alternativo aceito! Código: {result_alt}")
                success = connection.wait_for_historical_data(timeout_seconds=120)
                
                if success:
                    count = getattr(connection, '_historical_data_count', 0)
                    logger.info(f"🎉 SUCESSO ALTERNATIVO! {count} dados históricos recebidos")
                    return True
            
        return False
        
    except Exception as e:
        logger.error(f"💥 Erro no teste: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return False
    
    finally:
        # Cleanup
        try:
            if 'connection' in locals():
                connection.disconnect()
        except:
            pass

if __name__ == "__main__":
    print("🧪 Executando teste da API de dados históricos...")
    print()
    
    success = test_historical_api()
    
    print()
    if success:
        print("🎉 TESTE PASSOU! API de dados históricos funcionando")
        sys.exit(0)
    else:
        print("❌ TESTE FALHOU! Verifique os logs para detalhes")
        sys.exit(1)
