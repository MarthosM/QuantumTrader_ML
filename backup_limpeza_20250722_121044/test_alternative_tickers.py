#!/usr/bin/env python3
"""
Teste com tickers alternativos para verificar se o problema é específico do WDOQ25
ou um problema geral da API de dados históricos
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_alternative_tickers():
    """Testa a API com tickers alternativos mais comuns"""
    
    print("=== TESTE COM TICKERS ALTERNATIVOS ===")
    print()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger('TEST')
    
    try:
        # Importar e configurar
        from dotenv import load_dotenv
        from connection_manager import ConnectionManager
        
        load_dotenv()
        
        key = os.getenv('PROFIT_KEY')
        username = os.getenv('PROFIT_USERNAME') 
        password = os.getenv('PROFIT_PASSWORD')
        account = os.getenv('PROFIT_ACCOUNT')
        broker = os.getenv('PROFIT_BROKER')
        
        if not all([key, username, password]):
            print("❌ Credenciais não encontradas - favor configurar .env")
            return False
            
        # Conectar
        dll_path = os.getenv('PROFIT_DLL_PATH', r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll")
        connection = ConnectionManager(dll_path)
        
        success = connection.initialize(
            key=str(key),
            username=str(username), 
            password=str(password),
            account_id=account,
            broker_id=broker
        )
        
        if not success:
            print("❌ Falha na conexão")
            return False
            
        print("✅ Conexão estabelecida!")
        
        # Aguardar estabilização
        import time
        time.sleep(3)
        
        # Lista de tickers para testar (dos mais comuns aos específicos)
        tickers_to_test = [
            # Ações mais líquidas da Bovespa
            ("PETR4", "", "Petrobras PN"),
            ("VALE3", "", "Vale ON"), 
            ("ITUB4", "", "Itaú PN"),
            ("BBDC4", "", "Bradesco PN"),
            
            # Índices
            ("IBOV", "", "Índice Bovespa"),
            ("IBXX", "", "Índice IBrX-100"),
            
            # Futuros mais comuns (se existirem)
            ("WDO", "F", "Dólar Futuro Genérico"),
            ("DOL", "F", "Dólar Alternativo"),
            ("IND", "F", "Índice Futuro"),
            
            # Contratos específicos (mês atual)
            ("WDOQ25", "F", "WDO Julho 2025"),
            ("WDON25", "F", "WDO Junho 2025"),  
        ]
        
        # Testar cada ticker
        found_working_ticker = False
        
        for ticker, exchange, description in tickers_to_test:
            print(f"\n🔍 Testando: {ticker} ({description})")
            print(f"   Exchange: '{exchange}'")
            
            # Período de teste curto
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3)
            
            # Reset contador
            connection._historical_data_count = 0
            
            # Fazer requisição
            result = connection.request_historical_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )
            
            print(f"   Resultado: {result}")
            
            if result >= 0:
                print(f"   ✅ SUCESSO! Ticker {ticker} funcionou!")
                print(f"   ⏳ Aguardando dados...")
                
                # Aguardar dados
                success = connection.wait_for_historical_data(timeout_seconds=30)
                
                if success:
                    count = connection._historical_data_count
                    print(f"   🎉 {count} dados históricos recebidos!")
                    found_working_ticker = True
                    
                    # Se encontrou um que funciona, testar mais alguns similares
                    if not found_working_ticker:
                        print(f"   🚀 {ticker} é um ticker válido! Continuando testes...")
                    break
                else:
                    print(f"   ⚠️ Requisição aceita mas sem dados recebidos")
            else:
                print(f"   ❌ Falhou: {result}")
                
        # Resultado final
        if found_working_ticker:
            print(f"\n🎉 SUCESSO! Encontrou pelo menos um ticker funcionando")
            print(f"💡 Problema é específico do WDOQ25 - ticker pode estar vencido/inativo")
            print(f"📝 Recomendação: Usar ticker genérico 'WDO' ou verificar contratos ativos")
        else:
            print(f"\n❌ NENHUM ticker funcionou")
            print(f"💡 Problema é geral na API de dados históricos")
            print(f"📝 Possíveis causas:")
            print(f"   - Conta não tem permissão para dados históricos")
            print(f"   - Servidor de dados históricos indisponível") 
            print(f"   - API mudou e precisa de parâmetros diferentes")
            
        return found_working_ticker
        
    except Exception as e:
        logger.error(f"Erro no teste: {e}")
        return False
    
    finally:
        try:
            if 'connection' in locals():
                connection.disconnect()
        except:
            pass

if __name__ == "__main__":
    print("🧪 Testando tickers alternativos para diagnosticar API...")
    print()
    
    success = test_alternative_tickers()
    
    if success:
        print("\n✅ PROBLEMA IDENTIFICADO: Ticker específico")
        sys.exit(0)
    else:
        print("\n❌ PROBLEMA GERAL: API de dados históricos")
        sys.exit(1)
