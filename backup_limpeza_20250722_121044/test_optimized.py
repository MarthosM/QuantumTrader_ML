#!/usr/bin/env python3
"""
Teste Otimizado: Conexão com limite de 9 dias descoberto
Proteção contra loops e timeouts otimizados
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connection_manager import ConnectionManager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_optimized.log')
    ]
)

logger = logging.getLogger('TestOptimized')

def test_9_days_limit():
    """Testa conexão respeitando o limite de 9 dias descoberto"""
    
    logger.info("🚀 TESTE OTIMIZADO: Respeitando limite de 9 dias")
    logger.info("🎯 Com proteções contra loops e timeouts otimizados")
    logger.info("=" * 60)
    
    # Configurações
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    
    # Credenciais (substitua pelas suas)
    key = "SUA_CHAVE_AQUI"
    username = "SEU_USUARIO_AQUI" 
    password = "SUA_SENHA_AQUI"
    
    if not os.path.exists(dll_path):
        logger.error(f"❌ DLL não encontrada: {dll_path}")
        return False
    
    try:
        # Criar connection manager
        conn = ConnectionManager(dll_path)
        
        # Calcular período ótimo: exatos 9 dias atrás até hoje
        end_date = datetime.now()
        start_date = end_date - timedelta(days=9)
        
        # Se hoje é fim de semana, ajustar para última sexta-feira
        if end_date.weekday() >= 5:  # Sábado (5) ou Domingo (6)
            days_back = end_date.weekday() - 4  # Volta para sexta
            end_date = end_date - timedelta(days=days_back)
            start_date = end_date - timedelta(days=9)
        
        logger.info(f"📅 Período otimizado (9 dias exatos):")
        logger.info(f"   Início: {start_date.strftime('%d/%m/%Y %H:%M')}")
        logger.info(f"   Fim: {end_date.strftime('%d/%m/%Y %H:%M')}")
        logger.info(f"   Total: {(end_date - start_date).days} dias")
        logger.info(f"   Dia da semana início: {start_date.strftime('%A')}")
        logger.info(f"   Dia da semana fim: {end_date.strftime('%A')}")
        
        # Contador de dados
        historical_data_received = []
        connection_success = False
        
        def on_historical_data(trade_data):
            """Callback otimizado para dados históricos"""
            if trade_data.get('source') == 'historical':
                historical_data_received.append(trade_data)
                # Log menos frequente para evitar spam
                if len(historical_data_received) % 100 == 1:  # Log no 1º, 101º, 201º, etc.
                    logger.info(f"📊 {len(historical_data_received)} dados históricos...")
        
        # Registrar callback
        conn.register_trade_callback(on_historical_data)
        
        # Inicializar com timeout otimizado
        logger.info("🔌 Inicializando conexão (timeout otimizado)...")
        if not conn.initialize(key, username, password):
            logger.error("❌ Falha na inicialização")
            return False
        
        connection_success = True
        logger.info("✅ Conexão estabelecida!")
        
        # Testar tickers em ordem de prioridade
        priority_tickers = [
            ("WDO", "Dólar futuro genérico - mais provável de funcionar"),
            ("PETR4", "Petrobras ação - muito líquido"),
            ("VALE3", "Vale ação - muito líquido"),
            ("WDOQ25", "Contrato específico - pode estar vencido")
        ]
        
        for ticker, description in priority_tickers:
            logger.info(f"\n🎯 Testando: {ticker}")
            logger.info(f"📝 {description}")
            logger.info("-" * 50)
            
            # Limpar contadores
            conn._historical_data_count = 0
            historical_data_received.clear()
            
            # Fazer solicitação
            start_request = datetime.now()
            result = conn.request_historical_data(ticker, start_date, end_date)
            
            if result >= 0:
                logger.info(f"✅ Solicitação aceita para {ticker}!")
                logger.info(f"📊 Código retornado: {result}")
                
                # Aguardar com timeout otimizado (30s)
                logger.info("⏳ Aguardando dados (máx 30s)...")
                if conn.wait_for_historical_data(30):
                    request_duration = datetime.now() - start_request
                    final_count = len(historical_data_received)
                    
                    logger.info(f"🎉 SUCESSO TOTAL!")
                    logger.info(f"🏆 Ticker funcionando: {ticker}")
                    logger.info(f"📊 Dados recebidos: {final_count}")
                    logger.info(f"⏱️ Tempo total: {request_duration.total_seconds():.1f}s")
                    logger.info(f"📈 Taxa: {final_count/max(1, request_duration.total_seconds()):.1f} dados/s")
                    
                    # Mostrar amostra dos dados
                    if final_count > 0:
                        logger.info("📋 Amostra dos dados recebidos:")
                        for i in range(min(3, len(historical_data_received))):
                            data = historical_data_received[i]
                            logger.info(f"   {i+1}: Preço: {data.get('price')}, Volume: {data.get('volume')}, Hora: {data.get('timestamp')}")
                    
                    # Verificar qualidade dos dados
                    if final_count >= 100:
                        logger.info("✅ Dados suficientes para análise ML!")
                    elif final_count >= 10:
                        logger.info("⚠️ Poucos dados, mas utilizáveis")
                    else:
                        logger.info("❌ Dados insuficientes para análise")
                    
                    conn.disconnect()
                    return True
                    
                else:
                    logger.warning(f"⚠️ Timeout aguardando dados de {ticker}")
            else:
                logger.error(f"❌ Erro {result} para {ticker}")
                if result == -2147483645:
                    logger.error("   Possível causa: ticker inativo ou formato incorreto")
        
        # Se chegou aqui, nenhum ticker funcionou
        logger.error("❌ NENHUM TICKER FUNCIONOU")
        logger.error("🔍 Possíveis soluções:")
        logger.error("   1. Verificar se a conta tem permissão para dados históricos")
        logger.error("   2. Tentar em horário de mercado (9h-18h)")
        logger.error("   3. Contactar suporte da corretora")
        
        if connection_success:
            conn.disconnect()
        return False
        
    except Exception as e:
        logger.error(f"💥 Erro durante o teste: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🧪 TESTE OTIMIZADO COM DESCOBERTAS")
    logger.info("🎯 Limite: 9 dias máximo")
    logger.info("🛡️ Proteção: timeout 30s")
    logger.info("📊 Prioridade: tickers mais líquidos primeiro")
    logger.info("=" * 60)
    
    success = test_9_days_limit()
    
    logger.info("=" * 60)
    if success:
        logger.info("🏆 TESTE CONCLUÍDO COM SUCESSO!")
        logger.info("✅ Sistema funcionando com dados históricos")
        logger.info("📋 Pronto para integração com sistema ML")
    else:
        logger.error("❌ TESTE FALHADO")
        logger.error("🔧 Verificar configurações e permissões")
    
    logger.info("📄 Log completo salvo em: test_optimized.log")
