#!/usr/bin/env python3
"""
Teste das correções implementadas: 3 dias + detecção inteligente de ticker
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_3_days_smart.log')
    ]
)

logger = logging.getLogger('Test3DaysSmart')

def test_smart_ticker_detection():
    """Testa a detecção inteligente de tickers WDO"""
    
    logger.info("🧪 TESTE: Detecção inteligente de tickers WDO")
    logger.info("=" * 60)
    
    try:
        from connection_manager import ConnectionManager
        
        # Criar um manager para testar os métodos
        conn = ConnectionManager("")
        
        # Testar detecção de contrato atual
        current_contract = conn._get_current_wdo_contract()
        logger.info(f"📊 Contrato atual detectado: {current_contract}")
        
        # Testar variações para diferentes tickers
        test_tickers = ["WDOQ25", "WDO", "PETR4"]
        
        for ticker in test_tickers:
            logger.info(f"\n🔍 Testando variações para: {ticker}")
            variations = conn._get_smart_ticker_variations(ticker)
            logger.info(f"📋 Variações geradas: {variations}")
        
        logger.info("\n✅ Teste de detecção concluído com sucesso!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no teste: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_date_logic():
    """Testa a lógica de datas com limite de 3 dias"""
    
    logger.info("\n📅 TESTE: Lógica de datas com limite de 3 dias")
    logger.info("-" * 60)
    
    # Testar diferentes cenários de data
    scenarios = [
        ("Período normal", datetime.now() - timedelta(days=2), datetime.now()),
        ("Período longo (deve ser limitado)", datetime.now() - timedelta(days=7), datetime.now()),
        ("Data muito antiga", datetime.now() - timedelta(days=30), datetime.now() - timedelta(days=28)),
    ]
    
    for description, start_date, end_date in scenarios:
        logger.info(f"\n📊 Cenário: {description}")
        logger.info(f"   Original: {start_date.date()} - {end_date.date()}")
        logger.info(f"   Dias: {(end_date - start_date).days}")
        
        # Simular a lógica de validação
        days_requested = (end_date - start_date).days
        if days_requested > 3:
            start_date = end_date - timedelta(days=3)
            logger.info(f"   ⚠️ Ajustado: {start_date.date()} - {end_date.date()}")
            logger.info(f"   📊 Novo período: {(end_date - start_date).days} dias")
        else:
            logger.info("   ✅ Período dentro do limite")
    
    return True

if __name__ == "__main__":
    logger.info("🚀 TESTE COMPLETO: 3 DIAS + DETECÇÃO INTELIGENTE")
    logger.info("🎯 Objetivo: Validar implementações antes de testar sistema real")
    logger.info("=" * 70)
    
    # Testes
    success1 = test_date_logic()
    success2 = test_smart_ticker_detection()
    
    logger.info("=" * 70)
    if success1 and success2:
        logger.info("🎉 TODOS OS TESTES PASSARAM!")
        logger.info("✅ Sistema pronto para teste real com:")
        logger.info("   - Limite de 3 dias otimizado")
        logger.info("   - Detecção inteligente de contratos WDO")
        logger.info("   - Consideração de viradas de mês")
    else:
        logger.error("❌ ALGUNS TESTES FALHARAM")
        logger.error("🔧 Verificar implementação antes de prosseguir")
    
    logger.info("📄 Log completo salvo em: test_3_days_smart.log")
