"""
Teste das Otimizações de Carregamento de Dados
Verifica performance e detecção de gaps temporais
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_integration import DataIntegration
from src.data_loader import DataLoader

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('OptimizedDataTest')

class OptimizedDataTester:
    """Testa as otimizações de carregamento de dados"""
    
    def __init__(self):
        self.logger = logger
        self.data_loader = DataLoader()
        self.data_integration = DataIntegration(None, self.data_loader)
        
    def test_batch_processing(self):
        """Testa processamento em lote de candles"""
        self.logger.info("🚀 Testando processamento em lote...")
        
        start_time = time.time()
        
        # Simular dados históricos (marcar flag para não serem rejeitados)
        self.data_integration._historical_loading_complete = False
        base_time = datetime.now() - timedelta(hours=2)
        
        for i in range(200):  # 200 candles de teste
            trade_data = {
                'timestamp': base_time + timedelta(minutes=i),
                'price': 5500 + (i % 20),
                'volume': 1000 + (i % 500),
                'quantity': 1000 + (i % 500),  # Campo obrigatório
                'buy_volume': 600,
                'sell_volume': 400,
                'is_historical': True  # Marcar como histórico
            }
            
            # Simular processamento
            self.data_integration._on_trade(trade_data)
        
        # Simular fim do carregamento histórico
        completion_event = {'event_type': 'historical_data_complete'}
        self.data_integration._on_trade(completion_event)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verificar resultados
        candles_df = self.data_integration.candles_1min
        
        self.logger.info(f"⏱️ Tempo de processamento: {processing_time:.2f}s")
        self.logger.info(f"📊 Candles criados: {len(candles_df)}")
        self.logger.info(f"🚄 Taxa: {len(candles_df)/processing_time:.1f} candles/s")
        
        if not candles_df.empty:
            self.logger.info(f"✅ Primeiro candle: {candles_df.index[0]}")
            self.logger.info(f"✅ Último candle: {candles_df.index[-1]}")
            
        return len(candles_df) > 0
    
    def test_gap_detection(self):
        """Testa detecção de gaps temporais"""
        self.logger.info("🔍 Testando detecção de gaps...")
        
        # Criar dados com gap intencional
        old_time = datetime.now() - timedelta(minutes=30)  # 30 min atrás
        
        trade_data = {
            'timestamp': old_time,
            'price': 5500,
            'volume': 1000,
            'quantity': 1000,  # Campo obrigatório
            'buy_volume': 600,
            'sell_volume': 400
        }
        
        # Criar candle antigo
        candle_data = {
            'open': 5500,
            'high': 5510,
            'low': 5490,
            'close': 5505,
            'volume': 1000,
            'trades': 10
        }
        
        import pandas as pd
        candle = pd.DataFrame([candle_data], index=[old_time])
        self.data_integration.candles_1min = candle
        
        # Testar detecção de gap
        self.data_integration._check_and_fix_temporal_gap()
        
        return True
    
    def run_performance_test(self):
        """Executa todos os testes de performance"""
        self.logger.info("=" * 80)
        self.logger.info("🧪 INICIANDO TESTES DE PERFORMANCE OTIMIZADA")
        self.logger.info("=" * 80)
        
        results = {}
        
        # Teste 1: Processamento em lote
        self.logger.info("\n📦 TESTE 1: Processamento em Lote")
        results['batch_processing'] = self.test_batch_processing()
        
        # Teste 2: Detecção de gaps
        self.logger.info("\n🕳️ TESTE 2: Detecção de Gaps")
        results['gap_detection'] = self.test_gap_detection()
        
        # Resumo
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 RESUMO DOS TESTES")
        self.logger.info("=" * 80)
        
        success_count = sum(1 for result in results.values() if result)
        total_count = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSOU" if result else "❌ FALHOU"
            self.logger.info(f"   {test_name}: {status}")
        
        if success_count == total_count:
            self.logger.info(f"\n🎉 TODOS OS TESTES PASSARAM! ({success_count}/{total_count})")
        else:
            self.logger.info(f"\n⚠️ ALGUNS TESTES FALHARAM ({success_count}/{total_count})")
        
        return success_count == total_count


def main():
    """Função principal"""
    tester = OptimizedDataTester()
    
    try:
        success = tester.run_performance_test()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Erro durante teste: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)