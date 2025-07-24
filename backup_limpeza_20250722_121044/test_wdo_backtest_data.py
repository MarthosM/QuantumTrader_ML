#!/usr/bin/env python3
"""
Teste específico do BacktestRunner com dados reais de WDO
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Adicionar src ao path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Configurar logging sem emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_wdo_data_loading():
    """Testa carregamento dos dados reais de WDO"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=== TESTE DE CARREGAMENTO DE DADOS REAIS WDO ===")
        
        # Verificar se arquivo existe
        csv_path = r"C:\Users\marth\OneDrive\Programacao\Python\Projetos\ML_Tradingv2.0\src\training\data\historical\wdo_data_20_06_2025.csv"
        
        if not os.path.exists(csv_path):
            logger.error(f"Arquivo não encontrado: {csv_path}")
            return False
        
        logger.info(f"[OK] Arquivo encontrado: {csv_path}")
        
        # Carregar dados
        import pandas as pd
        
        df = pd.read_csv(csv_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        
        logger.info(f"[OK] Dados carregados com sucesso!")
        logger.info(f"   - Total de registros: {len(df):,}")
        logger.info(f"   - Período: {df.index.min()} até {df.index.max()}")
        logger.info(f"   - Colunas: {list(df.columns)}")
        logger.info(f"   - Preço médio: R$ {df['close'].mean():.2f}")
        logger.info(f"   - Variação: R$ {df['close'].min():.2f} - R$ {df['close'].max():.2f}")
        logger.info(f"   - Volume médio: {df['volume'].mean():,.0f}")
        
        # Testar período específico (últimos 7 dias dos dados)
        end_period = df.index.max()
        start_period = end_period - timedelta(days=7)
        
        df_period = df[(df.index >= start_period) & (df.index <= end_period)]
        logger.info(f"[OK] Teste período 7 dias:")
        logger.info(f"   - Registros: {len(df_period):,}")
        logger.info(f"   - Período: {df_period.index.min()} até {df_period.index.max()}")
        
        # Agora testar o BacktestRunner
        logger.info("\n=== TESTE DO BACKTEST RUNNER ===")
        
        from data_loader import DataLoader
        from model_manager import ModelManager
        from feature_engine import FeatureEngine
        from backtest_runner import BacktestRunner
        
        # Mock do sistema de trading
        class MockTradingSystem:
            def __init__(self):
                self.data_loader = DataLoader(data_dir="data/")
                self.model_manager = ModelManager(str(Path("models/trained/")))
                self.feature_engine = FeatureEngine()
        
        trading_system = MockTradingSystem()
        runner = BacktestRunner(trading_system)
        
        # Testar carregamento via runner
        logger.info("Testando carregamento via BacktestRunner...")
        
        test_data = runner._load_historical_data(start_period, end_period)
        
        if not test_data.empty:
            logger.info("[OK] BacktestRunner carregou dados com sucesso!")
            logger.info(f"   - Registros carregados: {len(test_data):,}")
            logger.info(f"   - Período: {test_data.index.min()} até {test_data.index.max()}")
            logger.info(f"   - Colunas disponíveis: {list(test_data.columns)}")
            
            # Verificar se tem as colunas necessárias para backtest
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in test_data.columns]
            
            if missing_cols:
                logger.warning(f"Colunas faltando para backtest: {missing_cols}")
            else:
                logger.info("[OK] Todas as colunas necessárias estão disponíveis!")
                
                # Mostrar estatísticas básicas
                logger.info("Estatísticas dos dados:")
                logger.info(f"   - Preço médio: R$ {test_data['close'].mean():.2f}")
                logger.info(f"   - Volatilidade: {test_data['close'].std():.2f}")
                logger.info(f"   - Volume médio: {test_data['volume'].mean():,.0f}")
                
                # Verificar consistência OHLC
                invalid_candles = (
                    (test_data['high'] < test_data['low']) |
                    (test_data['high'] < test_data['open']) |
                    (test_data['high'] < test_data['close']) |
                    (test_data['low'] > test_data['open']) |
                    (test_data['low'] > test_data['close'])
                ).sum()
                
                if invalid_candles > 0:
                    logger.warning(f"Candles inconsistentes encontrados: {invalid_candles}")
                else:
                    logger.info("[OK] Dados OHLC consistentes!")
            
            return True
        else:
            logger.error("[X] BacktestRunner não conseguiu carregar dados")
            return False
    
    except Exception as e:
        logger.error(f"[X] Erro durante teste: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Executa teste completo"""
    logger = logging.getLogger(__name__)
    
    logger.info("Iniciando teste de dados WDO para backtest...")
    
    success = test_wdo_data_loading()
    
    if success:
        logger.info("\n" + "="*60)
        logger.info("✅ TESTE CONCLUÍDO COM SUCESSO!")
        logger.info("Os dados WDO estão prontos para uso no backtest.")
        logger.info("="*60)
        return 0
    else:
        logger.error("\n" + "="*60)
        logger.error("❌ TESTE FALHOU!")
        logger.error("Verifique os problemas relatados acima.")
        logger.error("="*60)
        return 1

if __name__ == "__main__":
    exit(main())
