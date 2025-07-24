#!/usr/bin/env python3
"""
Script para executar backtest do sistema ML Trading
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Adicionar src ao path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backtest_execution.log')
    ]
)

def setup_mock_trading_system():
    """Cria um sistema de trading mock para demonstração"""
    from data_loader import DataLoader
    from model_manager import ModelManager
    from feature_engine import FeatureEngine
    
    # Mock do sistema de trading
    class MockTradingSystem:
        def __init__(self):
            self.data_loader = DataLoader(data_dir="data/")
            
            # Tentar carregar model manager se disponível
            try:
                models_dir = Path("models/trained/")
                self.model_manager = ModelManager(str(models_dir))
                if models_dir.exists():
                    self.model_manager.load_models()
            except Exception as e:
                logging.warning(f"Model manager não disponível: {e}")
                self.model_manager = None
            
            # Tentar carregar feature engine
            try:
                self.feature_engine = FeatureEngine()
            except Exception as e:
                logging.warning(f"Feature engine não disponível: {e}")
                self.feature_engine = None
    
    return MockTradingSystem()

def main():
    """Executa exemplo de backtest"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(">> Iniciando execução de backtest ML Trading")
        
        # Configurar período do backtest
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 30 dias de dados
        
        logger.info(f"Período do backtest: {start_date.date()} até {end_date.date()}")
        
        # Setup do sistema de trading
        logger.info(">> Configurando sistema de trading...")
        trading_system = setup_mock_trading_system()
        
        # Verificar dependências
        dependencies_ok = True
        
        # Verificar se existem módulos necessários
        required_modules = [
            'ml_backtester',
            'report_generator', 
            'stress_test_engine'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
                logger.info(f"[OK] {module} disponível")
            except ImportError:
                missing_modules.append(module)
                logger.warning(f"[X] {module} não encontrado")
                dependencies_ok = False
        
        # Se dependências estão ok, executar backtest
        if dependencies_ok:
            from backtest_runner import BacktestRunner, BacktestMode
            
            # Criar runner de backtest
            runner = BacktestRunner(trading_system)
            
            # Configuração do backtest
            config = {
                'initial_capital': 100000.0,
                'commission_per_contract': 0.50,
                'slippage_ticks': 1,
                'mode': BacktestMode.REALISTIC,
                'run_stress_tests': False  # Desabilitar por enquanto
            }
            
            logger.info(">> Executando backtest...")
            results = runner.run_ml_backtest(
                start_date=start_date,
                end_date=end_date,
                config=config
            )
            
            # Mostrar resultados
            logger.info("=" * 60)
            logger.info(">> RESULTADOS DO BACKTEST")
            logger.info("=" * 60)
            
            if 'metrics' in results:
                metrics = results['metrics']
                logger.info(f"Capital Final: R$ {metrics.get('final_equity', 0):,.2f}")
                logger.info(f"Retorno Total: {metrics.get('total_return', 0)*100:.2f}%")
                logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
                logger.info(f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            else:
                logger.info("Resultados disponíveis:")
                for key, value in results.items():
                    logger.info(f"  {key}: {value}")
            
        else:
            logger.error("[X] Dependências faltando para backtest completo:")
            for module in missing_modules:
                logger.error(f"   - {module}")
            
            logger.info(">> Executando teste básico de carregamento de dados...")
            
            # Teste básico de carregamento de dados
            from backtest_runner import BacktestRunner
            runner = BacktestRunner(trading_system)
            
            # Testar apenas carregamento de dados
            test_data = runner._load_historical_data(start_date, end_date)
            
            if not test_data.empty:
                logger.info("[OK] Teste de dados bem-sucedido:")
                logger.info(f"   Registros: {len(test_data)} carregados")
                logger.info(f"   Período: {test_data.index.min()} até {test_data.index.max()}")
                logger.info(f"   Preço médio: R$ {test_data['close'].mean():.2f}")
                logger.info(f"   Variação: R$ {test_data['close'].min():.2f} - R$ {test_data['close'].max():.2f}")
            else:
                logger.error("[X] Nenhum dado histórico carregado")
        
        logger.info("[OK] Execução concluída!")
        
    except Exception as e:
        logger.error(f"[X] Erro durante execução: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
