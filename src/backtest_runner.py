# src/backtesting/backtest_runner.py
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging
import os
from enum import Enum
from report_generator import BacktestReportGenerator
from ml_backtester import AdvancedMLBacktester, BacktestConfig, BacktestMode
from stress_test_engine import StressTestEngine

class BacktestRunner:
    """Runner para executar backtests com o sistema ML"""
    
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.logger = logging.getLogger(__name__)
        
    def run_ml_backtest(self, start_date: datetime, end_date: datetime,
                       config: Optional[Dict] = None) -> Dict:
        """
        Executa backtest do sistema ML
        
        Args:
            start_date: Data inicial
            end_date: Data final
            config: Configurações customizadas
            
        Returns:
            Resultados completos do backtest
        """
        self.logger.info(f"Iniciando backtest ML de {start_date} até {end_date}")
        
        # Configuração padrão
        if config is None:
            config = {
                'initial_capital': 100000.0,
                'commission_per_contract': 0.50,
                'slippage_ticks': 1,
                'mode': BacktestMode.REALISTIC
            }
        
        # Separar configurações específicas do BacktestConfig
        backtest_params = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': config.get('initial_capital', 100000.0),
            'commission_per_contract': config.get('commission_per_contract', 0.50),
            'slippage_ticks': config.get('slippage_ticks', 1),
            'mode': config.get('mode', BacktestMode.REALISTIC)
        }
        
        # Configurações adicionais não passadas para BacktestConfig
        run_stress_tests = config.get('run_stress_tests', False)
        
        # Criar configuração de backtest
        backtest_config = BacktestConfig(**backtest_params)
        
        # Criar backtester
        backtester = AdvancedMLBacktester(backtest_config)
        
        # Obter modelos e feature engine do sistema
        models = self.trading_system.model_manager.models  # Usar atributo direto
        feature_engine = self.trading_system.feature_engine
        
        # Inicializar backtester
        backtester.initialize(models, feature_engine)
        
        # Carregar dados históricos
        historical_data = self._load_historical_data(start_date, end_date)
        
        # Executar backtest
        results = backtester.run_backtest(historical_data)
        
        # Executar stress tests
        if run_stress_tests:
            stress_engine = StressTestEngine(backtester)
            stress_results = stress_engine.run_stress_tests(historical_data)
            results['stress_tests'] = stress_results
        
        # Gerar relatório
        report_generator = BacktestReportGenerator()
        report_path = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_generator.generate_comprehensive_report(results, report_path)
        
        self.logger.info(f"Backtest concluído. Relatório salvo em: {report_path}")
        
        return results
    
    def _load_historical_data(self, start_date: datetime, 
                            end_date: datetime) -> pd.DataFrame:
        """Carrega dados históricos para backtest"""
        self.logger.info("Carregando dados históricos para backtest...")
        
        # Primeiro, tentar carregar dados reais de WDO
        csv_path = r"C:\Users\marth\OneDrive\Programacao\Python\Projetos\ML_Tradingv2.0\src\training\data\historical\wdo_data_20_06_2025.csv"
        
        try:
            if os.path.exists(csv_path):
                self.logger.info(f"Carregando dados reais de WDO de: {csv_path}")
                
                # Carregar CSV com dados reais
                df = pd.read_csv(csv_path, parse_dates=['Date'])
                df.set_index('Date', inplace=True)
                
                # Remover colunas problemáticas que causam erro de conversão
                columns_to_remove = ['contract', 'preco']  # 'WDON25' causa erro de float
                for col in columns_to_remove:
                    if col in df.columns:
                        df = df.drop(columns=[col])
                
                # Manter apenas colunas numéricas essenciais
                essential_cols = ['open', 'high', 'low', 'close', 'volume']
                optional_cols = ['buy_volume', 'sell_volume', 'quantidade', 'trades']
                
                # Adicionar colunas opcionais se existirem
                for col in optional_cols:
                    if col in df.columns:
                        essential_cols.append(col)
                
                # Selecionar apenas colunas válidas
                valid_cols = [col for col in essential_cols if col in df.columns]
                df = df[valid_cols]
                
                # Converter para numérico e limpar
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna()
                
                # Filtrar pelo período solicitado
                mask = (df.index >= start_date) & (df.index <= end_date)
                df_filtered = df[mask]
                
                if not df_filtered.empty:
                    self.logger.info(f"✅ Carregados {len(df_filtered)} registros reais de WDO")
                    self.logger.info(f"📅 Período: {df_filtered.index.min()} até {df_filtered.index.max()}")
                    self.logger.info(f"💰 Preço médio: R$ {df_filtered['close'].mean():.2f}")
                    
                    # Renomear colunas se necessário para padronização
                    if 'quantidade' in df_filtered.columns:
                        df_filtered['trades'] = df_filtered['quantidade']
                    
                    return df_filtered
                else:
                    self.logger.warning("Período solicitado não encontrado nos dados reais")
            
        except Exception as e:
            self.logger.warning(f"Erro carregando dados reais: {e}")
        
        # Fallback para TrainingDataLoader
        try:
            from training.data_loader import TrainingDataLoader
            
            data_loader = TrainingDataLoader(data_path="src/training/data/historical/")
            
            # Carregar dados para WDO (símbolo padrão do sistema)
            symbols = ['WDO']
            
            historical_data = data_loader.load_historical_data(
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
                validate_realtime=False  # Para backtest, não precisamos validação rigorosa
            )
            
            if not historical_data.empty:
                self.logger.info(f"Carregados {len(historical_data)} registros via TrainingDataLoader")
                return historical_data
            
        except Exception as e:
            self.logger.warning(f"Erro carregando via TrainingDataLoader: {e}")
        
        # Último fallback - DataLoader padrão ou dados demo
        return self._load_data_fallback(start_date, end_date)
    
    def _load_data_fallback(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Método fallback para carregar dados históricos"""
        try:
            # Usar DataLoader padrão
            data_loader = self.trading_system.data_loader
            
            # Carregar candles históricos
            candles_data = data_loader.load_candles(
                start_date=start_date,
                end_date=end_date,
                interval="1m",
                symbol="WDO"
            )
            
            if not candles_data.empty:
                self.logger.info(f"Carregados {len(candles_data)} candles via DataLoader")
                return candles_data
            
            # Se ainda não tiver dados, gerar dados simulados para demo
            self.logger.warning("Gerando dados simulados para demo do backtest")
            return self._generate_demo_data(start_date, end_date)
            
        except Exception as e:
            self.logger.error(f"Erro no fallback de dados: {e}")
            return self._generate_demo_data(start_date, end_date)
    
    def _generate_demo_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Gera dados demo para teste do backtest"""
        import numpy as np
        
        # Gerar timestamps de 1 em 1 minuto
        timestamps = pd.date_range(start=start_date, end=end_date, freq='1T')
        
        # Preços realísticos para WDO
        base_price = 127000  # Preço base WDO
        n_points = len(timestamps)
        
        # Gerar série de preços com walk
        price_changes = np.random.normal(0, 50, n_points)  # Variação de 50 pontos
        prices = base_price + np.cumsum(price_changes)
        
        # Criar OHLCV
        data = []
        for i, (ts, close) in enumerate(zip(timestamps, prices)):
            open_price = prices[i-1] if i > 0 else base_price
            high = max(open_price, close) + abs(np.random.normal(0, 20))
            low = min(open_price, close) - abs(np.random.normal(0, 20))
            volume = np.random.randint(100, 1000)
            
            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high,
                'low': low, 
                'close': close,
                'volume': volume,
                'trades': np.random.randint(5, 50)
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        self.logger.info(f"Dados demo gerados: {len(df)} candles de {start_date} até {end_date}")
        return df

# Exemplo de uso
def run_backtest_example(trading_system):
    """Exemplo de como executar backtest"""
    
    runner = BacktestRunner(trading_system)
    
    # Configuração do backtest
    config = {
        'initial_capital': 100000.0,
        'commission_per_contract': 0.50,
        'slippage_ticks': 1,
        'mode': BacktestMode.REALISTIC,
        'run_stress_tests': True
    }
    
    # Executar backtest
    results = runner.run_ml_backtest(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 1),
        config=config
    )
    
    # Exibir resumo
    print(f"Capital Final: R$ {results['metrics']['final_equity']:,.2f}")
    print(f"Retorno Total: {results['metrics']['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['metrics']['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {results['metrics']['win_rate']*100:.1f}%")

def main():
    """Função principal para executar backtest quando rodado diretamente"""
    import sys
    from datetime import datetime, timedelta
    
    print("=== BACKTEST RUNNER - SISTEMA ML TRADING ===")
    
    try:
        # Setup do sistema de trading mock
        from data_loader import DataLoader
        from model_manager import ModelManager
        from feature_engine import FeatureEngine
        
        class MockTradingSystem:
            def __init__(self):
                self.data_loader = DataLoader(data_dir="../data/")
                # Usar diretório real dos modelos treinados
                models_path = r"C:\Users\marth\OneDrive\Programacao\Python\Projetos\ML_Tradingv2.0\src\training\models\training_20250720_184206\ensemble\ensemble_20250720_184206"
                self.model_manager = ModelManager(models_path)
                self.feature_engine = FeatureEngine()
        
        print("Configurando sistema de trading...")
        trading_system = MockTradingSystem()
        
        # Carregar modelos reais
        print("Carregando modelos ML...")
        success = trading_system.model_manager.load_models()
        if success:
            print(f"✅ Modelos carregados: {list(trading_system.model_manager.models.keys())}")
            print(f"Total de features disponíveis: {len(trading_system.model_manager.get_all_required_features())}")
        else:
            print("❌ Falha ao carregar modelos - continuando sem ML")
        
        # Criar runner
        runner = BacktestRunner(trading_system)
        
        # Configurar período - usar dados reais disponíveis
        # Com base no teste anterior: dados de 03/02/2025 até 20/06/2025
        start_date = datetime(2025, 6, 13)  # Últimos 7 dias dos dados
        end_date = datetime(2025, 6, 20)
        
        print(f"Período do backtest: {start_date.date()} até {end_date.date()}")
        
        # Primeiro testar apenas carregamento de dados
        print("Testando carregamento de dados históricos...")
        historical_data = runner._load_historical_data(start_date, end_date)
        
        if historical_data.empty:
            print("❌ Nenhum dado histórico carregado - não é possível executar backtest")
            return 1
        
        print(f"✅ Dados carregados: {len(historical_data)} registros")
        print(f"Período: {historical_data.index.min()} até {historical_data.index.max()}")
        print(f"Preço médio: R$ {historical_data['close'].mean():.2f}")
        
        # Configuração do backtest
        config = {
            'initial_capital': 100000.0,
            'commission_per_contract': 0.50,
            'slippage_ticks': 1,
            'mode': BacktestMode.REALISTIC,
            'run_stress_tests': False  # Desabilitar por enquanto
        }
        
        print("Executando backtest...")
        
        # Executar backtest
        results = runner.run_ml_backtest(
            start_date=start_date,
            end_date=end_date,
            config=config
        )
        
        print("\n=== RESULTADOS DO BACKTEST ===")
        if 'metrics' in results:
            metrics = results['metrics']
            print(f"Capital Final: R$ {metrics.get('final_equity', 0):,.2f}")
            print(f"Retorno Total: {metrics.get('total_return', 0)*100:.2f}%")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
        else:
            print("Estrutura de resultados:")
            for key in results.keys():
                print(f"  - {key}: {type(results[key])}")
            
            # Mostrar valores específicos se disponíveis
            if 'total_trades' in results:
                print(f"\n📊 MÉTRICAS BÁSICAS:")
                print(f"Total de Trades: {results.get('total_trades', 0)}")
                print(f"Trades Vencedoras: {results.get('winning_trades', 0)}")
                print(f"Trades Perdedoras: {results.get('losing_trades', 0)}")
                print(f"Win Rate: {results.get('win_rate', 0)*100:.1f}%")
                print(f"PnL Total: R$ {results.get('total_pnl', 0):,.2f}")
                print(f"Capital Final: R$ {results.get('final_equity', 100000):,.2f}")
        
        print("\n✅ Backtest executado com sucesso!")
        return 0
        
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())