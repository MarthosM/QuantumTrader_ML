"""
Sistema de Comparação com Backtest
Compara resultados do sistema ao vivo vs backtest histórico
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BacktestComparison')

# Adicionar paths
sys.path.insert(0, os.path.dirname(__file__))

from enhanced_production_system import EnhancedProductionSystem
from src.features.book_features_rt import BookFeatureEngineerRT
from src.data.book_data_manager import BookDataManager


class BacktestComparison:
    """Sistema de comparação entre produção e backtest"""
    
    def __init__(self):
        self.system = EnhancedProductionSystem()
        self.data_path = Path('data/')
        self.results = {
            'live': {
                'features': [],
                'predictions': [],
                'signals': [],
                'latencies': []
            },
            'backtest': {
                'features': [],
                'predictions': [],
                'signals': [],
                'latencies': []
            },
            'comparison': {
                'feature_correlation': 0,
                'prediction_correlation': 0,
                'signal_match_rate': 0,
                'differences': []
            }
        }
    
    def load_test_data(self) -> pd.DataFrame:
        """Carrega dados para teste"""
        logger.info("\nCarregando dados de teste...")
        
        # Tentar carregar CSV
        csv_path = self.data_path / 'csv_data' / 'wdo_5m_data.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = df.tail(100)  # Últimos 100 candles
            logger.info(f"  Carregados {len(df)} candles do CSV")
            return df
        
        # Simular dados se não existir
        logger.warning("  CSV não encontrado. Simulando dados...")
        
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=8),
            periods=100,
            freq='5min'
        )
        
        prices = [5450.0]
        for _ in range(99):
            ret = np.random.normal(0, 0.002)
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
            'volume': [100000 + np.random.randint(-20000, 20000) for _ in range(100)]
        })
        
        return df
    
    def run_live_system(self, data: pd.DataFrame):
        """Executa sistema ao vivo (simulado)"""
        logger.info("\n1. Executando Sistema AO VIVO...")
        
        for idx, row in data.iterrows():
            # Processar candle
            candle = {
                'timestamp': pd.to_datetime(row['timestamp']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
            
            self.system.feature_engineer._update_candle(candle)
            
            # Calcular features a cada 5 candles
            if idx % 5 == 0 and idx > 0:
                import time
                start = time.perf_counter()
                
                features = self.system._calculate_features()
                
                if len(features) == 65:
                    # Armazenar features
                    feature_vector = list(features.values())
                    self.results['live']['features'].append(feature_vector)
                    
                    # Fazer predição
                    prediction = self.system._make_ml_prediction(features)
                    self.results['live']['predictions'].append(prediction)
                    
                    # Gerar sinal
                    signal = self._generate_signal(prediction)
                    self.results['live']['signals'].append(signal)
                    
                    # Registrar latência
                    latency = (time.perf_counter() - start) * 1000
                    self.results['live']['latencies'].append(latency)
        
        logger.info(f"  Processados {len(data)} candles")
        logger.info(f"  Features calculadas: {len(self.results['live']['features'])}")
        logger.info(f"  Predições: {len(self.results['live']['predictions'])}")
        logger.info(f"  Latência média: {np.mean(self.results['live']['latencies']):.2f}ms")
    
    def run_backtest(self, data: pd.DataFrame):
        """Executa backtest tradicional"""
        logger.info("\n2. Executando BACKTEST...")
        
        # Criar novo sistema para backtest
        backtest_system = EnhancedProductionSystem()
        
        # Processar todos os dados de uma vez (simulando backtest batch)
        all_candles = []
        for _, row in data.iterrows():
            candle = {
                'timestamp': pd.to_datetime(row['timestamp']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
            all_candles.append(candle)
            backtest_system.feature_engineer._update_candle(candle)
        
        # Calcular features em batch (típico de backtest)
        for i in range(0, len(all_candles), 5):
            if i > 0:
                import time
                start = time.perf_counter()
                
                features = backtest_system._calculate_features()
                
                if len(features) == 65:
                    # Armazenar features
                    feature_vector = list(features.values())
                    self.results['backtest']['features'].append(feature_vector)
                    
                    # Fazer predição
                    prediction = backtest_system._make_ml_prediction(features)
                    self.results['backtest']['predictions'].append(prediction)
                    
                    # Gerar sinal
                    signal = self._generate_signal(prediction)
                    self.results['backtest']['signals'].append(signal)
                    
                    # Registrar latência
                    latency = (time.perf_counter() - start) * 1000
                    self.results['backtest']['latencies'].append(latency)
        
        logger.info(f"  Processados {len(data)} candles")
        logger.info(f"  Features calculadas: {len(self.results['backtest']['features'])}")
        logger.info(f"  Predições: {len(self.results['backtest']['predictions'])}")
        logger.info(f"  Latência média: {np.mean(self.results['backtest']['latencies']):.2f}ms")
    
    def _generate_signal(self, prediction: float) -> str:
        """Gera sinal de trading baseado na predição"""
        if prediction > 0.6:
            return 'BUY'
        elif prediction < 0.4:
            return 'SELL'
        else:
            return 'HOLD'
    
    def compare_results(self):
        """Compara resultados entre live e backtest"""
        logger.info("\n3. Comparando Resultados...")
        
        # 1. Comparar features
        if (len(self.results['live']['features']) > 0 and 
            len(self.results['backtest']['features']) > 0):
            
            # Pegar menor tamanho para comparação
            min_len = min(
                len(self.results['live']['features']),
                len(self.results['backtest']['features'])
            )
            
            live_features = np.array(self.results['live']['features'][:min_len])
            backtest_features = np.array(self.results['backtest']['features'][:min_len])
            
            # Calcular correlação média entre features
            correlations = []
            for i in range(min(65, live_features.shape[1])):
                if np.std(live_features[:, i]) > 0 and np.std(backtest_features[:, i]) > 0:
                    corr = np.corrcoef(live_features[:, i], backtest_features[:, i])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if correlations:
                self.results['comparison']['feature_correlation'] = np.mean(correlations)
                logger.info(f"  Correlação média de features: {self.results['comparison']['feature_correlation']:.3f}")
        
        # 2. Comparar predições
        if (len(self.results['live']['predictions']) > 0 and 
            len(self.results['backtest']['predictions']) > 0):
            
            min_len = min(
                len(self.results['live']['predictions']),
                len(self.results['backtest']['predictions'])
            )
            
            live_preds = self.results['live']['predictions'][:min_len]
            backtest_preds = self.results['backtest']['predictions'][:min_len]
            
            # Correlação de predições
            if np.std(live_preds) > 0 and np.std(backtest_preds) > 0:
                pred_corr = np.corrcoef(live_preds, backtest_preds)[0, 1]
                self.results['comparison']['prediction_correlation'] = pred_corr
                logger.info(f"  Correlação de predições: {pred_corr:.3f}")
        
        # 3. Comparar sinais
        if (len(self.results['live']['signals']) > 0 and 
            len(self.results['backtest']['signals']) > 0):
            
            min_len = min(
                len(self.results['live']['signals']),
                len(self.results['backtest']['signals'])
            )
            
            matches = sum(
                1 for i in range(min_len)
                if self.results['live']['signals'][i] == self.results['backtest']['signals'][i]
            )
            
            self.results['comparison']['signal_match_rate'] = matches / min_len
            logger.info(f"  Taxa de sinais idênticos: {self.results['comparison']['signal_match_rate']:.1%}")
        
        # 4. Comparar latências
        logger.info(f"\n  Latências:")
        logger.info(f"    Live: {np.mean(self.results['live']['latencies']):.2f}ms (média)")
        logger.info(f"    Backtest: {np.mean(self.results['backtest']['latencies']):.2f}ms (média)")
        
        # 5. Identificar principais diferenças
        self._identify_differences()
    
    def _identify_differences(self):
        """Identifica principais diferenças entre sistemas"""
        differences = []
        
        # Diferença no número de cálculos
        diff_calculations = abs(
            len(self.results['live']['features']) - 
            len(self.results['backtest']['features'])
        )
        if diff_calculations > 0:
            differences.append(f"Diferença de {diff_calculations} cálculos de features")
        
        # Diferença nas latências
        if self.results['live']['latencies'] and self.results['backtest']['latencies']:
            lat_diff = abs(
                np.mean(self.results['live']['latencies']) - 
                np.mean(self.results['backtest']['latencies'])
            )
            if lat_diff > 1.0:
                differences.append(f"Diferença de latência: {lat_diff:.2f}ms")
        
        # Diferença na distribuição de sinais
        if self.results['live']['signals'] and self.results['backtest']['signals']:
            live_buy_rate = self.results['live']['signals'].count('BUY') / len(self.results['live']['signals'])
            backtest_buy_rate = self.results['backtest']['signals'].count('BUY') / len(self.results['backtest']['signals'])
            
            if abs(live_buy_rate - backtest_buy_rate) > 0.1:
                differences.append(f"Taxa de BUY: Live={live_buy_rate:.1%} vs Backtest={backtest_buy_rate:.1%}")
        
        self.results['comparison']['differences'] = differences
        
        if differences:
            logger.info(f"\n  Principais diferenças encontradas:")
            for diff in differences:
                logger.info(f"    - {diff}")
    
    def validate_consistency(self) -> bool:
        """Valida consistência entre sistemas"""
        logger.info("\n" + "=" * 60)
        logger.info("VALIDAÇÃO DE CONSISTÊNCIA")
        logger.info("=" * 60)
        
        validations = []
        
        # 1. Correlação de features
        feat_corr = self.results['comparison']['feature_correlation']
        if feat_corr > 0.9:
            logger.info(f"[OK] Features altamente correlacionadas: {feat_corr:.3f}")
            validations.append(True)
        elif feat_corr > 0.7:
            logger.warning(f"[AVISO] Features moderadamente correlacionadas: {feat_corr:.3f}")
            validations.append(True)
        else:
            logger.error(f"[ERRO] Features com baixa correlação: {feat_corr:.3f}")
            validations.append(False)
        
        # 2. Correlação de predições
        pred_corr = self.results['comparison']['prediction_correlation']
        if pred_corr > 0.8:
            logger.info(f"[OK] Predições consistentes: {pred_corr:.3f}")
            validations.append(True)
        elif pred_corr > 0.6:
            logger.warning(f"[AVISO] Predições parcialmente consistentes: {pred_corr:.3f}")
            validations.append(True)
        else:
            logger.error(f"[ERRO] Predições inconsistentes: {pred_corr:.3f}")
            validations.append(False)
        
        # 3. Taxa de sinais idênticos
        signal_match = self.results['comparison']['signal_match_rate']
        if signal_match > 0.8:
            logger.info(f"[OK] Sinais consistentes: {signal_match:.1%}")
            validations.append(True)
        elif signal_match > 0.6:
            logger.warning(f"[AVISO] Sinais parcialmente consistentes: {signal_match:.1%}")
            validations.append(True)
        else:
            logger.error(f"[ERRO] Sinais inconsistentes: {signal_match:.1%}")
            validations.append(False)
        
        # Resultado final
        success = all(validations) if validations else False
        
        logger.info("\n" + "=" * 60)
        if success:
            logger.info("[SUCESSO] Sistemas são consistentes!")
            logger.info("O sistema ao vivo está operando conforme esperado.")
        else:
            logger.error("[FALHOU] Inconsistências detectadas entre sistemas")
            logger.error("Revisar implementação antes de produção.")
        logger.info("=" * 60)
        
        # Salvar relatório
        self._save_report()
        
        return success
    
    def _save_report(self):
        """Salva relatório de comparação"""
        report_path = Path('test_results/backtest_comparison.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Converter arrays numpy para listas
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        # Preparar relatório
        report = {
            'timestamp': datetime.now().isoformat(),
            'live_stats': {
                'total_features': len(self.results['live']['features']),
                'total_predictions': len(self.results['live']['predictions']),
                'avg_latency': np.mean(self.results['live']['latencies']) if self.results['live']['latencies'] else 0
            },
            'backtest_stats': {
                'total_features': len(self.results['backtest']['features']),
                'total_predictions': len(self.results['backtest']['predictions']),
                'avg_latency': np.mean(self.results['backtest']['latencies']) if self.results['backtest']['latencies'] else 0
            },
            'comparison': self.results['comparison']
        }
        
        # Converter valores numpy
        report = json.loads(json.dumps(report, default=convert_to_serializable))
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nRelatório salvo em: {report_path}")


def main():
    """Executa comparação entre live e backtest"""
    logger.info("\n" + "=" * 70)
    logger.info(" COMPARAÇÃO SISTEMA AO VIVO VS BACKTEST")
    logger.info("=" * 70)
    
    # Criar sistema de comparação
    comparison = BacktestComparison()
    
    # Carregar dados de teste
    test_data = comparison.load_test_data()
    
    # Executar sistema ao vivo
    comparison.run_live_system(test_data)
    
    # Executar backtest
    comparison.run_backtest(test_data)
    
    # Comparar resultados
    comparison.compare_results()
    
    # Validar consistência
    success = comparison.validate_consistency()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)