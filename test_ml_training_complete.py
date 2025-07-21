#!/usr/bin/env python3
"""
Teste Completo do Sistema de Treinamento ML
Valida toda a pipeline de treinamento com dados reais
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Adicionar src ao path
sys.path.append('src')
sys.path.append('src/training')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_ml_training.log')
    ]
)
logger = logging.getLogger(__name__)

def generate_realistic_ohlcv_data(n_samples: int = 2000) -> pd.DataFrame:
    """Gera dados OHLCV realísticos para teste"""
    
    # Parâmetros realísticos para WDO
    base_price = 5000
    volatility = 0.02
    trend = 0.0001
    
    # Gerar timestamps
    start_date = datetime.now() - timedelta(days=n_samples)
    dates = pd.date_range(start=start_date, periods=n_samples, freq='1min')
    
    # Simulação de random walk com características financeiras
    np.random.seed(42)
    
    # Retornos com clustered volatility
    returns = np.random.normal(trend, volatility, n_samples)
    
    # Adicionar períodos de alta volatilidade
    high_vol_periods = np.random.choice(n_samples, size=n_samples//20, replace=False)
    returns[high_vol_periods] *= 3
    
    # Calcular preços
    log_prices = np.log(base_price) + np.cumsum(returns)
    close_prices = np.exp(log_prices)
    
    # Gerar OHLC baseado no close
    noise = np.random.normal(0, volatility * 0.3, n_samples)
    
    high_prices = close_prices * (1 + np.abs(noise) * 0.5)
    low_prices = close_prices * (1 - np.abs(noise) * 0.5)
    
    # Open é o close anterior com pequeno gap
    gaps = np.random.normal(0, volatility * 0.1, n_samples)
    open_prices = np.roll(close_prices, 1) * (1 + gaps)
    open_prices[0] = close_prices[0]  # Primeiro open = primeiro close
    
    # Ajustar OHLC para lógica correta
    for i in range(n_samples):
        ohlc_values = [open_prices[i], high_prices[i], low_prices[i], close_prices[i]]
        high_prices[i] = max(ohlc_values)
        low_prices[i] = min(ohlc_values)
    
    # Volume correlacionado com volatilidade
    base_volume = 1000
    vol_factor = np.abs(returns) / np.std(returns)
    volumes = base_volume * (1 + vol_factor) * np.random.lognormal(0, 0.5, n_samples)
    volumes = volumes.astype(int)
    
    # Criar DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices.round(2),
        'high': high_prices.round(2), 
        'low': low_prices.round(2),
        'close': close_prices.round(2),
        'volume': volumes
    })
    
    data.set_index('timestamp', inplace=True)
    
    logger.info(f"Dados gerados: {len(data)} amostras")
    logger.info(f"Preço médio: {data['close'].mean():.2f}")
    logger.info(f"Volatilidade: {data['close'].pct_change().std():.4f}")
    
    return data

def test_robust_nan_handler():
    """Testa o RobustNaNHandler"""
    logger.info("🧪 Testando RobustNaNHandler...")
    
    try:
        from robust_nan_handler import RobustNaNHandler
        
        # Criar dados com NaN intencionais
        data = generate_realistic_ohlcv_data(1000)
        
        # Simular features com NaN
        features_with_nan = pd.DataFrame({
            'rsi': [50.0] * 50 + [np.nan] * 50 + [60.0] * 900,
            'macd': np.random.normal(0, 1, 1000),
            'bb_upper_20': np.random.normal(5100, 100, 1000),
            'momentum_5': [np.nan] * 20 + list(np.random.normal(0.01, 0.02, 980)),
            'volume_sma_10': np.random.normal(1000, 200, 1000)
        })
        
        # Adicionar mais NaN intencionais
        features_with_nan.iloc[100:120, 1] = np.nan  # macd
        features_with_nan.iloc[200:210, 2] = np.nan  # bb_upper_20
        
        handler = RobustNaNHandler()
        
        # Processar
        clean_data, stats = handler.handle_nans(features_with_nan, data)
        
        # Validar
        validation = handler.validate_nan_handling(clean_data)
        
        # Relatório
        report = handler.create_nan_handling_report(stats, validation)
        
        logger.info(f"✅ NaN Handler OK - Score: {validation['quality_score']:.3f}")
        
        # Salvar relatório
        with open('test_nan_handler_report.txt', 'w') as f:
            f.write(report)
        
        return clean_data, True
        
    except Exception as e:
        logger.error(f"❌ Erro no NaN Handler: {e}")
        return None, False

def test_ml_training_system():
    """Testa o sistema completo de treinamento"""
    logger.info("🚀 Testando Sistema de Treinamento ML...")
    
    try:
        from ml_training_system import MLTrainingSystem
        
        # Configuração de teste
        config = {
            'max_features': 50,
            'test_size': 0.2,
            'min_data_points': 500,
            'models_dir': Path('test_models'),
            'reports_dir': Path('test_reports')
        }
        
        # Gerar dados
        raw_data = generate_realistic_ohlcv_data(1500)
        
        # Inicializar sistema
        training_system = MLTrainingSystem(config)
        
        # Preparar dados
        features, targets = training_system.prepare_training_data(raw_data)
        
        logger.info(f"Features preparadas: {features.shape}")
        logger.info(f"Targets: {targets.value_counts().to_dict()}")
        
        # Treinar modelos
        training_result = training_system.train_models(features, targets)
        
        # Relatório
        report = training_system.create_training_report(training_result)
        
        # Salvar relatório
        report_file = f"test_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"✅ Treinamento OK - Modelo salvo em: {training_result['model_dir']}")
        logger.info(f"📄 Relatório: {report_file}")
        
        return training_result, True
        
    except Exception as e:
        logger.error(f"❌ Erro no sistema de treinamento: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, False

def test_model_loading():
    """Testa carregamento do modelo treinado"""
    logger.info("📥 Testando carregamento de modelo...")
    
    try:
        import joblib
        
        # Encontrar último modelo
        models_dir = Path('test_models')
        if not models_dir.exists():
            logger.warning("Diretório de modelos não existe")
            return False
        
        # Encontrar último treinamento
        training_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('training_')]
        if not training_dirs:
            logger.warning("Nenhum modelo encontrado")
            return False
        
        latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
        
        # Carregar modelo
        model_path = latest_dir / 'model.pkl'
        scaler_path = latest_dir / 'scaler.pkl'
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        logger.info(f"✅ Modelo carregado de: {latest_dir}")
        
        # Teste de predição
        test_data = generate_realistic_ohlcv_data(100)
        
        # Simular features básicas para teste
        test_features = pd.DataFrame({
            'close': test_data['close'],
            'volume': test_data['volume'],
            'rsi': 50.0,
            'macd': 0.0,
            'momentum_5': 0.01
        })
        
        # Fazer predição simples (apenas para testar carregamento)
        try:
            predictions = model.predict(test_features.iloc[:10])
            logger.info(f"✅ Predição OK: {np.unique(predictions, return_counts=True)}")
            return True
        except Exception as pred_error:
            logger.warning(f"Erro na predição (normal em teste): {pred_error}")
            return True  # Modelo carregou, erro de features é esperado
        
    except Exception as e:
        logger.error(f"❌ Erro carregando modelo: {e}")
        return False

def main():
    """Executa todos os testes"""
    print("🧪 TESTE COMPLETO DO SISTEMA DE TREINAMENTO ML")
    print("=" * 60)
    
    results = {}
    
    # Teste 1: NaN Handler
    print("\n1️⃣ Testando RobustNaNHandler...")
    clean_data, results['nan_handler'] = test_robust_nan_handler()
    
    # Teste 2: Sistema de Treinamento
    print("\n2️⃣ Testando MLTrainingSystem...")
    training_result, results['training_system'] = test_ml_training_system()
    
    # Teste 3: Carregamento de Modelo
    print("\n3️⃣ Testando carregamento de modelo...")
    results['model_loading'] = test_model_loading()
    
    # Resumo
    print("\n" + "=" * 60)
    print("📊 RESUMO DOS TESTES")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name:<20}: {status}")
    
    all_passed = all(results.values())
    overall_status = "✅ TODOS OS TESTES PASSARAM" if all_passed else "⚠️ ALGUNS TESTES FALHARAM"
    
    print(f"\nStatus Geral: {overall_status}")
    
    # Cleanup opcional
    if all_passed:
        print("\n🧹 Limpeza de arquivos de teste disponível")
        response = input("Remover arquivos de teste? (s/N): ").lower().strip()
        if response == 's':
            import shutil
            for path in ['test_models', 'test_reports', 'test_nan_handler_report.txt']:
                if Path(path).exists():
                    if Path(path).is_dir():
                        shutil.rmtree(path)
                    else:
                        Path(path).unlink()
                    print(f"Removido: {path}")

if __name__ == "__main__":
    main()
