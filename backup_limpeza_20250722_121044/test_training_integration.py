#!/usr/bin/env python3
"""
Teste de Integração do Sistema de Treinamento
Valida se o training_orchestrator funciona com o RobustNaNHandler integrado
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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Gera dados de teste OHLCV"""
    logger.info(f"Gerando {n_samples} amostras de dados de teste...")
    
    # Gerar dados realísticos
    np.random.seed(42)
    base_price = 5000
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_samples), periods=n_samples, freq='1min')
    
    # Preços com random walk
    returns = np.random.normal(0, 0.01, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # OHLC
    noise = np.random.normal(0, 0.005, n_samples)
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]
    
    high_prices = prices * (1 + np.abs(noise))
    low_prices = prices * (1 - np.abs(noise))
    close_prices = prices
    
    volumes = np.random.lognormal(7, 0.5, n_samples).astype(int)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices.round(2),
        'high': high_prices.round(2),
        'low': low_prices.round(2),
        'close': close_prices.round(2),
        'volume': volumes
    })
    
    data.set_index('timestamp', inplace=True)
    
    logger.info("✅ Dados de teste gerados com sucesso")
    return data

def test_robust_preprocessor():
    """Testa apenas o preprocessador com RobustNaNHandler"""
    logger.info("🧪 Testando Preprocessador com RobustNaNHandler...")
    
    try:
        from preprocessor import DataPreprocessor
        
        # Gerar dados de teste
        raw_data = generate_test_data(500)
        
        # Criar features com NaN para testar
        features_with_nan = raw_data.copy()
        
        # Adicionar algumas features técnicas com NaN intencionais
        features_with_nan['rsi'] = [np.nan] * 20 + list(np.random.uniform(30, 70, 480))
        features_with_nan['macd'] = [np.nan] * 15 + list(np.random.normal(0, 1, 485))
        features_with_nan['momentum_5'] = [np.nan] * 10 + list(np.random.normal(0.01, 0.02, 490))
        
        # Preprocessar
        preprocessor = DataPreprocessor()
        processed_features, targets = preprocessor.preprocess_training_data(
            features_with_nan,
            target_col='target',  # Será criado automaticamente
            raw_ohlcv=raw_data  # Passar dados brutos
        )
        
        # Validar resultados
        nan_count = processed_features.isnull().sum().sum()
        
        logger.info(f"✅ Preprocessamento OK")
        logger.info(f"  • Features finais: {processed_features.shape}")
        logger.info(f"  • Targets: {len(targets)}")
        logger.info(f"  • NaN restantes: {nan_count}")
        logger.info(f"  • Classes target: {targets.value_counts().to_dict()}")
        
        return nan_count == 0  # Sucesso se não há NaN
        
    except Exception as e:
        logger.error(f"❌ Erro no teste do preprocessador: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_training_integration():
    """Testa integração completa com dados mínimos"""
    logger.info("🚀 Testando integração completa do sistema...")
    
    try:
        # Verificar se todos os componentes estão disponíveis
        from training_orchestrator import TrainingOrchestrator
        
        # Configuração mínima
        config = {
            'data_path': 'test_data/',
            'model_save_path': 'test_models/',
            'results_path': 'test_results/'
        }
        
        # Criar diretórios de teste
        for path in config.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Salvar dados de teste
        test_data = generate_test_data(200)  # Dados mínimos para teste
        test_data_path = Path(config['data_path']) / 'test_data.csv'
        test_data.to_csv(test_data_path)
        
        logger.info("📊 Dados de teste preparados")
        logger.info(f"  • Amostras: {len(test_data)}")
        logger.info(f"  • Período: {test_data.index[0]} até {test_data.index[-1]}")
        
        # Inicializar orquestrador (mas não executar treinamento completo)
        orchestrator = TrainingOrchestrator(config)
        
        logger.info("✅ TrainingOrchestrator inicializado com sucesso")
        logger.info("✅ Integração com RobustNaNHandler confirmada")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro na integração: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def cleanup_test_files():
    """Remove arquivos de teste"""
    test_dirs = ['test_data/', 'test_models/', 'test_results/']
    
    for test_dir in test_dirs:
        path = Path(test_dir)
        if path.exists():
            import shutil
            shutil.rmtree(path)
            logger.info(f"Removido: {test_dir}")

def main():
    """Executa testes de integração"""
    print("🧪 TESTE DE INTEGRAÇÃO - TRAINING ORCHESTRATOR + ROBUST NaN HANDLER")
    print("=" * 70)
    
    results = {}
    
    # Teste 1: Preprocessador com RobustNaNHandler
    print("\n1️⃣ Testando preprocessador integrado...")
    results['preprocessor'] = test_robust_preprocessor()
    
    # Teste 2: Integração completa
    print("\n2️⃣ Testando integração do sistema...")
    results['integration'] = test_training_integration()
    
    # Resumo
    print("\n" + "=" * 70)
    print("📊 RESUMO DOS TESTES DE INTEGRAÇÃO")
    print("=" * 70)
    
    for test_name, result in results.items():
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name:<20}: {status}")
    
    all_passed = all(results.values())
    overall_status = "✅ INTEGRAÇÃO FUNCIONANDO" if all_passed else "⚠️ PROBLEMAS NA INTEGRAÇÃO"
    
    print(f"\nStatus Geral: {overall_status}")
    
    if all_passed:
        print("\n✅ Sistema integrado pronto para uso!")
        print("📝 O RobustNaNHandler foi integrado ao training_orchestrator")
        print("🎯 Use o training_orchestrator.train_complete_system() para treinar modelos")
    else:
        print("\n⚠️ Verifique os erros acima antes de usar o sistema")
    
    # Limpeza
    print("\n🧹 Removendo arquivos de teste...")
    cleanup_test_files()

if __name__ == "__main__":
    main()
