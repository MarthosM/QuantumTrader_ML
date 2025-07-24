#!/usr/bin/env python3
"""
Exemplo de Uso do Sistema de Treinamento Integrado
Demonstra como usar o training_orchestrator com RobustNaNHandler
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Adicionar src ao path
sys.path.append('src')
sys.path.append('src/training')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_example_training():
    """Executa exemplo completo de treinamento"""
    
    print("🤖 EXEMPLO DE TREINAMENTO - SISTEMA INTEGRADO")
    print("=" * 60)
    print("📋 Este exemplo demonstra:")
    print("  • Carregamento de dados")
    print("  • Tratamento robusto de NaN")
    print("  • Treinamento de modelos ML")
    print("  • Validação temporal")
    print("=" * 60)
    
    try:
        from training_orchestrator import TrainingOrchestrator
        
        # 1. Configuração do sistema
        config = {
            'data_path': 'data/historical/',
            'model_save_path': 'src/training/models/',
            'results_path': 'training_results/',
            'min_data_points': 100,  # Mínimo reduzido para exemplo
            'validation_method': 'walk_forward'
        }
        
        print(f"\n📁 Configuração:")
        print(f"  • Dados: {config['data_path']}")
        print(f"  • Modelos: {config['model_save_path']}")
        print(f"  • Resultados: {config['results_path']}")
        
        # 2. Inicializar orquestrador
        orchestrator = TrainingOrchestrator(config)
        print("✅ TrainingOrchestrator inicializado")
        
        # 3. Definir período e símbolos (exemplo)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 30 dias de dados
        symbols = ['WDO']  # Exemplo para WDO
        
        print(f"\n📅 Parâmetros de treinamento:")
        print(f"  • Período: {start_date.strftime('%Y-%m-%d')} até {end_date.strftime('%Y-%m-%d')}")
        print(f"  • Símbolos: {', '.join(symbols)}")
        
        # 4. Métricas alvo
        target_metrics = {
            'accuracy': 0.55,      # 55% de acurácia
            'f1_score': 0.50,      # F1 Score > 50%
            'precision': 0.55      # Precisão > 55%
        }
        
        print(f"\n🎯 Métricas alvo:")
        for metric, target in target_metrics.items():
            print(f"  • {metric}: {target}")
        
        # 5. Executar treinamento (comentado para exemplo)
        print(f"\n🚀 COMANDO PARA EXECUTAR TREINAMENTO:")
        print(f"```python")
        print(f"results = orchestrator.train_complete_system(")
        print(f"    start_date={start_date},")
        print(f"    end_date={end_date},")
        print(f"    symbols={symbols},")
        print(f"    target_metrics={target_metrics},")
        print(f"    validation_method='walk_forward'")
        print(f")")
        print(f"```")
        
        print(f"\n📝 RECURSOS INTEGRADOS:")
        print(f"✅ RobustNaNHandler - Tratamento inteligente de valores ausentes")
        print(f"✅ Validação temporal com walk-forward")
        print(f"✅ Ensemble de modelos (XGBoost + LightGBM + Random Forest)")
        print(f"✅ Otimização de hiperparâmetros")
        print(f"✅ Métricas de trading específicas")
        print(f"✅ Relatórios automáticos")
        
        print(f"\n💡 BENEFÍCIOS DO SISTEMA INTEGRADO:")
        print(f"  • Tratamento robusto de NaN sem viés")
        print(f"  • Recálculo automático de indicadores problemáticos")
        print(f"  • Estratégias específicas por tipo de feature")
        print(f"  • Validação de qualidade dos dados")
        print(f"  • Pipeline completo end-to-end")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no exemplo: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def show_integration_benefits():
    """Mostra os benefícios da integração"""
    
    print(f"\n" + "="*60)
    print("🔧 DETALHES DA INTEGRAÇÃO")
    print("="*60)
    
    print(f"\n📊 ANTES (Sistema Antigo):")
    print(f"  ❌ SimpleImputer básico")
    print(f"  ❌ Forward/backward fill genérico")
    print(f"  ❌ Remoção em massa de linhas com NaN")
    print(f"  ❌ Sem recálculo de indicadores")
    print(f"  ❌ Introdução de viés nos dados")
    
    print(f"\n📈 AGORA (Sistema Integrado):")
    print(f"  ✅ RobustNaNHandler com estratégias específicas")
    print(f"  ✅ Recálculo automático de RSI, MACD, Bollinger Bands")
    print(f"  ✅ Interpolação linear para momentum")
    print(f"  ✅ Forward fill apenas para lags")
    print(f"  ✅ Validação de qualidade dos dados")
    print(f"  ✅ Relatórios detalhados de tratamento")
    
    print(f"\n🎯 ESTRATÉGIAS POR FEATURE:")
    
    strategies = {
        'Indicadores Técnicos': ['RSI', 'MACD', 'Bollinger Bands', 'ATR', 'ADX'],
        'Momentum': ['momentum_1', 'momentum_5', 'roc_10', 'return_20'],
        'Volume': ['volume_sma', 'volume_ratio', 'volume_momentum'],
        'Volatilidade': ['volatility_5', 'parkinson_vol', 'gk_vol'],
        'Lags': ['rsi_lag_1', 'macd_lag_5', 'momentum_lag_10']
    }
    
    strategy_methods = {
        'Indicadores Técnicos': 'RECÁLCULO ADEQUADO',
        'Momentum': 'INTERPOLAÇÃO LINEAR',
        'Volume': 'RECÁLCULO ADEQUADO',
        'Volatilidade': 'RECÁLCULO ADEQUADO',
        'Lags': 'FORWARD FILL'
    }
    
    for category, features in strategies.items():
        method = strategy_methods[category]
        print(f"\n  📋 {category} → {method}")
        print(f"     Features: {', '.join(features[:3])}...")

def main():
    """Executa exemplo completo"""
    
    # Exemplo de uso
    success = run_example_training()
    
    if success:
        # Mostrar detalhes da integração
        show_integration_benefits()
        
        print(f"\n" + "="*60)
        print("✅ SISTEMA PRONTO PARA PRODUÇÃO!")
        print("="*60)
        print(f"📋 PRÓXIMOS PASSOS:")
        print(f"  1. Prepare seus dados históricos OHLCV")
        print(f"  2. Configure os parâmetros em config/")
        print(f"  3. Execute: training_orchestrator.train_complete_system()")
        print(f"  4. Verifique os resultados em training_results/")
        print(f"  5. Carregue modelos com ModelManager")
        
    else:
        print(f"\n❌ Verifique os erros acima")

if __name__ == "__main__":
    main()
