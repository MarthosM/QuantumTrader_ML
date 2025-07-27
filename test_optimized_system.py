"""
Script de Teste do Sistema Otimizado
"""

import logging
from datetime import datetime, timedelta
import json
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ml_coordinator import MLCoordinator
from signal_generator import SignalGenerator
from adaptive_threshold_manager import AdaptiveThresholdManager
from features.feature_debugger import FeatureDebugger


def test_threshold_optimization():
    """Testa sistema com thresholds otimizados"""
    
    # Configurar logging detalhado
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("="*80)
    print("TESTE DO SISTEMA OTIMIZADO")
    print("="*80)
    
    # 1. Carregar e mostrar novos thresholds
    print("\n1. Novos Thresholds:")
    with open('config/improved_thresholds.json', 'r') as f:
        thresholds = json.load(f)
    
    print(json.dumps(thresholds, indent=2))
    
    # 2. Testar Adaptive Threshold Manager
    print("\n2. Testando Adaptive Threshold Manager:")
    threshold_manager = AdaptiveThresholdManager()
    
    # Simular alguns trades
    test_trades = [
        {'regime': 'trend_up', 'win': True, 'return': 0.015, 'confidence': 0.65},
        {'regime': 'trend_up', 'win': False, 'return': -0.008, 'confidence': 0.58},
        {'regime': 'range', 'win': True, 'return': 0.010, 'confidence': 0.55},
        {'regime': 'range', 'win': True, 'return': 0.012, 'confidence': 0.62},
    ]
    
    for trade in test_trades:
        threshold_manager.update_trade_result(trade)
    
    print("Performance Summary:")
    print(json.dumps(threshold_manager.get_performance_summary(), indent=2))
    
    print("\nSugestões de ajuste:")
    for suggestion in threshold_manager.suggest_threshold_adjustments():
        print(f"  - {suggestion}")
    
    # 3. Testar Feature Debugger
    print("\n3. Testando Feature Debugger:")
    
    # Criar dados de teste
    import pandas as pd
    import numpy as np
    
    # Simular features com diferentes qualidades
    n_samples = 100
    test_features = pd.DataFrame({
        'good_feature': np.random.randn(n_samples) * 2 + np.sin(np.linspace(0, 10, n_samples)),
        'noisy_feature': np.random.randn(n_samples) * 0.1,
        'nan_feature': [np.nan if i % 5 == 0 else np.random.randn() for i in range(n_samples)],
        'zero_var_feature': [1.0] * n_samples,
        'returns_1': np.random.randn(n_samples) * 0.01
    })
    
    feature_debugger = FeatureDebugger()
    analysis = feature_debugger.analyze_features(test_features)
    
    print(f"\nQuality Score: {analysis['quality_score']:.3f}")
    print(f"NaN médio: {analysis['nan_analysis']['avg_nan_percentage']:.1f}%")
    print(f"Features sem variância: {analysis['variance_analysis']['zero_variance_features']}")
    
    # 4. Configuração otimizada para Signal Generator
    print("\n4. Configuração do Signal Generator:")
    
    optimized_config = {
        'direction_threshold': thresholds['signal_generator']['direction_threshold'],
        'magnitude_threshold': thresholds['signal_generator']['magnitude_threshold'],
        'confidence_threshold': thresholds['signal_generator']['confidence_threshold'],
        'risk_per_trade': thresholds['signal_generator']['risk_per_trade'],
        'point_value': 0.5,
        'min_stop_points': 5
    }
    
    print(json.dumps(optimized_config, indent=2))
    
    # 5. Simulação de predições com diferentes cenários
    print("\n5. Simulando cenários de trading:")
    
    scenarios = [
        {
            'name': 'Trend forte - deve gerar sinal',
            'prediction': {
                'direction': 0.35,
                'magnitude': 0.002,
                'confidence': 0.65,
                'regime': 'trend_up',
                'regime_confidence': 0.75,
                'can_trade': True
            }
        },
        {
            'name': 'Range próximo ao suporte - deve gerar sinal',
            'prediction': {
                'direction': 0.25,
                'magnitude': 0.0012,
                'confidence': 0.55,
                'regime': 'range',
                'support_resistance_proximity': 'near_support',
                'can_trade': True
            }
        },
        {
            'name': 'Sinal fraco - não deve gerar trade',
            'prediction': {
                'direction': 0.15,
                'magnitude': 0.0008,
                'confidence': 0.45,
                'regime': 'undefined',
                'can_trade': False
            }
        }
    ]
    
    signal_gen = SignalGenerator(optimized_config)
    
    # Criar dados de mercado fake para teste
    market_data = type('obj', (object,), {
        'candles': pd.DataFrame({
            'close': [100.0] * 10,
            'high': [101.0] * 10,
            'low': [99.0] * 10
        }),
        'indicators': pd.DataFrame({
            'atr': [0.5] * 10
        })
    })()
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Prediction: {scenario['prediction']}")
        
        # Verificar thresholds
        regime = scenario['prediction'].get('regime', 'undefined')
        regime_thresholds = threshold_manager.get_thresholds_for_regime(regime)
        
        print(f"  Thresholds para {regime}: {regime_thresholds}")
        
        # Verificar se passaria nos testes
        passes = []
        if scenario['prediction']['confidence'] >= regime_thresholds.get('confidence', 0.6):
            passes.append('confidence')
        if abs(scenario['prediction']['direction']) >= regime_thresholds.get('direction', 0.3):
            passes.append('direction')
        if scenario['prediction']['magnitude'] >= regime_thresholds.get('magnitude', 0.001):
            passes.append('magnitude')
            
        print(f"  Passa em: {passes}")
        print(f"  Deve gerar trade: {'SIM' if len(passes) == 3 else 'NÃO'}")
    
    # 6. Recomendações finais
    print("\n" + "="*80)
    print("RECOMENDAÇÕES PARA MELHORAR O SISTEMA:")
    print("="*80)
    
    recommendations = [
        "1. THRESHOLDS AJUSTADOS:",
        "   - Reduzidos em 10-20% para gerar mais sinais",
        "   - Diferentes por regime (trend vs range)",
        "   - Sistema adaptativo implementado",
        "",
        "2. MELHORIAS NAS FEATURES:",
        "   - Feature Debugger para identificar problemas",
        "   - Remover features com zero variância",
        "   - Melhorar tratamento de NaN",
        "",
        "3. ANÁLISE DE REGIME:",
        "   - Usar RegimeAnalyzer.confidence como filtro adicional",
        "   - Range: focar em proximidade de suporte/resistência",
        "   - Trend: confirmar alinhamento de EMAs",
        "",
        "4. MONITORAMENTO:",
        "   - Adaptive thresholds se ajustam com performance",
        "   - Debug detalhado mostra razões de rejeição",
        "   - Feature quality score identifica problemas",
        "",
        "5. PRÓXIMOS PASSOS:",
        "   - Rodar enhanced_ml_backtester.py para análise completa",
        "   - Retreinar modelos se quality score < 0.6",
        "   - Ajustar thresholds baseado nas sugestões automáticas"
    ]
    
    for rec in recommendations:
        print(rec)
    
    print("\n" + "="*80)
    print("Para executar backtest otimizado:")
    print("python src/enhanced_ml_backtester.py")
    print("="*80)


if __name__ == "__main__":
    test_threshold_optimization()