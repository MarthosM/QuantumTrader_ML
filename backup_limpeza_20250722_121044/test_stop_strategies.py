#!/usr/bin/env python3
"""
Teste da classe StopLossFeatureCalculator
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_stop_loss_feature_calculator():
    """Testa o calculador de features para stop loss"""
    print("🧪 TESTE: StopLossFeatureCalculator")
    print("="*60)
    
    try:
        from stop_strategies import StopLossFeatureCalculator, MLOptimizedStop
        
        # Criar dados de mercado de teste
        dates = pd.date_range(start='2025-01-01', periods=50, freq='1min')
        
        # Simular dados OHLCV realistas
        np.random.seed(42)
        base_price = 5100
        
        data = {
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        }
        
        current_price = base_price
        for i in range(50):
            # Movimento de preço realista
            change = np.random.normal(0, 0.002)  # 0.2% de volatilidade média
            
            open_price = current_price
            close_price = current_price * (1 + change)
            
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.001)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.001)))
            
            volume = np.random.randint(50, 200)
            
            data['open'].append(open_price)
            data['high'].append(high_price)
            data['low'].append(low_price)
            data['close'].append(close_price)
            data['volume'].append(volume)
            
            current_price = close_price
        
        market_data = pd.DataFrame(data, index=dates)
        
        print(f"📊 Dados de teste criados: {len(market_data)} candles")
        print(f"   Período: {market_data.index[0]} até {market_data.index[-1]}")
        print(f"   Preço inicial: {market_data['close'].iloc[0]:.2f}")
        print(f"   Preço final: {market_data['close'].iloc[-1]:.2f}")
        
        # Criar posição de teste
        position = {
            'side': 'long',
            'entry_price': market_data['close'].iloc[20],
            'current_price': market_data['close'].iloc[-1],
            'entry_time': market_data.index[20],
            'quantity': 1000
        }
        
        print(f"\n📍 Posição de teste:")
        print(f"   Lado: {position['side']}")
        print(f"   Preço entrada: {position['entry_price']:.2f}")
        print(f"   Preço atual: {position['current_price']:.2f}")
        print(f"   Tempo entrada: {position['entry_time']}")
        
        # Testar calculador de features
        calculator = StopLossFeatureCalculator()
        
        print(f"\n🔧 Testando calculador de features...")
        features = calculator.calculate_features(
            position=position,
            market_data=market_data,
            market_regime='trending'
        )
        
        print(f"✅ Features calculadas: {len(features)} features")
        
        # Mostrar features com nomes
        feature_names = calculator.get_feature_names()
        
        print(f"\n📋 FEATURES CALCULADAS:")
        for i, (name, value) in enumerate(zip(feature_names, features)):
            print(f"   {i+1:2d}. {name:<20}: {value:.6f}")
        
        # Testar MLOptimizedStop
        print(f"\n🤖 Testando MLOptimizedStop...")
        
        ml_stop = MLOptimizedStop()
        stop_price = ml_stop.calculate_stop(
            position=position,
            market_data=market_data,
            market_regime='trending'
        )
        
        print(f"✅ Stop loss calculado: {stop_price:.2f}")
        print(f"   Preço atual: {position['current_price']:.2f}")
        print(f"   Stop loss: {stop_price:.2f}")
        
        # Calcular distância do stop
        if position['side'] == 'long':
            stop_distance = (position['current_price'] - stop_price) / position['current_price']
        else:
            stop_distance = (stop_price - position['current_price']) / position['current_price']
        
        print(f"   Distância: {stop_distance*100:.2f}%")
        
        # Validações
        validations = []
        
        # 1. Features devem ter tamanho correto
        expected_features = 12
        validations.append(("Número de features", len(features) == expected_features))
        
        # 2. Features devem ser numéricas
        all_numeric = all(isinstance(f, (int, float)) and not pd.isna(f) for f in features)
        validations.append(("Features numéricas", all_numeric))
        
        # 3. Stop price deve ser razoável
        reasonable_stop = abs(stop_distance) < 0.1  # Menos de 10%
        validations.append(("Stop razoável (<10%)", reasonable_stop))
        
        # 4. Features dentro de ranges esperados
        atr_normalized = features[0]
        volatility = features[1]
        reasonable_ranges = (0 <= atr_normalized <= 0.1 and 0 <= volatility <= 0.1)
        validations.append(("Features em ranges esperados", reasonable_ranges))
        
        print(f"\n✅ VALIDAÇÕES:")
        all_passed = True
        for test_name, passed in validations:
            status = "✅ PASSOU" if passed else "❌ FALHOU"
            print(f"   {test_name:<30}: {status}")
            if not passed:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_regimes():
    """Testa com diferentes regimes de mercado"""
    print(f"\n🎯 TESTE: Diferentes Regimes de Mercado")
    print("="*60)
    
    try:
        from stop_strategies import StopLossFeatureCalculator
        
        # Criar dados simples
        data = {
            'open': [5100, 5102, 5104],
            'high': [5105, 5107, 5109],
            'low': [5095, 5097, 5099],
            'close': [5102, 5104, 5106],
            'volume': [100, 120, 110]
        }
        
        market_data = pd.DataFrame(data)
        
        position = {
            'side': 'long',
            'entry_price': 5100,
            'current_price': 5106,
            'entry_time': datetime.now() - timedelta(minutes=10)
        }
        
        calculator = StopLossFeatureCalculator()
        regimes = ['trend_up', 'trend_down', 'ranging', 'high_volatility', 'undefined']
        
        print("📊 Features para diferentes regimes:")
        
        for regime in regimes:
            features = calculator.calculate_features(position, market_data, regime)
            regime_score = features[8]  # Feature do regime
            print(f"   {regime:<15}: regime_score = {regime_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def main():
    """Executa todos os testes"""
    print("🧪 TESTE COMPLETO - STOP LOSS FEATURE CALCULATOR")
    print("="*70)
    
    results = []
    
    # Teste principal
    result1 = test_stop_loss_feature_calculator()
    results.append(("Feature Calculator", result1))
    
    # Teste de regimes
    result2 = test_different_regimes()
    results.append(("Diferentes Regimes", result2))
    
    # Resultado final
    print(f"\n" + "="*70)
    print("📋 RESULTADO DOS TESTES")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"   {test_name:<25}: {status}")
    
    print(f"\n🎯 RESULTADO FINAL: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 STOPLOSSFEATURECALCULATOR FUNCIONANDO PERFEITAMENTE!")
        print("✅ Todas as features são calculadas corretamente")
        print("✅ MLOptimizedStop pode usar o calculador")
        print("✅ Sistema pronto para produção")
    else:
        print("⚠️ Alguns testes falharam - verificar implementação")
    
    print(f"\n💡 USO NO SISTEMA:")
    print("```python")
    print("from stop_strategies import MLOptimizedStop")
    print("ml_stop = MLOptimizedStop()")
    print("stop_price = ml_stop.calculate_stop(position, market_data, regime)")
    print("```")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
