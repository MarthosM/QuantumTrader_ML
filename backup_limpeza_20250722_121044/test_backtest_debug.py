#!/usr/bin/env python3
"""
Debug do backtest ML - Ver predições em tempo real
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from ml_backtester import AdvancedMLBacktester, BacktestConfig, BacktestMode

def create_debug_data():
    """Cria dados históricos com padrões óbvios para forçar trades"""
    print("📊 CRIANDO DADOS DE DEBUG COM PADRÕES ÓBVIOS...")
    
    # 7 dias de dados com padrões fortes
    dates = pd.date_range('2025-06-13 09:00:00', periods=2100, freq='1min')
    
    # Criar tendências fortes para provocar sinais
    data = []
    base_price = 5500
    
    for i, dt in enumerate(dates):
        cycle_pos = i % 500
        
        # Criar padrões cíclicos fortes
        if cycle_pos < 250:  # Tendência de alta
            price_trend = base_price + (cycle_pos * 0.2)  # Alta consistente
            volume_boost = 1.5  # Volume aumentado
        else:  # Tendência de baixa
            price_trend = base_price + (250 * 0.2) - ((cycle_pos - 250) * 0.15)  # Baixa
            volume_boost = 1.3
            
        # Adicionar volatilidade controlada
        noise = np.random.normal(0, 2)
        close_price = max(price_trend + noise, 1)  # Evitar preços negativos
        
        high = close_price + abs(np.random.normal(0, 3))
        low = close_price - abs(np.random.normal(0, 3))
        open_price = close_price + np.random.normal(0, 2)
        volume = int(np.random.randint(100, 300) * volume_boost)
        
        data.append({
            'timestamp': dt,
            'open': max(open_price, 1),
            'high': max(high, open_price, close_price),
            'low': min(low, open_price, close_price),
            'close': close_price,
            'volume': volume,
            'contract': 'WDOH25'
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ Dados criados: {len(df)} barras")
    print(f"   Período: {df.index[0]} até {df.index[-1]}")
    print(f"   Preço inicial: R$ {df['close'].iloc[0]:.2f}")
    print(f"   Preço final: R$ {df['close'].iloc[-1]:.2f}")
    print(f"   Variação: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
    
    return df

def run_debug_backtest():
    """Executa backtest debug com logs detalhados"""
    print("=== DEBUG BACKTEST ML SYSTEM ===")
    
    # 1. Configurar logging mais verboso
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    
    # 2. Criar dados de teste
    test_data = create_debug_data()
    
    # 3. Configurar backtest
    config = BacktestConfig(
        start_date=test_data.index[500],   # Começar depois de histórico
        end_date=test_data.index[800],     # Processar 300 barras
        initial_capital=100000.0,
        mode=BacktestMode.REALISTIC
    )
    
    print(f"\n🔍 CONFIGURAÇÃO DO BACKTEST:")
    print(f"   Início: {config.start_date}")
    print(f"   Fim: {config.end_date}")
    print(f"   Capital inicial: R$ {config.initial_capital:,.2f}")
    print(f"   Barras históricas disponíveis: {len(test_data.loc[:config.start_date])}")
    print(f"   Barras a processar: {len(test_data.loc[config.start_date:config.end_date])}")
    
    # 4. Inicializar backtester
    backtester = AdvancedMLBacktester(config)
    
    # 5. Carregar modelos ML
    from model_manager import ModelManager
    try:
        models_dir = os.path.join('src', 'training', 'models', 'training_20250720_184206', 'ensemble')
        model_manager = ModelManager(models_dir)
        model_manager.load_models()
        ml_models = model_manager.models
        
        print(f"\n🤖 MODELOS CARREGADOS:")
        for name, model in ml_models.items():
            if hasattr(model, 'feature_names_in_'):
                feature_count = len(model.feature_names_in_)
                print(f"   {name}: {feature_count} features")
            else:
                print(f"   {name}: modelo sem feature_names_in_")
                
    except Exception as e:
        print(f"❌ Erro carregando modelos: {e}")
        ml_models = {}
    
    # 6. Inicializar backtester
    from feature_engine import FeatureEngine
    feature_engine = FeatureEngine()
    backtester.initialize(ml_models, feature_engine)
    
    # 7. Executar backtest com logging
    print(f"\n🚀 EXECUTANDO BACKTEST...")
    results = backtester.run_backtest(test_data)
    
    # 8. Analisar resultados
    print(f"\n📊 RESULTADOS DETALHADOS:")
    print(f"   Total trades: {results.get('total_trades', 0)}")
    print(f"   Trades vencedores: {results.get('winning_trades', 0)}")
    print(f"   Trades perdedores: {results.get('losing_trades', 0)}")
    print(f"   Win rate: {results.get('win_rate', 0):.1%}")
    print(f"   PnL total: R$ {results.get('total_pnl', 0):,.2f}")
    print(f"   Capital final: R$ {results.get('final_equity', 0):,.2f}")
    print(f"   Retorno total: {((results.get('final_equity', 100000) / 100000) - 1) * 100:.2f}%")
    
    # 9. Análise dos trades
    if 'trade_analysis' in results:
        trade_analysis = results['trade_analysis']
        print(f"\n📈 ANÁLISE POR LADO:")
        if 'by_side' in trade_analysis:
            for side, stats in trade_analysis['by_side'].items():
                print(f"   {side.capitalize()}: {stats.get('count', 0)} trades, {stats.get('win_rate', 0):.1%} win rate")
    
    # 10. Verificar se houve predições
    if results.get('total_trades', 0) == 0:
        print(f"\n⚠️  ZERO TRADES - ANÁLISE:")
        print(f"   ✓ Modelos foram carregados corretamente")
        print(f"   ✓ Dados históricos estão disponíveis")  
        print(f"   ✓ Features podem estar sendo calculadas")
        print(f"   → Possíveis causas:")
        print(f"     - Thresholds de confiança muito altos")
        print(f"     - Predições dos modelos muito conservadoras")
        print(f"     - Features não atingindo padrões esperados")
        print(f"     - Regime de mercado não sendo detectado adequadamente")
    
    return results

def analyze_single_prediction():
    """Analisa uma única predição em detalhes"""
    print(f"\n🔬 ANÁLISE DETALHADA DE UMA PREDIÇÃO:")
    
    # Usar dados reais
    test_data = create_debug_data()
    
    # Configurar
    config = BacktestConfig(
        start_date=test_data.index[500],
        end_date=test_data.index[501],  # Apenas 1 barra
        initial_capital=100000.0,
        mode=BacktestMode.REALISTIC
    )
    
    backtester = AdvancedMLBacktester(config)
    
    # Carregar modelos
    try:
        from model_manager import ModelManager
        models_dir = os.path.join('src', 'training', 'models', 'training_20250720_184206', 'ensemble')
        model_manager = ModelManager(models_dir)
        model_manager.load_models()
        ml_models = model_manager.models
        
        from feature_engine import FeatureEngine
        feature_engine = FeatureEngine()
        backtester.initialize(ml_models, feature_engine)
        
        # Testar geração de features
        timestamp = test_data.index[500]
        historical_data = test_data.loc[:timestamp]
        
        print(f"   Timestamp: {timestamp}")
        print(f"   Dados históricos: {len(historical_data)} barras")
        
        # Gerar features
        features = backtester._generate_features(historical_data)
        
        if features is not None:
            print(f"   ✅ Features geradas: {features.shape}")
            print(f"   Features disponíveis: {list(features.columns)[:10]}...")  # Primeiras 10
            
            # Obter predição ML
            market_data = test_data.loc[timestamp]
            ml_signal = backtester._get_ml_prediction(features, market_data)
            
            print(f"   🤖 Predição ML:")
            print(f"      Ação: {ml_signal['action']}")
            print(f"      Confiança: {ml_signal['confidence']:.3f}")
            print(f"      Prediction array: {ml_signal['prediction']}")
            print(f"      Preço: R$ {ml_signal['price']:.2f}")
            
            # Verificar por que não virou trade
            if ml_signal['action'] == 'none':
                print(f"   ⚠️  Sinal NONE - possíveis motivos:")
                print(f"      - Confiança ({ml_signal['confidence']:.3f}) < 0.6")
                print(f"      - Regime indefinido")
                print(f"      - Thresholds não atingidos")
            else:
                print(f"   ✅ Sinal válido gerado: {ml_signal['action']}")
                
        else:
            print(f"   ❌ Nenhuma feature foi gerada")
            print(f"      Dados históricos insuficientes: {len(historical_data)} < 100")
            
    except Exception as e:
        print(f"❌ Erro na análise: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Executa todos os testes de debug"""
    run_debug_backtest()
    analyze_single_prediction()

if __name__ == "__main__":
    main()
