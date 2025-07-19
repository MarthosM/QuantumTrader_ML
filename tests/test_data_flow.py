#!/usr/bin/env python3
"""
Teste Completo do Sistema ML Trading v2.0
Segue o mapeamento de fluxo de dados definido
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configurar TensorFlow silencioso
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Adicionar src ao path
src_path = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, src_path)

def setup_logging():
    """Configura logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('SystemTest')

def generate_realistic_candles(size=2000):
    """Gera dados realistas de candles WDO"""
    
    logger = logging.getLogger('SystemTest')
    logger.info(f"📊 Gerando {size} candles realistas para WDO...")
    
    # Horário de mercado apenas
    end_time = datetime.now().replace(hour=17, minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(days=10)
    
    # Gerar apenas horário de mercado (9h às 18h)
    all_dates = pd.date_range(start_time, end_time, freq='1min')
    market_dates = [d for d in all_dates if 9 <= d.hour <= 17]
    market_dates = market_dates[-size:]  # Pegar últimos N
    
    # Preço inicial realista para WDO
    initial_price = 132500  # Aproximadamente 132.500 pontos
    
    # Parâmetros realistas
    daily_volatility = 0.02  # 2% ao dia
    intraday_vol = daily_volatility / (24 * 60) ** 0.5  # Volatilidade por minuto
    
    # Gerar movimento com tendência e volatilidade clustering
    np.random.seed(42)  # Reproduzível
    
    # Componentes do preço
    trend = np.cumsum(np.random.normal(0, intraday_vol * 0.1, size))  # Tendência lenta
    noise = np.random.normal(0, intraday_vol, size)  # Ruído
    
    # Volatility clustering - períodos de alta/baixa volatilidade
    vol_regime = np.random.choice([0.5, 1.0, 1.5], size, p=[0.6, 0.3, 0.1])
    
    # Calcular retornos
    returns = (trend + noise) * vol_regime
    returns[0] = 0
    
    # Calcular preços
    prices = [initial_price]
    for i in range(1, size):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 1000))  # Preço mínimo 1000
    
    # Criar OHLCV realista
    candles_data = []
    
    for i, (timestamp, close) in enumerate(zip(market_dates, prices)):
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1]
        
        # Volatilidade intrabar baseada no ATR
        if i > 20:
            recent_ranges = [(prices[j] - prices[j-1]) for j in range(max(1, i-20), i)]
            atr = np.mean([abs(r) for r in recent_ranges])
        else:
            atr = close * 0.005  # 0.5% inicial
        
        # High/Low baseado na volatilidade
        range_factor = np.random.uniform(0.3, 1.5)  # Variação do range
        high = max(open_price, close) + abs(np.random.normal(0, atr * range_factor))
        low = min(open_price, close) - abs(np.random.normal(0, atr * range_factor))
        
        # Garantir consistência OHLC
        high = max(float(high), float(open_price), float(close))
        low = min(float(low), float(open_price), float(close))
        
        # Volume realista - maior em movimentos grandes
        base_volume = 200
        price_change = abs(close - open_price) / open_price if open_price != 0 else 0
        volume_factor = 1 + (price_change * 10)  # Mais volume em grandes movimentos
        volume = int(base_volume * volume_factor * np.random.uniform(0.5, 2.0))
        
        candles_data.append({
            'open': round(open_price, 0),
            'high': round(high, 0),
            'low': round(low, 0),
            'close': round(close, 0),
            'volume': max(volume, 10)  # Volume mínimo
        })
    
    df = pd.DataFrame(candles_data, index=market_dates)
    
    # Validação final
    assert (df['high'] >= df['low']).all()
    assert (df['high'] >= df['open']).all()
    assert (df['high'] >= df['close']).all()
    assert (df['low'] <= df['open']).all()
    assert (df['low'] <= df['close']).all()
    assert (df['volume'] > 0).all()
    
    logger.info(f"✅ Candles gerados: {len(df)} períodos")
    logger.info(f"📈 Range de preços: {df['low'].min():.0f} - {df['high'].max():.0f}")
    logger.info(f"📊 Volume médio: {df['volume'].mean():.0f}")
    
    return df

def test_complete_data_flow():
    """Teste completo seguindo o mapeamento de fluxo de dados"""
    
    logger = setup_logging()
    
    logger.info("🚀 INICIANDO TESTE COMPLETO DO FLUXO DE DADOS")
    logger.info("=" * 80)
    
    results = {}
    
    # ETAPA 1: Inicializar estrutura de dados
    logger.info("📊 ETAPA 1: Inicializando TradingDataStructure")
    
    from data_structure import TradingDataStructure
    
    data_structure = TradingDataStructure()
    data_structure.initialize_structure()
    
    logger.info("✅ TradingDataStructure inicializada")
    results['data_structure'] = True
    
    # ETAPA 2: Gerar dados de candles realistas
    logger.info("📈 ETAPA 2: Gerando dados históricos realistas")
    
    candles_df = generate_realistic_candles(1500)  # 1500 candles ~2.5 dias
    
    # Atualizar estrutura (verificar se método existe)
    if hasattr(data_structure, 'update_candles'):
        data_structure.update_candles(candles_df)
    else:
        data_structure.candles = candles_df
    
    logger.info("✅ Dados históricos carregados na estrutura")
    results['historical_data'] = len(candles_df)
    
    # ETAPA 3: Validação de dados com ProductionDataValidator
    logger.info("🛡️ ETAPA 3: Validando dados com ProductionDataValidator")
    
    from feature_engine import ProductionDataValidator
    
    validator = ProductionDataValidator(logger)
    
    try:
        is_valid = validator.validate_real_data(candles_df, "candles_historicos")
        logger.info("✅ Dados validados pelo ProductionDataValidator")
        results['data_validation'] = True
    except Exception as e:
        logger.warning(f"⚠️ Validação com restrições: {str(e)}")
        results['data_validation'] = False
    
    # ETAPA 4: Cálculo de indicadores técnicos
    logger.info("📊 ETAPA 4: Calculando indicadores técnicos")
    
    from technical_indicators import TechnicalIndicators
    
    tech_indicators = TechnicalIndicators()
    indicators_df = tech_indicators.calculate_all(candles_df)
    
    # Atualizar estrutura (verificar se método existe)
    if hasattr(data_structure, 'update_indicators'):
        data_structure.update_indicators(indicators_df)
    else:
        data_structure.indicators = indicators_df
    
    logger.info(f"✅ Indicadores calculados: {len(indicators_df.columns)} indicadores")
    logger.info(f"📈 Indicadores principais encontrados:")
    
    key_indicators = ['ema_9', 'ema_20', 'ema_50', 'rsi_14', 'macd', 'macd_signal', 
                     'bb_upper', 'bb_lower', 'atr_14', 'adx_14']
    
    found_indicators = [ind for ind in key_indicators if ind in indicators_df.columns]
    for ind in found_indicators[:10]:  # Mostrar primeiros 10
        last_value = indicators_df[ind].iloc[-1] if not indicators_df[ind].isna().all() else 'N/A'
        logger.info(f"  {ind}: {last_value}")
    
    results['technical_indicators'] = len(indicators_df.columns)
    
    # ETAPA 5: Geração de microestrutura simulada
    logger.info("🔬 ETAPA 5: Gerando dados de microestrutura")
    
    def generate_microstructure(candles):
        """Gera microestrutura baseada nos candles"""
        micro_data = []
        
        for idx, row in candles.iterrows():
            # Simular buy/sell pressure baseado no movimento
            price_move = (row['close'] - row['open']) / row['open'] if row['open'] != 0 else 0
            volume_total = row['volume']
            
            # Bias baseado no movimento
            if price_move > 0.001:  # Subiu mais de 0.1%
                buy_bias = 0.65
            elif price_move < -0.001:  # Caiu mais de 0.1%
                buy_bias = 0.35
            else:  # Movimento lateral
                buy_bias = 0.50
            
            # Adicionar ruído
            buy_bias += np.random.normal(0, 0.1)
            buy_bias = np.clip(buy_bias, 0.1, 0.9)
            
            buy_volume = int(volume_total * buy_bias)
            sell_volume = volume_total - buy_volume
            
            # Simular número de trades
            avg_trade_size = np.random.randint(15, 35)
            buy_trades = max(1, buy_volume // avg_trade_size)
            sell_trades = max(1, sell_volume // avg_trade_size)
            
            micro_data.append({
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'total_trades': buy_trades + sell_trades
            })
        
        return pd.DataFrame(micro_data, index=candles.index)
    
    microstructure_df = generate_microstructure(candles_df)
    
    # Atualizar estrutura (verificar se método existe)
    if hasattr(data_structure, 'update_microstructure'):
        data_structure.update_microstructure(microstructure_df)
    else:
        data_structure.microstructure = microstructure_df
    
    logger.info(f"✅ Microestrutura gerada: {len(microstructure_df.columns)} colunas")
    logger.info(f"📊 Buy/Sell ratio médio: {microstructure_df['buy_volume'].sum() / (microstructure_df['buy_volume'].sum() + microstructure_df['sell_volume'].sum()):.2%}")
    
    results['microstructure'] = len(microstructure_df.columns)
    
    # ETAPA 6: Cálculo de features ML
    logger.info("🤖 ETAPA 6: Calculando features ML")
    
    from ml_features import MLFeatures
    
    # Features essenciais para day trading
    required_features = [
        'momentum_5', 'momentum_10', 'momentum_20',
        'volatility_10', 'volatility_20', 'volatility_50',
        'return_1', 'return_5', 'return_10',
        'volume_sma_ratio_10', 'volume_sma_ratio_20',
        'price_position_sma20', 'price_position_ema20',
        'rsi_momentum_5', 'macd_momentum_5'
    ]
    
    ml_features = MLFeatures(required_features)
    features_df = ml_features.calculate_all(candles_df, microstructure_df, indicators_df)
    
    # Atualizar estrutura (verificar se método existe)
    if hasattr(data_structure, 'update_features'):
        data_structure.update_features(features_df)
    else:
        data_structure.features = features_df
    
    logger.info(f"✅ Features ML calculadas: {len(features_df.columns)} features")
    logger.info(f"📊 Features com dados válidos:")
    
    # Mostrar estatísticas das features
    for col in features_df.columns[:10]:  # Primeiras 10
        valid_count = features_df[col].notna().sum()
        if valid_count > 0:
            mean_val = features_df[col].mean()
            logger.info(f"  {col}: {valid_count} valores válidos, média: {mean_val:.4f}")
    
    results['ml_features'] = len(features_df.columns)
    
    # ETAPA 7: Teste do FeatureEngine completo
    logger.info("⚡ ETAPA 7: Testando FeatureEngine completo")
    
    from feature_engine import FeatureEngine
    
    feature_engine = FeatureEngine(required_features)
    
    # Calcular todas as features usando o engine
    try:
        all_features = feature_engine.calculate(data_structure, force_recalculate=True)
        
        logger.info("✅ FeatureEngine executado com sucesso")
        logger.info(f"📊 Resultado do FeatureEngine:")
        
        for key, df in all_features.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                logger.info(f"  {key}: {df.shape[1]} colunas, {df.shape[0]} linhas")
                
                # Verificar qualidade
                total_cells = df.shape[0] * df.shape[1]
                nan_cells = df.isna().sum().sum()
                nan_ratio = nan_cells / total_cells if total_cells > 0 else 0
                logger.info(f"    Taxa de NaN: {nan_ratio:.1%}")
                
        results['feature_engine'] = True
        
    except Exception as e:
        logger.warning(f"⚠️ FeatureEngine com problemas: {str(e)}")
        results['feature_engine'] = False
    
    # ETAPA 8: Teste básico de conexão (se DLL disponível)
    logger.info("📡 ETAPA 8: Testando ConnectionManager")
    
    from connection_manager import ConnectionManager
    
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    
    if os.path.exists(dll_path):
        try:
            connection = ConnectionManager(dll_path)
            logger.info("✅ ConnectionManager criado com DLL real")
            
            # Não tentar conectar sem credenciais, apenas testar criação
            results['connection'] = True
            
        except Exception as e:
            logger.warning(f"⚠️ ConnectionManager com problemas: {str(e)}")
            results['connection'] = False
    else:
        logger.info("ℹ️ DLL do Profit não encontrada - pulando teste de conexão")
        results['connection'] = 'skipped'
    
    # ETAPA 9: Teste do ModelManager
    logger.info("🧠 ETAPA 9: Testando ModelManager")
    
    from model_manager import ModelManager
    
    models_dirs = ['test_models', 'models', 'src/models']
    model_manager = None
    
    for models_dir in models_dirs:
        if os.path.exists(models_dir):
            try:
                model_manager = ModelManager(models_dir)
                model_manager.load_models()
                
                loaded_count = len(model_manager.models) if hasattr(model_manager, 'models') else 0
                
                if loaded_count > 0:
                    logger.info(f"✅ ModelManager: {loaded_count} modelos carregados de {models_dir}")
                    
                    # Obter features requeridas
                    try:
                        all_required_features = model_manager.get_all_required_features()
                        logger.info(f"📊 Features requeridas pelos modelos: {len(all_required_features)}")
                        
                        # Mostrar algumas features
                        sample_features = list(all_required_features)[:10]
                        logger.info(f"🔍 Exemplos de features: {sample_features}")
                        
                        results['model_manager'] = {
                            'models': loaded_count,
                            'features': len(all_required_features)
                        }
                        
                    except Exception as fe:
                        logger.warning(f"⚠️ Erro ao obter features: {str(fe)}")
                        results['model_manager'] = {'models': loaded_count, 'features': 0}
                    
                    break
                    
                else:
                    logger.info(f"ℹ️ ModelManager criado mas sem modelos em {models_dir}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Erro no ModelManager ({models_dir}): {str(e)}")
    
    if model_manager is None:
        logger.info("ℹ️ Nenhum modelo encontrado - sistema funcionará sem ML")
        results['model_manager'] = {'models': 0, 'features': 0}
    
    # ETAPA 10: Relatório final
    logger.info("📋 ETAPA 10: Relatório final do teste")
    logger.info("=" * 80)
    
    logger.info("🎯 RESULTADOS DO TESTE COMPLETO:")
    logger.info("")
    
    # Resumo por componente
    if results.get('data_structure'):
        logger.info("✅ TradingDataStructure: Funcional")
    
    if results.get('historical_data'):
        logger.info(f"✅ Dados históricos: {results['historical_data']} candles carregados")
    
    if results.get('data_validation'):
        logger.info("✅ ProductionDataValidator: Validação ativa")
    else:
        logger.info("⚠️ ProductionDataValidator: Com restrições (esperado para dados de teste)")
    
    if results.get('technical_indicators'):
        logger.info(f"✅ TechnicalIndicators: {results['technical_indicators']} indicadores")
    
    if results.get('microstructure'):
        logger.info(f"✅ Microestrutura: {results['microstructure']} colunas geradas")
    
    if results.get('ml_features'):
        logger.info(f"✅ MLFeatures: {results['ml_features']} features calculadas")
    
    if results.get('feature_engine'):
        logger.info("✅ FeatureEngine: Sistema completo funcional")
    else:
        logger.info("⚠️ FeatureEngine: Com problemas - verificar implementação")
    
    connection_status = results.get('connection', 'error')
    if connection_status == True:
        logger.info("✅ ConnectionManager: DLL carregada")
    elif connection_status == 'skipped':
        logger.info("ℹ️ ConnectionManager: DLL não disponível")
    else:
        logger.info("⚠️ ConnectionManager: Problemas")
    
    model_info = results.get('model_manager', {})
    if isinstance(model_info, dict):
        models_count = model_info.get('models', 0)
        features_count = model_info.get('features', 0)
        if models_count > 0:
            logger.info(f"✅ ModelManager: {models_count} modelos, {features_count} features")
        else:
            logger.info("ℹ️ ModelManager: Sem modelos (ML desabilitado)")
    
    logger.info("")
    logger.info("📊 RESUMO GERAL:")
    
    # Calcular score de funcionamento
    total_components = 8
    working_components = sum([
        bool(results.get('data_structure')),
        bool(results.get('historical_data')),
        bool(results.get('technical_indicators')),
        bool(results.get('microstructure')),
        bool(results.get('ml_features')),
        bool(results.get('feature_engine')),
        results.get('connection') == True,
        isinstance(results.get('model_manager'), dict) and results['model_manager'].get('models', 0) > 0
    ])
    
    score = working_components / total_components
    
    logger.info(f"🎯 Score geral: {working_components}/{total_components} ({score:.1%})")
    
    if score >= 0.8:
        logger.info("✅ SISTEMA FUNCIONANDO MUITO BEM!")
        status = "excellent"
    elif score >= 0.6:
        logger.info("✅ SISTEMA FUNCIONANDO ADEQUADAMENTE")
        status = "good"
    elif score >= 0.4:
        logger.info("⚠️ SISTEMA FUNCIONANDO COM LIMITAÇÕES")
        status = "limited"
    else:
        logger.info("❌ SISTEMA COM PROBLEMAS SIGNIFICATIVOS")
        status = "problems"
    
    logger.info("")
    logger.info("🚀 PRÓXIMOS PASSOS RECOMENDADOS:")
    
    if status in ['excellent', 'good']:
        logger.info("1. ✅ Sistema pronto para integração com dados reais")
        logger.info("2. ✅ Carregar modelos ML treinados")
        logger.info("3. ✅ Configurar conexão real com Profit")
        logger.info("4. ✅ Executar paper trading para validação final")
    else:
        logger.info("1. 🔧 Corrigir componentes com falhas")
        logger.info("2. 📊 Validar cálculos de indicadores e features")
        logger.info("3. 🧠 Instalar/treinar modelos ML")
        logger.info("4. 🔄 Repetir teste após correções")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🏁 TESTE COMPLETO FINALIZADO")
    logger.info("=" * 80)
    
    return status, results

if __name__ == "__main__":
    status, results = test_complete_data_flow()
    
    # Exit code baseado no resultado
    if status == 'excellent':
        exit(0)
    elif status == 'good':
        exit(0)
    elif status == 'limited':
        exit(1)
    else:
        exit(2)
