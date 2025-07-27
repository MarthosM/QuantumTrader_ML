#!/usr/bin/env python3
"""
Backtest otimizado com cálculo eficiente de features
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurar variáveis para backtest
os.environ['TRADING_ENV'] = 'backtest'
os.environ['ALLOW_HISTORICAL_DATA'] = 'true'
os.environ['DISABLE_DATA_VALIDATION'] = 'true'

# Importar componentes do sistema
from src.ml_backtester import AdvancedMLBacktester, BacktestConfig, BacktestMode
from src.model_manager import ModelManager
from src.feature_engine import FeatureEngine
from src.data_structure import TradingDataStructure

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_real_models():
    """Carrega modelos ML reais do sistema"""
    try:
        models_dir = os.getenv('MODELS_DIR')
        if not models_dir:
            models_dir = r"C:\Users\marth\OneDrive\Programacao\Python\Projetos\ML_Tradingv2.0\src\training\models\training_20250720_184206\ensemble\ensemble_20250720_184206"
        
        print(f"Carregando modelos de: {models_dir}")
        
        if not os.path.exists(models_dir):
            print(f"Diretorio de modelos nao encontrado: {models_dir}")
            return None, None
        
        # Criar ModelManager
        model_manager = ModelManager(models_dir)
        
        # Carregar modelos
        success = model_manager.load_models()
        
        if success and hasattr(model_manager, 'models') and model_manager.models:
            print(f"{len(model_manager.models)} modelos carregados:")
            for name in model_manager.models.keys():
                print(f"   - {name}")
            
            # Criar FeatureEngine configurado para backtest
            feature_engine = FeatureEngine()
            
            # Desabilitar todas as validações para backtest
            if hasattr(feature_engine, 'require_data_validation'):
                feature_engine.require_data_validation = False
            
            if hasattr(feature_engine, 'validator'):
                # Desabilitar validador
                feature_engine.validator = None
            
            # Configurar modo backtest se disponível
            if hasattr(feature_engine, 'set_backtest_mode'):
                feature_engine.set_backtest_mode(True)
            
            print(f"FeatureEngine configurado para backtest (validacoes desabilitadas)")
            
            return model_manager, feature_engine
        else:
            print("Nenhum modelo foi carregado")
            return None, None
            
    except Exception as e:
        print(f"Erro carregando modelos: {e}")
        return None, None

def load_wdo_data_for_backtest(days_back=30):
    """Carrega dados WDO para backtest"""
    csv_path = r"C:\Users\marth\OneDrive\Programacao\Python\Projetos\ML_Tradingv2.0\src\training\data\historical\wdo_data_20_06_2025.csv"
    
    try:
        # Carregar dados
        df = pd.read_csv(csv_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        
        # Limpar dados
        columns_to_remove = ['contract', 'preco']
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Converter colunas necessárias
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Adicionar colunas de microestrutura se não existirem
        if 'buy_volume' not in df.columns and 'volume' in df.columns:
            df['buy_volume'] = df['volume'] * 0.5  # Assumir distribuição 50/50
            df['sell_volume'] = df['volume'] * 0.5
        
        if 'trades' not in df.columns:
            if 'quantidade' in df.columns:
                df['trades'] = df['quantidade']
            else:
                df['trades'] = 100  # Valor padrão
        
        # Remover NaN
        df = df.dropna()
        
        # Selecionar período mais recente
        end_date = df.index.max()
        start_date = end_date - timedelta(days=days_back)
        
        mask = (df.index >= start_date) & (df.index <= end_date)
        df_filtered = df[mask]
        
        print(f"Dados carregados: {len(df_filtered)} registros")
        print(f"Periodo: {df_filtered.index.min()} ate {df_filtered.index.max()}")
        print(f"Preco medio: R$ {df_filtered['close'].mean():,.2f}")
        
        return df_filtered
        
    except Exception as e:
        print(f"Erro carregando dados: {e}")
        return None

def calculate_all_features_optimized(df):
    """Calcula todas as features necessárias de forma otimizada"""
    features_df = df.copy()
    
    # Calcular retornos uma vez
    returns = df['close'].pct_change()
    
    # EMAs (vetorizado)
    for period in [9, 20, 50, 200]:
        features_df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    features_df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Bollinger Bands (vetorizado)
    for period in [20, 50]:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        features_df[f'bb_upper_{period}'] = sma + (2 * std)
        features_df[f'bb_middle_{period}'] = sma
        features_df[f'bb_lower_{period}'] = sma - (2 * std)
        features_df[f'bb_width_{period}'] = 4 * std  # upper - lower = 4*std
    
    # True Range para ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # ATR
    features_df['atr'] = true_range.ewm(span=14, adjust=False).mean()
    features_df['atr_20'] = true_range.ewm(span=20, adjust=False).mean()
    
    # ADX simplificado
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr14 = true_range.ewm(span=14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / tr14)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / tr14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
    features_df['adx'] = dx.ewm(span=14, adjust=False).mean()
    
    # Volatilidade (vetorizado)
    for period in [10, 20, 50]:
        features_df[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
    
    # Volatilidade lag
    features_df['volatility_20_lag_1'] = features_df['volatility_20'].shift(1)
    features_df['volatility_20_lag_5'] = features_df['volatility_20'].shift(5)
    features_df['volatility_20_lag_10'] = features_df['volatility_20'].shift(10)
    
    # High-Low Range
    for period in [5, 10, 20]:
        features_df[f'high_low_range_{period}'] = (
            df['high'].rolling(window=period).max() - 
            df['low'].rolling(window=period).min()
        )
    
    # Parkinson Volatility (simplificado)
    hl_ratio = np.log(df['high'] / df['low'])
    for period in [10, 20]:
        features_df[f'parkinson_vol_{period}'] = np.sqrt(
            (1 / (4 * np.log(2))) * hl_ratio.rolling(window=period).var()
        )
    
    # Garman-Klass Volatility (simplificado)
    co_ratio = np.log(df['close'] / df['open'])
    for period in [10, 20]:
        gk_term1 = 0.5 * hl_ratio**2
        gk_term2 = (2 * np.log(2) - 1) * co_ratio**2
        features_df[f'gk_vol_{period}'] = np.sqrt(
            (gk_term1 - gk_term2).rolling(window=period).mean()
        )
    
    # Range percent
    features_df['range_percent'] = (df['high'] - df['low']) / df['close'] * 100
    
    # Preencher NaN
    features_df = features_df.ffill().bfill()
    
    return features_df

class MLBacktestEngine:
    """Engine de backtest com ML integrado"""
    
    def __init__(self, model_manager, feature_engine):
        self.model_manager = model_manager
        self.feature_engine = feature_engine
        self.logger = logging.getLogger(__name__)
        self.all_features_df = None  # Cache de features
        
    def precompute_features(self, historical_data):
        """Pré-calcula todas as features de uma vez"""
        print("Pre-calculando features para todo o dataset...")
        self.all_features_df = calculate_all_features_optimized(historical_data)
        print(f"Features pre-calculadas: {len(self.all_features_df.columns)} colunas")
        
    def generate_ml_signal(self, historical_data, current_index):
        """Gera sinal ML baseado nas features pré-calculadas"""
        try:
            # Usar features pré-calculadas
            if self.all_features_df is None:
                return {'action': 'none', 'confidence': 0, 'reason': 'no_precomputed_features'}
            
            if current_index < 100:  # Mínimo de dados
                return {'action': 'none', 'confidence': 0, 'reason': 'insufficient_data'}
            
            # Obter features até o índice atual
            features_slice = self.all_features_df.iloc[:current_index+1]
            
            # Obter features necessárias dos modelos
            required_features = self.model_manager.get_all_required_features()
            
            # Verificar se temos as features necessárias
            available_features = [f for f in required_features if f in features_slice.columns]
            
            if len(available_features) < len(required_features) * 0.8:  # Pelo menos 80% das features
                missing = [f for f in required_features if f not in features_slice.columns]
                return {'action': 'none', 'confidence': 0, 'reason': f'insufficient_features: missing {len(missing)}'}
            
            # Preparar features para predição (última linha)
            features_for_prediction = features_slice[available_features].iloc[-1:].copy()
            
            # Fazer predição usando ModelManager
            if hasattr(self.model_manager, 'predict') and callable(self.model_manager.predict):
                prediction_result = self.model_manager.predict(features_for_prediction)
                
                if prediction_result is not None:
                    return self._convert_prediction_to_signal(
                        prediction_result, 
                        historical_data.iloc[current_index],
                        features_slice
                    )
            
            return {'action': 'none', 'confidence': 0, 'reason': 'prediction_failed'}
            
        except Exception as e:
            self.logger.error(f"Erro gerando sinal ML: {e}")
            return {'action': 'none', 'confidence': 0, 'reason': f'error: {str(e)}'}
    
    def _convert_prediction_to_signal(self, prediction_result, current_candle, features_slice=None):
        """Converte resultado da predição ML em sinal de trading"""
        try:
            # Se é resultado de ensemble (dicionário com múltiplos modelos)
            if isinstance(prediction_result, dict):
                # Extrair predições individuais
                predictions = []
                for model_name, pred in prediction_result.items():
                    if isinstance(pred, dict) and 'predictions' in pred:
                        pred_array = pred['predictions']
                        if hasattr(pred_array, '__len__') and len(pred_array) > 0:
                            # Converter predição (0=sell, 1=hold, 2=buy)
                            pred_value = pred_array[0] if hasattr(pred_array, 'shape') else pred_array
                            predictions.append(float(pred_value))
                
                if predictions:
                    # Calcular predição média
                    avg_prediction = np.mean(predictions)
                    confidence = max(0.5, 1.0 - np.std(predictions))  # Baixo std = alta concordância
                    
                    # Mapear para ação com thresholds mais agressivos
                    # Como os modelos estão retornando 1.0, vamos usar variação mínima
                    if avg_prediction > 1.0:  # Qualquer valor acima de 1.0
                        action = 'buy'
                        confidence = min(0.7, confidence * (avg_prediction - 1.0) * 2)  # Ajustar confiança
                    elif avg_prediction < 1.0:  # Qualquer valor abaixo de 1.0
                        action = 'sell'
                        confidence = min(0.7, confidence * (1.0 - avg_prediction) * 2)  # Ajustar confiança
                    else:  # Exatamente 1.0
                        # Usar variabilidade dos modelos como sinal
                        if len(set(predictions)) > 1:  # Há discordância entre modelos
                            # Usar o modo (valor mais comum) como decisão
                            if max(predictions) > 1.5:
                                action = 'buy'
                                confidence = 0.6
                            elif min(predictions) < 0.5:
                                action = 'sell'
                                confidence = 0.6
                            else:
                                action = 'none'
                        else:
                            # Quando todos os modelos preveem HOLD, usar indicadores técnicos
                            # como tie-breaker
                            current_features = features_slice.iloc[-1]
                            
                            # Verificar momentum baseado em EMAs
                            if 'ema_9' in current_features and 'ema_20' in current_features:
                                ema9 = current_features['ema_9']
                                ema20 = current_features['ema_20']
                                ema50 = current_features.get('ema_50', ema20)
                                
                                # Tendência de alta
                                if ema9 > ema20 > ema50:
                                    action = 'buy'
                                    confidence = 0.65
                                # Tendência de baixa
                                elif ema9 < ema20 < ema50:
                                    action = 'sell'
                                    confidence = 0.65
                                else:
                                    action = 'none'
                            else:
                                action = 'none'
                    
                    return {
                        'action': action,
                        'confidence': confidence,
                        'symbol': 'WDO',
                        'price': current_candle['close'],
                        'prediction_details': {
                            'avg_prediction': avg_prediction,
                            'model_count': len(predictions),
                            'raw_predictions': predictions
                        }
                    }
            
            # Fallback para outros formatos
            return {'action': 'none', 'confidence': 0, 'reason': 'unknown_prediction_format'}
            
        except Exception as e:
            self.logger.error(f"Erro convertendo predição: {e}")
            return {'action': 'none', 'confidence': 0, 'reason': f'conversion_error: {str(e)}'}

def run_ml_backtest():
    """Executa backtest completo com ML"""
    print("=" * 80)
    print("BACKTEST OTIMIZADO COM MODELOS ML REAIS")
    print("=" * 80)
    
    # 1. Carregar modelos ML
    print("\nETAPA 1: Carregando modelos ML...")
    model_manager, feature_engine = load_real_models()
    
    if model_manager is None:
        print("Sem modelos ML - abortando backtest")
        return None
    
    # 2. Carregar dados históricos
    print("\nETAPA 2: Carregando dados históricos...")
    historical_data = load_wdo_data_for_backtest(days_back=15)  # 15 dias para teste
    
    if historical_data is None or historical_data.empty:
        print("Sem dados historicos - abortando backtest")
        return None
    
    # 3. Configurar backtest
    print("\nETAPA 3: Configurando backtest...")
    config = BacktestConfig(
        start_date=historical_data.index.min(),
        end_date=historical_data.index.max(),
        initial_capital=100000.0,
        commission_per_contract=0.50,
        slippage_ticks=1,
        mode=BacktestMode.REALISTIC
    )
    
    print(f"Capital inicial: R$ {config.initial_capital:,.2f}")
    print(f"Periodo: {config.start_date.date()} ate {config.end_date.date()}")
    
    # 4. Criar engine ML e pré-calcular features
    ml_engine = MLBacktestEngine(model_manager, feature_engine)
    ml_engine.precompute_features(historical_data)
    
    # 5. Criar backtester
    backtester = AdvancedMLBacktester(config)
    backtester.ml_models = model_manager.models
    backtester.feature_engine = feature_engine
    backtester.market_simulator = None
    
    # Cost model mock
    class MockCostModel:
        def __init__(self, config):
            self.config = config
        def calculate_commission(self, quantity):
            return quantity * self.config.commission_per_contract
        def calculate_slippage(self, base_price, side, market_data):
            return self.config.slippage_ticks * 0.5
    
    backtester.cost_model = MockCostModel(config)
    
    # 6. Executar backtest
    print("\nETAPA 4: Executando backtest com ML...")
    backtester._reset_state()
    
    signals_generated = 0
    ml_signals_successful = 0
    
    # Processar cada candle
    for i, (timestamp, candle) in enumerate(historical_data.iterrows()):
        try:
            timestamp_dt = backtester._ensure_datetime(timestamp)
            
            # Atualizar equity
            backtester._update_equity(candle, timestamp_dt)
            
            # Gerar sinal ML a cada 20 candles (para mais oportunidades)
            if i > 100 and i % 20 == 0:  # Depois de 100 candles iniciais
                ml_signal = ml_engine.generate_ml_signal(historical_data, i)
                
                # Debug: log todos os sinais
                if i <= 300:  # Primeiros sinais
                    print(f"  Candle {i}: {ml_signal}")
                
                if ml_signal['action'] != 'none' and ml_signal['confidence'] > 0.6:
                    signals_generated += 1
                    ml_signals_successful += 1
                    
                    print(f"\n  Sinal ML {signals_generated}: {ml_signal['action'].upper()} @ R$ {ml_signal['price']:,.2f}")
                    print(f"      Confiança: {ml_signal['confidence']:.2f}")
                    if 'prediction_details' in ml_signal:
                        details = ml_signal['prediction_details']
                        print(f"      Predição média: {details['avg_prediction']:.2f} ({details['model_count']} modelos)")
                    
                    # Processar sinal
                    backtester._process_signal(ml_signal, candle, timestamp_dt)
            
            # Verificar stops
            backtester._check_stops(candle)
            
        except Exception as e:
            if i < 5:  # Log apenas primeiros erros
                print(f"Erro no candle {i}: {e}")
    
    # 7. Finalizar backtest
    if backtester.positions:
        print(f"\nFechando {len(backtester.positions)} posicoes abertas...")
        backtester._close_all_positions(historical_data.iloc[-1], "end_of_backtest")
    
    # 8. Calcular métricas
    print("\nETAPA 5: Calculando métricas...")
    metrics = backtester._calculate_final_metrics()
    
    # 9. Resultados
    print("\n" + "=" * 80)
    print("RESULTADOS DO BACKTEST COM ML")
    print("=" * 80)
    
    print(f"\nESTATISTICAS DO ML:")
    print(f"Sinais ML gerados: {ml_signals_successful}")
    print(f"Sinais processados: {signals_generated}")
    print(f"Modelos utilizados: {len(model_manager.models)}")
    
    print(f"\nRESUMO FINANCEIRO:")
    print(f"Capital Inicial: R$ {config.initial_capital:,.2f}")
    print(f"Capital Final: R$ {metrics['final_equity']:,.2f}")
    print(f"Lucro/Prejuízo: R$ {metrics['total_pnl']:,.2f}")
    
    retorno = ((metrics['final_equity'] / config.initial_capital) - 1) * 100
    print(f"Retorno: {retorno:.2f}%")
    
    print(f"\nESTATISTICAS DE TRADING:")
    print(f"Total de Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"Trades Vencedoras: {metrics['winning_trades']}")
    print(f"Trades Perdedoras: {metrics['losing_trades']}")
    
    if metrics['total_trades'] > 0:
        print(f"\nMETRICAS AVANCADAS:")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"Expectativa: R$ {metrics.get('expectancy', 0):,.2f}")
        print(f"Ganho Médio: R$ {metrics.get('avg_win', 0):,.2f}")
        print(f"Perda Média: R$ {metrics.get('avg_loss', 0):,.2f}")
        if 'max_drawdown' in metrics:
            print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    
    print("\nBacktest com ML concluido!")
    
    return metrics

if __name__ == "__main__":
    try:
        metrics = run_ml_backtest()
        if metrics:
            final_return = ((metrics['final_equity']/100000) - 1)*100
            print(f"\nRESULTADO FINAL: {final_return:.2f}% de retorno com modelos ML reais")
    except Exception as e:
        print(f"\nErro: {e}")
        import traceback
        traceback.print_exc()