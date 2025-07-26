#!/usr/bin/env python3
"""
Backtest completo com modelos ML reais integrados
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

class MLBacktestEngine:
    """Engine de backtest com ML integrado"""
    
    def __init__(self, model_manager, feature_engine):
        self.model_manager = model_manager
        self.feature_engine = feature_engine
        self.logger = logging.getLogger(__name__)
        
    def generate_ml_signal(self, historical_data, current_index):
        """Gera sinal ML baseado nos dados históricos até o índice atual"""
        try:
            # Preparar dados até o índice atual (sem look-ahead bias)
            data_slice = historical_data.iloc[:current_index+1]
            
            if len(data_slice) < 100:  # Mínimo de dados para features
                return {'action': 'none', 'confidence': 0, 'reason': 'insufficient_data'}
            
            # Preparar estrutura de dados
            data_structure = TradingDataStructure()
            data_structure.initialize_structure()
            data_structure.candles = data_slice.copy()
            
            # Tentar diferentes métodos de extração de features
            features_result = None
            
            # Método 1: Usar o pipeline completo ML via FeatureEngine
            if hasattr(self.feature_engine, 'ml_features') and hasattr(self.feature_engine, 'technical_indicators'):
                try:
                    # Primeiro calcular indicadores técnicos
                    tech_indicators = self.feature_engine.technical_indicators.create_all_indicators(
                        candles_df=data_structure.candles
                    )
                    
                    # Depois calcular features ML completas  
                    ml_features_result = self.feature_engine.ml_features.calculate_all_features(
                        candles_df=data_structure.candles,
                        indicators_df=tech_indicators,
                        microstructure_df=data_structure.candles[['buy_volume', 'sell_volume', 'trades']]
                    )
                    
                    if isinstance(ml_features_result, dict) and 'model_ready' in ml_features_result:
                        features_result = ml_features_result['model_ready']
                    else:
                        features_result = ml_features_result
                        
                    if features_result is not None:
                        self.logger.info(f"Pipeline completo: {len(features_result.columns)} features geradas")
                        
                except Exception as e:
                    self.logger.warning(f"Pipeline completo falhou: {e}")
            
            # Método 2: get_features_for_prediction (fallback)
            if features_result is None and hasattr(self.feature_engine, 'get_features_for_prediction'):
                try:
                    features_result = self.feature_engine.get_features_for_prediction(
                        data=data_structure
                    )
                except Exception as e:
                    self.logger.warning(f"get_features_for_prediction falhou: {e}")
            
            # Método 3: create_features_separated (alternativo)
            if features_result is None and hasattr(self.feature_engine, 'create_features_separated'):
                try:
                    microstructure_df = data_structure.candles[['buy_volume', 'sell_volume', 'trades']]
                    indicators_df = pd.DataFrame()  # Vazio por enquanto
                    
                    features_result = self.feature_engine.create_features_separated(
                        candles_df=data_structure.candles,
                        microstructure_df=microstructure_df,
                        indicators_df=indicators_df
                    )
                except Exception as e:
                    self.logger.warning(f"create_features_separated falhou: {e}")
            
            # Tratar resultado que pode ser dict ou DataFrame
            if features_result is None:
                return {'action': 'none', 'confidence': 0, 'reason': 'no_features'}
            
            # Se é um dict, extrair o DataFrame
            if isinstance(features_result, dict):
                if 'model_ready' in features_result:
                    features_df = features_result['model_ready']
                elif 'features' in features_result:
                    features_df = features_result['features']
                elif 'all' in features_result:
                    features_df = features_result['all']
                else:
                    # Tentar usar qualquer DataFrame no dict
                    for key, value in features_result.items():
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            features_df = value
                            break
                    else:
                        return {'action': 'none', 'confidence': 0, 'reason': 'no_dataframe_in_dict'}
            else:
                features_df = features_result
            
            if features_df is None or (hasattr(features_df, 'empty') and features_df.empty):
                return {'action': 'none', 'confidence': 0, 'reason': 'empty_features'}
            
            # Obter features necessárias dos modelos
            required_features = self.model_manager.get_all_required_features()
            
            # Verificar se temos as features necessárias
            available_features = [f for f in required_features if f in features_df.columns]
            
            if len(available_features) < len(required_features) * 0.8:  # Pelo menos 80% das features
                return {'action': 'none', 'confidence': 0, 'reason': 'insufficient_features'}
            
            # Preparar features para predição (última linha)
            features_for_prediction = features_df[available_features].iloc[-1:].copy()
            
            # Fazer predição usando ModelManager
            if hasattr(self.model_manager, 'predict') and callable(self.model_manager.predict):
                prediction_result = self.model_manager.predict(features_for_prediction)
                
                if prediction_result is not None:
                    return self._convert_prediction_to_signal(prediction_result, data_slice.iloc[-1])
            
            return {'action': 'none', 'confidence': 0, 'reason': 'prediction_failed'}
            
        except Exception as e:
            self.logger.error(f"Erro gerando sinal ML: {e}")
            import traceback
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Debug adicional
            self.logger.error(f"FeatureEngine type: {type(self.feature_engine)}")
            all_methods = [m for m in dir(self.feature_engine) if not m.startswith('_') and callable(getattr(self.feature_engine, m))]
            self.logger.error(f"FeatureEngine métodos públicos: {all_methods[:10]}...")  # Primeiros 10
            feature_methods = [m for m in dir(self.feature_engine) if any(word in m.lower() for word in ['feature', 'extract', 'create', 'get'])]
            self.logger.error(f"Métodos relacionados a features: {feature_methods}")
            
            return {'action': 'none', 'confidence': 0, 'reason': f'error: {str(e)}'}
    
    def _convert_prediction_to_signal(self, prediction_result, current_candle):
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
                    confidence = np.std(predictions)  # Baixo std = alta concordância
                    confidence = max(0.5, 1.0 - confidence)  # Converter para confiança
                    
                    # Mapear para ação (thresholds mais sensíveis)
                    if avg_prediction >= 1.5:  # Próximo de 2 (buy)
                        action = 'buy'
                        confidence = confidence
                    elif avg_prediction <= 0.5:  # Próximo de 0 (sell)
                        action = 'sell'
                        confidence = confidence
                    else:  # Próximo de 1 (hold)
                        action = 'none'
                        confidence = confidence
                    
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
    print("BACKTEST COMPLETO COM MODELOS ML REAIS")
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
    
    # 4. Criar engine ML
    ml_engine = MLBacktestEngine(model_manager, feature_engine)
    
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
            
            # Gerar sinal ML a cada 50 candles (para mais oportunidades)
            if i > 100 and i % 50 == 0:  # Depois de 100 candles iniciais
                ml_signal = ml_engine.generate_ml_signal(historical_data, i)
                
                # Debug: log todos os sinais para diagnosticar o problema
                if i < 500:  # Log apenas primeiros casos para debug
                    print(f"  DEBUG candle {i}: {ml_signal}")
                
                if ml_signal['action'] != 'none' and ml_signal['confidence'] > 0.5:
                    signals_generated += 1
                    ml_signals_successful += 1
                    
                    print(f"  Sinal ML {signals_generated}: {ml_signal['action'].upper()} @ R$ {ml_signal['price']:,.2f}")
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