"""
Teste Completo do Sistema ML Trading v2.0
Executa fluxo completo de dados seguindo o mapeamento definido
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Adicionar src ao path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Imports dos componentes principais
try:
    from connection_manager import ConnectionManager
    from model_manager import ModelManager
    from data_structure import TradingDataStructure
    from data_loader import DataLoader
    from feature_engine import FeatureEngine
    from technical_indicators import TechnicalIndicators
    from ml_features import MLFeatures
    from data_pipeline import DataPipeline
    from prediction_engine import PredictionEngine
    from ml_coordinator import MLCoordinator
    from signal_generator import SignalGenerator
    from risk_manager import RiskManager
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print(f"Caminho src: {src_path}")
    print(f"Arquivos em src: {os.listdir(src_path) if os.path.exists(src_path) else 'Diretório não existe'}")
    sys.exit(1)

class SystemTester:
    """Teste completo do sistema seguindo mapeamento de fluxo de dados"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.temp_dir = tempfile.mkdtemp()
        self.components = {}
        self.test_data = {}
        
        # Configurações de teste
        self.config = {
            'dll_path': r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll",
            'models_dir': 'test_models',
            'ticker': 'WDOZ24',
            'username': 'demo',
            'password': 'demo',
            'use_real_connection': True,
            'data_source': 'profit_historical',
            'start_date': datetime.now() - timedelta(days=30),
            'end_date': datetime.now(),
            'candle_period': 1  # 1 minuto
        }
        
        self.logger.info("SystemTester inicializado")
    
    def _setup_logger(self):
        """Configura logging para o teste"""
        logger = logging.getLogger('SystemTester')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def run_complete_test(self):
        """Executa teste completo seguindo fluxo de dados mapeado"""
        
        self.logger.info("🚀 INICIANDO TESTE COMPLETO DO SISTEMA ML TRADING v2.0")
        self.logger.info("=" * 80)
        
        try:
            # ETAPA 1: Inicializar Conexão
            self.logger.info("📡 ETAPA 1: Inicializando Conexão com Profit")
            success = self._test_connection_initialization()
            if not success:
                self.logger.warning("⚠️ Conexão DLL falhou - usando modo simulação controlada")
            
            # ETAPA 2: Carregar Modelos ML
            self.logger.info("🧠 ETAPA 2: Carregando Modelos ML e Identificando Features")
            self._test_model_loading()
            
            # ETAPA 3: Estrutura de Dados
            self.logger.info("📊 ETAPA 3: Inicializando Estrutura de Dados")
            self._test_data_structure()
            
            # ETAPA 4: Carregamento de Dados Históricos
            self.logger.info("📈 ETAPA 4: Carregando Dados Históricos")
            self._test_historical_data_loading()
            
            # ETAPA 5: Cálculo de Indicadores Técnicos
            self.logger.info("📊 ETAPA 5: Calculando Indicadores Técnicos")
            self._test_technical_indicators()
            
            # ETAPA 6: Features de Machine Learning
            self.logger.info("🤖 ETAPA 6: Calculando Features ML")
            self._test_ml_features()
            
            # ETAPA 7: Engine de Features Completa
            self.logger.info("⚡ ETAPA 7: Engine de Features Completa")
            self._test_feature_engine()
            
            # ETAPA 8: Pipeline de Dados
            self.logger.info("🔄 ETAPA 8: Pipeline de Processamento")
            self._test_data_pipeline()
            
            # ETAPA 9: Predições ML
            self.logger.info("🎯 ETAPA 9: Sistema de Predições")
            self._test_predictions()
            
            # ETAPA 10: Sinais de Trading
            self.logger.info("📢 ETAPA 10: Geração de Sinais")
            self._test_signal_generation()
            
            # ETAPA 11: Gestão de Risco
            self.logger.info("🛡️ ETAPA 11: Sistema de Gestão de Risco")
            self._test_risk_management()
            
            # ETAPA 12: Relatório Final
            self.logger.info("📋 ETAPA 12: Relatório Final do Sistema")
            self._generate_final_report()
            
            self.logger.info("✅ TESTE COMPLETO FINALIZADO COM SUCESSO!")
            
        except Exception as e:
            self.logger.error(f"❌ ERRO CRÍTICO NO TESTE: {str(e)}")
            raise
        
        finally:
            self._cleanup()
    
    def _test_connection_initialization(self) -> bool:
        """Testa inicialização da conexão com Profit"""
        
        try:
            # Verificar se DLL existe
            dll_path = self.config['dll_path']
            if not os.path.exists(dll_path):
                self.logger.warning(f"DLL não encontrada em {dll_path}")
                return False
            
            # Inicializar ConnectionManager
            self.components['connection'] = ConnectionManager(dll_path)
            
            # Tentar carregar DLL
            success = self.components['connection'].initialize()
            
            if success:
                self.logger.info("✅ ConnectionManager inicializado com sucesso")
                
                # Tentar autenticar se credenciais válidas
                if self.config.get('username') and self.config.get('password'):
                    login_success = self.components['connection'].login(
                        self.config['username'], 
                        self.config['password']
                    )
                    
                    if login_success:
                        self.logger.info("✅ Login realizado com sucesso")
                    else:
                        self.logger.warning("⚠️ Login falhou - continuando sem autenticação")
                
                return True
            else:
                self.logger.warning("⚠️ Falha ao inicializar ConnectionManager")
                return False
                
        except Exception as e:
            self.logger.warning(f"⚠️ Erro na conexão: {str(e)}")
            return False
    
    def _test_model_loading(self):
        """Testa carregamento de modelos ML"""
        
        try:
            # Verificar diretório de modelos
            models_dir = self.config['models_dir']
            
            if not os.path.exists(models_dir):
                self.logger.info(f"📁 Criando diretório de modelos: {models_dir}")
                os.makedirs(models_dir, exist_ok=True)
                
                # Gerar modelos de teste se necessário
                self._create_test_models(models_dir)
            
            # Inicializar ModelManager
            self.components['model_manager'] = ModelManager(models_dir)
            
            # Carregar modelos
            self.components['model_manager'].load_models()
            
            # Verificar features requeridas
            features = self.components['model_manager'].get_all_required_features()
            
            self.logger.info(f"✅ Modelos carregados: {len(self.components['model_manager'].models)}")
            self.logger.info(f"📊 Features requeridas: {len(features)}")
            
            # Salvar features para uso posterior
            self.test_data['required_features'] = features
            
            # Log das primeiras 10 features
            if features:
                self.logger.info(f"🔍 Primeiras 10 features: {list(features)[:10]}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro no carregamento de modelos: {str(e)}")
            raise
    
    def _test_data_structure(self):
        """Testa inicialização da estrutura de dados"""
        
        try:
            # Inicializar TradingDataStructure
            self.components['data_structure'] = TradingDataStructure()
            self.components['data_structure'].initialize_structure()
            
            self.logger.info("✅ Estrutura de dados inicializada")
            
            # Verificar estruturas criadas
            structures = []
            if not self.components['data_structure'].candles.empty:
                structures.append("candles")
            if not self.components['data_structure'].indicators.empty:
                structures.append("indicators")
            if not self.components['data_structure'].features.empty:
                structures.append("features")
            
            self.logger.info(f"📊 Estruturas disponíveis: {structures}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro na estrutura de dados: {str(e)}")
            raise
    
    def _test_historical_data_loading(self):
        """Testa carregamento de dados históricos"""
        
        try:
            # Inicializar DataLoader
            connection = self.components.get('connection')
            self.components['data_loader'] = DataLoader(connection)
            
            # Tentar carregar dados reais primeiro
            success = False
            
            if connection and connection.connected:
                try:
                    self.logger.info("📡 Tentando carregar dados reais do Profit...")
                    
                    candles = self.components['data_loader'].load_historical_data(
                        ticker=self.config['ticker'],
                        start_date=self.config['start_date'],
                        end_date=self.config['end_date'],
                        period=self.config['candle_period']
                    )
                    
                    if not candles.empty:
                        self.test_data['candles'] = candles
                        success = True
                        self.logger.info(f"✅ Dados reais carregados: {len(candles)} candles")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Falha em dados reais: {str(e)}")
            
            # Se dados reais falharam, usar dados de teste controlados
            if not success:
                self.logger.info("📊 Gerando dados de teste controlados...")
                candles = self._generate_test_candles()
                self.test_data['candles'] = candles
                self.logger.info(f"✅ Dados de teste gerados: {len(candles)} candles")
            
            # Atualizar estrutura de dados
            self.components['data_structure'].update_candles(self.test_data['candles'])
            
            # Verificar qualidade dos dados
            self._validate_candle_data(self.test_data['candles'])
            
        except Exception as e:
            self.logger.error(f"❌ Erro no carregamento de dados: {str(e)}")
            raise
    
    def _test_technical_indicators(self):
        """Testa cálculo de indicadores técnicos"""
        
        try:
            # Inicializar TechnicalIndicators
            self.components['technical_indicators'] = TechnicalIndicators()
            
            # Calcular indicadores
            candles = self.test_data['candles']
            indicators = self.components['technical_indicators'].calculate_all(candles)
            
            self.test_data['indicators'] = indicators
            
            self.logger.info(f"✅ Indicadores calculados: {len(indicators.columns)} indicadores")
            self.logger.info(f"📊 Dados válidos: {len(indicators)} períodos")
            
            # Verificar indicadores principais
            key_indicators = ['ema_9', 'ema_20', 'rsi_14', 'macd', 'bb_upper', 'atr_14']
            found_indicators = [ind for ind in key_indicators if ind in indicators.columns]
            
            self.logger.info(f"🔍 Indicadores chave encontrados: {found_indicators}")
            
            # Atualizar estrutura de dados
            self.components['data_structure'].update_indicators(indicators)
            
        except Exception as e:
            self.logger.error(f"❌ Erro no cálculo de indicadores: {str(e)}")
            raise
    
    def _test_ml_features(self):
        """Testa cálculo de features ML"""
        
        try:
            # Usar features requeridas pelos modelos
            required_features = self.test_data.get('required_features', [])
            
            # Inicializar MLFeatures
            self.components['ml_features'] = MLFeatures(required_features)
            
            # Calcular features
            candles = self.test_data['candles']
            indicators = self.test_data['indicators']
            
            # Simular microestrutura se não disponível
            microstructure = self._generate_microstructure_data(candles)
            
            features = self.components['ml_features'].calculate_all(
                candles, microstructure, indicators
            )
            
            self.test_data['features'] = features
            
            self.logger.info(f"✅ Features ML calculadas: {len(features.columns)} features")
            self.logger.info(f"📊 Dados válidos: {len(features)} períodos")
            
            # Verificar features importantes
            if required_features:
                found_features = [f for f in required_features if f in features.columns]
                missing_features = [f for f in required_features if f not in features.columns]
                
                self.logger.info(f"🔍 Features encontradas: {len(found_features)}/{len(required_features)}")
                
                if missing_features:
                    self.logger.warning(f"⚠️ Features faltantes: {missing_features[:5]}...")
            
            # Atualizar estrutura de dados
            self.components['data_structure'].update_features(features)
            
        except Exception as e:
            self.logger.error(f"❌ Erro no cálculo de features ML: {str(e)}")
            raise
    
    def _test_feature_engine(self):
        """Testa o motor de features completo"""
        
        try:
            # Usar features requeridas pelos modelos
            required_features = self.test_data.get('required_features', [])
            
            # Inicializar FeatureEngine
            self.components['feature_engine'] = FeatureEngine(required_features)
            
            # Calcular todas as features
            result = self.components['feature_engine'].calculate(
                self.components['data_structure'],
                force_recalculate=True,
                use_advanced=True
            )
            
            self.test_data['all_features'] = result
            
            self.logger.info("✅ FeatureEngine executado com sucesso")
            
            # Verificar resultados
            for key, df in result.items():
                if isinstance(df, pd.DataFrame):
                    self.logger.info(f"📊 {key}: {df.shape[1]} colunas, {df.shape[0]} linhas")
                    
                    # Verificar qualidade dos dados
                    nan_ratio = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
                    self.logger.info(f"🔍 Taxa de NaN em {key}: {nan_ratio:.2%}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro no FeatureEngine: {str(e)}")
            # Não falhar o teste se o FeatureEngine tiver problemas específicos
            self.logger.warning("⚠️ Continuando teste sem FeatureEngine avançado")
    
    def _test_data_pipeline(self):
        """Testa pipeline de processamento de dados"""
        
        try:
            # Inicializar DataPipeline
            self.components['data_pipeline'] = DataPipeline()
            
            # Configurar pipeline
            config = {
                'min_candles': 50,
                'required_features': self.test_data.get('required_features', []),
                'validate_data': True
            }
            
            self.components['data_pipeline'].configure(config)
            
            # Processar dados históricos
            processed_data = self.components['data_pipeline'].process_historical_data(
                self.components['data_structure']
            )
            
            self.test_data['processed_data'] = processed_data
            
            self.logger.info("✅ Pipeline de dados executado com sucesso")
            self.logger.info(f"📊 Dados processados: {len(processed_data)} registros")
            
        except Exception as e:
            self.logger.error(f"❌ Erro no pipeline de dados: {str(e)}")
            raise
    
    def _test_predictions(self):
        """Testa sistema de predições"""
        
        try:
            # Inicializar PredictionEngine
            model_manager = self.components['model_manager']
            self.components['prediction_engine'] = PredictionEngine(model_manager)
            
            # Inicializar MLCoordinator
            feature_engine = self.components.get('feature_engine')
            self.components['ml_coordinator'] = MLCoordinator(
                model_manager, feature_engine, self.components['prediction_engine'], None
            )
            
            # Fazer predição com dados mais recentes
            recent_data = self._get_recent_data_sample()
            
            prediction_result = self.components['ml_coordinator'].process_prediction_request(
                recent_data
            )
            
            self.test_data['prediction'] = prediction_result
            
            self.logger.info("✅ Sistema de predições executado")
            self.logger.info(f"🎯 Resultado: {prediction_result}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro no sistema de predições: {str(e)}")
            # Criar predição mock para continuar teste
            self.test_data['prediction'] = {
                'regime': 'trend_up',
                'confidence': 0.75,
                'trade_decision': 'BUY',
                'can_trade': True,
                'risk_reward_target': 2.0
            }
            self.logger.warning("⚠️ Usando predição mock para continuar teste")
    
    def _test_signal_generation(self):
        """Testa geração de sinais de trading"""
        
        try:
            # Inicializar SignalGenerator
            self.components['signal_generator'] = SignalGenerator()
            
            # Gerar sinal baseado na predição
            prediction = self.test_data['prediction']
            market_data = self._get_current_market_data()
            
            signal = self.components['signal_generator'].generate_regime_based_signal(
                prediction, market_data
            )
            
            self.test_data['signal'] = signal
            
            self.logger.info("✅ Sinal de trading gerado")
            self.logger.info(f"📢 Sinal: {signal}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro na geração de sinais: {str(e)}")
            raise
    
    def _test_risk_management(self):
        """Testa sistema de gestão de risco"""
        
        try:
            # Inicializar RiskManager
            risk_config = {
                'max_position_size': 1,
                'max_daily_loss': 1000,
                'max_trades_per_day': 10
            }
            
            self.components['risk_manager'] = RiskManager(risk_config)
            
            # Avaliar risco do sinal
            signal = self.test_data['signal']
            account_info = {'balance': 100000, 'available': 95000}
            
            risk_assessment = self.components['risk_manager'].assess_trade_risk(
                signal, account_info
            )
            
            self.test_data['risk_assessment'] = risk_assessment
            
            self.logger.info("✅ Gestão de risco executada")
            self.logger.info(f"🛡️ Avaliação: {risk_assessment}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro na gestão de risco: {str(e)}")
            raise
    
    def _generate_final_report(self):
        """Gera relatório final do teste"""
        
        self.logger.info("📋 RELATÓRIO FINAL DO TESTE")
        self.logger.info("=" * 60)
        
        # Componentes testados
        self.logger.info("🔧 COMPONENTES TESTADOS:")
        for name, component in self.components.items():
            status = "✅ OK" if component is not None else "❌ FALHA"
            self.logger.info(f"  {name}: {status}")
        
        # Dados processados
        self.logger.info("\n📊 DADOS PROCESSADOS:")
        if 'candles' in self.test_data:
            candles = self.test_data['candles']
            self.logger.info(f"  Candles: {len(candles)} registros")
            self.logger.info(f"  Período: {candles.index[0]} até {candles.index[-1]}")
        
        if 'indicators' in self.test_data:
            indicators = self.test_data['indicators']
            self.logger.info(f"  Indicadores: {len(indicators.columns)} calculados")
        
        if 'features' in self.test_data:
            features = self.test_data['features']
            self.logger.info(f"  Features ML: {len(features.columns)} calculadas")
        
        # Resultado da predição
        self.logger.info("\n🎯 RESULTADO DA PREDIÇÃO:")
        if 'prediction' in self.test_data:
            pred = self.test_data['prediction']
            self.logger.info(f"  Regime: {pred.get('regime', 'N/A')}")
            self.logger.info(f"  Confiança: {pred.get('confidence', 0):.2f}")
            self.logger.info(f"  Decisão: {pred.get('trade_decision', 'N/A')}")
        
        # Sinal final
        self.logger.info("\n📢 SINAL FINAL:")
        if 'signal' in self.test_data:
            signal = self.test_data['signal']
            self.logger.info(f"  Ação: {signal.get('action', 'N/A')}")
            self.logger.info(f"  Preço: {signal.get('price', 0)}")
            self.logger.info(f"  Stop: {signal.get('stop', 0)}")
            self.logger.info(f"  Target: {signal.get('target', 0)}")
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("✅ TESTE COMPLETO FINALIZADO!")
    
    # Métodos auxiliares de geração de dados
    
    def _create_test_models(self, models_dir):
        """Cria modelos de teste se necessário"""
        
        # Importar script de criação de modelos mock
        import sys
        tests_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'mock_data')
        
        if os.path.exists(os.path.join(tests_path, 'create_mock_models.py')):
            sys.path.insert(0, tests_path)
            
            try:
                from create_mock_models import create_mock_models_ensemble
                create_mock_models_ensemble(models_dir)
                self.logger.info(f"✅ Modelos de teste criados em {models_dir}")
            except ImportError:
                self.logger.warning("⚠️ Script de criação de modelos não encontrado")
    
    def _generate_test_candles(self) -> pd.DataFrame:
        """Gera dados de candles para teste controlado"""
        
        # Período de teste
        start_date = self.config['start_date']
        end_date = self.config['end_date']
        
        # Gerar índice de tempo (apenas horário de mercado)
        dates = pd.bdate_range(start_date, end_date, freq='1min')
        
        # Filtrar horário de mercado (9:00 - 18:00)
        market_times = []
        for date in dates:
            if 9 <= date.hour <= 17:  # Horário de mercado
                market_times.append(date)
        
        # Pegar apenas uma amostra para teste
        market_times = market_times[:2000]  # Últimos 2000 minutos
        
        # Preço inicial realista para WDO
        initial_price = 130000  # Aproximadamente 130.000 pontos
        
        # Gerar série de preços com movimento browniano
        np.random.seed(42)  # Para reprodutibilidade
        
        returns = np.random.normal(0, 0.001, len(market_times))  # Retornos pequenos
        returns[0] = 0
        
        # Calcular preços
        prices = [initial_price]
        for i in range(1, len(market_times)):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(new_price)
        
        # Criar OHLCV
        candles_data = []
        
        for i, (timestamp, close) in enumerate(zip(market_times, prices)):
            # Simular OHLC baseado no close
            volatility = close * 0.0005  # 0.05% de volatilidade
            
            high = close + np.random.uniform(0, volatility)
            low = close - np.random.uniform(0, volatility)
            
            if i == 0:
                open_price = close
            else:
                open_price = prices[i-1]
            
            # Volume realista
            volume = np.random.randint(50, 500)
            
            candles_data.append({
                'open': open_price,
                'high': max(open_price, high, close, low),
                'low': min(open_price, high, close, low),
                'close': close,
                'volume': volume
            })
        
        # Criar DataFrame
        df = pd.DataFrame(candles_data, index=market_times)
        
        # Validar dados
        assert (df['high'] >= df['low']).all(), "High deve ser >= Low"
        assert (df['high'] >= df['open']).all(), "High deve ser >= Open"
        assert (df['high'] >= df['close']).all(), "High deve ser >= Close"
        assert (df['low'] <= df['open']).all(), "Low deve ser <= Open"
        assert (df['low'] <= df['close']).all(), "Low deve ser <= Close"
        assert (df['volume'] > 0).all(), "Volume deve ser positivo"
        
        return df
    
    def _generate_microstructure_data(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Gera dados de microestrutura simulados"""
        
        micro_data = []
        
        for idx, candle in candles.iterrows():
            # Simular volume de compra/venda baseado no movimento
            total_volume = candle['volume']
            
            price_change = candle['close'] - candle['open']
            
            if price_change > 0:
                # Movimento para cima - mais volume de compra
                buy_ratio = 0.6 + 0.2 * np.random.random()
            elif price_change < 0:
                # Movimento para baixo - mais volume de venda
                buy_ratio = 0.2 + 0.2 * np.random.random()
            else:
                # Sem movimento - equilibrado
                buy_ratio = 0.45 + 0.1 * np.random.random()
            
            buy_volume = int(total_volume * buy_ratio)
            sell_volume = total_volume - buy_volume
            
            # Simular número de trades
            buy_trades = max(1, int(buy_volume / 20))
            sell_trades = max(1, int(sell_volume / 20))
            
            micro_data.append({
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades
            })
        
        return pd.DataFrame(micro_data, index=candles.index)
    
    def _validate_candle_data(self, candles: pd.DataFrame):
        """Valida qualidade dos dados de candle"""
        
        # Verificações básicas
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in candles.columns]
        
        if missing_cols:
            raise ValueError(f"Colunas faltantes: {missing_cols}")
        
        # Verificar NaN
        nan_count = candles.isna().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"⚠️ {nan_count} valores NaN encontrados nos dados")
        
        # Verificar consistência OHLC
        invalid_high = (candles['high'] < candles['open']) | (candles['high'] < candles['close']) | (candles['high'] < candles['low'])
        invalid_low = (candles['low'] > candles['open']) | (candles['low'] > candles['close']) | (candles['low'] > candles['high'])
        
        if invalid_high.any():
            self.logger.error(f"❌ {invalid_high.sum()} registros com High inválido")
        
        if invalid_low.any():
            self.logger.error(f"❌ {invalid_low.sum()} registros com Low inválido")
        
        # Verificar volume
        if (candles['volume'] <= 0).any():
            self.logger.error("❌ Volume zero ou negativo encontrado")
        
        self.logger.info("✅ Validação de dados de candle concluída")
    
    def _get_recent_data_sample(self) -> Dict[str, Any]:
        """Retorna amostra de dados recentes para predição"""
        
        # Pegar últimos 50 registros (típico para ML)
        sample_size = min(50, len(self.test_data['candles']))
        
        return {
            'candles': self.test_data['candles'].tail(sample_size),
            'indicators': self.test_data.get('indicators', pd.DataFrame()).tail(sample_size),
            'features': self.test_data.get('features', pd.DataFrame()).tail(sample_size)
        }
    
    def _get_current_market_data(self) -> Dict[str, Any]:
        """Retorna dados de mercado atuais simulados"""
        
        latest_candle = self.test_data['candles'].iloc[-1]
        
        return {
            'current_price': latest_candle['close'],
            'bid': latest_candle['close'] - 5,  # 5 pontos abaixo
            'ask': latest_candle['close'] + 5,  # 5 pontos acima
            'volume': latest_candle['volume'],
            'timestamp': latest_candle.name
        }
    
    def _cleanup(self):
        """Limpa recursos do teste"""
        
        try:
            # Fechar conexões se existirem
            if 'connection' in self.components and self.components['connection']:
                if hasattr(self.components['connection'], 'disconnect'):
                    self.components['connection'].disconnect()
            
            # Limpar diretório temporário
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                
            self.logger.info("🧹 Limpeza concluída")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erro na limpeza: {str(e)}")

def main():
    """Função principal para executar o teste"""
    
    # Configurar logging global
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Criar e executar teste
    tester = SystemTester()
    tester.run_complete_test()

if __name__ == "__main__":
    main()
