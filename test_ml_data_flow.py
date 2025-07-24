"""
Script de Teste do Fluxo de Dados ML Trading v2.0
Testa o fluxo completo desde carregamento até predição
Baseado na arquitetura real do sistema
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Imports dos componentes do sistema
from src.connection_manager import ConnectionManager
from src.model_manager import ModelManager
from src.data_structure import TradingDataStructure
from src.data_loader import DataLoader
from src.feature_engine import FeatureEngine
from src.prediction_engine import PredictionEngine
from src.ml_coordinator import MLCoordinator
from src.signal_generator import SignalGenerator
from src.risk_manager import RiskManager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MLDataFlowTest')

class MLDataFlowTester:
    """Testa o fluxo completo de dados do sistema ML Trading"""
    
    def __init__(self):
        self.logger = logger
        self.components_ready = False
        
        # Componentes principais
        self.connection = None
        self.model_manager = None
        self.data_structure = None
        self.data_loader = None
        self.feature_engine = None
        self.prediction_engine = None
        self.ml_coordinator = None
        self.signal_generator = None
        self.risk_manager = None
        
        # Estado do sistema
        self.models_loaded = False
        self.features_required = set()
        self.data_loaded = False
        self.features_calculated = False
        
    def initialize_system(self):
        """Inicializa todos os componentes do sistema"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("🚀 INICIALIZANDO SISTEMA ML TRADING v2.0")
            self.logger.info("="*80)
            
            # 1. Connection Manager
            self.logger.info("\n📡 [1/9] Inicializando Connection Manager...")
            dll_path = os.getenv('PROFIT_DLL_PATH', r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll")
            self.connection = ConnectionManager(dll_path)
            
            # 2. Model Manager
            self.logger.info("🤖 [2/9] Inicializando Model Manager...")
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            self.model_manager = ModelManager(models_dir)
            
            # 3. Data Structure
            self.logger.info("📊 [3/9] Inicializando Data Structure...")
            self.data_structure = TradingDataStructure()
            self.data_structure.initialize_structure()
            
            # 4. Data Loader
            self.logger.info("💾 [4/9] Inicializando Data Loader...")
            self.data_loader = DataLoader()
            
            # 5. Feature Engine
            self.logger.info("⚙️ [5/9] Inicializando Feature Engine...")
            self.feature_engine = FeatureEngine(self.data_structure)
            
            # 6. Prediction Engine
            self.logger.info("🎯 [6/9] Inicializando Prediction Engine...")
            self.prediction_engine = PredictionEngine(self.model_manager)
            
            # 7. ML Coordinator
            self.logger.info("🧠 [7/9] Inicializando ML Coordinator...")
            self.ml_coordinator = MLCoordinator(
                self.model_manager,
                self.feature_engine,
                self.prediction_engine,
                None  # regime_trainer opcional
            )
            
            # 8. Signal Generator
            self.logger.info("📈 [8/9] Inicializando Signal Generator...")
            signal_config = {
                'direction_threshold': 0.3,
                'magnitude_threshold': 0.0001,
                'confidence_threshold': 0.6,
                'risk_per_trade': 0.02,
                'max_positions': 1
            }
            self.signal_generator = SignalGenerator(signal_config)
            
            # 9. Risk Manager
            self.logger.info("🛡️ [9/9] Inicializando Risk Manager...")
            config = {
                'max_daily_loss': 0.05,
                'max_positions': 1,
                'risk_per_trade': 0.02
            }
            self.risk_manager = RiskManager(config)
            
            self.components_ready = True
            self.logger.info("\n✅ Sistema inicializado com sucesso!")
            
        except Exception as e:
            self.logger.error(f"❌ Erro na inicialização: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def step1_load_models(self):
        """ETAPA 1: Carregar modelos e identificar features necessárias"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("📦 ETAPA 1: CARREGAMENTO DE MODELOS")
            self.logger.info("="*80)
            
            # Carregar modelos
            self.logger.info("\n🔄 Carregando modelos ML...")
            self.model_manager.load_models()
            
            # Obter informações dos modelos
            if not hasattr(self.model_manager, 'models') or not self.model_manager.models:
                self.logger.warning("⚠️ Nenhum modelo encontrado no diretório!")
                self.logger.info("\n📝 Criando modelos de exemplo para teste...")
                # Aqui poderíamos criar modelos de exemplo
                return False
            
            self.logger.info(f"\n✅ {len(self.model_manager.models)} modelos carregados:")
            
            # Analisar cada modelo
            for idx, (name, model_info) in enumerate(self.model_manager.models.items(), 1):
                self.logger.info(f"\n  [{idx}] {name}:")
                
                # Obter tipo do modelo
                model_type = type(model_info['model']).__name__ if 'model' in model_info else 'Unknown'
                self.logger.info(f"      📊 Tipo: {model_type}")
                
                # Obter features
                if 'features' in model_info:
                    self.logger.info(f"      🔢 Features: {len(model_info['features'])}")
                
                # Obter metadata se disponível
                if 'metadata' in model_info and model_info['metadata']:
                    if 'best_score' in model_info['metadata']:
                        self.logger.info(f"      🎯 Score: {model_info['metadata']['best_score']:.4f}")
                    if 'training_date' in model_info['metadata']:
                        self.logger.info(f"      📅 Treinado em: {model_info['metadata']['training_date']}")
            
            # Coletar todas as features necessárias
            self.features_required = self.model_manager.get_all_required_features()
            
            self.logger.info(f"\n📋 Features únicas necessárias: {len(self.features_required)}")
            
            # Mostrar categorias de features
            feature_categories = {
                'Básicas': ['open', 'high', 'low', 'close', 'volume'],
                'EMAs': [f for f in self.features_required if f.startswith('ema_')],
                'Momentum': [f for f in self.features_required if 'momentum' in f],
                'Volume': [f for f in self.features_required if 'volume' in f and not f in ['volume']],
                'Volatilidade': [f for f in self.features_required if 'volatility' in f],
                'Indicadores': ['rsi', 'macd', 'atr', 'adx', 'bb_width']
            }
            
            for category, features in feature_categories.items():
                matching = [f for f in features if f in self.features_required]
                if matching:
                    self.logger.info(f"\n  📊 {category}: {len(matching)} features")
                    for feat in matching[:3]:
                        self.logger.info(f"     - {feat}")
                    if len(matching) > 3:
                        self.logger.info(f"     ... e mais {len(matching)-3}")
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar modelos: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step2_load_data(self, symbol='WDO', days_back=5):
        """ETAPA 2: Carregar dados históricos e configurar tempo real"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("📊 ETAPA 2: CARREGAMENTO DE DADOS")
            self.logger.info("="*80)
            
            # Tentar conectar ao ProfitDLL
            self.logger.info(f"\n🔌 Tentando conectar ao ProfitDLL...")
            
            connected_to_profit = False
            
            try:
                # Inicializar DLL
                init_result = self.connection.initialize()
                if init_result == 1:
                    self.logger.info("✅ DLL inicializada!")
                    
                    # Conectar
                    if self.connection.connect():
                        self.logger.info("✅ Conectado ao Profit!")
                        connected_to_profit = True
                        
                        # Aqui implementaríamos a lógica para carregar dados históricos reais
                        # Por enquanto vamos usar dados de exemplo
                        
            except Exception as e:
                self.logger.warning(f"⚠️ ProfitDLL não disponível: {e}")
            
            # Carregar dados de exemplo para teste
            self.logger.info(f"\n📈 Carregando dados de {symbol} ({days_back} dias)...")
            
            # Criar dados de exemplo
            num_candles = days_back * 24 * 60  # Candles de 1 minuto
            candles_df = self.data_loader.load_historical_data(symbol, days_back)
            
            if candles_df.empty:
                # Criar dados manualmente se necessário
                self.logger.info("📊 Criando dados de exemplo...")
                candles_df = self._create_sample_candles(symbol, num_candles)
            
            # Atualizar data structure
            self.data_structure.update_candles(candles_df)
            
            # Criar dados de microestrutura simulados
            micro_df = self._create_sample_microstructure(candles_df)
            self.data_structure.update_microstructure(micro_df)
            
            # Diagnóstico dos dados
            self.logger.info(f"\n✅ Dados carregados com sucesso!")
            self.logger.info(f"   📊 Candles: {len(candles_df)} registros")
            self.logger.info(f"   📅 Período: {candles_df.index[0]} a {candles_df.index[-1]}")
            
            # Estatísticas do último candle
            last_candle = candles_df.iloc[-1]
            self.logger.info(f"\n📊 Último candle:")
            self.logger.info(f"   Open:  {last_candle['open']:.2f}")
            self.logger.info(f"   High:  {last_candle['high']:.2f}")
            self.logger.info(f"   Low:   {last_candle['low']:.2f}")
            self.logger.info(f"   Close: {last_candle['close']:.2f}")
            self.logger.info(f"   Volume: {last_candle['volume']:.0f}")
            
            self.data_loaded = True
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar dados: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step3_calculate_features(self):
        """ETAPA 3: Calcular indicadores técnicos e features ML"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("⚙️ ETAPA 3: CÁLCULO DE FEATURES")
            self.logger.info("="*80)
            
            # Sincronizar com modelos
            self.logger.info("\n🔄 Sincronizando features com modelos...")
            self.feature_engine.sync_with_models(self.features_required)
            
            # Calcular features
            self.logger.info("⚙️ Calculando indicadores e features...")
            start_time = time.time()
            
            success = self.feature_engine.calculate()
            
            if not success:
                raise Exception("Falha no cálculo de features")
            
            calc_time = time.time() - start_time
            self.logger.info(f"✅ Features calculadas em {calc_time:.2f}s")
            
            # Diagnóstico das features
            indicators_df = self.data_structure.get_indicators()
            features_df = self.data_structure.get_features()
            
            self.logger.info(f"\n📊 Indicadores técnicos: {len(indicators_df.columns)} calculados")
            self.logger.info(f"🤖 Features ML: {len(features_df.columns)} calculadas")
            
            # Verificar features necessárias
            available_features = set(features_df.columns)
            missing_features = self.features_required - available_features
            
            if missing_features:
                self.logger.warning(f"\n⚠️ Features faltando: {len(missing_features)}")
                for feat in list(missing_features)[:5]:
                    self.logger.warning(f"   - {feat}")
            else:
                self.logger.info("\n✅ Todas as features necessárias foram calculadas!")
            
            # Estatísticas de qualidade
            nan_percentage = (features_df.isna().sum() / len(features_df) * 100).mean()
            self.logger.info(f"\n📊 Qualidade dos dados:")
            self.logger.info(f"   - NaN médio: {nan_percentage:.1f}%")
            self.logger.info(f"   - Linhas completas: {features_df.dropna().shape[0]}/{len(features_df)}")
            
            # Mostrar valores de alguns indicadores
            if not indicators_df.empty:
                last_indicators = indicators_df.iloc[-1]
                self.logger.info(f"\n📈 Indicadores atuais:")
                
                key_indicators = ['ema_9', 'ema_20', 'ema_50', 'rsi', 'macd', 'atr', 'adx']
                for ind in key_indicators:
                    if ind in last_indicators and pd.notna(last_indicators[ind]):
                        self.logger.info(f"   {ind}: {last_indicators[ind]:.2f}")
            
            self.features_calculated = True
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao calcular features: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step4_detect_regime(self):
        """ETAPA 4: Detectar regime de mercado"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("🎯 ETAPA 4: DETECÇÃO DE REGIME DE MERCADO")
            self.logger.info("="*80)
            
            # Obter dados necessários
            indicators_df = self.data_structure.get_indicators()
            
            if indicators_df.empty:
                self.logger.warning("⚠️ Sem indicadores para detectar regime")
                return None
            
            # Pegar últimos valores
            last_row = indicators_df.iloc[-1]
            
            # Verificar EMAs para tendência
            ema_9 = last_row.get('ema_9', np.nan)
            ema_20 = last_row.get('ema_20', np.nan)
            ema_50 = last_row.get('ema_50', np.nan)
            adx = last_row.get('adx', np.nan)
            
            regime = 'undefined'
            confidence = 0.0
            
            if pd.notna(ema_9) and pd.notna(ema_20) and pd.notna(ema_50) and pd.notna(adx):
                # Detectar tendência
                if adx > 25:
                    if ema_9 > ema_20 > ema_50:
                        regime = 'trend_up'
                        confidence = min(adx / 100, 0.9)
                    elif ema_9 < ema_20 < ema_50:
                        regime = 'trend_down'
                        confidence = min(adx / 100, 0.9)
                else:
                    regime = 'range'
                    confidence = 0.6
            
            self.logger.info(f"\n📊 Regime detectado: {regime.upper()}")
            self.logger.info(f"💪 Confiança: {confidence:.1%}")
            
            if pd.notna(ema_9):
                self.logger.info(f"\n📈 EMAs:")
                self.logger.info(f"   EMA 9:  {ema_9:.2f}")
                self.logger.info(f"   EMA 20: {ema_20:.2f}")
                self.logger.info(f"   EMA 50: {ema_50:.2f}")
            
            if pd.notna(adx):
                self.logger.info(f"\n📊 ADX: {adx:.1f}")
            
            return {
                'regime': regime,
                'confidence': confidence,
                'ema_alignment': 'bullish' if regime == 'trend_up' else 'bearish' if regime == 'trend_down' else 'neutral'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao detectar regime: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def step5_make_prediction(self):
        """ETAPA 5: Realizar predição ML"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("🤖 ETAPA 5: PREDIÇÃO ML")
            self.logger.info("="*80)
            
            # Detectar regime primeiro
            regime_info = self.step4_detect_regime()
            
            if not regime_info:
                self.logger.warning("⚠️ Não foi possível detectar regime")
                return None
            
            # Fazer predição via ML Coordinator
            self.logger.info("\n🧠 Processando predição com ML Coordinator...")
            
            prediction_result = self.ml_coordinator.process_prediction_request(
                self.data_structure
            )
            
            if prediction_result:
                self.logger.info("\n✅ Predição realizada com sucesso!")
                
                # Mostrar resultados
                self.logger.info(f"\n📊 RESULTADO DA PREDIÇÃO:")
                self.logger.info(f"   🎯 Decisão: {prediction_result.get('trade_decision', 'HOLD')}")
                self.logger.info(f"   💪 Confiança: {prediction_result.get('confidence', 0):.1%}")
                self.logger.info(f"   📈 Direção: {prediction_result.get('direction', 0):.3f}")
                self.logger.info(f"   🎰 Probabilidade: {prediction_result.get('probability', 0.5):.3f}")
                
                if prediction_result.get('can_trade'):
                    self.logger.info(f"   ✅ Sinal válido para trading!")
                    self.logger.info(f"   🎯 Risk/Reward: {prediction_result.get('risk_reward_target', 1.5)}")
                else:
                    self.logger.info(f"   ⏸️ Sinal não atende critérios mínimos")
                
                # Mostrar estratégia aplicada
                strategy = prediction_result.get('strategy_applied', 'N/A')
                self.logger.info(f"\n📋 Estratégia: {strategy}")
                
                return prediction_result
            else:
                self.logger.warning("⚠️ Nenhuma predição foi gerada")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao fazer predição: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def step6_generate_signal(self, prediction_result):
        """ETAPA 6: Gerar sinal de trading"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("📈 ETAPA 6: GERAÇÃO DE SINAL")
            self.logger.info("="*80)
            
            if not prediction_result:
                self.logger.warning("⚠️ Sem predição para gerar sinal")
                return None
            
            # Obter dados de mercado
            candles_df = self.data_structure.get_candles()
            if candles_df.empty:
                return None
            
            current_price = candles_df.iloc[-1]['close']
            
            # Gerar sinal
            signal = self.signal_generator.generate_signal(
                prediction_result,
                {'current_price': current_price}
            )
            
            if signal:
                self.logger.info("\n✅ Sinal gerado com sucesso!")
                
                # Mostrar detalhes do sinal
                self.logger.info(f"\n📊 DETALHES DO SINAL:")
                self.logger.info(f"   🎯 Tipo: {signal.get('type', 'N/A')}")
                self.logger.info(f"   💰 Preço atual: {current_price:.2f}")
                
                if signal.get('type') != 'HOLD':
                    self.logger.info(f"   🛑 Stop Loss: {signal.get('stop_loss', 0):.2f}")
                    self.logger.info(f"   🎯 Take Profit: {signal.get('take_profit', 0):.2f}")
                    self.logger.info(f"   📊 Risk/Reward: {signal.get('risk_reward', 0):.1f}")
                
                # Validar com Risk Manager
                if signal.get('type') != 'HOLD':
                    validation = self.risk_manager.validate_signal(signal)
                    
                    if validation['approved']:
                        self.logger.info(f"\n✅ Sinal APROVADO pelo Risk Manager!")
                        self.logger.info(f"   📊 Tamanho da posição: {validation.get('position_size', 1)} contratos")
                    else:
                        self.logger.warning(f"\n❌ Sinal REJEITADO pelo Risk Manager!")
                        self.logger.warning(f"   Motivo: {validation.get('reason', 'N/A')}")
                
                return signal
            else:
                self.logger.info("\n⏸️ Nenhum sinal gerado (condições não atendidas)")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao gerar sinal: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_complete_flow(self):
        """Executa o fluxo completo de dados"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("🚀 EXECUTANDO FLUXO COMPLETO DE DADOS ML TRADING")
            self.logger.info("="*80)
            
            # Inicializar sistema
            if not self.components_ready:
                self.initialize_system()
            
            # Executar etapas
            steps = [
                ("Carregar Modelos", self.step1_load_models),
                ("Carregar Dados", lambda: self.step2_load_data()),
                ("Calcular Features", self.step3_calculate_features),
                ("Fazer Predição", self.step5_make_prediction),
            ]
            
            results = {}
            
            for step_name, step_func in steps:
                self.logger.info(f"\n⏳ Executando: {step_name}...")
                
                result = step_func()
                results[step_name] = result
                
                if result is False:
                    self.logger.error(f"❌ Falha em: {step_name}")
                    break
                    
                time.sleep(1)  # Pausa para visualização
            
            # Se tudo correu bem, gerar sinal
            if results.get("Fazer Predição"):
                signal = self.step6_generate_signal(results["Fazer Predição"])
                results["Gerar Sinal"] = signal
            
            # Resumo final
            self.logger.info("\n" + "="*80)
            self.logger.info("📊 RESUMO DO FLUXO DE DADOS")
            self.logger.info("="*80)
            
            success_count = sum(1 for r in results.values() if r not in [False, None])
            total_count = len(results)
            
            self.logger.info(f"\n✅ Etapas bem-sucedidas: {success_count}/{total_count}")
            
            for step, result in results.items():
                if result is False:
                    status = "❌ Falhou"
                elif result is None:
                    status = "⚠️ Sem resultado"
                else:
                    status = "✅ Sucesso"
                
                self.logger.info(f"   {step}: {status}")
            
            self.logger.info("\n" + "="*80)
            self.logger.info("✅ TESTE DO FLUXO CONCLUÍDO!")
            self.logger.info("="*80)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Erro no fluxo completo: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_sample_candles(self, symbol, num_candles):
        """Cria candles de exemplo para teste"""
        try:
            # Gerar timestamps
            end_time = pd.Timestamp.now()
            timestamps = pd.date_range(
                end=end_time,
                periods=num_candles,
                freq='1min'
            )
            
            # Gerar preços realistas
            base_price = 5000
            prices = []
            
            for i in range(num_candles):
                # Simular movimento de preço
                change = np.random.normal(0, 0.0002)  # 0.02% volatilidade
                base_price *= (1 + change)
                
                # OHLC
                open_price = base_price
                close_price = base_price * (1 + np.random.normal(0, 0.0001))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0001)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0001)))
                volume = np.random.randint(100, 1000)
                
                prices.append({
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
                
                base_price = close_price
            
            # Criar DataFrame
            df = pd.DataFrame(prices, index=timestamps)
            df.index.name = 'timestamp'
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erro criando candles: {e}")
            return pd.DataFrame()
    
    def _create_sample_microstructure(self, candles_df):
        """Cria dados de microestrutura simulados"""
        try:
            micro_data = []
            
            for timestamp, candle in candles_df.iterrows():
                # Simular pressão compradora/vendedora
                buy_pressure = np.random.uniform(0.3, 0.7)
                sell_pressure = 1 - buy_pressure
                
                # Simular flow imbalance
                flow_imbalance = buy_pressure - sell_pressure
                
                # Simular book imbalance
                book_imbalance = np.random.uniform(-0.5, 0.5)
                
                micro_data.append({
                    'buy_pressure': buy_pressure,
                    'sell_pressure': sell_pressure,
                    'flow_imbalance': flow_imbalance,
                    'book_imbalance': book_imbalance,
                    'spread': np.random.uniform(0.5, 2.0),
                    'trade_intensity': np.random.uniform(0.1, 1.0)
                })
            
            df = pd.DataFrame(micro_data, index=candles_df.index)
            return df
            
        except Exception as e:
            self.logger.error(f"Erro criando microestrutura: {e}")
            return pd.DataFrame()


def main():
    """Função principal"""
    tester = MLDataFlowTester()
    
    # Executar fluxo completo automaticamente
    tester.run_complete_flow()


if __name__ == "__main__":
    main()