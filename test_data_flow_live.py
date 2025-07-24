"""
Script de Teste do Fluxo Completo de Dados ML Trading v2.0
Testa desde o carregamento de modelos até a predição final
Com atualização em tempo real a cada novo candle
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
import threading
import queue

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Imports dos componentes do sistema
from src.connection_manager import ConnectionManager
from src.model_manager import ModelManager
from src.data_structure import TradingDataStructure
from src.data_loader import DataLoader
from src.feature_engine import FeatureEngine
from src.prediction_engine import PredictionEngine
from src.technical_indicators import TechnicalIndicators
from src.ml_features import MLFeatures

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataFlowTest')

class DataFlowTester:
    """Classe para testar o fluxo completo de dados do sistema"""
    
    def __init__(self):
        self.logger = logger
        self.models_loaded = False
        self.features_required = set()
        self.last_candle_time = None
        self.prediction_count = 0
        
        # Componentes do sistema
        self.connection = None
        self.model_manager = None
        self.data_structure = None
        self.data_loader = None
        self.feature_engine = None
        self.prediction_engine = None
        
        # Queue para dados em tempo real
        self.rt_queue = queue.Queue(maxsize=1000)
        self.running = True
        
    def initialize_components(self):
        """Inicializa todos os componentes do sistema"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("INICIANDO TESTE DO FLUXO DE DADOS ML TRADING v2.0")
            self.logger.info("=" * 80)
            
            # 1. Inicializar Connection Manager
            self.logger.info("\n1. Inicializando Connection Manager...")
            dll_path = os.getenv('PROFIT_DLL_PATH', r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll")
            self.connection = ConnectionManager(dll_path)
            
            # 2. Inicializar Model Manager
            self.logger.info("\n2. Inicializando Model Manager...")
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            self.model_manager = ModelManager(models_dir)
            
            # 3. Inicializar Data Structure
            self.logger.info("\n3. Inicializando Data Structure...")
            self.data_structure = TradingDataStructure()
            self.data_structure.initialize_structure()
            
            # 4. Inicializar Data Loader
            self.logger.info("\n4. Inicializando Data Loader...")
            self.data_loader = DataLoader()
            
            # 5. Inicializar Feature Engine
            self.logger.info("\n5. Inicializando Feature Engine...")
            self.feature_engine = FeatureEngine()
            
            # 6. Inicializar Prediction Engine
            self.logger.info("\n6. Inicializando Prediction Engine...")
            self.prediction_engine = PredictionEngine(self.model_manager)
            
            self.logger.info("\n✅ Todos os componentes inicializados com sucesso!")
            
        except Exception as e:
            self.logger.error(f"❌ Erro na inicialização: {e}")
            raise
    
    def load_and_analyze_models(self):
        """Carrega modelos e analisa features necessárias"""
        try:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("ETAPA 1: CARREGAMENTO E ANÁLISE DE MODELOS")
            self.logger.info("=" * 80)
            
            # Carregar modelos
            self.logger.info("\nCarregando modelos ML...")
            self.model_manager.load_models()
            
            # Verificar modelos carregados
            models_info = self.model_manager.get_models_info()
            self.logger.info(f"\n📊 Modelos carregados: {len(models_info)}")
            
            for name, info in models_info.items():
                self.logger.info(f"\n  🤖 {name}:")
                self.logger.info(f"     - Tipo: {info['type']}")
                self.logger.info(f"     - Features: {info['feature_count']}")
                self.logger.info(f"     - Score: {info.get('best_score', 'N/A'):.4f}")
                if 'metadata' in info and info['metadata']:
                    self.logger.info(f"     - Treinado em: {info['metadata'].get('training_date', 'N/A')}")
            
            # Obter todas as features necessárias
            self.features_required = self.model_manager.get_all_required_features()
            self.logger.info(f"\n📋 Total de features únicas necessárias: {len(self.features_required)}")
            
            # Mostrar amostra das features
            features_list = sorted(list(self.features_required))[:10]
            self.logger.info(f"\n🔍 Amostra das features necessárias:")
            for feature in features_list:
                self.logger.info(f"   - {feature}")
            
            if len(self.features_required) > 10:
                self.logger.info(f"   ... e mais {len(self.features_required) - 10} features")
            
            self.models_loaded = True
            self.logger.info("\n✅ Modelos carregados e analisados com sucesso!")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar modelos: {e}")
            raise
    
    def load_historical_data(self, symbol='WDO', days_back=5):
        """Carrega dados históricos via ProfitDLL ou dados de teste"""
        try:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("ETAPA 2: CARREGAMENTO DE DADOS HISTÓRICOS")
            self.logger.info("=" * 80)
            
            # Tentar conectar ao Profit
            self.logger.info(f"\n🔌 Tentando conectar ao ProfitDLL...")
            
            try:
                # Tentar inicializar conexão
                init_result = self.connection.initialize()
                if init_result == 1:
                    self.logger.info("✅ ProfitDLL inicializado com sucesso!")
                    
                    # Tentar conectar
                    connect_result = self.connection.connect()
                    if connect_result:
                        self.logger.info("✅ Conectado ao Profit com sucesso!")
                        
                        # Carregar dados históricos reais
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=days_back)
                        
                        self.logger.info(f"\n📊 Carregando dados históricos de {symbol}...")
                        self.logger.info(f"   Período: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
                        
                        success = self.data_loader.load_historical_data(
                            ticker=symbol,
                            start_date=start_date,
                            end_date=end_date
                        )
                        
                        if success:
                            self.logger.info("✅ Dados históricos carregados via ProfitDLL!")
                        else:
                            raise Exception("Falha ao carregar dados históricos")
                    else:
                        raise Exception("Falha ao conectar ao Profit")
                else:
                    raise Exception("Falha ao inicializar ProfitDLL")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ ProfitDLL não disponível: {e}")
                self.logger.info("📊 Carregando dados de teste para desenvolvimento...")
                
                # Carregar dados de teste
                success = self.data_loader.load_test_data(
                    ticker=symbol,
                    days_back=days_back
                )
                
                if success:
                    self.logger.info("✅ Dados de teste carregados com sucesso!")
                else:
                    raise Exception("Falha ao carregar dados de teste")
            
            # Diagnóstico dos dados carregados
            self.diagnose_data()
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar dados: {e}")
            raise
    
    def diagnose_data(self):
        """Realiza diagnóstico dos dados carregados"""
        try:
            self.logger.info("\n📊 DIAGNÓSTICO DOS DADOS CARREGADOS:")
            
            # Verificar candles
            candles_df = self.data_structure.get_candles()
            if not candles_df.empty:
                self.logger.info(f"\n📈 DataFrame de Candles:")
                self.logger.info(f"   - Shape: {candles_df.shape}")
                self.logger.info(f"   - Período: {candles_df.index[0]} a {candles_df.index[-1]}")
                self.logger.info(f"   - Colunas: {list(candles_df.columns)}")
                
                # Estatísticas básicas
                self.logger.info(f"\n   📊 Estatísticas do último candle:")
                last_candle = candles_df.iloc[-1]
                self.logger.info(f"      - Open: {last_candle['open']:.2f}")
                self.logger.info(f"      - High: {last_candle['high']:.2f}")
                self.logger.info(f"      - Low: {last_candle['low']:.2f}")
                self.logger.info(f"      - Close: {last_candle['close']:.2f}")
                self.logger.info(f"      - Volume: {last_candle['volume']:.0f}")
            else:
                self.logger.warning("⚠️ DataFrame de candles está vazio!")
            
            # Verificar microestrutura
            micro_df = self.data_structure.get_microstructure()
            if not micro_df.empty:
                self.logger.info(f"\n🔬 DataFrame de Microestrutura:")
                self.logger.info(f"   - Shape: {micro_df.shape}")
                self.logger.info(f"   - Colunas: {list(micro_df.columns)}")
            
            # Verificar orderbook
            orderbook_df = self.data_structure.get_orderbook()
            if not orderbook_df.empty:
                self.logger.info(f"\n📖 DataFrame de Orderbook:")
                self.logger.info(f"   - Shape: {orderbook_df.shape}")
                self.logger.info(f"   - Colunas: {list(orderbook_df.columns)}")
                
        except Exception as e:
            self.logger.error(f"❌ Erro no diagnóstico: {e}")
    
    def calculate_features(self):
        """Calcula todas as features necessárias"""
        try:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("ETAPA 3: CÁLCULO DE FEATURES")
            self.logger.info("=" * 80)
            
            # Sincronizar features com modelos
            self.logger.info("\n🔄 Sincronizando features com modelos...")
            self.feature_engine.sync_with_models(self.features_required)
            
            # Calcular features
            self.logger.info("\n⚙️ Calculando features...")
            start_time = time.time()
            
            success = self.feature_engine.calculate()
            
            calc_time = time.time() - start_time
            
            if success:
                self.logger.info(f"✅ Features calculadas com sucesso em {calc_time:.2f}s!")
                
                # Diagnóstico das features
                self.diagnose_features()
            else:
                raise Exception("Falha no cálculo de features")
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao calcular features: {e}")
            raise
    
    def diagnose_features(self):
        """Realiza diagnóstico das features calculadas"""
        try:
            self.logger.info("\n📊 DIAGNÓSTICO DAS FEATURES:")
            
            # Obter DataFrames
            indicators_df = self.data_structure.get_indicators()
            features_df = self.data_structure.get_features()
            
            # Diagnóstico de indicadores
            if not indicators_df.empty:
                self.logger.info(f"\n📈 DataFrame de Indicadores Técnicos:")
                self.logger.info(f"   - Shape: {indicators_df.shape}")
                self.logger.info(f"   - Total de indicadores: {len(indicators_df.columns)}")
                
                # Amostra de indicadores
                sample_indicators = ['ema_9', 'ema_20', 'ema_50', 'rsi', 'macd', 'atr', 'adx']
                available_indicators = [ind for ind in sample_indicators if ind in indicators_df.columns]
                
                if available_indicators:
                    self.logger.info(f"\n   📊 Valores atuais de indicadores principais:")
                    last_row = indicators_df.iloc[-1]
                    for ind in available_indicators:
                        value = last_row[ind]
                        if pd.notna(value):
                            self.logger.info(f"      - {ind}: {value:.2f}")
            
            # Diagnóstico de features ML
            if not features_df.empty:
                self.logger.info(f"\n🤖 DataFrame de Features ML:")
                self.logger.info(f"   - Shape: {features_df.shape}")
                self.logger.info(f"   - Total de features: {len(features_df.columns)}")
                
                # Verificar features necessárias
                available_features = set(features_df.columns)
                missing_features = self.features_required - available_features
                
                if missing_features:
                    self.logger.warning(f"\n⚠️ Features faltando: {len(missing_features)}")
                    for feat in list(missing_features)[:5]:
                        self.logger.warning(f"   - {feat}")
                else:
                    self.logger.info("✅ Todas as features necessárias foram calculadas!")
                
                # Estatísticas de NaN
                nan_counts = features_df.isna().sum()
                features_with_nan = nan_counts[nan_counts > 0]
                
                if len(features_with_nan) > 0:
                    self.logger.info(f"\n⚠️ Features com valores NaN: {len(features_with_nan)}")
                    for feat, count in features_with_nan.head(5).items():
                        self.logger.info(f"   - {feat}: {count} NaNs ({count/len(features_df)*100:.1f}%)")
                else:
                    self.logger.info("\n✅ Nenhuma feature com valores NaN!")
                    
        except Exception as e:
            self.logger.error(f"❌ Erro no diagnóstico de features: {e}")
    
    def make_prediction(self):
        """Realiza predição usando os modelos carregados"""
        try:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("ETAPA 4: REALIZANDO PREDIÇÃO")
            self.logger.info("=" * 80)
            
            # Preparar dados para predição
            features_df = self.data_structure.get_features()
            
            if features_df.empty:
                raise Exception("DataFrame de features está vazio!")
            
            # Usar última linha com features completas
            last_complete_idx = features_df.dropna(how='all').index[-1]
            prediction_data = features_df.loc[last_complete_idx]
            
            self.logger.info(f"\n🎯 Realizando predição para: {last_complete_idx}")
            
            # Fazer predição com cada modelo
            predictions = {}
            
            for model_name in self.model_manager.models.keys():
                try:
                    self.logger.info(f"\n🤖 Predição com {model_name}:")
                    
                    # Preparar features para o modelo
                    model_features = self.model_manager.get_model_features(model_name)
                    if not model_features:
                        self.logger.warning(f"   ⚠️ Sem features definidas para {model_name}")
                        continue
                    
                    # Filtrar apenas features disponíveis
                    available_features = [f for f in model_features if f in features_df.columns]
                    if len(available_features) < len(model_features):
                        self.logger.warning(f"   ⚠️ Usando {len(available_features)}/{len(model_features)} features")
                    
                    # Preparar dados
                    X = prediction_data[available_features].values.reshape(1, -1)
                    
                    # Fazer predição
                    prediction = self.prediction_engine.predict(model_name, X)
                    
                    if prediction is not None:
                        predictions[model_name] = prediction
                        
                        # Interpretar resultado
                        if hasattr(prediction, '__len__'):
                            pred_value = prediction[0]
                        else:
                            pred_value = prediction
                        
                        # Determinar sinal
                        if pred_value > 0.5:
                            signal = "BUY"
                            confidence = pred_value
                        elif pred_value < 0.5:
                            signal = "SELL"
                            confidence = 1 - pred_value
                        else:
                            signal = "HOLD"
                            confidence = 0.5
                        
                        self.logger.info(f"   📊 Predição: {pred_value:.4f}")
                        self.logger.info(f"   🎯 Sinal: {signal}")
                        self.logger.info(f"   💪 Confiança: {confidence:.2%}")
                        
                except Exception as e:
                    self.logger.error(f"   ❌ Erro na predição com {model_name}: {e}")
            
            # Consolidar predições
            if predictions:
                self.logger.info(f"\n✅ Predições realizadas com {len(predictions)} modelos!")
                
                # Calcular consenso
                pred_values = list(predictions.values())
                avg_prediction = np.mean(pred_values)
                
                self.logger.info(f"\n🎯 CONSENSO DOS MODELOS:")
                self.logger.info(f"   📊 Média das predições: {avg_prediction:.4f}")
                
                if avg_prediction > 0.55:
                    consensus = "BUY"
                elif avg_prediction < 0.45:
                    consensus = "SELL"
                else:
                    consensus = "HOLD"
                
                self.logger.info(f"   🎯 Sinal consenso: {consensus}")
                
                self.prediction_count += 1
                
                return {
                    'timestamp': last_complete_idx,
                    'predictions': predictions,
                    'consensus': consensus,
                    'avg_prediction': avg_prediction,
                    'prediction_count': self.prediction_count
                }
            else:
                self.logger.warning("⚠️ Nenhuma predição foi realizada!")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao fazer predição: {e}")
            return None
    
    def update_loop(self, update_interval=60):
        """Loop de atualização contínua a cada novo candle"""
        try:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("INICIANDO LOOP DE ATUALIZAÇÃO CONTÍNUA")
            self.logger.info(f"Intervalo de atualização: {update_interval}s")
            self.logger.info("=" * 80)
            
            while self.running:
                try:
                    # Aguardar próximo intervalo
                    self.logger.info(f"\n⏰ Aguardando próximo ciclo ({update_interval}s)...")
                    time.sleep(update_interval)
                    
                    # Verificar se há novos dados
                    self.logger.info("\n🔄 Verificando novos dados...")
                    
                    # Se conectado ao Profit, tentar obter dados em tempo real
                    if self.connection and self.connection.connected:
                        # Aqui implementaríamos a lógica para obter novos trades/candles
                        pass
                    
                    # Recalcular features
                    self.logger.info("⚙️ Recalculando features...")
                    self.feature_engine.calculate()
                    
                    # Fazer nova predição
                    prediction_result = self.make_prediction()
                    
                    if prediction_result:
                        self.logger.info(f"\n✅ Ciclo #{prediction_result['prediction_count']} concluído!")
                        
                        # Aqui podemos enviar resultado para GUI ou sistema de trading
                        self.display_results(prediction_result)
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    self.logger.error(f"❌ Erro no ciclo de atualização: {e}")
                    
        except KeyboardInterrupt:
            self.logger.info("\n⏹️ Loop de atualização interrompido pelo usuário")
            self.running = False
    
    def display_results(self, result):
        """Exibe resultados no monitor"""
        try:
            self.logger.info("\n" + "=" * 60)
            self.logger.info("📊 RESULTADO DA PREDIÇÃO")
            self.logger.info("=" * 60)
            self.logger.info(f"🕐 Timestamp: {result['timestamp']}")
            self.logger.info(f"📈 Predição média: {result['avg_prediction']:.4f}")
            self.logger.info(f"🎯 Sinal consenso: {result['consensus']}")
            self.logger.info(f"🔢 Total de predições: {result['prediction_count']}")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao exibir resultados: {e}")
    
    def run_complete_test(self):
        """Executa o teste completo do fluxo de dados"""
        try:
            # Inicializar componentes
            self.initialize_components()
            
            # Carregar e analisar modelos
            self.load_and_analyze_models()
            
            # Carregar dados históricos
            self.load_historical_data()
            
            # Calcular features
            self.calculate_features()
            
            # Fazer primeira predição
            self.make_prediction()
            
            # Perguntar se deseja continuar com loop
            response = input("\n🔄 Deseja iniciar loop de atualização contínua? (s/n): ")
            
            if response.lower() == 's':
                # Iniciar loop de atualização
                self.update_loop()
            else:
                self.logger.info("\n✅ Teste concluído!")
                
        except Exception as e:
            self.logger.error(f"\n❌ Erro durante teste: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Função principal"""
    # Criar e executar teste
    tester = DataFlowTester()
    tester.run_complete_test()


if __name__ == "__main__":
    main()