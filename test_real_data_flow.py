"""
Script de Teste com Dados Históricos Reais
Usa a arquitetura existente do ConnectionManager para dados históricos via ProfitDLL
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Imports dos componentes do sistema
from src.connection_manager import ConnectionManager
from src.model_manager import ModelManager
from src.data_structure import TradingDataStructure
from src.data_loader import DataLoader
from src.data_integration import DataIntegration
from src.feature_engine import FeatureEngine
from src.prediction_engine import PredictionEngine
from src.ml_coordinator import MLCoordinator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RealDataFlowTest')

class RealDataFlowTester:
    """Testa o fluxo completo usando dados históricos reais via ProfitDLL"""
    
    def __init__(self):
        self.logger = logger
        
        # Componentes do sistema
        self.connection = None
        self.model_manager = None
        self.data_structure = None
        self.data_loader = None
        self.data_integration = None
        self.feature_engine = None
        self.prediction_engine = None
        self.ml_coordinator = None
        
        # Estado
        self.connected = False
        self.models_loaded = False
        self.historical_data_loaded = False
        self.features_calculated = False
        
        # Configurações do .env
        self.dll_path = os.getenv('PROFIT_DLL_PATH')
        self.key = os.getenv('PROFIT_KEY')
        self.username = os.getenv('PROFIT_USER')
        self.password = os.getenv('PROFIT_PASSWORD')
        self.ticker = os.getenv('TICKER', 'WDOQ25')
        self.models_dir = os.getenv('MODELS_DIR')
        
    def initialize_components(self):
        """Inicializa componentes do sistema"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("🚀 INICIALIZANDO SISTEMA COM DADOS HISTÓRICOS REAIS")
            self.logger.info("="*80)
            
            # 1. Connection Manager
            self.logger.info("\n📡 [1/6] Connection Manager...")
            self.connection = ConnectionManager(self.dll_path)
            
            # 2. Data Structure
            self.logger.info("📊 [2/6] Data Structure...")
            self.data_structure = TradingDataStructure()
            self.data_structure.initialize_structure()
            
            # 2.5. Data Loader (handles candle creation)
            self.logger.info("📈 [2.5/6] Data Loader...")
            self.data_loader = DataLoader()
            
            # 3. Data Integration (ponte entre ConnectionManager e DataLoader)
            self.logger.info("🔗 [3/6] Data Integration...")
            self.data_integration = DataIntegration(self.connection, self.data_loader)
            
            # 4. Model Manager
            self.logger.info("🤖 [4/6] Model Manager...") 
            self.model_manager = ModelManager(self.models_dir)
            
            # 5. Feature Engine
            self.logger.info("⚙️ [5/6] Feature Engine...")
            self.feature_engine = FeatureEngine()
            # Configurar para modo desenvolvimento para este teste
            self.feature_engine.production_mode = False
            self.feature_engine.require_validation = False
            self.feature_engine.block_on_dummy_data = False
            
            # 6. Prediction Engine & ML Coordinator
            self.logger.info("🎯 [6/6] Prediction Engine & ML Coordinator...")
            self.prediction_engine = PredictionEngine(self.model_manager)
            self.ml_coordinator = MLCoordinator(
                self.model_manager,
                self.feature_engine,
                self.prediction_engine,
                None
            )
            
            self.logger.info("\n✅ Todos os componentes inicializados!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro na inicialização: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def connect_to_profit(self):
        """Conecta ao ProfitDLL"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("🔌 CONECTANDO AO PROFITDLL")
            self.logger.info("="*80)
            
            # Inicializar DLL
            self.logger.info("🔄 Inicializando DLL...")
            result = self.connection.initialize(
                key=self.key,
                username=self.username,
                password=self.password
            )
            
            if result == 1:
                self.logger.info("✅ DLL inicializada com sucesso!")
                
                # Aguardar estabilização da conexão
                self.logger.info("⏳ Aguardando estabilização (5s)...")
                time.sleep(5)
                
                # Verificar estado da conexão
                self.logger.info(f"\n📊 Estado da conexão:")
                self.logger.info(f"   🔗 Conectado: {'✅' if self.connection.connected else '❌'}")
                self.logger.info(f"   🏢 Login: {'✅' if self.connection.login_state == 0 else '❌'}")
                self.logger.info(f"   📈 Market Data: {'✅' if self.connection.market_connected else '❌'}")
                
                if self.connection.login_state == 0:  # LOGIN_CONNECTED
                    self.connected = True
                    self.logger.info("\n🎉 Conectado e pronto para dados históricos!")
                    return True
                else:
                    self.logger.error("❌ Login não conectado")
                    return False
            else:
                self.logger.error(f"❌ Falha na inicialização: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erro na conexão: {e}")
            return False
    
    def load_models(self):
        """Carrega modelos ML"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("🤖 CARREGANDO MODELOS ML")
            self.logger.info("="*80)
            
            success = self.model_manager.load_models()
            
            if success and self.model_manager.models:
                self.logger.info(f"\n✅ {len(self.model_manager.models)} modelos carregados:")
                
                for name, model in self.model_manager.models.items():
                    model_type = type(model).__name__
                    features_count = len(self.model_manager.model_features.get(name, []))
                    self.logger.info(f"   🤖 {name}: {model_type} ({features_count} features)")
                
                self.models_loaded = True
                return True
            else:
                self.logger.error("❌ Nenhum modelo encontrado")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar modelos: {e}")
            return False
    
    def request_historical_data(self, days_back=3):
        """Solicita dados históricos reais via ProfitDLL"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("📈 SOLICITANDO DADOS HISTÓRICOS REAIS")
            self.logger.info("="*80)
            
            if not self.connected:
                self.logger.error("❌ Não conectado ao ProfitDLL")
                return False
            
            # Definir período (máximo 3 dias como recomendado pelo sistema)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=min(days_back, 3))
            
            self.logger.info(f"\n📅 Solicitando dados históricos:")
            self.logger.info(f"   🎯 Ticker: {self.ticker}")
            self.logger.info(f"   📅 De: {start_date.strftime('%Y-%m-%d %H:%M')}")
            self.logger.info(f"   📅 Até: {end_date.strftime('%Y-%m-%d %H:%M')}")
            self.logger.info(f"   ⏱️ Período: {(end_date - start_date).days} dias")
            
            # Usar o método existente do ConnectionManager
            self.logger.info("\n🔄 Enviando requisição...")
            request_result = self.connection.request_historical_data(
                ticker=self.ticker,
                start_date=start_date,
                end_date=end_date
            )
            
            if request_result >= 0:
                self.logger.info(f"✅ Requisição enviada com sucesso! (ID: {request_result})")
                
                # Aguardar dados usando o método existente
                self.logger.info("\n⏳ Aguardando dados históricos...")
                wait_success = self.connection.wait_for_historical_data(timeout_seconds=60)
                
                if wait_success:
                    self.logger.info("✅ Dados históricos recebidos!")
                    
                    # Transferir dados do DataLoader para TradingDataStructure
                    if hasattr(self.data_loader, 'candles_df') and not self.data_loader.candles_df.empty:
                        self.logger.info("🔄 Transferindo candles do DataLoader para TradingDataStructure...")
                        success = self.data_structure.update_candles(self.data_loader.candles_df)
                        if success:
                            self.logger.info("✅ Candles transferidos com sucesso!")
                        else:
                            self.logger.warning("⚠️ Falha na transferência de candles")
                    
                    # Verificar dados na estrutura
                    candles_df = self.data_structure.get_candles()
                    
                    if not candles_df.empty:
                        self.logger.info(f"\n📊 DADOS HISTÓRICOS CARREGADOS:")
                        self.logger.info(f"   🕐 Total de candles: {len(candles_df)}")
                        self.logger.info(f"   📅 Período: {candles_df.index[0]} a {candles_df.index[-1]}")
                        self.logger.info(f"   📊 Colunas: {list(candles_df.columns)}")
                        
                        # Estatísticas do último candle
                        last_candle = candles_df.iloc[-1]
                        self.logger.info(f"\n📈 Último candle ({candles_df.index[-1]}):")
                        self.logger.info(f"   Open:  {last_candle['open']:.2f}")
                        self.logger.info(f"   High:  {last_candle['high']:.2f}")
                        self.logger.info(f"   Low:   {last_candle['low']:.2f}")
                        self.logger.info(f"   Close: {last_candle['close']:.2f}")
                        self.logger.info(f"   Volume: {last_candle['volume']:.0f}")
                        
                        # Verificar qualidade dos dados
                        price_range = (candles_df['high'].max() - candles_df['low'].min())
                        avg_volume = candles_df['volume'].mean()
                        
                        self.logger.info(f"\n📊 Qualidade dos dados:")
                        self.logger.info(f"   📈 Range de preços: {price_range:.2f}")
                        self.logger.info(f"   📊 Volume médio: {avg_volume:.0f}")
                        self.logger.info(f"   ✅ Dados válidos: {len(candles_df.dropna())}/{len(candles_df)}")
                        
                        self.historical_data_loaded = True
                        return True
                    else:
                        self.logger.warning("⚠️ Dados recebidos mas DataFrame está vazio")
                        return False
                else:
                    self.logger.error("❌ Timeout aguardando dados históricos")
                    return False
            else:
                self.logger.error(f"❌ Falha na requisição: {request_result}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao solicitar dados históricos: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_features_on_real_data(self):
        """Calcula features usando dados históricos reais"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("⚙️ CALCULANDO FEATURES COM DADOS REAIS")
            self.logger.info("="*80)
            
            if not self.historical_data_loaded:
                self.logger.error("❌ Dados históricos não carregados")
                return False
            
            # Verificar dados disponíveis
            candles_df = self.data_structure.get_candles()
            self.logger.info(f"\n📊 Dados disponíveis: {len(candles_df)} candles")
            
            # Calcular features
            self.logger.info("⚙️ Iniciando cálculo de features...")
            start_time = time.time()
            
            result = self.feature_engine.calculate(
                data=self.data_structure,
                force_recalculate=True,
                use_advanced=True
            )
            
            calc_time = time.time() - start_time
            
            if result:
                self.logger.info(f"✅ Features calculadas em {calc_time:.2f}s")
                
                # Diagnóstico das features
                indicators_df = self.data_structure.get_indicators()
                features_df = self.data_structure.get_features()
                
                self.logger.info(f"\n📈 Indicadores técnicos: {len(indicators_df.columns)} calculados")
                self.logger.info(f"🤖 Features ML: {len(features_df.columns)} calculadas")
                
                # Mostrar indicadores principais
                if not indicators_df.empty:
                    last_indicators = indicators_df.iloc[-1]
                    key_indicators = ['ema_9', 'ema_20', 'ema_50', 'rsi', 'macd', 'atr', 'adx']
                    
                    self.logger.info(f"\n📊 Indicadores atuais (dados reais):")
                    for ind in key_indicators:
                        if ind in last_indicators and pd.notna(last_indicators[ind]):
                            self.logger.info(f"   {ind}: {last_indicators[ind]:.2f}")
                
                # Verificar qualidade das features
                if not features_df.empty:
                    nan_count = features_df.isna().sum().sum()
                    total_values = len(features_df) * len(features_df.columns)
                    nan_percentage = (nan_count / total_values) * 100
                    
                    self.logger.info(f"\n📊 Qualidade das features:")
                    self.logger.info(f"   📊 Total de valores: {total_values}")
                    self.logger.info(f"   ❌ Valores NaN: {nan_count} ({nan_percentage:.1f}%)")
                    self.logger.info(f"   ✅ Valores válidos: {total_values - nan_count} ({100 - nan_percentage:.1f}%)")
                
                self.features_calculated = True
                return True
            else:
                self.logger.error("❌ Falha no cálculo de features")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao calcular features: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def make_prediction_with_real_data(self):
        """Realiza predição usando dados reais"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("🎯 PREDIÇÃO ML COM DADOS REAIS")
            self.logger.info("="*80)
            
            if not self.features_calculated:
                self.logger.error("❌ Features não calculadas")
                return False
            
            # Obter preço atual dos dados reais
            candles_df = self.data_structure.get_candles()
            current_price = candles_df.iloc[-1]['close']
            current_time = candles_df.index[-1]
            
            self.logger.info(f"\n📊 Contexto da predição:")
            self.logger.info(f"   🕐 Timestamp: {current_time}")
            self.logger.info(f"   💰 Preço atual: {current_price:.2f}")
            self.logger.info(f"   📊 Baseado em {len(candles_df)} candles reais")
            
            # Fazer predição
            self.logger.info("\n🧠 Processando predição com modelos ML...")
            prediction_result = self.ml_coordinator.process_prediction_request(
                self.data_structure
            )
            
            if prediction_result:
                self.logger.info("\n✅ Predição realizada com dados reais!")
                
                self.logger.info(f"\n🎯 RESULTADO DA PREDIÇÃO (DADOS REAIS):")
                self.logger.info(f"   📊 Regime: {prediction_result.get('regime', 'N/A').upper()}")
                self.logger.info(f"   🎯 Decisão: {prediction_result.get('trade_decision', 'HOLD')}")
                self.logger.info(f"   💪 Confiança: {prediction_result.get('confidence', 0):.1%}")
                self.logger.info(f"   📈 Direção: {prediction_result.get('direction', 0):.3f}")
                self.logger.info(f"   🎰 Probabilidade: {prediction_result.get('probability', 0.5):.3f}")
                
                # Analisar se pode operar
                can_trade = prediction_result.get('can_trade', False)
                if can_trade:
                    self.logger.info(f"   ✅ SINAL VÁLIDO para trading com dados reais!")
                    
                    # Calcular níveis de entrada
                    atr_value = self.data_structure.get_indicators().iloc[-1].get('atr', 10)
                    stop_distance = max(atr_value * 2, 5)  # Mínimo 5 pontos
                    
                    if prediction_result.get('trade_decision') == 'BUY':
                        entry = current_price
                        stop = entry - stop_distance
                        target = entry + (stop_distance * 2)  # Risk/Reward 1:2
                        
                        self.logger.info(f"\n📈 SETUP DE COMPRA (DADOS REAIS):")
                        self.logger.info(f"   💰 Entrada: {entry:.2f}")
                        self.logger.info(f"   🛑 Stop: {stop:.2f} (-{stop_distance:.1f} pts)")
                        self.logger.info(f"   🎯 Alvo: {target:.2f} (+{stop_distance*2:.1f} pts)")
                        
                    elif prediction_result.get('trade_decision') == 'SELL':
                        entry = current_price
                        stop = entry + stop_distance
                        target = entry - (stop_distance * 2)
                        
                        self.logger.info(f"\n📉 SETUP DE VENDA (DADOS REAIS):")
                        self.logger.info(f"   💰 Entrada: {entry:.2f}")
                        self.logger.info(f"   🛑 Stop: {stop:.2f} (+{stop_distance:.1f} pts)")
                        self.logger.info(f"   🎯 Alvo: {target:.2f} (-{stop_distance*2:.1f} pts)")
                else:
                    self.logger.info(f"   ⏸️ Aguardar melhor oportunidade")
                
                return prediction_result
            else:
                self.logger.warning("⚠️ Nenhuma predição gerada")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Erro na predição: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_complete_test(self):
        """Executa teste completo com dados históricos reais"""
        try:
            self.logger.info("\n" + "="*100)
            self.logger.info("🚀 TESTE COMPLETO COM DADOS HISTÓRICOS REAIS VIA PROFITDLL")
            self.logger.info("="*100)
            
            # Etapas do teste
            steps = [
                ("Inicializar Componentes", self.initialize_components),
                ("Conectar ProfitDLL", self.connect_to_profit),
                ("Carregar Modelos ML", self.load_models),
                ("Solicitar Dados Históricos", self.request_historical_data),
                ("Calcular Features", self.calculate_features_on_real_data),
                ("Realizar Predição", self.make_prediction_with_real_data)
            ]
            
            results = {}
            
            for step_name, step_func in steps:
                self.logger.info(f"\n▶️ Executando: {step_name}...")
                
                try:
                    result = step_func()
                    results[step_name] = result
                    
                    if result:
                        self.logger.info(f"✅ {step_name}: SUCESSO")
                    else:
                        self.logger.error(f"❌ {step_name}: FALHA")
                        
                        # Parar em falhas críticas
                        if step_name in ["Conectar ProfitDLL", "Carregar Modelos ML"]:
                            self.logger.error(f"🛑 Falha crítica - parando teste")
                            break
                            
                except Exception as e:
                    self.logger.error(f"❌ {step_name}: ERRO - {e}")
                    results[step_name] = False
                    break
                
                # Pausa entre etapas
                time.sleep(2)
            
            # Resumo final
            self.logger.info("\n" + "="*100)
            self.logger.info("📊 RESUMO DO TESTE COM DADOS REAIS")
            self.logger.info("="*100)
            
            success_count = sum(1 for r in results.values() if r)
            total_count = len(results)
            
            self.logger.info(f"\n✅ Etapas bem-sucedidas: {success_count}/{total_count}")
            
            for step_name, result in results.items():
                status = "✅ SUCESSO" if result else "❌ FALHA"
                self.logger.info(f"   {step_name}: {status}")
            
            if success_count == total_count:
                self.logger.info(f"\n🎉 SISTEMA COMPLETAMENTE FUNCIONAL COM DADOS REAIS!")
                
                # Mostrar estatísticas finais
                candles_df = self.data_structure.get_candles()
                if not candles_df.empty:
                    self.logger.info(f"\n📊 DADOS PROCESSADOS:")
                    self.logger.info(f"   📈 Candles: {len(candles_df)}")
                    self.logger.info(f"   🕐 Período: {candles_df.index[0]} a {candles_df.index[-1]}")
                    self.logger.info(f"   💰 Range de preços: {candles_df['low'].min():.2f} - {candles_df['high'].max():.2f}")
                    
                    # Informações do último candle
                    last_candle = candles_df.iloc[-1]
                    self.logger.info(f"\n📊 ÚLTIMO CANDLE (TEMPO REAL):")
                    self.logger.info(f"   🕐 {candles_df.index[-1]}")
                    self.logger.info(f"   💰 {last_candle['close']:.2f} (Vol: {last_candle['volume']:.0f})")
                
            elif success_count >= 4:
                self.logger.info(f"\n⚠️ SISTEMA PARCIALMENTE FUNCIONAL")
            else:
                self.logger.error(f"\n❌ SISTEMA COM FALHAS CRÍTICAS")
            
            self.logger.info("="*100)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Erro durante teste: {e}")
            return {}
    
    def cleanup(self):
        """Limpa recursos"""
        try:
            if self.connection:
                self.logger.info("\n🔌 Desconectando...")
                if hasattr(self.connection, 'disconnect'):
                    self.connection.disconnect()
                    self.logger.info("✅ Desconectado com sucesso!")
        except Exception as e:
            self.logger.error(f"Erro na limpeza: {e}")


def main():
    """Função principal"""
    tester = RealDataFlowTester()
    
    try:
        # Executar teste completo
        results = tester.run_complete_test()
        
        # Determinar sucesso
        success_count = sum(1 for r in results.values() if r)
        
        if success_count >= 4:  # Pelo menos 4 etapas funcionando
            print(f"\n{'='*60}")
            print("🎉 SISTEMA FUNCIONAL COM DADOS REAIS!")
            print(f"{'='*60}")
            return 0
        else:
            print(f"\n{'='*60}")
            print("❌ SISTEMA COM FALHAS CRÍTICAS")
            print(f"{'='*60}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Teste interrompido pelo usuário")
        return 0
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        return 1
    finally:
        # Sempre fazer cleanup
        tester.cleanup()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)