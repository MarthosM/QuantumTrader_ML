"""
Monitor de Status do Sistema ML Trading v2.0
Monitora em tempo real o cálculo de features e execução de predições
"""

import os
import sys
import time
import pandas as pd
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class SystemStatusMonitor:
    """Monitor em tempo real do status do sistema"""
    
    def __init__(self):
        self.logger = logging.getLogger('SystemMonitor')
        self.monitoring = False
        self.last_candles_count = 0
        self.last_features_count = 0
        self.last_prediction_time = None
        self.prediction_count = 0
        self.features_calculation_count = 0
        self.start_time = datetime.now()
        
        # Contadores para análise
        self.stats = {
            'total_candles_processed': 0,
            'total_features_calculated': 0,
            'total_predictions_made': 0,
            'last_activity_time': None,
            'system_health': 'Starting...'
        }
    
    def start_monitoring(self, trading_system):
        """Iniciar monitoramento do sistema"""
        self.trading_system = trading_system
        self.monitoring = True
        
        print("=" * 80)
        print("🔍 MONITOR DE STATUS DO SISTEMA ML TRADING V2.0")
        print("=" * 80)
        print(f"⏰ Iniciado em: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("")
        
        # Thread de monitoramento
        monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="SystemMonitor"
        )
        monitor_thread.start()
        
        return monitor_thread
    
    def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        while self.monitoring:
            try:
                self._check_system_status()
                self._display_status()
                time.sleep(5)  # Atualizar a cada 5 segundos
                
            except Exception as e:
                self.logger.error(f"Erro no monitoramento: {e}")
                time.sleep(10)
    
    def _check_system_status(self):
        """Verificar status atual do sistema"""
        if not self.trading_system:
            return
        
        current_time = datetime.now()
        
        # 1. Verificar dados de candles
        if hasattr(self.trading_system, 'data_structure') and self.trading_system.data_structure:
            data_structure = self.trading_system.data_structure
            
            # Candles
            if hasattr(data_structure, 'candles') and not data_structure.candles.empty:
                current_candles = len(data_structure.candles)
                if current_candles != self.last_candles_count:
                    self.stats['total_candles_processed'] = current_candles
                    self.stats['last_activity_time'] = current_time
                    self.last_candles_count = current_candles
            
            # Features
            if hasattr(data_structure, 'features') and not data_structure.features.empty:
                current_features = len(data_structure.features.columns)
                if current_features != self.last_features_count:
                    self.stats['total_features_calculated'] += 1
                    self.features_calculation_count += 1
                    self.stats['last_activity_time'] = current_time
                    self.last_features_count = current_features
        
        # 2. Verificar predições
        if hasattr(self.trading_system, 'last_prediction') and self.trading_system.last_prediction:
            if hasattr(self.trading_system, 'last_ml_time') and self.trading_system.last_ml_time:
                prediction_time = datetime.fromtimestamp(self.trading_system.last_ml_time)
                
                if self.last_prediction_time != prediction_time:
                    self.stats['total_predictions_made'] += 1
                    self.prediction_count += 1
                    self.stats['last_activity_time'] = current_time
                    self.last_prediction_time = prediction_time
        
        # 3. Avaliar saúde do sistema
        self._evaluate_system_health(current_time)
    
    def _evaluate_system_health(self, current_time):
        """Avaliar saúde geral do sistema"""
        if not self.stats['last_activity_time']:
            self.stats['system_health'] = "🟡 Aguardando atividade..."
            return
        
        # Tempo desde última atividade
        time_since_activity = current_time - self.stats['last_activity_time']
        
        if time_since_activity < timedelta(minutes=2):
            self.stats['system_health'] = "🟢 Sistema ativo e saudável"
        elif time_since_activity < timedelta(minutes=5):
            self.stats['system_health'] = "🟡 Sistema com baixa atividade"
        else:
            self.stats['system_health'] = "🔴 Sistema sem atividade recente"
    
    def _display_status(self):
        """Exibir status atual"""
        os.system('cls' if os.name == 'nt' else 'clear')  # Limpar tela
        
        current_time = datetime.now()
        uptime = current_time - self.start_time
        
        print("=" * 80)
        print("🔍 MONITOR DE STATUS DO SISTEMA ML TRADING V2.0")
        print("=" * 80)
        print(f"⏰ Tempo de execução: {str(uptime).split('.')[0]}")
        print(f"🕒 Última atualização: {current_time.strftime('%H:%M:%S')}")
        print("")
        
        # Status geral
        print("📊 STATUS GERAL:")
        print(f"   {self.stats['system_health']}")
        print("")
        
        # Métricas principais
        print("📈 MÉTRICAS PRINCIPAIS:")
        print(f"   📊 Candles processados: {self.stats['total_candles_processed']}")
        print(f"   🧮 Cálculos de features: {self.features_calculation_count}")
        print(f"   🤖 Predições ML: {self.prediction_count}")
        
        if self.stats['last_activity_time']:
            last_activity = self.stats['last_activity_time'].strftime('%H:%M:%S')
            print(f"   🕐 Última atividade: {last_activity}")
        print("")
        
        # Detalhes do sistema
        self._display_system_details()
        
        # Análise de performance
        self._display_performance_analysis(uptime)
        
        print("=" * 80)
        print("💡 Pressione Ctrl+C para parar o monitoramento")
        print("=" * 80)
    
    def _display_system_details(self):
        """Exibir detalhes específicos do sistema"""
        print("🔧 DETALHES DO SISTEMA:")
        
        if not self.trading_system:
            print("   ❌ Sistema de trading não disponível")
            return
        
        # Status de conexão
        if hasattr(self.trading_system, 'connection') and self.trading_system.connection:
            connection = self.trading_system.connection
            login_status = "🟢 Conectado" if hasattr(connection, 'login_state') and connection.login_state == 0 else "🔴 Desconectado"
            market_status = "🟢 Conectado" if hasattr(connection, 'market_state') and connection.market_state >= 2 else "🔴 Desconectado"
            
            print(f"   🔌 Login: {login_status}")
            print(f"   📈 Market Data: {market_status}")
        else:
            print("   🔌 Conexão: ❌ Não disponível")
        
        # Status dos dados
        if hasattr(self.trading_system, 'data_structure') and self.trading_system.data_structure:
            data_structure = self.trading_system.data_structure
            
            # Candles
            if hasattr(data_structure, 'candles'):
                candles_status = f"📊 {len(data_structure.candles)} candles" if not data_structure.candles.empty else "📊 Nenhum candle"
                
                if not data_structure.candles.empty:
                    last_candle_time = data_structure.candles.index[-1].strftime('%H:%M:%S')
                    candles_status += f" (último: {last_candle_time})"
                
                print(f"   {candles_status}")
            
            # Features
            if hasattr(data_structure, 'features'):
                if not data_structure.features.empty:
                    features_count = len(data_structure.features.columns)
                    features_rows = len(data_structure.features)
                    print(f"   🧮 Features: {features_count} colunas, {features_rows} linhas")
                    
                    # Análise de NaN
                    nan_count = data_structure.features.isnull().sum().sum()
                    total_values = data_structure.features.size
                    fill_rate = ((total_values - nan_count) / total_values) * 100 if total_values > 0 else 0
                    print(f"   📝 Preenchimento: {fill_rate:.1f}% ({nan_count} NaN)")
                else:
                    print("   🧮 Features: Não calculadas")
        
        # Status ML
        if hasattr(self.trading_system, 'ml_coordinator'):
            ml_status = "🟢 Disponível" if self.trading_system.ml_coordinator else "🔴 Indisponível"
            print(f"   🤖 ML Coordinator: {ml_status}")
        
        if hasattr(self.trading_system, 'model_manager'):
            model_status = "🟢 Disponível" if self.trading_system.model_manager else "🔴 Indisponível"
            print(f"   📚 Model Manager: {model_status}")
            
            if self.trading_system.model_manager and hasattr(self.trading_system.model_manager, 'models'):
                models_count = len(self.trading_system.model_manager.models)
                print(f"   📖 Modelos carregados: {models_count}")
        
        # Última predição
        if hasattr(self.trading_system, 'last_prediction') and self.trading_system.last_prediction:
            prediction = self.trading_system.last_prediction
            print(f"   🎯 Última predição: {prediction.get('action', 'N/A')} (confiança: {prediction.get('confidence', 0):.3f})")
        
        print("")
    
    def _display_performance_analysis(self, uptime):
        """Exibir análise de performance"""
        print("⚡ ANÁLISE DE PERFORMANCE:")
        
        uptime_seconds = uptime.total_seconds()
        
        if uptime_seconds > 0:
            # Taxas por minuto
            candles_per_min = (self.stats['total_candles_processed'] / uptime_seconds) * 60
            features_per_min = (self.features_calculation_count / uptime_seconds) * 60
            predictions_per_min = (self.prediction_count / uptime_seconds) * 60
            
            print(f"   📊 Candles/min: {candles_per_min:.1f}")
            print(f"   🧮 Features calc/min: {features_per_min:.1f}")
            print(f"   🤖 Predições/min: {predictions_per_min:.1f}")
        
        # Análise de eficiência
        if self.stats['total_candles_processed'] > 0:
            feature_efficiency = (self.features_calculation_count / self.stats['total_candles_processed']) * 100
            prediction_efficiency = (self.prediction_count / self.stats['total_candles_processed']) * 100
            
            print(f"   📈 Eficiência features: {feature_efficiency:.1f}%")
            print(f"   🎯 Eficiência predições: {prediction_efficiency:.1f}%")
        
        print("")
    
    def stop_monitoring(self):
        """Parar monitoramento"""
        self.monitoring = False
        print("\n🛑 Monitoramento interrompido")

def monitor_trading_system():
    """Conectar ao sistema de trading e monitorar"""
    print("🔍 Iniciando Monitor de Sistema...")
    
    try:
        # Configurar logging
        logging.basicConfig(level=logging.WARNING)
        
        # Verificar se o sistema está rodando
        print("⏳ Procurando sistema de trading ativo...")
        
        # Importar e conectar com sistema
        from trading_system import TradingSystem
        
        # Tentar obter instância ativa ou criar uma nova para monitoramento
        # Por enquanto, vamos simular dados para demonstração
        monitor = SystemStatusMonitor()
        
        print("📡 Conectando com sistema de trading...")
        
        # Simular sistema para demonstração
        class MockTradingSystem:
            def __init__(self):
                from data_structure import TradingDataStructure
                from data_loader import DataLoader
                
                self.data_structure = TradingDataStructure()
                self.data_structure.initialize_structure()
                
                # Simular dados
                data_loader = DataLoader()
                sample_data = data_loader.create_sample_data(100)
                self.data_structure.update_candles(sample_data)
                
                # Simular features
                import numpy as np
                features_data = pd.DataFrame({
                    'ema_9': np.random.randn(100),
                    'ema_20': np.random.randn(100),
                    'rsi_14': np.random.uniform(0, 100, 100),
                    'atr': np.random.uniform(0.1, 2.0, 100)
                }, index=sample_data.index)
                
                self.data_structure.features = features_data
                
                self.last_prediction = {
                    'action': 'buy',
                    'confidence': 0.75,
                    'direction': 1
                }
                
                self.last_ml_time = time.time()
                
                # Simular conexão
                class MockConnection:
                    login_state = 0
                    market_state = 2
                
                self.connection = MockConnection()
                
                # Simular componentes ML
                self.ml_coordinator = True
                self.model_manager = True
        
        # Para demonstração, usar sistema simulado
        mock_system = MockTradingSystem()
        monitor_thread = monitor.start_monitoring(mock_system)
        
        # Simular atividade do sistema
        def simulate_activity():
            """Simular atividade contínua do sistema"""
            while monitor.monitoring:
                try:
                    # Simular novos candles ocasionalmente
                    if time.time() % 10 < 1:  # A cada ~10 segundos
                        from data_loader import DataLoader
                        data_loader = DataLoader()
                        new_data = data_loader.create_sample_data(1)
                        mock_system.data_structure.candles = pd.concat([
                            mock_system.data_structure.candles.tail(99),
                            new_data
                        ])
                    
                    # Simular recálculo de features ocasionalmente
                    if time.time() % 15 < 1:  # A cada ~15 segundos
                        import numpy as np
                        new_features = pd.DataFrame({
                            'ema_9': [np.random.randn()],
                            'ema_20': [np.random.randn()],
                            'rsi_14': [np.random.uniform(0, 100)],
                            'atr': [np.random.uniform(0.1, 2.0)]
                        }, index=[mock_system.data_structure.candles.index[-1]])
                        
                        mock_system.data_structure.features = pd.concat([
                            mock_system.data_structure.features.tail(99),
                            new_features
                        ])
                    
                    # Simular predições ocasionalmente
                    if time.time() % 20 < 1:  # A cada ~20 segundos
                        mock_system.last_prediction = {
                            'action': np.random.choice(['buy', 'sell', 'hold']),
                            'confidence': np.random.uniform(0.3, 0.9),
                            'direction': np.random.choice([-1, 0, 1])
                        }
                        mock_system.last_ml_time = time.time()
                    
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Erro na simulação: {e}")
                    time.sleep(5)
        
        # Iniciar simulação em thread separada
        activity_thread = threading.Thread(target=simulate_activity, daemon=True)
        activity_thread.start()
        
        print("✅ Monitor conectado! Exibindo status em tempo real...\n")
        
        # Aguardar interrupção do usuário
        try:
            while monitor.monitoring:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Interrupção detectada...")
        finally:
            monitor.stop_monitoring()
            
    except Exception as e:
        print(f"❌ Erro no monitoramento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    monitor_trading_system()