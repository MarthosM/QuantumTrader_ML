#!/usr/bin/env python3
"""
Teste Completo do Sistema de Fluxo de Dados ML
Sistema de Trading v2.0

Este script testa o fluxo completo:
1. Criação de candles simulados
2. Cálculo de features
3. Execução de predições
4. Exibição no GUI

OBJETIVO: Confirmar que o dataframe de features está sendo
calculado corretamente a cada novo candle e que as predições
são executadas e exibidas no monitor GUI.
"""

import os
import sys
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
from typing import Dict, List, Any

# Adicionar src ao path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Imports locais
from data_flow_monitor import DataFlowMonitor, PredictionResult
from gui_prediction_extension import extend_gui_with_prediction_display
from ml_data_flow_integrator import MLDataFlowIntegrator


class MockTradingSystem:
    """Sistema de trading simulado para testes"""
    
    def __init__(self):
        self.logger = logging.getLogger('MockSystem')
        self.is_running = True
        
        # Simular componentes
        self.data_structure = MockDataStructure()
        self.feature_engine = MockFeatureEngine()
        self.ml_coordinator = MockMLCoordinator()
        self.monitor: Any = None  # Será definido se GUI for criado
        
        # Estado
        self.candle_count = 0
        
    def initialize(self) -> bool:
        return True
        
    def start(self) -> bool:
        return True
        
    def stop(self):
        self.is_running = False


class MockDataStructure:
    """Estrutura de dados simulada"""
    
    def __init__(self):
        self.logger = logging.getLogger('MockDataStructure')
        
        # DataFrames simulados
        self.candles_df = self._create_initial_candles()
        self.microstructure_df = pd.DataFrame()
        self.orderbook_df = pd.DataFrame()
        
        # Callbacks
        self.candle_callbacks = []
        
    def _create_initial_candles(self) -> pd.DataFrame:
        """Cria candles iniciais para teste"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=1),
            end=datetime.now(),
            freq='1min'
        )
        
        np.random.seed(42)  # Para resultados consistentes
        base_price = 125000.0
        
        data = []
        price = base_price
        
        for date in dates:
            # Simulação de movimento de preço
            change = np.random.normal(0, 50)  # Volatilidade de 50 pontos
            
            open_price = price
            high_price = open_price + abs(np.random.normal(0, 30))
            low_price = open_price - abs(np.random.normal(0, 30))
            close_price = open_price + change
            volume = int(np.random.uniform(100, 2000))
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
            
            price = close_price
            
        df = pd.DataFrame(data, index=dates)
        self.logger.info(f"Candles iniciais criados: {len(df)} registros")
        
        return df
        
    def add_new_candle(self):
        """Adiciona novo candle simulado"""
        last_timestamp = self.candles_df.index[-1]
        new_timestamp = last_timestamp + timedelta(minutes=1)
        
        # Preço baseado no último candle
        last_close = self.candles_df['close'].iloc[-1]
        change = np.random.normal(0, 50)
        
        open_price = last_close
        high_price = open_price + abs(np.random.normal(0, 30))
        low_price = open_price - abs(np.random.normal(0, 30))
        close_price = open_price + change
        volume = int(np.random.uniform(100, 2000))
        
        new_candle = pd.DataFrame({
            'open': [open_price],
            'high': [high_price],
            'low': [low_price],
            'close': [close_price],
            'volume': [volume]
        }, index=[new_timestamp])
        
        # Adicionar ao DataFrame
        self.candles_df = pd.concat([self.candles_df, new_candle])
        
        # Manter apenas últimos 100 candles
        if len(self.candles_df) > 100:
            self.candles_df = self.candles_df.tail(100)
            
        self.logger.info(f"Novo candle adicionado: {new_timestamp} - Close: {close_price:.2f}")
        
        # Chamar callbacks
        for callback in self.candle_callbacks:
            try:
                callback(new_candle.iloc[0])
            except Exception as e:
                self.logger.error(f"Erro em callback: {e}")
                
        return new_candle.iloc[0]
        
    def add_candle_callback(self, callback):
        """Adiciona callback para novos candles"""
        self.candle_callbacks.append(callback)


class MockFeatureEngine:
    """Feature engine simulado"""
    
    def __init__(self):
        self.logger = logging.getLogger('MockFeatureEngine')
        
    def request_indicator_calculation(self, candles_df: pd.DataFrame) -> Dict:
        """Simula cálculo de indicadores"""
        try:
            df = candles_df.copy()
            
            # Calcular indicadores básicos
            df['ema_9'] = df['close'].ewm(span=9).mean()
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            
            # RSI simulado
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)  # Evitar divisão por zero
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # ATR simulado
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.DataFrame({
                'hl': high_low,
                'hc': high_close,
                'lc': low_close
            }).max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # ADX simulado
            df['adx'] = np.random.uniform(20, 80, len(df))
            
            self.logger.debug(f"Indicadores calculados para {len(df)} candles")
            
            return {
                'success': True,
                'data': df
            }
            
        except Exception as e:
            self.logger.error(f"Erro calculando indicadores: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def create_features_separated(self, candles_df: pd.DataFrame, 
                                microstructure_df: pd.DataFrame,
                                indicators_df: pd.DataFrame) -> Dict:
        """Simula criação de features ML"""
        try:
            df = indicators_df.copy()
            
            # Features de momentum
            df['momentum_1'] = df['close'].pct_change(1)
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_10'] = df['close'].pct_change(10)
            
            # Features de volatilidade
            df['volatility_5'] = df['close'].rolling(5).std()
            df['volatility_10'] = df['close'].rolling(10).std()
            df['volatility_20'] = df['close'].rolling(20).std()
            
            # Features de volume
            df['volume_sma_10'] = df['volume'].rolling(10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_10']
            
            # Features de preço
            df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['hl_ratio'] = (df['high'] - df['low']) / df['close']
            
            # Features técnicas adicionais
            df['ema_diff'] = df['ema_9'] - df['ema_20']
            df['price_vs_ema9'] = (df['close'] - df['ema_9']) / df['ema_9']
            df['price_vs_ema20'] = (df['close'] - df['ema_20']) / df['ema_20']
            
            # Features de microestrutura simuladas
            df['spread_sim'] = np.random.uniform(0.5, 5.0, len(df))
            df['depth_sim'] = np.random.uniform(100, 1000, len(df))
            df['imbalance_sim'] = np.random.uniform(-0.5, 0.5, len(df))
            
            # Remover NaN das primeiras linhas
            df = df.dropna()
            
            self.logger.info(f"Features calculadas: {df.shape}")
            self.logger.debug(f"Colunas: {list(df.columns)}")
            
            return {
                'success': True,
                'features_df': df
            }
            
        except Exception as e:
            self.logger.error(f"Erro criando features: {e}")
            return {
                'success': False,
                'error': str(e)
            }


class MockMLCoordinator:
    """Coordenador ML simulado"""
    
    def __init__(self):
        self.logger = logging.getLogger('MockMLCoordinator')
        self.prediction_count = 0
        self.prediction_callbacks = []
        
    def process_prediction_request(self, prediction_data: Dict) -> Dict:
        """Simula processamento de predição"""
        try:
            features_df = prediction_data.get('features_df')
            if features_df is None or features_df.empty:
                return {'error': 'Features DataFrame vazio'}
                
            self.prediction_count += 1
            
            # Simular predição baseada em features
            last_momentum = features_df['momentum_1'].iloc[-1] if 'momentum_1' in features_df.columns else 0
            last_rsi = features_df['rsi'].iloc[-1] if 'rsi' in features_df.columns else 50
            
            # Lógica de predição simulada
            if last_rsi > 70:
                direction = -0.3 + np.random.normal(0, 0.1)  # Tendência de venda
                regime = 'overbought'
            elif last_rsi < 30:
                direction = 0.3 + np.random.normal(0, 0.1)   # Tendência de compra
                regime = 'oversold'
            else:
                direction = last_momentum * 2 + np.random.normal(0, 0.1)  # Seguir momentum
                regime = 'trending' if abs(direction) > 0.1 else 'ranging'
                
            # Magnitude baseada na volatilidade
            volatility = features_df['volatility_10'].iloc[-1] if 'volatility_10' in features_df.columns else 100
            magnitude = min(abs(direction) * (volatility / 100), 1.0)
            
            # Confiança baseada em múltiplos fatores
            confidence = min(0.9, max(0.3, 
                abs(direction) * 2 + 
                (1 - abs(last_rsi - 50) / 50) * 0.3 +
                np.random.uniform(0.1, 0.3)
            ))
            
            result = {
                'trade_decision': {
                    'action': 'buy' if direction > 0.1 else 'sell' if direction < -0.1 else 'hold',
                    'confidence': confidence
                },
                'direction': direction,
                'magnitude': magnitude,
                'confidence': confidence,
                'regime': regime,
                'model_used': f'MockModel_v{self.prediction_count % 3 + 1}',
                'features_used': len(features_df.columns),
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"Predição {self.prediction_count}: {direction:.3f} "
                           f"(confiança: {confidence:.3f})")
            
            # Chamar callbacks
            for callback in self.prediction_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    self.logger.error(f"Erro em callback de predição: {e}")
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Erro processando predição: {e}")
            return {'error': str(e)}
            
    def add_prediction_callback(self, callback):
        """Adiciona callback para predições"""
        self.prediction_callbacks.append(callback)


class DataFlowTester:
    """Testador completo do fluxo de dados"""
    
    def __init__(self, use_gui: bool = True):
        self.logger = logging.getLogger('DataFlowTester')
        self.use_gui = use_gui
        
        # Componentes
        self.mock_system = None
        self.integrator = None
        self.gui_thread = None
        
        # Controle de teste
        self.test_running = False
        self.candle_generation_thread = None
        
    def setup_test_environment(self) -> bool:
        """Configura ambiente de teste"""
        try:
            self.logger.info("🔧 Configurando ambiente de teste...")
            
            # 1. Criar sistema simulado
            self.mock_system = MockTradingSystem()
            
            # 2. Configurar GUI se solicitado
            if self.use_gui:
                success = self._setup_gui()
                if not success:
                    self.logger.warning("Falha configurando GUI, continuando sem interface visual")
                    self.use_gui = False
                    
            # 3. Criar integrador
            self.integrator = MLDataFlowIntegrator(self.mock_system)
            
            # 4. Inicializar integrador
            if not self.integrator.initialize():
                self.logger.error("Falha inicializando integrador")
                return False
                
            self.logger.info("✅ Ambiente de teste configurado")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro configurando ambiente: {e}")
            return False
            
    def _setup_gui(self) -> bool:
        """Configura GUI para teste"""
        try:
            import tkinter as tk
            from gui_prediction_extension import PredictionDisplayPanel, DataFlowStatusPanel
            
            # Criar janela principal
            root = tk.Tk()
            root.title("Teste - Fluxo de Dados ML")
            root.geometry("1000x700")
            
            # Criar mock do monitor GUI
            class MockMonitor:
                def __init__(self, root):
                    self.root = root
                    self.logger = logging.getLogger('MockMonitor')
                    self.prediction_panel = PredictionDisplayPanel(root, self.logger)
                    self.flow_status_panel = DataFlowStatusPanel(root, self.logger)
                    
                def update_prediction_data(self, data):
                    self.prediction_panel.update_prediction_data(data)
                    
                def update_flow_status(self, data):
                    self.flow_status_panel.update_flow_status(data)
                    
                def run(self):
                    self.root.mainloop()
                    
            self.mock_system.monitor = MockMonitor(root)
            
            # Iniciar GUI em thread separada
            self.gui_thread = threading.Thread(
                target=self.mock_system.monitor.run,
                daemon=True,
                name="TestGUI"
            )
            self.gui_thread.start()
            
            # Aguardar inicialização
            time.sleep(1)
            
            self.logger.info("✅ GUI de teste configurado")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro configurando GUI: {e}")
            return False
            
    def start_test(self, duration_minutes: int = 5, candle_interval_seconds: int = 10):
        """Inicia teste do fluxo de dados"""
        try:
            self.logger.info(f"🚀 Iniciando teste por {duration_minutes} minutos...")
            self.logger.info(f"📊 Novos candles a cada {candle_interval_seconds} segundos")
            
            # Configurar ambiente se necessário
            if not self.mock_system:
                if not self.setup_test_environment():
                    return False
                    
            # Iniciar integrador
            if not self.integrator.start_integration():
                self.logger.error("Falha iniciando integração")
                return False
                
            # Iniciar geração de candles
            self.test_running = True
            self.candle_generation_thread = threading.Thread(
                target=self._candle_generation_loop,
                args=(candle_interval_seconds,),
                daemon=True,
                name="CandleGenerator"
            )
            self.candle_generation_thread.start()
            
            # Executar teste por duração especificada
            self.logger.info("⏰ Teste em execução...")
            
            start_time = time.time()
            while time.time() - start_time < duration_minutes * 60 and self.test_running:
                # Imprimir status a cada 30 segundos
                if int(time.time() - start_time) % 30 == 0:
                    self.print_test_status()
                    
                time.sleep(1)
                
            # Parar teste
            self.stop_test()
            
            self.logger.info("✅ Teste concluído")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro executando teste: {e}")
            return False
            
    def _candle_generation_loop(self, interval_seconds: int):
        """Loop de geração de candles"""
        self.logger.info(f"📈 Iniciando geração de candles (intervalo: {interval_seconds}s)")
        
        candle_count = 0
        while self.test_running:
            try:
                # Gerar novo candle
                new_candle = self.mock_system.data_structure.add_new_candle()
                candle_count += 1
                
                self.logger.info(f"🕯️ Candle {candle_count} gerado: "
                               f"Close={new_candle['close']:.2f}, "
                               f"Volume={new_candle['volume']}")
                
                # Aguardar próximo candle
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Erro gerando candle: {e}")
                time.sleep(5)
                
        self.logger.info(f"📊 Geração de candles parada. Total gerado: {candle_count}")
        
    def stop_test(self):
        """Para o teste"""
        self.test_running = False
        
        if self.integrator:
            self.integrator.stop_integration()
            
        if self.candle_generation_thread and self.candle_generation_thread.is_alive():
            self.candle_generation_thread.join(timeout=5)
            
        self.logger.info("⏹️ Teste parado")
        
    def print_test_status(self):
        """Imprime status do teste"""
        if self.integrator:
            self.integrator.print_integration_status()
            
        # Status do sistema mock
        candles_count = len(self.mock_system.data_structure.candles_df)
        last_close = self.mock_system.data_structure.candles_df['close'].iloc[-1]
        
        print(f"\n📊 STATUS DO SISTEMA MOCK:")
        print(f"  • Total de Candles: {candles_count}")
        print(f"  • Último Preço: {last_close:.2f}")
        print(f"  • Sistema Rodando: {self.mock_system.is_running}")
        
    def run_simple_test(self):
        """Executa teste simples sem GUI"""
        self.use_gui = False
        
        if not self.setup_test_environment():
            return False
            
        # Teste básico: gerar alguns candles e verificar predições
        self.logger.info("🧪 Executando teste simples...")
        
        # Iniciar integração
        if not self.integrator.start_integration():
            return False
            
        # Gerar 5 candles com intervalo de 2 segundos
        for i in range(5):
            self.logger.info(f"📈 Gerando candle {i+1}/5...")
            new_candle = self.mock_system.data_structure.add_new_candle()
            
            # Aguardar processamento
            time.sleep(3)
            
            # Verificar se predição foi gerada
            if self.integrator.data_flow_monitor.current_prediction:
                pred = self.integrator.data_flow_monitor.current_prediction
                self.logger.info(f"✅ Predição gerada: {pred.direction:.3f} "
                               f"(confiança: {pred.confidence:.3f})")
            else:
                self.logger.warning("⚠️ Nenhuma predição gerada ainda")
                
        # Status final
        self.print_test_status()
        
        self.stop_test()
        return True


def main():
    """Função principal de teste"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('DataFlowTest')
    logger.info("🧪 Iniciando teste completo do fluxo de dados ML...")
    
    try:
        # Criar testador
        tester = DataFlowTester(use_gui=True)  # Mudar para False se não quiser GUI
        
        # Escolher tipo de teste
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == 'simple':
            # Teste simples sem GUI
            success = tester.run_simple_test()
        else:
            # Teste completo com GUI
            success = tester.start_test(duration_minutes=2, candle_interval_seconds=5)
            
        if success:
            logger.info("🎉 Teste concluído com sucesso!")
        else:
            logger.error("❌ Teste falhou")
            
    except KeyboardInterrupt:
        logger.info("⏹️ Teste interrompido pelo usuário")
        
    except Exception as e:
        logger.error(f"Erro no teste: {e}", exc_info=True)


if __name__ == "__main__":
    main()
