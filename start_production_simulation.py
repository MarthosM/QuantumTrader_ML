"""
Script de Produção com Simulação - Para Mercado Fechado
Gera dados simulados para testar o sistema quando o mercado está fechado
"""

import os
import sys
import time
import json
import logging
import threading
import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Carregar configurações
load_dotenv('.env.production')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/simulation_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SimulationSystem')

# Importar componentes
from enhanced_production_system import EnhancedProductionSystem
from src.logging.structured_logger import TradingLogger
from src.metrics.metrics_and_alerts import TradingMetricsSystem


class SimulatedDataGenerator:
    """Gera dados simulados para teste"""
    
    def __init__(self):
        self.base_price = 5450.0  # Preço base WDO
        self.volatility = 0.002   # Volatilidade 0.2%
        self.tick_size = 0.5
        self.volume_base = 100
        self.last_price = self.base_price
        
    def generate_book_data(self):
        """Gera book simulado"""
        spread = random.uniform(0.5, 2.0)
        mid_price = self.last_price
        
        # Gerar 5 níveis de bid/ask
        bids = []
        asks = []
        
        for i in range(5):
            bid_price = mid_price - (i + 1) * self.tick_size - spread/2
            ask_price = mid_price + (i + 1) * self.tick_size + spread/2
            
            bid_volume = random.randint(50, 200) * (5 - i)  # Mais volume nos níveis próximos
            ask_volume = random.randint(50, 200) * (5 - i)
            
            bids.append({'price': bid_price, 'volume': bid_volume})
            asks.append({'price': ask_price, 'volume': ask_volume})
        
        return {
            'timestamp': datetime.now(),
            'bids': bids,
            'asks': asks,
            'spread': spread,
            'mid_price': mid_price
        }
    
    def generate_tick_data(self):
        """Gera tick/trade simulado"""
        # Random walk
        change = np.random.randn() * self.volatility * self.base_price
        self.last_price += change
        
        # Arredondar para tick size
        self.last_price = round(self.last_price / self.tick_size) * self.tick_size
        
        return {
            'timestamp': datetime.now(),
            'price': self.last_price,
            'volume': random.randint(1, 10) * 10,
            'side': random.choice(['BUY', 'SELL']),
            'aggressor': random.choice(['BUY', 'SELL'])
        }
    
    def generate_candle_data(self):
        """Gera candle simulado"""
        open_price = self.last_price
        high = open_price + random.uniform(0, 5) * self.tick_size
        low = open_price - random.uniform(0, 5) * self.tick_size
        close = random.uniform(low, high)
        
        return {
            'timestamp': datetime.now(),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': random.randint(100, 1000)
        }


class QuantumTraderSimulation:
    """Sistema de simulação para mercado fechado"""
    
    def __init__(self):
        logger.info("=" * 70)
        logger.info(" QUANTUM TRADER ML - MODO SIMULAÇÃO")
        logger.info(" Para teste com mercado fechado")
        logger.info("=" * 70)
        
        self.running = False
        self.components = {}
        self.data_generator = SimulatedDataGenerator()
        self.simulation_thread = None
        
        # Carregar configuração
        self.load_configuration()
        
        # Inicializar componentes
        self.initialize_components()
    
    def load_configuration(self):
        """Carrega configuração"""
        config_path = Path('config_production.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"[OK] Configuração carregada")
        else:
            self.config = {'system': {'name': 'Simulation'}}
            logger.warning("[AVISO] Usando configuração padrão")
    
    def initialize_components(self):
        """Inicializa componentes sem ProfitDLL"""
        
        # Sistema Enhanced (modificado para não usar callbacks)
        logger.info("\n[1/3] Inicializando Sistema Enhanced...")
        try:
            self.components['production'] = EnhancedProductionSystem()
            logger.info("  [OK] Sistema Enhanced inicializado")
        except Exception as e:
            logger.error(f"  [ERRO] {e}")
        
        # Logging estruturado
        logger.info("\n[2/3] Inicializando Logging...")
        try:
            self.components['logger'] = TradingLogger()
            logger.info("  [OK] Logging inicializado")
        except Exception as e:
            logger.warning(f"  [AVISO] {e}")
        
        # Métricas
        logger.info("\n[3/3] Inicializando Métricas...")
        try:
            self.components['metrics'] = TradingMetricsSystem(0)
            logger.info("  [OK] Métricas inicializadas")
        except Exception as e:
            logger.warning(f"  [AVISO] {e}")
    
    def simulate_data_feed(self):
        """Thread que simula dados de mercado"""
        logger.info("\n[SIMULAÇÃO] Iniciando feed de dados simulados...")
        
        production = self.components.get('production')
        if not production:
            logger.error("Sistema Enhanced não disponível")
            return
        
        iteration = 0
        
        while self.running:
            try:
                iteration += 1
                
                # Gerar dados simulados
                book_data = self.data_generator.generate_book_data()
                tick_data = self.data_generator.generate_tick_data()
                
                # A cada 10 iterações, gerar um candle
                if iteration % 10 == 0:
                    candle_data = self.data_generator.generate_candle_data()
                    
                    # Alimentar buffers
                    if hasattr(production, 'candle_buffer'):
                        production.candle_buffer.add_candle(
                            timestamp=candle_data['timestamp'],
                            open=candle_data['open'],
                            high=candle_data['high'],
                            low=candle_data['low'],
                            close=candle_data['close'],
                            volume=candle_data['volume']
                        )
                
                # Alimentar book manager
                if hasattr(production, 'book_manager'):
                    production.book_manager.on_price_book_callback(book_data)
                
                # Alimentar trade buffer
                if hasattr(production, 'trade_buffer'):
                    production.trade_buffer.add_trade(
                        timestamp=tick_data['timestamp'],
                        price=tick_data['price'],
                        volume=tick_data['volume'],
                        side=tick_data['side']
                    )
                
                # Calcular features a cada 5 iterações
                if iteration % 5 == 0:
                    self.process_features()
                
                # Log periódico
                if iteration % 100 == 0:
                    logger.info(f"[SIMULAÇÃO] Iteração {iteration} - Preço: {tick_data['price']:.1f}")
                
                # Aguardar para simular tempo real
                time.sleep(0.5)  # 2 updates por segundo
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Erro na simulação: {e}")
                time.sleep(1)
    
    def process_features(self):
        """Processa features e simula trading"""
        try:
            production = self.components.get('production')
            if not production:
                return
            
            # Calcular features
            features = production._calculate_features()
            
            if len(features) > 0:
                logger.debug(f"Features calculadas: {len(features)}")
                
                # Fazer predição ML simulada
                ml_prediction = random.uniform(-0.5, 0.5)  # Simulado
                
                # Log estruturado
                if 'logger' in self.components:
                    self.components['logger'].log_feature_calculation(features, 1.5)
                    self.components['logger'].log_prediction(ml_prediction, 0.65, len(features))
                
                # Métricas
                if 'metrics' in self.components:
                    self.components['metrics'].record_feature_calculation(len(features), 1.5)
                    self.components['metrics'].record_prediction(0.65, 2.0)
                
                # Simular decisão de trading
                if abs(ml_prediction) > 0.3:
                    side = 'BUY' if ml_prediction > 0 else 'SELL'
                    logger.info(f"[TRADE SIMULADO] {side} - Confiança: {abs(ml_prediction):.2%}")
                    
                    if 'metrics' in self.components:
                        self.components['metrics'].record_trade(side, random.uniform(-100, 200))
                        
        except Exception as e:
            logger.error(f"Erro ao processar features: {e}")
    
    def start(self):
        """Inicia sistema de simulação"""
        self.running = True
        
        logger.info("\n" + "=" * 70)
        logger.info(" SIMULAÇÃO INICIADA")
        logger.info("=" * 70)
        
        # Iniciar thread de simulação
        self.simulation_thread = threading.Thread(target=self.simulate_data_feed, daemon=True)
        self.simulation_thread.start()
        
        # Loop principal
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n[AVISO] Interrupção detectada")
        finally:
            self.stop()
    
    def stop(self):
        """Para o sistema"""
        logger.info("\n[PARANDO] Sistema de simulação...")
        self.running = False
        
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=5)
        
        # Salvar métricas finais
        if 'metrics' in self.components:
            try:
                metrics_data = self.components['metrics'].get_dashboard_data()
                with open('metrics/simulation_final.json', 'w') as f:
                    json.dump(metrics_data, f, indent=2)
                logger.info("  [OK] Métricas salvas")
            except:
                pass
        
        logger.info("\n" + "=" * 70)
        logger.info(" SIMULAÇÃO FINALIZADA")
        logger.info("=" * 70)


def main():
    """Função principal"""
    
    print("\n" + "=" * 70)
    print(" MODO SIMULAÇÃO - Para teste com mercado fechado")
    print("=" * 70)
    print("\nEste modo gera dados simulados para testar o sistema")
    print("quando o mercado está fechado ou sem conexão com Profit.")
    print("\n[AVISO] NÃO use este modo em produção real!")
    print("=" * 70)
    
    response = input("\nIniciar simulação? (s/n): ")
    if response.lower() != 's':
        return
    
    # Criar diretórios necessários
    for dir_name in ['logs', 'metrics', 'data/book_tick_data']:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    # Iniciar simulação
    system = QuantumTraderSimulation()
    system.start()


if __name__ == "__main__":
    main()