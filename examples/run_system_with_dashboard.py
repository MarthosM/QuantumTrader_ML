"""
Exemplo de execução do sistema completo com dashboard
Demonstra a integração de todos os componentes
"""

import sys
import os
import time
import logging
from datetime import datetime
import threading
import random

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports dos componentes
from src.data.data_synchronizer import DataSynchronizer
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.portfolio.position_tracker import PositionTracker
from src.dashboard.dashboard_integration import DashboardIntegration

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingSystemDemo:
    """
    Sistema de trading completo com dashboard
    Para demonstração e testes
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Componentes do sistema
        self.data_sync = DataSynchronizer(config)
        self.order_manager = OrderManager(config)
        self.risk_manager = RiskManager(config)
        self.position_tracker = PositionTracker(config)
        self.dashboard = DashboardIntegration(config)
        
        # Estado
        self.is_running = False
        self.simulation_thread = None
        
    def start(self):
        """Inicia todos os componentes"""
        self.logger.info("Iniciando sistema de trading...")
        
        # Iniciar componentes
        components = [
            self.data_sync,
            self.order_manager,
            self.risk_manager,
            self.position_tracker
        ]
        
        for component in components:
            component.start()
            
        # Inicializar e iniciar dashboard
        self.dashboard.initialize_components(
            position_tracker=self.position_tracker,
            risk_manager=self.risk_manager,
            order_manager=self.order_manager,
            data_synchronizer=self.data_sync
        )
        self.dashboard.start()
        
        # Iniciar simulação
        self.is_running = True
        self.simulation_thread = threading.Thread(
            target=self._simulation_loop,
            name="SimulationLoop"
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        self.logger.info(f"Sistema iniciado! Dashboard: {self.dashboard.get_url()}")
        
    def stop(self):
        """Para todos os componentes"""
        self.logger.info("Parando sistema de trading...")
        
        self.is_running = False
        
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2)
            
        # Parar componentes
        for component in [self.data_sync, self.order_manager, 
                         self.risk_manager, self.position_tracker]:
            component.stop()
            
        self.dashboard.stop()
        
        self.logger.info("Sistema parado")
        
    def _simulation_loop(self):
        """Loop de simulação para demonstração"""
        symbols = ['WDOU25', 'WINV25', 'INDFUT']
        
        while self.is_running:
            try:
                # Simular dados de mercado
                for symbol in symbols:
                    price = 5000 + random.uniform(-100, 100)
                    
                    # Atualizar preços nas posições
                    self.position_tracker.update_price(symbol, price)
                    
                    # Simular dados tick
                    tick_data = {
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'price': price,
                        'volume': random.randint(1, 100)
                    }
                    
                    # Simular dados book
                    book_data = {
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'bid': price - random.uniform(0.5, 2),
                        'ask': price + random.uniform(0.5, 2),
                        'bid_size': random.randint(10, 100),
                        'ask_size': random.randint(10, 100)
                    }
                    
                    # Processar sincronização
                    self.data_sync.process_tick(tick_data)
                    self.data_sync.process_book(book_data)
                
                # Simular sinais de trading (ocasionalmente)
                if random.random() < 0.05:  # 5% de chance
                    self._simulate_trade_signal()
                    
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Erro na simulação: {e}")
                time.sleep(5)
                
    def _simulate_trade_signal(self):
        """Simula um sinal de trading"""
        symbol = random.choice(['WDOU25', 'WINV25', 'INDFUT'])
        side = random.choice(['BUY', 'SELL'])
        
        # Criar sinal
        signal_info = {
            'symbol': symbol,
            'side': side,
            'confidence': random.uniform(0.6, 0.9),
            'regime': random.choice(['trend', 'range']),
            'source': 'simulation'
        }
        
        # Validar com RiskManager
        current_price = 5000 + random.uniform(-100, 100)
        approved, reason = self.risk_manager.validate_signal(signal_info, current_price)
        
        if approved:
            # Calcular tamanho da posição
            position_size = self.risk_manager.calculate_position_size(
                signal_info, 
                current_price,
                self.position_tracker.cash_balance
            )
            
            # Criar ordem
            order = self.order_manager.create_order(
                symbol=symbol,
                side=side,
                quantity=position_size,
                order_type='MARKET',
                signal_info=signal_info
            )
            
            if order:
                # Submeter ordem
                success = self.order_manager.submit_order(order.order_id)
                
                if success:
                    self.logger.info(f"Ordem criada: {side} {position_size} {symbol}")
                    
                    # Simular execução após pequeno delay
                    threading.Timer(
                        random.uniform(0.5, 2.0),
                        self._simulate_order_fill,
                        args=(order, current_price)
                    ).start()
        else:
            self.logger.warning(f"Sinal rejeitado: {reason}")
            
    def _simulate_order_fill(self, order, price):
        """Simula preenchimento de ordem"""
        # Simular callback de execução
        self.order_manager.on_order_update(
            order.broker_order_id or order.order_id,
            {
                'status': 'FILLED',
                'filled_quantity': order.quantity,
                'filled_price': price + random.uniform(-5, 5),
                'commission': order.quantity * 5.0
            }
        )
        
        # Registrar no PositionTracker
        if order.state.value == 'FILLED':
            order_data = {
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'price': order.filled_price,
                'commission': order.commission,
                'order_id': order.order_id
            }
            
            # Verificar se está abrindo ou fechando posição
            position = self.position_tracker.get_position(order.symbol)
            
            if position is None or \
               (position.quantity > 0 and order.side.value == 'BUY') or \
               (position.quantity < 0 and order.side.value == 'SELL'):
                # Abrindo ou aumentando posição
                self.position_tracker.open_position(order_data)
            else:
                # Fechando ou reduzindo posição
                self.position_tracker.close_position(order_data)
                
            # Registrar trade no RiskManager
            self.risk_manager.register_trade({
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'price': order.filled_price
            })


def main():
    """Função principal"""
    # Configuração
    config = {
        # Capital e limites
        'initial_capital': 100000.0,
        'max_position_size': 10,
        'max_open_positions': 5,
        'max_daily_loss': 1000.0,
        'max_drawdown': 0.10,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.03,
        
        # Configurações do sistema
        'max_retry_attempts': 3,
        'order_timeout_seconds': 30,
        'commission_per_contract': 5.0,
        
        # Dashboard
        'dashboard_host': '127.0.0.1',
        'dashboard_port': 5000,
        'dashboard_debug': False
    }
    
    # Criar e iniciar sistema
    system = TradingSystemDemo(config)
    system.start()
    
    print("\n" + "="*60)
    print("🚀 SISTEMA DE TRADING INICIADO")
    print("="*60)
    print(f"📊 Dashboard: {system.dashboard.get_url()}")
    print(f"💰 Capital Inicial: ${config['initial_capital']:,.2f}")
    print(f"📉 Stop Loss: {config['stop_loss_pct']*100:.1f}%")
    print(f"📈 Take Profit: {config['take_profit_pct']*100:.1f}%")
    print("\nPressione Ctrl+C para parar o sistema")
    print("="*60 + "\n")
    
    try:
        # Manter sistema rodando
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nParando sistema...")
        system.stop()
        print("Sistema parado com sucesso!")


if __name__ == "__main__":
    main()