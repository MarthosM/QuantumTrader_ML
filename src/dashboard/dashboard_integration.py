"""
Integração do Dashboard com o Sistema de Trading
Facilita a inicialização e conexão dos componentes
"""

import threading
import logging
from typing import Optional
import sys
import os

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dashboard.app import dashboard_server, start_dashboard
from src.portfolio.position_tracker import PositionTracker
from src.risk.risk_manager import RiskManager
from src.execution.order_manager import OrderManager
from src.data.data_synchronizer import DataSynchronizer

class DashboardIntegration:
    """
    Classe para integrar o dashboard com o sistema de trading
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Thread do servidor
        self.server_thread = None
        self.is_running = False
        
        # Configurações do servidor
        self.host = config.get('dashboard_host', '0.0.0.0')
        self.port = config.get('dashboard_port', 5000)
        self.debug = config.get('dashboard_debug', False)
        
    def initialize_components(self, 
                            position_tracker: Optional[PositionTracker] = None,
                            risk_manager: Optional[RiskManager] = None,
                            order_manager: Optional[OrderManager] = None,
                            data_synchronizer: Optional[DataSynchronizer] = None):
        """
        Inicializa o dashboard com os componentes do sistema
        
        Args:
            position_tracker: Instância do PositionTracker
            risk_manager: Instância do RiskManager
            order_manager: Instância do OrderManager
            data_synchronizer: Instância do DataSynchronizer
        """
        self.logger.info("Inicializando componentes do dashboard")
        
        # Inicializar servidor com componentes
        dashboard_server.initialize(
            position_tracker=position_tracker,
            risk_manager=risk_manager,
            order_manager=order_manager,
            data_synchronizer=data_synchronizer
        )
        
        self.logger.info("Componentes do dashboard inicializados")
        
    def start(self):
        """Inicia o servidor do dashboard em thread separada"""
        if self.is_running:
            self.logger.warning("Dashboard já está em execução")
            return
            
        self.logger.info(f"Iniciando dashboard em {self.host}:{self.port}")
        
        # Criar thread para o servidor
        self.server_thread = threading.Thread(
            target=self._run_server,
            name="DashboardServer"
        )
        self.server_thread.daemon = True
        self.server_thread.start()
        
        self.is_running = True
        self.logger.info("Dashboard iniciado com sucesso")
        
    def _run_server(self):
        """Executa o servidor Flask"""
        try:
            start_dashboard(
                host=self.host,
                port=self.port,
                debug=self.debug
            )
        except Exception as e:
            self.logger.error(f"Erro ao executar servidor do dashboard: {e}")
            self.is_running = False
            
    def stop(self):
        """Para o servidor do dashboard"""
        if not self.is_running:
            return
            
        self.logger.info("Parando dashboard...")
        self.is_running = False
        
        # O Flask/SocketIO não tem uma forma limpa de parar
        # Em produção, você usaria um servidor WSGI apropriado
        self.logger.info("Dashboard parado")
        
    def get_url(self) -> str:
        """Retorna a URL do dashboard"""
        return f"http://{self.host if self.host != '0.0.0.0' else 'localhost'}:{self.port}"


# Exemplo de uso standalone
def create_standalone_dashboard(config: dict = None):
    """
    Cria um dashboard standalone para desenvolvimento/testes
    
    Args:
        config: Configurações do dashboard
        
    Returns:
        DashboardIntegration configurado
    """
    if config is None:
        config = {
            'dashboard_host': '127.0.0.1',
            'dashboard_port': 5000,
            'dashboard_debug': True,
            'initial_capital': 100000.0
        }
    
    # Criar componentes mock para testes
    position_tracker = PositionTracker(config)
    risk_manager = RiskManager(config)
    order_manager = OrderManager(config)
    data_synchronizer = DataSynchronizer(config)
    
    # Iniciar componentes
    for component in [position_tracker, risk_manager, order_manager, data_synchronizer]:
        component.start()
    
    # Criar e inicializar dashboard
    dashboard = DashboardIntegration(config)
    dashboard.initialize_components(
        position_tracker=position_tracker,
        risk_manager=risk_manager,
        order_manager=order_manager,
        data_synchronizer=data_synchronizer
    )
    
    return dashboard


if __name__ == "__main__":
    # Teste standalone
    logging.basicConfig(level=logging.INFO)
    
    dashboard = create_standalone_dashboard()
    dashboard.start()
    
    print(f"\n🌐 Dashboard disponível em: {dashboard.get_url()}")
    print("Pressione Ctrl+C para parar\n")
    
    try:
        # Manter rodando
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nParando dashboard...")
        dashboard.stop()