# src/execution/execution_integration.py
import logging
from typing import Dict, Optional

class ExecutionIntegration:
    """Integra execução de ordens com sistema de trading"""
    
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.logger = logging.getLogger(__name__)
        
        # Componentes
        self.order_manager = None
        self.execution_engine = None
        
    def initialize_execution_system(self):
        """Inicializa sistema de execução"""
        try:
            # Criar order manager
            from src.execution.order_manager import OrderExecutionManager
            self.order_manager = OrderExecutionManager(
                self.trading_system.connection_manager
            )
            self.order_manager.initialize()
            
            # Criar execution engine
            from src.execution.execution_engine import SimpleExecutionEngine
            self.execution_engine = SimpleExecutionEngine(
                self.order_manager,
                self.trading_system.ml_coordinator,
                self.trading_system.risk_manager
            )
            
            # Registrar no sistema principal
            self.trading_system.execution_engine = self.execution_engine
            
            # Conectar callbacks do ML
            self.trading_system.ml_coordinator.register_signal_callback(
                self._on_ml_signal
            )
            
            self.logger.info("Sistema de execução inicializado")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro inicializando execução: {e}")
            return False
    
    def _on_ml_signal(self, signal: Dict):
        """Callback para sinais do ML"""
        try:
            # Processar sinal
            order_id = self.execution_engine.process_ml_signal(signal)
            
            if order_id:
                self.logger.info(f"Ordem executada para sinal ML: {order_id}")
                
                # Atualizar métricas
                self.trading_system.metrics_collector.record_order_sent(signal)
            
        except Exception as e:
            self.logger.error(f"Erro processando sinal ML: {e}")
    
    def get_execution_status(self) -> Dict:
        """Retorna status do sistema de execução"""
        if not self.execution_engine:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'operational',
            'active_orders': len(self.execution_engine.get_active_orders()),
            'positions': self.execution_engine.get_positions(),
            'stats': self.execution_engine.get_execution_stats()
        }