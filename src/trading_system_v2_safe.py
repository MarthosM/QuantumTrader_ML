"""
Trading System V2 Safe - Versão segura sem callback problemático
Baseado no trading_system.py mas com proteções contra segfault
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

from src.connection_manager_v4 import ConnectionManagerV4
from src.data_structure import TradingDataStructure
from src.model_manager import ModelManager
from src.ml_coordinator import MLCoordinator
from src.signal_generator import SignalGenerator
from src.risk_manager import RiskManager
from src.order_manager import OrderManager
from src.data_pipeline import DataPipeline
from src.wdo_contract_manager import WDOContractManager
from src.production_validator import ProductionValidator

class TradingSystemV2Safe:
    """Sistema de Trading v2.0 - Versão Segura"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o sistema de trading
        
        Args:
            config: Dicionário de configuração
        """
        self.config = config
        self.logger = logging.getLogger('TradingSystemV2Safe')
        
        # Determinar ticker correto para WDO
        self._update_wdo_ticker()
        
        # Validador de produção
        self.validator = ProductionValidator()
        
        # Componentes principais
        self.connection = None
        self.data_structure = None
        self.data_pipeline = None
        self.model_manager = None
        self.ml_coordinator = None
        self.signal_generator = None
        self.risk_manager = None
        self.order_manager = None
        
        # Estado do sistema
        self.is_initialized = False
        self.is_running = False
        self.start_time = None
        
        # Estatísticas
        self.stats = {
            'trades': 0,
            'ml_predictions': 0,
            'candles_received': 0,
            'errors': 0
        }
        
        self.logger.info("="*60)
        self.logger.info("Sistema de Trading v2.0 Safe - Sem callback problemático")
        self.logger.info("="*60)
        
    def _update_wdo_ticker(self):
        """Atualiza ticker WDO para o contrato correto"""
        manager = WDOContractManager()
        today = datetime.now()
        correct_ticker = manager.get_current_contract(today)
        
        if self.config.get('ticker', '').startswith('WDO'):
            self.logger.info(f"Para data {today.strftime('%Y-%m-%d')}, usando contrato: {correct_ticker}")
            self.config['ticker'] = correct_ticker
            
    def _create_safe_connection(self):
        """Cria conexão segura sem callback problemático"""
        # Criar wrapper de configuração segura
        safe_config = self.config.copy()
        
        # Criar classe wrapper temporária
        class SafeConnectionManager(ConnectionManagerV4):
            def __init__(self, config):
                super().__init__(config)
                self._account_callback_disabled = True
                
            def _create_account_callback(self):
                """Override para desabilitar callback problemático"""
                from ctypes import WINFUNCTYPE, c_int, c_wchar_p
                
                @WINFUNCTYPE(c_int, c_int, c_wchar_p, c_wchar_p, c_wchar_p)
                def safe_account_callback(broker_id, account_id, account_name, titular):
                    # Apenas logar sem processar
                    self.logger.info(f"[SAFE] Account info recebido mas não processado (prevenção segfault)")
                    return 0
                    
                return safe_account_callback
        
        # Usar conexão segura
        return SafeConnectionManager(safe_config)
        
    def initialize(self) -> bool:
        """Inicializa todos os componentes do sistema"""
        try:
            self.logger.info("1. Inicializando conexão segura...")
            self.connection = self._create_safe_connection()
            
            if not self.connection.connect():
                self.logger.error("Falha ao conectar")
                return False
                
            self.logger.info("2. Inicializando estrutura de dados...")
            self.data_structure = TradingDataStructure()
            
            self.logger.info("3. Inicializando pipeline de dados...")
            self.data_pipeline = DataPipeline(
                self.connection,
                self.data_structure,
                self.config
            )
            
            self.logger.info("4. Carregando modelos ML...")
            self.model_manager = ModelManager(self.config.get('models_dir', 'models'))
            models_loaded = self.model_manager.load_models()
            
            if models_loaded:
                self.logger.info(f"   {len(self.model_manager.models)} modelos carregados")
            else:
                self.logger.warning("   Nenhum modelo ML carregado - operando sem ML")
                
            self.logger.info("5. Inicializando coordenador ML...")
            self.ml_coordinator = MLCoordinator(
                self.model_manager,
                self.data_structure,
                self.config
            )
            
            self.logger.info("6. Inicializando gerador de sinais...")
            self.signal_generator = SignalGenerator(self.config)
            
            self.logger.info("7. Inicializando gerenciador de risco...")
            self.risk_manager = RiskManager(self.config.get('risk', {}))
            
            self.logger.info("8. Inicializando gerenciador de ordens...")
            self.order_manager = OrderManager(
                self.connection,
                self.risk_manager,
                self.config
            )
            
            # Registrar handlers seguros
            self._register_safe_handlers()
            
            # Subscrever ao ticker
            ticker = self.config.get('ticker', 'WDOU25')
            self.logger.info(f"9. Subscrevendo ao ticker {ticker}...")
            
            if not self.connection.subscribe_ticker(ticker):
                self.logger.error(f"Falha ao subscrever {ticker}")
                return False
                
            self.is_initialized = True
            self.logger.info("="*60)
            self.logger.info("Sistema inicializado com sucesso!")
            self.logger.info("="*60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {e}", exc_info=True)
            return False
            
    def _register_safe_handlers(self):
        """Registra handlers seguros para callbacks"""
        try:
            # Handler para candles
            def on_candle(candle_data):
                self.stats['candles_received'] += 1
                # Processar candle de forma segura
                
            # Handler para trades
            def on_trade(trade_data):
                # Processar trade de forma segura
                pass
                
            # Registrar apenas handlers seguros
            if hasattr(self.connection, 'register_candle_handler'):
                self.connection.register_candle_handler(on_candle)
                
            if hasattr(self.connection, 'register_trade_handler'):
                self.connection.register_trade_handler(on_trade)
                
        except Exception as e:
            self.logger.warning(f"Erro ao registrar handlers: {e}")
            
    def start(self) -> bool:
        """Inicia o sistema de trading"""
        if not self.is_initialized:
            self.logger.error("Sistema não inicializado")
            return False
            
        try:
            self.logger.info("Iniciando componentes...")
            
            # Iniciar pipeline de dados
            if not self.data_pipeline.start():
                self.logger.error("Falha ao iniciar pipeline")
                return False
                
            # Iniciar ML coordinator
            if hasattr(self.ml_coordinator, 'start'):
                self.ml_coordinator.start()
                
            # Iniciar signal generator
            if hasattr(self.signal_generator, 'start'):
                self.signal_generator.start()
                
            self.is_running = True
            self.start_time = time.time()
            
            self.logger.info("Sistema iniciado com sucesso!")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar: {e}", exc_info=True)
            return False
            
    def stop(self):
        """Para o sistema de trading"""
        self.logger.info("Parando sistema...")
        self.is_running = False
        
        # Parar componentes na ordem reversa
        components = [
            ('Signal Generator', self.signal_generator),
            ('ML Coordinator', self.ml_coordinator),
            ('Data Pipeline', self.data_pipeline),
            ('Connection', self.connection)
        ]
        
        for name, component in components:
            if component and hasattr(component, 'stop'):
                try:
                    self.logger.info(f"Parando {name}...")
                    component.stop()
                except Exception as e:
                    self.logger.error(f"Erro ao parar {name}: {e}")
                    
        # Desconectar
        if self.connection and hasattr(self.connection, 'disconnect'):
            try:
                self.connection.disconnect()
            except:
                pass
                
        self.logger.info("Sistema parado")
        
    def is_running(self) -> bool:
        """Verifica se o sistema está rodando"""
        return self.is_running
        
    def get_status(self) -> Dict[str, Any]:
        """Retorna status atual do sistema"""
        status = {
            'running': self.is_running,
            'connected': False,
            'candles_count': 0,
            'last_prediction_time': None,
            'active_positions': 0
        }
        
        try:
            # Status da conexão
            if self.connection:
                status['connected'] = self.connection.is_connected()
                
            # Dados
            if self.data_structure:
                candles_df = self.data_structure.get_candles()
                if candles_df is not None:
                    status['candles_count'] = len(candles_df)
                    
            # ML
            if self.ml_coordinator and hasattr(self.ml_coordinator, 'last_prediction_time'):
                status['last_prediction_time'] = self.ml_coordinator.last_prediction_time
                
            # Posições
            if self.order_manager and hasattr(self.order_manager, 'get_positions'):
                positions = self.order_manager.get_positions()
                status['active_positions'] = len(positions) if positions else 0
                
        except Exception as e:
            self.logger.debug(f"Erro ao obter status: {e}")
            
        return status
        
    def get_final_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas finais"""
        if not self.start_time:
            return {}
            
        runtime = (time.time() - self.start_time) / 60
        
        stats = {
            'runtime_minutes': runtime,
            'total_trades': self.stats['trades'],
            'ml_predictions': self.stats['ml_predictions'],
            'candles_received': self.stats['candles_received'],
            'errors': self.stats['errors']
        }
        
        # Adicionar métricas de trading se disponíveis
        if self.risk_manager:
            risk_metrics = self.risk_manager.get_metrics()
            stats.update({
                'total_pnl': risk_metrics.get('total_pnl', 0),
                'win_rate': risk_metrics.get('win_rate', 0),
                'max_drawdown': risk_metrics.get('max_drawdown', 0)
            })
            
        return stats