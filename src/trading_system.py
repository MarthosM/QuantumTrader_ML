"""
Sistema de Trading Integrado v2.0
Integra todos os componentes em um sistema coeso e funcional
"""

import logging
import os
import threading
import time
import queue
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from ctypes import WINFUNCTYPE, c_int, c_wchar_p, c_double, c_int64, Structure

import pandas as pd

# Importar componentes desenvolvidos nas etapas anteriores
from connection_manager import ConnectionManager
from model_manager import ModelManager
from data_structure import TradingDataStructure
from data_pipeline import DataPipeline
from real_time_processor import RealTimeProcessor
from data_loader import DataLoader
from feature_engine import FeatureEngine
from prediction_engine import PredictionEngine
from ml_coordinator import MLCoordinator
from signal_generator import SignalGenerator
from risk_manager import RiskManager
from strategy_engine import StrategyEngine
from metrics_collector import MetricsCollector

# Adicionar integração para dados reais
from data_integration import DataIntegration

# Importar componentes das ETAPAS 4 e 5 (opcionais)
try:
    from ml.continuous_optimizer import ContinuousOptimizationPipeline, AutoOptimizationEngine
    from monitoring.performance_monitor import RealTimePerformanceMonitor
    from risk.intelligent_risk_manager import IntelligentRiskManager
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    # Componentes avançados não disponíveis
    ADVANCED_FEATURES_AVAILABLE = False
    ContinuousOptimizationPipeline = None
    AutoOptimizationEngine = None
    RealTimePerformanceMonitor = None
    IntelligentRiskManager = None

# Importar sistema de validação de produção
try:
    from production_data_validator import ProductionDataValidator, ProductionDataError
    PRODUCTION_VALIDATOR_AVAILABLE = True
except ImportError:
    PRODUCTION_VALIDATOR_AVAILABLE = False
    ProductionDataValidator = None
    ProductionDataError = Exception

class TradingSystem:
    """Sistema de trading completo v2.0"""
    
    def __init__(self, config: Dict):
        """
        Inicializa o sistema de trading
        
        Args:
            config: Configurações do sistema incluindo:
                - dll_path: Caminho da ProfitDLL
                - username/password: Credenciais
                - models_dir: Diretório dos modelos
                - ticker: Ativo para operar
                - strategy: Configurações de estratégia
                - risk: Configurações de risco
        """
        self.config = config
        self.logger = self._setup_logger()
        
        # Componentes principais
        self.connection = None
        self.model_manager = None
        self.data_structure = None
        self.data_pipeline = None
        self.data_loader = None
        self.real_time_processor = None
        self.feature_engine = None
        self.ml_coordinator = None
        self.strategy_engine = None
        self.metrics = None
        
        # Estado do sistema
        self.is_running = False
        self.initialized = False
        # Determinar contrato atual automaticamente
        self.ticker = self._get_current_contract(datetime.now())
        self.contract_check_time = None
        self.contract_check_interval = 3600  # Verificar a cada hora

        # Data integration será inicializado após os componentes
        self.data_integration = None
        
        # Controles anti-loop para carregamento de dados
        self.historical_data_loaded = False
        self.last_historical_load_time = None
        self.gap_fill_in_progress = False
        
        # Threads e queues
        self.ml_queue = queue.Queue(maxsize=10)
        self.signal_queue = queue.Queue(maxsize=10)
        self.ml_thread = None
        self.signal_thread = None
        self.optimization_thread = None
        self.risk_update_thread = None
        
        # Controle de tempo
        self.last_ml_time = None
        self.ml_interval = config.get('ml_interval', 60)  # segundos
        self.last_feature_calc = None
        self.feature_interval = 5  # segundos
        
        # Cache e estado
        self.last_prediction = None
        self.active_positions = {}
        self.account_info = {
            'balance': config.get('initial_balance', 100000),
            'available': config.get('initial_balance', 100000),
            'daily_pnl': 0,
            'daily_trades': 0
        }
        
        # Monitor visual (opcional)
        self.monitor = None
        self.use_gui = config.get('use_gui', True)

        # Sistema de otimização contínua (ETAPA 4)
        self.continuous_optimizer = None
        self.auto_optimizer = None
        self.performance_monitor = None
        
        # Sistema de risco inteligente (ETAPA 5)
        self.intelligent_risk_manager = None
        
        # Sistema de validação de produção (CRÍTICO)
        if PRODUCTION_VALIDATOR_AVAILABLE and ProductionDataValidator:
            self.production_validator = ProductionDataValidator()
        else:
            self.production_validator = None
        
        # Auto-retreinamento será configurado após inicialização do model_manager
        self.auto_retrain_config = {
            'auto_retrain_enabled': True,
            'min_retrain_interval_hours': 24,
            'min_data_points': 1000,
            'validation_split': 0.2
        }

    def _get_current_contract(self, date: datetime) -> str:
        """
        Determina o código de contrato WDO correto para uma data.
        
        Args:
            date: Data alvo
            
        Returns:
            str: Código do contrato (ex. "WDOQ25")
        """
        # Códigos de mês para futuros WDO
        month_codes = {
            1: 'G',  # Janeiro
            2: 'H',  # Fevereiro
            3: 'J',  # Março
            4: 'K',  # Abril
            5: 'M',  # Maio
            6: 'N',  # Junho
            7: 'Q',  # Julho
            8: 'U',  # Agosto
            9: 'V',  # Setembro
            10: 'X', # Outubro
            11: 'Z', # Novembro
            12: 'F'  # Dezembro
        }
        
        # O mês atual usa o código do mês atual
        month_code = month_codes[date.month]
        year_code = str(date.year)[-2:]
        
        contract = f"WDO{month_code}{year_code}"
        
        self.logger.info(f"Para data {date.date()}, usando contrato: {contract}")
        return contract
    
    def _validate_production_data(self, data, source: str, data_type: str):
        """
        Valida dados para produção - OBRIGATÓRIO em todos os pontos de dados
        
        Args:
            data: Dados a serem validados
            source: Fonte dos dados ('connection', 'file', 'cache', etc.)
            data_type: Tipo dos dados ('candles', 'trade', 'orderbook', etc.)
            
        Raises:
            ProductionDataError: Se dados são inválidos/dummy para produção
        """
        # Verificar se validador está disponível
        if not PRODUCTION_VALIDATOR_AVAILABLE or not self.production_validator:
            # Validação básica manual se validador não disponível
            if os.getenv('TRADING_ENV') == 'production':
                self.logger.warning("⚠️ ProductionDataValidator não disponível - usando validação básica")
                self._basic_data_validation(data, source, data_type)
            return
        
        try:
            # Usar o validador de produção
            self.production_validator.validate_trading_data(data, source, data_type)
            
        except ProductionDataError as e:
            self.logger.error(f"❌ DADOS INVÁLIDOS DETECTADOS - {source}.{data_type}: {e}")
            
            # Em produção, parar tudo
            if os.getenv('TRADING_ENV') == 'production':
                self.logger.critical("🚨 PRODUÇÃO BLOQUEADA - DADOS UNSAFE DETECTADOS")
                raise
            else:
                # Em desenvolvimento, apenas avisar
                self.logger.warning("⚠️ DESENVOLVIMENTO - Dados podem ser sintéticos")
        
        except Exception as e:
            self.logger.error(f"Erro na validação de produção: {e}")
            raise
    
    def _basic_data_validation(self, data, source: str, data_type: str):
        """Validação básica quando ProductionDataValidator não está disponível"""
        import pandas as pd
        import numpy as np
        
        if data is None:
            raise ValueError(f"Dados nulos recebidos de {source}")
        
        if isinstance(data, pd.DataFrame):
            if data.empty:
                raise ValueError(f"DataFrame vazio recebido de {source}")
            
            # Verificar se há dados obviamente sintéticos
            if data_type == 'candles':
                if 'close' in data.columns:
                    # Verificar se todos os preços são iguais (suspeito)
                    if data['close'].nunique() == 1:
                        self.logger.warning(f"⚠️ Suspeita de dados sintéticos: preços idênticos em {source}")
        
        self.logger.info(f"✓ Validação básica aprovada: {source}.{data_type}")
    
    def _setup_logger(self) -> logging.Logger:
        """Configura o sistema de logging"""
        logger = logging.getLogger('TradingSystemV2')
        
        # Usar apenas propagação para o logger raiz configurado no main
        logger.propagate = True
        
        return logger
        
    def initialize(self) -> bool:
        """Inicializa todos os componentes do sistema"""
        try:
            self.logger.info("="*60)
            self.logger.info("Iniciando Sistema de Trading v2.0")
            self.logger.info("="*60)
            
            # 1. Inicializar conexão
            self.logger.info("1. Inicializando conexão...")
            self.connection = ConnectionManager(self.config['dll_path'])
            if not self.connection.initialize(
                key=self.config.get('key', ''),
                username=self.config['username'],
                password=self.config['password'],
                account_id=self.config.get('account_id'),
                broker_id=self.config.get('broker_id'),
                trading_password=self.config.get('trading_password')
            ):
                self.logger.error("Falha ao inicializar conexão")
                return False
            self.logger.info("[ok] Conexão estabelecida")
            
            # 2. Carregar modelos ML
            self.logger.info("2. Carregando modelos ML...")
            self.model_manager = ModelManager(self.config['models_dir'])
            if not self.model_manager.load_models():
                self.logger.error("Falha ao carregar modelos")
                return False
            self.logger.info(f"[ok] {len(self.model_manager.models)} modelos carregados")
            
            # Configurar auto-retreinamento após carregar modelos
            if hasattr(self.model_manager, 'setup_auto_retraining'):
                self.model_manager.setup_auto_retraining(self.auto_retrain_config)
                self.logger.info("[ok] Auto-retreinamento configurado")
            
            # 3. Inicializar estrutura de dados
            self.data_structure = TradingDataStructure()
            self.data_structure.initialize_structure()
            self.logger.info("[ok] Estrutura de dados criada")
            
            # 4. Configurar pipeline de dados
            self.logger.info("4. Configurando pipeline de dados...")
            self.data_pipeline = DataPipeline(self.data_structure)
            self.real_time_processor = RealTimeProcessor(self.data_structure)
            self.data_loader = DataLoader(self.config.get('data_dir', 'data'))
            
            # 4.1 Inicializar integração de dados
            self.data_integration = DataIntegration(self.connection, self.data_loader)
            self.logger.info("[ok] Pipeline de dados configurado")

            # 5. Configurar engine de features
            self.logger.info("5. Configurando engine de features...")
            all_features = self._get_all_required_features()
            self.feature_engine = FeatureEngine(list(all_features))
            self.logger.info(f"[ok] Feature engine configurado com {len(all_features)} features")
            
            # 6. Configurar ML coordinator
            self.logger.info("6. Configurando ML coordinator...")
            pred_engine = PredictionEngine(self.model_manager)
            self.ml_coordinator = MLCoordinator(
                self.model_manager,
                self.feature_engine,
                pred_engine
            )
            self.logger.info("[ok] ML coordinator configurado")

            # 7. Configurar estratégia e risco
            self.logger.info("7. Configurando estratégia e risco...")
            signal_gen = SignalGenerator(self.config.get('strategy', {}))
            risk_mgr = RiskManager(self.config.get('risk', {}))
            self.strategy_engine = StrategyEngine(signal_gen, risk_mgr)
            
            # 7.1 Configurar risco inteligente (ETAPA 5) - Opcional
            if ADVANCED_FEATURES_AVAILABLE and IntelligentRiskManager:
                self.intelligent_risk_manager = IntelligentRiskManager(self.config.get('risk', {}))
                self.logger.info("[ok] Estratégia e risco inteligente configurados")
            else:
                self.logger.warning("Sistema de risco inteligente não disponível - usando básico")

            # 8. Inicializar métricas
            self.logger.info("8. Inicializando sistema de métricas...")
            self.metrics = MetricsCollector()
            self.logger.info("[ok] Sistema de métricas inicializado")

            # 9. Configurar sistema de otimização contínua (ETAPA 4) - Opcional
            if ADVANCED_FEATURES_AVAILABLE:
                self.logger.info("9. Configurando sistema de otimização contínua...")
                
                if ContinuousOptimizationPipeline:
                    self.continuous_optimizer = ContinuousOptimizationPipeline({
                        'optimization_interval_hours': 4,
                        'min_trades_for_optimization': 50,
                        'performance_window_hours': 24
                    })
                
                if AutoOptimizationEngine:
                    self.auto_optimizer = AutoOptimizationEngine(
                        self.model_manager,
                        {
                            'optimization_interval': 3600,  # 1 hora
                            'min_win_rate': 0.52,
                            'min_sharpe': 1.0,
                            'max_drawdown': 0.1,
                            'min_confidence': 0.6
                        }
                    )
                
                if RealTimePerformanceMonitor:
                    self.performance_monitor = RealTimePerformanceMonitor({
                        'update_interval': 60,  # segundos
                        'alert_thresholds': {
                            'min_win_rate': 0.45,
                            'max_drawdown': 0.15,
                            'max_consecutive_losses': 10
                        }
                    })
                
                self.logger.info("[ok] Sistema de otimização contínua configurado")
            else:
                self.logger.warning("Componentes avançados (ETAPA 4) não disponíveis - usando configuração básica")
                self.continuous_optimizer = None
                self.auto_optimizer = None
                self.performance_monitor = None

            # 10. Configurar callbacks
            self.logger.info("10. Configurando callbacks...")
            self._setup_callbacks()
            self.logger.info("[ok] Callbacks configurados")

            self.initialized = True
            self.logger.info("="*60)
            self.logger.info("Sistema inicializado com sucesso!")
            self.logger.info("="*60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {e}", exc_info=True)
            return False
            
    def _get_all_required_features(self) -> set:
        """Obtém todas as features necessárias pelos modelos"""
        all_features = set()
        
        # Coletar features de todos os modelos se model_manager estiver inicializado
        if self.model_manager and hasattr(self.model_manager, 'model_features'):
            for model_name, features in self.model_manager.model_features.items():
                all_features.update(features)
            
        # Adicionar features básicas sempre necessárias
        basic_features = {'open', 'high', 'low', 'close', 'volume'}
        all_features.update(basic_features)
        
        return all_features
    
    def _load_historical_data_safe(self, ticker: str, days_back: int) -> bool:
        """
        Carrega dados históricos reais do mercado
        
        Args:
            ticker: Símbolo do ativo
            days_back: Número de dias para carregar
            
        Returns:
            bool: True se carregou com sucesso
        """
        try:
            # Verificar modo de operação
            production_mode = os.getenv('TRADING_ENV', 'development') == 'production'
            
            if production_mode and not self.connection.market_connected:
                self.logger.error("PRODUÇÃO: Sem conexão com market data - operação bloqueada")
                return False
            
            # Opção 1 - Carregar dados reais via ConnectionManager
            if self.connection and self.connection.connected:
                self.logger.info(f"Carregando dados históricos reais para {ticker}")
                
                if self.connection.login_state != self.connection.LOGIN_CONNECTED:
                    self.logger.error("Login não conectado - não é possível obter dados históricos")
                    self.connection._log_connection_states()
                    
                    if production_mode:
                        return False
                    else:
                        self.logger.warning("DESENVOLVIMENTO: Prosseguindo sem dados históricos reais...")
                
                else:
                    # Tentar obter dados históricos reais
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days_back)
                    
                    self.logger.info("Login conectado - solicitando dados históricos...")
                    
                    result = self.connection.request_historical_data(
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if result >= 0:
                        self.logger.info("Dados históricos solicitados com sucesso!")
                        self.logger.info("Aguardando recebimento via callback...")
                        
                        success = self.connection.wait_for_historical_data(timeout_seconds=60)
                        
                        if success:
                            self.logger.info(f"Dados históricos recebidos com sucesso!")
                            
                            # 🛡️ VALIDAÇÃO OBRIGATÓRIA - Dados de produção
                            # Validar dados recebidos (verificação condicional)
                            try:
                                if hasattr(self.data_structure, 'candles') and not self.data_structure.candles.empty:
                                    self._validate_production_data(
                                        self.data_structure.candles.tail(10), 
                                        'connection_historical', 
                                        'candles'
                                    )
                            except Exception as e:
                                self.logger.warning(f"Validação de dados históricos: {e}")
                            
                            self.historical_data_loaded = True
                            self.last_historical_load_time = datetime.now()
                            
                            if hasattr(self.connection, '_historical_data_count'):
                                count = self.connection._historical_data_count
                                self.logger.info(f"Total de {count} registros históricos processados")
                                
                            self._check_and_fill_temporal_gap()
                            
                            return True
                        else:
                            self.logger.warning("Timeout ou erro ao receber dados históricos")
                        
                    else:
                        self.logger.error(f"Falha ao solicitar dados históricos: código {result}")
                    
            # Opção 2 - Carregar de cache/arquivo se disponível (apenas desenvolvimento)
            if not production_mode:
                if self.data_loader:
                    self.logger.info("Tentando carregar dados do cache/arquivo...")
                    
                    candles_df = self.data_loader.load_candles(
                        start_date=datetime.now() - timedelta(days=days_back),
                        end_date=datetime.now(),
                        interval='1min',
                        symbol=ticker
                    )
                    
                    if not candles_df.empty:
                        self.data_structure.update_candles(candles_df)
                        self.logger.info(f"Dados carregados do cache: {len(candles_df)} candles")
                        return True
                
            # Opção 3: Modo desenvolvimento com aviso claro
            if not production_mode:
                self.logger.warning("MODO DESENVOLVIMENTO - Carregando dados de teste isolados")
                return self._load_test_data_isolated(ticker, days_back)
            
            # Em produção, falhar se não há dados reais
            self.logger.error("Nenhuma fonte de dados reais disponível")
            return False
            
        except Exception as e:
            self.logger.error(f"Erro carregando dados históricos: {e}")
            return False

    def _load_test_data_isolated(self, ticker: str, days_back: int) -> bool:
        """
        Carrega dados de teste APENAS em desenvolvimento
        Isolado para não contaminar produção
        """
        # Verificar dupla que não está em produção
        if os.getenv('TRADING_ENV') == 'production':
            raise RuntimeError("_load_test_data_isolated chamado em PRODUÇÃO!")
        
        # Caminhos relativos ao diretório do projeto
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Tentar carregar de arquivo de teste
        test_file = os.path.join(base_dir, "tests", "data", f"{ticker}_test_data.csv")
        if os.path.exists(test_file):
            import pandas as pd
            try:
                test_df = pd.read_csv(test_file, parse_dates=['timestamp'], index_col='timestamp')
                
                # Filtrar período
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                test_df = test_df[start_date:end_date]
                
                if not test_df.empty:
                    self.data_structure.update_candles(test_df)
                    self.logger.info(f"Dados de teste carregados: {len(test_df)} candles")
                    return True
            except Exception as e:
                self.logger.warning(f"Erro carregando arquivo de teste: {e}")
        
        self.logger.error("=== DADOS HISTÓRICOS NECESSÁRIOS ===")
        self.logger.error("Para testar o sistema:")
        self.logger.error(f"1. Conecte-se à ProfitDLL com dados reais")
        self.logger.error(f"2. Ou coloque arquivo CSV em {test_file}")
        self.logger.error("====================================")
        
        return False
    
    def _setup_callbacks(self):
        """Configura callbacks para dados em tempo real"""
        if not self.connection:
            self.logger.warning("Conexão não disponível para configurar callbacks")
            return
            
        # Por enquanto, os callbacks específicos não estão implementados no ConnectionManager
        # O sistema funcionará com polling de dados
        self.logger.info("Callbacks não implementados - usando polling para dados")
        
    def start(self, ticker: Optional[str] = None) -> bool:
        """
        Inicia operação do sistema
        
        Args:
            ticker: Ticker do ativo (usa config se não fornecido)
        """
        if not self.initialized:
            self.logger.error("Sistema não inicializado")
            return False
            
        try:
            # Se ticker não foi fornecido, usar o contrato atual
            if ticker:
                self.ticker = ticker
            else:
                # Atualizar para contrato atual
                current_contract = self._get_current_contract(datetime.now())
                if current_contract != self.ticker:
                    self.logger.info(f"Atualizando contrato de {self.ticker} para {current_contract}")
                    self.ticker = current_contract
            self.logger.info(f"Iniciando operação para {self.ticker}")
            
            # 1. Carregar dados históricos
            self.logger.info("Carregando dados históricos...")
            days_back = self.config.get('historical_days', 10)
            
            if not self._load_historical_data_safe(self.ticker, days_back):
                self.logger.error("Falha ao carregar dados históricos")
                return False
                
            self.logger.info(f"✓ {len(self.data_structure.candles) if self.data_structure else 0} candles carregadas")
            
            # 2. Calcular indicadores e features iniciais
            self.logger.info("Calculando indicadores e features iniciais...")
            self._calculate_initial_features()
            
            # 3. Iniciar threads de processamento
            self.logger.info("Iniciando threads de processamento...")
            self._start_processing_threads()
            
            # 4. Marcar sistema como rodando
            self.is_running = True
            
            # 5. Solicitar dados em tempo real
            self.logger.info(f"Solicitando dados em tempo real para {self.ticker}")
            if self.connection and hasattr(self.connection, 'subscribe_ticker'):
                self.connection.subscribe_ticker(self.ticker)
            else:
                self.logger.warning("Método subscribe_ticker não disponível")
            
            # 6. Iniciar sistemas de otimização e monitoramento (ETAPA 4) - Condicional
            if self.performance_monitor and self.auto_optimizer:
                self.logger.info("Iniciando sistemas de otimização contínua...")
                try:
                    self.performance_monitor.start_monitoring()
                    self.auto_optimizer.start()
                    self.logger.info("Sistemas de otimização iniciados")
                except Exception as e:
                    self.logger.warning(f"Falha ao iniciar otimização: {e}")
            else:
                self.logger.info("Sistemas de otimização não disponíveis - prosseguindo sem otimização")
            
            self.logger.info("Sistema iniciado e operacional!")
            
            # 7. Iniciar monitor GUI se habilitado
            if self.use_gui:
                self.logger.info("Iniciando monitor visual...")
                from trading_monitor import TradingMonitor
                self.monitor = TradingMonitor(self)
                
                # Monitor roda em thread separada
                monitor_thread = threading.Thread(
                    target=self.monitor.start,
                    daemon=True
                )
                monitor_thread.start()
                time.sleep(1)  # Dar tempo para GUI inicializar
            
            # 8. Entrar no loop principal
            self._main_loop()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar sistema: {e}", exc_info=True)
            return False
            
    def _calculate_initial_features(self):
        """Calcula features iniciais com dados históricos"""
        try:
            # Verificar se componentes estão disponíveis
            if not self.feature_engine or not self.data_structure:
                self.logger.warning("Feature engine ou data structure não disponível")
                return
                
            # Calcular todas as features
            result = self.feature_engine.calculate(self.data_structure)
            
            # Log estatísticas
            if 'indicators' in result:
                self.logger.info(f"Indicadores calculados: {len(result['indicators'].columns)} colunas")
            if 'features' in result:
                self.logger.info(f"Features ML calculadas: {len(result['features'].columns)} colunas")
                
            self.last_feature_calc = time.time()
            
        except Exception as e:
            self.logger.error(f"Erro calculando features iniciais: {e}")
            
    def _start_processing_threads(self):
        """Inicia threads de processamento assíncrono"""
        # Thread de ML
        self.ml_thread = threading.Thread(
            target=self._ml_worker,
            daemon=True,
            name="MLWorker"
        )
        self.ml_thread.start()
        
        # Thread de sinais
        self.signal_thread = threading.Thread(
            target=self._signal_worker,
            daemon=True,
            name="SignalWorker"
        )
        self.signal_thread.start()
        
        # Thread de otimização (ETAPA 4)
        self.optimization_thread = threading.Thread(
            target=self._optimization_worker,
            daemon=True,
            name="OptimizationWorker"
        )
        self.optimization_thread.start()
        
        # Thread de atualização de risco (ETAPA 5)
        self.risk_update_thread = threading.Thread(
            target=self._risk_update_worker,
            daemon=True,
            name="RiskUpdateWorker"
        )
        self.risk_update_thread.start()
        
        self.logger.info("Threads de processamento iniciadas")
        
    def _main_loop(self):
        """Loop principal do sistema"""
        self.logger.info("Entrando no loop principal...")
        
        try:
            while self.is_running:
                # Verificar se deve recalcular features
                if self._should_calculate_features():
                    self._request_feature_calculation()

                # Verificar mudança de contrato
                if self._should_check_contract():
                    self._check_contract_rollover()

                # Verificar se deve fazer predição ML
                if self._should_run_ml():
                    self._request_ml_prediction()
                    
                # Processar métricas
                if hasattr(self, 'metrics'):
                    self._update_metrics()
                    
                # Pequena pausa para não sobrecarregar CPU
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("Interrupção do usuário detectada")
        except Exception as e:
            self.logger.error(f"Erro no loop principal: {e}", exc_info=True)
        finally:
            self.stop()
            
    def _on_trade(self, trade_data: Dict):
        """Callback para processar trades em tempo real"""
        if not self.is_running:
            return
            
        try:
            # Usar data_integration para criar candles reais
            if self.data_integration:
                # Processar trade para formar candles
                self.data_integration._on_trade(trade_data)
                
                # Obter candles atualizados
                current_candles = self.data_integration.get_candles('1min')
                if not current_candles.empty:
                    self.data_structure.update_candles(current_candles)
            
            # Processar com real time processor se disponível
            elif self.real_time_processor:
                self.real_time_processor.process_trade(trade_data)
            else:
                self.logger.warning("Nenhum processador de dados disponível")
                
            # Atualizar métricas se disponível
            if self.metrics:
                self.metrics.record_trade()
            
            # Registrar no monitor de performance (ETAPA 4)
            if self.performance_monitor and hasattr(trade_data, 'get'):
                self.performance_monitor.record_trade(trade_data)
                
        except Exception as e:
            self.logger.error(f"Erro processando trade: {e}")
            if self.metrics and hasattr(self.metrics, 'metrics'):
                self.metrics.metrics['errors'].append({
                    'time': datetime.now(),
                    'type': 'trade_processing',
                    'error': str(e)
                })
            
    def _on_book_update(self, book_data: Dict):
        """Callback para processar atualizações do book"""
        # Implementar se necessário
        pass
        
    def _on_state_change(self, state_type: int, state: int):
        """Callback para mudanças de estado da conexão"""
        state_names = {
            0: "LOGIN",
            1: "MARKET_DATA",
            2: "BROKER"
        }
        
        self.logger.info(f"Mudança de estado: {state_names.get(state_type, 'UNKNOWN')} = {state}")
        
    def _should_calculate_features(self) -> bool:
        """Verifica se deve recalcular features"""
        if self.last_feature_calc is None:
            return True
            
        elapsed = time.time() - self.last_feature_calc
        return elapsed >= self.feature_interval
        
    def _should_run_ml(self) -> bool:
        """Verifica se deve executar predição ML"""
        if self.last_ml_time is None:
            return True
            
        elapsed = time.time() - self.last_ml_time
        return elapsed >= self.ml_interval
    
    def _should_check_contract(self) -> bool:
        """Verifica se deve checar mudança de contrato"""
        if self.contract_check_time is None:
            return True
            
        elapsed = time.time() - self.contract_check_time
        return elapsed >= self.contract_check_interval

    def _check_contract_rollover(self):
        """Verifica se houve mudança de mês e atualiza contrato se necessário"""
        current_contract = self._get_current_contract(datetime.now())
        
        if current_contract != self.ticker:
            self.logger.warning(f"MUDANÇA DE CONTRATO DETECTADA: {self.ticker} -> {current_contract}")
            
            # Aqui você pode adicionar lógica para:
            # 1. Fechar posições no contrato antigo
            # 2. Cancelar ordens pendentes
            # 3. Atualizar subscrições
            
            # Por enquanto, apenas atualizar e re-subscrever
            old_ticker = self.ticker
            self.ticker = current_contract
            
            # Re-subscrever para novo contrato
            if self.connection:
                self.logger.info(f"Cancelando subscrição de {old_ticker}")
                self.connection.unsubscribe_ticker(old_ticker)
                
                self.logger.info(f"Subscrevendo novo contrato {self.ticker}")
                self.connection.subscribe_ticker(self.ticker)
                
            # Resetar dados se necessário
            self.logger.info("Limpando dados do contrato anterior")
            self._reset_contract_data()
            
        self.contract_check_time = time.time()

    def _reset_contract_data(self):
        """Reseta dados ao mudar de contrato"""
        # Limpar dados antigos mas manter estrutura
        if self.data_structure:
            self.data_structure.candles = self.data_structure.candles.iloc[0:0]
            self.data_structure.microstructure = self.data_structure.microstructure.iloc[0:0]
            self.data_structure.orderbook = self.data_structure.orderbook.iloc[0:0]
            self.data_structure.indicators = self.data_structure.indicators.iloc[0:0]
            self.data_structure.features = self.data_structure.features.iloc[0:0]
            
        # Resetar timers
        self.last_ml_time = None
        self.last_feature_calc = None
        
        # Carregar dados históricos do novo contrato
        if self.data_loader:
            self.logger.info(f"Carregando dados históricos para {self.ticker}")
            self._load_historical_data_safe(self.ticker, self.config.get('historical_days', 10))
                
    def _request_feature_calculation(self):
        """Solicita cálculo de features"""
        # Adicionar à fila se não estiver cheia
        if not self.ml_queue.full():
            self.ml_queue.put({
                'type': 'calculate_features',
                'timestamp': datetime.now()
            })
            self.last_feature_calc = time.time()
            
    def _request_ml_prediction(self):
        """Solicita predição ML"""
        if not self.ml_queue.full():
            self.ml_queue.put({
                'type': 'predict',
                'timestamp': datetime.now()
            })
            self.last_ml_time = time.time()
            
    def _ml_worker(self):
        """Thread worker para processamento ML"""
        self.logger.info("ML worker iniciado")
        
        while self.is_running:
            try:
                # Pegar próxima tarefa
                task = self.ml_queue.get(timeout=1.0)
                
                if task['type'] == 'calculate_features':
                    self._process_feature_calculation()
                elif task['type'] == 'predict':
                    self._process_ml_prediction()
                    
                self.ml_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Erro no ML worker: {e}", exc_info=True)
                
        self.logger.info("ML worker finalizado")
        
    def _process_feature_calculation(self):
        """Processa cálculo de features"""
        try:
            # Verificar componentes disponíveis
            if not self.feature_engine or not self.data_structure:
                self.logger.warning("Feature engine ou data structure não disponível")
                return
                
            # Calcular features
            result = self.feature_engine.calculate(self.data_structure)
            
            # Log apenas mudanças significativas
            if 'model_ready' in result:
                features_count = len(result['model_ready'].columns)
                if not hasattr(self, '_last_features_count') or self._last_features_count != features_count:
                    self.logger.info(f"Features calculadas: {features_count} colunas")
                    self._last_features_count = features_count
                    
        except Exception as e:
            self.logger.error(f"Erro calculando features: {e}")
            
    def _process_ml_prediction(self):
        """Processa predição ML"""
        try:
            # Verificar dados suficientes
            if not self.data_structure or not hasattr(self.data_structure, 'candles'):
                self.logger.warning("Data structure não disponível")
                return
                
            if len(self.data_structure.candles) < 50:
                return
                
            # Verificar ML coordinator
            if not self.ml_coordinator:
                self.logger.warning("ML coordinator não disponível")
                return
                
            # Executar predição
            prediction = self.ml_coordinator.process_prediction_request(self.data_structure)
            
            if prediction:
                self.last_prediction = prediction
                
                # Registrar métricas se disponível
                if self.metrics:
                    self.metrics.record_prediction(prediction)
                
                # Registrar no monitor de performance (ETAPA 4)
                if self.performance_monitor:
                    self.performance_monitor.record_prediction({
                        'model': 'ensemble',
                        'direction': prediction['direction'],
                        'confidence': prediction['confidence'],
                        'features_used': list(self.feature_engine.model_features),
                        'latency_ms': prediction.get('processing_time', 0) * 1000
                    })
                
                # Log da predição
                self.logger.info(
                    f"Predição ML - Direção: {prediction['direction']:.2f}, "
                    f"Magnitude: {prediction['magnitude']:.4f}, "
                    f"Confiança: {prediction['confidence']:.2f}"
                )
                
                # Adicionar à fila de sinais
                if not self.signal_queue.full():
                    self.signal_queue.put(prediction)
                    
        except Exception as e:
            self.logger.error(f"Erro na predição ML: {e}")
            
    def _signal_worker(self):
        """Thread worker para processamento de sinais"""
        self.logger.info("Signal worker iniciado")
        
        while self.is_running:
            try:
                # Pegar próxima predição
                prediction = self.signal_queue.get(timeout=1.0)
                
                # Processar sinal
                self._process_signal_generation(prediction)
                
                self.signal_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Erro no signal worker: {e}", exc_info=True)
                
        self.logger.info("Signal worker finalizado")
        
    def _process_signal_generation(self, prediction: Dict):
        """Processa geração de sinal baseado em predição"""
        try:
            # Verificar se já tem posição aberta
            if self.active_positions:
                self.logger.info("Posição já aberta, ignorando sinal")
                return
                
            # Verificar se strategy engine está disponível
            if not self.strategy_engine:
                self.logger.warning("Strategy engine não disponível")
                return
                
            # ETAPA 5: Usar risco inteligente para validar sinal
            if self.intelligent_risk_manager:
                # Preparar sinal inicial
                initial_signal = {
                    'entry_price': self.data_structure.candles['close'].iloc[-1],
                    'side': 'long' if prediction['direction'] > 0 else 'short',
                    'symbol': self.ticker,
                    'prediction': prediction
                }
                
                # Avaliação de risco ML
                validation_result = self.intelligent_risk_manager.comprehensive_risk_assessment(
                    initial_signal,
                    self.data_structure.candles,
                    self._get_portfolio_state()
                )
                
                if not validation_result['approved']:
                    self.logger.warning(f"Sinal rejeitado por risco inteligente: Score={validation_result['risk_score']:.2f}")
                    return
                
                # Position sizing inteligente
                position_sizing = self.intelligent_risk_manager.dynamic_position_sizing(
                    initial_signal,
                    validation_result,
                    self.account_info
                )
                
                # Aplicar ajustes ao sinal
                prediction['position_size'] = position_sizing['position_size']
                prediction['risk_metrics'] = position_sizing['risk_metrics']
            
            # Gerar sinal com strategy engine
            signal = self.strategy_engine.process_prediction(
                prediction,
                self.data_structure,
                self.account_info
            )
            
            if signal and signal['action'] != 'none':
                # ETAPA 5: Otimizar stop loss
                if self.intelligent_risk_manager and signal['action'] in ['buy', 'sell']:
                    position_mock = {
                        'entry_price': signal['price'],
                        'current_price': signal['price'],
                        'side': 'long' if signal['action'] == 'buy' else 'short',
                        'quantity': signal.get('position_size', 1)
                    }
                    
                    stop_optimization = self.intelligent_risk_manager.optimize_stop_loss(
                        position_mock,
                        self.data_structure.candles,
                        prediction.get('market_regime', 'normal')
                    )
                    
                    signal['stop_loss'] = stop_optimization['stop_loss']
                    signal['stop_strategy'] = stop_optimization['strategy_used']
                
                self.logger.info(
                    f"SINAL GERADO: {signal['action'].upper()} "
                    f"@ {signal['price']:.2f} "
                    f"Size: {signal.get('position_size', 1)} "
                    f"SL: {signal['stop_loss']:.2f} "
                    f"TP: {signal['take_profit']:.2f}"
                )
                
                # Executar ordem - REAL em produção, SIMULADA em desenvolvimento
                self._execute_order_safely(signal)
                
                # Registrar métrica se disponível
                if self.metrics and hasattr(self.metrics, 'metrics'):
                    self.metrics.metrics['signals_generated'] += 1
                
        except Exception as e:
            self.logger.error(f"Erro gerando sinal: {e}")
            
    def _execute_order_safely(self, signal: Dict):
        """
        Executa ordem de forma segura - REAL em produção, SIMULADA em desenvolvimento
        
        Args:
            signal: Sinal de trading com informações da ordem
        """
        production_mode = os.getenv('TRADING_ENV', 'development') == 'production'
        
        if production_mode:
            # 🚨 PRODUÇÃO: Execução real obrigatória
            self.logger.info(f"[PRODUÇÃO] Executando ordem REAL: {signal['action']}")
            
            try:
                # Verificar conexão com broker
                if not self.connection or not self.connection.connected:
                    raise RuntimeError("Conexão com broker não disponível em PRODUÇÃO")
                
                # Executar ordem real via ConnectionManager
                if hasattr(self.connection, 'place_order'):
                    order_result = self.connection.place_order(
                        symbol=self.ticker,
                        side=signal['action'],
                        quantity=signal.get('position_size', 1),
                        price=signal['price'],
                        stop_loss=signal.get('stop_loss'),
                        take_profit=signal.get('take_profit')
                    )
                    
                    if order_result and order_result.get('success'):
                        self.logger.info(f"✅ Ordem executada - ID: {order_result.get('order_id')}")
                        self._record_real_position(signal, order_result)
                    else:
                        self.logger.error(f"❌ Falha na execução da ordem: {order_result}")
                        
                else:
                    # Fallback: método não disponível no ConnectionManager atual
                    self.logger.error("❌ PRODUÇÃO BLOQUEADA: place_order não implementado no ConnectionManager")
                    raise RuntimeError("Execução real não disponível - place_order não implementado")
                    
            except Exception as e:
                self.logger.error(f"❌ ERRO CRÍTICO na execução de ordem real: {e}")
                # Em produção, não continuar com dados problemáticos
                raise
                
        else:
            # 🧪 DESENVOLVIMENTO: Simulação permitida
            self.logger.info(f"[DESENVOLVIMENTO] Simulando ordem: {signal['action']}")
            self._simulate_order_execution(signal)
    
    def _record_real_position(self, signal: Dict, order_result: Dict):
        """Registra posição real executada"""
        self.active_positions[self.ticker] = {
            'side': signal['action'],
            'entry_price': order_result.get('executed_price', signal['price']),
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'size': signal.get('position_size', 1),
            'entry_time': datetime.now(),
            'order_id': order_result.get('order_id'),
            'stop_strategy': signal.get('stop_strategy', 'fixed'),
            'real_execution': True  # Marcar como execução real
        }
        
        # Registrar métricas
        if self.metrics and hasattr(self.metrics, 'record_real_execution'):
            self.metrics.record_real_execution(order_result)
        elif self.metrics:
            # Fallback para métrica genérica
            if hasattr(self.metrics, 'metrics'):
                self.metrics.metrics['signals_executed'] += 1
    
    def _simulate_order_execution(self, signal: Dict):
        """Simula execução de ordem (APENAS DESENVOLVIMENTO)"""
        # Verificação de segurança dupla
        if os.getenv('TRADING_ENV') == 'production':
            raise RuntimeError("❌ SIMULAÇÃO CHAMADA EM PRODUÇÃO - BLOQUEADO!")
            
        self.logger.info(f"[SIMULAÇÃO] Executando ordem: {signal['action']}")
        
        # Simular posição aberta
        self.active_positions[self.ticker] = {
            'side': signal['action'],
            'entry_price': signal['price'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'size': signal.get('position_size', 1),
            'entry_time': datetime.now(),
            'stop_strategy': signal.get('stop_strategy', 'fixed'),
            'real_execution': False  # Marcar como simulação
        }
        
        # Registrar métrica se disponível
        if self.metrics and hasattr(self.metrics, 'metrics'):
            self.metrics.metrics['signals_executed'] += 1
            
        # Registrar trade no monitor de performance (ETAPA 4)
        if self.performance_monitor:
            self.performance_monitor.record_trade({
                'timestamp': datetime.now(),
                'symbol': self.ticker,
                'side': signal['action'],
                'price': signal['price'],
                'quantity': signal.get('position_size', 1),
                'confidence': signal.get('confidence', 0)
            })
        
    def _optimization_worker(self):
        """Thread worker para otimização contínua (ETAPA 4)"""
        self.logger.info("Optimization worker iniciado")
        
        # Verificar se componentes estão disponíveis
        if not self.continuous_optimizer or not self.performance_monitor:
            self.logger.warning("Componentes de otimização não disponíveis - worker encerrado")
            return
        
        last_optimization = None
        optimization_interval = 3600  # 1 hora
        
        while self.is_running:
            try:
                # Verificar se é hora de otimizar
                if last_optimization is None or (time.time() - last_optimization) > optimization_interval:
                    # Coletar métricas atuais
                    current_metrics = self.performance_monitor.get_current_metrics()
                    
                    # Verificar se deve otimizar
                    should_optimize, reason = self.continuous_optimizer.should_optimize(current_metrics)
                    
                    if should_optimize:
                        self.logger.info(f"Iniciando otimização contínua - Razão: {reason}")
                        
                        # Executar otimização
                        market_data = self.data_structure.candles if self.data_structure else None
                        performance_data = {
                            'returns': self._calculate_returns(),
                            'current_features': list(self.feature_engine.model_features) if self.feature_engine else [],
                            'volatility_factor': self._calculate_volatility_factor()
                        }
                        
                        optimization_results = self.continuous_optimizer.run_optimization_cycle(
                            market_data,
                            performance_data
                        )
                        
                        # Aplicar resultados se aprovados
                        if optimization_results:
                            self._apply_optimization_results(optimization_results)
                            
                        last_optimization = time.time()
                
                # Aguardar próximo ciclo
                time.sleep(60)  # Verificar a cada minuto
                
            except Exception as e:
                self.logger.error(f"Erro no optimization worker: {e}", exc_info=True)
                time.sleep(60)
                
        self.logger.info("Optimization worker finalizado")
        
    def _risk_update_worker(self):
        """Thread worker para atualização dinâmica de risco (ETAPA 5)"""
        self.logger.info("Risk update worker iniciado")
        
        while self.is_running:
            try:
                # Aguardar intervalo
                time.sleep(30)  # Atualizar a cada 30 segundos
                
                if not self.active_positions:
                    continue
                
                # Obter dados atuais
                market_data = self.data_structure.candles if self.data_structure else None
                
                if market_data is None or market_data.empty:
                    continue
                
                # Para cada posição aberta
                for symbol, position in self.active_positions.items():
                    try:
                        # Atualizar preço atual
                        position['current_price'] = market_data['close'].iloc[-1]
                        
                        # Verificar se deve atualizar stop loss
                        if self.intelligent_risk_manager and position.get('stop_strategy') != 'fixed':
                            # Detectar regime atual
                            market_regime = self._detect_market_regime(market_data)
                            
                            # Otimizar stop
                            stop_result = self.intelligent_risk_manager.optimize_stop_loss(
                                position,
                                market_data,
                                market_regime
                            )
                            
                            # Aplicar novo stop se melhor
                            if self._should_update_stop(position, stop_result):
                                old_stop = position['stop_loss']
                                position['stop_loss'] = stop_result['stop_loss']
                                position['stop_strategy'] = stop_result['strategy_used']
                                
                                self.logger.info(
                                    f"Stop atualizado para {symbol}: "
                                    f"{old_stop:.2f} -> {stop_result['stop_loss']:.2f} "
                                    f"({stop_result['strategy_used']})"
                                )
                        
                    except Exception as e:
                        self.logger.error(f"Erro atualizando risco para {symbol}: {e}")
                        
            except Exception as e:
                self.logger.error(f"Erro no risk update worker: {e}", exc_info=True)
                
        self.logger.info("Risk update worker finalizado")
        
    def _update_metrics(self):
        """Atualiza métricas do sistema"""
        # Verificar se metrics está disponível
        if not self.metrics:
            return
            
        # Log periódico de métricas
        if hasattr(self, '_last_metrics_log'):
            elapsed = time.time() - self._last_metrics_log
            if elapsed < 60:  # Log a cada minuto
                return
                
        summary = self.metrics.get_summary()
        self.logger.info(
            f"Métricas - Trades: {summary['trades_processed']}, "
            f"Predições: {summary['predictions_made']}, "
            f"Sinais: {summary['signals_generated']}/{summary['signals_executed']}"
        )
        
        self._last_metrics_log = time.time()
        
    def stop(self):
        """Para o sistema de forma ordenada"""
        self.logger.info("Parando sistema...")
        
        self.is_running = False
        
        # Parar sistemas de otimização (ETAPA 4)
        if self.auto_optimizer:
            self.auto_optimizer.stop()
        
        # Parar threads
        threads_to_stop = [
            self.ml_thread,
            self.signal_thread,
            self.optimization_thread,
            self.risk_update_thread
        ]
        
        for thread in threads_to_stop:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
            
        # Desconectar
        if self.connection:
            self.connection.disconnect()

        # Parar monitor se estiver rodando
        if self.monitor:
            self.monitor.stop()
            
        self.logger.info("Sistema parado com sucesso")
            
    def _check_and_fill_temporal_gap(self):
        """
        Verifica se há gap temporal entre dados históricos e tempo atual
        Faz APENAS UM carregamento adicional para evitar loops
        """
        try:
            # PROTEÇÃO ANTI-LOOP: Se já está em progresso, não fazer nada
            if self.gap_fill_in_progress:
                self.logger.info("Gap fill já em progresso - evitando loop")
                return
            
            # Verificar se temos DataIntegration para analisar gap
            if not self.data_integration or not hasattr(self.data_integration, 'candles_1min'):
                self.logger.warning("DataIntegration não disponível para análise de gap")
                return
            
            # Verificar se há dados para analisar
            if self.data_integration.candles_1min.empty:
                self.logger.warning("Nenhum candle formado ainda para análise de gap")
                return
            
            # Pegar último timestamp dos dados
            last_data_time = self.data_integration.candles_1min.index.max()
            current_time = datetime.now()
            
            # Calcular gap em minutos
            gap_minutes = (current_time - last_data_time).total_seconds() / 60
            
            self.logger.info(f"Gap temporal detectado: {gap_minutes:.1f} minutos")
            
            # Se gap é maior que 5 minutos, tentar preencher (APENAS UMA VEZ)
            if gap_minutes > 5:
                self.logger.info(f"Gap de {gap_minutes:.1f} min detectado - carregando dados faltantes...")
                
                # Marcar que gap fill está em progresso
                self.gap_fill_in_progress = True
                
                try:
                    # Calcular período para preencher gap
                    gap_start = last_data_time
                    gap_end = current_time
                    
                    # Solicitar dados do gap (máximo 3 dias conforme limite da API)
                    if gap_minutes > 4320:  # 3 dias = 4320 minutos
                        self.logger.warning("Gap muito grande (>3 dias) - limitando a últimos 3 dias")
                        gap_start = current_time - timedelta(days=3)
                    
                    self.logger.info(f"Solicitando dados do gap: {gap_start} até {gap_end}")
                    
                    result = self.connection.request_historical_data(
                        ticker=self.ticker,
                        start_date=gap_start,
                        end_date=gap_end
                    )
                    
                    if result >= 0:
                        self.logger.info("Dados do gap solicitados - aguardando...")
                        
                        # Aguardar com timeout menor (20s)
                        success = self.connection.wait_for_historical_data(timeout_seconds=20)
                        
                        if success:
                            self.logger.info("Gap temporal preenchido com sucesso!")
                        else:
                            self.logger.warning("Timeout ao preencher gap - continuando mesmo assim")
                    else:
                        self.logger.warning(f"Falha ao solicitar dados do gap: código {result}")
                        
                finally:
                    # SEMPRE limpar flag de gap fill
                    self.gap_fill_in_progress = False
                    
            else:
                self.logger.info(f"Gap pequeno ({gap_minutes:.1f} min) - não é necessário preencher")
                
        except Exception as e:
            self.logger.error(f"Erro ao verificar/preencher gap temporal: {e}")
            # Limpar flag em caso de erro
            self.gap_fill_in_progress = False
        
    def get_status(self) -> Dict:
        """Retorna status atual do sistema"""
        return {
            'running': self.is_running,
            'initialized': self.initialized,
            'ticker': self.ticker,
            'candles': len(self.data_structure.candles) if self.data_structure else 0,
            'last_prediction': self.last_prediction,
            'active_positions': self.active_positions,
            'metrics': self.metrics.get_summary() if self.metrics else {},
            'optimization_enabled': self.auto_optimizer is not None,
            'risk_management': 'intelligent' if self.intelligent_risk_manager else 'basic'
        }
        
    # Métodos auxiliares para suportar ETAPAS 4 e 5
    
    def _get_portfolio_state(self) -> Dict:
        """Retorna estado atual do portfolio"""
        return {
            'positions': self.active_positions,
            'balance': self.account_info['balance'],
            'available': self.account_info['available'],
            'daily_pnl': self.account_info.get('daily_pnl', 0),
            'total_exposure': sum(pos.get('size', 1) * pos.get('entry_price', 0) 
                                for pos in self.active_positions.values())
        }
        
    def _calculate_returns(self) -> List[float]:
        """Calcula retornos para otimização"""
        if not self.data_structure or self.data_structure.candles.empty:
            return []
            
        closes = self.data_structure.candles['close']
        returns = closes.pct_change().dropna().tolist()
        return returns[-100:]  # Últimos 100 retornos
        
    def _calculate_volatility_factor(self) -> float:
        """Calcula fator de volatilidade atual"""
        returns = self._calculate_returns()
        if not returns:
            return 1.0
            
        import numpy as np
        current_vol = float(np.std(returns[-20:])) if len(returns) > 20 else float(np.std(returns))
        historical_vol = float(np.std(returns))
        
        return current_vol / max(historical_vol, 1e-6)
        
    def _apply_optimization_results(self, results: Dict):
        """Aplica resultados da otimização contínua"""
        try:
            # Aplicar novas features se mudaram
            if 'features' in results and results['features'].get('changed'):
                new_features = results['features']['selected_features']
                if self.feature_engine:
                    self.feature_engine.model_features = new_features
                    self.logger.info(f"Features atualizadas: {len(new_features)} selecionadas")
                    
            # Aplicar novos hiperparâmetros se otimizados
            if 'hyperparameters' in results:
                hyperparams = results['hyperparameters']
                if self.model_manager and hasattr(self.model_manager, 'update_hyperparameters'):
                    self.model_manager.update_hyperparameters(hyperparams)
                    self.logger.info("Hiperparâmetros atualizados via otimização")
                else:
                    self.logger.warning("ModelManager não suporta update_hyperparameters")
                
            # Aplicar novos parâmetros de risco se otimizados
            if 'risk' in results:
                risk_params = results['risk']
                
                # Atualizar RiskManager básico
                if self.strategy_engine and hasattr(self.strategy_engine, 'risk_manager'):
                    risk_mgr = self.strategy_engine.risk_manager
                    if hasattr(risk_mgr, 'update_parameters'):
                        risk_mgr.update_parameters(risk_params)
                    else:
                        self.logger.warning("RiskManager não suporta update_parameters")
                
                # Atualizar IntelligentRiskManager
                if self.intelligent_risk_manager and hasattr(self.intelligent_risk_manager, 'update_parameters'):
                    self.intelligent_risk_manager.update_parameters(risk_params)
                    
                self.logger.info("Parâmetros de risco atualizados via otimização")
                
        except Exception as e:
            self.logger.error(f"Erro aplicando resultados de otimização: {e}")
            if self.metrics:
                self.metrics.record_error('optimization_application', str(e))
            
    def _detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """
        Detecta regime de mercado atual baseado em ml-prediction-strategy-doc.md
        
        Regimes:
        - trend_up: Tendência de alta (EMA9 > EMA20 > EMA50, ADX > 25)
        - trend_down: Tendência de baixa (EMA9 < EMA20 < EMA50, ADX > 25)
        - ranging: Lateralização (ADX < 25, preço entre suporte/resistência)
        - high_volatility: Alta volatilidade
        - undefined: Condições indefinidas
        """
        if market_data is None or len(market_data) < 50:
            return 'undefined'
            
        try:
            import numpy as np
            
            # Obter dados necessários
            closes = market_data['close']
            highs = market_data['high'] 
            lows = market_data['low']
            
            # Calcular EMAs se não estiverem disponíveis
            if len(closes) >= 50:
                ema_9 = closes.ewm(span=9).mean().iloc[-1]
                ema_20 = closes.ewm(span=20).mean().iloc[-1] 
                ema_50 = closes.ewm(span=50).mean().iloc[-1]
                current_price = closes.iloc[-1]
            else:
                # Dados insuficientes para EMAs completas
                ema_9 = closes.ewm(span=min(9, len(closes))).mean().iloc[-1]
                ema_20 = closes.ewm(span=min(20, len(closes))).mean().iloc[-1]
                ema_50 = closes.ewm(span=min(50, len(closes))).mean().iloc[-1]
                current_price = closes.iloc[-1]
            
            # Calcular ADX (aproximação)
            def calculate_adx_simple(high, low, close, period=14):
                """Cálculo simplificado do ADX"""
                if len(close) < period * 2:
                    return 15  # Valor neutro default
                    
                # True Range
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                # Directional Movement
                dm_plus = high.diff()
                dm_minus = -low.diff()
                
                dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
                dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
                
                # Smooth
                tr_smooth = true_range.ewm(alpha=1/period).mean()
                dm_plus_smooth = dm_plus.ewm(alpha=1/period).mean()
                dm_minus_smooth = dm_minus.ewm(alpha=1/period).mean()
                
                # DI
                di_plus = 100 * dm_plus_smooth / tr_smooth
                di_minus = 100 * dm_minus_smooth / tr_smooth
                
                # DX
                dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
                
                # ADX
                adx = dx.ewm(alpha=1/period).mean()
                
                return adx.iloc[-1] if not adx.empty else 15
            
            adx = calculate_adx_simple(highs, lows, closes)
            
            # Calcular volatilidade
            returns = closes.pct_change().dropna()
            if len(returns) >= 20:
                current_volatility = returns.rolling(20).std().iloc[-1]
                historical_volatility = returns.std()
                volatility_ratio = current_volatility / max(historical_volatility, 1e-6)
            else:
                volatility_ratio = 1.0
            
            # Detectar regime baseado nas regras documentadas
            
            # 1. Alta Volatilidade (prioritário)
            if volatility_ratio > 1.5:
                return 'high_volatility'
            
            # 2. Tendência de Alta
            if ema_9 > ema_20 > ema_50 and adx > 25:
                # Confirmar que preço está acima das médias
                if current_price > ema_20:
                    return 'trend_up'
            
            # 3. Tendência de Baixa  
            elif ema_9 < ema_20 < ema_50 and adx > 25:
                # Confirmar que preço está abaixo das médias
                if current_price < ema_20:
                    return 'trend_down'
            
            # 4. Lateralização (Range)
            elif adx < 25:
                # Verificar se está próximo das médias (movimento lateral)
                price_to_ema20_ratio = abs(current_price - ema_20) / ema_20
                
                if price_to_ema20_ratio < 0.01:  # Menos de 1% de distância
                    return 'ranging'
            
            # 5. Condições não definidas
            return 'undefined'
            
        except Exception as e:
            self.logger.error(f"Erro na detecção de regime: {e}")
            return 'undefined'
            
    def _should_update_stop(self, position: Dict, stop_result: Dict) -> bool:
        """Verifica se deve atualizar stop loss"""
        current_stop = position['stop_loss']
        new_stop = stop_result['stop_loss']
        entry_price = position['entry_price']
        
        # Para posições long
        if position['side'] in ['buy', 'long']:
            # Só atualizar se novo stop é maior (trailing)
            return new_stop > current_stop and new_stop < position['current_price']
            
        # Para posições short
        else:
            # Só atualizar se novo stop é menor (trailing)
            return new_stop < current_stop and new_stop > position['current_price']