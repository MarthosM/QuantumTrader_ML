"""
Sistema de Trading Integrado v2.0
Integra todos os componentes em um sistema coeso e funcional
"""

import logging
import threading
import time
import queue
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from ctypes import WINFUNCTYPE, c_int, c_wchar_p, c_double, c_int64, Structure

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
        
        # Threads e queues
        self.ml_queue = queue.Queue(maxsize=10)
        self.signal_queue = queue.Queue(maxsize=10)
        self.ml_thread = None
        self.signal_thread = None
        
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
            'available': config.get('initial_balance', 100000)
        }
        # Monitor visual (opcional)
        self.monitor = None
        self.use_gui = config.get('use_gui', True)

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
    
    def _setup_logger(self) -> logging.Logger:
        """Configura o sistema de logging"""
        logger = logging.getLogger('TradingSystemV2')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
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
                username=self.config['username'],
                password=self.config['password'],
                key=self.config.get('key', '')
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
            
            # 3. Inicializar estrutura de dados
            self.logger.info("3. Inicializando estrutura de dados...")
            self.data_structure = TradingDataStructure()
            self.data_structure.initialize_structure()
            self.logger.info("[ok] Estrutura de dados criada")
            
            # 4. Configurar pipeline de dados
            self.logger.info("4. Configurando pipeline de dados...")
            self.data_pipeline = DataPipeline(self.data_structure)
            self.real_time_processor = RealTimeProcessor(self.data_structure)
            self.data_loader = DataLoader(self.config.get('data_dir', 'data'))
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

            # 7. Configurar estratégia
            self.logger.info("7. Configurando estratégia...")
            signal_gen = SignalGenerator(self.config.get('strategy', {}))
            risk_mgr = RiskManager(self.config.get('risk', {}))
            self.strategy_engine = StrategyEngine(signal_gen, risk_mgr)
            self.logger.info("[ok] Estratégia configurada")

            # 8. Inicializar métricas
            self.logger.info("8. Inicializando sistema de métricas...")
            self.metrics = MetricsCollector()
            self.logger.info("[ok] Sistema de métricas inicializado")

            # 9. Configurar callbacks
            self.logger.info("9. Configurando callbacks...")
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
        Carrega dados históricos de forma segura usando métodos disponíveis
        
        Args:
            ticker: Símbolo do ativo
            days_back: Número de dias para carregar
            
        Returns:
            bool: True se carregou com sucesso
        """
        try:
            # Por enquanto, gerar dados sintéticos para teste
            # Em produção, usar API real ou arquivo
            from datetime import datetime, timedelta
            import pandas as pd
            import numpy as np
            
            # Gerar dados sintéticos
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Criar range de tempo (1 minuto)
            time_range = pd.date_range(start=start_date, end=end_date, freq='1min')
            
            if len(time_range) == 0:
                return False
            
            # Gerar dados OHLCV sintéticos
            base_price = 5000.0
            volatility = 0.02
            
            # Preços aleatórios mas realistas
            np.random.seed(42)  # Para reprodutibilidade
            returns = np.random.normal(0, volatility/100, len(time_range))
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Criar OHLCV
            candles_data = []
            for i, timestamp in enumerate(time_range):
                if i == 0:
                    open_price = base_price
                else:
                    open_price = candles_data[i-1]['close']
                
                close_price = prices[i]
                high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.001))
                low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.001))
                volume = np.random.uniform(100, 1000)
                
                candles_data.append({
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price, 
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
            
            # Converter para DataFrame e atualizar estrutura
            candles_df = pd.DataFrame(candles_data)
            candles_df.set_index('timestamp', inplace=True)
            
            # Atualizar estrutura de dados
            if self.data_structure:
                self.data_structure.update_candles(candles_df)
            else:
                self.logger.error("Data structure não inicializada")
            
            self.logger.info(f"Dados sintéticos gerados: {len(candles_df)} candles")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro carregando dados históricos: {e}")
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
            
            # Usar método correto do DataLoader
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
            
            self.logger.info("Sistema iniciado e operacional!")
            
            # 6. Entrar no loop principal
            self._main_loop()

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
            # Verificar se processador está disponível
            if not self.real_time_processor:
                self.logger.warning("Real time processor não disponível")
                return
                
            # Processar trade
            self.real_time_processor.process_trade(trade_data)
            
            # Atualizar métricas se disponível
            if self.metrics:
                self.metrics.record_trade()
            
                # Log periódico
                if hasattr(self.metrics, 'metrics') and self.metrics.metrics['trades_processed'] % 100 == 0:
                    self.logger.info(f"Trades processados: {self.metrics.metrics['trades_processed']}")
                
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
                
            # Gerar sinal
            signal = self.strategy_engine.process_prediction(
                prediction,
                self.data_structure,
                self.account_info
            )
            
            if signal and signal['action'] != 'none':
                self.logger.info(
                    f"SINAL GERADO: {signal['action'].upper()} "
                    f"@ {signal['price']:.2f} "
                    f"SL: {signal['stop_loss']:.2f} "
                    f"TP: {signal['take_profit']:.2f}"
                )
                
                # Aqui seria executada a ordem
                # Por enquanto, apenas simular
                self._simulate_order_execution(signal)
                
                # Registrar métrica se disponível
                if self.metrics and hasattr(self.metrics, 'metrics'):
                    self.metrics.metrics['signals_generated'] += 1
                
        except Exception as e:
            self.logger.error(f"Erro gerando sinal: {e}")
            
    def _simulate_order_execution(self, signal: Dict):
        """Simula execução de ordem (substituir por execução real)"""
        self.logger.info(f"[SIMULAÇÃO] Executando ordem: {signal['action']}")
        
        # Simular posição aberta
        self.active_positions[self.ticker] = {
            'side': signal['action'],
            'entry_price': signal['price'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'size': signal.get('position_size', 1),
            'entry_time': datetime.now()
        }
        
        # Registrar métrica se disponível
        if self.metrics and hasattr(self.metrics, 'metrics'):
            self.metrics.metrics['signals_executed'] += 1
        
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
        
        # Parar threads
        if self.ml_thread and self.ml_thread.is_alive():
            self.ml_thread.join(timeout=2.0)
            
        if self.signal_thread and self.signal_thread.is_alive():
            self.signal_thread.join(timeout=2.0)
            
        # Desconectar
        if self.connection:
            self.connection.disconnect()

        # Parar monitor se estiver rodando
        if self.monitor:
            self.monitor.stop()
            
        self.logger.info("Sistema parado com sucesso")
        
    def get_status(self) -> Dict:
        """Retorna status atual do sistema"""
        return {
            'running': self.is_running,
            'initialized': self.initialized,
            'ticker': self.ticker,
            'candles': len(self.data_structure.candles) if self.data_structure else 0,
            'last_prediction': self.last_prediction,
            'active_positions': self.active_positions,
            'metrics': self.metrics.get_summary() if self.metrics else {}
        }