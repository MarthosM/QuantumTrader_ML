"""
Connection Manager - Gerencia conexão com ProfitDLL e callbacks
Baseado em real_ml.py e enhanced_historical.py
"""

import logging
import time
import os
import traceback
from typing import Dict, Optional, Callable, Any
from ctypes import WINFUNCTYPE, WinDLL, c_int, c_wchar_p, c_double, c_uint, c_char, c_longlong, c_void_p
from datetime import datetime, timedelta

# Import da estrutura TAssetID do enhanced_historical
from ctypes import Structure

class TAssetID(Structure):
    _fields_ = [
        ("pwcTicker", c_wchar_p),
        ("pwcBolsa", c_wchar_p), 
        ("nFeed", c_int)
    ]

class ConnectionManager:
    """Gerencia conexão com Profit e callbacks essenciais"""
    
    def __init__(self, dll_path: str):
        self.dll_path = dll_path if dll_path else r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
        self.dll = None
        self.connected = False
        self.callbacks = {}
        self.logger = logging.getLogger('ConnectionManager')
        
        # OTIMIZAÇÃO: Configurar logging para reduzir spam no terminal
        self._configure_optimal_logging()
        
        # Estados de conexão (baseado em enhanced_historical.py)
        self.market_connected = False
        self.broker_connected = False
        self.routing_connected = False
        self.login_state = -1
        self.routing_state = -1
        self.market_state = -1

        # Constantes de estado do manual
        self.CONNECTION_STATE_LOGIN = 0
        self.CONNECTION_STATE_ROTEAMENTO = 1
        self.CONNECTION_STATE_MARKET_DATA = 2
        self.CONNECTION_STATE_MARKET_LOGIN = 3

        self.LOGIN_CONNECTED = 0
        self.MARKET_CONNECTED = 4
        self.ROTEAMENTO_BROKER_CONNECTED = 5
                
        # Configurações do servidor
        self.server_address = os.getenv("SERVER_ADDRESS", "producao.nelogica.com.br")
        self.server_port = os.getenv("SERVER_PORT", "8184")
        
        # Callbacks registrados
        self.trade_callbacks = []
        self.state_callbacks = []
        self.order_callbacks = []
        
        # Contadores para debug
        self._historical_data_count = 0
    
    def _configure_optimal_logging(self):
        """
        Configura logging otimizado para evitar spam no terminal
        Especialmente importante durante carregamento de dados históricos
        """
        try:
            # Configurar nível ROOT como WARNING para reduzir spam geral
            root_logger = logging.getLogger()
            if not root_logger.handlers:
                root_logger.setLevel(logging.WARNING)
            
            # Componentes que podem gerar muito spam - configurar como ERROR apenas
            spam_components = [
                'FeatureEngine',
                'ProductionDataValidator',
                'TechnicalIndicators',
                'MLFeatures',
                'RealTimeProcessor',
                'DataPipeline'
            ]
            
            for component in spam_components:
                spam_logger = logging.getLogger(component)
                spam_logger.setLevel(logging.ERROR)
            
            # ConnectionManager e componentes essenciais mantêm INFO
            essential_components = [
                'ConnectionManager',
                'TradingSystem',
                'DataIntegration'
            ]
            
            for component in essential_components:
                essential_logger = logging.getLogger(component)
                essential_logger.setLevel(logging.INFO)
            
            self.logger.info("🔧 Logging otimizado configurado - redução de spam ativada")
            
        except Exception as e:
            # Não falhar se houver problema com logging
            print(f"Aviso: Erro configurando logging otimizado: {e}")
        
    def configure_production_logging(self):
        """
        Configuração específica para ambiente de produção
        Chame este método para logging ainda mais limpo
        """
        try:
            # Em produção, apenas WARNING e ERROR
            production_components = [
                'ConnectionManager',
                'TradingSystem', 
                'DataIntegration',
                'FeatureEngine',
                'ProductionDataValidator'
            ]
            
            for component in production_components:
                prod_logger = logging.getLogger(component)
                prod_logger.setLevel(logging.WARNING)
            
            self.logger.warning("⚡ Modo produção ativado - logging mínimo")
            
        except Exception as e:
            print(f"Erro configurando modo produção: {e}")
        
    def get_logging_stats(self) -> dict:
        """Retorna estatísticas de logging atual"""
        return {
            'historical_data_count': self._historical_data_count,
            'connection_state': {
                'connected': self.connected,
                'market_connected': self.market_connected,
                'login_state': self.login_state
            }
        }
        
    def initialize(self, key: str, username: str, password: str, 
                  account_id: Optional[str] = None, broker_id: Optional[str] = None, 
                  trading_password: Optional[str] = None) -> bool:
        """
        Inicializa conexão com Profit
        Baseado em real_ml.py _initialize_dll()
        
        Args:
            key: Chave de acesso do Profit
            username: Nome de usuário
            password: Senha de login
            account_id: ID da conta (para simulador)
            broker_id: ID da corretora (para simulador)
            trading_password: Senha de trading (se necessária)
        """
        try:
            # Log dos parâmetros (sem senhas por segurança)
            self.logger.info(f"Inicializando conexão com usuário: {username}")
            if account_id:
                self.logger.info(f"Conta: {account_id}")
            if broker_id:
                self.logger.info(f"Corretora: {broker_id}")
            
            # Carregar DLL
            self.dll = self._load_dll()
            if not self.dll:
                return False
            
            # Configurar servidor
            self.logger.info(f"Configurando servidor: {self.server_address}:{self.server_port}")
            server_result = self.dll.SetServerAndPort(
                c_wchar_p(self.server_address),
                c_wchar_p(self.server_port)
            )
            self.logger.info(f"Resultado da configuração do servidor: {server_result}")
            
            # Configurar callbacks básicos
            self._setup_callbacks()
            
            # Conectar usando DLLInitializeLogin (inclui market data e routing)
            self.logger.info("Inicializando conexão completa com Profit...")
            init_result = self.dll.DLLInitializeLogin(
                c_wchar_p(key),
                c_wchar_p(username),
                c_wchar_p(password),
                self.callbacks['state'],           # StateCallback
                self.callbacks['order_history'],   # HistoryCallback
                self.callbacks['order_change'],    # OrderChangeCallback
                self.callbacks['account'],         # AccountCallback
                self.callbacks['trade'],           # NewTradeCallback
                None,                             # NewDailyCallback
                self.callbacks['price_book'],      # PriceBookCallback
                self.callbacks['offer_book'],      # OfferBookCallback
                self.callbacks['history'],         # HistoryTradeCallback
                self.callbacks['progress'],        # ProgressCallback
                None                              # TinyBookCallback
            )
            
            if init_result == 0:  # NL_OK
                self.logger.info("DLL inicializada com sucesso")
                
                # Aguardar conexões
                if self._wait_for_connections(timeout=30):
                    self.connected = True
                    self.logger.info("✅ Conexão estabelecida com sucesso!")
                    return True
                else:
                    self.logger.error("❌ Falha ao estabelecer conexões necessárias")
                    return False
                    
            else:
                self.logger.error(f"Falha na inicialização da DLL: código {init_result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {e}", exc_info=True)
            return False
    
    def _load_dll(self) -> Optional[WinDLL]:
        """Carrega a DLL do Profit"""
        try:
            if not os.path.exists(self.dll_path):
                self.logger.error(f"DLL não encontrada em: {self.dll_path}")
                return None
                
            dll = WinDLL(self.dll_path)
            self.logger.info("DLL carregada com sucesso")
            return dll
            
        except Exception as e:
            self.logger.error(f"Erro carregando DLL: {e}")
            return None
    
    def _setup_callbacks(self):
        """
        Configura callbacks básicos
        Baseado em enhanced_historical.py _initialize_callbacks()
        """
        
        # State callback
        @WINFUNCTYPE(None, c_int, c_int)
        def state_callback(nConnStateType, nResult):
            try:
                self.logger.info(f"State callback: Type={nConnStateType}, Result={nResult}")
                
                # Atualizar estados baseado no tipo
                if nConnStateType == self.CONNECTION_STATE_LOGIN:
                    self.login_state = nResult
                    if nResult == self.LOGIN_CONNECTED:
                        self.logger.info("✅ LOGIN conectado")
                    else:
                        self.logger.warning(f"❌ LOGIN erro: {nResult}")
                        
                elif nConnStateType == self.CONNECTION_STATE_ROTEAMENTO:
                    self.routing_state = nResult
                    self.routing_connected = (nResult == self.ROTEAMENTO_BROKER_CONNECTED)
                    if self.routing_connected:
                        self.logger.info("✅ ROTEAMENTO conectado")
                    else:
                        self.logger.warning(f"⚠️ ROTEAMENTO: {nResult}")
                        
                elif nConnStateType == self.CONNECTION_STATE_MARKET_DATA:
                    self.market_state = nResult
                    self.market_connected = (nResult == self.MARKET_CONNECTED)
                    if self.market_connected:
                        self.logger.info("✅ MARKET DATA conectado")
                    else:
                        self.logger.warning(f"⚠️ MARKET DATA: {nResult}")
                
                # Notificar callbacks registrados
                for callback in self.state_callbacks:
                    callback(nConnStateType, nResult)
                    
            except Exception as e:
                self.logger.error(f"Erro no state callback: {e}")
        
        # Trade callback (tempo real)
        @WINFUNCTYPE(None, TAssetID, c_wchar_p, c_uint, c_double, c_double, 
                     c_int, c_int, c_int, c_int, c_char)
        def trade_callback(asset_id, date, trade_number, price, vol, qtd, 
                          buy_agent, sell_agent, trade_type, b_edit):
            try:
                timestamp = datetime.strptime(str(date), '%d/%m/%Y %H:%M:%S.%f')
                
                trade_data = {
                    'timestamp': timestamp,
                    'ticker': asset_id.pwcTicker,
                    'price': float(price),
                    'volume': float(vol),
                    'quantity': int(qtd),
                    'trade_type': int(trade_type),
                    'trade_number': int(trade_number)
                }
                
                # Notificar callbacks registrados
                for callback in self.trade_callbacks:
                    callback(trade_data)
                    
            except Exception as e:
                self.logger.error(f"Erro no trade callback: {e}")
        
        # History trade callback - IMPLEMENTAÇÃO CRÍTICA COM LOG OTIMIZADO
        @WINFUNCTYPE(None, TAssetID, c_wchar_p, c_uint, c_double, c_double,
                     c_int, c_int, c_int, c_int)
        def history_callback(asset_id, date, trade_number, price, vol, qtd,
                           buy_agent, sell_agent, trade_type):
            try:
                # Este callback recebe os dados históricos!
                ticker_name = asset_id.pwcTicker if asset_id and asset_id.pwcTicker else 'N/A'
                
                # CORREÇÃO: Fazer parsing correto do timestamp
                try:
                    # Tentar diferentes formatos de data que podem vir da DLL
                    date_str = str(date) if date else ""
                    
                    if date_str:
                        # Formatos possíveis da ProfitDLL
                        for fmt in ['%d/%m/%Y %H:%M:%S.%f', '%d/%m/%Y %H:%M:%S', 
                                   '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S']:
                            try:
                                timestamp = datetime.strptime(date_str, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            # Se nenhum formato funcionou, usar timestamp atual
                            if self._historical_data_count < 5:  # Log apenas nos primeiros
                                self.logger.warning(f"Formato de data não reconhecido: {date_str}")
                            timestamp = datetime.now()
                    else:
                        timestamp = datetime.now()
                        
                except Exception as e:
                    if self._historical_data_count < 5:  # Log apenas nos primeiros
                        self.logger.error(f"Erro no parsing de timestamp: {e}")
                    timestamp = datetime.now()
                
                # LOG OTIMIZADO - apenas marcos importantes para evitar spam
                if self._historical_data_count == 0:
                    self.logger.info(f"📈 PRIMEIRO DADO HISTÓRICO: {ticker_name}")
                elif self._historical_data_count in [100, 500, 1000, 5000, 10000, 50000, 100000]:
                    rate = (self._historical_data_count / max(1, (datetime.now() - datetime.fromtimestamp(time.time() - 10)).total_seconds()))
                    self.logger.info(f"📊 {self._historical_data_count} dados recebidos... ({rate:.0f}/s)")
                elif self._historical_data_count % 10000 == 0:
                    self.logger.info(f"� {self._historical_data_count} dados históricos processados")
                
                # Incrementar contador
                self._historical_data_count += 1
                
                # Notificar callbacks registrados sobre novo dado histórico
                for callback in self.trade_callbacks:
                    callback({
                        'timestamp': timestamp,  # Usar timestamp processado
                        'ticker': ticker_name,
                        'price': float(price),
                        'volume': float(vol),
                        'quantity': int(qtd),
                        'trade_type': int(trade_type),
                        'trade_number': int(trade_number),
                        'is_historical': True
                    })
                    
            except Exception as e:
                self.logger.error(f"Erro no history callback: {e}")
        
        # Progress callback - MELHORADO
        @WINFUNCTYPE(None, TAssetID, c_int)
        def progress_callback(asset_id, progress):
            try:
                ticker_name = asset_id.pwcTicker if asset_id and asset_id.pwcTicker else 'N/A'
                if progress % 10 == 0 or progress >= 95:
                    self.logger.info(f"📊 Progresso {ticker_name}: {progress}%")
                
                # Se chegou a 100%, o download está completo
                if progress >= 100:
                    self.logger.info(f"✅ Download de {ticker_name} completo!")
                    
            except Exception as e:
                self.logger.error(f"Erro no progress callback: {e}")
        
        # Callbacks vazios para completar interface
        @WINFUNCTYPE(None, TAssetID, c_int, c_int, c_int, c_longlong, c_int,
                     c_longlong, c_double, c_char, c_char, c_char, c_char, 
                     c_char, c_wchar_p, c_void_p, c_void_p)
        def offer_book_callback(*args):
            pass
        
        @WINFUNCTYPE(None, TAssetID, c_int, c_int, c_int, c_longlong, c_int,
                     c_double, c_void_p, c_void_p)
        def price_book_callback(*args):
            pass
        
        # TOrderChangeCallback tem 17 parâmetros
        @WINFUNCTYPE(None, TAssetID, c_int, c_int, c_int, c_int, c_int, c_double,
                    c_double, c_double, c_longlong, c_wchar_p, c_wchar_p, c_wchar_p,
                    c_wchar_p, c_wchar_p, c_wchar_p, c_wchar_p)
        def order_change_callback(asset_id, corretora, qtd, traded_qtd, leaves_qtd,
                                side, price, stop_price, avg_price, profit_id,
                                tipo_ordem, conta, titular, cl_ord_id, status,
                                date, text_message):
            try:
                # Criar dicionário com dados da ordem
                order_data = {
                    'asset_id': asset_id,
                    'corretora': corretora,
                    'quantity': qtd,
                    'traded_quantity': traded_qtd,
                    'leaves_quantity': leaves_qtd,
                    'side': side,
                    'price': price,
                    'stop_price': stop_price,
                    'avg_price': avg_price,
                    'profit_id': profit_id,
                    'order_type': tipo_ordem,
                    'account': conta,
                    'titular': titular,
                    'client_order_id': cl_ord_id,
                    'status': status,
                    'date': date,
                    'text_message': text_message
                }
                
                # Notificar todos os callbacks registrados
                for callback in self.order_callbacks:
                    try:
                        callback(order_data)
                    except Exception as e:
                        self.logger.error(f"Erro executando callback de ordem: {e}")
                        
            except Exception as e:
                self.logger.error(f"Erro no order_change_callback: {e}")

        # THistoryCallback tem 16 parâmetros  
        @WINFUNCTYPE(None, TAssetID, c_int, c_int, c_int, c_int, c_int, c_double,
                    c_double, c_double, c_longlong, c_wchar_p, c_wchar_p, c_wchar_p,
                    c_wchar_p, c_wchar_p, c_wchar_p)
        def order_history_callback(asset_id, corretora, qtd, traded_qtd, leaves_qtd,
                                side, price, stop_price, avg_price, profit_id,
                                tipo_ordem, conta, titular, cl_ord_id, status, date):
            pass
        
        @WINFUNCTYPE(None, c_int, c_wchar_p, c_wchar_p, c_wchar_p)
        def account_callback(*args):
            pass
        
        # Armazenar callbacks
        self.callbacks = {
            'state': state_callback,
            'trade': trade_callback,
            'history': history_callback,
            'progress': progress_callback,
            'offer_book': offer_book_callback,
            'price_book': price_book_callback,
            'order_change': order_change_callback,
            'order_history': order_history_callback,
            'account': account_callback
        }
        
        self.logger.info("Callbacks configurados")
    
    def _wait_for_connections(self, timeout: int = 30) -> bool:
        """Aguarda conexões serem estabelecidas"""
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            # CORREÇÃO: Aguardar especificamente market data para dados históricos
            if self.market_connected and self.market_state == self.MARKET_CONNECTED:
                self.logger.info("✅ Market Data conectado - dados históricos disponíveis")
                return True
                
            # Log periódico mais detalhado
            elapsed = time.time() - start_time
            if int(elapsed) % 3 == 0 and int(elapsed) > 0:
                self.logger.info(f"Aguardando market data ({int(elapsed)}/{timeout}s)...")
                self._log_connection_states()
                
                # Verificar se pelo menos login funcionou
                if self.login_state != self.LOGIN_CONNECTED:
                    self.logger.warning("Login ainda não conectado - pode impedir market data")
                
            time.sleep(1)
            
        # Se chegou aqui, timeout ocorreu
        self.logger.error("TIMEOUT: Market data não conectou no tempo esperado")
        self._log_connection_states()
        return False
    
    def _log_connection_states(self):
        """Log dos estados de conexão com interpretação detalhada"""
        self.logger.info("=== Estados de Conexão ===")
        self.logger.info(f"Login: {self.login_state} {'✅' if self.login_state == self.LOGIN_CONNECTED else '❌'}")
        self.logger.info(f"Roteamento: {self.routing_state} (conectado: {self.routing_connected}) {'✅' if self.routing_connected else '⚠️'}")
        self.logger.info(f"Market Data: {self.market_state} (conectado: {self.market_connected}) {'✅' if self.market_connected else '⚠️'}")
        self.logger.info(f"Geral: {self.connected}")
        
        # Diagnóstico específico para dados históricos
        if self.login_state == self.LOGIN_CONNECTED:
            self.logger.info("💡 LOGIN OK - Dados históricos DEVEM estar disponíveis")
        else:
            self.logger.error("❌ LOGIN FALHOU - Dados históricos NÃO disponíveis")
            self.logger.error("   Possível solução: Verificar credenciais e conexão com servidor")
            
        self.logger.info("========================")
    
    def register_trade_callback(self, callback: Callable):
        """Registra callback para trades em tempo real"""
        self.trade_callbacks.append(callback)
        
    def register_state_callback(self, callback: Callable):
        """Registra callback para mudanças de estado"""
        self.state_callbacks.append(callback)
        
    def register_order_callback(self, callback: Callable):
        """Registra callback para atualizações de ordens"""
        self.order_callbacks.append(callback)
        
    def register_account_callback(self, callback: Callable):
        """Registra callback para informações da conta"""
        # Implementação temporária - pode ser expandida se necessário
        self.logger.debug("Callback de conta registrado")
        pass
    
    def subscribe_ticker(self, ticker: str, exchange: str = "F") -> bool:
        """Subscreve para receber dados de um ticker"""
        try:
            if self.dll is None:
                self.logger.error("DLL não está carregada. Inicialize antes de subscrever ticker.")
                return False
            if not hasattr(self.dll, "SubscribeTicker"):
                self.logger.error("Método SubscribeTicker não encontrado na DLL.")
                return False
            result = self.dll.SubscribeTicker(c_wchar_p(ticker), c_wchar_p(exchange))
            if result == 0:
                self.logger.info(f"Subscrito para {ticker} em {exchange}")
                return True
            else:
                self.logger.error(f"Falha ao subscrever {ticker}: código {result}")
                return False
        except Exception as e:
            self.logger.error(f"Erro ao subscrever ticker: {e}")
            return False

    def _get_current_wdo_contract(self, reference_date: Optional[datetime] = None) -> str:
        """
        Detecta o contrato WDO correto baseado na data atual e regras de virada
        
        Args:
            reference_date: Data de referência (padrão: agora)
            
        Returns:
            str: Código do contrato WDO atual
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        # Códigos de mês para futuros
        month_codes = {
            1: 'F',   # Janeiro
            2: 'G',   # Fevereiro  
            3: 'H',   # Março
            4: 'J',   # Abril
            5: 'K',   # Maio
            6: 'M',   # Junho
            7: 'N',   # Julho
            8: 'Q',   # Agosto
            9: 'U',   # Setembro
            10: 'V',  # Outubro
            11: 'X',  # Novembro
            12: 'Z'   # Dezembro
        }
        
        current_month = reference_date.month
        current_year = str(reference_date.year)[-2:]  # Últimos 2 dígitos
        current_day = reference_date.day
        
        # REGRA WDO: SEMPRE usar contrato do PRÓXIMO mês
        # Exemplo: TODO mês de julho (01/07 a 31/07) usa WDOQ25 (agosto)
        
        # Calcular próximo mês
        if current_month == 12:
            next_month = 1
            next_year = str(reference_date.year + 1)[-2:]
        else:
            next_month = current_month + 1
            next_year = current_year
            
        contract_month_code = month_codes[next_month]
        contract_year = next_year
        
        self.logger.info(f"📅 Mês {current_month} usa contrato do mês {next_month} (sempre próximo mês)")
        
        contract = f"WDO{contract_month_code}{contract_year}"
        self.logger.info(f"🎯 Contrato WDO detectado: {contract}")
        self.logger.info(f"📊 Data referência: {reference_date.strftime('%d/%m/%Y')}")
        self.logger.info(f"📊 Mês: {current_month}, Código: {contract_month_code}, Ano: {contract_year}")
        
        return contract
    
    def _get_smart_ticker_variations(self, ticker: str) -> list:
        """
        Gera variações inteligentes do ticker considerando viradas de mês
        
        Args:
            ticker: Ticker base
            
        Returns:
            list: Lista ordenada de tickers para tentar (mais provável primeiro)
        """
        variations = []
        
        if ticker.startswith("WDO"):
            self.logger.info(f"🔍 Detectado ticker WDO - aplicando lógica de contratos futuros")
            
            # 1. Contrato atual detectado automaticamente (MAIS PROVÁVEL)
            current_contract = self._get_current_wdo_contract()
            variations.append(current_contract)
            
            # 2. Ticker original fornecido
            if ticker != current_contract:
                variations.append(ticker)
            
            # 3. WDO genérico (funciona em algumas APIs)
            if "WDO" not in variations:
                variations.append("WDO")
            
            # Com a nova lógica corrigida, o contrato atual deve sempre funcionar
            # Removendo tentativas desnecessárias de outros meses
                
        else:
            # Para outros tickers, apenas usar o original
            variations.append(ticker)
        
        # Remover duplicatas mantendo ordem
        unique_variations = []
        for var in variations:
            if var not in unique_variations:
                unique_variations.append(var)
        
        self.logger.info(f"📋 Variações de ticker a serem testadas: {unique_variations}")
        return unique_variations
    
    def request_historical_data(self, ticker: str, start_date: datetime, 
                               end_date: datetime) -> int:
        """Solicita dados históricos - VERSÃO CORRIGIDA"""
        try:
            if self.dll is None:
                self.logger.error("DLL não está carregada. Inicialize antes de solicitar dados históricos.")
                return -1

            # CORREÇÃO PRINCIPAL: Dados históricos só precisam de login, não de market data
            if self.login_state != self.LOGIN_CONNECTED:
                self.logger.error("Login não conectado - dados históricos não disponíveis")
                self.logger.info("Estados atuais:")
                self._log_connection_states()
                return -1
            
            # CORREÇÃO: Para dados históricos, usar apenas login conectado
            self.logger.info("✅ Login conectado - prosseguindo com dados históricos")
            
            # VALIDAÇÃO: LIMITE OTIMIZADO DE 3 DIAS (melhor performance e confiabilidade)
            days_requested = (end_date - start_date).days
            if days_requested > 3:
                self.logger.warning(f"Período solicitado muito longo ({days_requested} dias). API otimizada para máximo de 3 dias.")
                start_date = end_date - timedelta(days=3)
            
            # VALIDAÇÃO: Verificar se as datas não são muito antigas - MÁXIMO 3 DIAS
            days_ago = (datetime.now() - start_date).days
            if days_ago > 3:
                self.logger.warning(f"Data inicial muito antiga ({days_ago} dias atrás). Ajustando para últimos 3 dias.")
                start_date = datetime.now() - timedelta(days=3)
                end_date = datetime.now()
            
            # VALIDAÇÃO: Garantir que não é fim de semana ou feriado
            # Ajustar end_date para último dia útil se necessário
            if end_date.weekday() >= 5:  # Sábado (5) ou Domingo (6)
                days_to_subtract = end_date.weekday() - 4  # Volta para sexta-feira
                end_date = end_date - timedelta(days=days_to_subtract)
                self.logger.info(f"End_date ajustado para último dia útil: {end_date.date()}")
            
            # Recalcular start_date após ajuste de end_date
            start_date = end_date - timedelta(days=min(days_requested, 3))
            
            self.logger.info(f"Solicitando dados históricos para {ticker} - LIMITE: 3 DIAS OTIMIZADO")
            self.logger.info(f"Período final: {start_date.date()} até {end_date.date()} ({(end_date - start_date).days} dias)")
            self.logger.info("⚡ OTIMIZADO: API funciona melhor com limite de 3 dias!")
            
            # RESET: Limpar contadores antes de nova requisição
            self._historical_data_count = 0
            
            # CORREÇÃO: Usar sistema inteligente de detecção de ticker
            tickers_to_try = self._get_smart_ticker_variations(ticker)

            # CORREÇÃO: Usar formatos de data que funcionam com WDO na B3
            # WDO é negociado na BM&F (exchange "F"), com horário específico
            start_str = start_date.replace(hour=9, minute=0, second=0).strftime('%d/%m/%Y %H:%M:%S')
            end_str = end_date.replace(hour=18, minute=0, second=0).strftime('%d/%m/%Y %H:%M:%S')
            
            self.logger.info(f"Período: {start_str} até {end_str}")
            
            # CORREÇÃO: Usar GetHistoryTrades com parâmetros corretos
            if hasattr(self.dll, "GetHistoryTrades"):
                
                for test_ticker in tickers_to_try:
                    self.logger.info(f"Testando ticker: {test_ticker}")
                    
                    try:
                        # CORREÇÃO PRINCIPAL: Usar bolsa/exchange correta para WDO
                        exchange = "F" if test_ticker.startswith("WDO") else ""
                        
                        # CORREÇÃO 1: Tentar primeiro com formato correto de data
                        start_full = start_date.strftime('%d/%m/%Y %H:%M:%S')
                        end_full = end_date.strftime('%d/%m/%Y %H:%M:%S')
                        
                        self.logger.info(f"Tentando {test_ticker} na bolsa '{exchange}' com período {start_full} - {end_full}")
                        
                        result = self.dll.GetHistoryTrades(
                            c_wchar_p(test_ticker),
                            c_wchar_p(exchange),
                            c_wchar_p(start_full),
                            c_wchar_p(end_full)
                        )
                        self.logger.info(f"Resultado GetHistoryTrades para {test_ticker}: {result}")
                        
                        # CORREÇÃO: Interpretar códigos de retorno corretamente
                        if result >= 0:
                            self.logger.info(f"✅ Solicitação aceita para ticker {test_ticker}!")
                            return result
                        elif result == -2147483645:
                            self.logger.warning(f"Erro de parâmetros para {test_ticker}, tentando outras variações...")
                            
                            # Tentar sem horários específicos (apenas data)
                            start_simple = start_date.strftime('%d/%m/%Y')
                            end_simple = end_date.strftime('%d/%m/%Y')
                            
                            self.logger.info(f"Tentando formato simples: {start_simple} - {end_simple}")
                            result2 = self.dll.GetHistoryTrades(
                                c_wchar_p(test_ticker),
                                c_wchar_p(exchange),
                                c_wchar_p(start_simple),
                                c_wchar_p(end_simple)
                            )
                            
                            if result2 >= 0:
                                self.logger.info(f"✅ Sucesso com formato simples para {test_ticker}!")
                                return result2
                                
                            # Tentar com exchange vazio
                            self.logger.info(f"Tentando com exchange vazio")
                            result3 = self.dll.GetHistoryTrades(
                                c_wchar_p(test_ticker),
                                c_wchar_p(""),
                                c_wchar_p(start_simple),
                                c_wchar_p(end_simple)
                            )
                            
                            if result3 >= 0:
                                self.logger.info(f"✅ Sucesso com exchange vazio para {test_ticker}!")
                                return result3
                                
                            # Último recurso: formato americano
                            start_us = start_date.strftime('%m/%d/%Y')
                            end_us = end_date.strftime('%m/%d/%Y')
                            
                            self.logger.info(f"Tentando formato americano: {start_us} - {end_us}")
                            result4 = self.dll.GetHistoryTrades(
                                c_wchar_p(test_ticker),
                                c_wchar_p(exchange),
                                c_wchar_p(start_us),
                                c_wchar_p(end_us)
                            )
                            
                            if result4 >= 0:
                                self.logger.info(f"✅ Sucesso com formato americano para {test_ticker}!")
                                return result4
                                
                        else:
                            self.logger.warning(f"Erro {result} para ticker {test_ticker}")
                            
                    except Exception as e:
                        self.logger.error(f"Erro ao testar ticker {test_ticker}: {e}")
            
            self.logger.error("=== DIAGNÓSTICO DE ERRO (ATUALIZADO) ===")
            self.logger.error("✅ DESCOBERTA: API funciona mas tem limitações específicas!")
            self.logger.error("")
            self.logger.error("🎯 LIMITE OTIMIZADO: MÁXIMO 3 DIAS de dados históricos")
            self.logger.error(f"   - Período solicitado: {start_date.date()} - {end_date.date()}")
            self.logger.error(f"   - Dias solicitados: {(end_date - start_date).days}")
            self.logger.error("   - Períodos longos causam erro -2147483645")
            self.logger.error("")
            self.logger.error("🔄 SISTEMA INTELIGENTE DE TICKERS:")
            self.logger.error("   - Detecta automaticamente contratos WDO atuais")
            self.logger.error("   - Considera viradas de mês (dia 15)")
            self.logger.error("   - Testa múltiplas variações em ordem de prioridade")
            self.logger.error("")
            self.logger.error("💡 RECOMENDAÇÕES:")
            self.logger.error("   1. SEMPRE usar períodos ≤ 3 dias")
            self.logger.error("   2. Sistema detecta contratos WDO automaticamente")
            self.logger.error("   3. Verificar se ticker está ativo no pregão")
            self.logger.error("   4. Timeout otimizado para 30s")
            self.logger.error("=============================")
            return -1
            
        except Exception as e:
            self.logger.error(f"Erro solicitando dados históricos: {e}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return -1
    
    def wait_for_historical_data(self, timeout_seconds: int = 60) -> bool:
        """
        Aguarda os dados históricos chegarem via callback
        PROTEÇÃO CONTRA LOOPS - timeout aumentado para 60s para dados históricos
        
        Args:
            timeout_seconds: Timeout em segundos (padrão 60s para dados históricos)
            
        Returns:
            bool: True se dados foram recebidos, False se timeout
        """
        try:
            start_time = time.time()
            last_count = self._historical_data_count
            stable_count = 0
            no_data_count = 0
            last_log_time = 0
            
            self.logger.info(f"⏳ Aguardando dados históricos (timeout: {timeout_seconds}s)...")
            self.logger.info(f"📊 Contador inicial: {self._historical_data_count}")
            
            while (time.time() - start_time) < timeout_seconds:
                current_count = self._historical_data_count
                elapsed = time.time() - start_time
                
                # Se há dados chegando
                if current_count > last_count:
                    last_count = current_count
                    stable_count = 0
                    no_data_count = 0
                    
                    # Log apenas a cada 5 segundos para evitar spam
                    if elapsed - last_log_time >= 5:
                        rate = (current_count / elapsed) if elapsed > 0 else 0
                        self.logger.info(f"📈 {current_count} dados recebidos... ({elapsed:.1f}s, {rate:.0f} trades/s)")
                        last_log_time = elapsed
                        
                else:
                    # Dados estáveis, contar tempo
                    stable_count += 1
                    no_data_count += 1
                    
                    # PROTEÇÃO 1: Se estável por 5 segundos e temos dados, considerar completo
                    if stable_count >= 10 and current_count > 0:  # 10 * 0.5s = 5s
                        self.logger.info(f"✅ Dados históricos carregados: {current_count} registros em {elapsed:.1f}s")
                        
                        # NOVO: Notificar callbacks sobre fim do carregamento histórico
                        self._notify_historical_data_complete()
                        
                        return True
                    
                    # PROTEÇÃO 2: Se passou 90 segundos sem dados, desistir
                    if no_data_count >= 180 and current_count == 0:  # 180 * 0.5s = 90s
                        self.logger.warning(f"⚠️ 90 segundos sem dados - assumindo que não há dados disponíveis")
                        return False
                    
                    # PROTEÇÃO 3: Log periódico menos frequente
                    if int(elapsed) % 10 == 0 and int(elapsed) > 0 and elapsed - last_log_time >= 10:
                        self.logger.info(f"⏳ Aguardando... {current_count} dados recebidos em {elapsed:.0f}s")
                        last_log_time = elapsed
                
                time.sleep(0.5)
            
            # Timeout atingido
            final_count = self._historical_data_count
            elapsed_final = time.time() - start_time
            
            if final_count > 0:
                self.logger.warning(f"⚠️ Timeout após {elapsed_final:.1f}s, mas {final_count} dados foram recebidos")
                
                # NOVO: Mesmo com timeout, notificar fim do carregamento se temos dados
                self._notify_historical_data_complete()
                
                return True
            else:
                self.logger.error(f"❌ Timeout após {elapsed_final:.1f}s sem nenhum dado recebido")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro aguardando dados históricos: {e}")
            return False
    
    def _notify_historical_data_complete(self):
        """
        Notifica que o carregamento de dados históricos foi completado
        """
        try:
            self.logger.info("🎉 Carregamento de dados históricos FINALIZADO!")
            
            # Notificar todos os callbacks registrados sobre conclusão
            for callback in self.trade_callbacks:
                # Enviar sinal especial de conclusão
                try:
                    callback({
                        'event_type': 'historical_data_complete',
                        'total_records': self._historical_data_count,
                        'timestamp': datetime.now()
                    })
                except Exception as e:
                    self.logger.debug(f"Callback não suporta evento de conclusão: {e}")
                    
        except Exception as e:
            self.logger.error(f"Erro notificando fim dos dados históricos: {e}")
    
    def unsubscribe_ticker(self, ticker: str) -> bool:
        """
        Cancela subscrição de um ticker
        
        Args:
            ticker: Código do ativo
            
        Returns:
            bool: Sucesso da operação
        """
        try:
            # Implementar chamada específica da DLL
            # Por enquanto, apenas log
            self.logger.info(f"Cancelando subscrição de {ticker}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao cancelar subscrição: {e}")
            return False
        
    def disconnect(self):
        """Desconecta e limpa recursos"""
        try:
            if self.dll:
                self.dll.DLLFinalize()
                self.logger.info("DLL finalizada")
            self.connected = False
        except Exception as e:
            self.logger.error(f"Erro ao desconectar: {e}")

    def get_account_info(self) -> bool:
        """
        Solicita informações das contas disponíveis
        
        Returns:
            bool: Sucesso da operação
        """
        try:
            if not self.dll or not self.connected:
                self.logger.error("DLL não está conectada")
                return False
                
            result = self.dll.GetAccount()
            if result == 0:  # NL_OK
                self.logger.info("Solicitação de contas enviada")
                return True
            else:
                self.logger.error(f"Erro ao solicitar contas: código {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro ao obter informações de conta: {e}")
            return False
        
    def _validate_market_data(self, data: Dict) -> bool:
        """
        Valida que dados são reais e não dummy
        
        Args:
            data: Dados recebidos do broker
            
        Returns:
            bool: True se dados são válidos
        """
        # Em produção, validação rigorosa
        if os.getenv('TRADING_ENV') == 'PRODUCTION':
            # Verificar fonte
            if not self.market_connected:
                self.logger.error("Dados recebidos sem conexão de market data")
                return False
                
            # Verificar timestamp
            if 'timestamp' in data:
                data_age = (datetime.now() - data['timestamp']).total_seconds()
                if data_age > 5:  # Mais de 5 segundos
                    self.logger.error(f"Dados muito antigos: {data_age}s")
                    return False
            
            # Verificar valores suspeitos
            if 'price' in data:
                # WDO tem preços típicos entre 4000-6000
                if data['price'] < 3000 or data['price'] > 10000:
                    self.logger.error(f"Preço suspeito para WDO: {data['price']}")
                    return False
        
        return True
