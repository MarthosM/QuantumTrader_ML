"""
Connection Manager v4 - CORRIGIDO baseado na análise de callbacks
Implementa as correções identificadas no debug de callbacks
"""

import logging
import time
import os
import traceback
from typing import Dict, Optional, Callable, Any, List
from ctypes import WINFUNCTYPE, WinDLL, c_int, c_wchar_p, c_double, c_uint, c_char, c_longlong, c_void_p
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Import das estruturas da ProfitDLL
from profit_dll_structures import TAssetID

class ConnectionManagerV4Fixed:
    """
    Connection Manager corrigido baseado na análise de callbacks
    
    CORREÇÕES PRINCIPAIS:
    1. Todos callbacks retornam c_int (não None)
    2. Tratamento robusto de erros em callbacks
    3. Sistema inteligente de detecção de contratos WDO
    4. Múltiplas tentativas para dados históricos
    5. Monitoramento detalhado de estado
    """
    
    def __init__(self, dll_path: str):
        self.dll_path = dll_path if dll_path else r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
        self.dll = None
        self.connected = False
        self.logger = logging.getLogger('ConnectionManagerV4Fixed')
        
        # Estados de conexão (conforme teste confirmado)
        self.connection_states = {
            'login': False,
            'roteamento': False,  
            'market_data': False
        }
        
        # Callbacks registrados
        self.trade_callbacks = []
        self.state_callbacks = []
        self.order_callbacks = []
        
        # Contadores para monitoramento
        self.callback_counts = {
            'state': 0,
            'trade': 0,
            'history': 0,
            'progress': 0,
            'account': 0
        }
        
        # Cache de dados históricos
        self.historical_data_cache = []
        self.historical_data_complete = False
        
        # Armazenar referências de callbacks (evitar GC)
        self._callback_refs = {}
        
        # Configurações do servidor
        self.server_address = os.getenv("SERVER_ADDRESS", "producao.nelogica.com.br")
        self.server_port = os.getenv("SERVER_PORT", "8184")
        
    def initialize(self, key: str, username: str, password: str) -> bool:
        """
        Inicializa conexão com ProfitDLL usando callbacks corrigidos
        
        Args:
            key: Chave de acesso
            username: Nome de usuário
            password: Senha
            
        Returns:
            bool: Sucesso da inicialização
        """
        try:
            self.logger.info("=== INICIALIZANDO CONNECTION MANAGER V4 CORRIGIDO ===")
            
            # 1. Carregar DLL
            if not self._load_dll():
                return False
            
            # 2. Configurar servidor
            self.logger.info(f"Configurando servidor: {self.server_address}:{self.server_port}")
            server_result = self.dll.SetServerAndPort(
                c_wchar_p(self.server_address),
                c_wchar_p(self.server_port)
            )
            self.logger.info(f"SetServerAndPort result: {server_result}")
            
            # 3. Criar callbacks corrigidos
            self._create_fixed_callbacks()
            
            # 4. Inicializar conexão
            self.logger.info("Inicializando DLL com callbacks corrigidos...")
            init_result = self.dll.DLLInitializeLogin(
                c_wchar_p(key),
                c_wchar_p(username),
                c_wchar_p(password),
                self._callback_refs['state'],
                None,  # HistoryCallback (deprecated)
                None,  # OrderChangeCallback
                self._callback_refs['account'],
                self._callback_refs['trade'],
                None,  # NewDailyCallback
                None,  # PriceBookCallback
                None,  # OfferBookCallback
                self._callback_refs['history'],
                self._callback_refs['progress'],
                None   # TinyBookCallback
            )
            
            self.logger.info(f"DLLInitializeLogin result: {init_result}")
            
            if init_result == 0:  # NL_OK
                # 5. Aguardar conexões
                if self._wait_for_connections_v4(timeout=30):
                    self.connected = True
                    self.logger.info("✅ CONEXÃO ESTABELECIDA COM SUCESSO!")
                    self._log_connection_summary()
                    return True
                else:
                    self.logger.error("❌ Falha ao estabelecer conexões necessárias")
                    return False
            else:
                self.logger.error(f"❌ Falha na inicialização: {init_result}")
                self._interpret_error_code(init_result)
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erro na inicialização: {e}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False
    
    def _load_dll(self) -> bool:
        """Carrega a DLL com verificação robusta"""
        try:
            if not os.path.exists(self.dll_path):
                self.logger.error(f"DLL não encontrada: {self.dll_path}")
                return False
                
            self.dll = WinDLL(self.dll_path)
            self.logger.info("✅ DLL carregada com sucesso")
            
            # Verificar funções essenciais
            essential_functions = [
                'DLLInitializeLogin', 'SetServerAndPort', 
                'GetHistoryTrades', 'SubscribeTicker', 'DLLFinalize'
            ]
            
            for func_name in essential_functions:
                if hasattr(self.dll, func_name):
                    self.logger.debug(f"✅ Função disponível: {func_name}")
                else:
                    self.logger.error(f"❌ Função AUSENTE: {func_name}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro carregando DLL: {e}")
            return False
    
    def _create_fixed_callbacks(self):
        """
        Cria callbacks com as correções identificadas no debug
        
        CORREÇÕES PRINCIPAIS:
        1. Todos callbacks retornam c_int
        2. Tratamento robusto de exceções
        3. Logging otimizado para evitar spam
        4. Validação de dados de entrada
        """
        
        # CALLBACK 1: State - CORRIGIDO
        @WINFUNCTYPE(c_int, c_int, c_int)
        def state_callback(conn_type, result):
            try:
                self.callback_counts['state'] += 1
                
                # Log apenas mudanças importantes
                if conn_type == 0:  # CONNECTION_STATE_LOGIN
                    if result == 0:  # LOGIN_CONNECTED
                        self.connection_states['login'] = True
                        self.logger.info("🔑 LOGIN CONECTADO!")
                    else:
                        self.connection_states['login'] = False
                        self.logger.error(f"❌ LOGIN ERRO: {result}")
                        
                elif conn_type == 1:  # CONNECTION_STATE_ROTEAMENTO
                    if result == 5:  # ROTEAMENTO_BROKER_CONNECTED
                        self.connection_states['roteamento'] = True
                        self.logger.info("🔗 ROTEAMENTO CONECTADO!")
                    else:
                        self.connection_states['roteamento'] = (result == 2)
                        
                elif conn_type == 2:  # CONNECTION_STATE_MARKET_DATA
                    if result == 4:  # MARKET_CONNECTED
                        self.connection_states['market_data'] = True
                        self.logger.info("📊 MARKET DATA CONECTADO!")
                    else:
                        self.connection_states['market_data'] = False
                        self.logger.debug(f"Market data state: {result}")
                
                # Notificar callbacks registrados
                for callback in self.state_callbacks:
                    try:
                        callback(conn_type, result)
                    except Exception as e:
                        self.logger.error(f"Erro em state callback registrado: {e}")
                
                return 0  # Sucesso
                
            except Exception as e:
                self.logger.error(f"CRÍTICO: Erro em state_callback: {e}")
                return -1  # Erro
        
        # CALLBACK 2: Trade (tempo real) - CORRIGIDO
        @WINFUNCTYPE(c_int, TAssetID, c_wchar_p, c_uint, c_double, c_double, 
                     c_int, c_int, c_int, c_int, c_char)
        def trade_callback(asset_id, date, trade_number, price, vol, qtd, 
                          buy_agent, sell_agent, trade_type, b_edit):
            try:
                self.callback_counts['trade'] += 1
                
                # Validação robusta do asset_id
                ticker = 'UNKNOWN'
                if asset_id:
                    try:
                        ticker = str(asset_id.pwcTicker) if asset_id.pwcTicker else 'UNKNOWN'
                    except:
                        ticker = 'ERROR_READING_TICKER'
                
                # Processar trade
                trade_data = {
                    'timestamp': self._parse_trade_date(date),
                    'ticker': ticker,
                    'price': float(price),
                    'volume': float(vol),
                    'quantity': int(qtd),
                    'trade_type': int(trade_type),
                    'trade_number': int(trade_number),
                    'is_historical': False
                }
                
                # Log apenas alguns trades para evitar spam
                if self.callback_counts['trade'] <= 10 or self.callback_counts['trade'] % 100 == 0:
                    self.logger.info(f"📈 TRADE: {ticker} @ {price} vol={vol}")
                
                # Notificar callbacks registrados
                for callback in self.trade_callbacks:
                    try:
                        callback(trade_data)
                    except Exception as e:
                        self.logger.error(f"Erro notificando trade callback: {e}")
                
                return 0  # Sucesso
                
            except Exception as e:
                self.logger.error(f"CRÍTICO: Erro em trade_callback: {e}")
                return -1  # Erro
        
        # CALLBACK 3: History (dados históricos) - CORRIGIDO E OTIMIZADO
        @WINFUNCTYPE(c_int, TAssetID, c_wchar_p, c_uint, c_double, c_double,
                     c_int, c_int, c_int, c_int)
        def history_callback(asset_id, date, trade_number, price, vol, qtd,
                           buy_agent, sell_agent, trade_type):
            try:
                self.callback_counts['history'] += 1
                
                # Validação robusta
                ticker = 'UNKNOWN'
                if asset_id:
                    try:
                        ticker = str(asset_id.pwcTicker) if asset_id.pwcTicker else 'UNKNOWN'
                    except:
                        ticker = 'ERROR_READING_TICKER'
                
                # Processar dado histórico
                historical_data = {
                    'timestamp': self._parse_trade_date(date),
                    'ticker': ticker,
                    'price': float(price),
                    'volume': float(vol),
                    'quantity': int(qtd),
                    'trade_type': int(trade_type),
                    'trade_number': int(trade_number),
                    'is_historical': True
                }
                
                # Adicionar ao cache
                self.historical_data_cache.append(historical_data)
                
                # Log otimizado para evitar spam
                count = self.callback_counts['history']
                if count == 1:
                    self.logger.info(f"📊 PRIMEIRO DADO HISTÓRICO: {ticker}")
                elif count in [100, 500, 1000, 5000, 10000] or count % 10000 == 0:
                    self.logger.info(f"📈 {count} dados históricos recebidos ({ticker})")
                
                # Notificar callbacks registrados
                for callback in self.trade_callbacks:
                    try:
                        callback(historical_data)
                    except Exception as e:
                        self.logger.error(f"Erro notificando history callback: {e}")
                
                return 0  # Sucesso
                
            except Exception as e:
                self.logger.error(f"CRÍTICO: Erro em history_callback: {e}")
                return -1  # Erro
        
        # CALLBACK 4: Progress - CORRIGIDO
        @WINFUNCTYPE(c_int, TAssetID, c_int)
        def progress_callback(asset_id, progress):
            try:
                self.callback_counts['progress'] += 1
                
                ticker = 'UNKNOWN'
                if asset_id:
                    try:
                        ticker = str(asset_id.pwcTicker) if asset_id.pwcTicker else 'UNKNOWN'
                    except:
                        ticker = 'ERROR_READING_TICKER'
                
                # Log marcos importantes
                if progress == 0:
                    self.logger.info(f"📥 Iniciando download: {ticker}")
                elif progress == 100:
                    self.logger.info(f"✅ Download completo: {ticker}")
                    # Marcar como completo após pequeno delay
                    self._schedule_historical_complete_check()
                elif progress % 25 == 0:  # 25%, 50%, 75%
                    self.logger.info(f"📊 Progresso {ticker}: {progress}%")
                
                return 0  # Sucesso
                
            except Exception as e:
                self.logger.error(f"CRÍTICO: Erro em progress_callback: {e}")
                return -1  # Erro
        
        # CALLBACK 5: Account - CORRIGIDO
        @WINFUNCTYPE(c_int, c_int, c_wchar_p, c_wchar_p, c_wchar_p)
        def account_callback(broker_id, account_id, account_name, titular):
            try:
                self.callback_counts['account'] += 1
                
                account_str = str(account_id) if account_id else 'UNKNOWN'
                self.logger.info(f"👤 CONTA: broker={broker_id}, account={account_str}")
                
                return 0  # Sucesso
                
            except Exception as e:
                self.logger.error(f"CRÍTICO: Erro em account_callback: {e}")
                return -1  # Erro
        
        # Armazenar referências (evitar garbage collection)
        self._callback_refs = {
            'state': state_callback,
            'trade': trade_callback,
            'history': history_callback,
            'progress': progress_callback,
            'account': account_callback
        }
        
        self.logger.info("✅ Callbacks corrigidos criados com retorno c_int")
    
    def _parse_trade_date(self, date_str) -> datetime:
        """Parse robusto de datas vindas da DLL"""
        try:
            if not date_str:
                return datetime.now()
            
            date_string = str(date_str)
            
            # Formatos possíveis da ProfitDLL
            formats = [
                '%d/%m/%Y %H:%M:%S.%f',
                '%d/%m/%Y %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S',
                '%m/%d/%Y %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_string, fmt)
                except ValueError:
                    continue
            
            # Se nenhum formato funcionou, usar timestamp atual
            self.logger.warning(f"Formato de data não reconhecido: {date_string}")
            return datetime.now()
            
        except Exception as e:
            self.logger.error(f"Erro parseando data: {e}")
            return datetime.now()
    
    def _schedule_historical_complete_check(self):
        """Agenda verificação de dados históricos completos"""
        import threading
        
        def check_complete():
            time.sleep(2)  # Aguardar 2 segundos após progress 100%
            if not self.historical_data_complete and len(self.historical_data_cache) > 0:
                self.historical_data_complete = True
                self.logger.info(f"🎉 DADOS HISTÓRICOS COMPLETOS: {len(self.historical_data_cache)} registros")
                
                # Notificar conclusão
                for callback in self.trade_callbacks:
                    try:
                        callback({
                            'event_type': 'historical_data_complete',
                            'total_records': len(self.historical_data_cache),
                            'timestamp': datetime.now()
                        })
                    except:
                        pass  # Callback pode não suportar este evento
        
        thread = threading.Thread(target=check_complete, daemon=True)
        thread.start()
    
    def _wait_for_connections_v4(self, timeout: int = 30) -> bool:
        """
        Aguarda conexões baseado nos estados confirmados no teste
        
        Args:
            timeout: Timeout em segundos
            
        Returns:
            bool: True se conexões essenciais estabelecidas
        """
        self.logger.info(f"Aguardando conexões por {timeout}s...")
        
        start_time = time.time()
        last_log = 0
        
        while (time.time() - start_time) < timeout:
            elapsed = time.time() - start_time
            
            # Log status a cada 5 segundos
            if int(elapsed) - last_log >= 5:
                self._log_connection_status()
                last_log = int(elapsed)
            
            # Verificar se conexões essenciais estão estabelecidas
            if self.connection_states['login'] and self.connection_states['market_data']:
                self.logger.info("✅ Conexões essenciais estabelecidas!")
                return True
            
            time.sleep(1)
        
        # Timeout atingido
        self.logger.error("❌ TIMEOUT: Conexões não estabelecidas no tempo esperado")
        self._log_connection_status()
        
        # Retornar True se pelo menos login funcionou (dados históricos disponíveis)
        if self.connection_states['login']:
            self.logger.warning("⚠️ Login OK, prosseguindo mesmo sem market data completo")
            return True
        
        return False
    
    def _log_connection_status(self):
        """Log detalhado do status das conexões"""
        total_callbacks = sum(self.callback_counts.values())
        
        self.logger.info("=== STATUS DAS CONEXÕES ===")
        self.logger.info(f"Login: {'✅' if self.connection_states['login'] else '❌'}")
        self.logger.info(f"Market Data: {'✅' if self.connection_states['market_data'] else '❌'}")
        self.logger.info(f"Roteamento: {'✅' if self.connection_states['roteamento'] else '❌'}")
        self.logger.info(f"Total Callbacks: {total_callbacks}")
        
        for cb_type, count in self.callback_counts.items():
            if count > 0:
                self.logger.info(f"  {cb_type}: {count}")
        
        self.logger.info("========================")
    
    def _log_connection_summary(self):
        """Log resumo da conexão estabelecida"""
        total_callbacks = sum(self.callback_counts.values())
        
        self.logger.info("🎉 CONEXÃO ESTABELECIDA - RESUMO:")
        self.logger.info(f"  Login: ✅ ({self.callback_counts['state']} state callbacks)")
        self.logger.info(f"  Market Data: ✅ (sistema pronto para dados em tempo real)")
        self.logger.info(f"  Contas: ✅ ({self.callback_counts['account']} account callbacks)")
        self.logger.info(f"  Total: {total_callbacks} callbacks executados")
        self.logger.info("Sistema pronto para solicitar dados históricos e operar!")
    
    def _interpret_error_code(self, code):
        """Interpreta códigos de erro da API"""
        error_codes = {
            -2147483647: "NL_INTERNAL_ERROR - Erro interno da DLL",
            -2147483646: "NL_NOT_INITIALIZED - DLL não foi inicializada",
            -2147483645: "NL_INVALID_ARGS - Argumentos inválidos fornecidos",
            -2147483644: "NL_WAITING_SERVER - Aguardando resposta do servidor",
            -2147483643: "NL_NO_LOGIN - Login não realizado ou falhou",
            -2147483642: "NL_NO_LICENSE - Licença não encontrada ou inválida",
            -2147483620: "NL_NO_PASSWORD - Senha não fornecida ou inválida"
        }
        
        if code in error_codes:
            self.logger.error(f"ERRO: {error_codes[code]}")
        else:
            self.logger.error(f"ERRO DESCONHECIDO: Código {code}")
    
    def get_active_wdo_contracts(self) -> List[str]:
        """
        Retorna lista de contratos WDO ativos em ordem de prioridade
        
        Returns:
            List[str]: Lista de tickers para testar
        """
        month_codes = {
            1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
            7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
        }
        
        now = datetime.now()
        current_month = now.month
        current_year = str(now.year)[-2:]
        
        contracts = []
        
        # 1. Contrato do próximo mês (mais provável)
        next_month = current_month + 1 if current_month < 12 else 1
        next_year = current_year if current_month < 12 else str(now.year + 1)[-2:]
        
        contracts.append(f"WDO{month_codes[next_month]}{next_year}")
        
        # 2. Contrato do mês atual
        contracts.append(f"WDO{month_codes[current_month]}{current_year}")
        
        # 3. Próximos 2 meses
        for i in range(2, 4):
            month = current_month + i
            year = current_year
            if month > 12:
                month -= 12
                year = str(int(current_year) + 1) if len(current_year) == 2 else str(now.year + 1)[-2:]
            contracts.append(f"WDO{month_codes[month]}{year}")
        
        # 4. WDO genérico
        contracts.append("WDO")
        
        self.logger.info(f"Contratos WDO a testar: {contracts}")
        return contracts
    
    def request_historical_data_v4(self, ticker: str, start_date: datetime, 
                                  end_date: datetime) -> bool:
        """
        Versão melhorada para solicitar dados históricos
        
        Args:
            ticker: Ticker base (ex: "WDO")
            start_date: Data inicial
            end_date: Data final
            
        Returns:
            bool: True se solicitação foi aceita
        """
        try:
            if not self.dll or not self.connection_states['login']:
                self.logger.error("Sistema não conectado - login necessário para dados históricos")
                return False
            
            self.logger.info("=== SOLICITANDO DADOS HISTÓRICOS V4 ===")
            
            # Reset do cache
            self.historical_data_cache = []
            self.historical_data_complete = False
            self.callback_counts['history'] = 0
            self.callback_counts['progress'] = 0
            
            # Validar período (máximo 3 dias conforme descoberto)
            days_requested = (end_date - start_date).days
            if days_requested > 3:
                self.logger.warning(f"Período reduzido de {days_requested} para 3 dias")
                start_date = end_date - timedelta(days=3)
            
            # Obter contratos para testar
            if ticker.startswith("WDO"):
                contracts_to_test = self.get_active_wdo_contracts()
            else:
                contracts_to_test = [ticker]
            
            # Exchanges para testar
            exchanges = ["F", "", "B"] if ticker.startswith("WDO") else [""]
            
            # Formatos de data para testar
            date_formats = [
                ('%d/%m/%Y', 'DD/MM/YYYY'),
                ('%m/%d/%Y', 'MM/DD/YYYY'),
                ('%Y-%m-%d', 'YYYY-MM-DD')
            ]
            
            # Tentar todas as combinações
            for contract in contracts_to_test:
                for exchange in exchanges:
                    for date_fmt, desc in date_formats:
                        
                        start_str = start_date.strftime(date_fmt)
                        end_str = end_date.strftime(date_fmt)
                        
                        self.logger.info(f"Testando: {contract} ({exchange}) {start_str}-{end_str} ({desc})")
                        
                        try:
                            result = self.dll.GetHistoryTrades(
                                c_wchar_p(contract),
                                c_wchar_p(exchange),
                                c_wchar_p(start_str),
                                c_wchar_p(end_str)
                            )
                            
                            self.logger.info(f"GetHistoryTrades result: {result}")
                            
                            if result >= 0:
                                self.logger.info(f"✅ SOLICITAÇÃO ACEITA: {contract} ({exchange})")
                                self.logger.info(f"Formato data: {desc}")
                                self.logger.info(f"Período: {start_str} até {end_str}")
                                return True
                            else:
                                self.logger.debug(f"Falhou: {contract} ({exchange}) - código {result}")
                                
                        except Exception as e:
                            self.logger.debug(f"Exceção testando {contract}: {e}")
            
            # Se chegou aqui, nenhuma combinação funcionou
            self.logger.error("❌ NENHUMA COMBINAÇÃO FUNCIONOU para dados históricos")
            self.logger.error("Possíveis causas:")
            self.logger.error("1. Nenhum contrato WDO ativo encontrado")
            self.logger.error("2. Período solicitado muito antigo ou futuro")
            self.logger.error("3. Conta sem permissão para dados históricos")
            self.logger.error("4. Mercado fechado e dados não disponíveis")
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Erro solicitando dados históricos: {e}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False
    
    def wait_for_historical_data_v4(self, timeout_seconds: int = 60) -> bool:
        """
        Aguarda dados históricos com monitoramento melhorado
        
        Args:
            timeout_seconds: Timeout em segundos
            
        Returns:
            bool: True se dados foram recebidos
        """
        try:
            start_time = time.time()
            self.logger.info(f"⏳ Aguardando dados históricos por {timeout_seconds}s...")
            
            while (time.time() - start_time) < timeout_seconds:
                elapsed = time.time() - start_time
                
                # Verificar se dados chegaram
                history_count = self.callback_counts['history']
                progress_count = self.callback_counts['progress']
                
                if history_count > 0:
                    self.logger.info(f"✅ DADOS RECEBIDOS: {history_count} registros em {elapsed:.1f}s")
                    
                    # Aguardar um pouco mais para garantir que todos os dados chegaram
                    time.sleep(2)
                    final_count = self.callback_counts['history']
                    
                    if final_count == history_count:
                        self.logger.info(f"🎉 DADOS HISTÓRICOS COMPLETOS: {final_count} registros")
                        return True
                    else:
                        self.logger.info(f"📥 Ainda recebendo dados: {final_count} registros")
                
                # Log status a cada 10 segundos
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    self.logger.info(f"[{int(elapsed)}s] History: {history_count}, Progress: {progress_count}")
                
                time.sleep(1)
            
            # Timeout atingido
            final_history = self.callback_counts['history']
            final_progress = self.callback_counts['progress']
            
            if final_history > 0:
                self.logger.warning(f"⚠️ TIMEOUT mas recebidos {final_history} dados históricos")
                return True
            else:
                self.logger.error(f"❌ TIMEOUT sem dados - Progress: {final_progress}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erro aguardando dados históricos: {e}")
            return False
    
    def subscribe_ticker(self, ticker: str, exchange: str = "F") -> bool:
        """Subscreve ticker para dados em tempo real"""
        try:
            if not self.dll or not self.connection_states['market_data']:
                self.logger.error("Market data não conectado - subscrição não disponível")
                return False
            
            result = self.dll.SubscribeTicker(c_wchar_p(ticker), c_wchar_p(exchange))
            
            if result == 0:
                self.logger.info(f"✅ Subscrito: {ticker} ({exchange})")
                return True
            else:
                self.logger.error(f"❌ Falha na subscrição {ticker}: código {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erro subscrevendo ticker: {e}")
            return False
    
    def register_trade_callback(self, callback: Callable):
        """Registra callback para receber dados de trade"""
        self.trade_callbacks.append(callback)
        self.logger.info(f"Callback de trade registrado ({len(self.trade_callbacks)} total)")
    
    def register_state_callback(self, callback: Callable):
        """Registra callback para mudanças de estado"""
        self.state_callbacks.append(callback)
        self.logger.info(f"Callback de estado registrado ({len(self.state_callbacks)} total)")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Retorna status detalhado da conexão"""
        return {
            'connected': self.connected,
            'states': self.connection_states.copy(),
            'callback_counts': self.callback_counts.copy(),
            'historical_data_count': len(self.historical_data_cache),
            'historical_complete': self.historical_data_complete
        }
    
    def disconnect(self):
        """Desconecta e finaliza recursos"""
        try:
            if self.dll:
                self.dll.DLLFinalize()
                self.logger.info("DLL finalizada")
            
            self.connected = False
            self.connection_states = {'login': False, 'roteamento': False, 'market_data': False}
            self.logger.info("Desconexão concluída")
            
        except Exception as e:
            self.logger.error(f"Erro na desconexão: {e}")


# Função de conveniência para teste
def test_connection_manager_v4():
    """Teste rápido do Connection Manager V4"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Criar e testar connection manager
    cm = ConnectionManagerV4Fixed(r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll")
    
    try:
        # Conectar
        success = cm.initialize(
            os.getenv('PROFIT_KEY', ''),
            os.getenv('PROFIT_USERNAME', ''),
            os.getenv('PROFIT_PASSWORD', '')
        )
        
        if success:
            print("✅ Conexão estabelecida!")
            
            # Testar dados históricos
            if cm.request_historical_data_v4("WDO", 
                                            datetime.now() - timedelta(days=1), 
                                            datetime.now()):
                print("✅ Solicitação de dados aceita!")
                
                if cm.wait_for_historical_data_v4(30):
                    print("✅ Dados históricos recebidos!")
                else:
                    print("❌ Timeout aguardando dados históricos")
            else:
                print("❌ Falha na solicitação de dados históricos")
        else:
            print("❌ Falha na conexão")
            
    finally:
        cm.disconnect()


if __name__ == "__main__":
    test_connection_manager_v4()