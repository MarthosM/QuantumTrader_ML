"""
ConnectionManagerV3 - Gerenciador de conexão otimizado com coleta de dados real
Baseado no DEVELOPER_GUIDE_V3_REFACTORING.md
"""

import logging
from ctypes import *
from datetime import datetime
import os
from typing import Dict, Optional, Callable
import threading
import queue

# Imports internos
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from realtime.realtime_processor_v3 import RealTimeProcessorV3


class ConnectionManagerV3:
    """
    Gerenciador de conexão otimizado com ProfitDLL
    
    Features:
    - Callbacks otimizados para máxima coleta de dados
    - Integração com RealTimeProcessorV3
    - Thread-safe para múltiplos acessos
    - Monitoramento de qualidade de conexão
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o gerenciador de conexão
        
        Args:
            config: Configurações opcionais
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # DLL Path
        self.dll_path = self.config.get('dll_path', r"C:\Arquivos de Programas\Profit\ProfitDLL.dll")
        
        # Estado da conexão
        self.connected = False
        self.dll = None
        self.user = None
        self.password = None
        
        # Processador de tempo real
        self.realtime_processor = RealTimeProcessorV3(self.config.get('processor_config', {}))
        
        # Callbacks registrados
        self.callbacks = {
            'trade': None,
            'book': None,
            'connection': None,
            'status': None
        }
        
        # Métricas
        self.metrics = {
            'trades_received': 0,
            'books_received': 0,
            'connection_drops': 0,
            'last_trade_time': None,
            'last_book_time': None
        }
        
        # Lock para thread safety
        self._lock = threading.RLock()
        
    def initialize_dll(self, dll_path: Optional[str] = None) -> bool:
        """
        Inicializa a DLL do Profit
        
        Args:
            dll_path: Caminho opcional para a DLL
            
        Returns:
            bool: True se inicializou com sucesso
        """
        try:
            path = dll_path or self.dll_path
            
            if not os.path.exists(path):
                self.logger.error(f"DLL não encontrada: {path}")
                return False
            
            # Carregar DLL
            self.dll = CDLL(path)
            
            # Configurar assinaturas das funções principais
            self._configure_dll_functions()
            
            self.logger.info(f"DLL inicializada: {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro inicializando DLL: {e}")
            return False
            
    def _configure_dll_functions(self):
        """Configura assinaturas das funções da DLL"""
        
        if not self.dll:
            return
        
        # Funções de conexão
        self.dll.DLLInitializeLogin.argtypes = [c_wchar_p, c_wchar_p]
        self.dll.DLLInitializeLogin.restype = c_int
        
        self.dll.DLLConnect.argtypes = []
        self.dll.DLLConnect.restype = c_int
        
        self.dll.DLLDisconnect.argtypes = []
        self.dll.DLLDisconnect.restype = None
        
        # Funções de estado
        self.dll.DLLIsConnected.argtypes = []
        self.dll.DLLIsConnected.restype = c_int
        
        # Funções de ticker
        self.dll.DLLSubscribeTicker.argtypes = [c_wchar_p]
        self.dll.DLLSubscribeTicker.restype = None
        
        self.dll.DLLUnsubscribeTicker.argtypes = [c_wchar_p]
        self.dll.DLLUnsubscribeTicker.restype = None
        
    def connect(self, user: str, password: str) -> bool:
        """
        Conecta ao servidor Profit
        
        Args:
            user: Usuário
            password: Senha
            
        Returns:
            bool: True se conectou com sucesso
        """
        with self._lock:
            try:
                if not self.dll:
                    self.logger.error("DLL não inicializada")
                    return False
                
                # Inicializar login
                result = self.dll.DLLInitializeLogin(user, password)
                if result != 0:
                    self.logger.error(f"Falha no login: código {result}")
                    return False
                
                # Configurar callbacks antes de conectar
                self.setup_enhanced_callbacks()
                
                # Conectar
                result = self.dll.DLLConnect()
                if result == 0:
                    self.connected = True
                    self.user = user
                    self.password = password
                    
                    # Iniciar processador
                    self.realtime_processor.start()
                    
                    self.logger.info("Conectado ao Profit com sucesso")
                    return True
                else:
                    self.logger.error(f"Falha na conexão: código {result}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Erro conectando: {e}")
                return False
                
    def disconnect(self):
        """Desconecta do servidor"""
        with self._lock:
            if self.dll and self.connected:
                try:
                    # Parar processador
                    self.realtime_processor.stop()
                    
                    # Desconectar
                    self.dll.DLLDisconnect()
                    self.connected = False
                    
                    self.logger.info("Desconectado do Profit")
                except Exception as e:
                    self.logger.error(f"Erro desconectando: {e}")
                    
    def setup_enhanced_callbacks(self):
        """Configura callbacks otimizados para coleta máxima de dados"""
        
        if not self.dll:
            return
        
        # Callback de trades com detalhes completos
        @WINFUNCTYPE(None, c_wchar_p, c_wchar_p, c_double, c_double, 
                     c_int, c_wchar_p, c_wchar_p, c_int)
        def enhanced_trade_callback(date, time, price, volume, quantity,
                                   aggressor_side, trade_id, sequence):
            try:
                # Parse datetime
                trade_datetime = self._parse_datetime(date, time)
                
                # Criar dados do trade
                trade_data = {
                    'datetime': trade_datetime,
                    'price': float(price),
                    'volume': float(volume),
                    'quantity': int(quantity),
                    'side': aggressor_side,  # BUY/SELL real
                    'trade_id': trade_id,
                    'sequence': int(sequence)
                }
                
                # Enviar para processador
                self.realtime_processor.add_trade(trade_data)
                
                # Métricas
                with self._lock:
                    self.metrics['trades_received'] += 1
                    self.metrics['last_trade_time'] = trade_datetime
                    
            except Exception as e:
                self.logger.error(f"Erro no callback de trade: {e}")
        
        # Callback de book com múltiplos níveis
        @WINFUNCTYPE(None, c_wchar_p, c_wchar_p, c_int, c_double, 
                     c_int, c_double, c_int)
        def enhanced_book_callback(date, time, side, price, quantity, 
                                 total_quantity, level):
            try:
                # Parse datetime
                book_datetime = self._parse_datetime(date, time)
                
                # Criar dados do book
                book_data = {
                    'datetime': book_datetime,
                    'side': 'bid' if side == 0 else 'ask',
                    'level': int(level),
                    'price': float(price),
                    'quantity': int(quantity),
                    'total_quantity': int(total_quantity)
                }
                
                # Enviar para processador
                self.realtime_processor.add_book_update(book_data)
                
                # Métricas
                with self._lock:
                    self.metrics['books_received'] += 1
                    self.metrics['last_book_time'] = book_datetime
                    
            except Exception as e:
                self.logger.error(f"Erro no callback de book: {e}")
        
        # Callback de status de conexão
        @WINFUNCTYPE(None, c_int)
        def connection_status_callback(status):
            try:
                if status == 0:
                    self.logger.warning("Conexão perdida")
                    with self._lock:
                        self.connected = False
                        self.metrics['connection_drops'] += 1
                elif status == 1:
                    self.logger.info("Conexão restabelecida")
                    with self._lock:
                        self.connected = True
                        
            except Exception as e:
                self.logger.error(f"Erro no callback de status: {e}")
        
        # Registrar callbacks
        try:
            if hasattr(self.dll, 'DLLSetNewTradeCallback'):
                self.dll.DLLSetNewTradeCallback(enhanced_trade_callback)
                self.callbacks['trade'] = enhanced_trade_callback
                
            if hasattr(self.dll, 'DLLSetNewBookCallback'):
                self.dll.DLLSetNewBookCallback(enhanced_book_callback)
                self.callbacks['book'] = enhanced_book_callback
                
            if hasattr(self.dll, 'DLLSetConnectionStatusCallback'):
                self.dll.DLLSetConnectionStatusCallback(connection_status_callback)
                self.callbacks['connection'] = connection_status_callback
                
            self.logger.info("Callbacks enhanced configurados")
            
        except Exception as e:
            self.logger.error(f"Erro configurando callbacks: {e}")
            
    def subscribe_ticker(self, ticker: str) -> bool:
        """
        Inscreve em um ticker para receber dados
        
        Args:
            ticker: Símbolo do ativo
            
        Returns:
            bool: True se inscreveu com sucesso
        """
        try:
            if not self.dll or not self.connected:
                self.logger.error("Não conectado")
                return False
            
            self.dll.DLLSubscribeTicker(ticker)
            self.logger.info(f"Inscrito em {ticker}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro inscrevendo em {ticker}: {e}")
            return False
            
    def unsubscribe_ticker(self, ticker: str):
        """Cancela inscrição em um ticker"""
        try:
            if self.dll and self.connected:
                self.dll.DLLUnsubscribeTicker(ticker)
                self.logger.info(f"Desinscrito de {ticker}")
                
        except Exception as e:
            self.logger.error(f"Erro desinscrevendo de {ticker}: {e}")
            
    def _parse_datetime(self, date_str, time_str) -> datetime:
        """Parse de data e hora do Profit"""
        try:
            # Formato esperado: YYYYMMDD HHMMSS
            dt_str = f"{date_str} {time_str}"
            return datetime.strptime(dt_str, "%Y%m%d %H%M%S")
        except:
            return datetime.now()
            
    def get_metrics(self) -> Dict:
        """Retorna métricas de conexão"""
        with self._lock:
            metrics = self.metrics.copy()
            
            # Adicionar métricas do processador
            processor_metrics = self.realtime_processor.get_metrics()
            metrics['processor'] = processor_metrics
            
            return metrics
            
    def get_latest_features(self, n_rows: int = 1):
        """
        Retorna features mais recentes do processador
        
        Args:
            n_rows: Número de linhas
            
        Returns:
            DataFrame com features
        """
        return self.realtime_processor.get_latest_features(n_rows)
        
    def health_check(self) -> Dict[str, bool]:
        """Verifica saúde do sistema"""
        
        health = {
            'dll_loaded': self.dll is not None,
            'connected': self.connected,
            'callbacks_registered': all(self.callbacks.values()),
            'processor_healthy': self.realtime_processor.health_check()['overall'],
            'recent_data': (
                self.metrics['last_trade_time'] and
                (datetime.now() - self.metrics['last_trade_time']).seconds < 60
            )
        }
        
        health['overall'] = all(health.values())
        
        return health


def main():
    """Teste do ConnectionManagerV3"""
    
    print("="*60)
    print("TESTE DO CONNECTION MANAGER V3")
    print("="*60)
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Criar gerenciador
    manager = ConnectionManagerV3()
    
    # Inicializar DLL
    if not manager.initialize_dll():
        print("[ERRO] Falha ao inicializar DLL")
        return
    
    print("[OK] DLL inicializada")
    
    # Simular callbacks
    print("\nSimulando callbacks...")
    
    # Como não temos conexão real, vamos testar os componentes
    print("\nVerificando componentes:")
    
    # Health check
    health = manager.health_check()
    for component, status in health.items():
        status_str = "[OK]" if status else "[FAIL]"
        print(f"  {component}: {status_str}")
    
    # Métricas
    metrics = manager.get_metrics()
    print(f"\nMétricas:")
    print(f"  Trades recebidos: {metrics['trades_received']}")
    print(f"  Books recebidos: {metrics['books_received']}")
    
    print("\n[OK] Teste concluído!")


if __name__ == "__main__":
    main()