"""
Buffer Thread-Safe para integração ProfitDLL + HMARL
Resolve problema de Segmentation Fault isolando threads
"""

import queue
import threading
import time
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import traceback


@dataclass
class BufferedData:
    """Estrutura para dados no buffer"""
    timestamp: datetime
    data_type: str  # 'trade', 'order', 'book', etc
    data: Dict[str, Any]
    priority: int = 0  # 0 = normal, 1 = high
    
    def __lt__(self, other):
        """Permite comparação para PriorityQueue"""
        if not isinstance(other, BufferedData):
            return NotImplemented
        # Prioridade maior primeiro, depois timestamp mais antigo
        return (self.priority, self.timestamp) > (other.priority, other.timestamp)


class ThreadSafeBuffer:
    """
    Buffer thread-safe para isolar ProfitDLL de HMARL
    Previne Segmentation Fault ao separar contextos de execução
    """
    
    def __init__(self, max_size: int = 10000):
        self.logger = logging.getLogger('ThreadSafeBuffer')
        
        # Queue thread-safe simples (sem prioridade por enquanto)
        self.buffer = queue.Queue(maxsize=max_size)
        
        # Controle de threads
        self.is_running = False
        self.consumer_thread = None
        self.producer_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        
        # Callbacks para processar dados
        self.processors = {}
        
        # Estatísticas
        self.stats = {
            'produced': 0,
            'consumed': 0,
            'dropped': 0,
            'errors': 0,
            'last_error': None
        }
        
        # Event para sincronização
        self.stop_event = threading.Event()
        
        self.logger.info("ThreadSafeBuffer inicializado")
    
    def register_processor(self, data_type: str, processor: Callable):
        """Registra função para processar tipo específico de dado"""
        self.processors[data_type] = processor
        self.logger.info(f"Processor registrado para {data_type}")
    
    def put(self, data: BufferedData, timeout: float = 0.1) -> bool:
        """
        Adiciona dados ao buffer (thread-safe)
        Usado pelo ProfitDLL callbacks
        """
        try:
            # Usar lock para garantir atomicidade
            with self.producer_lock:
                # Adicionar diretamente o objeto BufferedData
                self.buffer.put(data, block=True, timeout=timeout)
                self.stats['produced'] += 1
                return True
                
        except queue.Full:
            self.stats['dropped'] += 1
            self.logger.warning(f"Buffer cheio - descartando {data.data_type}")
            return False
            
        except Exception as e:
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            self.logger.error(f"Erro ao adicionar ao buffer: {e}")
            return False
    
    def _consumer_loop(self):
        """
        Loop consumidor executado em thread separada
        Processa dados do buffer de forma isolada
        """
        self.logger.info("Consumer thread iniciada")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Timeout permite verificar stop_event periodicamente
                data = self.buffer.get(timeout=0.1)
                
                # Processar com lock para evitar condições de corrida
                with self.consumer_lock:
                    self._process_data(data)
                    self.stats['consumed'] += 1
                    
            except queue.Empty:
                # Normal - sem dados para processar
                continue
                
            except Exception as e:
                self.stats['errors'] += 1
                self.logger.error(f"Erro no consumer loop: {e}")
                self.logger.error(traceback.format_exc())
                time.sleep(0.1)  # Evitar loop infinito em caso de erro
    
    def _process_data(self, data: BufferedData):
        """Processa dados usando callback apropriado"""
        try:
            processor = self.processors.get(data.data_type)
            
            if processor:
                # Executar em contexto isolado
                processor(data.data)
            else:
                self.logger.warning(f"Sem processor para {data.data_type}")
                
        except Exception as e:
            self.logger.error(f"Erro processando {data.data_type}: {e}")
            self.stats['errors'] += 1
    
    def start(self):
        """Inicia thread consumidora"""
        if self.is_running:
            self.logger.warning("Buffer já está rodando")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # Criar e iniciar thread consumidora
        self.consumer_thread = threading.Thread(
            target=self._consumer_loop,
            name="BufferConsumer",
            daemon=True
        )
        self.consumer_thread.start()
        
        self.logger.info("ThreadSafeBuffer iniciado")
    
    def stop(self, timeout: float = 5.0):
        """Para thread consumidora de forma segura"""
        if not self.is_running:
            return
        
        self.logger.info("Parando ThreadSafeBuffer...")
        
        self.is_running = False
        self.stop_event.set()
        
        # Aguardar thread terminar
        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout)
            
            if self.consumer_thread.is_alive():
                self.logger.warning("Consumer thread não terminou no tempo esperado")
        
        self.logger.info("ThreadSafeBuffer parado")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do buffer"""
        return {
            **self.stats,
            'buffer_size': self.buffer.qsize(),
            'is_running': self.is_running,
            'efficiency': self.stats['consumed'] / max(self.stats['produced'], 1)
        }
    
    def clear(self):
        """Limpa o buffer"""
        with self.producer_lock:
            while not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                except queue.Empty:
                    break
        
        self.logger.info("Buffer limpo")


class IsolatedProcessor:
    """
    Processador isolado para HMARL
    Executa em contexto separado do ProfitDLL
    """
    
    def __init__(self, hmarl_integration):
        self.hmarl = hmarl_integration
        self.logger = logging.getLogger('IsolatedProcessor')
        
        # Lock para sincronização interna
        self.process_lock = threading.Lock()
        
    def process_trade(self, trade_data: Dict):
        """Processa trade de forma isolada"""
        try:
            with self.process_lock:
                # Processar em contexto isolado
                enriched = self.hmarl._process_trade_data(trade_data)
                
                if enriched:
                    # Publicar via ZMQ (thread-safe)
                    self.hmarl._publish_market_data(enriched)
                    
                    # Armazenar no Valkey (thread-safe)
                    self.hmarl._store_in_valkey(enriched)
                    
        except Exception as e:
            self.logger.error(f"Erro processando trade isolado: {e}")
            self.hmarl.metrics['errors'] += 1
    
    def process_order(self, order_data: Dict):
        """Processa ordem de forma isolada"""
        try:
            with self.process_lock:
                # Processar ordem
                self.logger.debug(f"Ordem processada: {order_data.get('order_id')}")
                
        except Exception as e:
            self.logger.error(f"Erro processando ordem: {e}")
    
    def process_book(self, book_data: Dict):
        """Processa book de forma isolada"""
        try:
            with self.process_lock:
                # Processar book
                if hasattr(self.hmarl, 'liquidity_monitor'):
                    self.hmarl.infrastructure.liquidity_monitor.update_book(book_data)
                    
        except Exception as e:
            self.logger.error(f"Erro processando book: {e}")


# Exemplo de uso
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Criar buffer
    buffer = ThreadSafeBuffer(max_size=1000)
    
    # Registrar processadores
    def mock_processor(data):
        print(f"Processando: {data}")
    
    buffer.register_processor('trade', mock_processor)
    
    # Iniciar
    buffer.start()
    
    # Simular dados
    for i in range(5):
        data = BufferedData(
            timestamp=datetime.now(),
            data_type='trade',
            data={'id': i, 'price': 5000 + i}
        )
        buffer.put(data)
        time.sleep(0.1)
    
    # Aguardar processamento
    time.sleep(1)
    
    # Stats
    print(f"Stats: {buffer.get_stats()}")
    
    # Parar
    buffer.stop()