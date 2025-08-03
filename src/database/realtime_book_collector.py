"""
Sistema de coleta de book de ofertas em tempo real do ProfitDLL
Este módulo se conecta ao servidor isolado e coleta dados de book continuamente
"""

import os
import sys
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from multiprocessing.connection import Client
import threading
import queue
import logging
from typing import Dict, List, Optional, Tuple

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class RealtimeBookCollector:
    """
    Coletor de book de ofertas em tempo real via servidor isolado
    """
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.logger = self._setup_logger()
        
        # Configuração do servidor
        self.server_address = config.get('server_address', ('localhost', 6789))
        self.auth_key = b'profit_dll_secret'
        
        # Cliente IPC
        self.client = None
        self.connected = False
        
        # Armazenamento
        self.data_dir = Path(config.get('data_dir', 'data/realtime/book'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Buffers para dados
        self.offer_book_buffer = []
        self.price_book_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Controle
        self.is_running = False
        self.receiver_thread = None
        self.saver_thread = None
        
        # Estatísticas
        self.stats = {
            'offer_book_count': 0,
            'price_book_count': 0,
            'files_saved': 0,
            'start_time': None
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Configura logger"""
        logger = logging.getLogger('RealtimeBookCollector')
        logger.setLevel(logging.INFO)
        
        # Handler para arquivo
        fh = logging.FileHandler('logs/realtime_book_collector.log')
        fh.setLevel(logging.INFO)
        
        # Handler para console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formato
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def connect(self) -> bool:
        """Conecta ao servidor ProfitDLL"""
        try:
            self.logger.info(f"Conectando ao servidor em {self.server_address}...")
            
            self.client = Client(self.server_address, authkey=self.auth_key)
            self.connected = True
            
            # Aguardar mensagem de boas-vindas
            if self.client.poll(timeout=5):
                welcome = self.client.recv()
                self.logger.info(f"Servidor conectado: {welcome}")
                return True
            else:
                self.logger.error("Timeout aguardando resposta do servidor")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro ao conectar: {e}")
            return False
    
    def subscribe_book(self, ticker: str, book_type: str = 'both') -> bool:
        """
        Subscreve ao book de um ticker
        
        Args:
            ticker: Ticker do ativo
            book_type: 'offer', 'price' ou 'both'
        """
        try:
            if not self.connected:
                self.logger.error("Cliente não conectado")
                return False
            
            success = True
            
            # Subscrever offer book
            if book_type in ['offer', 'both']:
                self.client.send({
                    'type': 'subscribe_offer_book',
                    'ticker': ticker
                })
                
                # Aguardar resposta
                if self.client.poll(timeout=5):
                    response = self.client.recv()
                    if response.get('success'):
                        self.logger.info(f"✅ Subscrito ao offer book de {ticker}")
                    else:
                        self.logger.error(f"Falha ao subscrever offer book de {ticker}")
                        success = False
            
            # Subscrever price book
            if book_type in ['price', 'both']:
                self.client.send({
                    'type': 'subscribe_price_book',
                    'ticker': ticker
                })
                
                # Aguardar resposta
                if self.client.poll(timeout=5):
                    response = self.client.recv()
                    if response.get('success'):
                        self.logger.info(f"✅ Subscrito ao price book de {ticker}")
                    else:
                        self.logger.error(f"Falha ao subscrever price book de {ticker}")
                        success = False
            
            return success
            
        except Exception as e:
            self.logger.error(f"Erro ao subscrever book: {e}")
            return False
    
    def start_collection(self, duration_minutes: int = 0):
        """
        Inicia coleta de book em tempo real
        
        Args:
            duration_minutes: Duração da coleta em minutos (0 = infinito)
        """
        try:
            self.is_running = True
            self.stats['start_time'] = datetime.now()
            
            # Iniciar thread de recepção
            self.receiver_thread = threading.Thread(target=self._receive_loop)
            self.receiver_thread.daemon = True
            self.receiver_thread.start()
            
            # Iniciar thread de salvamento
            self.saver_thread = threading.Thread(target=self._save_loop)
            self.saver_thread.daemon = True
            self.saver_thread.start()
            
            self.logger.info("🚀 Coleta de book iniciada")
            
            # Se duração especificada, aguardar
            if duration_minutes > 0:
                end_time = datetime.now() + timedelta(minutes=duration_minutes)
                self.logger.info(f"Coletando por {duration_minutes} minutos...")
                
                while datetime.now() < end_time and self.is_running:
                    time.sleep(1)
                    
                    # Log periódico
                    if int(time.time()) % 30 == 0:
                        self._log_stats()
                
                self.stop_collection()
            
        except Exception as e:
            self.logger.error(f"Erro na coleta: {e}")
            self.stop_collection()
    
    def _receive_loop(self):
        """Loop para receber dados do servidor"""
        self.logger.info("Thread de recepção iniciada")
        
        while self.is_running and self.connected:
            try:
                # Verificar se há dados
                if self.client.poll(timeout=0.1):
                    message = self.client.recv()
                    
                    msg_type = message.get('type')
                    
                    if msg_type == 'offer_book':
                        # Dados de offer book
                        book_data = message.get('data', {})
                        with self.buffer_lock:
                            self.offer_book_buffer.append(book_data)
                            self.stats['offer_book_count'] += 1
                            
                    elif msg_type == 'price_book':
                        # Dados de price book
                        book_data = message.get('data', {})
                        with self.buffer_lock:
                            self.price_book_buffer.append(book_data)
                            self.stats['price_book_count'] += 1
                            
                    elif msg_type == 'heartbeat':
                        # Heartbeat do servidor
                        self.logger.debug(f"Heartbeat: {message.get('stats', {})}")
                        
                    elif msg_type == 'error':
                        self.logger.error(f"Erro do servidor: {message.get('message')}")
                        
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Erro no loop de recepção: {e}")
                    time.sleep(1)
    
    def _save_loop(self):
        """Loop para salvar dados periodicamente"""
        self.logger.info("Thread de salvamento iniciada")
        
        save_interval = 60  # Salvar a cada 60 segundos
        last_save = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Verificar se é hora de salvar
                if current_time - last_save >= save_interval:
                    self._save_buffers()
                    last_save = current_time
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Erro no loop de salvamento: {e}")
                time.sleep(5)
    
    def _save_buffers(self):
        """Salva buffers em arquivos Parquet"""
        try:
            timestamp = datetime.now()
            date_dir = self.data_dir / timestamp.strftime('%Y%m%d')
            date_dir.mkdir(exist_ok=True)
            
            # Copiar buffers (thread-safe)
            with self.buffer_lock:
                offer_data = self.offer_book_buffer.copy()
                price_data = self.price_book_buffer.copy()
                
                # Limpar buffers
                self.offer_book_buffer.clear()
                self.price_book_buffer.clear()
            
            # Salvar offer book
            if offer_data:
                df_offer = pd.DataFrame(offer_data)
                filename = f"offer_book_{timestamp.strftime('%H%M%S')}.parquet"
                filepath = date_dir / filename
                
                df_offer.to_parquet(filepath, compression='snappy')
                self.logger.info(f"💾 Salvo offer book: {len(offer_data)} registros em {filename}")
                self.stats['files_saved'] += 1
            
            # Salvar price book
            if price_data:
                df_price = pd.DataFrame(price_data)
                filename = f"price_book_{timestamp.strftime('%H%M%S')}.parquet"
                filepath = date_dir / filename
                
                df_price.to_parquet(filepath, compression='snappy')
                self.logger.info(f"💾 Salvo price book: {len(price_data)} registros em {filename}")
                self.stats['files_saved'] += 1
                
        except Exception as e:
            self.logger.error(f"Erro ao salvar buffers: {e}")
    
    def _log_stats(self):
        """Log estatísticas de coleta"""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds() if self.stats['start_time'] else 0
        
        self.logger.info(f"""
        📊 ESTATÍSTICAS DE COLETA
        ========================
        Tempo de execução: {int(uptime)}s
        Offer book: {self.stats['offer_book_count']} registros
        Price book: {self.stats['price_book_count']} registros
        Arquivos salvos: {self.stats['files_saved']}
        Buffer atual: {len(self.offer_book_buffer)} offer, {len(self.price_book_buffer)} price
        """)
    
    def stop_collection(self):
        """Para a coleta"""
        self.logger.info("Parando coleta...")
        
        self.is_running = False
        
        # Salvar buffers restantes
        self._save_buffers()
        
        # Aguardar threads
        if self.receiver_thread:
            self.receiver_thread.join(timeout=5)
        if self.saver_thread:
            self.saver_thread.join(timeout=5)
        
        # Desconectar
        if self.client:
            try:
                self.client.close()
            except:
                pass
        
        # Log final
        self._log_stats()
        self.logger.info("✅ Coleta finalizada")
    
    def get_latest_book_snapshot(self, ticker: str, book_type: str = 'offer') -> Optional[pd.DataFrame]:
        """
        Retorna o último snapshot do book
        
        Args:
            ticker: Ticker do ativo
            book_type: 'offer' ou 'price'
            
        Returns:
            DataFrame com dados do book ou None
        """
        try:
            # Buscar arquivo mais recente
            today_dir = self.data_dir / datetime.now().strftime('%Y%m%d')
            if not today_dir.exists():
                return None
            
            pattern = f"{book_type}_book_*.parquet"
            files = list(today_dir.glob(pattern))
            
            if not files:
                return None
            
            # Carregar arquivo mais recente
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            df = pd.read_parquet(latest_file)
            
            # Filtrar por ticker se especificado
            if 'ticker' in df.columns and ticker:
                df = df[df['ticker'] == ticker]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao obter snapshot: {e}")
            return None


if __name__ == "__main__":
    # Teste do coletor
    config = {
        'data_dir': 'data/realtime/book',
        'server_address': ('localhost', 6789)
    }
    
    collector = RealtimeBookCollector(config)
    
    # Conectar ao servidor
    if collector.connect():
        # Subscrever ao book do WDO
        ticker = "WDOU25"  # Contrato atual
        
        if collector.subscribe_book(ticker, 'both'):
            # Coletar por 5 minutos
            collector.start_collection(duration_minutes=5)
        else:
            print("Falha ao subscrever book")
    else:
        print("Falha ao conectar ao servidor")