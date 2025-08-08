"""
Sistema de Produção HMARL com Captura de Dados para Treinamento
Versão otimizada que captura book/tick sem impactar performance
"""

import os
import sys
import time
import threading
import logging
import json
from datetime import datetime
from pathlib import Path
import multiprocessing
from collections import deque
import queue
import pandas as pd
import numpy as np
from ctypes import WINFUNCTYPE, POINTER, c_double, c_int

# Adicionar src ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar base do sistema enhanced
from start_hmarl_production_enhanced import EnhancedHMARLProductionSystem

# Importar estruturas necessárias
from production_fixed import TAssetIDRec

logger = logging.getLogger('HMARL_With_Capture')

class DataCaptureBuffer:
    """Buffer otimizado para captura de dados sem impactar performance"""
    
    def __init__(self, max_size=50000, flush_interval=60):
        self.book_buffer = deque(maxlen=max_size)
        self.tick_buffer = deque(maxlen=max_size)
        self.candle_buffer = deque(maxlen=max_size)
        self.prediction_buffer = deque(maxlen=max_size)
        
        self.flush_interval = flush_interval
        self.last_flush = time.time()
        self.lock = threading.RLock()
        
        # Estatísticas
        self.stats = {
            'book_captured': 0,
            'tick_captured': 0,
            'candles_captured': 0,
            'predictions_captured': 0,
            'flushes': 0,
            'bytes_written': 0
        }
    
    def add_book(self, data):
        """Adiciona dados de book ao buffer"""
        with self.lock:
            self.book_buffer.append({
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                **data
            })
            self.stats['book_captured'] += 1
    
    def add_tick(self, data):
        """Adiciona tick ao buffer"""
        with self.lock:
            self.tick_buffer.append({
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                **data
            })
            self.stats['tick_captured'] += 1
    
    def add_candle(self, data):
        """Adiciona candle ao buffer"""
        with self.lock:
            self.candle_buffer.append({
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                **data
            })
            self.stats['candles_captured'] += 1
    
    def add_prediction(self, features, prediction, actual_result=None):
        """Adiciona predição e features para treinamento"""
        with self.lock:
            self.prediction_buffer.append({
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'features': features,
                'prediction': prediction,
                'actual': actual_result
            })
            self.stats['predictions_captured'] += 1
    
    def should_flush(self):
        """Verifica se é hora de fazer flush"""
        return time.time() - self.last_flush > self.flush_interval
    
    def get_buffers_copy(self):
        """Retorna cópia dos buffers para salvamento"""
        with self.lock:
            book_copy = list(self.book_buffer)
            tick_copy = list(self.tick_buffer)
            candle_copy = list(self.candle_buffer)
            prediction_copy = list(self.prediction_buffer)
            
            # Limpar buffers após copiar
            self.book_buffer.clear()
            self.tick_buffer.clear()
            self.candle_buffer.clear()
            self.prediction_buffer.clear()
            
            self.last_flush = time.time()
            self.stats['flushes'] += 1
            
            return {
                'book': book_copy,
                'tick': tick_copy,
                'candles': candle_copy,
                'predictions': prediction_copy
            }


class AsyncDataSaver:
    """Salvamento assíncrono de dados em thread separada"""
    
    def __init__(self, base_path='data/training'):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.save_queue = queue.Queue()
        self.is_running = True
        self.save_thread = None
        
        # Criar subdiretórios
        self.paths = {
            'book': self.base_path / 'book',
            'tick': self.base_path / 'tick',
            'candles': self.base_path / 'candles',
            'predictions': self.base_path / 'predictions'
        }
        
        for path in self.paths.values():
            path.mkdir(exist_ok=True)
        
        self.stats = {
            'saves': 0,
            'errors': 0,
            'total_bytes': 0
        }
    
    def start(self):
        """Inicia thread de salvamento"""
        self.save_thread = threading.Thread(
            target=self._save_worker,
            name="DataSaver",
            daemon=True
        )
        self.save_thread.start()
        logger.info("[DataSaver] Thread de salvamento iniciada")
    
    def stop(self):
        """Para thread de salvamento"""
        self.is_running = False
        if self.save_thread:
            self.save_thread.join(timeout=5)
    
    def save_async(self, data_dict):
        """Adiciona dados à fila de salvamento"""
        try:
            self.save_queue.put(data_dict, block=False)
        except queue.Full:
            logger.warning("[DataSaver] Fila cheia, descartando dados antigos")
            # Descartar item mais antigo e tentar novamente
            try:
                self.save_queue.get_nowait()
                self.save_queue.put(data_dict, block=False)
            except:
                pass
    
    def _save_worker(self):
        """Worker que processa fila de salvamento"""
        while self.is_running:
            try:
                # Pegar dados da fila com timeout
                data_dict = self.save_queue.get(timeout=1)
                
                # Salvar cada tipo de dado
                date_str = datetime.now().strftime('%Y%m%d')
                hour_str = datetime.now().strftime('%H')
                
                for data_type, data_list in data_dict.items():
                    if not data_list:
                        continue
                    
                    # Nome do arquivo com data e hora
                    filename = f"{data_type}_{date_str}_{hour_str}.jsonl"
                    filepath = self.paths[data_type] / filename
                    
                    # Salvar em formato JSONL (uma linha por registro)
                    try:
                        with open(filepath, 'a', encoding='utf-8') as f:
                            for record in data_list:
                                json_line = json.dumps(record, ensure_ascii=False)
                                f.write(json_line + '\n')
                                self.stats['total_bytes'] += len(json_line)
                        
                        self.stats['saves'] += 1
                        
                        # Log periódico
                        if self.stats['saves'] % 10 == 0:
                            logger.info(f"[DataSaver] {self.stats['saves']} saves, "
                                      f"{self.stats['total_bytes']/1024/1024:.2f} MB escritos")
                    
                    except Exception as e:
                        logger.error(f"[DataSaver] Erro salvando {data_type}: {e}")
                        self.stats['errors'] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[DataSaver] Erro no worker: {e}")
                self.stats['errors'] += 1


class HMARLProductionWithCapture(EnhancedHMARLProductionSystem):
    """
    Sistema HMARL com captura eficiente de dados para treinamento
    """
    
    def __init__(self):
        super().__init__()
        
        # Inicializar atributos faltantes
        self.pnl = 0.0
        self.position = 0
        
        # Sistema de captura
        self.capture_enabled = True
        self.capture_buffer = DataCaptureBuffer(
            max_size=50000,      # Buffer de 50k registros
            flush_interval=60    # Flush a cada 60 segundos
        )
        self.data_saver = AsyncDataSaver()
        
        # Thread de flush periódico
        self.flush_thread = None
        
        # Configurações de captura
        self.capture_config = {
            'capture_book': True,
            'capture_tick': True,
            'capture_candles': True,
            'capture_predictions': True,
            'capture_interval_ms': 100,  # Capturar book a cada 100ms
            'min_price_change': 0.5,     # Só capturar se preço mudou 0.5+
            'compress_book': True        # Comprimir dados do book
        }
        
        # Últimos valores para detectar mudanças
        self.last_captured = {
            'price': 0,
            'book_time': 0,
            'tick_time': 0
        }
        
        logger.info("[CAPTURE] Sistema de captura inicializado")
    
    def initialize(self):
        """Inicializa sistema base + captura"""
        # Inicializar sistema base
        if not super().initialize():
            return False
        
        # Iniciar sistema de captura
        if self.capture_enabled:
            self.data_saver.start()
            self._start_flush_thread()
            logger.info("[CAPTURE] Sistema de captura ativado")
        
        return True
    
    def _start_flush_thread(self):
        """Inicia thread de flush periódico"""
        self.flush_thread = threading.Thread(
            target=self._flush_worker,
            name="FlushWorker",
            daemon=True
        )
        self.flush_thread.start()
    
    def _flush_worker(self):
        """Worker que faz flush periódico dos buffers"""
        while self.is_running:
            try:
                time.sleep(10)  # Verificar a cada 10 segundos
                
                if self.capture_buffer.should_flush():
                    self._flush_buffers()
                    
            except Exception as e:
                logger.error(f"[FLUSH] Erro no flush worker: {e}")
    
    def _flush_buffers(self):
        """Faz flush dos buffers para salvamento"""
        try:
            # Pegar cópia dos buffers
            data_to_save = self.capture_buffer.get_buffers_copy()
            
            # Enviar para salvamento assíncrono
            if any(data_to_save.values()):
                self.data_saver.save_async(data_to_save)
                
                # Log de estatísticas
                stats = self.capture_buffer.stats
                logger.info(f"[CAPTURE] Flush: Book={stats['book_captured']} "
                          f"Tick={stats['tick_captured']} "
                          f"Candles={stats['candles_captured']} "
                          f"Predictions={stats['predictions_captured']}")
        
        except Exception as e:
            logger.error(f"[FLUSH] Erro ao fazer flush: {e}")
    
    # Override dos callbacks para capturar dados
    
    def _create_all_callbacks(self):
        """Cria callbacks com captura de dados integrada"""
        super()._create_all_callbacks()
        
        # Salvar referência ao callback original do tinyBook
        original_tiny_callback = self.callback_refs.get('tiny_book')
        
        # Criar novo callback que captura E processa
        @WINFUNCTYPE(None, POINTER(TAssetIDRec), c_double, c_int, c_int)
        def tinyBookWithCapture(assetId, price, qtd, side):
            # Chamar callback original
            if original_tiny_callback:
                original_tiny_callback(assetId, price, qtd, side)
            
            # Capturar se configurado
            if self.capture_enabled and self.capture_config['capture_tick']:
                current_time = time.time()
                
                # Capturar se mudou preço ou passou tempo suficiente
                price_changed = abs(price - self.last_captured['price']) >= self.capture_config['min_price_change']
                time_passed = (current_time - self.last_captured['tick_time']) * 1000 >= self.capture_config['capture_interval_ms']
                
                if price_changed or time_passed:
                    self.capture_buffer.add_tick({
                        'price': float(price),
                        'qty': int(qtd),
                        'side': 'BID' if side == 0 else 'ASK',
                        'ticker': self.target_ticker
                    })
                    
                    self.last_captured['price'] = price
                    self.last_captured['tick_time'] = current_time
            
            return None
        
        # Substituir callback
        self.callback_refs['tiny_book'] = tinyBookWithCapture
    
    def _process_daily_callback(self, candle_data):
        """Processa callback daily e captura dados"""
        # Processar normalmente
        super()._process_daily_callback(candle_data)
        
        # Capturar para treinamento
        if self.capture_enabled and self.capture_config['capture_candles']:
            self.capture_buffer.add_candle({
                'ticker': self.target_ticker,
                'open': candle_data['open'],
                'high': candle_data['high'],
                'low': candle_data['low'],
                'close': candle_data['close'],
                'volume': candle_data['volume'],
                'trades': candle_data.get('trades', 0)
            })
    
    def _make_prediction(self):
        """Faz predição e captura para treinamento"""
        # Calcular features ANTES da predição
        features = self._calculate_features()
        
        # Fazer predição normal
        prediction = super()._make_prediction()
        
        # Capturar para treinamento se configurado
        if (self.capture_enabled and 
            self.capture_config['capture_predictions'] and 
            features and prediction):
            
            self.capture_buffer.add_prediction(
                features=features,
                prediction={
                    'direction': prediction.get('direction'),
                    'confidence': prediction.get('confidence'),
                    'enhanced': prediction.get('hmarl_enhanced', False)
                }
            )
        
        return prediction
    
    def _capture_book_snapshot(self):
        """Captura snapshot do book (chamado periodicamente)"""
        if not self.capture_enabled or not self.capture_config['capture_book']:
            return
        
        current_time = time.time()
        if (current_time - self.last_captured['book_time']) * 1000 < self.capture_config['capture_interval_ms']:
            return
        
        # Criar snapshot do book se tivermos dados
        if hasattr(self, 'last_book_data') and self.last_book_data:
            book_snapshot = {
                'ticker': self.target_ticker,
                'price': self.current_price,
                'spread': self.last_book_data.get('spread', 0),
                'bid_volume': self.last_book_data.get('bid_volume', 0),
                'ask_volume': self.last_book_data.get('ask_volume', 0),
                'imbalance': self.last_book_data.get('imbalance', 0)
            }
            
            # Comprimir se configurado
            if self.capture_config['compress_book']:
                # Só adicionar níveis top do book
                book_snapshot['bid_1'] = self.last_book_data.get('bid_1', {})
                book_snapshot['ask_1'] = self.last_book_data.get('ask_1', {})
            else:
                # Adicionar book completo
                book_snapshot['full_book'] = self.last_book_data
            
            self.capture_buffer.add_book(book_snapshot)
            self.last_captured['book_time'] = current_time
    
    def _log_status(self):
        """Log status incluindo estatísticas de captura"""
        # Sincronizar pnl com daily_pnl
        self.pnl = getattr(self, 'daily_pnl', 0.0)
        
        super()._log_status()
        
        if self.capture_enabled:
            # Capturar snapshot do book periodicamente
            self._capture_book_snapshot()
            
            # Log de captura
            stats = self.capture_buffer.stats
            capture_msg = (f"[CAPTURE] Book: {stats['book_captured']} | "
                         f"Tick: {stats['tick_captured']} | "
                         f"Candles: {stats['candles_captured']} | "
                         f"Predictions: {stats['predictions_captured']}")
            
            if stats['book_captured'] % 100 == 0:  # Log a cada 100 capturas
                self.logger.info(capture_msg)
                self._add_log(capture_msg)
    
    def cleanup(self):
        """Cleanup incluindo flush final e salvamento"""
        # Sinalizar parada primeiro
        self.is_running = False
        
        logger.info("[CAPTURE] Finalizando sistema...")
        
        # Flush final dos buffers (rapido)
        if self.capture_enabled:
            try:
                # Flush com timeout
                self._flush_buffers()
                
                # Parar sistema de salvamento
                self.data_saver.is_running = False
                
                # Estatísticas finais (sem esperar)
                logger.info(f"[CAPTURE] Estatísticas finais:")
                logger.info(f"  Book capturados: {self.capture_buffer.stats['book_captured']}")
                logger.info(f"  Ticks capturados: {self.capture_buffer.stats['tick_captured']}")
                logger.info(f"  Candles capturados: {self.capture_buffer.stats['candles_captured']}")
                logger.info(f"  Predições capturadas: {self.capture_buffer.stats['predictions_captured']}")
                logger.info(f"  Total escrito: {self.data_saver.stats['total_bytes']/1024/1024:.2f} MB")
            except Exception as e:
                logger.error(f"Erro no flush final: {e}")
        
        # Cleanup do sistema base (ja tem timeout curto para agentes)
        super().cleanup()


def main():
    print("\n" + "="*60)
    print("QUANTUM TRADER ML - HMARL + CAPTURA DE DADOS")
    print("="*60)
    print(f"Data: {datetime.now()}")
    print("Funcionalidades:")
    print("  ✓ Trading com ML + HMARL")
    print("  ✓ Captura de book/tick para treinamento")
    print("  ✓ Salvamento assíncrono otimizado")
    print("  ✓ Buffer em memória com flush periódico")
    print("="*60)
    
    system = None
    
    # Handler para Ctrl+C
    import signal
    def signal_handler(sig, frame):
        print("\n\n[CTRL+C] Encerrando sistema rapidamente...")
        if system:
            system.is_running = False
        # Nao chamar cleanup aqui - deixar o finally fazer isso
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Criar sistema com captura
        system = HMARLProductionWithCapture()
        
        # Configurar captura (opcional)
        capture_config = input("\nAtivar captura de dados? (S/n): ").strip().lower()
        if capture_config == 'n':
            system.capture_enabled = False
            print("[INFO] Captura desativada - rodando apenas trading")
        else:
            print("[INFO] Captura ativada - dados serão salvos em data/training/")
            
            # Configurações adicionais
            print("\nConfigurações de captura:")
            print(f"  Buffer: 50.000 registros")
            print(f"  Flush: a cada 60 segundos")
            print(f"  Formato: JSONL comprimido")
            print(f"  Destino: data/training/")
        
        print("\nInicializando sistema...")
        
        # Inicializar
        if not system.initialize():
            print("\nERRO: Falha na inicialização")
            return 1
        
        # Inicializar HMARL
        if system.initialize_hmarl():
            print("[OK] HMARL inicializado com agentes reais")
        
        # Inicializar Monitor
        if system.start_monitor():
            print("[OK] Enhanced Monitor iniciado")
        
        # Aguardar estabilização
        print("\nConectando ao mercado...")
        time.sleep(3)
        
        # Subscrever ticker
        ticker = os.getenv('TICKER', 'WDOU25')
        if not system.subscribe_ticker(ticker):
            print(f"\nERRO: Falha ao subscrever {ticker}")
            return 1
        
        print(f"\nRecebendo dados de {ticker}...")
        time.sleep(5)
        
        # Mostrar callbacks
        print(f"\nCallbacks ativos:")
        for cb_type, count in system.callbacks.items():
            if count > 0:
                print(f"  {cb_type}: {count:,}")
        
        # Iniciar trading
        if not system.start():
            return 1
        
        print("\n" + "="*60)
        print("SISTEMA OPERACIONAL")
        print(f"Trading: {'Ativo' if system.is_running else 'Inativo'}")
        print(f"HMARL: {'Ativo' if system.hmarl_enabled else 'Inativo'}")
        print(f"Captura: {'Ativa' if system.capture_enabled else 'Inativa'}")
        print(f"Ticker: {ticker}")
        print("\nComandos:")
        print("  CTRL+C - Parar sistema")
        print("  Dados salvos automaticamente em data/training/")
        print("="*60)
        
        # Loop principal
        last_stats_time = time.time()
        while system.is_running:
            time.sleep(1)
            
            # Mostrar estatísticas a cada 30 segundos
            if time.time() - last_stats_time > 30:
                if system.capture_enabled:
                    stats = system.capture_buffer.stats
                    print(f"\n[STATS] Captura - Book: {stats['book_captured']} | "
                          f"Tick: {stats['tick_captured']} | "
                          f"Candles: {stats['candles_captured']} | "
                          f"Predictions: {stats['predictions_captured']}")
                last_stats_time = time.time()
        
    except KeyboardInterrupt:
        print("\n\nEncerrando sistema...")
        print("Salvando dados capturados...")
        
    except Exception as e:
        print(f"\nERRO FATAL: {e}")
        logger.error(f"Erro fatal: {e}", exc_info=True)
        
    finally:
        if 'system' in locals() and system:
            print("\nFinalizando sistema...")
            
            # Stop rapido
            system.is_running = False
            system.stop()
            
            # Cleanup com timeout
            try:
                system.cleanup()
            except Exception as e:
                print(f"Erro no cleanup: {e}")
            
            # Estatísticas finais (apenas se nao foi Ctrl+C)
            if system.stats.get('predictions', 0) > 0:
                print("\n" + "="*60)
                print("RESUMO DA SESSÃO")
                print("="*60)
            
            # Trading stats
            print(f"Predições ML: {system.stats['predictions']}")
            print(f"Trades: {system.stats['trades']}")
            print(f"PnL: R$ {system.pnl:.2f}")
            
            # HMARL stats
            if system.hmarl_enabled:
                print(f"\nHMARL:")
                print(f"  Sinais processados: {system.hmarl_stats['real_agent_signals']}")
                print(f"  Predições enhanced: {system.hmarl_stats['enhanced_predictions']}")
            
            # Capture stats
            if system.capture_enabled:
                print(f"\nDados Capturados:")
                stats = system.capture_buffer.stats
                print(f"  Book snapshots: {stats['book_captured']:,}")
                print(f"  Ticks: {stats['tick_captured']:,}")
                print(f"  Candles: {stats['candles_captured']:,}")
                print(f"  Predições: {stats['predictions_captured']:,}")
                print(f"  Dados salvos: {system.data_saver.stats['total_bytes']/1024/1024:.2f} MB")
                print(f"  Arquivos: {system.data_saver.stats['saves']}")
                print(f"\nDados salvos em: data/training/")
            
            print("="*60)

if __name__ == "__main__":
    # Configurar encoding para Windows
    if sys.platform == 'win32':
        os.system('chcp 65001 >nul 2>&1')
    
    sys.exit(main())