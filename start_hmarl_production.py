"""
Sistema de Produção com HMARL Integrado
Combina production_fixed.py com sistema HMARL
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

# Adicionar src ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar production_fixed
from production_fixed import ProductionFixedSystem

# Importar enhanced_monitor
try:
    from enhanced_monitor import EnhancedMonitor
    MONITOR_AVAILABLE = True
except ImportError as e:
    MONITOR_AVAILABLE = False
    print(f"[AVISO] Enhanced Monitor não disponível: {e}")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('HMARL_Production')

try:
    import valkey
    import zmq
    HMARL_AVAILABLE = True
    logger.info("[OK] Infraestrutura HMARL disponível (Valkey + ZMQ)")
except ImportError as e:
    HMARL_AVAILABLE = False
    logger.warning(f"[AVISO] HMARL parcialmente disponível: {e}")

class HMARLProductionSystem(ProductionFixedSystem):
    """
    Extensão do production_fixed com capacidades HMARL
    """
    
    def __init__(self):
        super().__init__()
        self.hmarl_enabled = HMARL_AVAILABLE
        self.valkey_client = None
        self.zmq_context = None
        self.flow_data_buffer = []
        self._last_prediction = None
        self.hmarl_stats = {
            'flow_signals': 0,
            'agent_consensus': 0,
            'enhanced_predictions': 0
        }
        self.monitor_process = None
        self.monitor_data_file = Path('data/monitor_data.json')
        self.recent_logs = []  # Para armazenar logs recentes
        
    def initialize_hmarl(self):
        """Inicializa componentes HMARL"""
        if not self.hmarl_enabled:
            self.logger.warning("HMARL não disponível - rodando sem agentes")
            return False
            
        try:
            # Conectar ao Valkey
            self.valkey_client = valkey.Valkey(
                host='localhost',
                port=6379,
                decode_responses=True
            )
            self.valkey_client.ping()
            self.logger.info("[OK] Conectado ao Valkey")
            
            # Configurar ZMQ
            self.zmq_context = zmq.Context()
            
            # Publisher para broadcast de dados
            self.flow_publisher = self.zmq_context.socket(zmq.PUB)
            self.flow_publisher.bind("tcp://*:5557")
            
            # Subscriber para sinais dos agentes
            self.agent_subscriber = self.zmq_context.socket(zmq.SUB)
            self.agent_subscriber.connect("tcp://localhost:5561")
            self.agent_subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
            
            self.logger.info("[OK] ZMQ configurado (pub:5557, sub:5561)")
            
            # Iniciar thread de processamento HMARL
            hmarl_thread = threading.Thread(target=self._process_hmarl_signals)
            hmarl_thread.daemon = True
            hmarl_thread.start()
            
            # Adicionar interceptor para broadcast de dados
            self._setup_data_interceptors()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar HMARL: {e}")
            self.hmarl_enabled = False
            return False
            
    def _process_hmarl_signals(self):
        """Processa sinais dos agentes HMARL"""
        poller = zmq.Poller()
        poller.register(self.agent_subscriber, zmq.POLLIN)
        
        while self.is_running:
            try:
                socks = dict(poller.poll(100))
                
                if self.agent_subscriber in socks:
                    message = self.agent_subscriber.recv_json(zmq.NOBLOCK)
                    self._handle_agent_signal(message)
                    
            except zmq.Again:
                continue
            except Exception as e:
                self.logger.error(f"Erro processando sinais HMARL: {e}")
                
    def _handle_agent_signal(self, signal):
        """Processa sinal de agente"""
        self.hmarl_stats['flow_signals'] += 1
        
        # Log apenas sinais importantes
        if signal.get('confidence', 0) > 0.7:
            self.logger.info(f"[HMARL] Sinal forte: {signal.get('type')} "
                           f"conf={signal.get('confidence'):.2f}")
            
    def _broadcast_to_hmarl(self, data_type, data):
        """Envia dados para agentes HMARL"""
        if not self.hmarl_enabled or not self.flow_publisher:
            return
            
        try:
            message = {
                'timestamp': datetime.now().isoformat(),
                'type': data_type,
                'data': data
            }
            self.flow_publisher.send_json(message, zmq.NOBLOCK)
            
            # Também salvar no Valkey para análise histórica
            if self.valkey_client:
                stream_key = f"flow:{self.target_ticker}"
                self.valkey_client.xadd(
                    stream_key,
                    {'type': data_type, 'value': str(data)},
                    maxlen=10000
                )
                
        except Exception as e:
            self.logger.debug(f"Erro ao broadcast HMARL: {e}")
            
    def _make_prediction(self):
        """Faz predição ML com enhancement HMARL"""
        base_prediction = super()._make_prediction()
        
        if not base_prediction:
            return None
            
        # Salvar predição para o monitor
        self._last_prediction = base_prediction.copy()
        
        # Log da predição
        pred_msg = f"ML Prediction: Dir={base_prediction.get('direction', 0):.3f} Conf={base_prediction.get('confidence', 0):.3f}"
        self._add_log(pred_msg)
        
        if not self.hmarl_enabled:
            return base_prediction
            
        try:
            # Buscar consenso dos agentes no Valkey
            consensus_key = f"consensus:{self.target_ticker}"
            agent_consensus = self.valkey_client.get(consensus_key)
            
            if agent_consensus:
                consensus_data = json.loads(agent_consensus)
                
                # Ajustar predição baseado no consenso
                if consensus_data.get('confidence', 0) > 0.7:
                    weight = 0.3  # 30% peso para HMARL
                    base_prediction['direction'] = (
                        base_prediction['direction'] * (1 - weight) +
                        consensus_data.get('direction', 0.5) * weight
                    )
                    base_prediction['hmarl_enhanced'] = True
                    self.hmarl_stats['enhanced_predictions'] += 1
                    self._add_log("[HMARL] Prediction enhanced by agents")
            
            # Salvar predição atualizada
            self._last_prediction = base_prediction.copy()
            
            # Também salvar no Valkey para análise
            if self.valkey_client:
                self.valkey_client.xadd(
                    f"predictions:{self.target_ticker}",
                    {
                        'direction': base_prediction.get('direction', 0),
                        'confidence': base_prediction.get('confidence', 0),
                        'enhanced': base_prediction.get('hmarl_enhanced', False)
                    },
                    maxlen=1000
                )
                    
            return base_prediction
            
        except Exception as e:
            self.logger.debug(f"Erro no enhancement HMARL: {e}")
            return base_prediction
            
    def _create_all_callbacks(self):
        """Cria callbacks e adiciona hooks HMARL"""
        super()._create_all_callbacks()
        
        # Hook será adicionado após inicialização HMARL
        # para evitar problemas com ctypes
        
    def _setup_data_interceptors(self):
        """Configura interceptors para broadcast HMARL"""
        # Salvamos referência ao callback original de tiny_book
        self._original_process_tiny = None
        
        # Interceptamos atualizações de preço para broadcast
        if hasattr(self, 'current_price'):
            self._last_broadcast_price = 0
        
    def _add_log(self, message):
        """Adiciona log ao buffer de logs recentes"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.recent_logs.append(log_entry)
        # Manter apenas últimos 50 logs
        if len(self.recent_logs) > 50:
            self.recent_logs = self.recent_logs[-50:]
    
    def _log_status(self):
        """Log de status incluindo HMARL"""
        super()._log_status()
        
        status_msg = f"Price: R${self.current_price:.2f} | Pos: {self.position} | PnL: R${self.pnl:.2f}"
        self._add_log(status_msg)
        
        if self.hmarl_enabled:
            hmarl_msg = f"[HMARL] Signals: {self.hmarl_stats['flow_signals']} | Enhanced: {self.hmarl_stats['enhanced_predictions']}"
            self.logger.info(hmarl_msg)
            self._add_log(hmarl_msg)
            
            # Salvar status dos agentes no Valkey
            if self.valkey_client:
                try:
                    # Simular status dos agentes para teste
                    agents_status = {
                        'order_flow': {'signals': 10, 'avg_confidence': 0.75},
                        'liquidity': {'signals': 8, 'avg_confidence': 0.68},
                        'tape_reading': {'signals': 12, 'avg_confidence': 0.82},
                        'footprint': {'signals': 6, 'avg_confidence': 0.71}
                    }
                    
                    for agent, status in agents_status.items():
                        self.valkey_client.set(
                            f"agent:{agent}:status",
                            json.dumps(status),
                            ex=60  # Expire em 60 segundos
                        )
                except:
                    pass
        
        # Atualizar dados para o monitor
        self._update_monitor_data()
            
    def _update_monitor_data(self):
        """Atualiza arquivo de dados para o monitor"""
        try:
            # Garantir que o diretório existe
            self.monitor_data_file.parent.mkdir(exist_ok=True)
            
            # Coletar dados do sistema
            monitor_data = {
                'timestamp': datetime.now().isoformat(),
                'status': 'Operacional' if self.is_running else 'Parado',
                'ticker': self.target_ticker,
                'price': self.current_price,
                'position': self.position,
                'entry_price': self.entry_price,
                'pnl': self.pnl,
                'daily_pnl': self.daily_pnl,
                'trades': self.stats['trades'],
                'wins': self.stats['wins'],
                'losses': self.stats['losses'],
                'callbacks': self.callbacks.copy(),
                'active_models': list(self.models.keys()) if hasattr(self, 'models') else [],
                'hmarl_stats': self.hmarl_stats.copy() if self.hmarl_enabled else {},
                'last_prediction': self._last_prediction if self._last_prediction else {}
            }
            
            # Adicionar logs recentes
            if hasattr(self, 'recent_logs'):
                monitor_data['recent_logs'] = self.recent_logs[-20:]
            
            # Salvar no arquivo
            with open(self.monitor_data_file, 'w') as f:
                json.dump(monitor_data, f, indent=2)
                
        except Exception as e:
            self.logger.debug(f"Erro ao atualizar monitor data: {e}")
    
    def start_monitor(self):
        """Inicia o Enhanced Monitor em processo separado"""
        if not MONITOR_AVAILABLE:
            self.logger.warning("Enhanced Monitor não disponível")
            return False
            
        try:
            def run_monitor():
                try:
                    monitor = EnhancedMonitor()
                    monitor.run()
                except Exception as e:
                    logger.error(f"Erro no monitor: {e}")
            
            # Iniciar monitor em processo separado
            self.monitor_process = multiprocessing.Process(target=run_monitor)
            self.monitor_process.daemon = True
            self.monitor_process.start()
            
            self.logger.info("[OK] Enhanced Monitor iniciado")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar monitor: {e}")
            return False
    
    def stop_monitor(self):
        """Para o Enhanced Monitor"""
        if self.monitor_process and self.monitor_process.is_alive():
            self.monitor_process.terminate()
            self.monitor_process.join(timeout=2)
            if self.monitor_process.is_alive():
                self.monitor_process.kill()
            self.logger.info("Enhanced Monitor finalizado")
    
    def cleanup(self):
        """Cleanup incluindo HMARL e Monitor"""
        super().cleanup()
        
        # Parar monitor
        self.stop_monitor()
        
        if self.zmq_context:
            self.zmq_context.term()
        if self.valkey_client:
            self.valkey_client.close()

def main():
    print("\n" + "="*60)
    print("QUANTUM TRADER ML - PRODUÇÃO COM HMARL E MONITOR")
    print("="*60)
    print(f"Data: {datetime.now()}")
    print(f"HMARL: {'[OK] Ativado' if HMARL_AVAILABLE else '[AVISO] Desativado'}")
    print(f"Monitor: {'[OK] Disponível' if MONITOR_AVAILABLE else '[AVISO] Não disponível'}")
    print("="*60)
    
    try:
        # Criar sistema
        system = HMARLProductionSystem()
        
        # Inicializar base
        if not system.initialize():
            print("\nERRO: Falha na inicialização base")
            return 1
            
        # Inicializar HMARL
        if HMARL_AVAILABLE:
            if system.initialize_hmarl():
                print("[OK] HMARL inicializado com sucesso")
            else:
                print("[AVISO] HMARL não inicializado - continuando sem agentes")
        
        # Inicializar Enhanced Monitor
        if MONITOR_AVAILABLE:
            if system.start_monitor():
                print("[OK] Enhanced Monitor iniciado")
                print("    → Interface gráfica disponível")
            else:
                print("[AVISO] Monitor não pôde ser iniciado")
        
        # Aguardar estabilização
        print("\nSistema conectado. Aguardando dados...")
        time.sleep(3)
        
        # Subscrever
        ticker = os.getenv('TICKER', 'WDOU25')
        if not system.subscribe_ticker(ticker):
            print(f"\nERRO: Falha ao subscrever {ticker}")
            return 1
            
        # Aguardar dados
        print(f"\nAguardando dados de {ticker}...")
        time.sleep(5)
        
        # Verificar recepção
        print(f"\nCallbacks recebidos:")
        for cb_type, count in system.callbacks.items():
            if count > 0:
                print(f"  {cb_type}: {count:,}")
                
        # Iniciar estratégia
        if not system.start():
            return 1
            
        print("\n" + "="*60)
        print(f"SISTEMA OPERACIONAL")
        print(f"Modelos ML: {len(system.models)}")
        print(f"HMARL: {'Ativo' if system.hmarl_enabled else 'Inativo'}")
        print(f"Monitor: {'Ativo' if system.monitor_process and system.monitor_process.is_alive() else 'Inativo'}")
        print(f"Ticker: {ticker}")
        print("Para parar: CTRL+C")
        print("="*60)
        
        # Loop principal
        while system.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário")
        
    except Exception as e:
        print(f"\nERRO FATAL: {e}")
        logger.error(f"Erro fatal: {e}", exc_info=True)
        
    finally:
        if 'system' in locals():
            system.stop()
            system.cleanup()
            
            # Stats finais
            print("\n" + "="*60)
            print("ESTATÍSTICAS FINAIS")
            print("="*60)
            print(f"Callbacks totais:")
            for cb_type, count in system.callbacks.items():
                if count > 0:
                    print(f"  {cb_type}: {count:,}")
            print(f"Predições ML: {system.stats['predictions']}")
            if system.hmarl_enabled:
                print(f"HMARL Signals: {system.hmarl_stats['flow_signals']}")
                print(f"Enhanced Predictions: {system.hmarl_stats['enhanced_predictions']}")
            print("="*60)

if __name__ == "__main__":
    sys.exit(main())