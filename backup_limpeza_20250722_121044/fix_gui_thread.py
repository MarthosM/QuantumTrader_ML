#!/usr/bin/env python3
"""
🔧 CORREÇÃO CRÍTICA GUI - ML TRADING v2.0
========================================
Corrige erro "main thread is not in main loop"

PROBLEMA IDENTIFICADO:
❌ GUI está sendo executado em thread secundária
❌ Tkinter DEVE rodar na thread principal
❌ Sistema está invertido: trading em main, GUI em thread

SOLUÇÃO:
✅ GUI na thread principal
✅ Trading system em thread background
✅ Comunicação thread-safe entre eles
"""

import os
import sys
import time
import threading
import queue
from datetime import datetime

# Configurar path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def load_config():
    """Carrega configuração do .env"""
    config = {}
    
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    config[key] = value
    
    return {
        'dll_path': config.get('PROFIT_DLL_PATH', ''),
        'username': config.get('PROFIT_USER', ''),
        'password': config.get('PROFIT_PASSWORD', ''),
        'account_id': config.get('PROFIT_ACCOUNT_ID', ''),
        'broker_id': config.get('PROFIT_BROKER_ID', ''),
        'trading_password': config.get('PROFIT_TRADING_PASSWORD', ''),
        'key': config.get('PROFIT_KEY', ''),
        'models_dir': config.get('MODELS_DIR', ''),
        'ml_interval': int(config.get('ML_INTERVAL', 15)),
        'use_gui': True,
        'historical_days': int(config.get('HISTORICAL_DAYS', 10)),
        'initial_balance': float(config.get('INITIAL_BALANCE', 100000))
    }

class TradingSystemThread(threading.Thread):
    """Thread para executar o sistema de trading"""
    
    def __init__(self, config, status_queue):
        super().__init__(daemon=True)
        self.config = config
        self.status_queue = status_queue
        self.system = None
        self.running = False
        
    def run(self):
        """Executa o sistema de trading em background"""
        try:
            # Importar módulos
            from trading_system import TradingSystem
            
            # Criar sistema
            self.system = TradingSystem(self.config)
            
            # Enviar status
            self.status_queue.put(('status', 'Inicializando sistema...'))
            
            # Inicializar
            if not self.system.initialize():
                self.status_queue.put(('error', 'Falha na inicialização'))
                return
                
            self.status_queue.put(('status', 'Sistema inicializado'))
            
            # Iniciar operação
            if self.system.start():
                self.status_queue.put(('status', 'Sistema operacional'))
                self.running = True
                
                # Loop de monitoramento
                while self.running:
                    # Enviar dados para GUI
                    if hasattr(self.system, 'get_status'):
                        status = self.system.get_status()
                        self.status_queue.put(('data', status))
                    
                    time.sleep(1)  # Atualizar a cada segundo
                    
            else:
                self.status_queue.put(('error', 'Falha ao iniciar sistema'))
                
        except Exception as e:
            self.status_queue.put(('error', f'Erro no sistema: {e}'))
            
    def stop(self):
        """Para o sistema"""
        self.running = False
        if self.system:
            self.system.stop()

class TradingGUIMain:
    """GUI principal que roda na thread main"""
    
    def __init__(self):
        self.config = load_config()
        self.status_queue = queue.Queue()
        self.trading_thread = None
        
        # Importar GUI
        from trading_monitor_gui import TradingMonitorGUI
        self.gui_class = TradingMonitorGUI
        
    def start(self):
        """Inicia sistema com GUI na thread principal"""
        print("🚀 INICIANDO ML TRADING v2.0 - GUI CORRIGIDO")
        print("="*50)
        print(f"🕐 Início: {datetime.now().strftime('%H:%M:%S')}")
        print("✅ GUI rodará na thread PRINCIPAL")
        print("✅ Trading rodará em thread BACKGROUND")
        print("")
        
        try:
            # 1. Criar sistema mock inicial para GUI
            mock_system = self._create_mock_system()
            
            # 2. Criar GUI na thread principal
            print("🖥️ Criando GUI na thread principal...")
            gui = self.gui_class(mock_system)
            
            # 3. Iniciar sistema de trading em thread background
            print("🔧 Iniciando sistema de trading em background...")
            self.trading_thread = TradingSystemThread(self.config, self.status_queue)
            self.trading_thread.start()
            
            # 4. Configurar atualizações do GUI
            def update_gui():
                """Atualiza GUI com dados do sistema"""
                try:
                    while not self.status_queue.empty():
                        msg_type, data = self.status_queue.get_nowait()
                        
                        if msg_type == 'status':
                            print(f"📊 Status: {data}")
                        elif msg_type == 'error':
                            print(f"❌ Erro: {data}")
                        elif msg_type == 'data':
                            # Atualizar GUI com dados reais
                            if hasattr(gui, 'update_with_data'):
                                gui.update_with_data(data)
                                
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"Erro atualizando GUI: {e}")
                
                # Reagendar atualização
                if hasattr(gui, 'root') and gui.root:
                    gui.root.after(1000, update_gui)  # A cada 1 segundo
            
            # 5. Configurar GUI para iniciar atualizações
            if hasattr(gui, 'root'):
                gui.root.after(1000, update_gui)
                
                # Configurar fechamento
                def on_closing():
                    print("🛑 Fechando sistema...")
                    if self.trading_thread:
                        self.trading_thread.stop()
                    gui.root.quit()
                    
                gui.root.protocol("WM_DELETE_WINDOW", on_closing)
            
            print("✅ GUI configurado na thread principal")
            print("📊 Monitor deve abrir agora...")
            print("")
            
            # 6. EXECUTAR GUI NA THREAD PRINCIPAL
            gui.run()
            
        except Exception as e:
            print(f"❌ Erro: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_mock_system(self):
        """Cria sistema mock para inicializar GUI"""
        class MockSystem:
            def __init__(self):
                self.is_running = False
                self.ticker = "WDOQ25"
                self.last_prediction = None
                self.metrics = MockMetrics()
                
            def get_status(self):
                return {
                    'running': self.is_running,
                    'ticker': self.ticker,
                    'last_prediction': self.last_prediction,
                    'metrics': self.metrics.get_summary() if self.metrics else {}
                }
        
        class MockMetrics:
            def get_summary(self):
                return {
                    'trades_processed': 0,
                    'predictions_made': 0,
                    'signals_generated': 0,
                    'signals_executed': 0,
                    'uptime': '00:00:00',
                    'current_price': 5584.0
                }
        
        return MockSystem()

def main():
    """Função principal"""
    print("🔧 CORREÇÃO CRÍTICA GUI - THREAD PRINCIPAL")
    print("="*45)
    
    try:
        gui_main = TradingGUIMain()
        gui_main.start()
        
    except KeyboardInterrupt:
        print("\n⏹️ Interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🕐 Finalizado: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
