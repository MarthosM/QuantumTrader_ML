"""
Wrapper para executar ProfitDLL em processo totalmente isolado
Com proteção contra crashes e comunicação via arquivos/pipes
"""

import os
import sys
import time
import json
import signal
import logging
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ProfitDLLWrapper')


class ProfitDLLWrapper:
    """
    Wrapper que executa ProfitDLL em processo isolado
    Usa arquivos temporários para comunicação para evitar problemas de IPC
    """
    
    def __init__(self, work_dir: Optional[str] = None):
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="profit_dll_")
        self.command_file = Path(self.work_dir) / "command.json"
        self.response_file = Path(self.work_dir) / "response.json"
        self.status_file = Path(self.work_dir) / "status.json"
        self.data_dir = Path(self.work_dir) / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.is_running = False
        self.connection_manager = None
        
        logger.info(f"ProfitDLL Wrapper inicializado")
        logger.info(f"Diretório de trabalho: {self.work_dir}")
        
    def run(self):
        """Executa o loop principal do wrapper"""
        # Ignorar sinais que podem derrubar o processo
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        
        self.is_running = True
        
        # Atualizar status
        self._update_status({
            'state': 'running',
            'pid': os.getpid(),
            'started': datetime.now().isoformat()
        })
        
        logger.info("Wrapper iniciado, aguardando comandos...")
        
        try:
            while self.is_running:
                # Verificar se há novo comando
                if self.command_file.exists():
                    try:
                        # Ler comando
                        with open(self.command_file, 'r') as f:
                            command = json.load(f)
                        
                        # Remover arquivo de comando
                        self.command_file.unlink()
                        
                        logger.info(f"Comando recebido: {command.get('type')}")
                        
                        # Processar comando
                        response = self._process_command(command)
                        
                        # Salvar resposta
                        with open(self.response_file, 'w') as f:
                            json.dump(response, f)
                            
                    except Exception as e:
                        logger.error(f"Erro processando comando: {e}")
                        self._save_error_response(str(e))
                
                # Pequena pausa para não consumir CPU
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Erro fatal no wrapper: {e}")
            self._update_status({
                'state': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        
        finally:
            self._cleanup()
    
    def _process_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Processa um comando recebido"""
        cmd_type = command.get('type')
        
        if cmd_type == 'connect':
            return self._handle_connect(command)
        
        elif cmd_type == 'disconnect':
            return self._handle_disconnect()
        
        elif cmd_type == 'collect_historical':
            return self._handle_collect_historical(command)
        
        elif cmd_type == 'status':
            return self._handle_status()
        
        elif cmd_type == 'shutdown':
            self.is_running = False
            return {'success': True, 'message': 'Shutdown iniciado'}
        
        else:
            return {'success': False, 'error': f'Comando desconhecido: {cmd_type}'}
    
    def _handle_connect(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Conecta ao ProfitDLL"""
        try:
            from src.connection_manager_v4 import ConnectionManagerV4
            
            dll_path = command.get('dll_path')
            username = command.get('username')
            password = command.get('password')
            key = command.get('key')
            
            logger.info("Criando ConnectionManager...")
            self.connection_manager = ConnectionManagerV4(dll_path)
            
            logger.info("Conectando ao ProfitDLL...")
            success = self.connection_manager.initialize(
                key=key,
                username=username,
                password=password
            )
            
            if success:
                logger.info("✅ Conectado ao ProfitDLL!")
                self._update_status({
                    'state': 'connected',
                    'timestamp': datetime.now().isoformat()
                })
                return {'success': True, 'message': 'Conectado com sucesso'}
            else:
                return {'success': False, 'error': 'Falha na conexão'}
                
        except Exception as e:
            logger.error(f"Erro conectando: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_disconnect(self) -> Dict[str, Any]:
        """Desconecta do ProfitDLL"""
        try:
            if self.connection_manager:
                self.connection_manager.disconnect()
                self.connection_manager = None
                
            self._update_status({
                'state': 'disconnected',
                'timestamp': datetime.now().isoformat()
            })
            
            return {'success': True, 'message': 'Desconectado'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _handle_collect_historical(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Coleta dados históricos"""
        try:
            if not self.connection_manager:
                return {'success': False, 'error': 'Não conectado'}
            
            symbol = command.get('symbol')
            start_date = command.get('start_date')
            end_date = command.get('end_date')
            
            # Configurar callback para salvar dados
            data_file = self.data_dir / f"{symbol}_{start_date.replace('/', '-')}.json"
            trades_received = []
            
            def on_history_trade(data):
                trades_received.append(data)
                # Salvar incrementalmente
                if len(trades_received) % 100 == 0:
                    with open(data_file, 'w') as f:
                        json.dump(trades_received, f)
            
            self.connection_manager.register_history_trade_callback(on_history_trade)
            
            # Solicitar dados
            logger.info(f"Solicitando histórico: {symbol} de {start_date} até {end_date}")
            success = self.connection_manager.get_history_trades(
                ticker=symbol,
                exchange='F',
                date_start=start_date,
                date_end=end_date
            )
            
            if success:
                # Aguardar dados
                time.sleep(10)
                
                # Salvar dados finais
                with open(data_file, 'w') as f:
                    json.dump(trades_received, f)
                
                return {
                    'success': True,
                    'data_file': str(data_file),
                    'count': len(trades_received)
                }
            else:
                return {'success': False, 'error': 'Falha ao solicitar dados'}
                
        except Exception as e:
            logger.error(f"Erro coletando histórico: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_status(self) -> Dict[str, Any]:
        """Retorna status do wrapper"""
        try:
            status = {
                'success': True,
                'running': self.is_running,
                'connected': self.connection_manager is not None,
                'work_dir': self.work_dir,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    status.update(json.load(f))
            
            return status
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _update_status(self, status: Dict[str, Any]):
        """Atualiza arquivo de status"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status, f)
        except:
            pass
    
    def _save_error_response(self, error: str):
        """Salva resposta de erro"""
        try:
            with open(self.response_file, 'w') as f:
                json.dump({'success': False, 'error': error}, f)
        except:
            pass
    
    def _cleanup(self):
        """Limpa recursos"""
        logger.info("Limpando recursos...")
        
        if self.connection_manager:
            try:
                self.connection_manager.disconnect()
            except:
                pass
        
        self._update_status({
            'state': 'stopped',
            'timestamp': datetime.now().isoformat()
        })


def main():
    """Função principal para executar o wrapper"""
    # Pegar diretório de trabalho dos argumentos ou criar novo
    work_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Criar e executar wrapper
    wrapper = ProfitDLLWrapper(work_dir)
    
    try:
        wrapper.run()
    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()