"""
Cliente para testar o ProfitDLL Wrapper
Usa comunicação via arquivos para máxima segurança
"""

import os
import sys
import time
import json
import subprocess
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Carregar variáveis
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('WrapperClient')


class WrapperClient:
    """Cliente para comunicar com o ProfitDLL Wrapper"""
    
    def __init__(self):
        self.work_dir = tempfile.mkdtemp(prefix="profit_dll_")
        self.command_file = Path(self.work_dir) / "command.json"
        self.response_file = Path(self.work_dir) / "response.json"
        self.status_file = Path(self.work_dir) / "status.json"
        self.wrapper_process = None
        
        logger.info(f"Cliente inicializado")
        logger.info(f"Diretório de trabalho: {self.work_dir}")
    
    def start_wrapper(self):
        """Inicia o processo wrapper"""
        logger.info("🚀 Iniciando wrapper ProfitDLL...")
        
        # Comando para executar o wrapper
        cmd = [
            sys.executable,
            "scripts/profit_dll_wrapper.py",
            self.work_dir
        ]
        
        # Iniciar processo
        self.wrapper_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Wrapper iniciado com PID: {self.wrapper_process.pid}")
        
        # Aguardar inicialização
        for i in range(10):
            if self.status_file.exists():
                logger.info("✅ Wrapper pronto!")
                return True
            time.sleep(1)
        
        logger.error("❌ Wrapper não inicializou")
        return False
    
    def send_command(self, command: dict, timeout: int = 30) -> dict:
        """Envia comando e aguarda resposta"""
        # Limpar resposta anterior
        if self.response_file.exists():
            self.response_file.unlink()
        
        # Enviar comando
        with open(self.command_file, 'w') as f:
            json.dump(command, f)
        
        logger.info(f"📤 Comando enviado: {command.get('type')}")
        
        # Aguardar resposta
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.response_file.exists():
                try:
                    with open(self.response_file, 'r') as f:
                        response = json.load(f)
                    
                    # Remover arquivo de resposta
                    self.response_file.unlink()
                    
                    logger.info(f"📥 Resposta recebida: {response}")
                    return response
                    
                except Exception as e:
                    logger.error(f"Erro lendo resposta: {e}")
            
            # Verificar se wrapper ainda está vivo
            if self.wrapper_process and self.wrapper_process.poll() is not None:
                logger.error(f"❌ Wrapper morreu! Exit code: {self.wrapper_process.returncode}")
                return {'success': False, 'error': 'Wrapper crashed'}
            
            time.sleep(0.5)
        
        return {'success': False, 'error': 'Timeout aguardando resposta'}
    
    def get_status(self) -> dict:
        """Obtém status do wrapper"""
        return self.send_command({'type': 'status'}, timeout=5)
    
    def connect_dll(self) -> dict:
        """Conecta ao ProfitDLL"""
        return self.send_command({
            'type': 'connect',
            'dll_path': r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll",
            'username': os.getenv('PROFIT_USERNAME'),
            'password': os.getenv('PROFIT_PASSWORD'),
            'key': os.getenv('PROFIT_KEY')
        })
    
    def collect_historical(self, symbol: str, start_date: str, end_date: str) -> dict:
        """Coleta dados históricos"""
        return self.send_command({
            'type': 'collect_historical',
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date
        })
    
    def shutdown(self):
        """Finaliza o wrapper"""
        logger.info("🛑 Finalizando wrapper...")
        
        # Enviar comando de shutdown
        self.send_command({'type': 'shutdown'}, timeout=5)
        
        # Aguardar processo terminar
        if self.wrapper_process:
            self.wrapper_process.wait(timeout=10)
            
            if self.wrapper_process.poll() is None:
                logger.warning("⚠️ Forçando término...")
                self.wrapper_process.terminate()
                self.wrapper_process.wait()
        
        # Limpar diretório temporário
        import shutil
        try:
            shutil.rmtree(self.work_dir)
        except:
            pass


def main():
    """Teste principal"""
    logger.info("="*80)
    logger.info("🧪 TESTE DO PROFITDLL WRAPPER")
    logger.info("="*80)
    
    client = WrapperClient()
    
    try:
        # 1. Iniciar wrapper
        if not client.start_wrapper():
            logger.error("Falha ao iniciar wrapper")
            return
        
        # Aguardar estabilização
        time.sleep(2)
        
        # 2. Verificar status
        logger.info("\n📊 Verificando status inicial...")
        status = client.get_status()
        logger.info(f"Status: {status}")
        
        # 3. Conectar ao ProfitDLL
        logger.info("\n🔌 Conectando ao ProfitDLL...")
        result = client.connect_dll()
        
        if not result.get('success'):
            logger.error(f"Falha ao conectar: {result}")
            return
        
        logger.info("✅ Conectado com sucesso!")
        
        # 4. Verificar status após conexão
        time.sleep(2)
        status = client.get_status()
        logger.info(f"Status após conexão: {status}")
        
        # 5. Coletar dados históricos
        logger.info("\n📈 Coletando dados históricos...")
        result = client.collect_historical(
            symbol='WDOU25',
            start_date='01/08/2025',
            end_date='01/08/2025'
        )
        
        if result.get('success'):
            logger.info(f"✅ Coleta bem-sucedida!")
            logger.info(f"   Arquivo: {result.get('data_file')}")
            logger.info(f"   Registros: {result.get('count')}")
            
            # Ler dados se houver
            if result.get('data_file') and Path(result['data_file']).exists():
                with open(result['data_file'], 'r') as f:
                    data = json.load(f)
                    if data:
                        logger.info(f"   Primeiro: {data[0]}")
                        logger.info(f"   Último: {data[-1]}")
        else:
            logger.error(f"Falha na coleta: {result}")
        
        # 6. Status final
        logger.info("\n📊 Status final...")
        status = client.get_status()
        logger.info(f"Status: {status}")
        
    except Exception as e:
        logger.error(f"Erro no teste: {e}", exc_info=True)
    
    finally:
        # Sempre finalizar o wrapper
        client.shutdown()
    
    logger.info("\n" + "="*80)
    logger.info("🏁 TESTE CONCLUÍDO")
    logger.info("="*80)


if __name__ == "__main__":
    main()