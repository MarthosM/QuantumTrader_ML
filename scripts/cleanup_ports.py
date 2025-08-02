"""
Script para limpar portas ZMQ em uso
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CleanupPorts')

PORTS = [5555, 5556, 5557, 5558, 5559, 5560, 5561]

def find_process_using_port(port):
    """Encontra processo usando uma porta específica"""
    try:
        # Windows: netstat -ano | findstr :PORT
        cmd = f'netstat -ano | findstr :{port}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'LISTENING' in line:
                    # Extrair PID (último campo)
                    parts = line.split()
                    if parts:
                        pid = parts[-1]
                        return pid
        return None
        
    except Exception as e:
        logger.error(f"Erro ao buscar processo na porta {port}: {e}")
        return None

def kill_process(pid):
    """Mata processo pelo PID"""
    try:
        # Windows: taskkill /F /PID <pid>
        cmd = f'taskkill /F /PID {pid}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            logger.error(f"Erro ao matar processo {pid}: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Erro ao matar processo {pid}: {e}")
        return False

def cleanup_ports():
    """Limpa todas as portas ZMQ"""
    logger.info("Limpando portas ZMQ...")
    
    processes_killed = 0
    
    for port in PORTS:
        pid = find_process_using_port(port)
        if pid:
            logger.info(f"Porta {port} em uso pelo processo {pid}")
            if kill_process(pid):
                logger.info(f"✅ Processo {pid} finalizado")
                processes_killed += 1
            else:
                logger.error(f"❌ Falha ao finalizar processo {pid}")
        else:
            logger.info(f"✅ Porta {port} disponível")
    
    logger.info(f"\n{processes_killed} processos finalizados")
    
    # Também tentar matar processos Python zumbi
    try:
        # Encontrar processos Python
        cmd = 'tasklist | findstr python'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            logger.info("\nProcessos Python encontrados:")
            print(result.stdout)
            
    except Exception as e:
        logger.error(f"Erro listando processos Python: {e}")

if __name__ == "__main__":
    cleanup_ports()