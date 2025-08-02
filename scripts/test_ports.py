"""
Verifica se as portas ZMQ estão disponíveis
"""

import socket
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TestPorts')

# Portas usadas pelo sistema HMARL
PORTS = {
    5555: 'tick',
    5556: 'book', 
    5557: 'flow',
    5558: 'footprint',
    5559: 'liquidity/signal',
    5560: 'tape',
    5561: 'decisions'
}

def check_port(port):
    """Verifica se uma porta está disponível"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            return result != 0  # True se disponível
    except Exception as e:
        logger.error(f"Erro verificando porta {port}: {e}")
        return False

def main():
    logger.info("Verificando portas ZMQ...\n")
    
    all_available = True
    
    for port, name in PORTS.items():
        available = check_port(port)
        status = "✅ Disponível" if available else "❌ Em uso"
        logger.info(f"Porta {port} ({name}): {status}")
        
        if not available:
            all_available = False
    
    if all_available:
        logger.info("\n✅ Todas as portas estão disponíveis")
    else:
        logger.info("\n❌ Algumas portas estão em uso")
        logger.info("Execute: netstat -an | findstr :555")
        logger.info("Para liberar: taskkill /F /PID <pid>")

if __name__ == "__main__":
    main()