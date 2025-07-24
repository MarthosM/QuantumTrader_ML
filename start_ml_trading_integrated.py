#!/usr/bin/env python3
"""
Sistema de Trading ML v2.0 - Startup Integrado
Versão com monitoramento completo do fluxo ML

RECURSOS INCLUÍDOS:
- Monitor de fluxo de dados automático
- GUI estendido com painéis de predição
- Mapeamento: Candles → Features → Predições → GUI
- Validação de dados em tempo real
- Histórico de predições
"""

import os
import sys
import logging
from datetime import datetime

# Configurar paths
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def setup_enhanced_logging():
    """Configura logging aprimorado para ML flow"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Logger raiz
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'ml_trading_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )
    
    # Loggers específicos
    logging.getLogger('DataFlowMonitor').setLevel(logging.INFO)
    logging.getLogger('MLIntegrator').setLevel(logging.INFO)
    logging.getLogger('GUIExtension').setLevel(logging.INFO)

def print_startup_banner():
    """Imprime banner de inicialização"""
    banner = f'''
{'='*70}
    SISTEMA DE TRADING ML v2.0 - VERSÃO INTEGRADA
{'='*70}
    
    FLUXO COMPLETO: Candles → Features → Predições → GUI
    Monitor de dados em tempo real
    Painéis de predição ML no GUI
    Histórico e análise de performance
    
    Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
{'='*70}
'''
    print(banner)

def main():
    """Função principal integrada"""
    print_startup_banner()
    setup_enhanced_logging()
    
    logger = logging.getLogger('MLStartup')
    
    try:
        logger.info("Iniciando sistema ML integrado...")
        
        # Verificar dependências
        logger.info("Verificando módulos ML...")
        
        try:
            from data_flow_monitor import DataFlowMonitor
            from gui_prediction_extension import extend_gui_with_prediction_display
            from ml_data_flow_integrator import MLDataFlowIntegrator
            logger.info("Módulos ML carregados")
        except ImportError as e:
            logger.error(f"Erro carregando módulos ML: {e}")
            logger.info("Execute este script do diretório raiz do projeto")
            return 1
            
        # Executar sistema principal
        logger.info("Iniciando sistema principal...")
        
        # Importar e executar main
        from main import main as trading_main
        return trading_main()
            
    except KeyboardInterrupt:
        logger.info("Sistema interrompido pelo usuário")
        return 0
        
    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
