#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Trading ML v2.0 - Versão Integrada SEM EMOJIS
Entrada principal para o sistema integrado com monitoramento de fluxo ML
Data: 2025-07-22
"""

import sys
import os
import logging
from datetime import datetime

def configure_logging():
    """Configura logging sem unicode"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ml_trading_integrated.log', encoding='utf-8')
        ]
    )

def main():
    """Função principal de entrada"""
    
    print("=" * 80)
    print("    SISTEMA DE TRADING ML v2.0 - VERSAO INTEGRADA")
    print("=" * 80)
    print("    FLUXO COMPLETO: Candles -> Features -> Predicoes -> GUI")
    print("    Monitor de dados em tempo real")
    print("    Paineis de predicao ML no GUI")
    print("    Historico e analise de performance")
    print(f"    Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Configurar logging primeiro
    configure_logging()
    
    # Logger de startup
    logger = logging.getLogger('MLStartup')
    logger.info("Iniciando sistema ML integrado...")
    
    try:
        # Verificar se estamos no diretório src
        if not os.path.exists('main.py'):
            # Tentar ir para src
            if os.path.exists('src/main.py'):
                os.chdir('src')
                logger.info("Mudando para diretorio src/")
            else:
                logger.error("Nao foi possivel encontrar main.py")
                return 1
        
        # Importar os módulos ML primeiro
        logger.info("Verificando modulos ML...")
        try:
            from data_flow_monitor import DataFlowMonitor
            from gui_prediction_extension import PredictionDisplayPanel 
            from ml_data_flow_integrator import MLDataFlowIntegrator
            logger.info("Modulos ML carregados")
        except ImportError as e:
            logger.warning(f"Módulos ML não encontrados: {e}")
            logger.info("Sistema continuará sem integração ML avançada")
        
        # Importar e executar sistema principal
        logger.info("Iniciando sistema principal...")
        from main import main as trading_main
        return trading_main()
        
    except Exception as e:
        logger.error(f"Erro fatal durante inicialização: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
