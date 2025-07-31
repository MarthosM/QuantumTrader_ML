# -*- coding: utf-8 -*-
"""
Script de ativação da extensão de monitoramento ZMQ/Valkey
Exemplo de como integrar com o monitor GUI existente
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.trading_monitor_gui import TradingMonitorGUI
from src.gui_extensions.zmq_valkey_monitor_extension import integrate_zmq_valkey_monitor
from src.trading_system_enhanced import TradingSystemEnhanced
import logging

def activate_enhanced_monitoring():
    """
    Ativa o monitoramento enhanced no GUI existente
    """
    try:
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('EnhancedMonitorActivation')
        
        logger.info("Iniciando ativação do monitor enhanced...")
        
        # Criar ou obter instância do sistema de trading
        # (normalmente isso já existe no seu main.py)
        config = {
            'ticker': 'WDOU25',  # Ajustar conforme necessário
            'user_id': 'your_user_id',
            'broker_id': 'your_broker_id',
            'broker_name': 'your_broker',
            'account_id': 'your_account'
        }
        
        # Usar sistema enhanced se disponível
        try:
            trading_system = TradingSystemEnhanced(config)
            logger.info("Sistema enhanced detectado e carregado")
        except:
            # Fallback para sistema normal
            from src.trading_system import TradingSystem
            trading_system = TradingSystem(config)
            logger.info("Usando sistema padrão")
        
        # Criar monitor GUI
        monitor = TradingMonitorGUI(trading_system)
        
        # Integrar extensão ZMQ/Valkey se sistema enhanced estiver ativo
        extension = integrate_zmq_valkey_monitor(monitor, trading_system)
        
        if extension and extension.enhanced_active:
            logger.info("Extensão ZMQ/Valkey integrada com sucesso!")
            logger.info("Novas abas disponíveis: ZMQ/Valkey e Time Travel")
        else:
            logger.info("Sistema enhanced não detectado - monitor padrão ativo")
        
        # Iniciar GUI
        monitor.run()
        
    except Exception as e:
        logger.error(f"Erro ao ativar monitor enhanced: {e}")
        raise

if __name__ == "__main__":
    activate_enhanced_monitoring()