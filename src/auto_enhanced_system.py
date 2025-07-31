# -*- coding: utf-8 -*-
"""
Sistema de detecção automática de enhanced features
Importar este módulo no lugar de trading_system para detecção automática
"""

import os
import logging

logger = logging.getLogger('AutoEnhancedSystem')

def get_trading_system_class():
    """
    Retorna a classe apropriada baseado na configuração
    """
    # Verificar se enhanced está habilitado no ambiente
    enhanced_enabled = (
        os.getenv('ZMQ_ENABLED', 'false').lower() == 'true' or
        os.getenv('VALKEY_ENABLED', 'false').lower() == 'true' or
        os.getenv('TIME_TRAVEL_ENABLED', 'false').lower() == 'true' or
        os.getenv('ENHANCED_ML_ENABLED', 'false').lower() == 'true'
    )
    
    if enhanced_enabled:
        try:
            from trading_system_enhanced import TradingSystemEnhanced
            logger.info("Sistema Enhanced detectado e carregado")
            return TradingSystemEnhanced
        except ImportError as e:
            logger.warning(f"Enhanced habilitado mas não disponível: {e}")
    
    # Fallback para sistema normal
    from trading_system import TradingSystem
    logger.info("Usando sistema padrão")
    return TradingSystem

# Exportar como TradingSystem para substituição transparente
TradingSystem = get_trading_system_class()

# Monkey patch para o monitor GUI detectar automaticamente
try:
    import trading_monitor_gui
    
    # Salvar função original
    _original_create_monitor = trading_monitor_gui.create_monitor_gui
    
    def create_monitor_gui_enhanced(trading_system):
        """
        Versão enhanced do create_monitor_gui que adiciona extensões automaticamente
        """
        # Criar monitor normal
        monitor = _original_create_monitor(trading_system)
        
        # Tentar adicionar extensão se sistema for enhanced
        try:
            if hasattr(trading_system, 'get_enhanced_status'):
                from gui_extensions.zmq_valkey_monitor_extension import integrate_zmq_valkey_monitor
                
                extension = integrate_zmq_valkey_monitor(monitor, trading_system)
                if extension and extension.enhanced_active:
                    logger.info("✅ Monitor GUI aprimorado com extensões ZMQ/Valkey")
        except Exception as e:
            logger.debug(f"Extensão não aplicada: {e}")
        
        return monitor
    
    # Substituir função
    trading_monitor_gui.create_monitor_gui = create_monitor_gui_enhanced
    logger.info("Patch do monitor GUI aplicado com sucesso")
    
except ImportError:
    logger.debug("Monitor GUI não disponível para patch")
except Exception as e:
    logger.warning(f"Erro ao aplicar patch do monitor: {e}")

__all__ = ['TradingSystem']