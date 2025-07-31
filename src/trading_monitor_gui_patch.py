# -*- coding: utf-8 -*-
"""
Patch para adicionar suporte enhanced ao trading_monitor_gui.py existente
Adicione este código ao final do método __init__ da classe TradingMonitorGUI
"""

# CÓDIGO PARA ADICIONAR NO FINAL DO __init__ DO TradingMonitorGUI:

"""
        # Tentar integrar extensão ZMQ/Valkey se disponível
        self.zmq_valkey_extension = None
        try:
            # Verificar se é sistema enhanced
            if hasattr(self.trading_system, 'get_enhanced_status'):
                from gui_extensions.zmq_valkey_monitor_extension import integrate_zmq_valkey_monitor
                self.zmq_valkey_extension = integrate_zmq_valkey_monitor(self, self.trading_system)
                
                if self.zmq_valkey_extension and self.zmq_valkey_extension.enhanced_active:
                    self.logger.info("✅ Extensão ZMQ/Valkey integrada com sucesso")
                    
                    # Adicionar ao loop de atualização
                    if hasattr(self, 'update_functions'):
                        self.update_functions.append(self.zmq_valkey_extension.update_display)
                    else:
                        # Se não tiver lista de funções, modificar update_all
                        original_update_all = self.update_all
                        
                        def enhanced_update_all():
                            original_update_all()
                            if self.zmq_valkey_extension:
                                self.zmq_valkey_extension.update_display()
                        
                        self.update_all = enhanced_update_all
                        
        except ImportError:
            self.logger.debug("Módulo de extensão ZMQ/Valkey não disponível")
        except Exception as e:
            self.logger.debug(f"Extensão ZMQ/Valkey não ativada: {e}")
"""

# ALTERNATIVA: Modificar a função create_monitor_gui para incluir a extensão:

def create_enhanced_monitor_gui(trading_system):
    """
    Factory function aprimorada que cria monitor GUI com extensões se disponível
    
    Args:
        trading_system: Instância do TradingSystem
        
    Returns:
        TradingMonitorGUI: Instância do monitor GUI (possivelmente com extensões)
    """
    from trading_monitor_gui import TradingMonitorGUI
    
    # Criar monitor normal
    monitor = TradingMonitorGUI(trading_system)
    
    # Tentar adicionar extensão
    try:
        from gui_extensions.zmq_valkey_monitor_extension import integrate_zmq_valkey_monitor
        extension = integrate_zmq_valkey_monitor(monitor, trading_system)
        
        if extension and extension.enhanced_active:
            monitor.logger.info("Monitor GUI criado com extensões enhanced")
    except:
        pass  # Silenciosamente continuar sem extensão
    
    return monitor