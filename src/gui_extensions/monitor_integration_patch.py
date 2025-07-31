# -*- coding: utf-8 -*-
"""
Patch para integrar extensão ZMQ/Valkey ao monitor existente
Este código mostra como modificar o TradingMonitorGUI existente
"""

# OPÇÃO 1: Adicionar ao __init__ do TradingMonitorGUI existente
"""
# Adicionar após a inicialização do GUI no método __init__:

# Tentar integrar extensão ZMQ/Valkey
try:
    from gui_extensions.zmq_valkey_monitor_extension import integrate_zmq_valkey_monitor
    self.zmq_valkey_extension = integrate_zmq_valkey_monitor(self, self.trading_system)
    if self.zmq_valkey_extension and self.zmq_valkey_extension.enhanced_active:
        self.logger.info("Extensão ZMQ/Valkey integrada ao monitor")
except Exception as e:
    self.logger.debug(f"Extensão ZMQ/Valkey não disponível: {e}")
    self.zmq_valkey_extension = None
"""

# OPÇÃO 2: Adicionar como método separado
def integrate_enhanced_monitoring(monitor_gui):
    """
    Integra monitoramento enhanced ao GUI existente
    
    Args:
        monitor_gui: Instância de TradingMonitorGUI
    """
    try:
        from gui_extensions.zmq_valkey_monitor_extension import integrate_zmq_valkey_monitor
        
        extension = integrate_zmq_valkey_monitor(monitor_gui, monitor_gui.trading_system)
        
        if extension and extension.enhanced_active:
            monitor_gui.logger.info("Sistema enhanced detectado - novas abas adicionadas")
            
            # Se o monitor tiver um método update_all, adicionar a extensão
            if hasattr(monitor_gui, 'update_all'):
                original_update = monitor_gui.update_all
                
                def enhanced_update():
                    original_update()
                    if extension:
                        extension.update_display()
                
                monitor_gui.update_all = enhanced_update
            
            return extension
        else:
            monitor_gui.logger.info("Sistema enhanced não ativo - usando monitor padrão")
            return None
            
    except ImportError:
        monitor_gui.logger.debug("Módulo de extensão não encontrado")
        return None
    except Exception as e:
        monitor_gui.logger.error(f"Erro ao integrar extensão: {e}")
        return None

# OPÇÃO 3: Decorador para adicionar funcionalidade
def with_enhanced_monitoring(gui_class):
    """
    Decorador que adiciona monitoramento enhanced a uma classe GUI
    """
    original_init = gui_class.__init__
    
    def new_init(self, *args, **kwargs):
        # Chamar init original
        original_init(self, *args, **kwargs)
        
        # Tentar adicionar extensão
        try:
            from gui_extensions.zmq_valkey_monitor_extension import integrate_zmq_valkey_monitor
            self.zmq_valkey_extension = integrate_zmq_valkey_monitor(self, self.trading_system)
        except:
            self.zmq_valkey_extension = None
    
    gui_class.__init__ = new_init
    return gui_class

# Exemplo de uso do decorador:
"""
@with_enhanced_monitoring
class TradingMonitorGUI:
    # ... código existente ...
"""

# OPÇÃO 4: Função helper para modificação mínima
def setup_enhanced_monitor_if_available(monitor_gui):
    """
    Setup helper que pode ser chamado após criar o monitor
    
    Uso:
        monitor = TradingMonitorGUI(trading_system)
        setup_enhanced_monitor_if_available(monitor)
        monitor.run()
    """
    try:
        # Verificar se sistema é enhanced
        if hasattr(monitor_gui.trading_system, 'get_enhanced_status'):
            status = monitor_gui.trading_system.get_enhanced_status()
            if status['enhanced_features'].get('zmq_enabled') or \
               status['enhanced_features'].get('valkey_enabled'):
                
                from gui_extensions.zmq_valkey_monitor_extension import integrate_zmq_valkey_monitor
                extension = integrate_zmq_valkey_monitor(monitor_gui, monitor_gui.trading_system)
                
                if extension:
                    print("Monitor enhanced ativado com sucesso!")
                    print("Novas abas disponíveis:")
                    print("- ZMQ/Valkey: Status e estatísticas do sistema enhanced")
                    if extension._has_time_travel():
                        print("- Time Travel: Features avançadas de análise histórica")
                    return True
    except:
        pass
    
    return False