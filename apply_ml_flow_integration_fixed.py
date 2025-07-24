#!/usr/bin/env python3
"""
Script de Integração Final - Fluxo de Dados ML
Sistema de Trading v2.0

Este script modifica o sistema existente para incluir:
1. Monitor de fluxo de dados automático
2. Extensão do GUI com painéis de predição
3. Mapeamento completo: Candles → Features → Predições → GUI

INSTRUÇÕES DE USO:
1. Execute este script para aplicar as modificações
2. Use o sistema normalmente - as extensões serão ativadas automaticamente
3. O GUI mostrará predições ML em tempo real
"""

import os
import sys
import logging
from typing import Optional

# Adicionar src ao path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)


class MLFlowIntegrationPatcher:
    """
    Aplica patches no sistema existente para incluir monitoramento
    completo do fluxo de dados ML
    """
    
    def __init__(self):
        self.logger = logging.getLogger('MLFlowPatcher')
        
    def apply_all_patches(self) -> bool:
        """Aplica todas as modificações necessárias"""
        try:
            self.logger.info("🔧 Aplicando patches de integração ML...")
            
            success_count = 0
            total_patches = 3
            
            # 1. Modificar trading_system.py
            if self._patch_trading_system():
                success_count += 1
                self.logger.info("✅ 1/3 - Trading system modificado")
            else:
                self.logger.error("❌ 1/3 - Falha modificando trading system")
                
            # 2. Modificar trading_monitor_gui.py
            if self._patch_gui_monitor():
                success_count += 1
                self.logger.info("✅ 2/3 - GUI monitor modificado")
            else:
                self.logger.error("❌ 2/3 - Falha modificando GUI monitor")
                
            # 3. Criar script de inicialização integrado
            if self._create_integrated_startup():
                success_count += 1
                self.logger.info("✅ 3/3 - Script integrado criado")
            else:
                self.logger.error("❌ 3/3 - Falha criando script integrado")
                
            # Relatório final
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"INTEGRAÇÃO ML FLOW CONCLUÍDA")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Sucessos: {success_count}/{total_patches}")
            
            if success_count == total_patches:
                self.logger.info("🎉 TODAS as modificações aplicadas com sucesso!")
                self.logger.info("\n📋 COMO USAR O SISTEMA INTEGRADO:")
                self.logger.info("1. Execute: python start_ml_trading_integrated.py")
                self.logger.info("2. O sistema mostrará predições ML em tempo real")
                self.logger.info("3. GUI terá painel dedicado para features e predições")
                self.logger.info("4. Fluxo completo será monitorado automaticamente")
                return True
            else:
                self.logger.warning(f"⚠️ Apenas {success_count}/{total_patches} modificações bem-sucedidas")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro aplicando patches: {e}")
            return False
            
    def _patch_trading_system(self) -> bool:
        """Modifica trading_system.py para incluir integração ML"""
        try:
            trading_system_path = os.path.join(src_path, 'trading_system.py')
            
            if not os.path.exists(trading_system_path):
                self.logger.error("trading_system.py não encontrado")
                return False
                
            with open(trading_system_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Verificar se já foi modificado
            if 'ml_data_flow_integrator' in content:
                self.logger.info("trading_system.py já foi modificado")
                return True
                
            # Adicionar import no início
            import_section = '''# Integração ML Flow
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
try:
    from ml_data_flow_integrator import integrate_ml_data_flow_with_system
except ImportError:
    def integrate_ml_data_flow_with_system(system):
        return None

'''
            
            # Localizar imports existentes
            imports_end = content.find('\nclass TradingSystem')
            if imports_end == -1:
                self.logger.error("Não foi possível localizar classe TradingSystem")
                return False
                
            # Inserir import
            new_content = content[:imports_end] + '\n' + import_section + content[imports_end:]
            
            # Adicionar inicialização do integrador no método start()
            start_method_pattern = 'def start(self, ticker: Optional[str] = None) -> bool:'
            start_pos = new_content.find(start_method_pattern)
            if start_pos == -1:
                # Tentar padrão alternativo
                start_method_pattern = 'def start(self'
                start_pos = new_content.find(start_method_pattern)
                if start_pos == -1:
                    self.logger.error("Método start() não encontrado")
                    return False
                
            # Localizar final do método start
            next_method = new_content.find('\n    def ', start_pos + len(start_method_pattern))
            if next_method == -1:
                next_method = len(new_content)
                
            start_end = new_content.rfind('return True', start_pos, next_method)
            if start_end == -1:
                self.logger.error("return True não encontrado no método start")
                return False
                
            # Adicionar integração antes do return
            integration_code = '''
        # 🔧 INTEGRAÇÃO ML FLOW - Configurar monitoramento automático
        try:
            self.logger.info("Configurando integração ML Flow...")
            self.ml_integrator = integrate_ml_data_flow_with_system(self)
            if self.ml_integrator:
                self.logger.info("✅ Integração ML Flow configurada")
            else:
                self.logger.warning("⚠️ Integração ML Flow não disponível")
        except Exception as e:
            self.logger.warning(f"Erro configurando ML Flow: {e}")
            
        '''
            
            new_content = new_content[:start_end] + integration_code + new_content[start_end:]
            
            # Backup e salvar
            backup_path = trading_system_path + '.backup_ml_integration'
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            with open(trading_system_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            self.logger.info(f"Backup criado: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro modificando trading_system.py: {e}")
            return False
            
    def _patch_gui_monitor(self) -> bool:
        """Modifica GUI monitor para incluir painéis ML"""
        try:
            gui_path = os.path.join(src_path, 'trading_monitor_gui.py')
            
            if not os.path.exists(gui_path):
                self.logger.error("trading_monitor_gui.py não encontrado")
                return False
                
            with open(gui_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Verificar se já foi modificado
            if 'extend_gui_with_prediction_display' in content:
                self.logger.info("trading_monitor_gui.py já foi modificado")
                return True
                
            # Adicionar import
            import_addition = '''
# ML Flow Integration
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from gui_prediction_extension import extend_gui_with_prediction_display
except ImportError:
    def extend_gui_with_prediction_display(gui):
        return False
'''
            
            # Localizar imports
            class_start = content.find('class TradingMonitorGUI:')
            if class_start == -1:
                self.logger.error("Classe TradingMonitorGUI não encontrada")
                return False
                
            new_content = content[:class_start] + import_addition + '\n\n' + content[class_start:]
            
            # Adicionar extensão no __init__
            init_pattern = 'def __init__(self, trading_system):'
            init_pos = new_content.find(init_pattern)
            if init_pos == -1:
                self.logger.error("Método __init__ não encontrado")
                return False
                
            # Localizar final do __init__
            next_method = new_content.find('\n    def ', init_pos + len(init_pattern))
            if next_method == -1:
                next_method = len(new_content)
                
            # Adicionar extensão antes do final do __init__
            extension_code = '''
        
        # 🔧 ML FLOW INTEGRATION - Estender GUI com painéis de predição
        try:
            if extend_gui_with_prediction_display(self):
                self.logger.info("✅ GUI estendido com painéis ML")
            else:
                self.logger.info("ℹ️ Extensão ML não disponível")
        except Exception as e:
            self.logger.warning(f"Erro estendendo GUI: {e}")
'''
            
            insert_pos = next_method - 1
            new_content = new_content[:insert_pos] + extension_code + new_content[insert_pos:]
            
            # Backup e salvar
            backup_path = gui_path + '.backup_ml_integration'
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            with open(gui_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            self.logger.info(f"Backup criado: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro modificando GUI: {e}")
            return False
            
    def _create_integrated_startup(self) -> bool:
        """Cria script de inicialização integrado"""
        try:
            startup_path = os.path.join(project_root, 'start_ml_trading_integrated.py')
            
            startup_content = '''#!/usr/bin/env python3
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
    banner = f\'\'\'
{'='*70}
    SISTEMA DE TRADING ML v2.0 - VERSÃO INTEGRADA
{'='*70}
    
    FLUXO COMPLETO: Candles → Features → Predições → GUI
    Monitor de dados em tempo real
    Painéis de predição ML no GUI
    Histórico e análise de performance
    
    Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
{'='*70}
\'\'\'
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
'''
            
            with open(startup_path, 'w', encoding='utf-8') as f:
                f.write(startup_content)
                
            self.logger.info(f"Script integrado criado: {startup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro criando startup integrado: {e}")
            return False


def main():
    """Função principal do patcher"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('MLFlowPatch')
    logger.info("🔧 Iniciando aplicação de patches ML Flow...")
    
    try:
        patcher = MLFlowIntegrationPatcher()
        success = patcher.apply_all_patches()
        
        if success:
            logger.info("✅ Patches aplicados com sucesso!")
            logger.info("\n🚀 PRÓXIMOS PASSOS:")
            logger.info("1. Execute: python start_ml_trading_integrated.py")
            logger.info("2. Configure suas credenciais no .env")
            logger.info("3. O sistema mostrará predições ML em tempo real")
            return 0
        else:
            logger.error("❌ Falha aplicando patches")
            return 1
            
    except Exception as e:
        logger.error(f"Erro: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
