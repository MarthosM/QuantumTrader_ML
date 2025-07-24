#!/usr/bin/env python3
"""
Correção do erro de threading do GUI
Sistema de trading ML v2.0

O problema: GUI executando em thread daemon causa erro "main thread is not in main loop"
Solução: Executar GUI na thread principal e sistema em threads separadas
"""

import os
import sys
import logging
import threading
import time
from typing import Optional

# Adicionar src ao path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

class GUIThreadingFix:
    """
    Classe para corrigir problemas de threading do GUI
    """
    
    def __init__(self):
        self.logger = logging.getLogger('GUIFix')
        
    def apply_trading_system_fix(self):
        """Aplica correção no trading_system.py"""
        trading_system_path = os.path.join(src_path, 'trading_system.py')
        
        if not os.path.exists(trading_system_path):
            self.logger.error("Arquivo trading_system.py não encontrado")
            return False
            
        try:
            with open(trading_system_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Localizar seção do GUI
            gui_section_start = content.find("# 7. Iniciar monitor GUI se habilitado")
            if gui_section_start == -1:
                self.logger.error("Seção do GUI não encontrada")
                return False
                
            # Localizar o final da seção do GUI
            main_loop_start = content.find("# 8. Entrar no loop principal", gui_section_start)
            if main_loop_start == -1:
                self.logger.error("Início do loop principal não encontrado")
                return False
                
            # Extrair seção atual do GUI
            gui_section = content[gui_section_start:main_loop_start]
            
            # Nova implementação do GUI corrigida
            new_gui_section = '''            # 7. Iniciar monitor GUI se habilitado
            if self.use_gui:
                self.logger.info("Iniciando monitor visual...")
                try:
                    from trading_monitor_gui import create_monitor_gui
                    self.monitor = create_monitor_gui(self)
                    
                    # 🔧 CORREÇÃO: GUI deve rodar na thread principal
                    # Sistema roda em background, GUI na main thread
                    self.logger.info("✓ Monitor GUI configurado para thread principal")
                    
                    # Armazenar referência para controle do GUI
                    self._gui_ready = True
                    
                except Exception as e:
                    self.logger.warning(f"Erro configurando monitor GUI: {e}")
                    self.logger.info("Sistema continuará sem monitor visual")
                    self.monitor = None
                    
            # 7.5. Se GUI habilitado, configurar execução na thread principal
            if self.use_gui and self.monitor:
                self.logger.info("Sistema será executado em background thread")
                
                # Sistema roda em thread separada
                system_thread = threading.Thread(
                    target=self._main_loop_background,
                    daemon=False,  # Não daemon para controle adequado
                    name="TradingSystem"
                )
                system_thread.start()
                
                # GUI roda na thread principal
                self.logger.info("Iniciando GUI na thread principal...")
                try:
                    self.monitor.run()  # Bloqueia na thread principal
                finally:
                    # Cleanup quando GUI fechar
                    self.logger.info("GUI fechado, parando sistema...")
                    self.stop()
                    if system_thread.is_alive():
                        system_thread.join(timeout=5)
                        
                return True
            else:
                # Sem GUI - comportamento original
                
'''
            
            # Substituir seção do GUI
            new_content = content[:gui_section_start] + new_gui_section + content[main_loop_start:]
            
            # Backup do arquivo original
            backup_path = trading_system_path + '.backup_gui_fix'
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.logger.info(f"Backup criado: {backup_path}")
            
            # Salvar nova versão
            with open(trading_system_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self.logger.info("✓ Correção aplicada no trading_system.py")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro aplicando correção: {e}")
            return False
            
    def add_background_loop_method(self):
        """Adiciona método _main_loop_background ao trading_system.py"""
        trading_system_path = os.path.join(src_path, 'trading_system.py')
        
        try:
            with open(trading_system_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Localizar método _main_loop original
            main_loop_start = content.find("def _main_loop(self):")
            if main_loop_start == -1:
                self.logger.error("Método _main_loop não encontrado")
                return False
                
            # Localizar final do método _main_loop
            # Procurar próxima definição de método
            next_method = content.find("\n    def ", main_loop_start + 1)
            if next_method == -1:
                # Se não encontrar, procurar final da classe
                next_method = content.find("\nclass ", main_loop_start + 1)
                if next_method == -1:
                    # Se não encontrar, fim do arquivo
                    next_method = len(content)
                    
            main_loop_end = next_method
            
            # Método background para ser inserido
            background_method = '''
    def _main_loop_background(self):
        """
        Loop principal executado em background quando GUI está ativo
        Versão modificada do _main_loop original para threading
        """
        self.logger.info("Iniciando loop principal em background thread...")
        
        try:
            # Loop principal do sistema
            self._main_loop()
            
        except Exception as e:
            self.logger.error(f"Erro no loop background: {e}", exc_info=True)
        finally:
            self.logger.info("Loop background finalizado")
'''
            
            # Inserir método background antes do próximo método
            new_content = content[:main_loop_end] + background_method + content[main_loop_end:]
            
            # Salvar
            with open(trading_system_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            self.logger.info("✓ Método _main_loop_background adicionado")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro adicionando método background: {e}")
            return False
            
    def fix_gui_monitor_threading(self):
        """Corrige problemas de threading no monitor GUI"""
        gui_path = os.path.join(src_path, 'trading_monitor_gui.py')
        
        if not os.path.exists(gui_path):
            self.logger.error("Arquivo trading_monitor_gui.py não encontrado")
            return False
            
        try:
            with open(gui_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Localizar método run
            run_method_start = content.find("def run(self):")
            if run_method_start == -1:
                self.logger.error("Método run não encontrado no GUI")
                return False
                
            # Localizar final do método run
            next_method = content.find("\n\ndef ", run_method_start + 1)
            if next_method == -1:
                next_method = len(content)
                
            # Nova implementação do método run
            new_run_method = '''    def run(self):
        """
        Inicia a interface gráfica na thread principal
        🔧 CORREÇÃO: Garante execução na thread principal
        """
        self.logger = logging.getLogger('GUI')
        self.logger.info("Iniciando GUI na thread principal...")
        
        try:
            # Configurar protocolo de fechamento
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            
            # Iniciar monitoramento automaticamente se sistema rodando
            if hasattr(self.trading_system, 'is_running') and self.trading_system.is_running:
                self.root.after(1000, self.start_monitoring)  # Delay para garantir inicialização
                
            # Executar mainloop na thread principal
            self.root.mainloop()
            
        except Exception as e:
            self.logger.error(f"Erro executando GUI: {e}", exc_info=True)
        finally:
            self.logger.info("GUI finalizado")
'''
            
            # Substituir método run
            new_content = content[:run_method_start] + new_run_method + content[next_method:]
            
            # Backup
            backup_path = gui_path + '.backup_gui_fix'
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            # Salvar nova versão
            with open(gui_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            self.logger.info("✓ Correção aplicada no trading_monitor_gui.py")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro corrigindo GUI: {e}")
            return False
            
    def create_fixed_main(self):
        """Cria versão corrigida do main.py"""
        main_path = os.path.join(src_path, 'main.py')
        fixed_main_path = os.path.join(src_path, 'main_fixed.py')
        
        try:
            with open(main_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Localizar função main
            main_func_start = content.find("def main():")
            if main_func_start == -1:
                self.logger.error("Função main não encontrada")
                return False
                
            # Nova implementação da função main
            new_main_func = '''def main():
    """Função principal - VERSÃO CORRIGIDA PARA GUI"""
    # Configurar logging apenas uma vez
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    logger = logging.getLogger('Main')
    logger.info("Iniciando Sistema de Trading v2.0 - Versão GUI Corrigida")
    
    try:
        # Carregar configuração
        config = load_config()
        
        # Verificar credenciais obrigatórias
        required_fields = ['key', 'username', 'password']
        missing_fields = [field for field in required_fields if not config.get(field)]
        
        if missing_fields:
            logger.error(f"Campos obrigatórios não configurados no .env: {', '.join(missing_fields)}")
            return 1
            
        # Log das configurações (sem senhas)
        logger.info(f"Usuário: {config['username']}")
        logger.info(f"Conta: {config.get('account_id', 'Não especificada')}")
        logger.info(f"Corretora: {config.get('broker_id', 'Não especificada')}")
        logger.info(f"Ticker: {config.get('ticker', 'Auto-detectado')}")
        logger.info(f"GUI Habilitado: {config.get('use_gui', False)}")
            
        # Criar sistema
        system = TradingSystem(config)
        
        # Inicializar
        if not system.initialize():
            logger.error("Falha na inicialização do sistema")
            return 1
            
        # 🔧 CORREÇÃO CRÍTICA: Tratamento especial para GUI
        if config.get('use_gui', False):
            logger.info("Modo GUI: Sistema rodará em background, GUI na thread principal")
            
            # Iniciar operação (que agora gerencia threading automaticamente)
            if not system.start():
                logger.error("Falha ao iniciar operação")
                return 1
        else:
            logger.info("Modo Console: Execução tradicional")
            
            # Iniciar operação sem GUI
            if not system.start():
                logger.error("Falha ao iniciar operação")
                return 1
                
        return 0
        
    except KeyboardInterrupt:
        logger.info("Sistema interrompido pelo usuário")
        return 0
        
    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)
        return 1'''
        
            # Encontrar fim da função main original
            end_of_main = content.find("\n\nif __name__ == '__main__':")
            if end_of_main == -1:
                end_of_main = len(content)
                
            # Substituir função main
            before_main = content[:main_func_start]
            after_main = content[end_of_main:]
            
            new_content = before_main + new_main_func + after_main
            
            # Salvar versão corrigida
            with open(fixed_main_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            self.logger.info(f"✓ Versão corrigida criada: {fixed_main_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro criando main corrigido: {e}")
            return False
            
    def run_complete_fix(self):
        """Executa correção completa do sistema"""
        self.logger.info("🔧 Iniciando correção completa do threading GUI...")
        
        success_count = 0
        total_fixes = 4
        
        # 1. Corrigir trading_system.py
        if self.apply_trading_system_fix():
            success_count += 1
            self.logger.info("✓ 1/4 - Trading system corrigido")
        else:
            self.logger.error("✗ 1/4 - Falha na correção do trading system")
            
        # 2. Adicionar método background
        if self.add_background_loop_method():
            success_count += 1
            self.logger.info("✓ 2/4 - Método background adicionado")
        else:
            self.logger.error("✗ 2/4 - Falha ao adicionar método background")
            
        # 3. Corrigir GUI
        if self.fix_gui_monitor_threading():
            success_count += 1
            self.logger.info("✓ 3/4 - Monitor GUI corrigido")
        else:
            self.logger.error("✗ 3/4 - Falha na correção do monitor GUI")
            
        # 4. Criar main corrigido
        if self.create_fixed_main():
            success_count += 1
            self.logger.info("✓ 4/4 - Main corrigido criado")
        else:
            self.logger.error("✗ 4/4 - Falha ao criar main corrigido")
            
        # Relatório final
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"CORREÇÃO GUI THREADING CONCLUÍDA")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Sucessos: {success_count}/{total_fixes}")
        
        if success_count == total_fixes:
            self.logger.info("🎉 TODAS as correções aplicadas com sucesso!")
            self.logger.info("\n📋 INSTRUÇÕES DE USO:")
            self.logger.info("1. Use 'python src/main_fixed.py' para versão corrigida")
            self.logger.info("2. Ou substitua main.py original pela versão corrigida")
            self.logger.info("3. GUI agora roda na thread principal (sem erros)")
            self.logger.info("4. Sistema roda em background thread")
            self.logger.info("5. Fechamento do GUI para automaticamente o sistema")
            return True
        else:
            self.logger.warning(f"⚠️  Apenas {success_count}/{total_fixes} correções bem-sucedidas")
            self.logger.info("Verifique os logs para detalhes dos erros")
            return False


def main():
    """Função principal da correção"""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('GUIFix')
    logger.info("Iniciando correção de threading do GUI...")
    
    try:
        fixer = GUIThreadingFix()
        success = fixer.run_complete_fix()
        
        if success:
            logger.info("✓ Correção concluída com sucesso!")
            return 0
        else:
            logger.error("✗ Correção falhou")
            return 1
            
    except Exception as e:
        logger.error(f"Erro na correção: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
