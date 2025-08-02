#!/usr/bin/env python3
"""
Script de Migração Automática para ProfitDLL v4.0.0.30
Atualiza imports e referências no código
"""

import os
import re
import shutil
import sys
from datetime import datetime
from typing import List, Tuple, Dict
import argparse


class ProfitDLLMigrator:
    """Automatiza o processo de migração para v4.0.0.30"""
    
    def __init__(self, backup_dir: str = "backup_pre_v4", dry_run: bool = False):
        self.backup_dir = backup_dir
        self.dry_run = dry_run
        self.changes_made = []
        self.files_to_update = []
        
        # Mapeamento de substituições
        self.import_replacements = [
            # Connection Manager
            (r'from connection_manager import ConnectionManager',
             'from connection_manager_v4 import ConnectionManagerV4 as ConnectionManager'),
            (r'import connection_manager',
             'import connection_manager_v4 as connection_manager'),
            
            # Order Manager
            (r'from order_manager import OrderExecutionManager',
             'from order_manager_v4 import OrderExecutionManagerV4 as OrderExecutionManager'),
            (r'import order_manager',
             'import order_manager_v4 as order_manager'),
        ]
        
        # Substituições de código
        self.code_replacements = [
            # Funções depreciadas
            (r'\.SendBuyOrder\(', '.SendOrder('),
            (r'\.SendSellOrder\(', '.SendOrder('),
            (r'\.SendMarketBuyOrder\(', '.SendOrder('),
            (r'\.SendMarketSellOrder\(', '.SendOrder('),
            (r'\.SendCancelOrder\(', '.SendCancelOrderV2('),
            (r'\.GetPosition\(', '.GetPositionV2('),
            
            # Callbacks antigos
            (r'SetOrderChangeCallback', 'SetOrderCallback'),
            (r'SetHistoryCallback', 'SetOrderHistoryCallback'),
        ]
        
    def find_files_to_update(self) -> List[str]:
        """Encontra todos os arquivos Python que precisam ser atualizados"""
        files = []
        
        # Arquivos principais conhecidos
        main_files = [
            'src/trading_system.py',
            'src/data_integration.py',
            'src/execution_engine.py',
            'src/execution_integration.py',
            'src/database/historical_data_collector.py',
            'src/paper_trading/paper_trader_v3.py',
        ]
        
        # Adicionar arquivos que existem
        for file in main_files:
            if os.path.exists(file):
                files.append(file)
                
        # Buscar outros arquivos que podem precisar de atualização
        for root, dirs, filenames in os.walk('src'):
            # Ignorar diretórios de backup e __pycache__
            dirs[:] = [d for d in dirs if d not in ['__pycache__', 'backup_pre_v4']]
            
            for filename in filenames:
                if filename.endswith('.py') and not filename.endswith('_v4.py'):
                    filepath = os.path.join(root, filename)
                    # Verificar se o arquivo contém imports antigos
                    if self._needs_update(filepath):
                        if filepath not in files:
                            files.append(filepath)
        
        return files
    
    def _needs_update(self, filepath: str) -> bool:
        """Verifica se um arquivo precisa ser atualizado"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Verificar imports antigos
            for old_pattern, _ in self.import_replacements:
                if re.search(old_pattern, content):
                    return True
                    
            # Verificar código antigo
            for old_pattern, _ in self.code_replacements:
                if re.search(old_pattern, content):
                    return True
                    
            return False
            
        except Exception as e:
            print(f"Erro ao verificar {filepath}: {e}")
            return False
    
    def backup_files(self, files: List[str]):
        """Cria backup dos arquivos antes da migração"""
        if self.dry_run:
            print(f"[DRY RUN] Criaria backup de {len(files)} arquivos")
            return
            
        # Criar diretório de backup com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.backup_dir}_{timestamp}"
        
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)
            
        for file in files:
            try:
                # Manter estrutura de diretórios no backup
                relative_path = os.path.relpath(file, '.')
                backup_file = os.path.join(backup_path, relative_path)
                
                # Criar diretório se necessário
                backup_dir = os.path.dirname(backup_file)
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir)
                
                # Copiar arquivo
                shutil.copy2(file, backup_file)
                print(f"✅ Backup criado: {backup_file}")
                
            except Exception as e:
                print(f"❌ Erro ao fazer backup de {file}: {e}")
    
    def update_file(self, filepath: str) -> List[str]:
        """Atualiza um arquivo com as novas importações e código"""
        changes = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                original_content = content
            
            # Aplicar substituições de imports
            for old_pattern, new_pattern in self.import_replacements:
                if re.search(old_pattern, content):
                    content = re.sub(old_pattern, new_pattern, content)
                    changes.append(f"Import: {old_pattern} → {new_pattern}")
            
            # Aplicar substituições de código
            for old_pattern, new_pattern in self.code_replacements:
                matches = len(re.findall(old_pattern, content))
                if matches > 0:
                    content = re.sub(old_pattern, new_pattern, content)
                    changes.append(f"Código: {old_pattern} → {new_pattern} ({matches} ocorrências)")
            
            # Adicionar import de estruturas se necessário
            if 'SendOrder(' in content and 'from profit_dll_structures import' not in content:
                # Adicionar import após os imports existentes
                import_line = "\nfrom profit_dll_structures import (\n    OrderSide, OrderType, NResult,\n    create_account_identifier, create_send_order\n)\n"
                
                # Encontrar o último import
                import_matches = list(re.finditer(r'^(from|import)\s+.*$', content, re.MULTILINE))
                if import_matches:
                    last_import = import_matches[-1]
                    insert_pos = last_import.end()
                    content = content[:insert_pos] + import_line + content[insert_pos:]
                    changes.append("Adicionado import de profit_dll_structures")
            
            # Salvar arquivo se houve mudanças
            if content != original_content:
                if not self.dry_run:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"✅ Atualizado: {filepath}")
                else:
                    print(f"[DRY RUN] Atualizaria: {filepath}")
                
                for change in changes:
                    print(f"  - {change}")
            
            return changes
            
        except Exception as e:
            print(f"❌ Erro ao atualizar {filepath}: {e}")
            return []
    
    def create_validation_script(self):
        """Cria script para validar a migração"""
        validation_script = '''#!/usr/bin/env python3
"""
Script de Validação Pós-Migração
Verifica se a migração foi bem-sucedida
"""

import sys
import importlib
import traceback

def validate_imports():
    """Valida que os novos módulos podem ser importados"""
    print("🔍 Validando imports...")
    
    modules_to_test = [
        'profit_dll_structures',
        'connection_manager_v4',
        'order_manager_v4',
    ]
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module} importado com sucesso")
        except ImportError as e:
            print(f"  ❌ Erro ao importar {module}: {e}")
            return False
    
    return True

def validate_structures():
    """Valida que as estruturas estão corretas"""
    print("\\n🔍 Validando estruturas...")
    
    try:
        from profit_dll_structures import (
            TConnectorSendOrder, TConnectorAccountIdentifier,
            create_account_identifier, create_send_order,
            OrderSide, OrderType
        )
        
        # Testar criação de estruturas
        account = create_account_identifier(1, "12345")
        assert account.BrokerID == 1
        assert account.AccountID == "12345"
        print("  ✅ TConnectorAccountIdentifier OK")
        
        # Testar ordem
        order = create_send_order(
            account=account,
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=50.0
        )
        assert order.Ticker == "TEST"
        assert order.Quantity == 100
        print("  ✅ TConnectorSendOrder OK")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro validando estruturas: {e}")
        traceback.print_exc()
        return False

def validate_main_system():
    """Valida que o sistema principal pode ser importado"""
    print("\\n🔍 Validando sistema principal...")
    
    try:
        # Tentar importar trading_system
        import trading_system
        print("  ✅ trading_system importado")
        
        # Verificar que usa as classes v4
        if hasattr(trading_system, 'ConnectionManager'):
            conn_module = trading_system.ConnectionManager.__module__
            if 'v4' in conn_module:
                print(f"  ✅ Usando ConnectionManagerV4 de {conn_module}")
            else:
                print(f"  ⚠️  ConnectionManager de {conn_module} - verificar se é v4")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro ao importar trading_system: {e}")
        return False

def main():
    """Executa todas as validações"""
    print("=" * 60)
    print("VALIDAÇÃO PÓS-MIGRAÇÃO PROFITDLL v4.0.0.30")
    print("=" * 60)
    
    all_ok = True
    
    # Adicionar src ao path
    sys.path.insert(0, 'src')
    
    # Executar validações
    all_ok &= validate_imports()
    all_ok &= validate_structures()
    all_ok &= validate_main_system()
    
    print("\\n" + "=" * 60)
    if all_ok:
        print("✅ MIGRAÇÃO VALIDADA COM SUCESSO!")
    else:
        print("❌ PROBLEMAS ENCONTRADOS NA MIGRAÇÃO")
        print("Por favor, verifique os erros acima")
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
'''
        
        if not self.dry_run:
            with open('validate_migration.py', 'w') as f:
                f.write(validation_script)
            os.chmod('validate_migration.py', 0o755)
            print("\n✅ Script de validação criado: validate_migration.py")
        else:
            print("\n[DRY RUN] Criaria script de validação")
    
    def run(self):
        """Executa o processo de migração"""
        print("=" * 60)
        print("MIGRAÇÃO PARA PROFITDLL v4.0.0.30")
        print("=" * 60)
        
        if self.dry_run:
            print("🔍 MODO DRY RUN - Nenhuma alteração será feita")
        
        # 1. Encontrar arquivos
        print("\n1. Procurando arquivos para atualizar...")
        files = self.find_files_to_update()
        print(f"   Encontrados {len(files)} arquivos")
        
        if not files:
            print("\n✅ Nenhum arquivo precisa ser atualizado!")
            return
        
        # 2. Fazer backup
        print(f"\n2. Criando backup dos arquivos...")
        self.backup_files(files)
        
        # 3. Atualizar arquivos
        print(f"\n3. Atualizando arquivos...")
        total_changes = 0
        for file in files:
            changes = self.update_file(file)
            total_changes += len(changes)
            self.changes_made.extend([(file, change) for change in changes])
        
        # 4. Criar script de validação
        print(f"\n4. Criando script de validação...")
        self.create_validation_script()
        
        # 5. Resumo
        print("\n" + "=" * 60)
        print("RESUMO DA MIGRAÇÃO")
        print("=" * 60)
        print(f"Arquivos processados: {len(files)}")
        print(f"Total de mudanças: {total_changes}")
        
        if not self.dry_run:
            print("\n✅ Migração concluída!")
            print("\n📋 Próximos passos:")
            print("1. Execute: python validate_migration.py")
            print("2. Execute os testes: pytest tests/test_profitdll_v4_compatibility.py")
            print("3. Teste o sistema em ambiente de desenvolvimento")
        else:
            print("\n🔍 Análise concluída (dry run)")
            print("Execute sem --dry-run para aplicar as mudanças")


def main():
    parser = argparse.ArgumentParser(description='Migração para ProfitDLL v4.0.0.30')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Simula a migração sem fazer alterações')
    parser.add_argument('--backup-dir', default='backup_pre_v4',
                       help='Diretório para backup (default: backup_pre_v4)')
    
    args = parser.parse_args()
    
    # Verificar se está no diretório correto
    if not os.path.exists('src'):
        print("❌ Erro: Execute este script do diretório raiz do projeto")
        return 1
    
    # Executar migração
    migrator = ProfitDLLMigrator(
        backup_dir=args.backup_dir,
        dry_run=args.dry_run
    )
    
    try:
        migrator.run()
        return 0
    except Exception as e:
        print(f"\n❌ Erro durante migração: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())