"""
Script de Teste de Saúde do Sistema
Data: 03/08/2025

Verifica se todos os componentes principais estão funcionando após a limpeza
"""

import os
import sys
import time
import importlib
import subprocess
from datetime import datetime
from pathlib import Path
import json

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class SystemHealthChecker:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'warnings': 0
            }
        }
        
    def test(self, name, func, critical=True):
        """Executa um teste e registra resultado"""
        print(f"\nTestando: {name}")
        self.results['summary']['total'] += 1
        
        try:
            result, message = func()
            if result:
                print(f"   [OK] PASSOU: {message}")
                self.results['tests'][name] = {'status': 'passed', 'message': message}
                self.results['summary']['passed'] += 1
            else:
                if critical:
                    print(f"   [ERRO] FALHOU: {message}")
                    self.results['tests'][name] = {'status': 'failed', 'message': message}
                    self.results['summary']['failed'] += 1
                else:
                    print(f"   [AVISO] AVISO: {message}")
                    self.results['tests'][name] = {'status': 'warning', 'message': message}
                    self.results['summary']['warnings'] += 1
        except Exception as e:
            print(f"   [ERRO] ERRO: {str(e)}")
            self.results['tests'][name] = {'status': 'error', 'message': str(e)}
            self.results['summary']['failed'] += 1
    
    def test_imports(self):
        """Testa imports principais"""
        def check():
            modules = [
                'src.connection_manager_v4',
                'src.trading_system',
                'src.order_manager',
                'src.database.historical_data_collector',
                'src.database.realtime_book_collector',
                'src.ml_coordinator',
                'src.model_manager',
                'src.feature_engine'
            ]
            
            failed = []
            for module in modules:
                try:
                    importlib.import_module(module)
                except Exception as e:
                    failed.append(f"{module}: {str(e)}")
            
            if failed:
                return False, f"Falha ao importar: {', '.join(failed[:3])}"
            return True, f"Todos os {len(modules)} módulos principais importados"
        
        return check()
    
    def test_directories(self):
        """Verifica estrutura de diretórios"""
        def check():
            required_dirs = [
                'src',
                'src/database',
                'src/integration',
                'src/training',
                'data',
                'models',
                'logs',
                'scripts'
            ]
            
            missing = []
            for dir_path in required_dirs:
                if not os.path.exists(dir_path):
                    missing.append(dir_path)
            
            if missing:
                return False, f"Diretórios faltando: {', '.join(missing)}"
            return True, "Estrutura de diretórios intacta"
        
        return check()
    
    def test_config_files(self):
        """Verifica arquivos de configuração"""
        def check():
            config_files = [
                '.env',
                'requirements.txt',
                'CLAUDE.md'
            ]
            
            missing = []
            for file in config_files:
                if not os.path.exists(file):
                    missing.append(file)
            
            # .env é crítico
            if '.env' in missing:
                return False, ".env não encontrado - crítico para conexão"
            elif missing:
                return True, f"Alguns configs opcionais faltando: {', '.join(missing)}"
            return True, "Todos arquivos de configuração presentes"
        
        return check()
    
    def test_profit_dll(self):
        """Verifica se ProfitDLL está acessível"""
        def check():
            dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
            if not os.path.exists(dll_path):
                return False, f"ProfitDLL não encontrada em {dll_path}"
            
            # Tentar importar ConnectionManager
            try:
                from src.connection_manager_v4 import ConnectionManagerV4
                return True, "ProfitDLL encontrada e ConnectionManagerV4 importado"
            except Exception as e:
                return False, f"Erro ao importar ConnectionManager: {str(e)}"
        
        return check()
    
    def test_server_script(self):
        """Verifica se servidor isolado existe"""
        def check():
            server_path = "src/integration/profit_dll_server.py"
            if not os.path.exists(server_path):
                return False, "Script do servidor isolado não encontrado"
            
            # Verificar se pode ser importado
            try:
                spec = importlib.util.spec_from_file_location("profit_dll_server", server_path)
                module = importlib.util.module_from_spec(spec)
                return True, "Servidor isolado disponível"
            except Exception as e:
                return False, f"Erro ao verificar servidor: {str(e)}"
        
        return check()
    
    def test_historical_data(self):
        """Verifica dados históricos coletados"""
        def check():
            data_dir = Path("data/historical")
            if not data_dir.exists():
                return False, "Diretório de dados históricos não existe"
            
            # Procurar por arquivos parquet
            parquet_files = list(data_dir.rglob("*.parquet"))
            if not parquet_files:
                return True, "Nenhum dado histórico encontrado (executar coleta)"
            
            # Contar arquivos e calcular tamanho
            total_size = sum(f.stat().st_size for f in parquet_files) / (1024 * 1024)
            return True, f"Encontrados {len(parquet_files)} arquivos históricos ({total_size:.1f}MB)"
        
        return check()
    
    def test_models(self):
        """Verifica modelos ML"""
        def check():
            model_dir = Path("models")
            if not model_dir.exists():
                return False, "Diretório de modelos não existe"
            
            # Procurar modelos
            pkl_files = list(model_dir.glob("*.pkl"))
            if not pkl_files:
                return True, "Nenhum modelo treinado (executar treinamento)"
            
            # Verificar modelos v4
            v4_models = [f for f in pkl_files if 'v4' in f.name or 'v3' not in f.name]
            if v4_models:
                return True, f"Encontrados {len(v4_models)} modelos compatíveis"
            else:
                return False, "Apenas modelos v3 encontrados - retreinar necessário"
        
        return check()
    
    def test_zmq_valkey(self):
        """Verifica componentes ZMQ/Valkey"""
        def check():
            # Verificar se módulos existem
            zmq_files = [
                'src/integration/zmq_valkey_bridge.py',
                'src/integration/zmq_publisher_wrapper.py',
                'src/infrastructure/valkey_connection.py'
            ]
            
            missing = [f for f in zmq_files if not os.path.exists(f)]
            if missing:
                return True, f"Componentes ZMQ/Valkey opcionais não encontrados"
            
            # Tentar importar
            try:
                import zmq
                import valkey
                return True, "ZMQ e Valkey disponíveis para uso"
            except ImportError:
                return True, "ZMQ/Valkey não instalados (opcional)"
        
        return check()
    
    def test_book_collector(self):
        """Verifica sistema de coleta de book"""
        def check():
            files = [
                'src/database/realtime_book_collector.py',
                'scripts/book_collector.py',
                'scripts/test_book_collection.py'
            ]
            
            missing = [f for f in files if not os.path.exists(f)]
            if missing:
                return False, f"Arquivos de book faltando: {', '.join(missing)}"
            
            # Verificar se pode importar
            try:
                from src.database.realtime_book_collector import RealtimeBookCollector
                return True, "Sistema de coleta de book operacional"
            except Exception as e:
                return False, f"Erro ao importar book collector: {str(e)}"
        
        return check()
    
    def test_scripts(self):
        """Verifica scripts essenciais"""
        def check():
            essential_scripts = [
                'scripts/start_historical_collection.py',
                'scripts/view_historical_data.py',
                'scripts/book_collector.py',
                'test_connection.py'
            ]
            
            missing = []
            for script in essential_scripts:
                if not os.path.exists(script):
                    missing.append(script)
            
            if missing:
                return False, f"Scripts essenciais faltando: {', '.join(missing)}"
            return True, f"Todos os {len(essential_scripts)} scripts essenciais presentes"
        
        return check()
    
    def test_python_env(self):
        """Verifica ambiente Python"""
        def check():
            # Verificar versão Python
            version = sys.version_info
            if version.major != 3 or version.minor < 8:
                return False, f"Python {version.major}.{version.minor} - requer 3.8+"
            
            # Verificar pacotes essenciais
            required = ['pandas', 'numpy', 'pyarrow']
            missing = []
            
            for package in required:
                try:
                    importlib.import_module(package)
                except ImportError:
                    missing.append(package)
            
            if missing:
                return False, f"Pacotes faltando: {', '.join(missing)}"
            
            return True, f"Python {version.major}.{version.minor}.{version.micro} com pacotes essenciais"
        
        return check()
    
    def generate_report(self):
        """Gera relatório de saúde"""
        # Salvar JSON
        report_path = f"system_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Mostrar resumo
        print("\n" + "="*60)
        print("RESUMO DA SAUDE DO SISTEMA")
        print("="*60)
        
        summary = self.results['summary']
        total = summary['total']
        passed = summary['passed']
        failed = summary['failed']
        warnings = summary['warnings']
        
        # Calcular saúde
        if total > 0:
            health_score = (passed / total) * 100
        else:
            health_score = 0
        
        print(f"Total de testes: {total}")
        print(f"[OK] Passou: {passed}")
        print(f"[ERRO] Falhou: {failed}")
        print(f"[AVISO] Avisos: {warnings}")
        print(f"\nSaude do Sistema: {health_score:.1f}%")
        
        # Status geral
        if failed == 0 and warnings == 0:
            print("\n*** SISTEMA 100% OPERACIONAL! ***")
        elif failed == 0:
            print("\n[OK] Sistema operacional com pequenos avisos")
        elif failed <= 2:
            print("\n[AVISO] Sistema parcialmente operacional - correcoes necessarias")
        else:
            print("\n[ERRO] Sistema com problemas criticos - acao imediata necessaria")
        
        print(f"\nRelatório detalhado salvo em: {report_path}")
        
        # Mostrar ações recomendadas
        if failed > 0 or warnings > 0:
            print("\nACOES RECOMENDADAS:")
            
            # Verificar problemas específicos
            tests = self.results['tests']
            
            if 'Arquivos de configuração' in tests and tests['Arquivos de configuração']['status'] == 'failed':
                print("1. Criar arquivo .env com credenciais do ProfitDLL")
            
            if 'Modelos ML' in tests and tests['Modelos ML']['status'] != 'passed':
                print("2. Executar treinamento de modelos: python create_models.py")
            
            if 'Dados históricos' in tests and 'Nenhum dado' in tests['Dados históricos']['message']:
                print("3. Coletar dados históricos: python scripts/start_historical_collection.py")
            
            if any('faltando' in t.get('message', '') for t in tests.values() if t['status'] == 'failed'):
                print("4. Verificar se a limpeza removeu arquivos essenciais")


def main():
    print("""
    ========================================================
            TESTE DE SAUDE DO SISTEMA QUANTUMTRADER        
    ========================================================
    
    Verificando integridade do sistema apos limpeza...
    """)
    
    checker = SystemHealthChecker()
    
    # Executar testes
    print("VERIFICANDO COMPONENTES PRINCIPAIS\n")
    
    # Testes críticos
    checker.test("Ambiente Python", checker.test_python_env, critical=True)
    checker.test("Imports principais", checker.test_imports, critical=True)
    checker.test("Estrutura de diretórios", checker.test_directories, critical=True)
    checker.test("Arquivos de configuração", checker.test_config_files, critical=True)
    checker.test("ProfitDLL", checker.test_profit_dll, critical=True)
    checker.test("Servidor isolado", checker.test_server_script, critical=True)
    checker.test("Scripts essenciais", checker.test_scripts, critical=True)
    
    # Testes não-críticos
    checker.test("Dados históricos", checker.test_historical_data, critical=False)
    checker.test("Modelos ML", checker.test_models, critical=False)
    checker.test("Sistema de Book", checker.test_book_collector, critical=True)
    checker.test("ZMQ/Valkey", checker.test_zmq_valkey, critical=False)
    
    # Gerar relatório
    checker.generate_report()


if __name__ == "__main__":
    main()