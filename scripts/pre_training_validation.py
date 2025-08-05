"""
Script de validação completa pré-treinamento
Executa todas as verificações necessárias antes de iniciar o treinamento
"""

import sys
import os
import subprocess
import json
import psutil
import platform
from datetime import datetime
from pathlib import Path
import importlib.util

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PreTrainingValidator:
    """
    Validador completo para verificar se o sistema está pronto para treinamento
    """
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.info = []
        self.checks_passed = 0
        self.checks_total = 0
        
    def run_all_checks(self, symbol='WDOU25'):
        """Executa todas as verificações"""
        print("\n" + "="*70)
        print("🔍 VALIDAÇÃO PRÉ-TREINAMENTO - QuantumTrader ML v2.0")
        print("="*70)
        print(f"\nSímbolo: {symbol}")
        print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nIniciando verificações...\n")
        
        # 1. Verificar ambiente Python
        self.check_python_environment()
        
        # 2. Verificar estrutura de diretórios
        self.check_directory_structure()
        
        # 3. Verificar dados históricos
        self.check_historical_data(symbol)
        
        # 4. Verificar dados de book
        self.check_book_data(symbol)
        
        # 5. Verificar dependências
        self.check_dependencies()
        
        # 6. Verificar recursos do sistema
        self.check_system_resources()
        
        # 7. Verificar configurações
        self.check_configurations()
        
        # 8. Verificar infraestrutura HMARL (opcional)
        self.check_hmarl_infrastructure()
        
        # 9. Gerar relatório
        self.generate_report()
        
        return self.checks_passed == self.checks_total
        
    def check_python_environment(self):
        """Verifica ambiente Python"""
        self.checks_total += 1
        print("🐍 Verificando ambiente Python...")
        
        try:
            # Versão do Python
            python_version = sys.version_info
            if python_version.major == 3 and python_version.minor >= 8:
                self.info.append(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
                self.checks_passed += 1
                print("   ✅ Versão Python compatível")
            else:
                self.issues.append(f"Python {python_version.major}.{python_version.minor} - necessário 3.8+")
                print("   ❌ Versão Python incompatível")
                
            # Virtual environment
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                self.info.append("Virtual environment ativo")
                print("   ✅ Virtual environment detectado")
            else:
                self.warnings.append("Recomenda-se usar virtual environment")
                print("   ⚠️  Virtual environment não detectado")
                
        except Exception as e:
            self.issues.append(f"Erro ao verificar Python: {e}")
            print(f"   ❌ Erro: {e}")
            
    def check_directory_structure(self):
        """Verifica estrutura de diretórios"""
        self.checks_total += 1
        print("\n📁 Verificando estrutura de diretórios...")
        
        required_dirs = [
            'data/historical',
            'data/realtime/book',
            'models/tick_only',
            'models/book_enhanced',
            'config',
            'logs',
            'reports'
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing_dirs.append(dir_path)
                
        if missing_dirs:
            self.issues.append(f"Diretórios faltando: {missing_dirs}")
            print(f"   ❌ {len(missing_dirs)} diretórios faltando")
            print("   💡 Execute: python scripts/setup_directories.py")
        else:
            self.checks_passed += 1
            print("   ✅ Estrutura de diretórios OK")
            
    def check_historical_data(self, symbol):
        """Verifica dados históricos"""
        self.checks_total += 1
        print(f"\n📊 Verificando dados históricos para {symbol}...")
        
        try:
            # Executar script de verificação
            result = subprocess.run(
                [sys.executable, 'scripts/check_historical_data.py', '--symbol', symbol, '--days', '365'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.checks_passed += 1
                print("   ✅ Dados históricos disponíveis")
                
                # Extrair informações do output
                for line in result.stdout.split('\n'):
                    if 'Total de registros:' in line:
                        self.info.append(line.strip())
                    elif 'Cobertura:' in line:
                        self.info.append(line.strip())
            else:
                self.issues.append("Dados históricos insuficientes")
                print("   ❌ Dados históricos insuficientes")
                
        except FileNotFoundError:
            self.warnings.append("Script check_historical_data.py não encontrado")
            print("   ⚠️  Não foi possível verificar dados históricos")
            
    def check_book_data(self, symbol):
        """Verifica dados de book"""
        self.checks_total += 1
        print(f"\n📚 Verificando dados de book para {symbol}...")
        
        try:
            # Executar script de verificação
            result = subprocess.run(
                [sys.executable, 'scripts/check_book_data.py', '--symbol', symbol, '--days', '30'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.checks_passed += 1
                print("   ✅ Dados de book disponíveis")
            else:
                self.warnings.append("Dados de book não disponíveis - modelos book-enhanced não poderão ser treinados")
                print("   ⚠️  Dados de book não disponíveis")
                self.checks_passed += 1  # Não é crítico
                
        except FileNotFoundError:
            self.warnings.append("Script check_book_data.py não encontrado")
            print("   ⚠️  Não foi possível verificar dados de book")
            self.checks_passed += 1  # Não é crítico
            
    def check_dependencies(self):
        """Verifica dependências Python"""
        self.checks_total += 1
        print("\n📦 Verificando dependências...")
        
        required_packages = {
            'pandas': '1.3.0',
            'numpy': '1.21.0',
            'scikit-learn': '1.0.0',
            'xgboost': '1.7.0',
            'lightgbm': '3.3.0',
            'pyzmq': '22.0.0',
            'joblib': '1.0.0'
        }
        
        missing_packages = []
        wrong_version = []
        
        for package, min_version in required_packages.items():
            spec = importlib.util.find_spec(package)
            if spec is None:
                missing_packages.append(package)
            else:
                try:
                    module = importlib.import_module(package)
                    if hasattr(module, '__version__'):
                        version = module.__version__
                        # Comparação simplificada de versão
                        if version < min_version:
                            wrong_version.append(f"{package} ({version} < {min_version})")
                except:
                    pass
                    
        if missing_packages:
            self.issues.append(f"Pacotes faltando: {missing_packages}")
            print(f"   ❌ {len(missing_packages)} pacotes faltando")
        elif wrong_version:
            self.warnings.append(f"Versões desatualizadas: {wrong_version}")
            print(f"   ⚠️  {len(wrong_version)} pacotes com versão antiga")
            self.checks_passed += 1
        else:
            self.checks_passed += 1
            print("   ✅ Todas as dependências instaladas")
            
    def check_system_resources(self):
        """Verifica recursos do sistema"""
        self.checks_total += 1
        print("\n💻 Verificando recursos do sistema...")
        
        # CPU
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memória
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        # Disco
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024**3)
        
        self.info.append(f"CPU: {cpu_count} cores, {cpu_percent}% em uso")
        self.info.append(f"RAM: {memory_gb:.1f} GB total, {memory_available_gb:.1f} GB disponível")
        self.info.append(f"Disco: {disk_free_gb:.1f} GB livre")
        
        # Verificar mínimos
        issues = []
        if cpu_count < 4:
            issues.append("CPU com menos de 4 cores")
        if memory_gb < 8:
            issues.append("Menos de 8 GB de RAM")
        if disk_free_gb < 20:
            issues.append("Menos de 20 GB de espaço livre")
            
        if issues:
            self.warnings.extend(issues)
            print(f"   ⚠️  Recursos limitados: {', '.join(issues)}")
            self.checks_passed += 1  # Não é crítico
        else:
            self.checks_passed += 1
            print("   ✅ Recursos adequados")
            
        print(f"      CPU: {cpu_count} cores")
        print(f"      RAM: {memory_gb:.1f} GB")
        print(f"      Disco: {disk_free_gb:.1f} GB livre")
        
    def check_configurations(self):
        """Verifica arquivos de configuração"""
        self.checks_total += 1
        print("\n⚙️  Verificando configurações...")
        
        config_files = [
            'config/training_config.json',
            'config/features/all_required_features.json',
            'config/trading/risk_limits.json'
        ]
        
        missing_configs = []
        for config_file in config_files:
            if not Path(config_file).exists():
                missing_configs.append(config_file)
                
        if missing_configs:
            self.warnings.append(f"Configurações faltando: {missing_configs}")
            print(f"   ⚠️  {len(missing_configs)} arquivos de configuração faltando")
            print("   💡 Execute: python scripts/setup_directories.py")
            self.checks_passed += 1  # Não é crítico, serão criados defaults
        else:
            self.checks_passed += 1
            print("   ✅ Arquivos de configuração OK")
            
    def check_hmarl_infrastructure(self):
        """Verifica infraestrutura HMARL (opcional)"""
        self.checks_total += 1
        print("\n🚀 Verificando infraestrutura HMARL...")
        
        try:
            # Verificar Valkey/Redis
            import redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            self.info.append("Valkey/Redis disponível")
            print("   ✅ Valkey/Redis conectado")
            hmarl_available = True
        except:
            self.warnings.append("HMARL não disponível - sistema funcionará sem análise de fluxo")
            print("   ⚠️  Valkey/Redis não disponível")
            hmarl_available = False
            
        # Verificar portas ZMQ
        if hmarl_available:
            import socket
            zmq_ports = [5555, 5556, 5557, 5558, 5559, 5560]
            blocked_ports = []
            
            for port in zmq_ports:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                if result == 0:
                    blocked_ports.append(port)
                    
            if blocked_ports:
                self.warnings.append(f"Portas ZMQ em uso: {blocked_ports}")
                print(f"   ⚠️  Algumas portas ZMQ em uso: {blocked_ports}")
                
        self.checks_passed += 1  # HMARL é opcional
        
    def generate_report(self):
        """Gera relatório final"""
        print("\n" + "="*70)
        print("📋 RELATÓRIO DE VALIDAÇÃO")
        print("="*70)
        
        # Status geral
        all_passed = self.checks_passed == self.checks_total
        status = "✅ SISTEMA PRONTO PARA TREINAMENTO" if all_passed else "❌ CORREÇÕES NECESSÁRIAS"
        
        print(f"\nStatus: {status}")
        print(f"Verificações: {self.checks_passed}/{self.checks_total} aprovadas")
        
        # Problemas críticos
        if self.issues:
            print(f"\n❌ Problemas Críticos ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   • {issue}")
                
        # Avisos
        if self.warnings:
            print(f"\n⚠️  Avisos ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
                
        # Informações
        if self.info:
            print(f"\nℹ️  Informações do Sistema:")
            for info in self.info:
                print(f"   • {info}")
                
        # Próximos passos
        print("\n📌 Próximos Passos:")
        
        if self.issues:
            print("   1. Corrigir os problemas críticos listados acima")
            print("   2. Executar novamente este script de validação")
        else:
            print("   1. Revisar avisos (opcional)")
            print("   2. Executar treinamento: python examples/train_dual_models.py")
            
        # Salvar relatório
        report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'ready' if all_passed else 'not_ready',
            'checks_passed': self.checks_passed,
            'checks_total': self.checks_total,
            'issues': self.issues,
            'warnings': self.warnings,
            'info': self.info,
            'system': {
                'platform': platform.platform(),
                'python': sys.version,
                'cpu_cores': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3)
            }
        }
        
        report_path = Path('reports/pre_training_validation.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n💾 Relatório salvo em: {report_path}")
        print("="*70)
        
        return all_passed


def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validação pré-treinamento')
    parser.add_argument('--symbol', type=str, default='WDOU25', help='Símbolo para validar')
    
    args = parser.parse_args()
    
    validator = PreTrainingValidator()
    ready = validator.run_all_checks(args.symbol)
    
    # Exit code
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()