"""
Script de Verificação - Sistema pronto para produção?
Executa uma série de testes para validar se o sistema está pronto
"""

import os
import sys
from pathlib import Path
import json
import importlib
import socket
from datetime import datetime

# Cores para output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")

def print_success(text):
    print(f"{Colors.GREEN}[OK] {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}[WARN] {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}[ERRO] {text}{Colors.END}")

def check_environment():
    """Verifica variáveis de ambiente"""
    print_header("1. VERIFICANDO AMBIENTE")
    
    issues = []
    
    # Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print_success(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print_error(f"Python {python_version.major}.{python_version.minor} (mínimo 3.8)")
        issues.append("Python version")
    
    # Credenciais
    if os.getenv('PROFIT_USER') and os.getenv('PROFIT_PASS'):
        print_success("Credenciais ProfitDLL configuradas")
    else:
        print_warning("Credenciais ProfitDLL não encontradas (configure .env)")
        # Não é um erro crítico pois pode estar no .env
    
    return issues

def check_dependencies():
    """Verifica dependências instaladas"""
    print_header("2. VERIFICANDO DEPENDÊNCIAS")
    
    issues = []
    
    required_packages = [
        'pandas', 'numpy', 'lightgbm', 'tensorflow', 
        'flask', 'plotly', 'pytest', 'valkey',
        'zmq', 'psutil', 'dask'
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print_success(f"{package} instalado")
        except ImportError:
            print_error(f"{package} NÃO instalado")
            issues.append(package)
    
    return issues

def check_models():
    """Verifica se modelos existem"""
    print_header("3. VERIFICANDO MODELOS ML")
    
    issues = []
    
    models_to_check = [
        ('models/csv_5m/lightgbm_tick.txt', 'Modelo Tick'),
        ('models/csv_5m/scaler_tick.pkl', 'Scaler Tick'),
        ('models/csv_5m/features_tick.json', 'Features Tick'),
        ('models/book_only/lightgbm_book_only_optimized.txt', 'Modelo Book'),
        ('models/book_only/scaler_book_only.pkl', 'Scaler Book'),
        ('models/book_only/features_book_only.json', 'Features Book'),
    ]
    
    for model_path, model_name in models_to_check:
        if Path(model_path).exists():
            print_success(f"{model_name}: {model_path}")
        else:
            print_error(f"{model_name} NÃO encontrado: {model_path}")
            issues.append(model_name)
    
    return issues

def check_directories():
    """Verifica estrutura de diretórios"""
    print_header("4. VERIFICANDO ESTRUTURA DE DIRETÓRIOS")
    
    dirs_to_create = [
        'logs',
        'logs/production',
        'data',
        'data/checkpoints',
        'reports',
        'alerts',
        'database'
    ]
    
    for dir_path in dirs_to_create:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print_success(f"Criado: {dir_path}")
        else:
            print_success(f"Existe: {dir_path}")
    
    return []

def check_ports():
    """Verifica portas disponíveis"""
    print_header("5. VERIFICANDO PORTAS")
    
    issues = []
    
    ports_to_check = [
        (5000, "Dashboard Web"),
        (8001, "ProfitDLL"),
        (6379, "Valkey/Redis"),
        (5555, "ZMQ Publisher")
    ]
    
    for port, service in ports_to_check:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        
        if result == 0:
            if port == 8001:  # ProfitDLL deve estar rodando
                print_success(f"Porta {port} ({service}) - EM USO (esperado)")
            else:
                print_warning(f"Porta {port} ({service}) - EM USO")
        else:
            if port == 8001:
                print_warning(f"Porta {port} ({service}) - LIVRE (ProfitChart não está rodando?)")
            else:
                print_success(f"Porta {port} ({service}) - LIVRE")
    
    return issues

def check_imports():
    """Testa imports principais"""
    print_header("6. TESTANDO IMPORTS PRINCIPAIS")
    
    issues = []
    
    imports_to_test = [
        ('src.trading_system', 'TradingSystem'),
        ('src.strategies.hybrid_strategy', 'HybridStrategy'),
        ('src.risk.risk_manager', 'RiskManager'),
        ('src.execution.order_manager', 'OrderManager'),
        ('src.portfolio.position_tracker', 'PositionTracker'),
        ('src.data.data_synchronizer', 'DataSynchronizer'),
        ('src.connection_manager_v4', 'ConnectionManagerV4'),
    ]
    
    for module_name, class_name in imports_to_test:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                print_success(f"{class_name} de {module_name}")
            else:
                print_error(f"{class_name} não encontrado em {module_name}")
                issues.append(f"{module_name}.{class_name}")
        except Exception as e:
            print_error(f"Erro ao importar {module_name}: {str(e)[:50]}...")
            issues.append(module_name)
    
    return issues

def run_quick_test():
    """Executa teste rápido de componentes"""
    print_header("7. TESTE RÁPIDO DE COMPONENTES")
    
    issues = []
    
    try:
        # Testar criação da estratégia
        from src.strategies.hybrid_strategy import HybridStrategy
        strategy = HybridStrategy({})
        strategy.load_models()
        
        if strategy.tick_model and strategy.book_model:
            print_success("Modelos carregados com sucesso")
        else:
            print_warning("Modelos não carregados completamente")
            
    except Exception as e:
        print_error(f"Erro ao testar estratégia: {str(e)[:100]}...")
        issues.append("Strategy test")
    
    try:
        # Testar risk manager
        from src.risk.risk_manager import RiskManager
        risk = RiskManager({'initial_capital': 50000})
        print_success("RiskManager criado")
    except Exception as e:
        print_error(f"Erro ao criar RiskManager: {str(e)[:50]}...")
        issues.append("RiskManager")
    
    return issues

def generate_report(all_issues):
    """Gera relatório final"""
    print_header("RELATÓRIO FINAL")
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\nData: {timestamp}")
    
    if not any(all_issues.values()):
        print(f"\n{Colors.GREEN}>>> SISTEMA PRONTO PARA PRODUCAO! <<<{Colors.END}")
        print("\nPróximos passos:")
        print("1. Configure suas credenciais no arquivo .env")
        print("2. Certifique-se que o ProfitChart está aberto")
        print("3. Execute: python start_production.py")
        
        return True
    else:
        print(f"\n{Colors.RED}!!! SISTEMA PRECISA DE AJUSTES !!!{Colors.END}")
        print("\nProblemas encontrados:")
        
        for category, issues in all_issues.items():
            if issues:
                print(f"\n{category}:")
                for issue in issues:
                    print(f"  - {issue}")
        
        print("\nResolva os problemas acima antes de iniciar em produção.")
        
        return False

def main():
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}VERIFICAÇÃO DE PRONTIDÃO - QUANTUMTRADER ML{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    
    all_issues = {}
    
    # Executar verificações
    all_issues['environment'] = check_environment()
    all_issues['dependencies'] = check_dependencies()
    all_issues['models'] = check_models()
    all_issues['directories'] = check_directories()
    all_issues['ports'] = check_ports()
    all_issues['imports'] = check_imports()
    all_issues['components'] = run_quick_test()
    
    # Gerar relatório
    ready = generate_report(all_issues)
    
    # Salvar relatório
    report = {
        'timestamp': datetime.now().isoformat(),
        'ready': ready,
        'issues': all_issues,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }
    
    report_file = f"production_ready_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nRelatorio salvo em: {report_file}")
    
    return 0 if ready else 1

if __name__ == "__main__":
    sys.exit(main())