"""
Script para criar estrutura de diretórios necessária para o sistema
"""

import os
from pathlib import Path
import json
from datetime import datetime


def setup_directories():
    """
    Cria toda a estrutura de diretórios necessária
    """
    print(f"\n{'='*60}")
    print("CONFIGURAÇÃO DE DIRETÓRIOS - QuantumTrader ML")
    print(f"{'='*60}\n")
    
    # Definir estrutura
    directories = {
        # Dados
        'data/historical': 'Dados históricos tick-a-tick',
        'data/historical/WDOU25': 'Dados históricos WDO',
        'data/realtime/book': 'Dados de book em tempo real',
        'data/realtime/trades': 'Trades em tempo real',
        'data/processed': 'Dados processados',
        'data/cache': 'Cache de features',
        
        # Modelos
        'models/tick_only': 'Modelos treinados com dados tick',
        'models/tick_only/WDOU25/trend_up': 'Modelos regime alta',
        'models/tick_only/WDOU25/trend_down': 'Modelos regime baixa',
        'models/tick_only/WDOU25/range': 'Modelos regime lateral',
        'models/book_enhanced': 'Modelos com dados de book',
        'models/book_enhanced/WDOU25': 'Modelos book WDO',
        'models/ensemble': 'Modelos ensemble',
        'models/production': 'Modelos em produção',
        
        # Logs e relatórios
        'logs': 'Logs do sistema',
        'logs/training': 'Logs de treinamento',
        'logs/trading': 'Logs de trading',
        'logs/data_collection': 'Logs de coleta',
        'reports/training': 'Relatórios de treinamento',
        'reports/backtest': 'Relatórios de backtest',
        'reports/daily': 'Relatórios diários',
        
        # Configurações
        'config': 'Arquivos de configuração',
        'config/features': 'Configurações de features',
        'config/models': 'Configurações de modelos',
        'config/trading': 'Configurações de trading',
        
        # Backups
        'backups': 'Backups do sistema',
        'backups/models': 'Backup de modelos',
        'backups/config': 'Backup de configurações',
        
        # Temporários
        'tmp': 'Arquivos temporários',
        'tmp/cache': 'Cache temporário',
        'tmp/downloads': 'Downloads temporários',
        
        # Scripts e notebooks
        'notebooks': 'Jupyter notebooks',
        'notebooks/analysis': 'Análises exploratórias',
        'notebooks/experiments': 'Experimentos',
        
        # Testes
        'tests/data': 'Dados de teste',
        'tests/fixtures': 'Fixtures de teste',
        'tests/results': 'Resultados de teste'
    }
    
    created = 0
    existing = 0
    
    print("📁 Criando estrutura de diretórios...\n")
    
    for dir_path, description in directories.items():
        path = Path(dir_path)
        
        if path.exists():
            status = "✓ Existe"
            existing += 1
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                status = "✅ Criado"
                created += 1
            except Exception as e:
                status = f"❌ Erro: {e}"
                
        print(f"{status:12} {dir_path:45} # {description}")
        
    # Criar arquivos de configuração padrão se não existirem
    print(f"\n📄 Criando arquivos de configuração padrão...\n")
    
    config_files = {
        'config/training_config.json': {
            'description': 'Configuração de treinamento',
            'content': {
                'tick_only': {
                    'lookback_days': 365,
                    'validation_method': 'walk_forward',
                    'test_size': 0.2,
                    'n_splits': 5
                },
                'book_enhanced': {
                    'lookback_days': 30,
                    'targets': ['spread_next', 'imbalance_direction', 'price_move_5s'],
                    'min_samples': 1000
                },
                'features': {
                    'technical_indicators': True,
                    'microstructure': True,
                    'regime_features': True
                }
            }
        },
        'config/features/all_required_features.json': {
            'description': 'Lista de features necessárias',
            'content': {
                'basic_features': [
                    'price', 'volume', 'returns', 'log_returns',
                    'sma_5', 'sma_10', 'sma_20', 'ema_9', 'ema_21',
                    'rsi_14', 'macd', 'macd_signal', 'atr_14', 'adx_14'
                ],
                'book_features': [
                    'spread', 'book_imbalance', 'bid_depth_1', 'ask_depth_1',
                    'micro_price', 'kyle_lambda', 'amihud_illiquidity'
                ],
                'flow_features': [
                    'order_flow_imbalance', 'volume_imbalance',
                    'buy_pressure', 'sell_pressure', 'tape_speed'
                ]
            }
        },
        'config/trading/risk_limits.json': {
            'description': 'Limites de risco',
            'content': {
                'max_position': 10,
                'max_daily_loss': 5000.0,
                'max_drawdown': 0.10,
                'position_sizing': {
                    'method': 'fixed',
                    'base_size': 1,
                    'max_size': 5
                },
                'stop_loss': {
                    'enabled': True,
                    'type': 'percentage',
                    'value': 0.02
                }
            }
        }
    }
    
    for file_path, file_info in config_files.items():
        path = Path(file_path)
        
        if path.exists():
            print(f"✓ Existe     {file_path:45} # {file_info['description']}")
        else:
            try:
                with open(path, 'w') as f:
                    json.dump(file_info['content'], f, indent=2)
                print(f"✅ Criado     {file_path:45} # {file_info['description']}")
            except Exception as e:
                print(f"❌ Erro       {file_path:45} # {e}")
                
    # Criar README.md nos diretórios principais
    readme_dirs = {
        'data': "# Diretório de Dados\n\nContém todos os dados do sistema.",
        'models': "# Diretório de Modelos\n\nContém modelos treinados de ML.",
        'logs': "# Diretório de Logs\n\nLogs de execução do sistema.",
        'reports': "# Diretório de Relatórios\n\nRelatórios de análise e performance."
    }
    
    print(f"\n📝 Criando arquivos README...\n")
    
    for dir_name, content in readme_dirs.items():
        readme_path = Path(dir_name) / 'README.md'
        if not readme_path.exists():
            try:
                with open(readme_path, 'w') as f:
                    f.write(content)
                print(f"✅ Criado     {readme_path}")
            except Exception as e:
                print(f"❌ Erro       {readme_path} # {e}")
                
    # Criar arquivo .gitignore se não existir
    gitignore_path = Path('.gitignore')
    if not gitignore_path.exists():
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env

# Dados
data/historical/
data/realtime/
*.parquet
*.csv
*.h5

# Modelos
models/*/
*.pkl
*.joblib
*.h5
*.pt
*.pth

# Logs
logs/
*.log

# Temporários
tmp/
cache/
.ipynb_checkpoints/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Secrets
config/credentials.json
config/api_keys.json
"""
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print(f"\n✅ Criado .gitignore")
        
    # Resumo final
    print(f"\n{'='*60}")
    print(f"RESUMO:")
    print(f"  - Diretórios criados: {created}")
    print(f"  - Diretórios existentes: {existing}")
    print(f"  - Total de diretórios: {created + existing}")
    print(f"\n✅ Estrutura de diretórios configurada com sucesso!")
    print(f"{'='*60}\n")
    
    # Salvar timestamp da configuração
    setup_info = {
        'timestamp': datetime.now().isoformat(),
        'directories_created': created,
        'directories_total': created + existing,
        'structure': list(directories.keys())
    }
    
    with open('config/setup_info.json', 'w') as f:
        json.dump(setup_info, f, indent=2)


if __name__ == "__main__":
    setup_directories()