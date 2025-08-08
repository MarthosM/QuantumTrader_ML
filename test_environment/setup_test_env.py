"""
Setup de Ambiente de Teste Isolado
Cria estrutura para testar o sistema de 65 features sem afetar produção
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

def setup_test_environment():
    """Configura ambiente de teste isolado"""
    
    print("=" * 60)
    print("SETUP DE AMBIENTE DE TESTE ISOLADO")
    print("=" * 60)
    
    # 1. Criar estrutura de diretórios
    base_dir = Path("test_environment")
    dirs_to_create = [
        base_dir / "data" / "candles",
        base_dir / "data" / "book",
        base_dir / "data" / "trades",
        base_dir / "models",
        base_dir / "logs",
        base_dir / "tests",
        base_dir / "src" / "features",
        base_dir / "src" / "buffers",
        base_dir / "output"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Criado: {dir_path}")
    
    # 2. Criar arquivo de configuração de teste
    test_config = {
        "environment": "test",
        "data_source": "simulated",
        "features": {
            "total": 65,
            "categories": {
                "volatility": 10,
                "returns": 10,
                "order_flow": 8,
                "volume": 8,
                "technical": 8,
                "microstructure": 15,
                "temporal": 6
            }
        },
        "buffers": {
            "candles": {
                "max_size": 200,
                "min_required": 100
            },
            "book": {
                "max_size": 100,
                "min_required": 50
            },
            "trades": {
                "max_size": 1000,
                "min_required": 100
            }
        },
        "performance_targets": {
            "max_latency_ms": 200,
            "max_memory_mb": 100,
            "min_features_per_second": 10
        },
        "created_at": datetime.now().isoformat()
    }
    
    config_file = base_dir / "test_config.json"
    with open(config_file, 'w') as f:
        json.dump(test_config, f, indent=2)
    print(f"\n[OK] Configuração salva: {config_file}")
    
    # 3. Criar requirements específicos para teste
    requirements_test = """# Requirements para ambiente de teste
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-benchmark>=4.0.0
memory-profiler>=0.61.0
line-profiler>=4.1.0
"""
    
    req_file = base_dir / "requirements_test.txt"
    with open(req_file, 'w') as f:
        f.write(requirements_test)
    print(f"[OK] Requirements criado: {req_file}")
    
    # 4. Copiar módulos essenciais (sem afetar produção)
    modules_to_copy = [
        "src/features/book_features.py",
        "src/feature_engine.py",
        "src/ml_features.py",
        "src/technical_indicators.py"
    ]
    
    for module in modules_to_copy:
        src = Path(module)
        if src.exists():
            dst = base_dir / module
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"[OK] Copiado: {module}")
    
    # 5. Criar dados de teste simulados
    create_test_data(base_dir)
    
    # 6. Criar script de validação inicial
    validation_script = base_dir / "validate_setup.py"
    with open(validation_script, 'w') as f:
        f.write("""
import sys
import json
from pathlib import Path

def validate_test_environment():
    base_dir = Path(".")
    
    # Verificar estrutura
    required_dirs = ["data", "models", "logs", "tests", "src", "output"]
    for dir_name in required_dirs:
        if not (base_dir / dir_name).exists():
            print(f"[ERRO] Diretório faltando: {dir_name}")
            return False
    
    # Verificar config
    config_file = base_dir / "test_config.json"
    if not config_file.exists():
        print("[ERRO] Arquivo de configuração não encontrado")
        return False
    
    with open(config_file) as f:
        config = json.load(f)
        print(f"[OK] Configuração carregada: {config['features']['total']} features")
    
    # Verificar dados de teste
    test_data = base_dir / "data" / "candles" / "test_candles.json"
    if not test_data.exists():
        print("[ERRO] Dados de teste não encontrados")
        return False
    
    print("[SUCESSO] Ambiente de teste validado com sucesso!")
    return True

if __name__ == "__main__":
    validate_test_environment()
""")
    
    print(f"\n[OK] Script de validação criado: {validation_script}")
    
    print("\n" + "=" * 60)
    print("AMBIENTE DE TESTE CRIADO COM SUCESSO!")
    print("=" * 60)
    print("\nPróximos passos:")
    print("1. cd test_environment")
    print("2. pip install -r requirements_test.txt")
    print("3. python validate_setup.py")
    
    return base_dir

def create_test_data(base_dir):
    """Cria dados simulados para teste"""
    import numpy as np
    
    # Gerar candles simulados
    np.random.seed(42)
    candles = []
    base_price = 5450.0
    
    for i in range(200):
        price_change = np.random.randn() * 5
        open_price = base_price + price_change
        close_price = open_price + np.random.randn() * 2
        high_price = max(open_price, close_price) + abs(np.random.randn())
        low_price = min(open_price, close_price) - abs(np.random.randn())
        volume = 1000000 + np.random.randint(-100000, 100000)
        
        candles.append({
            "timestamp": f"2025-08-08T10:{i//60:02d}:{i%60:02d}:00",
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": volume
        })
        
        base_price = close_price
    
    candles_file = base_dir / "data" / "candles" / "test_candles.json"
    with open(candles_file, 'w') as f:
        json.dump(candles, f, indent=2)
    print(f"\n[OK] Dados de teste criados: 200 candles")
    
    # Gerar book simulado
    book_snapshots = []
    for i in range(100):
        bid_prices = [base_price - j*0.5 for j in range(1, 6)]
        ask_prices = [base_price + j*0.5 for j in range(1, 6)]
        bid_volumes = [np.random.randint(100, 1000) for _ in range(5)]
        ask_volumes = [np.random.randint(100, 1000) for _ in range(5)]
        
        book_snapshots.append({
            "timestamp": f"2025-08-08T10:00:{i:02d}",
            "bid_prices": bid_prices,
            "bid_volumes": bid_volumes,
            "ask_prices": ask_prices,
            "ask_volumes": ask_volumes
        })
    
    book_file = base_dir / "data" / "book" / "test_book.json"
    with open(book_file, 'w') as f:
        json.dump(book_snapshots, f, indent=2)
    print(f"[OK] Dados de book criados: 100 snapshots")
    
    # Gerar trades simulados
    trades = []
    for i in range(1000):
        trades.append({
            "timestamp": f"2025-08-08T10:00:{i//10:02d}.{i%10:03d}",
            "price": round(base_price + np.random.randn() * 2, 2),
            "volume": np.random.randint(1, 100),
            "side": np.random.choice(["buy", "sell"]),
            "aggressor": np.random.choice(["buyer", "seller"]),
            "trader_id": f"trader_{np.random.randint(1, 20)}"
        })
    
    trades_file = base_dir / "data" / "trades" / "test_trades.json"
    with open(trades_file, 'w') as f:
        json.dump(trades, f, indent=2)
    print(f"[OK] Dados de trades criados: 1000 trades")

if __name__ == "__main__":
    setup_test_environment()