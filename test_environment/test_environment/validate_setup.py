
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
