"""
Teste básico de conexão com ProfitDLL
"""

import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.connection_manager_v4 import ConnectionManagerV4


def test_connection():
    """Testa conexão básica com ProfitDLL"""
    print("="*60)
    print("TESTE DE CONEXAO PROFITDLL")
    print("="*60)
    
    # Carregar variáveis de ambiente
    load_dotenv()
    
    # Configurações
    dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
    username = os.getenv("PROFIT_USERNAME")
    password = os.getenv("PROFIT_PASSWORD") 
    key = os.getenv("PROFIT_KEY")
    
    if not all([username, password, key]):
        print("[ERRO] Credenciais não encontradas no arquivo .env")
        print("Crie um arquivo .env com:")
        print("PROFIT_USERNAME=seu_usuario")
        print("PROFIT_PASSWORD=sua_senha")
        print("PROFIT_KEY=sua_chave")
        return False
    
    print(f"DLL Path: {dll_path}")
    print(f"Username: {username}")
    print(f"Key: {key[:10]}...")
    
    # Criar connection manager
    try:
        print("\nCriando ConnectionManager...")
        connection = ConnectionManagerV4(dll_path)
        print("[OK] ConnectionManager criado")
        
        # Callback para mudanças de estado
        def on_state_change(state_type, result):
            print(f"[STATE] Tipo: {state_type}, Resultado: {result}")
        
        connection.register_state_callback(on_state_change)
        
        # Conectar
        print("\nConectando ao ProfitDLL...")
        success = connection.initialize(
            key=key,
            username=username,
            password=password
        )
        
        if success:
            print("[OK] Conexão estabelecida!")
            
            # Aguardar estados
            print("\nAguardando estados de conexão...")
            time.sleep(5)
            
            # Verificar estados
            print(f"\nLogin State: {connection.login_state}")
            print(f"Market State: {connection.market_state}")
            print(f"Routing State: {connection.routing_state}")
            
            # Desconectar
            print("\nDesconectando...")
            connection.disconnect()
            print("[OK] Desconectado")
            
            return True
        else:
            print("[ERRO] Falha ao conectar")
            return False
            
    except Exception as e:
        print(f"[ERRO] Exceção: {e}")
        return False


def main():
    print(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    
    if test_connection():
        print("\n*** TESTE CONCLUIDO COM SUCESSO ***")
    else:
        print("\n*** TESTE FALHOU ***")


if __name__ == "__main__":
    main()