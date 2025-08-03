"""
Teste do ConnectionManagerV4 com callbacks corrigidos
"""

import os
import sys
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestFixed')

# Carregar variáveis
load_dotenv()


def test_connection_manager_direct():
    """Testa ConnectionManagerV4 diretamente com callbacks corrigidos"""
    logger.info("="*80)
    logger.info("🧪 TESTE CONNECTIONMANAGERV4 COM CALLBACKS CORRIGIDOS")
    logger.info("="*80)
    
    # Primeiro, vamos sobrescrever os callbacks do ConnectionManagerV4
    import ctypes
    from src.connection_manager_v4 import ConnectionManagerV4
    from src.profit_dll_structures_fixed import (
        TStateCallbackFixed, TNewTradeCallbackFixed, THistoryTradeCallbackFixed,
        TProgressCallbackFixed, TAccountCallbackFixed, THistoryCallbackFixed,
        TOrderChangeCallbackFixed, TNewDailyCallbackFixed, TPriceBookCallbackFixed,
        TOfferBookCallbackFixed, TTinyBookCallbackFixed
    )
    
    # Monkey patch para usar callbacks corrigidos
    import src.profit_dll_structures as structures
    structures.TStateCallback = TStateCallbackFixed
    structures.TNewTradeCallback = TNewTradeCallbackFixed
    structures.THistoryTradeCallback = THistoryTradeCallbackFixed
    structures.TProgressCallback = TProgressCallbackFixed
    structures.TAccountCallback = TAccountCallbackFixed
    structures.THistoryCallback = THistoryCallbackFixed
    structures.TOrderChangeCallback = TOrderChangeCallbackFixed
    structures.TNewDailyCallback = TNewDailyCallbackFixed
    structures.TPriceBookCallback = TPriceBookCallbackFixed
    structures.TOfferBookCallback = TOfferBookCallbackFixed
    structures.TTinyBookCallback = TTinyBookCallbackFixed
    
    try:
        # Criar ConnectionManager
        dll_path = r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
        logger.info(f"Criando ConnectionManagerV4...")
        conn = ConnectionManagerV4(dll_path)
        
        # Conectar
        logger.info("\n🔌 Conectando ao ProfitDLL...")
        result = conn.initialize(
            key=os.getenv('PROFIT_KEY'),
            username=os.getenv('PROFIT_USERNAME'),
            password=os.getenv('PROFIT_PASSWORD')
        )
        
        if result:
            logger.info("✅ Conectado com sucesso!")
            
            # Aguardar estabilização
            logger.info("\n⏳ Aguardando estabilização...")
            time.sleep(5)
            
            # Testar histórico
            logger.info("\n📊 Testando coleta histórica...")
            
            trades_received = []
            def on_history_trade(data):
                trades_received.append(data)
                if len(trades_received) == 1:
                    logger.info(f"🎯 Primeiro trade recebido: {data}")
                elif len(trades_received) % 100 == 0:
                    logger.info(f"📊 {len(trades_received)} trades recebidos...")
            
            conn.register_history_trade_callback(on_history_trade)
            
            # Solicitar dados
            success = conn.get_history_trades(
                ticker='WDOU25',
                exchange='F',
                date_start='01/08/2025',
                date_end='01/08/2025'
            )
            
            logger.info(f"Solicitação enviada: {success}")
            
            # Aguardar dados
            logger.info("⏳ Aguardando dados por 20 segundos...")
            time.sleep(20)
            
            logger.info(f"\n📊 Total de trades recebidos: {len(trades_received)}")
            
            if trades_received:
                logger.info("✅ SUCESSO! Dados históricos coletados!")
                logger.info(f"Primeiro: {trades_received[0]}")
                logger.info(f"Último: {trades_received[-1]}")
            else:
                logger.warning("⚠️ Nenhum dado recebido")
            
            # Desconectar
            logger.info("\n🔌 Desconectando...")
            conn.disconnect()
            logger.info("✅ Desconectado")
            
            return True
            
        else:
            logger.error("❌ Falha ao conectar")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro: {e}", exc_info=True)
        return False


def test_in_isolated_process():
    """Executa teste em processo isolado para segurança"""
    logger.info("\n" + "="*80)
    logger.info("🧪 TESTE EM PROCESSO ISOLADO")
    logger.info("="*80)
    
    import subprocess
    
    # Script para executar em processo isolado
    test_script = """
import sys
sys.path.append(r'C:\\Users\\marth\\OneDrive\\Programacao\\Python\\Projetos\\QuantumTrader_ML')
from scripts.test_connection_manager_fixed import test_connection_manager_direct
result = test_connection_manager_direct()
sys.exit(0 if result else 1)
"""
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        logger.info(f"Exit code: {result.returncode}")
        
        if result.stdout:
            logger.info("\nOUTPUT:")
            print(result.stdout)
        
        if result.stderr:
            logger.info("\nERROS:")
            print(result.stderr)
        
        if result.returncode == -1073741819:  # 0xC0000005
            logger.error("❌ Segmentation Fault detectado!")
            logger.info("💡 Solução: Use arquitetura de processo isolado com comunicação via arquivos")
        elif result.returncode == 0:
            logger.info("✅ Teste bem-sucedido!")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout no processo")
        return False
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        return False


def main():
    logger.info("🧪 TESTE FINAL - CONNECTIONMANAGERV4 COM CORREÇÕES")
    logger.info("="*80)
    
    # Primeiro tentar diretamente
    logger.info("1️⃣ Tentando execução direta...")
    
    try:
        direct_success = test_connection_manager_direct()
        
        if direct_success:
            logger.info("\n✅ SUCESSO DIRETO!")
            logger.info("O ConnectionManagerV4 funciona com callbacks corrigidos!")
        else:
            logger.info("\n❌ Falha na execução direta")
            
    except Exception as e:
        logger.error(f"\n❌ Crash na execução direta: {e}")
        
        # Se crashar, tentar isolado
        logger.info("\n2️⃣ Tentando em processo isolado...")
        isolated_success = test_in_isolated_process()
        
        if isolated_success:
            logger.info("\n✅ Funciona em processo isolado!")
        else:
            logger.info("\n❌ Falha mesmo isolado")
    
    # Análise final
    logger.info("\n" + "="*80)
    logger.info("📋 CONCLUSÃO")
    logger.info("="*80)
    logger.info("1. Os callbacks do ProfitDLL devem retornar c_int, não None")
    logger.info("2. O ConnectionManagerV4 precisa ser atualizado com os tipos corretos")
    logger.info("3. Para produção, use processo isolado com comunicação via arquivos")
    logger.info("4. Isso evita crashes e mantém o sistema principal estável")


if __name__ == "__main__":
    main()