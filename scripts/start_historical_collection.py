"""
Script para iniciar a coleta de dados históricos usando servidor isolado
Este é o método recomendado para evitar Segmentation Fault
"""

import os
import sys
import time
import json
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HistoricalCollection')


def start_profit_dll_server():
    """Inicia o servidor ProfitDLL em processo isolado"""
    logger.info("🚀 Iniciando servidor ProfitDLL isolado...")
    
    # Comando para iniciar o servidor
    server_script = "src/integration/profit_dll_server.py"
    
    # Verificar se o script existe
    if not os.path.exists(server_script):
        logger.error(f"Script do servidor não encontrado: {server_script}")
        return None
    
    # Iniciar servidor em subprocess
    process = subprocess.Popen(
        [sys.executable, server_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    logger.info(f"Servidor iniciado com PID: {process.pid}")
    
    # Aguardar inicialização
    time.sleep(5)
    
    # Verificar se ainda está rodando
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        logger.error("Servidor morreu durante inicialização!")
        logger.error(f"STDOUT: {stdout}")
        logger.error(f"STDERR: {stderr}")
        return None
    
    return process


def collect_historical_data(symbol: str, days_back: int = 30):
    """Coleta dados históricos via servidor isolado"""
    
    from src.database.historical_data_collector import HistoricalDataCollector
    
    # Configuração
    config = {
        'data_dir': 'data/historical',
        'server_address': ('localhost', 6789)
    }
    
    # Criar coletor
    collector = HistoricalDataCollector(config)
    
    # Calcular período
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    logger.info(f"📊 Coletando dados de {symbol}")
    logger.info(f"   Período: {start_date.strftime('%d/%m/%Y')} até {end_date.strftime('%d/%m/%Y')}")
    
    # Coletar dados
    try:
        result = collector.collect_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            data_types=['trades']
        )
        
        if result:
            logger.info("✅ Coleta bem-sucedida!")
            
            # Mostrar estatísticas
            total_records = 0
            for date, data_types in result.items():
                if isinstance(data_types, dict):
                    for data_type, records in data_types.items():
                        if isinstance(records, list):
                            count = len(records)
                            if count > 0:
                                total_records += count
                                logger.info(f"   {date}: {data_type} - {count} registros")
            
            logger.info(f"   Total: {total_records} registros")
            
            # Verificar arquivos salvos
            data_dir = Path("data/historical") / symbol
            if data_dir.exists():
                # Buscar arquivos .parquet recursivamente
                files = list(data_dir.rglob("*.parquet"))
                logger.info(f"   Arquivos salvos: {len(files)}")
                
                # Mostrar últimos 5 arquivos
                for file in sorted(files)[-5:]:
                    size = file.stat().st_size / 1024  # KB
                    logger.info(f"     {file.relative_to(data_dir)} ({size:.1f} KB)")
        else:
            logger.error(f"❌ Nenhum dado coletado")
            
    except Exception as e:
        logger.error(f"❌ Erro durante coleta: {e}")


def main():
    logger.info("="*80)
    logger.info("🏗️ SISTEMA DE COLETA HISTÓRICA - PRODUÇÃO")
    logger.info("="*80)
    logger.info("Usando arquitetura de processo isolado para evitar crashes")
    logger.info("")
    
    # Configurações
    symbols = ['WDOU25']  # Contrato atual do WDO
    days_back = 30  # Últimos 30 dias
    
    # 1. Iniciar servidor (se não estiver rodando)
    server_process = None
    try:
        # Verificar se servidor já está rodando
        from multiprocessing.connection import Client
        
        try:
            # Tentar conectar ao servidor
            client = Client(('localhost', 6789), authkey=b'profit_dll_secret')
            client.send({'type': 'status'})
            
            if client.poll(timeout=2):
                response = client.recv()
                if response.get('connected'):
                    logger.info("✅ Servidor já está rodando e conectado!")
                else:
                    logger.info("⚠️ Servidor rodando mas não conectado ao ProfitDLL")
            client.close()
            
        except:
            # Servidor não está rodando, iniciar
            logger.info("Servidor não detectado, iniciando...")
            server_process = start_profit_dll_server()
            
            if not server_process:
                logger.error("Falha ao iniciar servidor!")
                return
        
        # 2. Coletar dados para cada símbolo
        for symbol in symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"Coletando: {symbol}")
            logger.info(f"{'='*60}")
            
            collect_historical_data(symbol, days_back)
            
            # Pausa entre símbolos
            if len(symbols) > 1:
                logger.info("\n⏸️ Aguardando 10 segundos antes do próximo símbolo...")
                time.sleep(10)
        
        # 3. Resumo final
        logger.info("\n" + "="*80)
        logger.info("📋 RESUMO DA COLETA")
        logger.info("="*80)
        
        # Verificar dados coletados
        total_files = 0
        total_size = 0
        
        for symbol in symbols:
            data_dir = Path("data/historical") / symbol
            if data_dir.exists():
                files = list(data_dir.rglob("*.parquet"))
                size = sum(f.stat().st_size for f in files) / (1024 * 1024)  # MB
                
                logger.info(f"{symbol}:")
                logger.info(f"   Arquivos: {len(files)}")
                logger.info(f"   Tamanho: {size:.1f} MB")
                
                total_files += len(files)
                total_size += size
        
        logger.info(f"\nTOTAL:")
        logger.info(f"   Arquivos: {total_files}")
        logger.info(f"   Tamanho: {total_size:.1f} MB")
        
    except Exception as e:
        logger.error(f"Erro no processo principal: {e}", exc_info=True)
        
    finally:
        # Finalizar servidor se foi iniciado por este script
        if server_process:
            logger.info("\n🛑 Finalizando servidor...")
            server_process.terminate()
            server_process.wait(timeout=5)
            
            if server_process.poll() is None:
                server_process.kill()
            
            logger.info("Servidor finalizado")
    
    logger.info("\n✅ COLETA CONCLUÍDA!")


if __name__ == "__main__":
    main()