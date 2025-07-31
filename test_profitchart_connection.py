"""
Script para testar conexão com ProfitChart e ticker disponível
"""

from datetime import datetime, timedelta
import logging
from src.data.profitdll_real_collector import ProfitDLLRealCollector

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_connection():
    print("="*60)
    print("TESTE DE CONEXAO COM PROFITCHART")
    print("="*60)
    
    collector = ProfitDLLRealCollector()
    
    # 1. Testar inicialização
    print("\n1. Testando inicializacao...")
    if not collector.initialize():
        print("ERRO: Falha na inicializacao da DLL")
        return
    print("OK: DLL inicializada")
    
    # 2. Testar conexão
    print("\n2. Testando conexao...")
    if not collector.connect_and_login():
        print("ERRO: Falha na conexao")
        print("Verifique se o ProfitChart esta aberto e logado")
        return
    print("OK: Conectado com sucesso")
    
    # 3. Testar diferentes períodos e tickers
    print("\n3. Testando coleta de dados...")
    
    # Testar últimas 2 horas
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=2)
    
    tickers_to_test = ["WDOQ25", "WINQ25", "WDOZ25"]
    
    for ticker in tickers_to_test:
        print(f"\nTestando ticker: {ticker}")
        print(f"Periodo: {start_date} ate {end_date}")
        
        try:
            trades_df = collector.get_historical_trades(ticker, start_date, end_date)
            
            if not trades_df.empty:
                print(f"SUCESSO: {len(trades_df)} trades coletados")
                print(f"Primeiro trade: {trades_df.index[0]}")
                print(f"Ultimo trade: {trades_df.index[-1]}")
                print(f"Preco medio: {trades_df['price'].mean():.2f}")
                break
            else:
                print("AVISO: Nenhum trade retornado")
                
        except Exception as e:
            print(f"ERRO: {e}")
    
    # 4. Desconectar
    print("\n4. Desconectando...")
    collector.disconnect()
    print("OK: Desconectado")
    
    print("\n" + "="*60)
    print("TESTE CONCLUIDO")
    print("="*60)

if __name__ == "__main__":
    test_connection()