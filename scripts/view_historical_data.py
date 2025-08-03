"""
Script para visualizar dados históricos coletados
"""

import sys
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def view_historical_data(symbol: str = "WDOU25"):
    """Visualiza dados históricos coletados"""
    
    data_dir = Path("data/historical") / symbol
    
    if not data_dir.exists():
        print(f"Diretório não encontrado: {data_dir}")
        return
    
    # Listar todos arquivos parquet
    files = list(data_dir.rglob("*.parquet"))
    
    if not files:
        print("Nenhum arquivo encontrado")
        return
    
    print(f"\nEncontrados {len(files)} arquivos para {symbol}")
    print("="*60)
    
    # Carregar e combinar todos os dados
    all_data = []
    
    for file in sorted(files):
        print(f"\nCarregando: {file.relative_to(data_dir)}")
        
        try:
            df = pd.read_parquet(file)
            print(f"  Registros: {len(df)}")
            
            if len(df) > 0:
                # Adicionar coluna de data do arquivo
                date_str = file.parent.name
                df['file_date'] = date_str
                
                # Mostrar amostra
                print(f"  Primeira negociação: {df.iloc[0]['timestamp'] if 'timestamp' in df.columns else 'N/A'}")
                print(f"  Última negociação: {df.iloc[-1]['timestamp'] if 'timestamp' in df.columns else 'N/A'}")
                
                # Estatísticas básicas
                if 'price' in df.columns:
                    print(f"  Preço mín: {df['price'].min():.2f}")
                    print(f"  Preço máx: {df['price'].max():.2f}")
                    print(f"  Preço médio: {df['price'].mean():.2f}")
                
                if 'quantity' in df.columns:
                    print(f"  Volume total: {df['quantity'].sum()}")
                
                all_data.append(df)
                
        except Exception as e:
            print(f"  Erro ao ler arquivo: {e}")
    
    # Combinar todos os dados
    if all_data:
        print("\n" + "="*60)
        print("RESUMO GERAL")
        print("="*60)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total de registros: {len(combined_df)}")
        
        # Converter timestamp se necessário
        if 'timestamp' in combined_df.columns:
            try:
                combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], 
                                                         format='%d/%m/%Y %H:%M:%S.%f',
                                                         dayfirst=True,
                                                         errors='coerce')
                
                # Remover timestamps inválidos
                combined_df = combined_df.dropna(subset=['timestamp'])
                
                print(f"Período: {combined_df['timestamp'].min()} até {combined_df['timestamp'].max()}")
                
                # Criar gráfico de preços
                if 'price' in combined_df.columns and len(combined_df) > 100:
                    plt.figure(figsize=(12, 6))
                    
                    # Agrupar por minuto e calcular OHLC
                    combined_df.set_index('timestamp', inplace=True)
                    ohlc = combined_df['price'].resample('5Min').agg(['first', 'max', 'min', 'last'])
                    ohlc.columns = ['Open', 'High', 'Low', 'Close']
                    
                    # Plot
                    plt.plot(ohlc.index, ohlc['Close'], label='Preço de Fechamento (5min)')
                    plt.fill_between(ohlc.index, ohlc['Low'], ohlc['High'], alpha=0.3)
                    
                    plt.title(f'Histórico de Preços - {symbol}')
                    plt.xlabel('Data/Hora')
                    plt.ylabel('Preço')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    # Salvar gráfico
                    output_file = f"data/historical/{symbol}_price_chart.png"
                    plt.savefig(output_file)
                    print(f"\nGráfico salvo em: {output_file}")
                    plt.close()
                    
            except Exception as e:
                print(f"Erro ao processar timestamps: {e}")
        
        # Salvar amostra em CSV para análise
        sample_file = f"data/historical/{symbol}_sample.csv"
        sample_size = min(1000, len(combined_df))
        combined_df.head(sample_size).to_csv(sample_file, index=False)
        print(f"\nAmostra salva em: {sample_file}")


if __name__ == "__main__":
    view_historical_data()