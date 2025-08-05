"""
Script para verificar disponibilidade e qualidade dos dados históricos
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_historical_data(symbol: str, days: int = 365):
    """
    Verifica disponibilidade e qualidade dos dados históricos
    """
    print(f"\n{'='*60}")
    print(f"VERIFICAÇÃO DE DADOS HISTÓRICOS - {symbol}")
    print(f"{'='*60}\n")
    
    # Paths possíveis
    base_paths = [
        Path('data/historical'),
        Path('data') / symbol,
        Path('data/historical') / symbol
    ]
    
    # Encontrar path correto
    data_path = None
    for path in base_paths:
        if path.exists():
            data_path = path
            break
            
    if not data_path:
        print("❌ Nenhum diretório de dados encontrado!")
        print(f"   Procurado em: {[str(p) for p in base_paths]}")
        return False
        
    print(f"📁 Diretório de dados: {data_path}")
    
    # Estatísticas
    stats = {
        'total_files': 0,
        'valid_files': 0,
        'corrupted_files': 0,
        'total_records': 0,
        'date_range': {'start': None, 'end': None},
        'missing_dates': [],
        'file_sizes': [],
        'issues': []
    }
    
    # Buscar arquivos parquet
    parquet_files = list(data_path.rglob('*.parquet'))
    stats['total_files'] = len(parquet_files)
    
    print(f"\n📊 Arquivos encontrados: {stats['total_files']}")
    
    if stats['total_files'] == 0:
        print("❌ Nenhum arquivo parquet encontrado!")
        return False
        
    # Analisar cada arquivo
    dates_found = set()
    
    for i, file_path in enumerate(parquet_files):
        try:
            # Tentar ler arquivo
            df = pd.read_parquet(file_path)
            stats['valid_files'] += 1
            stats['total_records'] += len(df)
            stats['file_sizes'].append(file_path.stat().st_size / 1024 / 1024)  # MB
            
            # Verificar estrutura
            required_columns = ['price', 'volume', 'timestamp']
            missing_cols = set(required_columns) - set(df.columns)
            
            if missing_cols:
                stats['issues'].append(f"Colunas faltando em {file_path.name}: {missing_cols}")
                
            # Verificar dados
            if 'price' in df.columns:
                if df['price'].isna().sum() > 0:
                    stats['issues'].append(f"NaN encontrados em price: {file_path.name}")
                if (df['price'] <= 0).any():
                    stats['issues'].append(f"Preços inválidos (<=0) em: {file_path.name}")
                    
            # Extrair data do arquivo
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                file_date = df['timestamp'].dt.date.iloc[0]
                dates_found.add(file_date)
                
                # Atualizar range de datas
                if stats['date_range']['start'] is None or file_date < stats['date_range']['start']:
                    stats['date_range']['start'] = file_date
                if stats['date_range']['end'] is None or file_date > stats['date_range']['end']:
                    stats['date_range']['end'] = file_date
                    
            # Progress
            if (i + 1) % 10 == 0:
                print(f"   Processados: {i + 1}/{stats['total_files']} arquivos...", end='\r')
                
        except Exception as e:
            stats['corrupted_files'] += 1
            stats['issues'].append(f"Erro ao ler {file_path.name}: {str(e)}")
            
    print(f"\n\n📈 Resumo da Análise:")
    print(f"   - Arquivos válidos: {stats['valid_files']}/{stats['total_files']}")
    print(f"   - Arquivos corrompidos: {stats['corrupted_files']}")
    print(f"   - Total de registros: {stats['total_records']:,}")
    
    if stats['file_sizes']:
        print(f"   - Tamanho médio por arquivo: {np.mean(stats['file_sizes']):.1f} MB")
        print(f"   - Tamanho total: {sum(stats['file_sizes']):.1f} MB")
        
    # Verificar gaps de datas
    if stats['date_range']['start'] and stats['date_range']['end']:
        print(f"\n📅 Período dos dados:")
        print(f"   - Início: {stats['date_range']['start']}")
        print(f"   - Fim: {stats['date_range']['end']}")
        
        # Calcular dias úteis esperados
        expected_dates = []
        current = stats['date_range']['start']
        while current <= stats['date_range']['end']:
            if current.weekday() < 5:  # Segunda a sexta
                expected_dates.append(current)
            current += timedelta(days=1)
            
        missing_dates = set(expected_dates) - dates_found
        coverage = len(dates_found) / len(expected_dates) * 100 if expected_dates else 0
        
        print(f"   - Dias úteis esperados: {len(expected_dates)}")
        print(f"   - Dias com dados: {len(dates_found)}")
        print(f"   - Cobertura: {coverage:.1f}%")
        
        if missing_dates:
            print(f"   - Dias faltando: {len(missing_dates)}")
            if len(missing_dates) <= 10:
                for date in sorted(missing_dates)[:10]:
                    print(f"     • {date}")
                    
    # Verificar requisitos mínimos
    print(f"\n✅ Verificação de Requisitos:")
    
    min_days = days * 0.7  # 70% dos dias solicitados
    has_min_data = len(dates_found) >= min_days
    print(f"   - Mínimo de {min_days:.0f} dias: {'✅ OK' if has_min_data else '❌ FALHOU'}")
    
    has_recent_data = False
    if stats['date_range']['end']:
        days_old = (datetime.now().date() - stats['date_range']['end']).days
        has_recent_data = days_old < 30
        print(f"   - Dados recentes (< 30 dias): {'✅ OK' if has_recent_data else f'❌ {days_old} dias atrás'}")
        
    no_major_issues = len(stats['issues']) < 5
    print(f"   - Integridade dos dados: {'✅ OK' if no_major_issues else f'⚠️  {len(stats["issues"])} problemas'}")
    
    # Mostrar problemas se houver
    if stats['issues']:
        print(f"\n⚠️  Problemas encontrados ({len(stats['issues'])}):")
        for issue in stats['issues'][:10]:  # Mostrar no máximo 10
            print(f"   - {issue}")
        if len(stats['issues']) > 10:
            print(f"   ... e mais {len(stats['issues']) - 10} problemas")
            
    # Recomendações
    print(f"\n💡 Recomendações:")
    
    if not has_min_data:
        print("   - Coletar mais dados históricos antes do treinamento")
        print(f"   - Necessário: pelo menos {int(min_days)} dias")
        
    if not has_recent_data:
        print("   - Atualizar dados históricos com período mais recente")
        
    if stats['corrupted_files'] > 0:
        print(f"   - Investigar e corrigir {stats['corrupted_files']} arquivos corrompidos")
        
    if len(missing_dates) > len(dates_found) * 0.1:  # Mais de 10% faltando
        print("   - Muitos gaps nos dados - verificar coleta")
        
    # Resultado final
    print(f"\n{'='*60}")
    ready = has_min_data and no_major_issues
    if ready:
        print("✅ DADOS PRONTOS PARA TREINAMENTO!")
    else:
        print("❌ DADOS INSUFICIENTES - VERIFICAR RECOMENDAÇÕES ACIMA")
    print(f"{'='*60}\n")
    
    return ready


def main():
    parser = argparse.ArgumentParser(description='Verifica dados históricos para treinamento')
    parser.add_argument('--symbol', type=str, default='WDOU25', help='Símbolo para verificar')
    parser.add_argument('--days', type=int, default=365, help='Dias de histórico esperados')
    
    args = parser.parse_args()
    
    ready = check_historical_data(args.symbol, args.days)
    
    # Exit code
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()