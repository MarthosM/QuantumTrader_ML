"""
Script para verificar disponibilidade e qualidade dos dados de book
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_book_data(symbol: str, days: int = 30):
    """
    Verifica disponibilidade e qualidade dos dados de book
    """
    print(f"\n{'='*60}")
    print(f"VERIFICAÇÃO DE DADOS DE BOOK - {symbol}")
    print(f"{'='*60}\n")
    
    # Path dos dados de book
    book_path = Path('data/realtime/book')
    
    if not book_path.exists():
        print("❌ Diretório de book não encontrado!")
        print(f"   Esperado em: {book_path}")
        print("\n💡 Execute o coletor de book primeiro:")
        print("   python scripts/book_collector.py")
        return False
        
    print(f"📁 Diretório de book: {book_path}")
    
    # Estatísticas
    stats = {
        'total_files': 0,
        'offer_book_files': 0,
        'price_book_files': 0,
        'valid_files': 0,
        'total_records': 0,
        'date_range': {'start': None, 'end': None},
        'coverage_hours': {},
        'issues': []
    }
    
    # Buscar arquivos de book
    offer_files = list(book_path.rglob(f'offer_book_{symbol}_*.parquet'))
    price_files = list(book_path.rglob(f'price_book_{symbol}_*.parquet'))
    
    stats['offer_book_files'] = len(offer_files)
    stats['price_book_files'] = len(price_files)
    stats['total_files'] = len(offer_files) + len(price_files)
    
    print(f"\n📊 Arquivos encontrados:")
    print(f"   - Offer Book: {stats['offer_book_files']}")
    print(f"   - Price Book: {stats['price_book_files']}")
    print(f"   - Total: {stats['total_files']}")
    
    if stats['total_files'] == 0:
        print("\n❌ Nenhum arquivo de book encontrado!")
        print("\n💡 Para coletar dados de book:")
        print("   1. Certifique-se que o mercado está aberto")
        print("   2. Execute: python scripts/book_collector.py")
        return False
        
    # Analisar arquivos
    all_files = offer_files + price_files
    dates_found = set()
    hourly_coverage = {}
    
    for i, file_path in enumerate(all_files):
        try:
            # Ler arquivo
            df = pd.read_parquet(file_path)
            stats['valid_files'] += 1
            stats['total_records'] += len(df)
            
            # Verificar estrutura para offer book
            if 'offer_book' in file_path.name:
                required_cols = ['timestamp', 'position', 'price', 'quantity', 'agent_code']
                missing = set(required_cols) - set(df.columns)
                if missing:
                    stats['issues'].append(f"Colunas faltando em offer book: {missing}")
                    
            # Verificar estrutura para price book
            elif 'price_book' in file_path.name:
                required_cols = ['timestamp', 'position', 'side', 'price', 'quantity']
                missing = set(required_cols) - set(df.columns)
                if missing:
                    stats['issues'].append(f"Colunas faltando em price book: {missing}")
                    
            # Analisar timestamps
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Datas únicas
                file_dates = df['timestamp'].dt.date.unique()
                dates_found.update(file_dates)
                
                # Coverage por hora
                for date in file_dates:
                    if date not in hourly_coverage:
                        hourly_coverage[date] = set()
                    
                    day_data = df[df['timestamp'].dt.date == date]
                    hours = day_data['timestamp'].dt.hour.unique()
                    hourly_coverage[date].update(hours)
                    
                # Atualizar range
                min_date = df['timestamp'].min().date()
                max_date = df['timestamp'].max().date()
                
                if stats['date_range']['start'] is None or min_date < stats['date_range']['start']:
                    stats['date_range']['start'] = min_date
                if stats['date_range']['end'] is None or max_date > stats['date_range']['end']:
                    stats['date_range']['end'] = max_date
                    
            # Progress
            if (i + 1) % 10 == 0:
                print(f"   Processados: {i + 1}/{stats['total_files']} arquivos...", end='\r')
                
        except Exception as e:
            stats['issues'].append(f"Erro ao ler {file_path.name}: {str(e)}")
            
    print(f"\n\n📈 Resumo da Análise:")
    print(f"   - Arquivos válidos: {stats['valid_files']}/{stats['total_files']}")
    print(f"   - Total de registros: {stats['total_records']:,}")
    print(f"   - Média por arquivo: {stats['total_records'] // stats['valid_files']:,}" if stats['valid_files'] > 0 else "")
    
    # Período dos dados
    if stats['date_range']['start'] and stats['date_range']['end']:
        print(f"\n📅 Período dos dados:")
        print(f"   - Início: {stats['date_range']['start']}")
        print(f"   - Fim: {stats['date_range']['end']}")
        
        total_days = (stats['date_range']['end'] - stats['date_range']['start']).days + 1
        print(f"   - Total de dias: {total_days}")
        print(f"   - Dias com dados: {len(dates_found)}")
        
    # Análise de cobertura horária
    print(f"\n⏰ Cobertura do Pregão:")
    
    pregao_hours = list(range(9, 18))  # 9h às 17h
    good_coverage_days = 0
    
    for date in sorted(hourly_coverage.keys())[-5:]:  # Últimos 5 dias
        hours = hourly_coverage[date]
        pregao_covered = [h for h in pregao_hours if h in hours]
        coverage_pct = len(pregao_covered) / len(pregao_hours) * 100
        
        print(f"   {date}: {coverage_pct:.0f}% ({len(pregao_covered)}/{len(pregao_hours)} horas)")
        
        if coverage_pct >= 80:
            good_coverage_days += 1
            
    # Verificar tipos de book
    print(f"\n📊 Tipos de Book:")
    
    has_offer = stats['offer_book_files'] > 0
    has_price = stats['price_book_files'] > 0
    
    print(f"   - Offer Book (detalhado): {'✅ Disponível' if has_offer else '❌ Não encontrado'}")
    print(f"   - Price Book (agregado): {'✅ Disponível' if has_price else '❌ Não encontrado'}")
    
    # Verificar requisitos mínimos
    print(f"\n✅ Verificação de Requisitos:")
    
    min_days = max(15, days * 0.5)  # Pelo menos 15 dias ou 50% do solicitado
    has_min_days = len(dates_found) >= min_days
    print(f"   - Mínimo de {min_days:.0f} dias: {'✅ OK' if has_min_days else f'❌ Apenas {len(dates_found)} dias'}")
    
    has_recent = False
    if stats['date_range']['end']:
        days_old = (datetime.now().date() - stats['date_range']['end']).days
        has_recent = days_old < 7
        print(f"   - Dados recentes (< 7 dias): {'✅ OK' if has_recent else f'❌ {days_old} dias atrás'}")
        
    has_coverage = good_coverage_days >= 3
    print(f"   - Cobertura do pregão: {'✅ OK' if has_coverage else '❌ Cobertura insuficiente'}")
    
    has_both_types = has_offer and has_price
    print(f"   - Ambos tipos de book: {'✅ OK' if has_both_types else '⚠️  Recomendado ter ambos'}")
    
    # Problemas encontrados
    if stats['issues']:
        print(f"\n⚠️  Problemas encontrados ({len(stats['issues'])}):")
        for issue in stats['issues'][:5]:
            print(f"   - {issue}")
            
    # Recomendações
    print(f"\n💡 Recomendações:")
    
    if not has_min_days:
        print(f"   - Coletar mais {min_days - len(dates_found):.0f} dias de dados")
        print("   - Execute: python scripts/book_collector.py durante o pregão")
        
    if not has_recent:
        print("   - Atualizar coleta de book com dados mais recentes")
        
    if not has_coverage:
        print("   - Melhorar cobertura horária - coletar durante todo o pregão")
        
    if not has_both_types:
        print("   - Configurar coleta para ambos offer book e price book")
        
    # Análise de qualidade dos dados
    if stats['valid_files'] > 0:
        print(f"\n📊 Qualidade dos Dados:")
        
        # Tamanho médio dos arquivos
        file_sizes = [f.stat().st_size / 1024 / 1024 for f in all_files]  # MB
        print(f"   - Tamanho médio: {np.mean(file_sizes):.1f} MB")
        print(f"   - Tamanho total: {sum(file_sizes):.1f} MB")
        
        # Densidade de dados
        if stats['total_records'] > 0 and len(dates_found) > 0:
            records_per_day = stats['total_records'] / len(dates_found)
            print(f"   - Registros por dia: {records_per_day:,.0f}")
            
            # Estimativa de updates por segundo durante pregão
            pregao_seconds = 8 * 3600  # 8 horas
            updates_per_second = records_per_day / pregao_seconds
            print(f"   - Updates por segundo (estimado): {updates_per_second:.1f}")
            
    # Resultado final
    print(f"\n{'='*60}")
    
    ready_basic = has_min_days and has_recent
    ready_enhanced = ready_basic and has_coverage and has_both_types
    
    if ready_enhanced:
        print("✅ DADOS DE BOOK PRONTOS PARA TREINAMENTO COMPLETO!")
    elif ready_basic:
        print("⚠️  DADOS DE BOOK MÍNIMOS DISPONÍVEIS")
        print("   Recomenda-se coletar mais dados para melhores resultados")
    else:
        print("❌ DADOS DE BOOK INSUFICIENTES")
        print("   Execute a coleta por mais dias antes do treinamento")
        
    print(f"{'='*60}\n")
    
    return ready_basic


def main():
    parser = argparse.ArgumentParser(description='Verifica dados de book para treinamento')
    parser.add_argument('--symbol', type=str, default='WDOU25', help='Símbolo para verificar')
    parser.add_argument('--days', type=int, default=30, help='Dias de book esperados')
    
    args = parser.parse_args()
    
    ready = check_book_data(args.symbol, args.days)
    
    # Exit code
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()