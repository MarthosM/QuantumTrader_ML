"""
Debug da estrutura de dados do Book Collector
Analisa inconsistências nos dados coletados
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
import sys

def analyze_data_structure(data_list):
    """Analisa estrutura dos dados para encontrar inconsistências"""
    print(f"\nTotal de registros: {len(data_list)}")
    
    # Contar tipos
    types = Counter(d.get('type', 'NO_TYPE') for d in data_list)
    print(f"\nTipos de dados:")
    for tipo, count in types.items():
        print(f"  {tipo}: {count}")
    
    # Analisar campos por tipo
    print(f"\n{'='*60}")
    print("ANÁLISE DE CAMPOS POR TIPO:")
    print(f"{'='*60}")
    
    fields_by_type = {}
    for item in data_list:
        tipo = item.get('type', 'NO_TYPE')
        if tipo not in fields_by_type:
            fields_by_type[tipo] = set()
        fields_by_type[tipo].update(item.keys())
    
    # Mostrar campos de cada tipo
    for tipo, fields in fields_by_type.items():
        print(f"\n{tipo}:")
        print(f"  Campos: {sorted(fields)}")
        
        # Verificar se todos os registros do tipo têm os mesmos campos
        records_of_type = [d for d in data_list if d.get('type') == tipo]
        field_counts = Counter()
        
        for record in records_of_type:
            field_counts[len(record)] += 1
            
        if len(field_counts) > 1:
            print(f"  ⚠️  AVISO: Registros com diferentes números de campos:")
            for num_fields, count in sorted(field_counts.items()):
                print(f"    {count} registros com {num_fields} campos")
        
        # Amostrar alguns registros problemáticos
        if len(field_counts) > 1:
            print(f"  Exemplos de registros:")
            shown = 0
            for record in records_of_type[:5]:
                print(f"    Campos: {list(record.keys())}")
                shown += 1
                if shown >= 3:
                    break
    
    # Verificar registros sem tipo
    no_type = [i for i, d in enumerate(data_list) if 'type' not in d]
    if no_type:
        print(f"\n⚠️  {len(no_type)} registros sem campo 'type' nos índices: {no_type[:10]}...")
        
    # Verificar valores None ou vazios
    print(f"\n{'='*60}")
    print("VERIFICAÇÃO DE VALORES PROBLEMÁTICOS:")
    print(f"{'='*60}")
    
    for tipo in types:
        records_of_type = [d for d in data_list if d.get('type') == tipo]
        if records_of_type:
            sample = records_of_type[0]
            for field in sample.keys():
                none_count = sum(1 for r in records_of_type if r.get(field) is None)
                empty_count = sum(1 for r in records_of_type if r.get(field) == "")
                if none_count > 0 or empty_count > 0:
                    print(f"\n{tipo}.{field}:")
                    if none_count > 0:
                        print(f"  None: {none_count} registros")
                    if empty_count > 0:
                        print(f"  Vazio: {empty_count} registros")

def main():
    # Procurar arquivos JSON mais recentes
    data_dir = Path('data/realtime/book')
    
    if not data_dir.exists():
        print(f"Diretório {data_dir} não encontrado!")
        return
        
    # Listar todos os arquivos JSON
    json_files = list(data_dir.rglob('*raw.json'))
    
    if not json_files:
        print("Nenhum arquivo JSON raw encontrado!")
        # Tentar parquet
        parquet_files = list(data_dir.rglob('*.parquet'))
        if parquet_files:
            print(f"\nEncontrados {len(parquet_files)} arquivos parquet:")
            for pf in parquet_files[-5:]:
                print(f"  {pf}")
                try:
                    df = pd.read_parquet(pf)
                    print(f"    Shape: {df.shape}")
                    print(f"    Colunas: {list(df.columns)}")
                except Exception as e:
                    print(f"    Erro: {e}")
        return
        
    # Analisar o arquivo mais recente
    latest_file = sorted(json_files)[-1]
    print(f"Analisando: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    analyze_data_structure(data)
    
    # Tentar criar DataFrame para reproduzir o erro
    print(f"\n{'='*60}")
    print("TENTANDO CRIAR DATAFRAME:")
    print(f"{'='*60}")
    
    try:
        df = pd.DataFrame(data)
        print(f"✅ DataFrame criado com sucesso!")
        print(f"   Shape: {df.shape}")
        print(f"   Colunas: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Erro ao criar DataFrame: {e}")
        print(f"   Tipo do erro: {type(e).__name__}")
        
        # Tentar criar por tipo
        for tipo in types:
            records = [d for d in data if d.get('type') == tipo]
            if records:
                try:
                    df_tipo = pd.DataFrame(records)
                    print(f"\n✅ DataFrame para tipo '{tipo}' criado com sucesso")
                    print(f"   Shape: {df_tipo.shape}")
                except Exception as e2:
                    print(f"\n❌ Erro no tipo '{tipo}': {e2}")

if __name__ == "__main__":
    main()