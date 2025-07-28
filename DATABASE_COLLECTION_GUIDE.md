# Guia do Sistema de Coleta e Banco de Dados Históricos

## Resumo Executivo

Sistema completo para coleta, validação, armazenamento e merge de dados históricos de trading, considerando as limitações do ProfitDLL (máximo 3 meses, 9 dias por vez).

## Componentes Implementados

### 1. HistoricalDataCollector
**Arquivo**: `src/database/historical_data_collector.py`

Gerencia coleta de dados de múltiplas fontes:
- **ProfitDLL**: Dados recentes (últimos 3 meses)
- **CSV**: Dados históricos arquivados
- **APIs**: Fontes externas como fallback

**Funcionalidades principais**:
- Estratégia inteligente de coleta baseada no período
- Divisão automática em chunks de 9 dias para ProfitDLL
- Cache de metadados para evitar re-coleta
- Suporte a múltiplos tipos de dados (trades, candles, book)

### 2. DatabaseManager
**Arquivo**: `src/database/database_manager.py`

Sistema de armazenamento otimizado:
- **Formato**: Parquet comprimido com gzip
- **Estrutura**: Organizado por símbolo/tipo/data
- **Metadados**: SQLite para índices e estatísticas
- **Otimizações**: Compressão adaptativa e tipos de dados otimizados

**Funcionalidades**:
- Thread-safe com locks
- Backup automático
- Estatísticas e métricas de qualidade
- Query eficiente por período

### 3. DataValidator
**Arquivo**: `src/database/data_validator.py`

Validação completa de qualidade:
- **Estrutura**: Verifica colunas e tipos obrigatórios
- **Integridade**: Detecta duplicatas, outliers e gaps
- **Consistência**: Valida OHLC, spreads e horários
- **Auto-correção**: Opção de corrigir problemas automaticamente

**Métricas de qualidade**:
- Score de 0 a 1
- Detecção de padrões anômalos
- Sugestões de melhorias

### 4. DataMerger
**Arquivo**: `src/database/data_merger.py`

Combina dados de múltiplas fontes:
- **Resolução de conflitos**: Por qualidade, prioridade ou média
- **Deduplicação**: Hash-based para eficiência
- **Merge paralelo**: Para grandes volumes
- **Relatório de conflitos**: Documenta todas as decisões

### 5. Script de Coleta Automatizada
**Arquivo**: `scripts/collect_historical_data.py`

Automatiza todo o processo:
- **Modos**: Full, Update, Gaps, Schedule
- **Agendamento**: Coleta diária automática
- **Estado persistente**: Rastreia última coleta
- **Relatórios**: Geração automática

## Uso do Sistema

### 1. Configuração Inicial

Criar arquivo `config/collector_config.json`:
```json
{
    "symbols": ["WDOU25", "WDOV25", "WDOQ25"],
    "data_types": ["trades", "candles"],
    "db_path": "data/trading_db",
    "csv_dir": "data/csv",
    "log_dir": "logs",
    "report_dir": "reports",
    "connection": {
        "dll_path": "C:\\ProfitDLL\\profit.dll",
        "server": "demo.profit.com",
        "username": "user",
        "password": "pass"
    },
    "thresholds": {
        "min_quality_score": 0.7,
        "max_missing_pct": 0.05
    }
}
```

### 2. Coleta Inicial (6 meses)

```bash
# Coleta completa dos últimos 6 meses
python scripts/collect_historical_data.py --mode full

# Ou com datas específicas
python scripts/collect_historical_data.py --mode full \
    --start-date 2024-07-01 \
    --end-date 2025-01-27
```

### 3. Atualização Diária

```bash
# Atualizar últimos 7 dias (padrão)
python scripts/collect_historical_data.py --mode update

# Ou especificar dias
python scripts/collect_historical_data.py --mode update --days-back 14
```

### 4. Preencher Gaps

```bash
# Identificar e preencher dados faltantes
python scripts/collect_historical_data.py --mode gaps
```

### 5. Modo Agendado

```bash
# Executar coleta diária às 19h
python scripts/collect_historical_data.py --mode schedule
```

## Fluxo de Coleta

### Estratégia por Período

1. **Dados Recentes (< 3 meses)**:
   - Fonte: ProfitDLL
   - Chunks: 9 dias por vez
   - Delay: 1 segundo entre requisições

2. **Dados Históricos (> 3 meses)**:
   - Fonte primária: CSV arquivados
   - Fonte secundária: APIs externas
   - Validação: Score mínimo 0.7

### Processo de Validação

1. **Pré-validação**: Estrutura e tipos
2. **Validação de qualidade**: Outliers, gaps, duplicatas
3. **Auto-correção** (se habilitada):
   - Ordenação temporal
   - Remoção de duplicatas
   - Correção de tipos
   - Forward-fill de gaps pequenos

### Armazenamento Otimizado

```
data/
└── trading_db/
    ├── metadata.db          # SQLite com índices
    ├── parquet/            # Dados comprimidos
    │   └── WDOU25/
    │       ├── trades/
    │       │   ├── 20250127.parquet.gz
    │       │   └── 20250128.parquet.gz
    │       └── candles/
    │           ├── 20250127.parquet.gz
    │           └── 20250128.parquet.gz
    └── backups/            # Backups automáticos
```

## Limitações e Soluções

### ProfitDLL: Máximo 3 meses
**Solução**: Sistema detecta automaticamente o período e usa fontes alternativas para dados mais antigos.

### ProfitDLL: 9 dias por requisição
**Solução**: Divisão automática em chunks com delay entre requisições.

### Dados faltantes em fins de semana/feriados
**Solução**: Validador ignora fins de semana e detecta feriados baseado em volume.

### Conflitos entre fontes
**Solução**: DataMerger com múltiplas estratégias de resolução configuráveis.

## Monitoramento

### Logs
- Localização: `logs/data_collection_YYYYMMDD.log`
- Rotação: Diária
- Níveis: INFO, WARNING, ERROR

### Relatórios
- Localização: `reports/collection_report_*.txt`
- Conteúdo: Estatísticas, erros, símbolos coletados

### Estado
- Arquivo: `data/collector_state.json`
- Rastreia: Última coleta, erros, status por símbolo

## Métricas de Qualidade

### Score de Qualidade
- **> 0.9**: Excelente, pronto para produção
- **0.7-0.9**: Bom, adequado para treinamento
- **0.5-0.7**: Regular, revisar antes de usar
- **< 0.5**: Rejeitado automaticamente

### Estatísticas Disponíveis
```python
from src.database.database_manager import DatabaseManager

db = DatabaseManager()
stats = db.get_data_stats('WDOU25')

for stat in stats:
    print(f"{stat.data_type}:")
    print(f"  Período: {stat.start_date} a {stat.end_date}")
    print(f"  Registros: {stat.total_records:,}")
    print(f"  Tamanho: {stat.total_size_mb:.1f} MB")
    print(f"  Qualidade: {stat.quality_score:.2%}")
```

## Manutenção

### Backup Manual
```python
from src.database.database_manager import DatabaseManager

db = DatabaseManager()
backup_path = db.backup_database("backup_manual_20250128")
print(f"Backup criado: {backup_path}")
```

### Otimização de Armazenamento
```python
# Recomprimir dados antigos
db.optimize_storage(older_than_days=30)
```

### Limpeza de Logs
```bash
# Remover logs com mais de 30 dias
find logs/ -name "*.log" -mtime +30 -delete
```

## Integração com Sistema de Trading

### Carregar Dados para Treinamento
```python
from src.database.database_manager import DatabaseManager
from datetime import datetime, timedelta

db = DatabaseManager()

# Carregar 6 meses de dados
end_date = datetime.now()
start_date = end_date - timedelta(days=180)

trades = db.load_data(
    symbol='WDOU25',
    data_type='trades',
    start_date=start_date,
    end_date=end_date
)

print(f"Carregados {len(trades):,} trades")
```

### Validar Consistência
```python
from src.database.data_validator import DataValidator

validator = DataValidator()

# Validar consistência entre trades e candles
result = validator.validate_consistency(trades_df, candles_df)

if not result.is_valid:
    print("Inconsistências encontradas:")
    for error in result.errors:
        print(f"  - {error}")
```

## Troubleshooting

### "Connection timeout" no ProfitDLL
- Verificar credenciais e servidor
- Aumentar timeout nas configurações
- Verificar firewall/proxy

### "Dados rejeitados por baixa qualidade"
- Revisar fonte de dados
- Habilitar auto-correção
- Ajustar thresholds se apropriado

### "Memória insuficiente"
- Reduzir chunk_size
- Processar por períodos menores
- Habilitar otimização de storage

### "Conflitos excessivos no merge"
- Revisar prioridade das fontes
- Ajustar tolerâncias
- Verificar sincronização de relógios

## Conclusão

O sistema de coleta e banco de dados está completo e pronto para uso. Ele resolve automaticamente as limitações do ProfitDLL e fornece dados de alta qualidade para treinamento e backtesting do sistema ML Trading V3.

**Próximos passos**:
1. Executar coleta inicial de 6 meses
2. Configurar coleta diária automatizada
3. Treinar modelos ML com os dados coletados