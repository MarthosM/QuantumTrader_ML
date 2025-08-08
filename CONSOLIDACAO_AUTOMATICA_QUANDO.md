# Quando a Consolidação Automática é Acionada

## Opções de Automação Disponíveis

### 1. **Durante a Coleta Contínua** (Integrado no `book_collector_continuous.py`)
- **Quando**: A cada hora cheia (10:00, 11:00, 12:00, etc.)
- **Como funciona**: 
  - Durante os primeiros 5 minutos de cada hora
  - Executa `auto_consolidate_book_data.py` automaticamente
  - Não interrompe a coleta de dados
- **Vantagem**: Dados sempre atualizados durante o dia
- **Uso**: Já está ativo quando você executa `python book_collector_continuous.py`

### 2. **Agendamento Separado** (`schedule_consolidation.py`)
Várias opções de agendamento:

#### a) **Após Fechamento do Mercado**
```bash
python schedule_consolidation.py market_close
```
- Consolida às 18:05 (após fechamento)
- Ideal para processamento diário final

#### b) **A Cada Hora**
```bash
python schedule_consolidation.py hourly
```
- Consolida no minuto :05 de cada hora
- Verifica se já foi consolidado recentemente

#### c) **A Cada 30 Minutos**
```bash
python schedule_consolidation.py every_30min
```
- Consolidação mais frequente
- Para análise quase em tempo real

#### d) **Modo Contínuo**
```bash
python schedule_consolidation.py continuous
```
- Verifica a cada 5 minutos se há novos dados
- Consolida apenas se necessário

### 3. **Execução Manual**
```bash
# Consolidar dados de hoje
python auto_consolidate_book_data.py

# Consolidar data específica
python auto_consolidate_book_data.py 20250805
```

## Recomendações de Uso

### Para Day Trading / Análise Intraday
- Use o **book_collector_continuous.py** (já tem consolidação automática a cada hora)
- Ou rode `schedule_consolidation.py hourly` em paralelo

### Para Análise End-of-Day
- Use `schedule_consolidation.py market_close`
- Roda uma vez após fechamento do mercado

### Para Backtesting / Pesquisa
- Execute manualmente quando necessário
- Use os arquivos consolidados em `data/realtime/book/YYYYMMDD/consolidated/`

## Arquivos Gerados pela Consolidação

Sempre que a consolidação é executada, ela cria:

1. **Por Tipo de Dado**:
   - `consolidated_offer_book_YYYYMMDD.parquet`
   - `consolidated_tiny_book_YYYYMMDD.parquet`
   - `consolidated_daily_YYYYMMDD.parquet`

2. **Completo**:
   - `consolidated_complete_YYYYMMDD.parquet` (todos os dados)

3. **Por Hora** (se configurado):
   - `consolidated_hour_09_YYYYMMDD.parquet`
   - `consolidated_hour_10_YYYYMMDD.parquet`
   - etc.

4. **Otimizado para ML**:
   - `training_data_YYYYMMDD.parquet` (com features temporais adicionadas)

## Status da Consolidação

Para verificar se os dados já foram consolidados:
- Procure o arquivo `consolidation_metadata_YYYYMMDD.json`
- Contém timestamp da última consolidação
- Se modificado há menos de 1 hora, não reconsolida

## Integração com Pipeline de Treinamento

O sistema de treinamento pode detectar automaticamente novos dados consolidados:

```python
from src.training.flexible_data_loader import FlexibleBookDataLoader

loader = FlexibleBookDataLoader()
# Carrega automaticamente os últimos 7 dias consolidados
data = loader.load_for_training(date_range=7)