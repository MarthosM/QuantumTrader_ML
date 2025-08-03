# Guia de Coleta de Book de Ofertas - ProfitDLL

**Data: 03/08/2025**

## Resumo

Este documento descreve o sistema de coleta de book de ofertas em tempo real implementado para o ProfitDLL v4.0.0.30.

## ⚠️ Limitação Importante

**O ProfitDLL NÃO fornece dados históricos de book de ofertas**. Apenas dados em tempo real estão disponíveis através de callbacks. Isso significa que:

- ✅ **Disponível**: Book em tempo real durante o pregão
- ❌ **Não disponível**: Book histórico de dias anteriores
- ❌ **Não disponível**: Função equivalente a `GetHistoryTrades` para book

## Arquitetura Implementada

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────┐
│ Book Collector  │ --> │ Servidor Isolado │ --> │ ProfitDLL  │
│  (Cliente IPC)  │ <-- │  (Callbacks)     │ <-- │ (Realtime) │
└─────────────────┘     └──────────────────┘     └────────────┘
         │                       │
         │                       ▼
         │              ┌────────────────┐
         └─────────────>│ Armazenamento  │
                        │   (Parquet)    │
                        └────────────────┘
```

## Tipos de Book Disponíveis

### 1. Offer Book (Book de Ofertas Detalhado)
- Mostra cada oferta individual com corretora
- Campos: broker_id, position, side, volume, quantity, offer_id, price
- Callback: `SetOfferBookCallbackV2`

### 2. Price Book (Book de Preços Agregado)
- Mostra níveis de preço agregados
- Campos: position, side, order_count, quantity, price
- Callback: `SetPriceBookCallbackV2`

## Como Usar

### 1. Iniciar o Servidor (se necessário)
```bash
python src/integration/profit_dll_server.py
```

### 2. Coletar Book em Tempo Real
```bash
python scripts/test_book_collection.py
```

### 3. Usar o Coletor Programaticamente
```python
from src.database.realtime_book_collector import RealtimeBookCollector

# Configurar
config = {
    'data_dir': 'data/realtime/book',
    'server_address': ('localhost', 6789)
}

# Criar coletor
collector = RealtimeBookCollector(config)

# Conectar e subscrever
if collector.connect():
    if collector.subscribe_book('WDOU25', 'both'):
        # Coletar por 5 minutos
        collector.start_collection(duration_minutes=5)
```

## Estrutura de Armazenamento

```
data/realtime/book/
└── 20250803/                      # Data
    ├── offer_book_143000.parquet  # HH:MM:SS
    ├── offer_book_143100.parquet
    ├── price_book_143000.parquet
    └── price_book_143100.parquet
```

## Formato dos Dados

### Offer Book
```python
{
    'timestamp': '03/08/2025 14:30:00.123',
    'ticker': 'WDOU25',
    'broker_id': 3,
    'position': 1,
    'side': 0,  # 0=Buy, 1=Sell
    'volume': 10000,
    'quantity': 10,
    'offer_id': 123456789,
    'price': 5650.5,
    'has_price': True,
    'has_quantity': True,
    'has_date': True,
    'has_offer_id': True,
    'is_edit': False,
    'date': '03/08/2025 14:30:00'
}
```

### Price Book
```python
{
    'timestamp': '03/08/2025 14:30:00.123',
    'ticker': 'WDOU25',
    'position': 1,
    'side': 0,  # 0=Buy, 1=Sell
    'order_count': 5,
    'quantity': 50,
    'display_quantity': 50,
    'price': 5650.5
}
```

## Callbacks Implementados

### ConnectionManagerV4
- `register_offer_book_callback()`: Registra callback para offer book
- `register_price_book_callback()`: Registra callback para price book
- `subscribe_offer_book()`: Subscreve ao offer book de um ticker
- `subscribe_price_book()`: Subscreve ao price book de um ticker

### ProfitDLLServer
- Processa comandos de subscrição/cancelamento
- Encaminha dados de book via IPC
- Gerencia fila de mensagens com limite de 10.000

## Considerações de Performance

1. **Volume de Dados**: O book pode gerar milhares de atualizações por minuto
2. **Buffer**: Dados são bufferizados e salvos a cada 60 segundos
3. **Compressão**: Arquivos Parquet com compressão Snappy
4. **Thread-safe**: Uso de locks para acesso concorrente aos buffers

## Horários de Funcionamento

- **Pregão Regular**: 9h às 17h45 (horário de Brasília)
- **After Market**: 17h50 às 18h30
- **Dias úteis**: Segunda a sexta-feira
- **Feriados**: Sem dados (mercado fechado)

## Troubleshooting

### "Market data não conectado"
- O servidor precisa estar conectado ao market data
- Aguarde mensagem "Market Connected" antes de subscrever

### "Nenhum dado recebido"
- Verifique se está dentro do horário de pregão
- Confirme que o ticker está correto (ex: WDOU25)
- Verifique logs do servidor em `logs/profit_dll_server.log`

### "Buffer overflow"
- Aumente o intervalo de salvamento se necessário
- Considere filtrar apenas níveis específicos do book

## Próximos Passos

1. **Análise de Microestrutura**: Usar dados de book para análise de liquidez
2. **Detecção de Iceberg**: Identificar ordens ocultas no book
3. **Order Flow**: Combinar book com trades para análise de fluxo
4. **Imbalance**: Calcular desequilíbrio entre bid/ask

## Exemplo de Análise

```python
import pandas as pd

# Carregar dados
df_offer = pd.read_parquet('data/realtime/book/20250803/offer_book_143000.parquet')

# Separar bid/ask
bid_book = df_offer[df_offer['side'] == 0].sort_values('position')
ask_book = df_offer[df_offer['side'] == 1].sort_values('position')

# Calcular spread
best_bid = bid_book.iloc[0]['price'] if not bid_book.empty else 0
best_ask = ask_book.iloc[0]['price'] if not ask_book.empty else 0
spread = best_ask - best_bid

print(f"Best Bid: {best_bid}")
print(f"Best Ask: {best_ask}")
print(f"Spread: {spread}")

# Volume no topo do book
bid_volume = bid_book.iloc[0]['quantity'] if not bid_book.empty else 0
ask_volume = ask_book.iloc[0]['quantity'] if not ask_book.empty else 0
print(f"Bid Volume: {bid_volume}")
print(f"Ask Volume: {ask_volume}")
```

## Conclusão

O sistema de coleta de book está **100% operacional** para dados em tempo real. A limitação de não haver dados históricos é uma característica da API ProfitDLL, não do sistema implementado.

Para análises que requerem histórico de book, será necessário:
1. Coletar e armazenar dados continuamente durante o pregão
2. Construir sua própria base histórica ao longo do tempo
3. Considerar fontes alternativas para dados históricos de book