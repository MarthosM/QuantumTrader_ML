# Status da Coleta de Dados Históricos - ProfitDLL

## 📅 Data: 02/08/2025

## 🎯 Objetivo
Implementar coleta de dados históricos do ProfitDLL para criar base de dados para treinamento do sistema HMARL.

## ✅ O que foi implementado

### 1. Sistema de Coleta Completo
- **HistoricalDataCollector** (`src/database/historical_data_collector.py`)
  - Coleta dados históricos via servidor isolado
  - Divide períodos grandes em blocos de 9 dias
  - Armazena dados em formato Parquet (comprimido e eficiente)
  - Suporta diferentes tipos de dados (trades, book, etc.)

### 2. Servidor Isolado
- **ProfitDLLServer** (`src/integration/profit_dll_server.py`)
  - Roda ProfitDLL em processo separado para evitar Segmentation Fault
  - Comunicação via IPC (Inter-Process Communication)
  - Gerencia callbacks e requisições de forma segura

### 3. Integração com ConnectionManagerV4
- Adicionados métodos:
  - `register_history_trade_callback()` - registra callback para dados históricos
  - `get_history_trades()` - solicita dados históricos
- Configurado para usar exchange "F" para futuros (não "BMF")

### 4. Scripts de Teste
- `collect_recent_data.py` - coleta últimos 30 dias
- `run_full_historical_collection.py` - coleta completa (3 meses)
- `test_wdou25_current.py` - testa contrato atual
- Scripts de diagnóstico diversos

## ❌ Problema Encontrado

### Sintomas
1. GetHistoryTrades aceita a requisição (retorna 0)
2. Mostra progresso de 0% a 100%
3. Mas **não retorna nenhum dado** no callback

### Código de Erro
- Quando usamos exchange "BMF": erro -2147483645
- Quando usamos exchange "F": requisição aceita mas sem dados

### Testes Realizados
1. ✅ Diferentes formatos de data (DD/MM/YYYY e DD/MM/YYYY HH:mm:SS)
2. ✅ Diferentes exchanges ("F", "BMF", "")
3. ✅ Diferentes contratos (WDOU25, WDOQ25, WDO)
4. ✅ Diferentes períodos (1 dia, 7 dias, 30 dias)
5. ✅ Datas passadas (2024 e 2025)

## 🔍 Diagnóstico

### Possíveis Causas
1. **Mercado Fechado**: Testando fora do horário de pregão (9h-18h)
2. **Dados Futuros**: Pedindo dados de 01/08/2025 (ontem) que podem não estar disponíveis
3. **Limitações da Conta**: Conta pode não ter permissão para dados históricos
4. **Limitações da API**: GetHistoryTrades pode ter restrições não documentadas

### O que está funcionando
- ✅ Conexão com ProfitDLL
- ✅ Autenticação bem-sucedida
- ✅ Callbacks configurados corretamente
- ✅ Servidor isolado evita Segmentation Fault
- ✅ GetHistoryTrades disponível e aceita requisições

## 🚀 Próximos Passos

### 1. Testar Durante o Pregão
Execute o sistema durante horário de mercado aberto (9h-18h) para verificar se os dados ficam disponíveis.

### 2. Verificar com Suporte ProfitDLL
Confirmar:
- Se a conta tem acesso a dados históricos
- Se há limitações específicas na API
- Formato correto de parâmetros

### 3. Alternativa: Coleta em Tempo Real
Como o sistema já está preparado, pode-se:
- Iniciar coleta de dados em tempo real durante o pregão
- Acumular base histórica gradualmente
- Usar esses dados para treinar o HMARL

### 4. Verificar Documentação Atualizada
Buscar atualizações na documentação do ProfitDLL sobre:
- Mudanças na API GetHistoryTrades
- Novos requisitos ou parâmetros
- Limitações conhecidas

## 💻 Como Usar o Sistema

### Coleta de Dados Recentes
```bash
python scripts/collect_recent_data.py
```

### Coleta Completa (3 meses)
```bash
python scripts/run_full_historical_collection.py
```

### Teste Direto
```bash
python scripts/simple_history_test.py
```

## 📊 Estrutura de Dados

### Formato de Armazenamento
```
data/
├── historical/
│   ├── WDOU25/
│   │   ├── 20250801.parquet
│   │   ├── 20250802.parquet
│   │   └── ...
│   └── WDOQ25/
│       └── ...
└── csv/  # Exportação opcional em CSV
```

### Campos dos Dados
- `timestamp`: Data/hora do trade
- `price`: Preço
- `volume`: Volume financeiro
- `quantity`: Quantidade
- `trade_type`: Tipo (compra/venda)
- `buy_agent`: Agente comprador
- `sell_agent`: Agente vendedor

## 🔧 Configuração

### Variáveis de Ambiente (.env)
```env
PROFIT_USERNAME=seu_usuario
PROFIT_PASSWORD=sua_senha
PROFIT_KEY=sua_chave
```

### Configuração do Coletor
```json
{
  "symbols": ["WDOU25"],
  "data_types": ["trades"],
  "data_dir": "data/historical",
  "dll_path": "C:\\path\\to\\ProfitDLL.dll"
}
```

## 📝 Conclusão

O sistema de coleta histórica está **completamente implementado** e pronto para uso. No entanto, não está conseguindo recuperar dados históricos do ProfitDLL nas condições atuais de teste.

Recomenda-se:
1. Testar durante o pregão
2. Verificar permissões da conta
3. Considerar coleta em tempo real como alternativa

O código está robusto, com tratamento de erros, logging detalhado e arquitetura escalável para quando os dados estiverem disponíveis.