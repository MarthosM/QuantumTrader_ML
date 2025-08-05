# Comandos para Coleta de Book e Treinamento

## 📊 Status Atual

- ✅ **Conexão ProfitDLL**: Funcionando corretamente
- ✅ **Callbacks**: Configurados com WinDLL/WINFUNCTYPE 
- ✅ **Login/Market**: Conectando com sucesso
- ⚠️ **Dados**: Só disponíveis durante pregão (seg-sex 09:00-18:00)

## 🚀 Comandos Principais

### 1. Coletar Dados de Book (Durante Pregão)

```bash
# Versão mais estável e testada
python book_collector_final.py

# Alternativas:
python book_collector_fixed_v2.py
python book_collector_complete.py  # Original (pode precisar ajustes)
```

### 2. Verificar Dados Coletados

```bash
# Checar arquivos salvos
python scripts/check_book_data.py

# Ver estrutura dos dados
python scripts/view_historical_data.py

# Verificar diretório de dados
dir data\realtime\book\20250804\*.parquet
```

### 3. Treinar Modelos com Book

```bash
# Sistema completo dual (tick + book)
python examples/train_dual_models.py

# Treinar apenas com book
python -m src.training.book_training_pipeline --symbol WDOQ25 --days 30

# Pipeline de tick-only
python -m src.training.tick_training_pipeline --symbol WDOQ25 --days 365
```

### 4. Executar Sistema de Trading

```bash
# Com modelos treinados
python examples/hmarl_integrated_trading.py

# Sistema principal
python src/main.py
```

## 📁 Estrutura de Arquivos

```
data/
├── realtime/
│   └── book/
│       └── YYYYMMDD/
│           ├── book_data_HHMMSS.parquet
│           └── book_data_HHMMSS.json
└── historical/
    └── SYMBOL/
        └── YYYYMMDD/
            └── trades.parquet

models/
├── tick_only/
│   └── SYMBOL/
│       └── REGIME/
│           └── model.pkl
└── book_enhanced/
    └── SYMBOL/
        └── TARGET/
            └── model.pkl
```

## ⚠️ Requisitos Importantes

### Para Coletar Book:
1. **Profit Chart PRO** deve estar aberto e logado
2. **Mercado aberto** (segunda a sexta, 09:00-18:00)
3. **Credenciais** configuradas nas variáveis de ambiente

### Para Treinar:
1. Pelo menos **5-10 dias** de dados de book
2. **Dados históricos** de tick (já disponíveis)
3. **8GB+ RAM** para processar features

## 🔧 Troubleshooting

### Erro: "Nenhum dado coletado"
- Verificar se é dia útil e horário de pregão
- Confirmar que Profit Chart está aberto
- Testar com `python test_connection_v4_final.py`

### Erro: "DLL não encontrada"
```bash
# Copiar DLL se necessário
copy C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll ProfitDLL64.dll
```

### Erro: "OverflowError no callback"
- Use `book_collector_final.py` (versão simplificada)
- Evita problemas com TranslateTrade

## 📊 Exemplo de Uso Completo

```bash
# 1. Segunda-feira 09:30 - Iniciar coleta
python book_collector_final.py

# 2. Deixar rodando durante o dia...

# 3. Final do dia - Verificar dados
python scripts/check_book_data.py

# 4. Após 5-10 dias - Treinar modelos
python examples/train_dual_models.py

# 5. Testar em simulação
python src/main.py --mode simulation

# 6. Executar em produção
python examples/hmarl_integrated_trading.py
```

## 🎯 Próximos Passos

1. **Aguardar abertura do mercado** (segunda-feira)
2. **Coletar dados por 1 semana**
3. **Treinar modelos dual**
4. **Validar em backtest**
5. **Deploy em produção**

---

💡 **Dica**: Execute `python book_collector_final.py` em uma tela/tmux separada para coletar durante todo o pregão.