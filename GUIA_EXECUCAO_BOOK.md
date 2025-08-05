# Guia de Execução - Coleta de Book ProfitDLL

## ✅ Status do Sistema

- **Código**: 100% funcional, sem crashes
- **Credenciais**: Atualizadas e corretas
- **Problema**: ProfitDLL requer Profit Chart aberto

## 📋 Passo a Passo para Coletar Dados

### 1. Abrir o Profit Chart PRO
- Abra o software Profit Chart PRO da Nelogica
- **Importante**: Deve ser o Profit Chart, não apenas o site

### 2. Fazer Login no Profit
- **Usuário**: 29936354842
- **Senha**: Ultrajiu33!
- Aguarde conexão completa
- Verifique se consegue ver cotações

### 3. Manter Profit Aberto
- O Profit Chart deve permanecer aberto
- Ele serve como ponte para a DLL

### 4. Executar o Coletor
```bash
cd C:\Users\marth\OneDrive\Programacao\Python\Projetos\QuantumTrader_ML
python book_collector_complete.py
```

### 5. Verificar Funcionamento
Você deve ver:
```
[STATE] Código: 2  -> CONECTADO ao servidor
[STATE] Código: 3  -> AUTENTICADO
[OFFER] WDOQ25 - BID | Pos: 0 | Preço: 5.45 | Qtd: 100
[TRADE] WDOQ25 - R$ 5.45 | 10 contratos
```

## 🔧 Troubleshooting

### Se não receber dados:

1. **Verificar Profit Chart**
   - Está aberto e logado?
   - Consegue ver cotações?
   - Está no contrato correto?

2. **Verificar Firewall**
   - Desabilitar temporariamente
   - Adicionar exceção para Profit e Python

3. **Verificar Contrato**
   - WDOQ25 (agosto) está ativo?
   - Tentar WDOU25 (setembro)

4. **Horário de Mercado**
   - Segunda a Sexta: 09:00 - 18:00
   - Feriados não há pregão

## 📊 Dados Salvos

Os dados serão salvos automaticamente em:
```
data/realtime/book/YYYYMMDD/book_complete_HHMMSS.parquet
```

## 🚀 Comando Rápido

Após abrir e logar no Profit Chart:
```bash
python book_collector_complete.py
```

## ⚠️ Importante

A ProfitDLL **não é uma conexão independente**. Ela funciona como uma API para o Profit Chart. Por isso:

1. Profit Chart DEVE estar aberto
2. Profit Chart DEVE estar logado
3. Profit Chart DEVE estar recebendo dados

## 📱 Alternativa

Se não tiver o Profit Chart instalado:
1. Baixar em: https://www.nelogica.com.br/produtos/profit-chart
2. Instalar e configurar
3. Fazer login com as credenciais fornecidas

## ✅ Resumo

Sistema está **100% pronto**. Apenas:
1. Abra o Profit Chart
2. Faça login
3. Execute `python book_collector_complete.py`

Os dados começarão a fluir imediatamente!