# Guia de Resolução - Sistema não recebe dados em tempo real

## 🔴 Problema Atual
- Sistema mostra preço de R$ 5493.50 (dado antigo)
- Mercado real está em R$ 5465
- Erro de subscrição: -2147483647
- Login Error: 200

## 📋 Passos para Resolver (Execute em ordem)

### 1. Limpar Processos Travados
```bash
python clean_python_processes.py
```
- Mata todos os processos Python relacionados ao trading
- Libera a DLL de possíveis travas

### 2. Verificar Status do Sistema
```bash
python check_profitchart_status.py
```
- Verifica se ProfitChart está instalado e rodando
- Testa conectividade de rede
- Verifica firewall e DLLs

### 3. Teste Minimalista de Conexão
```bash
# Aguarde 1 minuto após limpar processos
python test_connection_minimal.py
```
- Testa conexão básica com a DLL
- Usa ticker WDOV25 (outubro)
- Mostra exatamente onde está falhando

### 4. Se Ainda Não Funcionar

#### A. Verificar ProfitChart
1. Abra o ProfitChart manualmente
2. Faça login com suas credenciais
3. Verifique se consegue ver cotações do WDO
4. Se funcionar lá, o problema é específico da API

#### B. Reiniciar Completamente
1. Feche TODOS os programas relacionados
2. Reinicie o computador
3. Aguarde 5 minutos
4. Tente novamente: `python test_connection_minimal.py`

#### C. Aguardar Desbloqueio
Se nada funcionar, provavelmente há um bloqueio temporário:
1. Aguarde 30-60 minutos
2. NÃO tente múltiplas conexões nesse período
3. Após aguardar, execute apenas: `python test_connection_minimal.py`

### 5. Quando Voltar a Funcionar

Execute o sistema de produção otimizado:
```bash
python production_fixed.py
```

## 🔍 Interpretando os Erros

### Login Error 200
- Problema de autenticação/permissão
- Conta pode estar bloqueada temporariamente
- Credenciais podem estar incorretas

### Subscribe Error -2147483647
- Falha total na subscrição ao ticker
- Servidor de dados não está respondendo
- Limite de conexões atingido

### STATE Broker: 0 (repetido)
- Sistema está desconectando repetidamente
- Indica instabilidade na conexão

## ⚠️ Causas Mais Prováveis

1. **Múltiplas Conexões Simultâneas**
   - Vários scripts rodando ao mesmo tempo
   - DLL não finalizou corretamente
   - Solução: Limpar processos e aguardar

2. **Bloqueio Temporário da Conta**
   - Muitas tentativas de conexão
   - Sistema de segurança da corretora
   - Solução: Aguardar 30-60 minutos

3. **Problema de Servidor**
   - Manutenção não programada
   - Instabilidade temporária
   - Solução: Verificar com ProfitChart

## ✅ Checklist de Verificação

- [ ] Processos Python limpos
- [ ] ProfitChart funcionando normalmente
- [ ] Internet conectada
- [ ] Firewall não está bloqueando
- [ ] DLLs presentes no diretório
- [ ] Aguardou tempo suficiente (se bloqueado)

## 📞 Último Recurso

Se após todos os passos o problema persistir:
1. Entre em contato com o suporte da corretora
2. Informe o erro específico: -2147483647
3. Pergunte sobre limites de conexão da conta
4. Verifique se há manutenção programada

## 🎯 Lembre-se

- O código está **correto e funcional**
- O sistema funcionou perfeitamente das 16:16 às 16:23
- O problema é **externo** (conectividade/autenticação)
- Quando a conexão for restabelecida, tudo voltará a funcionar