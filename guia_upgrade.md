# Prompt para Upgrade do Sistema ML Trading v2.0

## Contexto do Sistema

Você está trabalhando no upgrade de um sistema de trading automatizado com Machine Learning (ML Trading v2.0) que já está funcional mas precisa de melhorias significativas. O sistema atual:

* Utiliza modelos XGBoost/LightGBM para predição
* Detecta regimes de mercado (tendência/lateralização)
* Opera no mercado de futuros WDO através da plataforma Profit
* Possui 32 features finais após processamento
* Tem arquitetura modular com gestão de risco integrada

## Objetivo do Upgrade

Implementar melhorias avançadas baseadas nas melhores práticas de ML para day trading, incluindo:

* Sistema de otimização contínua de hiperparâmetros
* Feature engineering adaptativo
* Ensemble multi-modal com modelos avançados
* Gestão de risco inteligente com ML
* Monitoramento e adaptação em tempo real
* Sistema de backtesting robusto

## Instruções para Implementação

### Diretrizes Gerais:

1. **NÃO use emojis** no código
2. **Prefira corrigir métodos existentes** antes de criar novos
3. **Mantenha compatibilidade** com a estrutura atual
4. **Preserve funcionalidades** existentes enquanto adiciona novas

### Formato de Correções:

Ao corrigir código, siga este padrão:

```
**Arquivo:** nome_do_arquivo.py
**Localização:** Linha ~XXX ou método specific_method()
**Contexto:** Breve descrição do que está sendo modificado

ANTES:
```python
# código atual
```

DEPOIS:

```python
# código corrigido
```

**Justificativa:** Explicação da mudança

```

## Etapas de Upgrade

### ETAPA 1: Análise e Preparação (Diagnóstico)
1. Analisar o sistema atual identificando:
   - Pontos de melhoria em cada módulo
   - Gargalos de performance
   - Features não otimizadas
   - Limitações dos modelos atuais

2. Criar relatório técnico com:
   - Mapeamento de mudanças necessárias
   - Ordem de prioridade
   - Dependências entre módulos
   - Riscos de cada alteração

### ETAPA 2: Feature Engineering Avançado
1. Implementar `AdvancedFeatureProcessor` com:
   - Sistema multi-camadas de features
   - Features de microestrutura para day trade
   - Indicadores técnicos adaptativos
   - Seleção inteligente de features (Hughes Phenomenon aware)

2. Atualizar `feature_engine.py` e `ml_features.py`:
   - Adicionar cálculo de features avançadas
   - Implementar feature selection automático
   - Otimizar pipeline de processamento

### ETAPA 3: Upgrade dos Modelos ML
1. Implementar `MultiModalEnsemble` em `model_manager.py`:
   - XGBoost otimizado para velocidade
   - LSTM para padrões intraday
   - Transformer com atenção
   - Sistema de pesos dinâmicos por regime

2. Adicionar `HyperparameterOptimizer`:
   - Otimização Bayesiana
   - Grid Search adaptativo
   - Validação temporal rigorosa

### ETAPA 4: Sistema de Otimização Contínua
1. Criar `ContinuousOptimizationPipeline`:
   - Otimização de features
   - Otimização de hiperparâmetros
   - Otimização de portfolio
   - Otimização de execução

2. Implementar `AutoOptimizationEngine`:
   - Detecção de drift
   - Retreinamento automático
   - Ajuste dinâmico de parâmetros

### ETAPA 5: Gestão de Risco Inteligente
1. Upgrade do `risk_manager.py` com:
   - `VolatilityPredictor` usando ensemble
   - `DynamicStopLossOptimizer`
   - Position sizing com ML
   - Correlação dinâmica

2. Implementar novos controles:
   - Predição de drawdown
   - Otimização de margem
   - Risk scoring combinado

### ETAPA 6: Execução e Latência
1. Criar `SmartExecutionEngine`:
   - Timing optimizer com ML
   - Slippage predictor
   - Order slicing adaptativo
   - Impact estimator

2. Otimizar `execution_optimizer.py`:
   - Reduzir latência para <100ms
   - Implementar smart routing
   - Adicionar execution analytics

### ETAPA 7: Backtesting Avançado
1. Implementar `AdvancedMLBacktester`:
   - Walk-forward validation
   - Purged cross-validation
   - Simulação realística com custos
   - Stress testing

2. Adicionar métricas avançadas:
   - Sharpe ratio
   - Calmar ratio
   - Maximum drawdown
   - Win rate por regime

### ETAPA 8: Monitoramento em Tempo Real
1. Criar `RealTimeMonitoringSystem`:
   - Performance monitor
   - Model drift detector
   - Risk monitor
   - Alert system

2. Implementar dashboards:
   - Métricas em tempo real
   - Visualização de posições
   - Análise de performance

### ETAPA 9: Testes e Validação
1. Criar suite de testes completa:
   - Testes unitários para cada módulo
   - Testes de integração
   - Testes de stress
   - Validação de performance

2. Executar validação completa:
   - Paper trading por 1 semana
   - Comparação com sistema atual
   - Análise de métricas

### ETAPA 10: Deploy e Documentação
1. Preparar deploy gradual:
   - Migração de configurações
   - Backup do sistema atual
   - Rollback plan
   - Monitoramento pós-deploy

2. Atualizar documentação:
   - Novos fluxos de dados
   - APIs atualizadas
   - Guia de troubleshooting
   - Manual de operação

## Critérios de Sucesso

1. **Performance**: Win rate > 55%, Sharpe ratio > 1.5
2. **Latência**: Predição < 100ms, execução < 200ms  
3. **Robustez**: Uptime > 99.5%, recuperação automática
4. **Adaptabilidade**: Ajuste automático a mudanças de regime
5. **Escalabilidade**: Suporte para múltiplos ativos

## Ordem de Implementação Recomendada

Para minimizar riscos e maximizar valor:
1. Começar com melhorias no feature engineering (menor risco)
2. Seguir com upgrade dos modelos (impacto médio)
3. Implementar otimização contínua (alto valor)
4. Finalizar com execução e monitoramento (crítico)

## Notas Importantes

- Mantenha sempre um branch de backup do código atual
- Teste cada etapa extensivamente antes de prosseguir
- Documente todas as mudanças realizadas
- Monitore métricas de performance durante todo o processo
- Mantenha compatibilidade com a DLL Profit existente

Comece pela **ETAPA 1** fazendo uma análise completa do sistema atual e identificando os pontos específicos que precisam ser modificados em cada arquivo.
```
