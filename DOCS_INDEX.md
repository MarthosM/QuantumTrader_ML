# 📚 Índice de Documentação - ML Trading v2.0

## 🗺️ Documentação Técnica Principal

### 🎯 **ESSENCIAL** - Para IA/Copilot
| Arquivo | Propósito | Uso pela IA |
|---------|-----------|-------------|
| `src/features/complete_ml_data_flow_map.md` | **Mapeamento completo do fluxo de dados** | ⭐ **OBRIGATÓRIO** - Referência arquitetural |
| `.github/copilot-instructions.md` | Instruções personalizadas do GitHub Copilot | ⭐ **ESSENCIAL** - Diretrizes de desenvolvimento |
| `.copilot-instructions.md` | Versão resumida das instruções | 🔖 Referência rápida |

### 🔧 Configuração de Ambiente
| Arquivo | Propósito |
|---------|-----------|
| `.vscode/settings.json` | Configurações VS Code otimizadas para IA |
| `.vscode/workspace.code-workspace` | Workspace configurado |
| `requirements.txt` | Dependências Python |
| `.env.example` | Variáveis de ambiente |

### 📋 Documentação do Projeto
| Arquivo | Propósito |
|---------|-----------|
| `README.md` | Visão geral e setup do projeto |
| `tests/test_etapa1.py` | Padrões de teste estabelecidos |
| Este arquivo (`DOCS_INDEX.md`) | Índice de toda documentação |

## 🏗️ Arquitetura de Referência

### 📊 Fluxo Principal de Dados
```
1. ModelManager → Carrega modelos e identifica features necessárias
2. DataStructure → Organiza dados em DataFrames separados
3. FeatureGenerator → Calcula indicadores e features ML
4. MLIntegration → Executa predições com ensemble
5. TradingStrategy → Gera sinais de trading
```

### 🗂️ Estrutura de Classes
```
src/
├── connection_manager.py    # Interface com ProfitDLL
├── model_manager.py        # Gestão de modelos ML
├── data_structure.py       # Estrutura centralizada de dados
├── data/                   # Processamento de dados
├── features/               # Engenharia de features
│   └── complete_ml_data_flow_map.md  ⭐ DOCUMENTO PRINCIPAL
├── models/                 # Modelos treinados
└── utils/                  # Utilitários
```

## 🤖 Como a IA Deve Usar Esta Documentação

### 1. **Início de Qualquer Tarefa**
```
1. Ler: src/features/complete_ml_data_flow_map.md
2. Consultar: .github/copilot-instructions.md
3. Verificar: Padrões em tests/test_etapa1.py
```

### 2. **Durante Desenvolvimento**
- **Sempre** seguir o fluxo de dados mapeado
- **Validar** compatibilidade com ModelManager
- **Manter** padrões de DataFrames separados
- **Usar** threading para operações pesadas

### 3. **Para Novos Módulos**
- **Seguir** estrutura estabelecida em src/
- **Implementar** testes seguindo padrão pytest
- **Documentar** seguindo formato markdown
- **Validar** integração com fluxo existente

## 📈 Métricas de Qualidade

### ✅ Indicadores de Qualidade
- **Features**: ~80-100 por modelo
- **Cobertura de Testes**: >90%
- **Performance**: <1s para predições
- **Validação**: Rigorosa em cada etapa

### 🎯 Padrões Obrigatórios
- **Logging**: Por classe com getLogger
- **Typing**: Hints completos
- **Validação**: Dados e parâmetros
- **Threading**: Para cálculos pesados
- **Testes**: Com e sem dependências externas

---

## 🚀 Quick Start para IA

1. **Ler primeiro**: `src/features/complete_ml_data_flow_map.md`
2. **Entender padrões**: `tests/test_etapa1.py`
3. **Seguir diretrizes**: `.github/copilot-instructions.md`
4. **Implementar com qualidade**: Validação + Logging + Testes

**Lembre-se**: Este é um sistema de trading real - precisão e confiabilidade são fundamentais!
