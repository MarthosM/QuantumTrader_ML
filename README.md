# ML Trading v2.0

Um sistema avançado de trading algorítmico usando Machine Learning para análise de mercado financeiro.

## 🚀 Características

- **Machine Learning**: Utiliza algoritmos de ML para previsão de preços
- **Análise Técnica**: Implementa diversos indicadores técnicos usando a biblioteca `ta`
- **Otimização de Hiperparâmetros**: Integração com Optuna para otimização automática
- **Múltiplos Modelos**: Suporte para LightGBM, XGBoost e Scikit-learn
- **Balanceamento de Classes**: Uso do imbalanced-learn para lidar com dados desbalanceados
- **Visualização**: Gráficos e análises com Matplotlib e Seaborn
- **Testes**: Suite de testes com pytest

## 📋 Pré-requisitos

- Python 3.12+
- pip ou conda

## 🛠️ Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/ML_Tradingv2.0.git
cd ML_Tradingv2.0
```

2. Crie um ambiente virtual:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# ou
source .venv/bin/activate  # Linux/Mac
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 📦 Dependências Principais

- **pandas & numpy**: Manipulação e análise de dados
- **scikit-learn**: Algoritmos de machine learning
- **lightgbm & xgboost**: Modelos de gradient boosting
- **ta**: Análise técnica
- **optuna**: Otimização de hiperparâmetros
- **matplotlib & seaborn**: Visualização de dados
- **SQLAlchemy**: ORM para banco de dados
- **python-dotenv**: Gerenciamento de variáveis de ambiente

## 🏗️ Estrutura do Projeto

```
ML_Tradingv2.0/
│
├── data/                 # Dados de mercado
├── models/              # Modelos treinados
├── src/                 # Código fonte
│   ├── data/           # Processamento de dados
│   ├── features/       # Engenharia de features
│   ├── models/         # Modelos de ML
│   └── utils/          # Utilitários
├── tests/              # Testes
├── notebooks/          # Jupyter notebooks
└── config/             # Arquivos de configuração
```

## 📊 Uso

(Instruções de uso serão adicionadas conforme o desenvolvimento)

## 🧪 Testes

Execute os testes com:
```bash
pytest
```

## 📈 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

## ⚠️ Disclaimer

Este software é para fins educacionais e de pesquisa. Trading de ativos financeiros envolve riscos significativos. Use por sua própria conta e risco.
