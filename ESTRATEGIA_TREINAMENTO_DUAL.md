# Estratégia de Treinamento Dual: CSV + Book Collector

## 📊 Visão Geral

Como a coleta via Book Collector é limitada em volume, implementaremos uma estratégia dual:

1. **Modelos Base** - Treinados com grandes volumes de dados CSV históricos
2. **Modelos HMARL** - Treinados/ajustados com dados de microestrutura do Book

## 🎯 Arquitetura Proposta

### Tier 1: Modelos Base (CSV)
```python
# Modelos treinados com anos de dados históricos
base_models = {
    'price_prediction': XGBoostRegressor(),      # Previsão de preço
    'trend_classifier': RandomForestClassifier(), # Classificação de tendência
    'volatility_model': LightGBMRegressor(),     # Estimativa de volatilidade
    'volume_predictor': CatBoostRegressor()      # Previsão de volume
}

# Features tradicionais do CSV
csv_features = [
    'returns', 'volume', 'vwap', 'rsi', 'macd',
    'bollinger_bands', 'support_resistance'
]
```

### Tier 2: Modelos HMARL (Book)
```python
# Modelos especializados em microestrutura
hmarl_models = {
    'flow_toxicity': FlowToxicityModel(),        # Análise de fluxo tóxico
    'market_impact': MarketImpactModel(),        # Impacto de mercado
    'order_imbalance': OrderImbalanceModel(),    # Desequilíbrio do book
    'liquidity_prediction': LiquidityModel()     # Previsão de liquidez
}

# Features de microestrutura
book_features = [
    'bid_ask_spread', 'book_imbalance', 'order_flow',
    'queue_position', 'aggressive_trades', 'hidden_liquidity'
]
```

## 🔄 Pipeline de Treinamento

### Fase 1: Treinamento Base (Offline)
```python
class BaseModelTraining:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.models = {}
        
    def train_all_base_models(self):
        # 1. Carregar dados históricos (anos)
        df = pd.read_csv(self.csv_path)
        
        # 2. Gerar features tradicionais
        features = self.generate_traditional_features(df)
        
        # 3. Treinar cada modelo base
        for model_name, model in base_models.items():
            X, y = self.prepare_data(features, model_name)
            
            # Walk-forward validation
            model.fit(X_train, y_train)
            self.models[model_name] = model
            
        # 4. Salvar modelos
        self.save_models('models/base/')
```

### Fase 2: Treinamento HMARL (Online/Incremental)
```python
class HMARLModelTraining:
    def __init__(self, book_collector):
        self.book_collector = book_collector
        self.models = {}
        self.min_samples = 10000  # Mínimo para treinar
        
    def incremental_training(self):
        # 1. Coletar dados de book em tempo real
        book_data = self.book_collector.get_latest_data()
        
        # 2. Quando atingir volume mínimo
        if len(book_data) >= self.min_samples:
            # 3. Gerar features de microestrutura
            features = self.generate_microstructure_features(book_data)
            
            # 4. Treinar/atualizar modelos HMARL
            for model_name, model in hmarl_models.items():
                if hasattr(model, 'partial_fit'):
                    # Aprendizado incremental
                    model.partial_fit(features, labels)
                else:
                    # Re-treinar com janela deslizante
                    model.fit(features[-50000:], labels[-50000:])
```

## 🎭 Sistema de Ensemble

### Combinação Inteligente
```python
class DualModelEnsemble:
    def __init__(self, base_models, hmarl_models):
        self.base_models = base_models
        self.hmarl_models = hmarl_models
        self.weights = self.calculate_optimal_weights()
        
    def predict(self, market_data, book_data=None):
        predictions = {}
        
        # 1. Sempre usar modelos base (sempre disponíveis)
        base_pred = self.base_models.predict(market_data)
        predictions['base'] = base_pred
        
        # 2. Se tiver dados de book, usar HMARL
        if book_data is not None and self.hmarl_models_ready():
            hmarl_pred = self.hmarl_models.predict(book_data)
            predictions['hmarl'] = hmarl_pred
            
            # 3. Combinar predições com pesos adaptativos
            final_pred = self.weighted_combination(predictions)
        else:
            # Usar apenas base se HMARL não disponível
            final_pred = base_pred
            
        return final_pred
        
    def weighted_combination(self, predictions):
        # Pesos dinâmicos baseados em:
        # - Confiança do modelo
        # - Performance recente
        # - Condições de mercado
        return weighted_average(predictions, self.weights)
```

## 📈 Vantagens da Abordagem

### 1. **Robustez**
- Sempre temos predições (modelos base)
- HMARL adiciona precisão quando disponível

### 2. **Escalabilidade**
- Base: Treina offline com TB de dados
- HMARL: Aprende incrementalmente

### 3. **Especialização**
- Base: Padrões macro (tendências, ciclos)
- HMARL: Padrões micro (fluxo, liquidez)

## 🚀 Implementação Prática

### Configuração
```yaml
# config/dual_training.yaml
base_models:
  data_source: "C:/Users/marth/Downloads/WINFUT_SAMPLE"
  features:
    - technical_indicators
    - price_patterns
    - volume_profile
  training:
    lookback_years: 5
    validation_split: 0.2
    
hmarl_models:
  data_source: "data/realtime/book"
  features:
    - microstructure
    - order_flow
    - market_depth
  training:
    min_samples: 10000
    update_frequency: "hourly"
    
ensemble:
  initial_weights:
    base: 0.7
    hmarl: 0.3
  adaptation:
    enabled: true
    window: 1000
```

### Código de Execução
```python
# train_dual_system.py
from src.training.base_trainer import BaseModelTraining
from src.training.hmarl_trainer import HMARLModelTraining
from src.ensemble.dual_ensemble import DualModelEnsemble

# 1. Treinar modelos base (uma vez)
base_trainer = BaseModelTraining('data/csv/historical')
base_models = base_trainer.train_all_base_models()

# 2. Iniciar coleta e treinamento HMARL
hmarl_trainer = HMARLModelTraining('data/realtime/book')
hmarl_trainer.start_incremental_training()

# 3. Criar ensemble
ensemble = DualModelEnsemble(base_models, hmarl_trainer.models)

# 4. Usar em produção
while True:
    market_data = get_latest_market_data()
    book_data = get_latest_book_data()
    
    prediction = ensemble.predict(market_data, book_data)
    execute_strategy(prediction)
```

## 📊 Métricas de Sucesso

### Modelos Base
- Sharpe Ratio > 1.5
- Win Rate > 55%
- Max Drawdown < 15%

### Modelos HMARL
- Redução de slippage > 20%
- Melhoria no timing > 15%
- Detecção de fluxo tóxico > 80%

### Ensemble
- Performance > max(base, hmarl) + 5%
- Consistência inter-regime
- Adaptação a mudanças de mercado

## 🔮 Evolução Futura

1. **Transfer Learning**: Usar modelos base para inicializar HMARL
2. **Meta-Learning**: Aprender quando confiar em cada tipo
3. **Reinforcement Learning**: Otimizar pesos do ensemble online
4. **Feature Sharing**: Compartilhar representações entre modelos