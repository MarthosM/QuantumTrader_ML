# 📋 Checklist Pré-Treinamento - QuantumTrader ML

## 🔍 Verificações Essenciais Antes do Treinamento

### 1. 📊 Verificar Disponibilidade de Dados

#### Dados Tick-a-Tick (Históricos)
```bash
# Verificar se há dados históricos suficientes
python scripts/check_historical_data.py --symbol WDOU25 --days 365

# Verificar integridade dos arquivos parquet
python scripts/validate_parquet_files.py --path data/historical/
```

**Requisitos mínimos**:
- ✅ Pelo menos 6 meses de dados para treino robusto
- ✅ Dados sem gaps significativos (< 5% de dias faltando)
- ✅ Arquivos parquet válidos e não corrompidos

#### Dados de Book (Tempo Real)
```bash
# Verificar dados de book coletados
python scripts/check_book_data.py --symbol WDOU25 --days 30

# Se não houver dados suficientes, coletar primeiro:
python scripts/book_collector.py --symbol WDOU25
```

**Requisitos mínimos**:
- ✅ Pelo menos 15 dias de dados de book
- ✅ Cobertura do horário de pregão completo
- ✅ Offer book e Price book disponíveis

### 2. 🔧 Configurar Ambiente

#### Dependências Python
```bash
# Instalar todas as dependências necessárias
pip install -r requirements.txt

# Verificar versões específicas
python scripts/check_dependencies.py
```

**Pacotes críticos**:
- `xgboost >= 1.7.0`
- `lightgbm >= 3.3.0`
- `scikit-learn >= 1.0.0`
- `pandas >= 1.3.0`
- `numpy >= 1.21.0`
- `pyzmq >= 22.0.0`
- `valkey >= 0.1.0` ou `redis >= 4.0.0`

#### Infraestrutura HMARL (Opcional mas Recomendado)
```bash
# Iniciar Valkey/Redis
docker-compose up -d valkey

# Verificar conexão
python scripts/test_valkey_connection.py

# Verificar portas ZMQ disponíveis
netstat -an | grep -E "555[5-9]|5560"
```

### 3. 📁 Estrutura de Diretórios

```bash
# Criar estrutura necessária
python scripts/setup_directories.py
```

Estrutura esperada:
```
QuantumTrader_ML/
├── data/
│   ├── historical/      # Dados tick históricos
│   │   └── WDOU25/     # Por símbolo
│   └── realtime/
│       └── book/       # Dados de book coletados
├── models/
│   ├── tick_only/      # Modelos tick-only
│   └── book_enhanced/  # Modelos com book
├── logs/               # Logs do sistema
└── reports/           # Relatórios de treinamento
```

### 4. 🧪 Testes de Componentes

#### Testar Pipelines de Features
```python
# test_feature_pipelines.py
from src.training.feature_pipeline import FeatureEngineeringPipeline
from src.features.book_features import BookFeatureEngineer

# Testar pipeline de features técnicas
pipeline = FeatureEngineeringPipeline()
test_data = pd.DataFrame({
    'price': np.random.randn(1000).cumsum() + 5000,
    'volume': np.random.randint(1, 100, 1000)
})
features = pipeline.create_training_features(test_data)
assert len(features.columns) > 20, "Features insuficientes"

# Testar book features
book_eng = BookFeatureEngineer()
book_data = pd.DataFrame({
    'best_bid': np.random.randn(1000) + 5000,
    'best_ask': np.random.randn(1000) + 5001,
    'bid_volume_1': np.random.randint(10, 100, 1000),
    'ask_volume_1': np.random.randint(10, 100, 1000)
})
book_features = book_eng.calculate_all_features(book_data)
assert 'spread' in book_features.columns, "Spread não calculado"
```

#### Testar Regime Analyzer
```python
from src.training.regime_analyzer import RegimeAnalyzer

analyzer = RegimeAnalyzer()
# Criar dados OHLCV de teste
candles = pd.DataFrame({
    'open': np.random.randn(100).cumsum() + 5000,
    'high': np.random.randn(100).cumsum() + 5010,
    'low': np.random.randn(100).cumsum() + 4990,
    'close': np.random.randn(100).cumsum() + 5000,
    'volume': np.random.randint(100, 1000, 100)
})
regime_info = analyzer.analyze_market(candles)
assert regime_info['regime'] in ['trend_up', 'trend_down', 'range', 'undefined']
```

### 5. 🔍 Validar Qualidade dos Dados

```python
# scripts/validate_data_quality.py
import pandas as pd
import numpy as np
from pathlib import Path

def validate_tick_data(symbol: str, days_back: int = 30):
    """Valida qualidade dos dados tick"""
    issues = []
    
    # Verificar arquivos
    data_path = Path(f'data/historical/{symbol}')
    if not data_path.exists():
        issues.append(f"Path não existe: {data_path}")
        return issues
        
    # Carregar amostra
    files = list(data_path.glob('**/trades.parquet'))
    if len(files) < days_back * 0.7:  # 70% dos dias esperados
        issues.append(f"Poucos arquivos: {len(files)} encontrados, {days_back} esperados")
        
    # Verificar conteúdo
    for file in files[:5]:  # Amostra
        try:
            df = pd.read_parquet(file)
            
            # Verificar colunas essenciais
            required_cols = ['price', 'volume', 'timestamp']
            missing = set(required_cols) - set(df.columns)
            if missing:
                issues.append(f"Colunas faltando em {file}: {missing}")
                
            # Verificar valores
            if df['price'].isna().sum() > 0:
                issues.append(f"NaN em price: {file}")
            if (df['price'] <= 0).any():
                issues.append(f"Preços inválidos: {file}")
                
        except Exception as e:
            issues.append(f"Erro ao ler {file}: {e}")
            
    return issues

# Executar validação
issues = validate_tick_data('WDOU25', 365)
if issues:
    print("⚠️ Problemas encontrados:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✅ Dados validados com sucesso!")
```

### 6. 🎯 Definir Objetivos e Métricas

```yaml
# config/training_objectives.yaml
training_objectives:
  tick_only_models:
    target_metrics:
      accuracy: 0.55      # Mínimo aceitável
      f1_score: 0.50
      sharpe_ratio: 1.0
    validation:
      method: walk_forward
      test_size: 0.2
      n_splits: 5
      
  book_enhanced_models:
    spread_prediction:
      mae: 0.0002        # 2 bps de erro máximo
      r2: 0.6
    price_direction:
      accuracy: 0.52     # Melhor que random
      precision: 0.5
      
  risk_limits:
    max_drawdown: 0.10   # 10% máximo
    max_position: 10     # Contratos
    daily_loss_limit: 5000.00  # R$
```

### 7. 💾 Backup de Dados Importantes

```bash
# Fazer backup antes de treinar
python scripts/backup_data.py --output backups/pre_training_$(date +%Y%m%d)

# Itens para backup:
# - Modelos existentes (se houver)
# - Configurações atuais
# - Dados históricos (opcional, são grandes)
```

### 8. 🖥️ Recursos Computacionais

**Verificar disponibilidade**:
```python
import psutil
import torch  # se usar deep learning

# CPU
print(f"CPU cores: {psutil.cpu_count()}")
print(f"CPU usage: {psutil.cpu_percent()}%")

# Memória
mem = psutil.virtual_memory()
print(f"RAM total: {mem.total / 1e9:.1f} GB")
print(f"RAM disponível: {mem.available / 1e9:.1f} GB")

# GPU (se disponível)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**Requisitos mínimos**:
- CPU: 4+ cores
- RAM: 8+ GB (16 GB recomendado)
- Espaço em disco: 50+ GB livres
- GPU: Opcional, mas acelera XGBoost/LightGBM

### 9. 📝 Documentar Configurações

```python
# scripts/save_training_config.py
import json
import yaml
from datetime import datetime

config = {
    'timestamp': datetime.now().isoformat(),
    'environment': {
        'python_version': sys.version,
        'platform': platform.platform(),
        'packages': {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    },
    'data_status': {
        'tick_data_days': 365,
        'book_data_days': 30,
        'symbols': ['WDOU25']
    },
    'training_params': {
        'tick_only': {
            'lookback_days': 365,
            'regimes': ['trend_up', 'trend_down', 'range'],
            'validation_method': 'walk_forward'
        },
        'book_enhanced': {
            'lookback_days': 30,
            'targets': ['spread_next', 'imbalance_direction', 'price_move_5s']
        }
    }
}

# Salvar configuração
with open('config/training_config_snapshot.json', 'w') as f:
    json.dump(config, f, indent=2)
```

### 10. 🚦 Checklist Final

Antes de executar o treinamento, confirme:

- [ ] **Dados disponíveis e validados**
- [ ] **Ambiente Python configurado corretamente**
- [ ] **Estrutura de diretórios criada**
- [ ] **Testes de componentes passando**
- [ ] **Recursos computacionais adequados**
- [ ] **Objetivos e métricas definidos**
- [ ] **Backup realizado**
- [ ] **HMARL/Valkey rodando (se usar)**

### 🚀 Comando para Iniciar Treinamento

Após todas as verificações:

```bash
# Treinamento completo com relatório
python examples/train_dual_models.py --symbol WDOU25 --validate

# Ou apenas tick-only
python -c "
from src.training.tick_training_pipeline import TickTrainingPipeline
config = {'tick_data_path': 'data/historical', 'models_path': 'models/tick_only'}
pipeline = TickTrainingPipeline(config)
result = pipeline.train_complete_pipeline('WDOU25', lookback_days=365)
"

# Ou apenas book-enhanced
python -c "
from src.training.book_training_pipeline import BookTrainingPipeline
config = {'book_data_path': 'data/realtime/book', 'models_path': 'models/book_enhanced'}
pipeline = BookTrainingPipeline(config)
result = pipeline.train_complete_pipeline('WDOU25', lookback_days=30)
"
```

## ⚠️ Problemas Comuns

1. **"No data available"**: Executar coleta de dados primeiro
2. **"Import error"**: Verificar instalação de pacotes
3. **"Memory error"**: Reduzir tamanho do batch ou usar menos features
4. **"Valkey connection refused"**: Iniciar Docker/Valkey primeiro

---
**Última atualização**: Agosto 2025