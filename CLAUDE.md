# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚀 Quick Start Commands

### Development Setup
```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Common Commands
```bash
# Run the main trading system
python src/main.py

# Run all tests
pytest

# Run specific test modules
pytest tests/test_etapa1.py -v
pytest src/test_etapa2.py -v

# Run tests with coverage
pytest --cov=src tests/

# Code quality checks
pylint src/
mypy src/
black src/  # Auto-format code
isort src/  # Sort imports
```

### Training System Commands
```bash
# Train complete ML system
python -c "
from src.training.training_orchestrator import TrainingOrchestrator
from src.training.regime_analyzer import RegimeAnalyzer
from datetime import datetime, timedelta

config = {'data_path': 'data/', 'model_save_path': 'models/'}
orchestrator = TrainingOrchestrator(config)

results = orchestrator.train_complete_system(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
    symbols=['WDO'],
    target_metrics={'accuracy': 0.55}
)

# Test regime detection
regime_analyzer = RegimeAnalyzer()
# regime_info = regime_analyzer.analyze_market(candles_df)
"

# Run backtest
python src/ml_backtester.py
```

## 🏗️ High-Level Architecture

### System Overview
ML Trading v2.0 is a production-grade algorithmic trading system that uses machine learning for market analysis and automated trading decisions. It integrates with ProfitDLL for real-time market data and order execution.

### Core Data Flow
```
1. Models → Load ML models and extract required features
2. Data → Fetch real-time and historical market data
3. Indicators → Calculate technical indicators (45+ indicators)
4. Features → Generate ML features (80-100 features)
5. Regime Detection → Identify market regime (trend/range/undefined)
6. Prediction → Execute ML models based on regime
7. Signal → Generate trading signals with risk management
8. Execution → Send orders through ProfitDLL
```

### Key Components

#### Data Management Layer
- **ConnectionManager** (`connection_manager.py`): ProfitDLL interface with callbacks
- **TradingDataStructure** (`data_structure.py`): Centralized thread-safe data storage
- **DataLoader** (`data_loader.py`): Historical data loading with validation
- **DataPipeline** (`data_pipeline.py`): Data processing pipeline

#### ML Layer
- **ModelManager** (`model_manager.py`): ML model loading and management (real models only)
- **FeatureEngine** (`feature_engine.py`): Feature calculation orchestration  
- **RegimeAnalyzer** (`training/regime_analyzer.py`): Market regime detection (ADX + EMAs)
- **MLCoordinator** (`ml_coordinator.py`): Regime-based strategy coordination
- **PredictionEngine** (`prediction_engine.py`): **REAL predictions only** - no mock/synthetic data

#### Trading Layer
- **TradingSystem** (`trading_system.py`): Main system orchestrator
- **SignalGenerator** (`signal_generator.py`): Convert predictions to trading signals
- **RiskManager** (`risk_manager.py`): Position sizing and risk control
- **OrderManager** (`order_manager.py`): Order execution management

#### Training System (`src/training/`)
- **TrainingOrchestrator**: Unified end-to-end training pipeline
- **RobustNaNHandler**: Intelligent NaN handling without bias
- **EnsembleTrainer**: Multi-model ensemble training (XGBoost, LightGBM, RandomForest)
- **ValidationEngine**: Walk-forward temporal validation

### Threading Architecture
The system uses multiple threads for real-time processing:
- Main thread: System coordination
- ML thread: Feature calculation and predictions
- Signal thread: Signal generation and risk management
- Data threads: Real-time data processing from callbacks

## 🛡️ Critical Production Safety Rules

### Data Policy (MUST FOLLOW)
1. **ALWAYS use real market data** from ProfitDLL in production
2. **NEVER use mock/synthetic data** for:
   - Final integration tests
   - Backtests
   - Production environment
3. **Mock data is ONLY allowed** for isolated component testing during development
4. **System automatically blocks** synthetic data in production environment

### ML Predictions Policy (CRITICAL)
1. **NEVER use mock predictions** in any environment
2. **PredictionEngine MUST use real ModelManager** exclusively
3. **System MUST fail with None** when models unavailable
4. **NO fallback to random/synthetic predictions**

### Validation Points
```python
# Every data entry point must validate:
if os.getenv('TRADING_ENV') == 'production':
    if data_source.startswith('mock') or data_source.startswith('test'):
        raise ProductionDataError("Mock data detected in production!")

# PredictionEngine validation:
if not hasattr(self.model_manager, 'models') or not self.model_manager.models:
    self.logger.error("❌ NENHUM MODELO DISPONÍVEL - Predição impossível")
    return None  # NEVER return mock predictions
```

## 📊 Market Regime System

### RegimeAnalyzer Implementation
The system uses `training/regime_analyzer.py` for automatic regime detection:

```python
# RegimeAnalyzer automatically integrated in MLCoordinator
from training.regime_analyzer import RegimeAnalyzer
regime_analyzer = RegimeAnalyzer(logger)
regime_info = regime_analyzer.analyze_market(unified_data)
```

### Regime Detection (MANDATORY)
The RegimeAnalyzer detects market regime before any prediction:

1. **Trend Regime** (ADX > 25, aligned EMAs)
   - `trend_up`: EMA9 > EMA20 > EMA50
   - `trend_down`: EMA9 < EMA20 < EMA50
   - Strategy: Follow trend with 1:2 risk/reward
   - Auto-calculates confidence based on ADX strength

2. **Range Regime** (ADX < 25, independent of EMAs)
   - Strategy: Trade reversals at boundaries
   - Risk/Reward: 1:1.5
   - Fixed confidence: 0.6

3. **Undefined Regime** (ADX > 25 but EMAs not aligned)
   - Action: HOLD (no trading)
   - Conservative thresholds: 0.8 confidence required

### Trading Thresholds by Regime
```python
# Trend Trading
TREND_THRESHOLDS = {
    'confidence': 0.60,
    'probability': 0.60,
    'direction': 0.70,
    'magnitude': 0.003
}

# Range Trading
RANGE_THRESHOLDS = {
    'confidence': 0.60,
    'probability': 0.55,
    'direction': 0.50,
    'magnitude': 0.0015
}
```

## 🔧 Development Guidelines

### Adding New Features
1. Add calculation logic to `ml_features.py` or `technical_indicators.py`
2. Register with RobustNaNHandler for proper NaN treatment
3. Update `all_required_features.json` if needed
4. Add tests for new feature calculation

### Adding New Models
1. Train using TrainingOrchestrator for consistency
2. Ensure model outputs required feature list
3. Save model with metadata including features
4. Update ModelManager to support new model type

### Testing Best Practices
```python
# Component tests (mock allowed)
def test_component():
    mock_data = create_test_data()
    result = component.process(mock_data)
    del mock_data  # Always cleanup

# Integration tests (real data required)
def test_integration():
    real_data = load_real_historical_data()
    if real_data.empty:
        pytest.skip("Real data unavailable")
    result = system.process(real_data)
```

## 📈 Performance Targets

### System Performance
- Feature calculation: < 5 seconds
- ML prediction: < 1 second
- Order execution: < 100ms
- Historical data loading: 1 day (optimized for faster startup)

### Trading Metrics
- Win Rate: > 55%
- Profit Factor: > 1.5
- Sharpe Ratio: > 1.0
- Max Drawdown: < 10%
- Daily trades: 3-10 (market dependent)

## 🚨 Common Issues and Solutions

### Contract Ticker Issues (WDO)
- **Regra**: SEMPRE usa contrato do PRÓXIMO mês
- **Não importa o dia**: Do 1º ao último dia do mês, usa próximo mês
- **Exemplos**: 
  - TODO julho (01/07 a 31/07) → WDOQ25 (agosto)
  - TODO agosto (01/08 a 31/08) → WDOU25 (setembro)
  - TODO dezembro (01/12 a 31/12) → WDOF26 (janeiro/26)

### Features Not Found
- Check if feature is calculated in `ml_features.py` or `technical_indicators.py`
- Verify feature is in model's required features list
- Check for calculation errors in logs

### Models Not Loading
- Verify `.pkl` files exist in models directory
- Check if `_features.json` metadata exists
- Ensure model type is supported by ModelManager

### Data Alignment Issues
- All DataFrames must use datetime index
- Use `pd.concat(..., axis=1)` for synchronization
- Apply `.ffill()` for temporal gaps

### Performance Issues
- Enable caching in feature calculations
- Check if parallel processing is enabled
- Reduce lookback periods if possible
- Use vectorized pandas operations

## 📚 Key Documentation References

1. **Data Flow Map**: `src/features/complete_ml_data_flow_map.md` - Complete system architecture
2. **ML Strategy Doc**: `src/features/ml-prediction-strategy-doc.md` - Trading strategies by regime
3. **Developer Guide**: `DEVELOPER_GUIDE.md` - Detailed technical guide with recent updates
4. **Training System**: `SISTEMA_TREINAMENTO_INTEGRADO.md` - Unified training documentation

## 🔍 Debugging Tips

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Key Log Messages to Monitor
- "Features não encontradas" - Missing features
- "Modelo X: Y features" - Model loading confirmation
- "Regime detected:" - Market regime identification
- "NaN quality score:" - Data quality metrics
- "❌ NENHUM MODELO DISPONÍVEL" - **EXPECTED** when no models loaded
- "🎯 Predição REAL gerada" - Confirms real ML predictions

### System Health Check
```python
# Check all components
from src.trading_system import TradingSystem
system = TradingSystem(config)
print(f"Models loaded: {len(system.model_manager.models)}")
print(f"Features required: {len(system.model_manager.get_all_required_features())}")
print(f"Connection status: {system.connection.connected}")
```

## 🎯 Final Reminders

1. **ALWAYS validate data source** before processing
2. **NEVER use mock predictions** - system must fail appropriately
3. **NEVER skip regime detection** before predictions
4. **FOLLOW threshold requirements** for each regime
5. **TEST with real data** for production readiness
6. **MONITOR system metrics** continuously
7. **UPDATE documentation** after significant changes

### Expected System Failures (NORMAL)
- **PredictionEngine returns None** when no models loaded
- **System blocks synthetic data** in production
- **Features validation fails** with insufficient data
- These are **SAFETY FEATURES**, not bugs

## 🌟 Filosofia Simple Made Easy

Este sistema segue os princípios de Rich Hickey:

### Simple = Desemaranhado
- Cada componente tem UMA responsabilidade
- Dados fluem em UMA direção
- Features são funções puras
- Sem fallbacks mágicos

### Easy ≠ Simple
- Não usamos atalhos que criam dívida técnica
- Preferimos APIs explícitas vs. "convenientes"
- Escolhemos clareza sobre brevidade

### Regras de Ouro
1. **Compose, Don't Complect**: Use composição, não entrelaçamento
2. **Values over State**: Dados imutáveis quando possível
3. **Pure Functions**: Features sem side effects
4. **Explicit over Implicit**: Sem mágica, sem surpresas
5. **One Thing Well**: Cada módulo faz uma coisa bem

### Aplicação Prática
- DataProvider: APENAS fornece dados
- FeatureEngine: APENAS calcula features
- ModelManager: APENAS gerencia modelos
- Nenhum componente "ajuda" outro com fallbacks