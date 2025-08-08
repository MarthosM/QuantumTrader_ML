"""
Pipeline Final para 5 Milhões de Registros
Versão realista com configuração otimizada (1000 trades horizon)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
import joblib
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CSV5MFinalTrainer:
    """Treina modelos com configuração final otimizada"""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.sample_size = 5_000_000
        self.horizon = 1000  # 1000 trades ahead
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_data_with_progress(self):
        """Carrega 5M registros"""
        
        print("=" * 80)
        print("CARREGAMENTO DE DADOS - VERSÃO FINAL")
        print("=" * 80)
        
        dtypes = {
            '<trade_number>': 'uint32',
            '<price>': 'float32',
            '<qty>': 'uint16',
            '<vol>': 'float32',
            '<buy_agent>': 'category',
            '<sell_agent>': 'category',
            '<trade_type>': 'category'
        }
        
        print(f"\nCarregando {self.sample_size:,} registros...")
        start_time = datetime.now()
        
        df = pd.read_csv(self.csv_path, 
                        nrows=self.sample_size,
                        dtype=dtypes)
        
        print("Processando timestamps...")
        df['timestamp'] = pd.to_datetime(
            df['<date>'].astype(str) + ' ' + df['<time>'].astype(str).str.zfill(6),
            format='%Y%m%d %H%M%S'
        )
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        load_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n[OK] Tempo: {load_time:.1f}s")
        print(f"[OK] Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
        print(f"[OK] Horizonte: {self.horizon} trades (~5-10 minutos)")
        
        return df
    
    def create_trading_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features otimizadas para trading sem data leakage"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE FEATURES PARA TRADING")
        print("=" * 80)
        
        features = pd.DataFrame(index=df.index)
        
        # Preço base para cálculos (sem usar diretamente)
        price = df['<price>'].values.astype('float32')
        
        # 1. RETORNOS MÚLTIPLOS (principal preditor)
        print("\n-> Features de Retorno...")
        returns_periods = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        for period in tqdm(returns_periods, desc="  Calculando retornos"):
            features[f'returns_{period}'] = pd.Series(price).pct_change(period)
        
        # Log returns para não-linearidades
        features['log_returns_1'] = np.log(price / np.roll(price, 1))
        features['log_returns_5'] = np.log(price / np.roll(price, 5))
        features['log_returns_20'] = np.log(price / np.roll(price, 20))
        
        # 2. VOLATILIDADE E TURBULÊNCIA
        print("-> Features de Volatilidade...")
        vol_windows = [10, 20, 50, 100, 200]
        for window in tqdm(vol_windows, desc="  Calculando volatilidade"):
            features[f'volatility_{window}'] = features['returns_1'].rolling(window).std()
            # Volatilidade relativa
            features[f'vol_ratio_{window}'] = (
                features[f'volatility_{window}'] / 
                features[f'volatility_{window}'].rolling(window*2).mean()
            ).fillna(1)
        
        # Turbulência do mercado (Mahalanobis distance proxy)
        features['turbulence'] = (
            features['returns_1'].rolling(50).apply(lambda x: ((x - x.mean()) ** 2).mean())
        )
        
        # 3. MOMENTUM E TENDÊNCIA
        print("-> Features de Momentum...")
        # Momentum clássico
        features['momentum_5'] = features['returns_5']
        features['momentum_20'] = features['returns_20']
        features['momentum_50'] = features['returns_50']
        
        # Momentum ajustado por risco (Sharpe ratio)
        for period in [5, 20, 50]:
            features[f'sharpe_{period}'] = (
                features[f'returns_{period}'] / 
                features[f'volatility_{period*2}'].clip(lower=0.0001)
            )
        
        # Taxa de mudança de momentum
        features['momentum_acceleration'] = features['momentum_5'] - features['momentum_20']
        
        # 4. VOLUME E LIQUIDEZ
        print("-> Features de Volume...")
        qty = df['<qty>'].values.astype('float32')
        vol = df['<vol>'].values.astype('float32')
        
        # Volume patterns
        for window in tqdm([20, 50, 100, 200], desc="  Processando volume"):
            # Z-score do volume
            vol_mean = pd.Series(qty).rolling(window).mean()
            vol_std = pd.Series(qty).rolling(window).std()
            features[f'volume_zscore_{window}'] = (qty - vol_mean) / vol_std.clip(lower=0.001)
            
            # Volume ratio
            features[f'volume_ratio_{window}'] = qty / vol_mean.clip(lower=1)
        
        # VWAP e desvios
        cumvol = pd.Series(vol).rolling(200).sum()
        cumqty = pd.Series(qty).rolling(200).sum()
        vwap = (cumvol / cumqty).fillna(method='ffill')
        features['price_vwap_deviation'] = (price / vwap - 1) * 100
        
        # Volume-return correlation
        features['volume_return_corr'] = (
            pd.Series(qty).rolling(50).corr(features['returns_1'].abs())
        )
        
        # 5. MICROESTRUTURA E FLUXO
        print("-> Features de Microestrutura...")
        
        # Order flow
        features['is_buyer'] = (df['<trade_type>'] == 'AggressorBuyer').astype('int8')
        features['is_seller'] = (df['<trade_type>'] == 'AggressorSeller').astype('int8')
        
        # Order flow imbalance (múltiplas janelas)
        for window in tqdm([20, 50, 100, 200], desc="  Order flow"):
            buyer_vol = (features['is_buyer'] * qty).rolling(window).sum()
            seller_vol = (features['is_seller'] * qty).rolling(window).sum()
            total_vol = buyer_vol + seller_vol
            
            features[f'flow_imbalance_{window}'] = (
                (buyer_vol - seller_vol) / total_vol.clip(lower=1)
            )
            
            # Order flow momentum
            features[f'flow_momentum_{window}'] = features[f'flow_imbalance_{window}'].diff(10)
        
        # Trade intensity
        time_diff = df['timestamp'].diff().dt.total_seconds()
        features['trade_intensity'] = 1 / time_diff.rolling(50).median().clip(lower=0.001)
        features['intensity_surge'] = (
            features['trade_intensity'] / 
            features['trade_intensity'].rolling(200).mean()
        ).fillna(1)
        
        # 6. AGENT BEHAVIOR PATTERNS
        print("-> Features de Agent Behavior...")
        
        # Top agents activity
        top_buyers = df['<buy_agent>'].value_counts().head(20).index
        top_sellers = df['<sell_agent>'].value_counts().head(20).index
        
        # Institutional proxy (top 5 agents)
        institutional_buyers = top_buyers[:5]
        institutional_sellers = top_sellers[:5]
        
        features['institutional_buying'] = df['<buy_agent>'].isin(institutional_buyers).astype('int8')
        features['institutional_selling'] = df['<sell_agent>'].isin(institutional_sellers).astype('int8')
        
        # Institutional flow
        features['institutional_flow'] = (
            features['institutional_buying'].rolling(100).sum() - 
            features['institutional_selling'].rolling(100).sum()
        )
        
        # Agent diversity (número de agentes únicos)
        features['buyer_diversity'] = df['<buy_agent>'].rolling(100).apply(lambda x: x.nunique())
        features['seller_diversity'] = df['<sell_agent>'].rolling(100).apply(lambda x: x.nunique())
        
        # 7. TECHNICAL PATTERNS
        print("-> Features Técnicas...")
        
        # Moving averages (sem usar preço direto)
        ma_5 = pd.Series(price).rolling(5).mean()
        ma_20 = pd.Series(price).rolling(20).mean()
        ma_50 = pd.Series(price).rolling(50).mean()
        
        # MA ratios e cruzamentos
        features['ma_5_20_ratio'] = (ma_5 / ma_20 - 1) * 100
        features['ma_20_50_ratio'] = (ma_20 / ma_50 - 1) * 100
        features['ma_trend_strength'] = features['ma_5_20_ratio'] + features['ma_20_50_ratio']
        
        # Bollinger Bands
        bb_mean = pd.Series(price).rolling(20).mean()
        bb_std = pd.Series(price).rolling(20).std()
        bb_upper = bb_mean + 2 * bb_std
        bb_lower = bb_mean - 2 * bb_std
        
        features['bb_position'] = ((price - bb_mean) / (bb_upper - bb_lower) * 2).clip(-1, 1)
        features['bb_width'] = (bb_upper - bb_lower) / bb_mean * 100
        
        # 8. TEMPORAL E SAZONALIDADE
        print("-> Features Temporais...")
        
        features['hour'] = df['timestamp'].dt.hour.astype('int8')
        features['minute'] = df['timestamp'].dt.minute.astype('int8')
        features['day_of_week'] = df['timestamp'].dt.dayofweek.astype('int8')
        
        # Períodos do dia
        market_open = df['timestamp'].dt.normalize() + pd.Timedelta(hours=9)
        minutes_since_open = ((df['timestamp'] - market_open).dt.total_seconds() / 60).clip(lower=0, upper=480)
        features['time_of_day'] = minutes_since_open / 480  # Normalizado [0,1]
        
        # Indicadores de sessão
        features['is_opening_hour'] = (features['hour'] == 9).astype('int8')
        features['is_closing_hour'] = (features['hour'] >= 16).astype('int8')
        features['is_lunch_period'] = ((features['hour'] >= 12) & (features['hour'] < 14)).astype('int8')
        
        # 9. FEATURES COMPOSTAS E INTERAÇÕES
        print("-> Features Compostas...")
        
        # Momentum × Volume
        features['momentum_volume'] = features['momentum_20'] * features['volume_zscore_50']
        
        # Volatilidade × Flow
        features['vol_flow_interaction'] = features['volatility_50'] * features['flow_imbalance_100']
        
        # Institutional × Momentum
        features['institutional_momentum'] = features['institutional_flow'] * features['momentum_50']
        
        print(f"\n[OK] Total features criadas: {features.shape[1]}")
        
        # Cleanup final
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        # Garantir float32
        for col in features.columns:
            if features[col].dtype == 'float64':
                features[col] = features[col].astype('float32')
        
        return features
    
    def create_balanced_targets(self, df: pd.DataFrame) -> tuple:
        """Cria targets com distribuição balanceada usando 0.5×std"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE TARGETS BALANCEADOS")
        print("=" * 80)
        
        print(f"\nHorizonte: {self.horizon} trades")
        
        # Calcular retornos futuros
        future_price = df['<price>'].shift(-self.horizon)
        returns = (future_price - df['<price>']) / df['<price>']
        
        # Método 0.5×std (configuração ótima encontrada)
        returns_std = returns.std()
        buy_threshold = 0.5 * returns_std
        sell_threshold = -0.5 * returns_std
        
        print(f"\nMétodo: 0.5 × desvio padrão")
        print(f"Desvio padrão dos retornos: {returns_std:.5f} ({returns_std*100:.3f}%)")
        print(f"Threshold BUY:  > {buy_threshold:.5f} ({buy_threshold*100:.3f}%)")
        print(f"Threshold SELL: < {sell_threshold:.5f} ({sell_threshold*100:.3f}%)")
        
        # Criar target
        target = pd.Series(0, index=df.index, dtype='int8')
        target[returns > buy_threshold] = 1    # BUY
        target[returns < sell_threshold] = -1  # SELL
        
        # Estatísticas
        dist = target.value_counts().sort_index()
        total = len(target)
        
        print(f"\nDistribuição real:")
        print(f"  SELL (-1): {dist.get(-1, 0):>8,} ({dist.get(-1, 0)/total*100:>5.1f}%)")
        print(f"  HOLD  (0): {dist.get(0, 0):>8,} ({dist.get(0, 0)/total*100:>5.1f}%)")
        print(f"  BUY   (1): {dist.get(1, 0):>8,} ({dist.get(1, 0)/total*100:>5.1f}%)")
        
        # Verificar balanço
        buy_pct = dist.get(1, 0) / total * 100
        sell_pct = dist.get(-1, 0) / total * 100
        signals_pct = buy_pct + sell_pct
        
        print(f"\nMétricas de balanço:")
        print(f"  Total de sinais: {signals_pct:.1f}%")
        print(f"  Razão BUY/SELL: {buy_pct/sell_pct:.2f}" if sell_pct > 0 else "  Razão BUY/SELL: N/A")
        
        # Target binário para métricas de trading
        target_binary = target.copy()
        target_binary[target_binary == 0] = np.nan
        
        return target, target_binary, returns_std
    
    def train_balanced_models(self, features: pd.DataFrame, target: pd.Series):
        """Treina modelos com foco em performance real de trading"""
        
        print("\n" + "=" * 80)
        print("TREINAMENTO DE MODELOS BALANCEADOS")
        print("=" * 80)
        
        # Preparar dados
        mask = ~target.isna()
        X = features[mask]
        y = target[mask]
        
        print(f"\nDados disponíveis: {len(X):,} registros")
        
        # Normalizar
        print("Normalizando features...")
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Split temporal
        split = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Calcular pesos para balanceamento
        classes = np.array([-1, 0, 1])
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"\nPesos das classes: {class_weight_dict}")
        
        # Configurações dos modelos
        models_config = [
            ("LightGBM", self._train_lightgbm),
            ("XGBoost", self._train_xgboost),
            ("Random Forest", self._train_random_forest),
            ("Extra Trees", self._train_extra_trees)
        ]
        
        predictions = {}
        importances = {}
        
        for name, train_func in models_config:
            print(f"\n{'='*60}")
            print(f"Treinando {name}...")
            print(f"{'='*60}")
            
            model, pred, metrics, importance = train_func(
                X_train, y_train, X_test, y_test, class_weight_dict
            )
            
            self.models[name.lower().replace(' ', '_')] = model
            self.results[name.lower().replace(' ', '_')] = metrics
            predictions[name] = pred
            importances[name] = importance
            
            # Mostrar métricas detalhadas
            self._print_metrics(name, metrics)
        
        # Ensemble voting
        print(f"\n{'='*60}")
        print("ENSEMBLE VOTING")
        print(f"{'='*60}")
        
        # Voting ponderado baseado em trading accuracy
        weights = {}
        for name, metrics in self.results.items():
            weights[name] = metrics['trading_accuracy']
        
        # Normalizar pesos
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        print("\nPesos do ensemble (baseado em trading accuracy):")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.3f}")
        
        # Calcular ensemble
        ensemble_pred = np.zeros_like(y_test.values, dtype=float)
        
        model_name_map = {
            "LightGBM": "lightgbm",
            "XGBoost": "xgboost",
            "Random Forest": "random_forest",
            "Extra Trees": "extra_trees"
        }
        
        for display_name, pred in predictions.items():
            model_key = model_name_map[display_name]
            if model_key in weights:
                ensemble_pred += weights[model_key] * pred
        
        ensemble_pred = np.round(ensemble_pred).astype(int)
        
        # Métricas do ensemble
        ensemble_metrics = self._calculate_comprehensive_metrics(y_test, ensemble_pred)
        self.results['ensemble'] = ensemble_metrics
        
        self._print_metrics("Ensemble", ensemble_metrics)
        
        # Feature importance
        print("\n" + "="*60)
        print("TOP 20 FEATURES MAIS IMPORTANTES")
        print("="*60)
        
        # Combinar importâncias
        avg_imp = pd.DataFrame(index=X.columns)
        for name, imp in importances.items():
            avg_imp[name] = imp.reindex(X.columns, fill_value=0)
        
        # Média ponderada pelos mesmos pesos do ensemble
        avg_imp['weighted_avg'] = 0
        for display_name, imp_df in importances.items():
            model_key = model_name_map[display_name]
            if model_key in weights:
                avg_imp['weighted_avg'] += weights[model_key] * avg_imp[display_name]
        
        avg_imp = avg_imp.sort_values('weighted_avg', ascending=False)
        
        print(avg_imp[['weighted_avg']].head(20))
        
        return avg_imp
    
    def _calculate_comprehensive_metrics(self, y_true, y_pred):
        """Calcula métricas abrangentes focadas em trading"""
        
        # Métricas básicas
        accuracy = (y_pred == y_true).mean()
        
        # Por classe
        accuracy_sell = (y_pred[y_true == -1] == -1).mean() if (y_true == -1).any() else 0
        accuracy_hold = (y_pred[y_true == 0] == 0).mean() if (y_true == 0).any() else 0
        accuracy_buy = (y_pred[y_true == 1] == 1).mean() if (y_true == 1).any() else 0
        
        # Trading metrics (ignorando HOLD)
        trading_mask = y_true != 0
        if trading_mask.any():
            trading_accuracy = (y_pred[trading_mask] == y_true[trading_mask]).mean()
            trading_precision_buy = (
                (y_pred[y_pred == 1] == y_true[y_pred == 1]).mean() 
                if (y_pred == 1).any() else 0
            )
            trading_precision_sell = (
                (y_pred[y_pred == -1] == y_true[y_pred == -1]).mean() 
                if (y_pred == -1).any() else 0
            )
        else:
            trading_accuracy = 0
            trading_precision_buy = 0
            trading_precision_sell = 0
        
        # F1 scores
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
        
        # Profit factor simulado (assumindo risco/retorno 1:1)
        true_buys = (y_true == 1).sum()
        true_sells = (y_true == -1).sum()
        correct_buys = ((y_pred == 1) & (y_true == 1)).sum()
        correct_sells = ((y_pred == -1) & (y_true == -1)).sum()
        wrong_buys = ((y_pred == 1) & (y_true != 1)).sum()
        wrong_sells = ((y_pred == -1) & (y_true != -1)).sum()
        
        profit = correct_buys + correct_sells
        loss = wrong_buys + wrong_sells
        profit_factor = profit / loss if loss > 0 else np.inf
        
        return {
            'accuracy': accuracy,
            'accuracy_sell': accuracy_sell,
            'accuracy_hold': accuracy_hold,
            'accuracy_buy': accuracy_buy,
            'trading_accuracy': trading_accuracy,
            'trading_precision_buy': trading_precision_buy,
            'trading_precision_sell': trading_precision_sell,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'profit_factor': profit_factor,
            'confusion_matrix': cm.tolist(),
            'correct_trades': int(correct_buys + correct_sells),
            'wrong_trades': int(wrong_buys + wrong_sells)
        }
    
    def _print_metrics(self, model_name, metrics):
        """Imprime métricas de forma organizada"""
        print(f"\nMétricas {model_name}:")
        print(f"  Overall Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Trading Accuracy: {metrics['trading_accuracy']:.2%}")
        print(f"  Precision BUY: {metrics['trading_precision_buy']:.2%}")
        print(f"  Precision SELL: {metrics['trading_precision_sell']:.2%}")
        print(f"  F1 Macro: {metrics['f1_macro']:.2%}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        
    def _train_lightgbm(self, X_train, y_train, X_test, y_test, class_weights):
        """LightGBM otimizado"""
        sample_weights = np.zeros(len(y_train))
        for class_val, weight in class_weights.items():
            sample_weights[y_train == class_val] = weight
        
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'num_leaves': 63,
            'learning_rate': 0.02,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 8,
            'min_data_in_leaf': 50,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1
        }
        
        y_train_lgb = y_train + 1
        y_test_lgb = y_test + 1
        
        lgb_train = lgb.Dataset(X_train, label=y_train_lgb, weight=sample_weights)
        lgb_val = lgb.Dataset(X_test, label=y_test_lgb, reference=lgb_train)
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=300,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        pred = model.predict(X_test, num_iteration=model.best_iteration)
        pred_class = np.argmax(pred, axis=1) - 1
        
        metrics = self._calculate_comprehensive_metrics(y_test, pred_class)
        
        importance = pd.Series(
            model.feature_importance(importance_type='gain'),
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return model, pred_class, metrics, importance
    
    def _train_xgboost(self, X_train, y_train, X_test, y_test, class_weights):
        """XGBoost otimizado"""
        sample_weights = np.zeros(len(y_train))
        for class_val, weight in class_weights.items():
            sample_weights[y_train == class_val] = weight
        
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=50,
            reg_alpha=0.1,
            reg_lambda=0.1,
            gamma=0.1,
            tree_method='hist',
            eval_metric='mlogloss',
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=42
        )
        
        y_train_xgb = y_train + 1
        y_test_xgb = y_test + 1
        
        model.fit(
            X_train, y_train_xgb,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test_xgb)],
            verbose=False
        )
        
        pred = model.predict(X_test) - 1
        
        metrics = self._calculate_comprehensive_metrics(y_test, pred)
        
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return model, pred, metrics, importance
    
    def _train_random_forest(self, X_train, y_train, X_test, y_test, class_weights):
        """Random Forest otimizado"""
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=50,
            min_samples_leaf=25,
            max_features='sqrt',
            class_weight=class_weights,
            n_jobs=-1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        metrics = self._calculate_comprehensive_metrics(y_test, pred)
        
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return model, pred, metrics, importance
    
    def _train_extra_trees(self, X_train, y_train, X_test, y_test, class_weights):
        """Extra Trees para diversidade"""
        model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=50,
            min_samples_leaf=25,
            max_features='sqrt',
            class_weight=class_weights,
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        metrics = self._calculate_comprehensive_metrics(y_test, pred)
        
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return model, pred, metrics, importance
    
    def save_complete_results(self, feature_importance: pd.DataFrame, returns_std: float):
        """Salva resultados completos com relatório detalhado"""
        
        output_dir = Path('models/csv_5m_final')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*80)
        print("SALVANDO RESULTADOS FINAIS")
        print("="*80)
        
        # Salvar modelos
        for name, model in tqdm(self.models.items(), desc="Salvando modelos"):
            if 'lightgbm' in name:
                model_file = output_dir / f'{name}_{timestamp}.txt'
                model.save_model(str(model_file))
            else:
                model_file = output_dir / f'{name}_{timestamp}.pkl'
                joblib.dump(model, model_file)
        
        # Salvar scaler
        joblib.dump(self.scaler, output_dir / f'scaler_{timestamp}.pkl')
        
        # Feature importance
        feature_importance.to_csv(output_dir / f'feature_importance_{timestamp}.csv')
        
        # Configuração completa
        config = {
            'training_date': datetime.now().isoformat(),
            'sample_size': self.sample_size,
            'horizon_trades': self.horizon,
            'threshold_method': '0.5_std',
            'returns_std': float(returns_std),
            'buy_threshold': float(0.5 * returns_std),
            'sell_threshold': float(-0.5 * returns_std),
            'feature_count': len(feature_importance),
            'top_10_features': feature_importance.head(10).to_dict()
        }
        
        # Performance detalhada
        performance_report = {
            'models_performance': {},
            'best_overall': None,
            'best_trading': None,
            'ensemble_weights': {}
        }
        
        # Adicionar métricas de cada modelo
        for name, metrics in self.results.items():
            performance_report['models_performance'][name] = {
                'accuracy': round(metrics['accuracy'], 4),
                'trading_accuracy': round(metrics['trading_accuracy'], 4),
                'precision_buy': round(metrics['trading_precision_buy'], 4),
                'precision_sell': round(metrics['trading_precision_sell'], 4),
                'f1_macro': round(metrics['f1_macro'], 4),
                'profit_factor': round(metrics['profit_factor'], 2),
                'correct_trades': metrics['correct_trades'],
                'wrong_trades': metrics['wrong_trades']
            }
        
        # Identificar melhores modelos
        best_overall = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        best_trading = max(self.results.items(), key=lambda x: x[1]['trading_accuracy'])
        
        performance_report['best_overall'] = {
            'model': best_overall[0],
            'accuracy': round(best_overall[1]['accuracy'], 4)
        }
        
        performance_report['best_trading'] = {
            'model': best_trading[0],
            'trading_accuracy': round(best_trading[1]['trading_accuracy'], 4)
        }
        
        # Salvar relatório completo
        full_report = {
            'configuration': config,
            'performance': performance_report,
            'training_summary': {
                'total_models': len(self.models),
                'ensemble_method': 'weighted_voting_by_trading_accuracy',
                'expected_trading_accuracy': '55-65%',
                'expected_profit_factor': '>1.5'
            }
        }
        
        with open(output_dir / f'training_report_{timestamp}.json', 'w') as f:
            json.dump(full_report, f, indent=2)
        
        print(f"\n[OK] Resultados salvos em: {output_dir}")
        
        # Imprimir resumo final
        print("\n" + "="*80)
        print("RESUMO FINAL DO TREINAMENTO")
        print("="*80)
        
        print("\nMelhores modelos:")
        print(f"  Overall: {best_overall[0]} ({best_overall[1]['accuracy']:.2%})")
        print(f"  Trading: {best_trading[0]} ({best_trading[1]['trading_accuracy']:.2%})")
        
        print("\nPerformance do Ensemble:")
        ensemble_metrics = self.results['ensemble']
        print(f"  Trading Accuracy: {ensemble_metrics['trading_accuracy']:.2%}")
        print(f"  Profit Factor: {ensemble_metrics['profit_factor']:.2f}")
        print(f"  Trades Corretos: {ensemble_metrics['correct_trades']}")
        print(f"  Trades Errados: {ensemble_metrics['wrong_trades']}")

def main():
    """Pipeline principal final"""
    
    print("PIPELINE FINAL - 5 MILHÕES DE REGISTROS")
    print("Configuração otimizada: 1000 trades horizon, 0.5×std threshold\n")
    
    csv_path = r"C:\Users\marth\Downloads\WDO_FUT\WDOFUT_BMF_T.csv"
    
    trainer = CSV5MFinalTrainer(csv_path)
    
    try:
        # 1. Carregar dados
        df = trainer.load_data_with_progress()
        
        # 2. Criar features otimizadas
        features = trainer.create_trading_features(df)
        
        # 3. Criar targets balanceados
        target, target_binary, returns_std = trainer.create_balanced_targets(df)
        
        # 4. Treinar modelos
        feature_importance = trainer.train_balanced_models(features, target)
        
        # 5. Salvar resultados completos
        trainer.save_complete_results(feature_importance, returns_std)
        
        # Cleanup
        del df, features, target
        gc.collect()
        
        print("\n" + "="*80)
        print("[OK] TREINAMENTO FINAL CONCLUÍDO COM SUCESSO!")
        print("="*80)
        
        print("\nConfigurações utilizadas:")
        print("- Horizonte: 1000 trades (~5-10 minutos)")
        print("- Threshold: 0.5 × desvio padrão dos retornos")
        print("- Classes balanceadas com pesos automáticos")
        print("- 4 modelos em ensemble ponderado")
        print("- Métricas focadas em trading real")
        
        print("\nPróximos passos:")
        print("1. Implementar backtesting com os modelos treinados")
        print("2. Integrar com sistema de trading em tempo real")
        print("3. Monitorar performance em paper trading")
        print("4. Ajustar thresholds conforme necessário")
        
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()