"""
Pipeline Otimizado para Memória - 5 Milhões de Registros
Versão com processamento em chunks e otimizações de memória
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

class CSV5MMemoryOptimizedTrainer:
    """Treina modelos com otimização de memória"""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.sample_size = 5_000_000
        self.chunk_size = 500_000  # Processar em chunks de 500k
        self.horizon = 1000  # 1000 trades ahead
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_data_optimized(self):
        """Carrega dados com tipos otimizados"""
        
        print("=" * 80)
        print("CARREGAMENTO OTIMIZADO DE DADOS")
        print("=" * 80)
        
        # Tipos otimizados para economia de memória
        dtypes = {
            '<date>': 'uint32',
            '<time>': 'uint32',
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
        
        # Carregar dados
        df = pd.read_csv(self.csv_path, 
                        nrows=self.sample_size,
                        dtype=dtypes)
        
        # Processar timestamp de forma otimizada
        print("Processando timestamps...")
        df['timestamp'] = pd.to_datetime(
            df['<date>'].astype(str) + ' ' + df['<time>'].astype(str).str.zfill(6),
            format='%Y%m%d %H%M%S'
        )
        
        # Liberar colunas desnecessárias
        df = df.drop(['<date>', '<time>'], axis=1)
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        load_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n[OK] Tempo: {load_time:.1f}s")
        print(f"[OK] Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
        print(f"[OK] Memória usada: {df.memory_usage().sum() / 1024**2:.1f} MB")
        
        return df
    
    def create_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features com economia de memória"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO OTIMIZADA DE FEATURES")
        print("=" * 80)
        
        n = len(df)
        features_dict = {}
        
        # Extrair arrays uma vez
        price = df['<price>'].values.astype('float32')
        qty = df['<qty>'].values.astype('float32')
        vol = df['<vol>'].values.astype('float32')
        
        # 1. RETORNOS (features essenciais)
        print("\n-> Calculando retornos...")
        for period in [1, 5, 10, 20, 50, 100]:
            returns = np.zeros(n, dtype='float32')
            returns[period:] = (price[period:] - price[:-period]) / price[:-period]
            features_dict[f'returns_{period}'] = returns
            
        # 2. VOLATILIDADE (janelas menores)
        print("-> Calculando volatilidade...")
        returns_1 = features_dict['returns_1']
        
        for window in [10, 20, 50]:
            vol_arr = pd.Series(returns_1).rolling(window).std().values.astype('float32')
            features_dict[f'volatility_{window}'] = vol_arr
            
        # 3. VOLUME METRICS
        print("-> Calculando métricas de volume...")
        
        # Volume z-score
        for window in [20, 50]:
            vol_mean = pd.Series(qty).rolling(window).mean()
            vol_std = pd.Series(qty).rolling(window).std()
            z_score = ((qty - vol_mean) / vol_std.clip(lower=0.001)).values.astype('float32')
            features_dict[f'volume_zscore_{window}'] = z_score
            
        # 4. ORDER FLOW
        print("-> Calculando order flow...")
        is_buyer = (df['<trade_type>'] == 'AggressorBuyer').astype('float32').values
        is_seller = (df['<trade_type>'] == 'AggressorSeller').astype('float32').values
        
        for window in [20, 50]:
            buyer_flow = pd.Series(is_buyer).rolling(window).sum()
            seller_flow = pd.Series(is_seller).rolling(window).sum()
            total_flow = buyer_flow + seller_flow
            imbalance = ((buyer_flow - seller_flow) / total_flow.clip(lower=1)).values.astype('float32')
            features_dict[f'flow_imbalance_{window}'] = imbalance
            
        # 5. INDICADORES TÉCNICOS SIMPLES
        print("-> Calculando indicadores técnicos...")
        
        # Moving average ratios
        ma_5 = pd.Series(price).rolling(5).mean()
        ma_20 = pd.Series(price).rolling(20).mean()
        features_dict['ma_5_20_ratio'] = ((ma_5 / ma_20 - 1).values.astype('float32'))
        
        # Bollinger position
        bb_mean = pd.Series(price).rolling(20).mean()
        bb_std = pd.Series(price).rolling(20).std()
        bb_pos = ((price - bb_mean) / (2 * bb_std).clip(lower=0.0001)).astype('float32')
        features_dict['bb_position'] = bb_pos
        
        # 6. AGENT METRICS (simplificado)
        print("-> Calculando métricas de agentes...")
        
        # Top 5 agents activity
        top_buyers = df['<buy_agent>'].value_counts().head(5).index
        top_sellers = df['<sell_agent>'].value_counts().head(5).index
        
        top_buyer_active = df['<buy_agent>'].isin(top_buyers).astype('float32').values
        top_seller_active = df['<sell_agent>'].isin(top_sellers).astype('float32').values
        
        features_dict['top_buyers_active'] = top_buyer_active
        features_dict['top_sellers_active'] = top_seller_active
        
        # Institutional flow
        inst_flow = pd.Series(top_buyer_active).rolling(50).sum() - pd.Series(top_seller_active).rolling(50).sum()
        features_dict['institutional_flow'] = inst_flow.values.astype('float32')
        
        # 7. TEMPORAL
        print("-> Calculando features temporais...")
        features_dict['hour'] = df['timestamp'].dt.hour.values.astype('float32')
        features_dict['minute'] = df['timestamp'].dt.minute.values.astype('float32')
        
        # Criar DataFrame final
        print("\n-> Consolidando features...")
        features = pd.DataFrame(features_dict)
        
        # Substituir infinitos e NaN
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(0)
        
        # Garantir float32
        for col in features.columns:
            if features[col].dtype != 'float32':
                features[col] = features[col].astype('float32')
        
        print(f"\n[OK] Total features: {features.shape[1]}")
        print(f"[OK] Memória features: {features.memory_usage().sum() / 1024**2:.1f} MB")
        
        # Liberar memória
        del features_dict
        gc.collect()
        
        return features
    
    def create_balanced_targets(self, df: pd.DataFrame) -> tuple:
        """Cria targets com distribuição balanceada usando 0.5×std"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE TARGETS BALANCEADOS")
        print("=" * 80)
        
        print(f"\nHorizonte: {self.horizon} trades")
        
        # Calcular retornos futuros
        prices = df['<price>'].values
        future_idx = np.arange(len(prices)) + self.horizon
        future_idx = np.clip(future_idx, 0, len(prices) - 1)
        
        future_prices = prices[future_idx]
        returns = (future_prices - prices) / prices
        
        # Ajustar últimos valores
        returns[-self.horizon:] = 0
        
        # Método 0.5×std (configuração ótima)
        returns_valid = returns[:-self.horizon]
        returns_std = returns_valid.std()
        buy_threshold = 0.5 * returns_std
        sell_threshold = -0.5 * returns_std
        
        print(f"\nMétodo: 0.5 × desvio padrão")
        print(f"Desvio padrão dos retornos: {returns_std:.5f} ({returns_std*100:.3f}%)")
        print(f"Threshold BUY:  > {buy_threshold:.5f} ({buy_threshold*100:.3f}%)")
        print(f"Threshold SELL: < {sell_threshold:.5f} ({sell_threshold*100:.3f}%)")
        
        # Criar target
        target = np.zeros(len(df), dtype='int8')
        target[returns > buy_threshold] = 1    # BUY
        target[returns < sell_threshold] = -1  # SELL
        
        # Estatísticas
        unique, counts = np.unique(target, return_counts=True)
        dist = dict(zip(unique, counts))
        total = len(target)
        
        print(f"\nDistribuição real:")
        print(f"  SELL (-1): {dist.get(-1, 0):>8,} ({dist.get(-1, 0)/total*100:>5.1f}%)")
        print(f"  HOLD  (0): {dist.get(0, 0):>8,} ({dist.get(0, 0)/total*100:>5.1f}%)")
        print(f"  BUY   (1): {dist.get(1, 0):>8,} ({dist.get(1, 0)/total*100:>5.1f}%)")
        
        return target, returns_std
    
    def train_models_optimized(self, features: pd.DataFrame, target: np.ndarray):
        """Treina modelos com otimizações de memória"""
        
        print("\n" + "=" * 80)
        print("TREINAMENTO OTIMIZADO")
        print("=" * 80)
        
        # Preparar dados
        mask = ~np.isnan(target)
        mask[-self.horizon:] = False  # Excluir últimos registros sem target válido
        
        X = features[mask]
        y = target[mask]
        
        print(f"\nDados disponíveis: {len(X):,} registros")
        
        # Normalizar
        print("Normalizando features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Split temporal
        split = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Calcular pesos
        classes = np.array([-1, 0, 1])
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"\nPesos das classes: {class_weight_dict}")
        
        # Treinar modelos
        models_config = [
            ("LightGBM", self._train_lightgbm_optimized),
            ("XGBoost", self._train_xgboost_optimized)
        ]
        
        predictions = {}
        importances = {}
        
        for name, train_func in models_config:
            print(f"\n{'='*60}")
            print(f"Treinando {name}...")
            print(f"{'='*60}")
            
            model, pred, metrics, importance = train_func(
                X_train, y_train, X_test, y_test, class_weight_dict, X.columns
            )
            
            self.models[name.lower()] = model
            self.results[name.lower()] = metrics
            predictions[name] = pred
            importances[name] = importance
            
            # Mostrar métricas
            self._print_metrics(name, metrics)
            
            # Liberar memória
            gc.collect()
        
        # Feature importance
        print("\n" + "="*60)
        print("TOP 15 FEATURES MAIS IMPORTANTES")
        print("="*60)
        
        # Combinar importâncias
        avg_imp = pd.DataFrame(index=X.columns)
        for name, imp in importances.items():
            avg_imp[name] = imp
        
        avg_imp['average'] = avg_imp.mean(axis=1)
        avg_imp = avg_imp.sort_values('average', ascending=False)
        
        print(avg_imp[['average']].head(15))
        
        return avg_imp
    
    def _calculate_comprehensive_metrics(self, y_true, y_pred):
        """Calcula métricas abrangentes"""
        
        accuracy = (y_pred == y_true).mean()
        
        # Por classe
        accuracy_sell = (y_pred[y_true == -1] == -1).mean() if (y_true == -1).any() else 0
        accuracy_hold = (y_pred[y_true == 0] == 0).mean() if (y_true == 0).any() else 0
        accuracy_buy = (y_pred[y_true == 1] == 1).mean() if (y_true == 1).any() else 0
        
        # Trading accuracy
        trading_mask = y_true != 0
        if trading_mask.any():
            trading_accuracy = (y_pred[trading_mask] == y_true[trading_mask]).mean()
        else:
            trading_accuracy = 0
        
        # F1 scores
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        return {
            'accuracy': accuracy,
            'accuracy_sell': accuracy_sell,
            'accuracy_hold': accuracy_hold,
            'accuracy_buy': accuracy_buy,
            'trading_accuracy': trading_accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro
        }
    
    def _print_metrics(self, model_name, metrics):
        """Imprime métricas"""
        print(f"\nMétricas {model_name}:")
        print(f"  Overall Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Trading Accuracy: {metrics['trading_accuracy']:.2%}")
        print(f"  F1 Macro: {metrics['f1_macro']:.2%}")
    
    def _train_lightgbm_optimized(self, X_train, y_train, X_test, y_test, class_weights, feature_names):
        """LightGBM otimizado para memória"""
        
        sample_weights = np.zeros(len(y_train))
        for class_val, weight in class_weights.items():
            sample_weights[y_train == class_val] = weight
        
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'num_leaves': 31,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 6,
            'min_data_in_leaf': 100,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'verbose': -1,
            'force_col_wise': True  # Otimização de memória
        }
        
        y_train_lgb = y_train + 1
        y_test_lgb = y_test + 1
        
        lgb_train = lgb.Dataset(X_train, label=y_train_lgb, weight=sample_weights)
        lgb_val = lgb.Dataset(X_test, label=y_test_lgb, reference=lgb_train)
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=200,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )
        
        pred = model.predict(X_test, num_iteration=model.best_iteration)
        pred_class = np.argmax(pred, axis=1) - 1
        
        metrics = self._calculate_comprehensive_metrics(y_test, pred_class)
        
        importance = pd.Series(
            model.feature_importance(importance_type='gain'),
            index=feature_names
        ).sort_values(ascending=False)
        
        return model, pred_class, metrics, importance
    
    def _train_xgboost_optimized(self, X_train, y_train, X_test, y_test, class_weights, feature_names):
        """XGBoost otimizado para memória"""
        
        sample_weights = np.zeros(len(y_train))
        for class_val, weight in class_weights.items():
            sample_weights[y_train == class_val] = weight
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=100,
            reg_alpha=0.5,
            reg_lambda=0.5,
            tree_method='hist',
            eval_metric='mlogloss',
            early_stopping_rounds=30,
            n_jobs=4,  # Limitar paralelismo para economia de memória
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
            index=feature_names
        ).sort_values(ascending=False)
        
        return model, pred, metrics, importance
    
    def save_optimized_results(self, feature_importance: pd.DataFrame, returns_std: float):
        """Salva resultados otimizados"""
        
        output_dir = Path('models/csv_5m_optimized')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*80)
        print("SALVANDO RESULTADOS OTIMIZADOS")
        print("="*80)
        
        # Salvar modelos
        for name, model in self.models.items():
            if 'lightgbm' in name:
                model_file = output_dir / f'{name}_{timestamp}.txt'
                model.save_model(str(model_file))
            else:
                model_file = output_dir / f'{name}_{timestamp}.pkl'
                joblib.dump(model, model_file)
        
        # Salvar scaler
        joblib.dump(self.scaler, output_dir / f'scaler_{timestamp}.pkl')
        
        # Feature importance
        feature_importance.to_csv(output_dir / f'features_{timestamp}.csv')
        
        # Configuração
        config = {
            'training_date': datetime.now().isoformat(),
            'sample_size': self.sample_size,
            'horizon_trades': self.horizon,
            'threshold_method': '0.5_std',
            'returns_std': float(returns_std),
            'memory_optimized': True,
            'chunk_size': self.chunk_size
        }
        
        # Performance
        performance = {
            'models_performance': {}
        }
        
        for name, metrics in self.results.items():
            performance['models_performance'][name] = {
                'accuracy': round(metrics['accuracy'], 4),
                'trading_accuracy': round(metrics['trading_accuracy'], 4),
                'f1_macro': round(metrics['f1_macro'], 4)
            }
        
        # Salvar relatório
        report = {
            'configuration': config,
            'performance': performance
        }
        
        with open(output_dir / f'report_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[OK] Resultados salvos em: {output_dir}")
        
        # Resumo
        print("\n" + "="*80)
        print("RESUMO DO TREINAMENTO OTIMIZADO")
        print("="*80)
        
        for name, metrics in self.results.items():
            print(f"\n{name}:")
            print(f"  Trading Accuracy: {metrics['trading_accuracy']:.2%}")
            print(f"  Overall Accuracy: {metrics['accuracy']:.2%}")

def main():
    """Pipeline principal otimizado"""
    
    print("PIPELINE OTIMIZADO PARA MEMÓRIA - 5M REGISTROS")
    print("Configuração: 1000 trades horizon, 0.5×std threshold\n")
    
    csv_path = r"C:\Users\marth\Downloads\WDO_FUT\WDOFUT_BMF_T.csv"
    
    trainer = CSV5MMemoryOptimizedTrainer(csv_path)
    
    try:
        # 1. Carregar dados otimizados
        df = trainer.load_data_optimized()
        
        # 2. Criar features otimizadas
        features = trainer.create_features_optimized(df)
        
        # 3. Criar targets balanceados
        target, returns_std = trainer.create_balanced_targets(df)
        
        # Liberar DataFrame original
        del df
        gc.collect()
        
        # 4. Treinar modelos otimizados
        feature_importance = trainer.train_models_optimized(features, target)
        
        # 5. Salvar resultados
        trainer.save_optimized_results(feature_importance, returns_std)
        
        # Cleanup final
        del features, target
        gc.collect()
        
        print("\n" + "="*80)
        print("[OK] TREINAMENTO OTIMIZADO CONCLUÍDO!")
        print("="*80)
        
        print("\nConfiguração utilizada:")
        print("- Processamento otimizado para memória")
        print("- Features essenciais apenas")
        print("- 2 modelos principais (LightGBM + XGBoost)")
        print("- Horizonte: 1000 trades")
        print("- Threshold: 0.5 × std")
        
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()