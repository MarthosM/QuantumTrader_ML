"""
Pipeline para Treinar Modelo Book-Only
Segue arquitetura HMARL: modelo separado apenas com book data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import xgboost as xgb
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BookOnlyTrainer:
    """Treina modelo usando apenas book data conforme arquitetura HMARL"""
    
    def __init__(self):
        self.book_path = Path("data/realtime/book")
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_book_data(self, date: str = "20250806"):
        """Carrega dados do book"""
        
        print("=" * 80)
        print("BOOK-ONLY MODEL TRAINING (HMARL Architecture)")
        print("=" * 80)
        
        # Carregar training data
        training_file = self.book_path / date / "training_ready" / f"training_data_{date}.parquet"
        
        print(f"\nCarregando: {training_file}")
        book_data = pd.read_parquet(training_file)
        
        print(f"[OK] {len(book_data):,} registros")
        print(f"[OK] Período: {book_data['timestamp'].min()} até {book_data['timestamp'].max()}")
        
        # Filtrar apenas offer_book e price_book
        mask = book_data['type'].isin(['offer_book', 'price_book'])
        book_data = book_data[mask].copy()
        
        print(f"[OK] Dados filtrados: {len(book_data):,} registros")
        print(f"[OK] Tipos: {book_data['type'].value_counts().to_dict()}")
        
        return book_data
    
    def create_book_features(self, book_data: pd.DataFrame) -> pd.DataFrame:
        """Cria features específicas do book para microestrutura"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE FEATURES DE MICROESTRUTURA")
        print("=" * 80)
        
        features_dict = {}
        
        # 1. PRICE FEATURES
        print("\n-> Price features...")
        if 'price' in book_data.columns:
            # Price levels
            features_dict['price'] = book_data['price'].values.astype('float32')
            
            # Price changes por ticker
            for ticker in book_data['ticker'].unique():
                ticker_mask = book_data['ticker'] == ticker
                ticker_data = book_data[ticker_mask]
                
                # Mudanças de preço
                price_change = ticker_data['price'].diff()
                price_pct = ticker_data['price'].pct_change()
                
                # Mapear de volta
                features_dict[f'price_change_{ticker}'] = pd.Series(
                    price_change, index=ticker_data.index
                ).reindex(book_data.index, fill_value=0).values.astype('float32')
                
                features_dict[f'price_pct_{ticker}'] = pd.Series(
                    price_pct, index=ticker_data.index
                ).reindex(book_data.index, fill_value=0).values.astype('float32')
        
        # 2. BOOK DEPTH FEATURES
        print("-> Book depth features...")
        if 'position' in book_data.columns:
            features_dict['position'] = book_data['position'].values.astype('float32')
            features_dict['is_top_book'] = (book_data['position'] <= 5).astype('float32')
            features_dict['is_deep_book'] = (book_data['position'] > 20).astype('float32')
            
            # Position bins
            features_dict['position_bin'] = pd.cut(
                book_data['position'], 
                bins=[0, 5, 10, 20, 50, 100], 
                labels=[1, 2, 3, 4, 5]
            ).astype('float32')
        
        # 3. VOLUME FEATURES
        print("-> Volume features...")
        if 'quantity' in book_data.columns:
            features_dict['quantity'] = book_data['quantity'].values.astype('float32')
            features_dict['quantity_log'] = np.log1p(book_data['quantity']).values.astype('float32')
            
            # Volume por posição
            if 'position' in book_data.columns:
                # Volume nos top levels
                top_mask = book_data['position'] <= 5
                features_dict['is_large_top'] = (
                    (book_data['quantity'] > book_data['quantity'].quantile(0.8)) & top_mask
                ).astype('float32')
        
        # 4. SIDE IMBALANCE
        print("-> Side imbalance features...")
        if 'side' in book_data.columns:
            # Mapear side
            side_map = {'bid': 0, 'ask': 1}
            features_dict['side'] = book_data['side'].map(side_map).values.astype('float32')
            
            # Calcular imbalance por timestamp e ticker
            print("  Calculando imbalances...")
            imbalances = []
            
            grouped = book_data.groupby(['timestamp', 'ticker'])
            for (ts, ticker), group in tqdm(grouped, desc="  Grupos"):
                bid_vol = group[group['side'] == 'bid']['quantity'].sum()
                ask_vol = group[group['side'] == 'ask']['quantity'].sum()
                total = bid_vol + ask_vol
                
                if total > 0:
                    imbalance = (bid_vol - ask_vol) / total
                    bid_ratio = bid_vol / total
                    ask_ratio = ask_vol / total
                else:
                    imbalance = 0
                    bid_ratio = 0.5
                    ask_ratio = 0.5
                
                for idx in group.index:
                    imbalances.append({
                        'idx': idx,
                        'imbalance': imbalance,
                        'bid_ratio': bid_ratio,
                        'ask_ratio': ask_ratio
                    })
            
            # Mapear de volta
            imb_df = pd.DataFrame(imbalances).set_index('idx')
            features_dict['book_imbalance'] = imb_df['imbalance'].reindex(book_data.index, fill_value=0).values.astype('float32')
            features_dict['bid_ratio'] = imb_df['bid_ratio'].reindex(book_data.index, fill_value=0.5).values.astype('float32')
            features_dict['ask_ratio'] = imb_df['ask_ratio'].reindex(book_data.index, fill_value=0.5).values.astype('float32')
        
        # 5. TEMPORAL FEATURES
        print("-> Temporal features...")
        features_dict['hour'] = book_data['hour'].values.astype('float32')
        features_dict['minute'] = book_data['minute'].values.astype('float32')
        features_dict['time_of_day'] = (book_data['hour'] * 60 + book_data['minute']) / (24 * 60)
        
        # 6. UPDATE FREQUENCY
        print("-> Update frequency features...")
        # Contar updates por minuto
        book_data_copy = book_data.copy()
        book_data_copy['minute_key'] = book_data_copy['timestamp'].dt.floor('1min')
        updates_per_min = book_data_copy.groupby(['minute_key', 'ticker']).size()
        
        # Mapear de forma mais eficiente
        book_data_copy['freq_key'] = list(zip(book_data_copy['minute_key'], book_data_copy['ticker']))
        book_data_copy['update_frequency'] = book_data_copy['freq_key'].map(updates_per_min.to_dict()).fillna(0)
        
        features_dict['update_frequency'] = book_data_copy['update_frequency'].values.astype('float32')
        
        # Converter para DataFrame
        features = pd.DataFrame(features_dict)
        
        print(f"\n[OK] {len(features.columns)} features criadas")
        
        return features, book_data
    
    def create_book_targets(self, book_data: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
        """Cria targets baseados em movimentos futuros do book"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE TARGETS PARA BOOK")
        print("=" * 80)
        
        # Resetar índices para garantir alinhamento
        book_data = book_data.reset_index(drop=True)
        
        # Agrupar por ticker para calcular retornos futuros
        targets = np.zeros(len(book_data), dtype='int16')
        
        for ticker in book_data['ticker'].unique():
            print(f"\nProcessando {ticker}...")
            
            ticker_mask = book_data['ticker'] == ticker
            ticker_data = book_data[ticker_mask].copy()
            ticker_indices = np.where(ticker_mask)[0]
            
            if 'price' in ticker_data.columns:
                prices = ticker_data['price'].values
                
                # Calcular mudança de preço nos próximos N updates
                horizon = 20  # 20 updates ahead
                
                future_prices = np.zeros_like(prices)
                for i in range(len(prices) - horizon):
                    future_prices[i] = prices[i + horizon]
                
                # Últimos valores sem futuro
                future_prices[-horizon:] = prices[-horizon:]
                
                # Retornos
                returns = (future_prices - prices) / prices
                returns[-horizon:] = np.nan
                
                # Thresholds baseados no desvio padrão
                valid_returns = returns[~np.isnan(returns)]
                if len(valid_returns) > 0:
                    std = valid_returns.std()
                    threshold = 0.5 * std
                    
                    # Criar targets
                    ticker_targets = np.zeros(len(prices), dtype='int16')
                    ticker_targets[returns > threshold] = 1
                    ticker_targets[returns < -threshold] = -1
                    ticker_targets[np.isnan(returns)] = -999
                    
                    # Mapear de volta usando índices numéricos
                    targets[ticker_indices] = ticker_targets
                    
                    # Stats
                    valid_targets = ticker_targets[ticker_targets != -999]
                    unique, counts = np.unique(valid_targets, return_counts=True)
                    
                    print(f"  Threshold: ±{threshold:.5f} ({threshold*100:.3f}%)")
                    print(f"  Distribuição: {dict(zip(unique, counts))}")
        
        return targets
    
    def train_book_models(self, features: pd.DataFrame, targets: np.ndarray):
        """Treina modelos específicos para book data"""
        
        print("\n" + "=" * 80)
        print("TREINAMENTO DE MODELOS BOOK-ONLY")
        print("=" * 80)
        
        # Remover inválidos
        valid_mask = targets != -999
        X = features[valid_mask]
        y = targets[valid_mask]
        
        print(f"\nDados válidos: {len(X):,}")
        
        # Normalizar
        X_scaled = self.scaler.fit_transform(X)
        
        # Split temporal
        split = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Class weights
        classes = np.array([-1, 0, 1])
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        results = {}
        
        # 1. LightGBM
        print("\n" + "="*60)
        print("Treinando LightGBM para Book...")
        print("="*60)
        
        lgb_model, lgb_metrics = self._train_lightgbm(
            X_train, y_train, X_test, y_test, 
            class_weight_dict, features.columns
        )
        
        self.models['lightgbm_book'] = lgb_model
        results['lightgbm'] = lgb_metrics
        
        # 2. XGBoost
        print("\n" + "="*60)
        print("Treinando XGBoost para Book...")
        print("="*60)
        
        xgb_model, xgb_metrics = self._train_xgboost(
            X_train, y_train, X_test, y_test,
            class_weight_dict, features.columns
        )
        
        self.models['xgboost_book'] = xgb_model
        results['xgboost'] = xgb_metrics
        
        return results
    
    def _train_lightgbm(self, X_train, y_train, X_test, y_test, class_weights, feature_names):
        """Treina LightGBM para book data"""
        
        # Sample weights
        sample_weights = np.zeros(len(y_train))
        for class_val, weight in class_weights.items():
            sample_weights[y_train == class_val] = weight
        
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'num_leaves': 63,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 6,
            'min_data_in_leaf': 50,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1
        }
        
        lgb_train = lgb.Dataset(X_train, label=y_train + 1, weight=sample_weights)
        lgb_val = lgb.Dataset(X_test, label=y_test + 1, reference=lgb_train)
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=300,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        
        # Evaluate
        pred = model.predict(X_test, num_iteration=model.best_iteration)
        pred_class = np.argmax(pred, axis=1) - 1
        
        # Metrics
        accuracy = (pred_class == y_test).mean()
        trading_mask = y_test != 0
        trading_accuracy = (pred_class[trading_mask] == y_test[trading_mask]).mean() if trading_mask.any() else 0
        
        # Feature importance
        importance = pd.Series(
            model.feature_importance(importance_type='gain'),
            index=feature_names
        ).sort_values(ascending=False)
        
        print(f"\nResultados LightGBM:")
        print(f"  Overall Accuracy: {accuracy:.2%}")
        print(f"  Trading Accuracy: {trading_accuracy:.2%}")
        print(f"\nTop 10 Features:")
        print(importance.head(10))
        
        metrics = {
            'accuracy': accuracy,
            'trading_accuracy': trading_accuracy,
            'importance': importance
        }
        
        return model, metrics
    
    def _train_xgboost(self, X_train, y_train, X_test, y_test, class_weights, feature_names):
        """Treina XGBoost para book data"""
        
        # Sample weights
        sample_weights = np.zeros(len(y_train))
        for class_val, weight in class_weights.items():
            sample_weights[y_train == class_val] = weight
        
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=50,
            reg_alpha=0.1,
            reg_lambda=0.1,
            tree_method='hist',
            eval_metric='mlogloss',
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=42
        )
        
        model.fit(
            X_train, y_train + 1,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test + 1)],
            verbose=False
        )
        
        # Evaluate
        pred = model.predict(X_test) - 1
        
        accuracy = (pred == y_test).mean()
        trading_mask = y_test != 0
        trading_accuracy = (pred[trading_mask] == y_test[trading_mask]).mean() if trading_mask.any() else 0
        
        print(f"\nResultados XGBoost:")
        print(f"  Overall Accuracy: {accuracy:.2%}")
        print(f"  Trading Accuracy: {trading_accuracy:.2%}")
        
        metrics = {
            'accuracy': accuracy,
            'trading_accuracy': trading_accuracy
        }
        
        return model, metrics
    
    def save_models(self, results: dict):
        """Salva modelos book-only"""
        
        output_dir = Path('models/book_only')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*80)
        print("SALVANDO MODELOS BOOK-ONLY")
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
        joblib.dump(self.scaler, output_dir / f'scaler_book_{timestamp}.pkl')
        
        # Salvar metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'model_type': 'book_only',
            'architecture': 'HMARL_dual_training',
            'purpose': 'microstructure_timing',
            'results': {
                name: {
                    'accuracy': float(metrics['accuracy']),
                    'trading_accuracy': float(metrics['trading_accuracy'])
                }
                for name, metrics in results.items()
            }
        }
        
        with open(output_dir / f'metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[OK] Modelos salvos em: {output_dir}")
        
        # Resumo final
        print("\n" + "="*80)
        print("RESUMO DO TREINAMENTO BOOK-ONLY")
        print("="*80)
        print("\nArquitetura HMARL:")
        print("- Modelo 1: Tick-Only (histórico longo) ✓")
        print("- Modelo 2: Book-Only (microestrutura) ✓")
        print("\nPróximos passos:")
        print("1. Implementar HybridStrategy para combinar modelos")
        print("2. Usar tick-only para regime/tendência")
        print("3. Usar book-only para timing de entrada/saída")

def main():
    """Pipeline principal"""
    
    print("BOOK-ONLY MODEL TRAINING")
    print("Seguindo arquitetura HMARL com modelos separados\n")
    
    trainer = BookOnlyTrainer()
    
    try:
        # 1. Carregar book data
        book_data = trainer.load_book_data("20250806")
        
        # 2. Criar features de microestrutura
        features, book_data_processed = trainer.create_book_features(book_data)
        
        # 3. Criar targets baseados em book
        targets = trainer.create_book_targets(book_data_processed, features)
        
        # 4. Treinar modelos book-only
        results = trainer.train_book_models(features, targets)
        
        # 5. Salvar modelos
        trainer.save_models(results)
        
        print("\n[OK] TREINAMENTO BOOK-ONLY CONCLUÍDO!")
        
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()