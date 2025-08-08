"""
Pipeline Book-Only com Complexidade Moderada
Versão balanceada entre simplicidade e realismo
Segue arquitetura HMARL - modelo separado para book
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class BookModerateTrainer:
    """Treina modelo book-only com complexidade moderada"""
    
    def __init__(self):
        self.book_path = Path("data/realtime/book")
        self.models = {}
        self.scaler = RobustScaler()
        self.horizon = 20  # 20 book updates ahead
        
    def load_and_validate_book_data(self, date: str = "20250806"):
        """Carrega e valida dados do book"""
        
        print("=" * 80)
        print("BOOK-ONLY MODERATE COMPLEXITY TRAINING")
        print("=" * 80)
        
        # Carregar dados
        training_file = self.book_path / date / "training_ready" / f"training_data_{date}.parquet"
        print(f"\nCarregando: {training_file}")
        
        book_data = pd.read_parquet(training_file)
        print(f"[OK] {len(book_data):,} registros carregados")
        
        # Filtrar apenas book data
        mask = book_data['type'].isin(['offer_book', 'price_book'])
        book_data = book_data[mask].copy()
        
        # Validação básica
        if 'price' in book_data.columns:
            # Remover preços inválidos
            valid_price = (book_data['price'] > 0) & book_data['price'].notna()
            book_data = book_data[valid_price]
            
            # Remover outliers extremos (preços que mudam mais de 10% de uma vez)
            for ticker in book_data['ticker'].unique():
                ticker_mask = book_data['ticker'] == ticker
                ticker_data = book_data[ticker_mask]
                
                price_changes = ticker_data['price'].pct_change().abs()
                outlier_mask = price_changes > 0.1  # 10% change
                
                if outlier_mask.any():
                    print(f"[LIMPEZA] Removendo {outlier_mask.sum()} outliers de {ticker}")
                    # Remover outliers do dataframe principal
                    outlier_indices = ticker_data[outlier_mask].index
                    book_data = book_data.drop(outlier_indices)
        
        # Resetar índices
        book_data = book_data.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
        
        print(f"[OK] {len(book_data):,} registros após limpeza")
        print(f"[OK] Período: {book_data['timestamp'].min()} até {book_data['timestamp'].max()}")
        
        return book_data
    
    def create_moderate_features(self, book_data: pd.DataFrame) -> pd.DataFrame:
        """Cria features moderadamente complexas e eficientes"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE FEATURES MODERADAS")
        print("=" * 80)
        
        features_list = []
        
        # Processar por ticker
        for ticker in book_data['ticker'].unique():
            print(f"\nProcessando {ticker}...")
            ticker_mask = book_data['ticker'] == ticker
            ticker_data = book_data[ticker_mask].copy()
            
            if len(ticker_data) < 1000:
                continue
            
            ticker_features = pd.DataFrame(index=ticker_data.index)
            
            # 1. PRICE FEATURES ESSENCIAIS
            print("  -> Price features...")
            prices = ticker_data['price'].values
            
            # Returns básicos
            ticker_features['return_1'] = np.concatenate([[0], (prices[1:] - prices[:-1]) / prices[:-1]])
            ticker_features['log_return_1'] = np.concatenate([[0], np.log(prices[1:] / prices[:-1])])
            
            # Moving averages rápidas
            for window in [5, 10, 20]:
                if window < len(prices):
                    ma = pd.Series(prices).rolling(window).mean()
                    ticker_features[f'price_ma_{window}'] = (prices - ma) / ma
            
            # Volatilidade simples
            returns = ticker_features['return_1'].values
            for window in [10, 30]:
                vol = pd.Series(returns).rolling(window).std()
                ticker_features[f'volatility_{window}'] = vol
            
            # 2. BOOK POSITION FEATURES
            print("  -> Book position features...")
            if 'position' in ticker_data.columns:
                positions = ticker_data['position'].values
                
                # Position básica
                ticker_features['position'] = positions
                ticker_features['position_inverse'] = 1.0 / (positions + 1)
                ticker_features['is_top_5'] = (positions <= 5).astype(float)
                ticker_features['is_top_10'] = (positions <= 10).astype(float)
            
            # 3. VOLUME FEATURES
            print("  -> Volume features...")
            if 'quantity' in ticker_data.columns:
                quantities = ticker_data['quantity'].values
                
                # Volume básico
                ticker_features['quantity_log'] = np.log1p(quantities)
                
                # Volume relativo
                vol_ma = pd.Series(quantities).rolling(20).mean()
                ticker_features['volume_ratio'] = quantities / (vol_ma + 1)
                
                # Large volume indicator
                vol_threshold = np.percentile(quantities, 90)
                ticker_features['is_large_volume'] = (quantities > vol_threshold).astype(float)
            
            # 4. ORDER FLOW SIMPLES
            print("  -> Order flow...")
            if 'side' in ticker_data.columns:
                # Calcular OFI por janelas de 1 minuto
                ticker_data['minute'] = ticker_data['timestamp'].dt.floor('1min')
                
                ofi_dict = {}
                for minute, group in ticker_data.groupby('minute'):
                    if 'quantity' in group.columns:
                        bid_vol = group[group['side'] == 'bid']['quantity'].sum()
                        ask_vol = group[group['side'] == 'ask']['quantity'].sum()
                        total = bid_vol + ask_vol
                        
                        if total > 0:
                            ofi = (bid_vol - ask_vol) / total
                        else:
                            ofi = 0
                        
                        for idx in group.index:
                            ofi_dict[idx] = ofi
                
                # Mapear OFI
                ticker_features['ofi'] = pd.Series(ofi_dict, index=ticker_data.index).fillna(0)
                
                # Side indicator
                ticker_features['is_bid'] = (ticker_data['side'] == 'bid').astype(float)
                ticker_features['is_ask'] = (ticker_data['side'] == 'ask').astype(float)
            
            # 5. TEMPORAL FEATURES
            print("  -> Temporal features...")
            if 'hour' in ticker_data.columns and 'minute' in ticker_data.columns:
                ticker_features['hour'] = ticker_data['hour'].values.astype(float)
                ticker_features['minute'] = ticker_data['minute'].values.astype(float)
                ticker_features['time_normalized'] = (ticker_features['hour'] * 60 + ticker_features['minute']) / (18 * 60)
                
                # Session indicators
                ticker_features['is_morning'] = (ticker_features['hour'] < 12).astype(float)
                ticker_features['is_afternoon'] = (ticker_features['hour'] >= 12).astype(float)
            
            # 6. MOMENTUM SIMPLES
            print("  -> Momentum...")
            for lag in [5, 10, 20]:
                if lag < len(prices):
                    momentum = np.zeros(len(prices))
                    momentum[lag:] = (prices[lag:] - prices[:-lag]) / prices[:-lag]
                    ticker_features[f'momentum_{lag}'] = momentum
            
            # Adicionar à lista
            features_list.append(ticker_features)
        
        # Combinar features
        if not features_list:
            raise ValueError("Nenhuma feature foi criada!")
        
        features = pd.concat(features_list, axis=0).sort_index()
        
        # Limpar dados
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(0)
        
        # Converter para float32
        for col in features.columns:
            features[col] = features[col].astype('float32')
        
        print(f"\n[OK] {len(features.columns)} features criadas")
        print(f"[OK] Shape: {features.shape}")
        
        return features
    
    def create_realistic_targets(self, book_data: pd.DataFrame) -> np.ndarray:
        """Cria targets realistas baseados em movimentos de preço"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE TARGETS REALISTAS")
        print("=" * 80)
        
        targets = np.full(len(book_data), -99, dtype='int8')
        
        # Processar por ticker
        for ticker in book_data['ticker'].unique():
            print(f"\nProcessando targets para {ticker}...")
            
            ticker_mask = book_data['ticker'] == ticker
            ticker_indices = np.where(ticker_mask)[0]
            ticker_data = book_data[ticker_mask]
            
            if len(ticker_data) < self.horizon * 2:
                continue
            
            prices = ticker_data['price'].values
            
            # Calcular retornos futuros
            returns = np.zeros(len(prices))
            
            for i in range(len(prices) - self.horizon):
                future_price = prices[i + self.horizon]
                current_price = prices[i]
                
                if current_price > 0:
                    returns[i] = (future_price - current_price) / current_price
            
            # Usar apenas retornos válidos
            valid_mask = (returns != 0) & np.isfinite(returns)
            valid_returns = returns[valid_mask]
            
            if len(valid_returns) > 100:
                # Calcular percentis para thresholds dinâmicos
                p60 = np.percentile(valid_returns, 60)  # Buy threshold
                p40 = np.percentile(valid_returns, 40)  # Sell threshold
                
                # Garantir thresholds mínimos
                min_threshold = 0.0002  # 0.02%
                if p60 < min_threshold:
                    p60 = min_threshold
                if p40 > -min_threshold:
                    p40 = -min_threshold
                
                print(f"  Horizonte: {self.horizon} updates")
                print(f"  Buy threshold (P60): {p60:.5f} ({p60*100:.3f}%)")
                print(f"  Sell threshold (P40): {p40:.5f} ({p40*100:.3f}%)")
                
                # Criar targets
                ticker_targets = np.zeros(len(prices), dtype='int8')
                ticker_targets[returns > p60] = 1
                ticker_targets[returns < p40] = -1
                
                # Marcar últimos como inválidos
                ticker_targets[-self.horizon:] = -99
                
                # Mapear de volta
                targets[ticker_indices] = ticker_targets
                
                # Stats
                valid_targets = ticker_targets[ticker_targets != -99]
                unique, counts = np.unique(valid_targets, return_counts=True)
                print(f"  Distribuição: {dict(zip(unique, counts))}")
                
                # Verificar balance
                if len(unique) > 1:
                    balance = min(counts) / max(counts)
                    print(f"  Balance ratio: {balance:.2f}")
        
        return targets
    
    def train_moderate_model(self, features: pd.DataFrame, targets: np.ndarray):
        """Treina modelo com configuração moderada"""
        
        print("\n" + "=" * 80)
        print("TREINAMENTO DO MODELO MODERADO")
        print("=" * 80)
        
        # Remover inválidos
        valid_mask = targets != -99
        X = features[valid_mask]
        y = targets[valid_mask]
        
        print(f"\nDados válidos: {len(X):,}")
        
        # Verificar classes
        unique_classes = np.unique(y)
        print(f"Classes encontradas: {unique_classes}")
        
        if len(unique_classes) < 2:
            print("[ERRO] Apenas uma classe encontrada!")
            return None
        
        # Normalizar
        print("\nNormalizando features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Split temporal 80/20
        split = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Class weights
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
        sample_weights = np.zeros(len(y_train))
        for class_val, weight in zip(unique_classes, class_weights):
            sample_weights[y_train == class_val] = weight
        
        # Parâmetros moderados
        params = {
            'objective': 'multiclass',
            'num_class': 3,  # -1, 0, 1
            'metric': 'multi_logloss',
            'num_leaves': 63,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 7,
            'min_data_in_leaf': 50,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'force_row_wise': True
        }
        
        # Treinar
        print("\nTreinando LightGBM...")
        
        # Ajustar labels para LightGBM (deve ser 0, 1, 2)
        y_train_lgb = y_train + 1
        y_test_lgb = y_test + 1
        
        lgb_train = lgb.Dataset(X_train, label=y_train_lgb, weight=sample_weights)
        lgb_val = lgb.Dataset(X_test, label=y_test_lgb, reference=lgb_train)
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=300,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        
        # Avaliar
        pred = model.predict(X_test, num_iteration=model.best_iteration)
        pred_class = np.argmax(pred, axis=1) - 1  # Voltar para -1, 0, 1
        
        # Métricas
        accuracy = (pred_class == y_test).mean()
        
        # Trading accuracy (ignorando HOLD)
        trading_mask = y_test != 0
        if trading_mask.any():
            trading_accuracy = (pred_class[trading_mask] == y_test[trading_mask]).mean()
        else:
            trading_accuracy = 0
        
        print(f"\nResultados:")
        print(f"  Overall Accuracy: {accuracy:.2%}")
        print(f"  Trading Accuracy: {trading_accuracy:.2%}")
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        print(classification_report(y_test, pred_class, 
                                  labels=[-1, 0, 1],
                                  target_names=['SELL', 'HOLD', 'BUY']))
        
        # Feature importance
        importance = pd.Series(
            model.feature_importance(importance_type='gain'),
            index=features.columns
        ).sort_values(ascending=False)
        
        print("\nTop 15 Features:")
        for i, (feat, imp) in enumerate(importance.head(15).items(), 1):
            print(f"{i:2d}. {feat:25s} {imp:8.2f}")
        
        self.models['lightgbm_book_moderate'] = model
        
        return {
            'model': model,
            'accuracy': accuracy,
            'trading_accuracy': trading_accuracy,
            'importance': importance
        }
    
    def save_results(self, results: dict):
        """Salva modelo e resultados"""
        
        if not results:
            print("\n[ERRO] Nenhum resultado para salvar")
            return
            
        output_dir = Path('models/book_moderate')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*80)
        print("SALVANDO RESULTADOS")
        print("="*80)
        
        # Salvar modelo
        model = results['model']
        model.save_model(str(output_dir / f'lightgbm_book_moderate_{timestamp}.txt'))
        
        # Salvar scaler
        joblib.dump(self.scaler, output_dir / f'scaler_{timestamp}.pkl')
        
        # Salvar feature importance
        results['importance'].to_csv(output_dir / f'features_{timestamp}.csv')
        
        # Metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'model_type': 'book_only_moderate',
            'architecture': 'HMARL_separate',
            'horizon': self.horizon,
            'accuracy': float(results['accuracy']),
            'trading_accuracy': float(results['trading_accuracy']),
            'top_features': results['importance'].head(10).to_dict()
        }
        
        with open(output_dir / f'metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[OK] Resultados salvos em: {output_dir}")

def main():
    """Pipeline principal moderado"""
    
    print("BOOK-ONLY MODERATE COMPLEXITY TRAINING")
    print("Versão otimizada com complexidade balanceada\n")
    
    trainer = BookModerateTrainer()
    
    try:
        # 1. Carregar e validar dados
        book_data = trainer.load_and_validate_book_data("20250806")
        
        # 2. Criar features moderadas
        features = trainer.create_moderate_features(book_data)
        
        # 3. Criar targets realistas
        targets = trainer.create_realistic_targets(book_data)
        
        # 4. Treinar modelo
        results = trainer.train_moderate_model(features, targets)
        
        if results:
            # 5. Salvar resultados
            trainer.save_results(results)
            
            print("\n" + "="*80)
            print("RESUMO FINAL - ARQUITETURA HMARL")
            print("="*80)
            print("\nModelos treinados (separados):")
            print("1. Tick-Only (CSV): 47% trading accuracy [OK]")
            print(f"2. Book-Only (Moderate): {results['trading_accuracy']:.1%} trading accuracy [OK]")
            print("\nPróximos passos:")
            print("1. Implementar HybridStrategy")
            print("2. Validar em dados real-time")
            print("3. Ajustar thresholds baseado em performance")
        else:
            print("\n[ERRO] Treinamento falhou")
        
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()