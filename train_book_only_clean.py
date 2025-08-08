"""
Pipeline Book-Only Limpo e Robusto
Versão corrigida para lidar com dados problemáticos
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BookOnlyCleanTrainer:
    """Treina modelo book-only com tratamento robusto de dados"""
    
    def __init__(self):
        self.book_path = Path("data/realtime/book")
        self.models = {}
        self.scaler = RobustScaler()  # Mais robusto que StandardScaler
        
    def load_and_clean_book_data(self, date: str = "20250806"):
        """Carrega e limpa dados do book"""
        
        print("=" * 80)
        print("BOOK-ONLY CLEAN TRAINING")
        print("=" * 80)
        
        # Carregar dados
        training_file = self.book_path / date / "training_ready" / f"training_data_{date}.parquet"
        print(f"\nCarregando: {training_file}")
        
        book_data = pd.read_parquet(training_file)
        print(f"[OK] {len(book_data):,} registros carregados")
        
        # Filtrar tipos relevantes
        mask = book_data['type'].isin(['offer_book', 'price_book'])
        book_data = book_data[mask].copy()
        
        # Verificar e limpar preços
        print("\nVerificando qualidade dos dados...")
        
        # Remover preços zero ou negativos
        if 'price' in book_data.columns:
            invalid_price = (book_data['price'] <= 0) | book_data['price'].isna()
            if invalid_price.any():
                print(f"[LIMPEZA] Removendo {invalid_price.sum()} registros com preço inválido")
                book_data = book_data[~invalid_price]
        
        # Resetar índices
        book_data = book_data.reset_index(drop=True)
        
        print(f"[OK] {len(book_data):,} registros após limpeza")
        print(f"[OK] Tipos: {book_data['type'].value_counts().to_dict()}")
        
        return book_data
    
    def create_simple_features(self, book_data: pd.DataFrame) -> pd.DataFrame:
        """Cria features simples e robustas"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE FEATURES SIMPLES")
        print("=" * 80)
        
        features_list = []
        
        # 1. FEATURES BÁSICAS
        print("\n-> Features básicas...")
        
        # Preço normalizado por ticker
        for ticker in book_data['ticker'].unique():
            ticker_mask = book_data['ticker'] == ticker
            ticker_data = book_data[ticker_mask]
            
            if 'price' in ticker_data.columns and len(ticker_data) > 100:
                prices = ticker_data['price'].values
                
                # Normalizar preços (0-1 por ticker)
                price_min = prices.min()
                price_max = prices.max()
                
                if price_max > price_min:
                    normalized_price = (prices - price_min) / (price_max - price_min)
                else:
                    normalized_price = np.zeros_like(prices)
                
                # Features de preço
                price_features = pd.DataFrame({
                    'price_normalized': normalized_price,
                    'price_pct_change': pd.Series(prices).pct_change().fillna(0).values,
                    'price_rolling_std': pd.Series(prices).rolling(10).std().fillna(0).values / prices.mean()
                }, index=ticker_data.index)
                
                features_list.append(price_features)
        
        # 2. FEATURES DE POSIÇÃO
        print("-> Features de posição...")
        
        position_features = pd.DataFrame({
            'position': book_data['position'].fillna(50),
            'is_top_5': (book_data['position'] <= 5).astype(float),
            'position_normalized': book_data['position'] / 100  # Assumindo max 100 níveis
        }, index=book_data.index)
        
        features_list.append(position_features)
        
        # 3. FEATURES DE VOLUME
        print("-> Features de volume...")
        
        if 'quantity' in book_data.columns:
            qty = book_data['quantity'].fillna(0)
            
            volume_features = pd.DataFrame({
                'quantity_log': np.log1p(qty),
                'quantity_zscore': (qty - qty.mean()) / (qty.std() + 1e-8)
            }, index=book_data.index)
            
            features_list.append(volume_features)
        
        # 4. FEATURES DE LADO
        print("-> Features de lado...")
        
        if 'side' in book_data.columns:
            side_map = {'bid': 0, 'ask': 1}
            side_numeric = book_data['side'].map(side_map).fillna(0.5)
            
            side_features = pd.DataFrame({
                'side': side_numeric,
                'is_bid': (book_data['side'] == 'bid').astype(float),
                'is_ask': (book_data['side'] == 'ask').astype(float)
            }, index=book_data.index)
            
            features_list.append(side_features)
        
        # 5. FEATURES TEMPORAIS
        print("-> Features temporais...")
        
        temporal_features = pd.DataFrame({
            'hour': book_data['hour'],
            'minute': book_data['minute'],
            'time_of_day': (book_data['hour'] * 60 + book_data['minute']) / (18 * 60)  # Normalizado 0-1
        }, index=book_data.index)
        
        features_list.append(temporal_features)
        
        # Combinar todas as features
        features = pd.concat(features_list, axis=1)
        
        # Preencher valores faltantes
        features = features.fillna(0)
        
        # Garantir que não há infinitos
        features = features.replace([np.inf, -np.inf], 0)
        
        # Converter para float32
        for col in features.columns:
            features[col] = features[col].astype('float32')
        
        print(f"\n[OK] {len(features.columns)} features criadas")
        print(f"[OK] Features: {list(features.columns)}")
        
        return features
    
    def create_simple_targets(self, book_data: pd.DataFrame) -> np.ndarray:
        """Cria targets simples baseados em mudança de preço"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE TARGETS SIMPLES")
        print("=" * 80)
        
        targets = np.zeros(len(book_data), dtype='int8')
        
        # Por ticker
        for ticker in book_data['ticker'].unique():
            ticker_mask = book_data['ticker'] == ticker
            ticker_indices = np.where(ticker_mask)[0]
            
            if len(ticker_indices) < 100:
                continue
                
            ticker_data = book_data[ticker_mask]
            
            if 'price' in ticker_data.columns:
                prices = ticker_data['price'].values
                
                # Mudança de preço simples (próximos 10 updates)
                horizon = 10
                
                for i in range(len(prices) - horizon):
                    current_price = prices[i]
                    future_price = prices[i + horizon]
                    
                    if current_price > 0:
                        price_change = (future_price - current_price) / current_price
                        
                        # Thresholds simples
                        if price_change > 0.0001:  # 0.01%
                            targets[ticker_indices[i]] = 1
                        elif price_change < -0.0001:
                            targets[ticker_indices[i]] = -1
                        # else: permanece 0 (HOLD)
        
        # Estatísticas
        unique, counts = np.unique(targets, return_counts=True)
        print("\nDistribuição de targets:")
        for val, count in zip(unique, counts):
            label = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}[val]
            pct = count / len(targets) * 100
            print(f"  {label}: {count:,} ({pct:.1f}%)")
        
        return targets
    
    def train_simple_model(self, features: pd.DataFrame, targets: np.ndarray):
        """Treina modelo LightGBM simples"""
        
        print("\n" + "=" * 80)
        print("TREINAMENTO DO MODELO")
        print("=" * 80)
        
        # Remover registros sem variação
        valid_mask = ~np.isnan(features).any(axis=1)
        X = features[valid_mask]
        y = targets[valid_mask]
        
        print(f"\nDados válidos: {len(X):,}")
        
        # Verificar se temos exemplos de todas as classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print("[ERRO] Dados insuficientes - apenas uma classe encontrada")
            return None
        
        # Normalizar com RobustScaler
        print("Normalizando features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Split temporal
        split = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Parâmetros simples
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'force_row_wise': True
        }
        
        # Treinar
        print("\nTreinando LightGBM...")
        
        lgb_train = lgb.Dataset(X_train, label=y_train + 1)
        lgb_val = lgb.Dataset(X_test, label=y_test + 1, reference=lgb_train)
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(25)]
        )
        
        # Avaliar
        pred = model.predict(X_test, num_iteration=model.best_iteration)
        pred_class = np.argmax(pred, axis=1) - 1
        
        # Métricas
        accuracy = (pred_class == y_test).mean()
        
        # Trading accuracy
        trading_mask = y_test != 0
        if trading_mask.any():
            trading_accuracy = (pred_class[trading_mask] == y_test[trading_mask]).mean()
        else:
            trading_accuracy = 0
        
        print(f"\nResultados:")
        print(f"  Overall Accuracy: {accuracy:.2%}")
        print(f"  Trading Accuracy: {trading_accuracy:.2%}")
        
        # Feature importance
        importance = pd.Series(
            model.feature_importance(importance_type='gain'),
            index=features.columns
        ).sort_values(ascending=False)
        
        print("\nTop 10 Features:")
        print(importance.head(10))
        
        self.models['lightgbm_book_clean'] = model
        
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
            
        output_dir = Path('models/book_clean')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*80)
        print("SALVANDO RESULTADOS")
        print("="*80)
        
        # Salvar modelo
        model = results['model']
        model.save_model(str(output_dir / f'lightgbm_book_clean_{timestamp}.txt'))
        
        # Salvar scaler
        joblib.dump(self.scaler, output_dir / f'scaler_{timestamp}.pkl')
        
        # Salvar feature importance
        results['importance'].to_csv(output_dir / f'features_{timestamp}.csv')
        
        # Metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'model_type': 'book_only_clean',
            'accuracy': float(results['accuracy']),
            'trading_accuracy': float(results['trading_accuracy']),
            'top_features': results['importance'].head(5).to_dict()
        }
        
        with open(output_dir / f'metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[OK] Resultados salvos em: {output_dir}")

def main():
    """Pipeline principal"""
    
    print("BOOK-ONLY CLEAN TRAINING")
    print("Versão robusta para dados com problemas\n")
    
    trainer = BookOnlyCleanTrainer()
    
    try:
        # 1. Carregar e limpar dados
        book_data = trainer.load_and_clean_book_data("20250806")
        
        # 2. Criar features simples
        features = trainer.create_simple_features(book_data)
        
        # 3. Criar targets simples
        targets = trainer.create_simple_targets(book_data)
        
        # 4. Treinar modelo
        results = trainer.train_simple_model(features, targets)
        
        # 5. Salvar resultados
        if results:
            trainer.save_results(results)
            
            print("\n" + "="*80)
            print("RESUMO FINAL")
            print("="*80)
            print("\nModelos treinados:")
            print("1. Tick-Only (CSV): 47% trading accuracy ✓")
            print(f"2. Book-Only (Clean): {results['trading_accuracy']:.1%} trading accuracy ✓")
            print("\nPróximo passo: Implementar HybridStrategy")
        else:
            print("\n[ERRO] Treinamento falhou")
        
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()