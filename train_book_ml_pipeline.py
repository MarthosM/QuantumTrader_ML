"""
Pipeline de Treinamento com Book Data
Combina dados tick-a-tick com dados do book para melhorar accuracy
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

class BookMLPipeline:
    """Pipeline para treinar modelos com dados do book"""
    
    def __init__(self):
        self.book_path = Path("data/realtime/book")
        self.csv_path = Path(r"C:\Users\marth\Downloads\WDO_FUT\WDOFUT_BMF_T.csv")
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.horizon = 100  # 100 trades ahead (mais curto para book data)
        
    def load_book_data(self, date: str = "20250806"):
        """Carrega dados consolidados do book"""
        
        print("=" * 80)
        print("CARREGAMENTO DE BOOK DATA")
        print("=" * 80)
        
        # Paths para os arquivos consolidados
        consolidated_path = self.book_path / date / "consolidated"
        training_path = self.book_path / date / "training_ready"
        
        # Verificar arquivo de training
        training_file = training_path / f"training_data_{date}.parquet"
        if training_file.exists():
            print(f"\n[OK] Carregando dados de training prontos: {training_file}")
            book_data = pd.read_parquet(training_file)
            print(f"[OK] {len(book_data):,} registros carregados")
            return book_data
        
        # Verificar se há coluna 'type' nos dados
        if 'type' in book_data.columns:
            print(f"[OK] Tipos de dados: {book_data['type'].value_counts().to_dict()}")
        
        # Ordenar por timestamp se disponível
        if 'timestamp' in book_data.columns:
            book_data = book_data.sort_values('timestamp').reset_index(drop=True)
        
        return book_data
    
    def load_tick_data(self, n_records: int = 1_000_000):
        """Carrega dados tick-a-tick para complementar"""
        
        print("\n" + "=" * 80)
        print("CARREGAMENTO DE TICK DATA")
        print("=" * 80)
        
        dtypes = {
            '<date>': 'uint32',
            '<time>': 'uint32',
            '<price>': 'float32',
            '<qty>': 'uint16',
            '<vol>': 'float32',
            '<buy_agent>': 'category',
            '<sell_agent>': 'category',
            '<trade_type>': 'category'
        }
        
        print(f"\nCarregando {n_records:,} registros tick-a-tick...")
        tick_data = pd.read_csv(self.csv_path, nrows=n_records, dtype=dtypes)
        
        # Criar timestamp
        tick_data['timestamp'] = pd.to_datetime(
            tick_data['<date>'].astype(str) + ' ' + tick_data['<time>'].astype(str).str.zfill(6),
            format='%Y%m%d %H%M%S'
        )
        
        tick_data = tick_data.sort_values('timestamp').reset_index(drop=True)
        
        print(f"[OK] Período: {tick_data['timestamp'].min()} até {tick_data['timestamp'].max()}")
        
        return tick_data
    
    def create_book_features(self, book_data: pd.DataFrame) -> pd.DataFrame:
        """Cria features específicas do book"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE FEATURES DO BOOK")
        print("=" * 80)
        
        features = pd.DataFrame(index=book_data.index)
        
        # Separar por tipo de dado (usar coluna 'type')
        type_col = 'type' if 'type' in book_data.columns else 'data_type'
        
        offer_mask = book_data[type_col] == 'offer_book'
        price_mask = book_data[type_col] == 'price_book'
        tiny_mask = book_data[type_col] == 'tiny_book'
        daily_mask = book_data[type_col] == 'daily'
        
        # 1. FEATURES BASEADAS NO TIPO DE BOOK
        if offer_mask.any() or price_mask.any():
            print("\n-> Features de Book...")
            
            # Para offer_book e price_book
            book_mask = offer_mask | price_mask
            book_data_subset = book_data[book_mask]
            
            # Features básicas de preço e quantidade
            if 'price' in book_data_subset.columns:
                features.loc[book_mask, 'book_price'] = book_data_subset['price']
                
                # Mudanças de preço
                price_changes = book_data_subset.groupby('ticker')['price'].diff()
                features.loc[book_mask, 'price_change'] = price_changes
                features.loc[book_mask, 'price_change_pct'] = book_data_subset.groupby('ticker')['price'].pct_change()
            
            if 'quantity' in book_data_subset.columns:
                features.loc[book_mask, 'book_quantity'] = book_data_subset['quantity']
                
                # Volume por posição
                if 'position' in book_data_subset.columns:
                    # Top of book (position 0-5)
                    top_book = book_data_subset['position'] <= 5
                    features.loc[book_mask & top_book, 'is_top_book'] = 1.0
            
            # Side do book (0=buy, 1=sell)
            if 'side' in book_data_subset.columns:
                features.loc[book_mask, 'book_side'] = book_data_subset['side']
                
                # Calcular imbalance por ticker e timestamp
                if 'timestamp' in book_data_subset.columns and 'quantity' in book_data_subset.columns:
                    # Agrupar por timestamp e ticker
                    for (ts, ticker), group in book_data_subset.groupby(['timestamp', 'ticker']):
                        buy_vol = group[group['side'] == 0]['quantity'].sum()
                        sell_vol = group[group['side'] == 1]['quantity'].sum()
                        total_vol = buy_vol + sell_vol
                        
                        if total_vol > 0:
                            imbalance = (buy_vol - sell_vol) / total_vol
                            mask = (book_data['timestamp'] == ts) & (book_data['ticker'] == ticker)
                            features.loc[mask & book_mask, 'book_imbalance'] = imbalance
        
        # 2. FEATURES DE DAILY/TINY
        if daily_mask.any() or tiny_mask.any():
            print("-> Features de Summary Data...")
            summary_mask = daily_mask | tiny_mask
            summary_data = book_data[summary_mask]
            
            # Volume metrics
            if 'volume_total' in summary_data.columns:
                features.loc[summary_mask, 'volume_total'] = summary_data['volume_total']
                features.loc[summary_mask, 'volume_delta'] = summary_data.get('volume_delta', 0)
            
            # Trade count metrics
            if 'trades_total' in summary_data.columns:
                features.loc[summary_mask, 'trades_total'] = summary_data['trades_total']
                features.loc[summary_mask, 'trades_delta'] = summary_data.get('trades_delta', 0)
        
        # 3. FEATURES AGREGADAS
        print("-> Features Agregadas...")
        
        # Contagem de updates por tipo
        if type_col in book_data.columns:
            type_counts = book_data.groupby(['timestamp', 'ticker'])[type_col].value_counts().unstack(fill_value=0)
            
            # Ratio de offer vs price updates
            if 'offer_book' in type_counts.columns and 'price_book' in type_counts.columns:
                update_ratio = type_counts['offer_book'] / (type_counts['price_book'] + 1)
                
                # Mapear de volta para o DataFrame principal
                for (ts, ticker), ratio in update_ratio.items():
                    mask = (book_data['timestamp'] == ts) & (book_data['ticker'] == ticker)
                    features.loc[mask, 'offer_price_ratio'] = ratio
        
        # 4. FEATURES TEMPORAIS
        print("-> Features Temporais...")
        if 'hour' in book_data.columns:
            features['hour'] = book_data['hour']
        if 'minute' in book_data.columns:
            features['minute'] = book_data['minute']
        if 'timestamp' in book_data.columns:
            features['seconds_since_open'] = (
                (book_data['timestamp'] - book_data['timestamp'].dt.normalize() - pd.Timedelta(hours=9))
                .dt.total_seconds()
            )
        
        # Preencher NaN
        features = features.fillna(0)
        
        # Garantir float32
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                features[col] = features[col].astype('float32')
        
        print(f"\n[OK] Total de features do book: {features.shape[1]}")
        
        return features
    
    def merge_tick_and_book_data(self, tick_data: pd.DataFrame, book_data: pd.DataFrame):
        """Merge dados tick com book baseado em timestamp"""
        
        print("\n" + "=" * 80)
        print("MERGE DE TICK E BOOK DATA")
        print("=" * 80)
        
        # Garantir que ambos têm timestamp
        if 'timestamp' not in tick_data.columns or 'timestamp' not in book_data.columns:
            raise ValueError("Ambos datasets precisam ter coluna timestamp")
        
        # Merge_asof para encontrar book data mais próximo para cada tick
        print("\nRealizando merge temporal...")
        
        # Preparar book data - usar type se disponível
        type_col = 'type' if 'type' in book_data.columns else 'data_type'
        
        # Filtrar apenas dados relevantes (offer_book ou price_book)
        if type_col in book_data.columns:
            relevant_types = ['offer_book', 'price_book']
            book_subset = book_data[book_data[type_col].isin(relevant_types)].copy()
        else:
            book_subset = book_data.copy()
        
        book_subset = book_subset.sort_values('timestamp')
        
        # Selecionar colunas importantes do book
        book_cols = ['timestamp', 'ticker', 'price', 'quantity', 'side', 'position']
        book_cols = [col for col in book_cols if col in book_subset.columns]
        
        # Adicionar features calculadas se existirem
        for col in ['book_imbalance', 'book_price', 'price_change']:
            if col in book_subset.columns:
                book_cols.append(col)
        
        book_subset = book_subset[book_cols].copy()
        
        # Merge asof (encontra book mais próximo antes de cada tick)
        merged = pd.merge_asof(
            tick_data.sort_values('timestamp'),
            book_subset,
            on='timestamp',
            direction='backward',
            tolerance=pd.Timedelta('1min')
        )
        
        print(f"[OK] Merge completo: {len(merged):,} registros")
        
        # Verificar quantos registros têm dados do book
        if 'price_y' in merged.columns:  # price do book
            print(f"[OK] Registros com book data: {merged['price_y'].notna().sum():,}")
        elif 'book_price' in merged.columns:
            print(f"[OK] Registros com book data: {merged['book_price'].notna().sum():,}")
        
        return merged
    
    def create_combined_features(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        """Cria features combinadas tick + book"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE FEATURES COMBINADAS")
        print("=" * 80)
        
        features = pd.DataFrame(index=merged_data.index)
        
        # Arrays base
        price = merged_data['<price>'].values.astype('float32')
        qty = merged_data['<qty>'].values.astype('float32')
        
        # 1. FEATURES DE PREÇO E RETORNO
        print("\n-> Features de Preço e Retorno...")
        for period in [1, 5, 10, 20, 50]:
            returns = np.zeros(len(price), dtype='float32')
            returns[period:] = (price[period:] - price[:-period]) / price[:-period]
            features[f'returns_{period}'] = returns
        
        # 2. FEATURES DE VOLUME
        print("-> Features de Volume...")
        for window in [10, 20, 50]:
            vol_ma = pd.Series(qty).rolling(window).mean()
            features[f'volume_ma_{window}'] = vol_ma
            features[f'volume_ratio_{window}'] = qty / vol_ma.clip(lower=1)
        
        # 3. FEATURES DE MICROESTRUTURA (se temos book data)
        # Adaptar para estrutura real dos dados
        if 'price_y' in merged_data.columns:  # price do book
            print("-> Features de Microestrutura...")
            
            book_price = merged_data['price_y'].values
            
            # Diferença entre preço do trade e preço do book
            features['price_vs_book'] = (price - book_price) / book_price * 100
            
            # Trade acima/abaixo do book
            features['trade_above_book'] = (price > book_price).astype('float32')
            features['trade_below_book'] = (price < book_price).astype('float32')
            
        # Se temos side information
        if 'side' in merged_data.columns:
            # Preço vs side do book (0=buy, 1=sell)
            is_buy_side = (merged_data['side'] == 0).astype('float32')
            is_sell_side = (merged_data['side'] == 1).astype('float32')
            
            features['matches_buy_side'] = is_buy_side
            features['matches_sell_side'] = is_sell_side
        
        # 4. FEATURES DE POSIÇÃO E PROFUNDIDADE
        if 'position' in merged_data.columns:
            print("-> Features de Posição no Book...")
            
            # Posição no book (quanto mais próximo de 0, mais próximo do topo)
            features['book_position'] = merged_data['position'].values
            features['is_top_5'] = (merged_data['position'] <= 5).astype('float32')
            features['is_top_10'] = (merged_data['position'] <= 10).astype('float32')
            
        # Quantidade no book
        if 'quantity' in merged_data.columns:
            book_qty = merged_data['quantity'].values
            trade_qty = merged_data['<qty>'].values
            
            # Ratio entre quantidade do trade e quantidade no book
            features['qty_vs_book'] = trade_qty / (book_qty + 1)
            features['large_vs_book'] = (trade_qty > book_qty).astype('float32')
        
        # 5. FEATURES DE ORDER FLOW
        print("-> Features de Order Flow...")
        is_buyer = (merged_data['<trade_type>'] == 'AggressorBuyer').astype('float32')
        is_seller = (merged_data['<trade_type>'] == 'AggressorSeller').astype('float32')
        
        for window in [10, 20, 50]:
            buyer_flow = pd.Series(is_buyer * qty).rolling(window).sum()
            seller_flow = pd.Series(is_seller * qty).rolling(window).sum()
            total_flow = buyer_flow + seller_flow
            
            features[f'flow_imbalance_{window}'] = (
                (buyer_flow - seller_flow) / total_flow.clip(lower=1)
            )
        
        # 6. FEATURES TÉCNICAS
        print("-> Features Técnicas...")
        
        # Bollinger Bands
        price_series = pd.Series(price)
        for window in [20, 50]:
            ma = price_series.rolling(window).mean()
            std = price_series.rolling(window).std()
            features[f'bb_position_{window}'] = ((price_series - ma) / (2 * std)).clip(-3, 3)
        
        # 7. FEATURES TEMPORAIS
        print("-> Features Temporais...")
        features['hour'] = merged_data['timestamp'].dt.hour
        features['minute'] = merged_data['timestamp'].dt.minute
        
        # Cleanup
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(0)
        
        # Float32
        for col in features.columns:
            if features[col].dtype != 'float32':
                features[col] = features[col].astype('float32')
        
        print(f"\n[OK] Total features combinadas: {features.shape[1]}")
        
        return features
    
    def create_targets(self, data: pd.DataFrame, price_col: str = '<price>') -> np.ndarray:
        """Cria targets balanceados"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE TARGETS")
        print("=" * 80)
        
        prices = data[price_col].values
        
        # Calcular retornos futuros
        future_idx = np.arange(len(prices)) + self.horizon
        future_idx = np.clip(future_idx, 0, len(prices) - 1)
        
        future_prices = prices[future_idx]
        returns = (future_prices - prices) / prices
        
        # Invalidar últimos registros
        returns[-self.horizon:] = np.nan
        
        # Calcular thresholds
        valid_returns = returns[~np.isnan(returns)]
        std = valid_returns.std()
        
        buy_threshold = 0.5 * std
        sell_threshold = -0.5 * std
        
        print(f"\nHorizonte: {self.horizon} trades")
        print(f"Desvio padrão: {std:.5f} ({std*100:.3f}%)")
        print(f"Threshold BUY: {buy_threshold:.5f} ({buy_threshold*100:.3f}%)")
        print(f"Threshold SELL: {sell_threshold:.5f} ({sell_threshold*100:.3f}%)")
        
        # Criar targets
        targets = np.zeros(len(data), dtype='int8')
        targets[returns > buy_threshold] = 1
        targets[returns < sell_threshold] = -1
        targets[np.isnan(returns)] = -999  # Marcar como inválido
        
        # Stats
        valid_targets = targets[targets != -999]
        unique, counts = np.unique(valid_targets, return_counts=True)
        
        print("\nDistribuição:")
        for val, count in zip(unique, counts):
            label = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}[val]
            print(f"  {label}: {count:,} ({count/len(valid_targets)*100:.1f}%)")
        
        return targets
    
    def train_models(self, features: pd.DataFrame, targets: np.ndarray):
        """Treina modelos com book features"""
        
        print("\n" + "=" * 80)
        print("TREINAMENTO DE MODELOS")
        print("=" * 80)
        
        # Remover targets inválidos
        valid_mask = targets != -999
        X = features[valid_mask]
        y = targets[valid_mask]
        
        print(f"\nDados válidos: {len(X):,}")
        
        # Normalizar
        X_scaled = self.scaler.fit_transform(X)
        
        # Split
        split = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Class weights
        classes = np.array([-1, 0, 1])
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        # LightGBM
        print("\n" + "="*60)
        print("Treinando LightGBM com Book Features...")
        print("="*60)
        
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
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
            'verbose': -1
        }
        
        # Sample weights
        sample_weights = np.zeros(len(y_train))
        for class_val, weight in class_weight_dict.items():
            sample_weights[y_train == class_val] = weight
        
        lgb_train = lgb.Dataset(
            X_train, 
            label=y_train + 1,
            weight=sample_weights
        )
        lgb_val = lgb.Dataset(
            X_test, 
            label=y_test + 1,
            reference=lgb_train
        )
        
        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=300,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Predictions
        lgb_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
        lgb_pred_class = np.argmax(lgb_pred, axis=1) - 1
        
        # Metrics
        self._print_metrics("LightGBM+Book", y_test, lgb_pred_class)
        
        # Feature importance
        importance = pd.Series(
            lgb_model.feature_importance(importance_type='gain'),
            index=features.columns
        ).sort_values(ascending=False)
        
        print("\nTop 15 Features:")
        print(importance.head(15))
        
        # Salvar modelo
        self.models['lightgbm_book'] = lgb_model
        
        return importance
    
    def _print_metrics(self, name: str, y_true, y_pred):
        """Imprime métricas de avaliação"""
        
        # Overall accuracy
        accuracy = (y_pred == y_true).mean()
        
        # Trading accuracy (ignorando HOLD)
        trading_mask = y_true != 0
        if trading_mask.any():
            trading_accuracy = (y_pred[trading_mask] == y_true[trading_mask]).mean()
        else:
            trading_accuracy = 0
        
        # Por classe
        cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
        
        print(f"\nMétricas {name}:")
        print(f"  Overall Accuracy: {accuracy:.2%}")
        print(f"  Trading Accuracy: {trading_accuracy:.2%}")
        print(f"\nConfusion Matrix:")
        print(f"        SELL  HOLD   BUY")
        print(f"  SELL  {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
        print(f"  HOLD  {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
        print(f"  BUY   {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")
    
    def save_results(self, importance: pd.DataFrame):
        """Salva modelos e resultados"""
        
        output_dir = Path('models/book_ml')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*80)
        print("SALVANDO RESULTADOS")
        print("="*80)
        
        # Salvar modelo
        for name, model in self.models.items():
            if 'lightgbm' in name:
                model_file = output_dir / f'{name}_{timestamp}.txt'
                model.save_model(str(model_file))
        
        # Salvar scaler
        joblib.dump(self.scaler, output_dir / f'scaler_{timestamp}.pkl')
        
        # Feature importance
        importance.to_csv(output_dir / f'features_{timestamp}.csv')
        
        # Metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'horizon': self.horizon,
            'features_count': len(importance),
            'top_features': importance.head(10).to_dict()
        }
        
        with open(output_dir / f'metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[OK] Resultados salvos em: {output_dir}")

def main():
    """Pipeline principal"""
    
    print("PIPELINE DE TREINAMENTO COM BOOK DATA")
    print("Combinando tick data + book data para melhor accuracy\n")
    
    pipeline = BookMLPipeline()
    
    try:
        # 1. Carregar book data
        book_data = pipeline.load_book_data("20250806")
        
        # 2. Carregar tick data
        tick_data = pipeline.load_tick_data(n_records=500_000)
        
        # 3. Criar features do book
        book_features = pipeline.create_book_features(book_data)
        
        # 4. Merge tick e book data
        merged_data = pipeline.merge_tick_and_book_data(tick_data, book_data)
        
        # 5. Criar features combinadas
        features = pipeline.create_combined_features(merged_data)
        
        # 6. Criar targets
        targets = pipeline.create_targets(merged_data)
        
        # 7. Treinar modelos
        importance = pipeline.train_models(features, targets)
        
        # 8. Salvar resultados
        pipeline.save_results(importance)
        
        print("\n" + "="*80)
        print("[OK] PIPELINE CONCLUÍDO COM SUCESSO!")
        print("="*80)
        
        print("\nPróximos passos:")
        print("1. Avaliar melhoria na accuracy vs modelo tick-only")
        print("2. Ajustar features do book baseado na importância")
        print("3. Testar diferentes horizontes de previsão")
        print("4. Implementar ensemble tick+book")
        
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()