"""
Pipeline Otimizado para Book Data
Versão simplificada e eficiente para evitar travamentos
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import joblib
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BookOptimizedPipeline:
    """Pipeline otimizado para treinar com book data"""
    
    def __init__(self):
        self.book_path = Path("data/realtime/book")
        self.csv_path = Path(r"C:\Users\marth\Downloads\WDO_FUT\WDOFUT_BMF_T.csv")
        self.models = {}
        self.scaler = StandardScaler()
        self.horizon = 100  # trades ahead
        
    def load_book_data_simple(self, date: str = "20250806"):
        """Carrega book data de forma simplificada"""
        
        print("=" * 80)
        print("CARREGAMENTO SIMPLIFICADO DE BOOK DATA")
        print("=" * 80)
        
        # Carregar arquivo de training
        training_file = self.book_path / date / "training_ready" / f"training_data_{date}.parquet"
        
        if not training_file.exists():
            print(f"[ERRO] Arquivo não encontrado: {training_file}")
            return None
            
        print(f"\nCarregando: {training_file}")
        book_data = pd.read_parquet(training_file)
        
        # Informações básicas
        print(f"[OK] {len(book_data):,} registros carregados")
        print(f"[OK] Período: {book_data['timestamp'].min()} até {book_data['timestamp'].max()}")
        
        if 'type' in book_data.columns:
            print(f"[OK] Tipos: {book_data['type'].value_counts().to_dict()}")
        
        return book_data
    
    def create_simple_book_features(self, book_data: pd.DataFrame) -> pd.DataFrame:
        """Cria features simplificadas do book"""
        
        print("\n" + "=" * 80)
        print("CRIAÇÃO DE FEATURES SIMPLIFICADAS")
        print("=" * 80)
        
        features_dict = {}
        
        # 1. FEATURES BÁSICAS
        print("\n-> Features básicas...")
        
        # Preço
        if 'price' in book_data.columns:
            features_dict['book_price'] = book_data['price'].values.astype('float32')
            
            # Mudanças de preço por ticker
            for ticker in book_data['ticker'].unique()[:2]:  # Limitar a 2 tickers
                mask = book_data['ticker'] == ticker
                ticker_data = book_data[mask]
                
                price_change = ticker_data['price'].diff().fillna(0)
                features_dict[f'price_change_{ticker}'] = pd.Series(
                    price_change, index=ticker_data.index
                ).reindex(book_data.index, fill_value=0).values.astype('float32')
        
        # Quantidade
        if 'quantity' in book_data.columns:
            features_dict['book_quantity'] = book_data['quantity'].values.astype('float32')
        
        # 2. FEATURES DE POSIÇÃO
        print("-> Features de posição...")
        
        if 'position' in book_data.columns:
            features_dict['book_position'] = book_data['position'].values.astype('float32')
            features_dict['is_top_5'] = (book_data['position'] <= 5).astype('float32')
        
        # 3. FEATURES DE SIDE
        print("-> Features de side...")
        
        if 'side' in book_data.columns:
            # Converter side para numérico (bid=0, ask=1)
            side_map = {'bid': 0, 'ask': 1, 'buy': 0, 'sell': 1}
            features_dict['book_side'] = book_data['side'].map(side_map).fillna(0.5).values.astype('float32')
            
            # Calcular imbalance simples por timestamp
            print("  Calculando imbalance...")
            imbalance_dict = {}
            
            # Limitar número de timestamps para evitar travamento
            unique_timestamps = book_data['timestamp'].unique()
            sample_size = min(len(unique_timestamps), 1000)
            sampled_timestamps = np.random.choice(unique_timestamps, sample_size, replace=False)
            
            # Agrupar por timestamp amostrado
            for ts in tqdm(sampled_timestamps, desc="  Timestamps"):
                group = book_data[book_data['timestamp'] == ts]
                if 'quantity' in group.columns:
                    # Usar mapeamento de side
                    buy_vol = group[group['side'].isin(['bid', 'buy'])]['quantity'].sum()
                    sell_vol = group[group['side'].isin(['ask', 'sell'])]['quantity'].sum()
                    total = buy_vol + sell_vol
                    
                    if total > 0:
                        imbalance = (buy_vol - sell_vol) / total
                    else:
                        imbalance = 0
                    
                    imbalance_dict[ts] = imbalance
            
            # Mapear de volta
            features_dict['book_imbalance'] = book_data['timestamp'].map(imbalance_dict).fillna(0).values.astype('float32')
        
        # 4. FEATURES TEMPORAIS
        print("-> Features temporais...")
        
        if 'hour' in book_data.columns:
            features_dict['hour'] = book_data['hour'].values.astype('float32')
        if 'minute' in book_data.columns:
            features_dict['minute'] = book_data['minute'].values.astype('float32')
        
        # Converter para DataFrame
        features = pd.DataFrame(features_dict)
        
        print(f"\n[OK] {len(features.columns)} features criadas")
        print(f"[OK] Memória: {features.memory_usage().sum() / 1024**2:.1f} MB")
        
        return features
    
    def load_tick_data_sample(self, start_date, end_date, n_records: int = 100000):
        """Carrega amostra de tick data no período do book"""
        
        print("\n" + "=" * 80)
        print("CARREGAMENTO DE TICK DATA SAMPLE")
        print("=" * 80)
        
        print(f"\nBuscando dados entre {start_date} e {end_date}")
        
        # Carregar chunks até encontrar o período desejado
        chunk_size = 100000
        found_data = []
        total_read = 0
        
        for chunk in pd.read_csv(self.csv_path, chunksize=chunk_size, nrows=n_records*10):
            # Criar timestamp
            chunk['timestamp'] = pd.to_datetime(
                chunk['<date>'].astype(str) + ' ' + chunk['<time>'].astype(str).str.zfill(6),
                format='%Y%m%d %H%M%S'
            )
            
            # Filtrar período
            mask = (chunk['timestamp'] >= start_date) & (chunk['timestamp'] <= end_date)
            relevant_data = chunk[mask]
            
            if len(relevant_data) > 0:
                found_data.append(relevant_data)
                print(f"  Encontrados {len(relevant_data):,} registros")
            
            total_read += len(chunk)
            
            if len(pd.concat(found_data)) >= n_records or total_read >= n_records * 10:
                break
        
        if not found_data:
            print("[AVISO] Nenhum dado encontrado no período")
            return None
        
        tick_data = pd.concat(found_data).head(n_records)
        tick_data = tick_data.sort_values('timestamp').reset_index(drop=True)
        
        print(f"\n[OK] {len(tick_data):,} registros tick carregados")
        print(f"[OK] Período real: {tick_data['timestamp'].min()} até {tick_data['timestamp'].max()}")
        
        return tick_data
    
    def merge_simple(self, tick_data: pd.DataFrame, book_features: pd.DataFrame, 
                     book_timestamps: pd.Series) -> pd.DataFrame:
        """Merge simplificado de tick e book data"""
        
        print("\n" + "=" * 80)
        print("MERGE SIMPLIFICADO")
        print("=" * 80)
        
        # Como os dados são de períodos diferentes, vamos simular um merge
        # Usar features médias do book para todos os ticks
        print("\n[INFO] Simulando merge devido a períodos diferentes")
        print("[INFO] Aplicando features médias do book aos ticks")
        
        # Calcular médias das features do book
        book_means = book_features.mean()
        
        # Criar DataFrame merged
        merged = tick_data.copy()
        
        # Adicionar features do book como valores constantes
        for col in book_features.columns:
            if col in book_means:
                merged[col] = book_means[col]
        
        print(f"\n[OK] Merge simulado: {len(merged):,} registros")
        print(f"[OK] Features do book adicionadas: {len(book_features.columns)}")
        
        return merged
    
    def create_final_features(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        """Cria features finais combinadas"""
        
        print("\n" + "=" * 80)
        print("FEATURES FINAIS")
        print("=" * 80)
        
        features_dict = {}
        
        # Dados base
        price = merged_data['<price>'].values.astype('float32')
        qty = merged_data['<qty>'].values.astype('float32')
        
        # 1. RETORNOS
        print("-> Retornos...")
        for period in [1, 5, 10, 20, 50]:
            returns = np.zeros(len(price), dtype='float32')
            if period < len(price):
                returns[period:] = (price[period:] - price[:-period]) / price[:-period]
            features_dict[f'returns_{period}'] = returns
        
        # 2. VOLUME
        print("-> Volume...")
        features_dict['volume'] = qty
        features_dict['volume_log'] = np.log1p(qty)
        
        # 3. BOOK FEATURES (se disponíveis)
        print("-> Book features...")
        
        book_cols = [col for col in merged_data.columns if col.startswith('book_') or col == 'is_top_5']
        for col in book_cols:
            if col in merged_data.columns:
                features_dict[col] = merged_data[col].fillna(0).values.astype('float32')
        
        # 4. PRICE VS BOOK
        if 'book_price' in merged_data.columns:
            book_price = merged_data['book_price'].values
            valid_book = ~np.isnan(book_price)
            
            price_vs_book = np.zeros(len(price), dtype='float32')
            price_vs_book[valid_book] = (price[valid_book] - book_price[valid_book]) / book_price[valid_book] * 100
            features_dict['price_vs_book'] = price_vs_book
        
        # 5. ORDER FLOW
        print("-> Order flow...")
        is_buyer = (merged_data['<trade_type>'] == 'AggressorBuyer').astype('float32')
        features_dict['is_buyer'] = is_buyer
        
        # 6. TEMPORAL
        features_dict['hour'] = merged_data['timestamp'].dt.hour.astype('float32')
        features_dict['minute'] = merged_data['timestamp'].dt.minute.astype('float32')
        
        # Criar DataFrame
        features = pd.DataFrame(features_dict)
        
        # Limpar
        features = features.fillna(0)
        
        print(f"\n[OK] {len(features.columns)} features finais")
        
        return features
    
    def train_lightgbm_simple(self, features: pd.DataFrame, targets: np.ndarray):
        """Treina modelo LightGBM simplificado"""
        
        print("\n" + "=" * 80)
        print("TREINAMENTO LIGHTGBM")
        print("=" * 80)
        
        # Remover inválidos
        valid_mask = targets != -999
        X = features[valid_mask]
        y = targets[valid_mask]
        
        print(f"Dados válidos: {len(X):,}")
        
        # Normalizar
        X_scaled = self.scaler.fit_transform(X)
        
        # Split
        split = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Weights
        classes = np.array([-1, 0, 1])
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        sample_weights = np.zeros(len(y_train))
        for class_val, weight in class_weight_dict.items():
            sample_weights[y_train == class_val] = weight
        
        # Parâmetros otimizados
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 5,
            'min_data_in_leaf': 100,
            'verbose': -1
        }
        
        # Train
        lgb_train = lgb.Dataset(X_train, label=y_train + 1, weight=sample_weights)
        lgb_val = lgb.Dataset(X_test, label=y_test + 1, reference=lgb_train)
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=200,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
        )
        
        # Evaluate
        pred = model.predict(X_test, num_iteration=model.best_iteration)
        pred_class = np.argmax(pred, axis=1) - 1
        
        # Metrics
        accuracy = (pred_class == y_test).mean()
        trading_mask = y_test != 0
        trading_accuracy = (pred_class[trading_mask] == y_test[trading_mask]).mean() if trading_mask.any() else 0
        
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
        
        self.models['lightgbm'] = model
        
        return model, importance

def main():
    """Pipeline principal otimizado"""
    
    print("PIPELINE OTIMIZADO - BOOK DATA")
    print("Versão simplificada para evitar travamentos\n")
    
    pipeline = BookOptimizedPipeline()
    
    try:
        # 1. Carregar book data
        book_data = pipeline.load_book_data_simple("20250806")
        if book_data is None:
            return
        
        # 2. Criar features do book
        book_features = pipeline.create_simple_book_features(book_data)
        
        # 3. Carregar tick data (períodos diferentes)
        print("\n[INFO] Book data é de 2025, tick data é de 2024")
        print("[INFO] Carregando amostra de tick data para simulação...")
        
        # Carregar amostra de tick data
        tick_data = pd.read_csv(pipeline.csv_path, nrows=100000)
        tick_data['timestamp'] = pd.to_datetime(
            tick_data['<date>'].astype(str) + ' ' + tick_data['<time>'].astype(str).str.zfill(6),
            format='%Y%m%d %H%M%S'
        )
        
        print(f"\n[OK] Tick data carregado: {len(tick_data):,} registros")
        print(f"[OK] Período tick: {tick_data['timestamp'].min()} até {tick_data['timestamp'].max()}")
        
        # 4. Merge simplificado
        merged = pipeline.merge_simple(tick_data, book_features, book_data['timestamp'])
        
        # 5. Features finais
        features = pipeline.create_final_features(merged)
        
        # 6. Criar targets
        prices = merged['<price>'].values
        horizon = 100
        
        future_idx = np.minimum(np.arange(len(prices)) + horizon, len(prices) - 1)
        returns = (prices[future_idx] - prices) / prices
        returns[-horizon:] = np.nan
        
        valid_returns = returns[~np.isnan(returns)]
        threshold = 0.5 * valid_returns.std()
        
        targets = np.zeros(len(merged), dtype='int16')  # Mudado para int16
        targets[returns > threshold] = 1
        targets[returns < -threshold] = -1
        targets[np.isnan(returns)] = -999
        
        print(f"\nDistribuição de targets:")
        unique, counts = np.unique(targets[targets != -999], return_counts=True)
        for val, count in zip(unique, counts):
            print(f"  {val}: {count:,}")
        
        # 7. Treinar
        model, importance = pipeline.train_lightgbm_simple(features, targets)
        
        # 8. Salvar
        output_dir = Path('models/book_optimized')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model.save_model(str(output_dir / f'lightgbm_book_{timestamp}.txt'))
        joblib.dump(pipeline.scaler, output_dir / f'scaler_{timestamp}.pkl')
        importance.to_csv(output_dir / f'features_{timestamp}.csv')
        
        print(f"\n[OK] Modelos salvos em: {output_dir}")
        print("\n[OK] PIPELINE CONCLUÍDO!")
        
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()