"""
Treinamento HMARL Simplificado com Dados do Book Collector
Usa apenas módulos core sem dependências complexas
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


def extract_book_features(data_path: str) -> pd.DataFrame:
    """Extrai features dos dados do book collector"""
    
    print("=" * 70)
    print("EXTRAÇÃO DE FEATURES DO BOOK COLLECTOR")
    print("=" * 70)
    
    # Carregar dados
    df = pd.read_parquet(data_path)
    print(f"\nDados carregados: {len(df):,} registros")
    
    # Filtrar tiny_book (melhores ofertas)
    tiny_book = df[df['type'] == 'tiny_book'].copy()
    print(f"Registros tiny_book: {len(tiny_book):,}")
    
    if tiny_book.empty:
        print("[ERRO] Sem dados de tiny_book")
        return pd.DataFrame()
    
    # Preparar dados
    tiny_book['timestamp'] = pd.to_datetime(tiny_book['timestamp'])
    tiny_book = tiny_book.sort_values('timestamp')
    
    # Separar bid e ask
    bid_data = tiny_book[tiny_book['side'] == 'bid'][['timestamp', 'price', 'quantity']].copy()
    ask_data = tiny_book[tiny_book['side'] == 'ask'][['timestamp', 'price', 'quantity']].copy()
    
    # Renomear colunas
    bid_data.columns = ['timestamp', 'bid_price', 'bid_qty']
    ask_data.columns = ['timestamp', 'ask_price', 'ask_qty']
    
    # Resample para 1 segundo (agregar dados)
    bid_resampled = bid_data.set_index('timestamp').resample('1s').agg({
        'bid_price': 'last',
        'bid_qty': 'mean'
    })
    
    ask_resampled = ask_data.set_index('timestamp').resample('1s').agg({
        'ask_price': 'last',
        'ask_qty': 'mean'
    })
    
    # Combinar
    book_data = pd.concat([bid_resampled, ask_resampled], axis=1).dropna()
    print(f"\nDados após resample (1s): {len(book_data):,} registros")
    
    # Calcular features de microestrutura
    features = pd.DataFrame(index=book_data.index)
    
    # 1. Spread
    features['spread'] = book_data['ask_price'] - book_data['bid_price']
    features['spread_pct'] = (features['spread'] / book_data['bid_price']) * 100
    
    # 2. Mid price
    features['mid_price'] = (book_data['ask_price'] + book_data['bid_price']) / 2
    
    # 3. Imbalance de quantidade
    features['qty_imbalance'] = (book_data['bid_qty'] - book_data['ask_qty']) / \
                                (book_data['bid_qty'] + book_data['ask_qty'])
    
    # 4. Pressão de preço
    features['price_pressure'] = book_data['bid_qty'] / (book_data['bid_qty'] + book_data['ask_qty'])
    
    # 5. Features temporais
    features['returns_1s'] = features['mid_price'].pct_change()
    features['returns_5s'] = features['mid_price'].pct_change(5)
    features['returns_10s'] = features['mid_price'].pct_change(10)
    
    # 6. Volatilidade
    features['volatility_10s'] = features['returns_1s'].rolling(10).std()
    features['volatility_30s'] = features['returns_1s'].rolling(30).std()
    
    # 7. Médias móveis do spread
    features['spread_ma_10'] = features['spread'].rolling(10).mean()
    features['spread_ma_30'] = features['spread'].rolling(30).mean()
    
    # 8. Volume médio
    features['avg_bid_qty'] = book_data['bid_qty'].rolling(10).mean()
    features['avg_ask_qty'] = book_data['ask_qty'].rolling(10).mean()
    
    # Target: Direção do movimento de preço
    features['future_return'] = features['mid_price'].pct_change().shift(-5)  # 5 segundos no futuro
    features['direction'] = np.where(features['future_return'] > 0.0001, 1,  # Alta
                                   np.where(features['future_return'] < -0.0001, -1,  # Baixa
                                          0))  # Neutro
    
    # Remover NaN
    features = features.dropna()
    
    print(f"\nFeatures calculadas: {features.shape}")
    print(f"Distribuição do target:")
    print(features['direction'].value_counts())
    
    return features


def train_hmarl_model(features_df: pd.DataFrame):
    """Treina modelo HMARL com features extraídas"""
    
    print("\n" + "=" * 70)
    print("TREINAMENTO DO MODELO HMARL")
    print("=" * 70)
    
    # Separar features e target
    feature_cols = [col for col in features_df.columns 
                   if col not in ['direction', 'future_return']]
    
    X = features_df[feature_cols]
    y = features_df['direction']
    
    print(f"\nFeatures para treinamento: {len(feature_cols)}")
    print(feature_cols)
    
    # Split temporal (80/20)
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"\nTamanho do conjunto de treino: {len(X_train)}")
    print(f"Tamanho do conjunto de teste: {len(X_test)}")
    
    # Treinar Random Forest (robusto e interpretável)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTreinando Random Forest...")
    model.fit(X_train, y_train)
    
    # Avaliar
    y_pred = model.predict(X_test)
    
    print("\n" + "=" * 50)
    print("RESULTADOS DO MODELO")
    print("=" * 50)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Baixa', 'Neutro', 'Alta']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features Mais Importantes:")
    print(feature_importance.head(10))
    
    # Salvar modelo
    output_dir = Path('models/hmarl/book_based')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = output_dir / f'rf_direction_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    joblib.dump(model, model_file)
    
    # Salvar metadados
    metadata = {
        'model_type': 'RandomForest',
        'target': 'price_direction',
        'features': feature_cols,
        'feature_importance': feature_importance.to_dict('records'),
        'training_date': datetime.now().isoformat(),
        'training_records': len(X_train),
        'test_records': len(X_test),
        'data_source': 'book_collector_continuous'
    }
    
    metadata_file = output_dir / f'model_metadata_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[OK] Modelo salvo em: {model_file}")
    print(f"[OK] Metadados salvos em: {metadata_file}")
    
    return model, metadata


def analyze_offer_book_patterns(data_path: str):
    """Analisa padrões no offer book completo"""
    
    print("\n" + "=" * 70)
    print("ANÁLISE DE PADRÕES DO OFFER BOOK")
    print("=" * 70)
    
    df = pd.read_parquet(data_path)
    offer_book = df[df['type'] == 'offer_book'].copy()
    
    if offer_book.empty:
        print("[AVISO] Sem dados de offer_book")
        return
    
    print(f"\nRegistros offer_book: {len(offer_book):,}")
    
    # Analisar ações
    if 'action' in offer_book.columns:
        print("\nDistribuição de ações:")
        action_map = {0: 'New', 1: 'Edit', 2: 'Delete', 3: 'DeleteFrom', 4: 'DeleteThru'}
        for action, count in offer_book['action'].value_counts().items():
            action_name = action_map.get(action, f'Unknown({action})')
            print(f"  {action_name}: {count:,}")
    
    # Analisar profundidade
    if 'position' in offer_book.columns:
        print(f"\nProfundidade máxima: {offer_book['position'].max():.0f}")
        print(f"Profundidade média: {offer_book['position'].mean():.1f}")
    
    # Analisar distribuição de preços
    if 'price' in offer_book.columns:
        valid_prices = offer_book[offer_book['price'] > 0]['price']
        if not valid_prices.empty:
            print(f"\nDistribuição de preços:")
            print(f"  Min: {valid_prices.min():.2f}")
            print(f"  Max: {valid_prices.max():.2f}")
            print(f"  Média: {valid_prices.mean():.2f}")
            print(f"  Desvio: {valid_prices.std():.2f}")


def main():
    """Pipeline principal"""
    
    # Path dos dados
    data_path = 'data/realtime/book/20250805/training/consolidated_training_20250805.parquet'
    
    # 1. Extrair features
    features = extract_book_features(data_path)
    
    if features.empty:
        print("[ERRO] Falha na extração de features")
        return
    
    # 2. Treinar modelo
    model, metadata = train_hmarl_model(features)
    
    # 3. Análise adicional do offer book
    analyze_offer_book_patterns(data_path)
    
    print("\n" + "=" * 70)
    print("TREINAMENTO HMARL CONCLUÍDO COM SUCESSO!")
    print("=" * 70)
    print("\nPróximos passos:")
    print("1. Validar modelo com dados mais recentes")
    print("2. Implementar estratégia de trading baseada nas predições")
    print("3. Integrar com sistema de execução em tempo real")
    print("4. Monitorar performance em paper trading")


if __name__ == "__main__":
    main()