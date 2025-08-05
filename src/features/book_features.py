"""
Feature Engineering para Book de Ofertas
Extrai features avançadas de microestrutura de mercado
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta


class BookFeatureEngineer:
    """
    Engenharia de features específicas para book de ofertas
    Focado em microestrutura e dinâmica de liquidez
    """
    
    def __init__(self):
        self.logger = logging.getLogger('BookFeatureEngineer')
        
        # Configurações
        self.max_levels = 5  # Níveis de profundidade do book
        self.time_windows = [1, 5, 10, 30, 60]  # Segundos
        
    def calculate_all_features(self, book_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula todas as features de book
        
        Args:
            book_data: DataFrame com dados de book (offer ou price)
            
        Returns:
            DataFrame com features calculadas
        """
        features = pd.DataFrame(index=book_data.index)
        
        # 1. Features básicas de spread
        spread_features = self.calculate_spread_features(book_data)
        features = pd.concat([features, spread_features], axis=1)
        
        # 2. Features de desequilíbrio
        imbalance_features = self.calculate_imbalance_features(book_data)
        features = pd.concat([features, imbalance_features], axis=1)
        
        # 3. Features de profundidade
        depth_features = self.calculate_depth_features(book_data)
        features = pd.concat([features, depth_features], axis=1)
        
        # 4. Features de dinâmica temporal
        dynamic_features = self.calculate_dynamic_features(book_data)
        features = pd.concat([features, dynamic_features], axis=1)
        
        # 5. Features de microestrutura avançadas
        microstructure_features = self.calculate_microstructure_features(book_data)
        features = pd.concat([features, microstructure_features], axis=1)
        
        # 6. Features de detecção de padrões
        pattern_features = self.detect_book_patterns(book_data)
        features = pd.concat([features, pattern_features], axis=1)
        
        return features
        
    def calculate_spread_features(self, book_data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features relacionadas ao spread"""
        features = pd.DataFrame(index=book_data.index)
        
        # Assumindo colunas de book
        if 'best_bid' in book_data.columns and 'best_ask' in book_data.columns:
            # Spread absoluto
            features['spread'] = book_data['best_ask'] - book_data['best_bid']
            
            # Spread relativo (%)
            mid_price = (book_data['best_ask'] + book_data['best_bid']) / 2
            features['spread_pct'] = (features['spread'] / mid_price) * 100
            
            # Spread médio móvel
            for window in [10, 50, 100]:
                features[f'spread_ma_{window}'] = features['spread'].rolling(window).mean()
                
            # Volatilidade do spread
            features['spread_volatility'] = features['spread'].rolling(100).std()
            
            # Spread normalizado por volatilidade
            daily_volatility = book_data.get('daily_volatility', 1)
            features['normalized_spread'] = features['spread'] / daily_volatility
            
        return features
        
    def calculate_imbalance_features(self, book_data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de desequilíbrio do book"""
        features = pd.DataFrame(index=book_data.index)
        
        # Volume imbalance por nível
        for level in range(1, self.max_levels + 1):
            bid_vol_col = f'bid_volume_{level}'
            ask_vol_col = f'ask_volume_{level}'
            
            if bid_vol_col in book_data.columns and ask_vol_col in book_data.columns:
                bid_vol = book_data[bid_vol_col]
                ask_vol = book_data[ask_vol_col]
                
                # Imbalance ratio
                features[f'imbalance_ratio_l{level}'] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1)
                
                # Imbalance absoluto
                features[f'imbalance_abs_l{level}'] = bid_vol - ask_vol
                
        # Imbalance agregado (primeiros 3 níveis)
        if all(f'imbalance_ratio_l{i}' in features.columns for i in range(1, 4)):
            features['imbalance_weighted'] = (
                features['imbalance_ratio_l1'] * 0.5 +
                features['imbalance_ratio_l2'] * 0.3 +
                features['imbalance_ratio_l3'] * 0.2
            )
            
        # Pressão do book
        total_bid_volume = sum(book_data.get(f'bid_volume_{i}', 0) for i in range(1, 4))
        total_ask_volume = sum(book_data.get(f'ask_volume_{i}', 0) for i in range(1, 4))
        
        features['book_pressure'] = np.log(total_bid_volume + 1) - np.log(total_ask_volume + 1)
        
        # Skew do book
        features['book_skew'] = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume + 1)
        
        return features
        
    def calculate_depth_features(self, book_data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de profundidade do book"""
        features = pd.DataFrame(index=book_data.index)
        
        # Profundidade total cada lado
        bid_depth = 0
        ask_depth = 0
        
        for level in range(1, self.max_levels + 1):
            bid_vol = book_data.get(f'bid_volume_{level}', 0)
            ask_vol = book_data.get(f'ask_volume_{level}', 0)
            
            bid_depth += bid_vol
            ask_depth += ask_vol
            
            # Profundidade cumulativa
            features[f'bid_depth_cum_{level}'] = bid_depth
            features[f'ask_depth_cum_{level}'] = ask_depth
            
        features['total_depth'] = bid_depth + ask_depth
        features['depth_ratio'] = bid_depth / (ask_depth + 1)
        
        # Concentração de liquidez
        if 'bid_volume_1' in book_data.columns:
            features['bid_concentration'] = book_data['bid_volume_1'] / (bid_depth + 1)
            features['ask_concentration'] = book_data['ask_volume_1'] / (ask_depth + 1)
            
        # Shape do book (inclinação)
        bid_prices = [book_data.get(f'bid_price_{i}', 0) for i in range(1, 4)]
        ask_prices = [book_data.get(f'ask_price_{i}', 0) for i in range(1, 4)]
        
        if len(bid_prices) >= 3 and all(bid_prices):
            features['bid_slope'] = (bid_prices[0] - bid_prices[2]) / 2
            features['ask_slope'] = (ask_prices[2] - ask_prices[0]) / 2
            
        return features
        
    def calculate_dynamic_features(self, book_data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de dinâmica temporal do book"""
        features = pd.DataFrame(index=book_data.index)
        
        # Velocidade de mudança do book
        if 'timestamp' in book_data.columns:
            book_data['timestamp'] = pd.to_datetime(book_data['timestamp'])
            
            # Para cada janela temporal
            for window in self.time_windows:
                # Taxa de mudança do spread
                if 'spread' in features.columns:
                    features[f'spread_velocity_{window}s'] = (
                        features['spread'].diff() / window
                    ).rolling(f'{window}s').mean()
                    
                # Taxa de mudança do imbalance
                if 'book_pressure' in features.columns:
                    features[f'pressure_velocity_{window}s'] = (
                        features['book_pressure'].diff() / window
                    ).rolling(f'{window}s').mean()
                    
        # Estabilidade do book
        for col in ['best_bid', 'best_ask']:
            if col in book_data.columns:
                # Número de mudanças
                features[f'{col}_changes'] = (book_data[col].diff() != 0).rolling(100).sum()
                
                # Desvio padrão
                features[f'{col}_stability'] = book_data[col].rolling(100).std()
                
        return features
        
    def calculate_microstructure_features(self, book_data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features avançadas de microestrutura"""
        features = pd.DataFrame(index=book_data.index)
        
        # Micro-price (preço ponderado por volume)
        if all(col in book_data.columns for col in ['best_bid', 'best_ask', 'bid_volume_1', 'ask_volume_1']):
            bid_weight = book_data['ask_volume_1'] / (book_data['bid_volume_1'] + book_data['ask_volume_1'] + 1)
            ask_weight = book_data['bid_volume_1'] / (book_data['bid_volume_1'] + book_data['ask_volume_1'] + 1)
            
            features['micro_price'] = (
                book_data['best_bid'] * bid_weight + 
                book_data['best_ask'] * ask_weight
            )
            
            # Desvio do micro-price em relação ao mid
            mid_price = (book_data['best_bid'] + book_data['best_ask']) / 2
            features['micro_price_deviation'] = features['micro_price'] - mid_price
            
        # Kyle's Lambda (impacto de preço)
        if 'price' in book_data.columns and 'volume' in book_data.columns:
            # Aproximação simplificada
            price_changes = book_data['price'].diff()
            volume_signed = book_data['volume'] * np.sign(price_changes)
            
            features['kyle_lambda'] = (
                price_changes.rolling(100).std() / 
                volume_signed.rolling(100).std()
            )
            
        # Amihud Illiquidity
        if 'price' in book_data.columns and 'volume' in book_data.columns:
            returns = book_data['price'].pct_change()
            features['amihud_illiquidity'] = (
                returns.abs() / (book_data['volume'] + 1)
            ).rolling(100).mean()
            
        # Effective Spread (se tivermos trades)
        if 'trade_price' in book_data.columns:
            mid_price = (book_data['best_bid'] + book_data['best_ask']) / 2
            features['effective_spread'] = 2 * abs(book_data['trade_price'] - mid_price)
            
        # Realized Spread (requer dados futuros)
        # features['realized_spread'] = ... (implementar se necessário)
        
        return features
        
    def detect_book_patterns(self, book_data: pd.DataFrame) -> pd.DataFrame:
        """Detecta padrões específicos no book"""
        features = pd.DataFrame(index=book_data.index)
        
        # Detecção de ordens grandes (elephant orders)
        for level in range(1, 4):
            bid_vol = book_data.get(f'bid_volume_{level}', pd.Series(0))
            ask_vol = book_data.get(f'ask_volume_{level}', pd.Series(0))
            
            # Threshold: 3x o volume médio
            bid_threshold = bid_vol.rolling(1000).mean() * 3
            ask_threshold = ask_vol.rolling(1000).mean() * 3
            
            features[f'large_bid_l{level}'] = (bid_vol > bid_threshold).astype(int)
            features[f'large_ask_l{level}'] = (ask_vol > ask_threshold).astype(int)
            
        # Detecção de iceberg (ordens escondidas)
        if 'bid_volume_1' in book_data.columns:
            # Padrão: volume constante sendo reposto
            bid_vol_diff = book_data['bid_volume_1'].diff()
            ask_vol_diff = book_data['ask_volume_1'].diff()
            
            # Detecta reposição constante
            features['iceberg_bid_signal'] = (
                (bid_vol_diff.abs() < book_data['bid_volume_1'] * 0.1) & 
                (book_data['bid_volume_1'] > book_data['bid_volume_1'].rolling(100).mean())
            ).astype(int)
            
            features['iceberg_ask_signal'] = (
                (ask_vol_diff.abs() < book_data['ask_volume_1'] * 0.1) & 
                (book_data['ask_volume_1'] > book_data['ask_volume_1'].rolling(100).mean())
            ).astype(int)
            
        # Detecção de sweep (limpeza de níveis)
        if all(f'bid_volume_{i}' in book_data.columns for i in range(1, 4)):
            # Sweep: múltiplos níveis zerados simultaneamente
            bid_sweep = (
                (book_data['bid_volume_1'] == 0) & 
                (book_data['bid_volume_2'] == 0)
            ).astype(int)
            
            ask_sweep = (
                (book_data['ask_volume_1'] == 0) & 
                (book_data['ask_volume_2'] == 0)
            ).astype(int)
            
            features['bid_sweep_signal'] = bid_sweep
            features['ask_sweep_signal'] = ask_sweep
            
        # Padrão de acumulação/distribuição
        if 'book_pressure' in features.columns:
            pressure_ma = features['book_pressure'].rolling(100).mean()
            pressure_std = features['book_pressure'].rolling(100).std()
            
            features['accumulation_signal'] = (
                features['book_pressure'] > pressure_ma + pressure_std
            ).astype(int)
            
            features['distribution_signal'] = (
                features['book_pressure'] < pressure_ma - pressure_std
            ).astype(int)
            
        return features
        
    def calculate_order_flow_features(self, book_data: pd.DataFrame, trade_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calcula features de order flow combinando book e trades
        
        Args:
            book_data: Dados do book
            trade_data: Dados de trades (opcional)
            
        Returns:
            Features de order flow
        """
        features = pd.DataFrame(index=book_data.index)
        
        if trade_data is not None:
            # Merge book e trade data
            combined = pd.merge_asof(
                book_data.sort_values('timestamp'),
                trade_data.sort_values('timestamp'),
                on='timestamp',
                direction='backward'
            )
            
            # Order Flow Imbalance
            buy_volume = combined[combined['side'] == 'buy']['volume'].rolling('1min').sum()
            sell_volume = combined[combined['side'] == 'sell']['volume'].rolling('1min').sum()
            
            features['order_flow_imbalance'] = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1)
            
            # Volume at best prices
            at_bid = combined[combined['price'] == combined['best_bid']]['volume'].rolling('1min').sum()
            at_ask = combined[combined['price'] == combined['best_ask']]['volume'].rolling('1min').sum()
            
            features['volume_at_bid_pct'] = at_bid / (at_bid + at_ask + 1)
            
            # Aggressiveness ratio
            market_orders = combined[combined['order_type'] == 'market']['volume'].rolling('1min').sum()
            limit_orders = combined[combined['order_type'] == 'limit']['volume'].rolling('1min').sum()
            
            features['aggressiveness_ratio'] = market_orders / (limit_orders + 1)
            
        return features
        
    def aggregate_book_features(self, offer_book_features: pd.DataFrame, 
                               price_book_features: pd.DataFrame) -> pd.DataFrame:
        """
        Combina features de offer book e price book
        
        Args:
            offer_book_features: Features do offer book
            price_book_features: Features do price book agregado
            
        Returns:
            Features combinadas
        """
        # Combinar as features mais relevantes de cada tipo
        combined = pd.DataFrame(index=offer_book_features.index)
        
        # Do offer book: detalhes de microestrutura
        offer_cols = [
            'micro_price', 'kyle_lambda', 'amihud_illiquidity',
            'iceberg_bid_signal', 'iceberg_ask_signal'
        ]
        for col in offer_cols:
            if col in offer_book_features.columns:
                combined[f'offer_{col}'] = offer_book_features[col]
                
        # Do price book: visão agregada
        price_cols = [
            'spread', 'book_pressure', 'depth_ratio',
            'bid_slope', 'ask_slope'
        ]
        for col in price_cols:
            if col in price_book_features.columns:
                combined[f'price_{col}'] = price_book_features[col]
                
        # Features cruzadas
        if 'offer_micro_price' in combined.columns and 'price_spread' in combined.columns:
            # Divergência entre micro price e mid price
            mid_price = combined.get('price_mid', 0)
            combined['micro_mid_divergence'] = combined['offer_micro_price'] - mid_price
            
        return combined
        

def example_usage():
    """Exemplo de uso do BookFeatureEngineer"""
    # Criar engenheiro de features
    engineer = BookFeatureEngineer()
    
    # Dados simulados de book
    book_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01 09:00:00', periods=1000, freq='100ms'),
        'best_bid': np.random.normal(5000, 10, 1000),
        'best_ask': np.random.normal(5001, 10, 1000),
        'bid_volume_1': np.random.poisson(100, 1000),
        'ask_volume_1': np.random.poisson(100, 1000),
        'bid_volume_2': np.random.poisson(80, 1000),
        'ask_volume_2': np.random.poisson(80, 1000),
        'bid_volume_3': np.random.poisson(60, 1000),
        'ask_volume_3': np.random.poisson(60, 1000),
    })
    
    # Calcular todas as features
    features = engineer.calculate_all_features(book_data)
    
    print(f"Features calculadas: {features.shape[1]}")
    print(f"Amostras: {features.shape[0]}")
    print(f"\nPrimeiras features:")
    print(features.columns[:10].tolist())
    
    # Estatísticas básicas
    print(f"\nEstatísticas do spread:")
    print(features['spread'].describe())
    
    print(f"\nEstatísticas do book pressure:")
    print(features['book_pressure'].describe())


if __name__ == "__main__":
    example_usage()