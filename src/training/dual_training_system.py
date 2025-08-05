"""
Sistema de Treinamento Dual - Tick-Only vs Book-Enhanced
Permite treinar modelos com diferentes fontes de dados
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from dataclasses import dataclass
from enum import Enum

from src.training.training_orchestrator import TrainingOrchestrator
from src.training.regime_analyzer import RegimeAnalyzer
from src.features.book_features import BookFeatureEngineer
from src.infrastructure.zmq_valkey_flow_setup import TradingInfrastructureWithFlow
from src.coordination.flow_aware_coordinator import FlowAwareCoordinator
import zmq
import orjson


class ModelType(Enum):
    """Tipos de modelos baseados em dados disponíveis"""
    TICK_ONLY = "tick_only"          # Apenas dados tick-a-tick (1 ano)
    BOOK_ENHANCED = "book_enhanced"   # Com dados de book (período menor)
    HYBRID = "hybrid"                 # Combinação de ambos


@dataclass
class TrainingConfig:
    """Configuração de treinamento"""
    model_type: ModelType
    data_path: str
    output_path: str
    feature_set: str  # 'basic', 'extended', 'book'
    lookback_days: int
    validation_split: float = 0.2
    test_split: float = 0.1
    
    
class DualTrainingSystem:
    """
    Sistema de treinamento dual que suporta:
    1. Modelos Tick-Only: Treinados com 1 ano de dados tick-a-tick
    2. Modelos Book-Enhanced: Treinados com dados de book (período menor mas mais rico)
    """
    
    def __init__(self, base_config: Dict[str, Any]):
        self.config = base_config
        self.logger = logging.getLogger('DualTrainingSystem')
        
        # Paths
        self.tick_data_path = Path(self.config.get('tick_data_path', 'data/historical'))
        self.book_data_path = Path(self.config.get('book_data_path', 'data/realtime/book'))
        self.models_path = Path(self.config.get('models_path', 'models'))
        
        # Criar diretórios
        self.models_path.mkdir(parents=True, exist_ok=True)
        (self.models_path / 'tick_only').mkdir(exist_ok=True)
        (self.models_path / 'book_enhanced').mkdir(exist_ok=True)
        
        # Componentes
        self.regime_analyzer = RegimeAnalyzer()
        self.orchestrator = TrainingOrchestrator(self.config)
        self.book_engineer = BookFeatureEngineer()
        
        # HMARL components (se disponível)
        self.hmarl_infrastructure = None
        self.flow_coordinator = None
        self._setup_hmarl_integration()
        
        # Feature sets
        self.feature_sets = {
            'basic': self._get_basic_features(),
            'extended': self._get_extended_features(),
            'book': self._get_book_features()
        }
        
        # HMARL flow features
        self.flow_features = [
            'flow_ofi_1m', 'flow_ofi_5m', 'flow_ofi_15m',
            'flow_volume_imbalance', 'flow_aggression_ratio',
            'flow_large_trade_ratio', 'tape_speed',
            'liquidity_score', 'footprint_pattern'
        ]
        
    def _setup_hmarl_integration(self):
        """Configura integração com HMARL se disponível"""
        try:
            # Configuração HMARL
            hmarl_config = {
                'symbol': 'WDOU25',
                'zmq': {
                    'tick_port': 5555,
                    'book_port': 5556,
                    'flow_port': 5557,
                    'footprint_port': 5558
                },
                'valkey': {
                    'host': 'localhost',
                    'port': 6379
                }
            }
            
            # Criar infraestrutura HMARL
            self.hmarl_infrastructure = TradingInfrastructureWithFlow(hmarl_config)
            self.hmarl_infrastructure.initialize()
            
            # Criar coordenador de fluxo
            self.flow_coordinator = FlowAwareCoordinator(hmarl_config['valkey'])
            
            self.logger.info("✅ HMARL integrado ao sistema de treinamento")
            
        except Exception as e:
            self.logger.warning(f"HMARL não disponível: {e}")
            self.logger.info("Continuando sem integração HMARL")
        
    def _get_basic_features(self) -> List[str]:
        """Features básicas disponíveis em dados tick-only"""
        return [
            # Preço
            'price', 'returns', 'log_returns',
            'price_change', 'price_change_pct',
            
            # Volume
            'volume', 'volume_imbalance',
            'buy_volume', 'sell_volume',
            'volume_ratio', 'trade_count',
            
            # Técnicos
            'sma_5', 'sma_10', 'sma_20',
            'ema_9', 'ema_21',
            'rsi_14', 'macd', 'macd_signal',
            'bollinger_upper', 'bollinger_lower',
            'atr_14', 'adx_14',
            
            # Microestrutura
            'vwap', 'spread_mean',
            'tick_direction', 'tick_count',
            'avg_trade_size'
        ]
        
    def _get_extended_features(self) -> List[str]:
        """Features estendidas incluindo derivadas"""
        basic = self._get_basic_features()
        extended = [
            # Momentum
            'momentum_5', 'momentum_10',
            'roc_5', 'roc_10',
            
            # Volatilidade
            'volatility_5', 'volatility_20',
            'volatility_ratio',
            
            # Padrões
            'higher_highs', 'lower_lows',
            'inside_bar', 'outside_bar',
            
            # Order Flow
            'cumulative_delta', 'delta_momentum',
            'buy_pressure', 'sell_pressure',
            
            # Regime
            'trend_strength', 'regime_confidence'
        ]
        return basic + extended
        
    def _get_book_features(self) -> List[str]:
        """Features específicas do book de ofertas"""
        return [
            # Book Depth
            'bid_ask_spread', 'mid_price',
            'book_imbalance', 'book_pressure',
            
            # Níveis
            'bid_depth_1', 'bid_depth_2', 'bid_depth_3',
            'ask_depth_1', 'ask_depth_2', 'ask_depth_3',
            
            # Volumes
            'bid_volume_1', 'bid_volume_2', 'bid_volume_3',
            'ask_volume_1', 'ask_volume_2', 'ask_volume_3',
            
            # Dinâmica
            'book_velocity', 'book_acceleration',
            'order_flow_imbalance',
            'micro_price', 'weighted_mid_price',
            
            # Métricas avançadas
            'kyle_lambda', 'amihud_illiquidity',
            'effective_spread', 'realized_spread',
            
            # Eventos
            'large_order_indicator',
            'iceberg_detection',
            'sweep_detection'
        ]
        
    def train_tick_only_models(self, 
                              symbols: List[str],
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Treina modelos usando apenas dados tick-a-tick (1 ano de histórico)
        
        Args:
            symbols: Lista de símbolos para treinar
            start_date: Data inicial (default: 1 ano atrás)
            end_date: Data final (default: hoje)
            
        Returns:
            Resultados do treinamento
        """
        self.logger.info("=== TREINAMENTO TICK-ONLY ===")
        
        # Configurar datas
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=365)
            
        self.logger.info(f"Período: {start_date.date()} até {end_date.date()}")
        
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"\nTreinando modelos tick-only para {symbol}...")
            
            # Carregar dados
            tick_data = self._load_tick_data(symbol, start_date, end_date)
            if tick_data.empty:
                self.logger.warning(f"Sem dados tick para {symbol}")
                continue
                
            # Preparar features
            feature_data = self._prepare_tick_features(tick_data, 'extended')
            
            # Treinar por regime
            regime_models = {}
            for regime in ['trend_up', 'trend_down', 'range']:
                self.logger.info(f"Treinando regime {regime}...")
                
                # Filtrar dados por regime
                regime_data = self._filter_by_regime(feature_data, regime)
                if len(regime_data) < 1000:
                    self.logger.warning(f"Dados insuficientes para regime {regime}: {len(regime_data)} amostras")
                    continue
                
                # Treinar ensemble
                model_path = self.models_path / 'tick_only' / f'{symbol}_{regime}'
                ensemble_results = self._train_ensemble(
                    regime_data,
                    model_path,
                    feature_set='extended'
                )
                
                regime_models[regime] = ensemble_results
                
            results[symbol] = {
                'type': 'tick_only',
                'period': f"{start_date.date()} to {end_date.date()}",
                'samples': len(tick_data),
                'regimes': regime_models
            }
            
        return results
        
    def train_book_enhanced_models(self,
                                  symbols: List[str],
                                  lookback_days: int = 30) -> Dict[str, Any]:
        """
        Treina modelos usando dados de book (período menor mas mais detalhado)
        
        Args:
            symbols: Lista de símbolos
            lookback_days: Dias de histórico de book
            
        Returns:
            Resultados do treinamento
        """
        self.logger.info("=== TREINAMENTO BOOK-ENHANCED ===")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        self.logger.info(f"Período: {start_date.date()} até {end_date.date()}")
        
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"\nTreinando modelos book-enhanced para {symbol}...")
            
            # Carregar dados de book
            book_data = self._load_book_data(symbol, start_date, end_date)
            if book_data.empty:
                self.logger.warning(f"Sem dados de book para {symbol}")
                continue
                
            # Preparar features incluindo book
            feature_data = self._prepare_book_features(book_data)
            
            # Treinar modelo especializado em book
            model_path = self.models_path / 'book_enhanced' / symbol
            
            # Modelo único para análise de microestrutura
            book_model = self._train_book_model(
                feature_data,
                model_path
            )
            
            results[symbol] = {
                'type': 'book_enhanced',
                'period': f"{start_date.date()} to {end_date.date()}",
                'samples': len(book_data),
                'model': book_model
            }
            
        return results
        
    def _load_tick_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Carrega dados tick históricos"""
        all_data = []
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            file_path = self.tick_data_path / symbol / date_str / 'trades.parquet'
            
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    all_data.append(df)
                except Exception as e:
                    self.logger.error(f"Erro ao ler {file_path}: {e}")
                    
            current_date += timedelta(days=1)
            
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
        
    def _load_book_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Carrega dados de book"""
        all_data = []
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            date_dir = self.book_data_path / date_str
            
            if date_dir.exists():
                # Carregar offer book e price book
                for book_type in ['offer_book_*.parquet', 'price_book_*.parquet']:
                    for file_path in date_dir.glob(book_type):
                        try:
                            df = pd.read_parquet(file_path)
                            df['book_type'] = book_type.split('_')[0]
                            all_data.append(df)
                        except Exception as e:
                            self.logger.error(f"Erro ao ler {file_path}: {e}")
                            
            current_date += timedelta(days=1)
            
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
        
    def _prepare_tick_features(self, data: pd.DataFrame, feature_set: str) -> pd.DataFrame:
        """Prepara features a partir de dados tick"""
        features = pd.DataFrame(index=data.index)
        
        # Features diretas do DataFrame
        direct_features = ['price', 'volume', 'buy_volume', 'sell_volume', 'trade_count']
        for feat in direct_features:
            if feat in data.columns:
                features[feat] = data[feat]
        
        # Calcular features derivadas
        if 'price' in data.columns:
            # Retornos
            features['returns'] = data['price'].pct_change()
            features['log_returns'] = np.log(data['price'] / data['price'].shift(1))
            features['price_change'] = data['price'].diff()
            features['price_change_pct'] = features['returns'] * 100
            
            # Médias móveis
            features['sma_5'] = data['price'].rolling(5).mean()
            features['sma_10'] = data['price'].rolling(10).mean()
            features['sma_20'] = data['price'].rolling(20).mean()
            features['ema_9'] = data['price'].ewm(span=9, adjust=False).mean()
            features['ema_21'] = data['price'].ewm(span=21, adjust=False).mean()
            
            # Bollinger Bands
            sma20 = features['sma_20']
            std20 = data['price'].rolling(20).std()
            features['bollinger_upper'] = sma20 + (2 * std20)
            features['bollinger_lower'] = sma20 - (2 * std20)
            
            # RSI
            features['rsi_14'] = self._calculate_rsi(data['price'], 14)
            
            # MACD
            ema12 = data['price'].ewm(span=12, adjust=False).mean()
            ema26 = data['price'].ewm(span=26, adjust=False).mean()
            features['macd'] = ema12 - ema26
            features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
            
            # ATR e ADX (simplificados)
            if all(col in data.columns for col in ['high', 'low', 'close']):
                features['atr_14'] = self._calculate_atr(data, 14)
                features['adx_14'] = self._calculate_adx(data, 14)
            else:
                # Aproximação usando apenas price
                features['atr_14'] = data['price'].rolling(14).std()
                features['adx_14'] = 25  # Valor neutro
        
        # Volume features
        if 'volume' in data.columns:
            features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            features['volume_imbalance'] = (data.get('buy_volume', 0) - data.get('sell_volume', 0)) / (data['volume'] + 1)
        
        # VWAP
        if 'price' in data.columns and 'volume' in data.columns:
            features['vwap'] = (data['price'] * data['volume']).cumsum() / data['volume'].cumsum()
        
        # Features estendidas se solicitado
        if feature_set == 'extended':
            # Momentum
            if 'price' in data.columns:
                features['momentum_5'] = data['price'] - data['price'].shift(5)
                features['momentum_10'] = data['price'] - data['price'].shift(10)
                features['roc_5'] = features['momentum_5'] / data['price'].shift(5)
                features['roc_10'] = features['momentum_10'] / data['price'].shift(10)
                
                # Volatilidade
                features['volatility_5'] = data['price'].rolling(5).std()
                features['volatility_20'] = data['price'].rolling(20).std()
                features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']
                
                # Padrões simples
                features['higher_highs'] = (data['price'] > data['price'].shift(1)).astype(int)
                features['lower_lows'] = (data['price'] < data['price'].shift(1)).astype(int)
            
            # Order flow
            if 'buy_volume' in data.columns and 'sell_volume' in data.columns:
                features['cumulative_delta'] = (data['buy_volume'] - data['sell_volume']).cumsum()
                features['delta_momentum'] = features['cumulative_delta'].diff(5)
                features['buy_pressure'] = data['buy_volume'] / (data['volume'] + 1)
                features['sell_pressure'] = data['sell_volume'] / (data['volume'] + 1)
        
        # Preencher NaN com métodos apropriados
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula ATR (Average True Range)"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula ADX simplificado"""
        # Implementação simplificada
        high = data['high']
        low = data['low']
        close = data['close']
        
        # +DM e -DM
        plus_dm = high.diff()
        minus_dm = low.shift() - low
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Suavizar
        plus_di = 100 * (plus_dm.rolling(period).mean() / self._calculate_atr(data, period))
        minus_di = 100 * (minus_dm.rolling(period).mean() / self._calculate_atr(data, period))
        
        # DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1)
        
        # ADX
        adx = dx.rolling(period).mean()
        
        return adx
        
    def _prepare_book_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepara features incluindo dados de book"""
        # Usar BookFeatureEngineer para calcular features
        book_features = self.book_engineer.calculate_all_features(data)
        
        # Se HMARL disponível, adicionar flow features
        if self.hmarl_infrastructure:
            flow_features = self._get_flow_enhanced_features(data)
            book_features = pd.concat([book_features, flow_features], axis=1)
            
        return book_features
    
    def _get_flow_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Obtém features de fluxo via HMARL"""
        flow_features = pd.DataFrame(index=data.index)
        
        if not self.hmarl_infrastructure:
            return flow_features
            
        try:
            # Publicar dados para análise de fluxo
            for idx, row in data.iterrows():
                tick_data = {
                    'symbol': row.get('symbol', 'WDOU25'),
                    'timestamp': row.get('timestamp'),
                    'price': row.get('price', 0),
                    'volume': row.get('volume', 0),
                    'trade_type': row.get('trade_type', 0)
                }
                
                # Publicar e obter análise de fluxo
                flow_analysis = self.hmarl_infrastructure.publish_tick_with_flow(tick_data)
                
                # Extrair features de fluxo
                if flow_analysis:
                    for feature in self.flow_features:
                        flow_features.loc[idx, feature] = flow_analysis.get(feature, 0)
                        
        except Exception as e:
            self.logger.error(f"Erro ao obter flow features: {e}")
            
        return flow_features
        
    def _filter_by_regime(self, data: pd.DataFrame, regime: str) -> pd.DataFrame:
        """Filtra dados por regime de mercado usando RegimeAnalyzer"""
        # Calcular regime para cada linha
        regime_data = []
        
        # Precisamos de pelo menos 50 candles para análise de regime
        window_size = 50
        
        for i in range(window_size, len(data)):
            # Janela de dados para análise
            window = data.iloc[i-window_size:i]
            
            # Preparar dados no formato esperado pelo RegimeAnalyzer
            candles_df = pd.DataFrame({
                'open': window.get('open', window.get('price', 0)),
                'high': window.get('high', window.get('price', 0)),
                'low': window.get('low', window.get('price', 0)),
                'close': window.get('close', window.get('price', 0)),
                'volume': window.get('volume', 0)
            })
            
            # Analisar regime
            regime_info = self.regime_analyzer.analyze_market(candles_df)
            
            # Adicionar resultado
            if regime_info['regime'] == regime:
                regime_data.append(i)
        
        # Filtrar dados
        if regime_data:
            filtered = data.iloc[regime_data]
            self.logger.info(f"Regime {regime}: {len(filtered)} amostras de {len(data)} total")
            return filtered
        else:
            self.logger.warning(f"Nenhuma amostra encontrada para regime {regime}")
            return pd.DataFrame()  # Retornar DataFrame vazio
        
    def _train_ensemble(self, data: pd.DataFrame, output_path: Path, feature_set: str) -> Dict[str, Any]:
        """Treina ensemble de modelos usando TrainingOrchestrator"""
        # Preparar dados para o orchestrator
        feature_cols = [col for col in data.columns if col not in ['timestamp', 'symbol', 'target']]
        X = data[feature_cols]
        
        # Criar target se não existir
        if 'target' in data.columns:
            y = data['target']
        else:
            # Gerar target baseado em retornos futuros
            if 'price' in data.columns:
                returns = data['price'].pct_change(5).shift(-5)  # Retorno de 5 períodos à frente
                y = pd.cut(returns, bins=[-np.inf, -0.001, 0.001, np.inf], labels=[0, 1, 2])
            else:
                self.logger.warning("Sem coluna price para gerar target")
                return {}
        
        # Alinhar dados
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 100:
            self.logger.warning(f"Dados insuficientes após limpeza: {len(X_clean)} amostras")
            return {}
        
        # Dividir dados
        split_idx = int(len(X_clean) * 0.8)
        X_train = X_clean.iloc[:split_idx]
        y_train = y_clean.iloc[:split_idx]
        X_val = X_clean.iloc[split_idx:]
        y_val = y_clean.iloc[split_idx:]
        
        # Usar componentes do orchestrator
        from src.training.ensemble_trainer import EnsembleTrainer
        from src.training.model_trainer import ModelTrainer
        
        # Criar trainers
        model_trainer = ModelTrainer(str(output_path))
        ensemble_trainer = EnsembleTrainer(model_trainer)
        
        # Treinar ensemble
        ensemble_result = ensemble_trainer.train_ensemble(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_types=['xgboost_fast', 'lightgbm_balanced', 'random_forest_stable'],
            parallel=False
        )
        
        # Salvar ensemble
        ensemble_path = ensemble_trainer.save_ensemble(
            ensemble_result,
            save_path=str(output_path / 'ensemble')
        )
        
        # Calcular métricas de trading
        from src.training.validation_engine import ValidationEngine
        validator = ValidationEngine()
        
        # Predições no conjunto de validação
        predictions = self._get_ensemble_predictions(ensemble_result, X_val)
        
        # Calcular métricas
        metrics = validator.calculate_validation_metrics(
            y_val,
            predictions,
            self._get_ensemble_probabilities(ensemble_result, X_val)
        )
        
        return {
            'models': list(ensemble_result['models'].keys()),
            'weights': ensemble_result['weights'],
            'performance': {
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall']
            },
            'ensemble_path': ensemble_path,
            'feature_importance': self._get_feature_importance(ensemble_result, feature_cols)
        }
    
    def _get_ensemble_predictions(self, ensemble_result: Dict, X: pd.DataFrame) -> np.ndarray:
        """Obtém predições do ensemble"""
        predictions = []
        weights = ensemble_result['weights']
        
        for name, model_data in ensemble_result['models'].items():
            model = model_data['model']
            weight = weights[name]
            
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                predictions.append(pred * weight)
        
        if not predictions:
            return np.zeros(len(X))
        
        # Combinar predições ponderadas
        ensemble_pred = np.sum(predictions, axis=0)
        
        # Converter para classes discretas
        return np.round(ensemble_pred).astype(int)
    
    def _get_ensemble_probabilities(self, ensemble_result: Dict, X: pd.DataFrame) -> np.ndarray:
        """Obtém probabilidades do ensemble"""
        probabilities = []
        weights = ensemble_result['weights']
        
        for name, model_data in ensemble_result['models'].items():
            model = model_data['model']
            weight = weights[name]
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probabilities.append(proba * weight)
        
        if not probabilities:
            # Retornar probabilidades uniformes se não há modelos
            n_samples = len(X)
            return np.ones((n_samples, 3)) / 3
        
        # Combinar probabilidades ponderadas
        ensemble_proba = np.sum(probabilities, axis=0)
        
        # Normalizar
        row_sums = ensemble_proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        ensemble_proba = ensemble_proba / row_sums
        
        return ensemble_proba
    
    def _get_feature_importance(self, ensemble_result: Dict, feature_names: List[str]) -> Dict[str, float]:
        """Extrai importância de features do ensemble"""
        importance_dict = {}
        
        for name, model_data in ensemble_result['models'].items():
            model = model_data['model']
            weight = ensemble_result['weights'][name]
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                for feat_name, imp in zip(feature_names, importances):
                    if feat_name not in importance_dict:
                        importance_dict[feat_name] = 0
                    importance_dict[feat_name] += imp * weight
        
        # Normalizar
        total_importance = sum(importance_dict.values())
        if total_importance > 0:
            importance_dict = {k: v/total_importance for k, v in importance_dict.items()}
        
        # Ordenar por importância
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
    def _train_book_model(self, data: pd.DataFrame, output_path: Path) -> Dict[str, Any]:
        """Treina modelo especializado em análise de book"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_absolute_error, r2_score
        import xgboost as xgb
        
        # Preparar dados
        feature_cols = [col for col in data.columns if col not in ['target', 'timestamp', 'symbol']]
        X = data[feature_cols]
        
        # Targets para microestrutura
        targets = {
            'spread_next': data['spread'].shift(-1),
            'imbalance_next': data['book_imbalance'].shift(-1),
            'price_move': data['price'].pct_change().shift(-1)
        }
        
        models = {}
        results = {}
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        for target_name, y in targets.items():
            if y.isna().all():
                continue
                
            # Remover NaNs
            mask = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 100:
                continue
                
            # Treinar ensemble
            ensemble = {
                'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
                'xgb': xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
            }
            
            target_models = {}
            target_scores = []
            
            for model_name, model in ensemble.items():
                scores = []
                
                for train_idx, val_idx in tscv.split(X_clean):
                    X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                    y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    
                    mae = mean_absolute_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    scores.append({'mae': mae, 'r2': r2})
                    
                avg_mae = np.mean([s['mae'] for s in scores])
                avg_r2 = np.mean([s['r2'] for s in scores])
                
                target_models[model_name] = model
                target_scores.append({'model': model_name, 'mae': avg_mae, 'r2': avg_r2})
                
            # Salvar melhor modelo
            best_model_info = min(target_scores, key=lambda x: x['mae'])
            best_model = target_models[best_model_info['model']]
            
            # Treinar no dataset completo
            best_model.fit(X_clean, y_clean)
            
            model_path = output_path / f'{target_name}_model.pkl'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model, model_path)
            
            models[target_name] = best_model_info
            results[target_name] = {
                'best_model': best_model_info['model'],
                'mae': best_model_info['mae'],
                'r2': best_model_info['r2']
            }
            
            # Salvar feature importance
            if hasattr(best_model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance.to_csv(output_path / f'{target_name}_importance.csv', index=False)
                
        return {
            'type': 'microstructure_analyzer',
            'features_used': len(feature_cols),
            'targets': list(models.keys()),
            'performance': results,
            'hmarl_enhanced': self.hmarl_infrastructure is not None
        }
        
    def create_hybrid_strategy(self, symbol: str) -> Dict[str, Any]:
        """
        Cria estratégia híbrida combinando modelos tick-only e book-enhanced
        com integração HMARL para análise de fluxo
        """
        self.logger.info(f"Criando estratégia híbrida para {symbol}")
        
        strategy = {
            'symbol': symbol,
            'components': {
                'regime_detection': 'tick_only',  # Usa histórico longo
                'signal_generation': 'tick_only',  # Sinais base
                'entry_timing': 'book_enhanced',   # Timing preciso
                'exit_optimization': 'book_enhanced',  # Saídas otimizadas
                'risk_management': 'hybrid',  # Combina ambos
                'flow_analysis': 'hmarl' if self.hmarl_infrastructure else None
            },
            'thresholds': {
                'regime_confidence': 0.7,  # Do modelo tick-only
                'book_imbalance': 0.6,     # Do modelo book-enhanced
                'flow_consensus': 0.65,    # Do HMARL
                'combined_signal': 0.65    # Threshold híbrido
            },
            'hmarl_integration': {
                'enabled': self.hmarl_infrastructure is not None,
                'flow_features': self.flow_features if self.hmarl_infrastructure else [],
                'coordinator': 'FlowAwareCoordinator' if self.flow_coordinator else None
            }
        }
        
        # Se HMARL disponível, configurar agentes
        if self.hmarl_infrastructure:
            strategy['hmarl_agents'] = [
                {'type': 'order_flow_specialist', 'weight': 0.3},
                {'type': 'liquidity_specialist', 'weight': 0.2},
                {'type': 'tape_reading', 'weight': 0.2},
                {'type': 'footprint_pattern', 'weight': 0.3}
            ]
            
        return strategy
        
    def evaluate_model_performance(self, model_type: ModelType, test_data: pd.DataFrame) -> Dict[str, float]:
        """Avalia performance de um tipo de modelo"""
        metrics = {}
        
        if model_type == ModelType.TICK_ONLY:
            # Métricas para modelos de tendência/regime
            metrics['trend_accuracy'] = 0.0
            metrics['regime_stability'] = 0.0
            metrics['signal_quality'] = 0.0
            
        elif model_type == ModelType.BOOK_ENHANCED:
            # Métricas para modelos de microestrutura
            metrics['spread_forecast_error'] = 0.0
            metrics['execution_improvement'] = 0.0
            metrics['slippage_reduction'] = 0.0
            
        return metrics
        
    def save_training_report(self, results: Dict[str, Any], filename: str = None):
        """Salva relatório detalhado do treinamento"""
        if not filename:
            filename = f"dual_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        report_path = self.models_path / filename
        
        import json
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        self.logger.info(f"Relatório salvo em: {report_path}")
        

def main():
    """Exemplo de uso do sistema de treinamento dual com HMARL"""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = {
        'tick_data_path': 'data/historical',
        'book_data_path': 'data/realtime/book',
        'models_path': 'models',
        'log_level': 'INFO'
    }
    
    # Criar sistema
    dual_trainer = DualTrainingSystem(config)
    
    # Símbolos para treinar
    symbols = ['WDOU25']
    
    # 1. Treinar modelos tick-only (usa 1 ano de dados)
    tick_results = dual_trainer.train_tick_only_models(symbols)
    
    # 2. Treinar modelos book-enhanced (usa 30 dias com book)
    book_results = dual_trainer.train_book_enhanced_models(symbols, lookback_days=30)
    
    # 3. Criar estratégia híbrida
    for symbol in symbols:
        hybrid_strategy = dual_trainer.create_hybrid_strategy(symbol)
        print(f"\nEstratégia híbrida para {symbol}:")
        print(hybrid_strategy)
    
    # 4. Salvar relatório
    all_results = {
        'tick_only': tick_results,
        'book_enhanced': book_results,
        'timestamp': datetime.now()
    }
    dual_trainer.save_training_report(all_results)


if __name__ == "__main__":
    main()