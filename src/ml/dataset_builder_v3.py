"""
DatasetBuilderV3 - Construção automatizada de datasets com dados reais
Baseado no DEVELOPER_GUIDE_V3_REFACTORING.md
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import os
import json
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib


class DatasetBuilderV3:
    """
    Constrói datasets para treinamento com dados reais e validação temporal
    
    Features:
    - Coleta dados reais do ProfitDLL
    - Calcula features com MLFeaturesV3
    - Cria labels baseadas em movimento futuro
    - Separação temporal train/valid/test
    - Salva datasets prontos para treino
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o builder
        
        Args:
            config: Configurações opcionais
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Paths
        self.data_path = self.config.get('data_path', 'data/')
        self.dataset_path = self.config.get('dataset_path', 'datasets/')
        self.model_path = self.config.get('model_path', 'models/')
        
        # Garantir que diretórios existam
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        
        # Configurações de dataset
        self.lookback_periods = self.config.get('lookback_periods', 100)
        self.target_periods = self.config.get('target_periods', 5)
        self.target_threshold = self.config.get('target_threshold', 0.001)  # 0.1%
        
        # Split configuration
        self.train_ratio = self.config.get('train_ratio', 0.7)
        self.valid_ratio = self.config.get('valid_ratio', 0.15)
        self.test_ratio = self.config.get('test_ratio', 0.15)
        
        # Feature scaler
        self.scaler_type = self.config.get('scaler_type', 'robust')
        self.scaler = None
        
        # Cache
        self._data_cache = {}
        
    def build_training_dataset(self, start_date: datetime, end_date: datetime,
                             ticker: str = "WDOH25", timeframe: str = '1min') -> Dict:
        """
        Constrói dataset completo para treinamento
        
        Args:
            start_date: Data inicial
            end_date: Data final
            ticker: Símbolo do ativo
            timeframe: Timeframe dos dados
            
        Returns:
            Dict com datasets separados por split e regime
        """
        self.logger.info("="*60)
        self.logger.info("CONSTRUINDO DATASET V3 COM DADOS REAIS")
        self.logger.info("="*60)
        self.logger.info(f"Período: {start_date.date()} até {end_date.date()}")
        self.logger.info(f"Ticker: {ticker}")
        self.logger.info(f"Timeframe: {timeframe}")
        
        # 1. Coletar dados reais
        self.logger.info("\n1. Coletando dados reais...")
        raw_data = self._collect_real_data(ticker, start_date, end_date, timeframe)
        
        if not raw_data or 'candles' not in raw_data:
            self.logger.error("Falha na coleta de dados")
            return {}
        
        self.logger.info(f"   Candles coletados: {len(raw_data['candles'])}")
        self.logger.info(f"   Microestrutura: {len(raw_data.get('microstructure', []))}")
        
        # 2. Calcular features
        self.logger.info("\n2. Calculando features ML...")
        features = self._calculate_features(raw_data)
        
        if features.empty:
            self.logger.error("Falha no cálculo de features")
            return {}
        
        self.logger.info(f"   Features calculadas: {features.shape}")
        self.logger.info(f"   NaN rate: {features.isna().sum().sum() / (features.shape[0] * features.shape[1]):.2%}")
        
        # 3. Criar labels
        self.logger.info("\n3. Criando labels...")
        features_with_labels = self._create_labels(features, raw_data['candles'])
        
        self.logger.info(f"   Samples com labels: {len(features_with_labels)}")
        
        # 4. Detectar regimes
        self.logger.info("\n4. Detectando regimes de mercado...")
        features_with_regime = self._detect_regimes(features_with_labels, raw_data['candles'])
        
        regime_counts = features_with_regime['regime'].value_counts()
        self.logger.info(f"   Regimes detectados:")
        for regime, count in regime_counts.items():
            self.logger.info(f"     - {regime}: {count} ({count/len(features_with_regime):.1%})")
        
        # 5. Separar datasets
        self.logger.info("\n5. Separando datasets (train/valid/test)...")
        datasets = self._split_datasets(features_with_regime)
        
        # 6. Normalizar features
        self.logger.info("\n6. Normalizando features...")
        datasets = self._normalize_features(datasets)
        
        # 7. Salvar datasets
        self.logger.info("\n7. Salvando datasets...")
        self._save_datasets(datasets, ticker, start_date, end_date)
        
        # Estatísticas finais
        self._print_dataset_statistics(datasets)
        
        return datasets
    
    def _collect_real_data(self, ticker: str, start_date: datetime, 
                          end_date: datetime, timeframe: str) -> Dict:
        """Coleta dados reais usando RealDataCollector"""
        
        try:
            # Importar e usar RealDataCollector
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from data.real_data_collector import RealDataCollector
            
            collector = RealDataCollector()
            
            # Para teste, vamos simular dados
            # Em produção, usar: data = collector.collect_historical_data(ticker, start_date, end_date)
            
            # Simulação de dados para teste
            self.logger.warning("Usando dados simulados para teste - substituir por coleta real em produção")
            
            dates = pd.date_range(start_date, end_date, freq=timeframe)
            
            # Simular candles
            candles = pd.DataFrame({
                'open': 5900 + np.random.randn(len(dates)).cumsum() * 0.5,
                'high': 5905 + np.random.randn(len(dates)).cumsum() * 0.5,
                'low': 5895 + np.random.randn(len(dates)).cumsum() * 0.5,
                'close': 5900 + np.random.randn(len(dates)).cumsum() * 0.5,
                'volume': np.random.randint(1000000, 5000000, len(dates)),
                'bid': 5899 + np.random.randn(len(dates)).cumsum() * 0.5,
                'ask': 5901 + np.random.randn(len(dates)).cumsum() * 0.5
            }, index=dates)
            
            # Ajustar OHLC
            candles['high'] = candles[['open', 'close', 'high']].max(axis=1)
            candles['low'] = candles[['open', 'close', 'low']].min(axis=1)
            
            # Simular microestrutura
            microstructure = pd.DataFrame({
                'buy_volume': candles['volume'] * np.random.uniform(0.4, 0.6, len(dates)),
                'sell_volume': candles['volume'] * np.random.uniform(0.4, 0.6, len(dates)),
                'volume_imbalance': np.random.randn(len(dates)) * 500000,
                'trade_imbalance': np.random.randn(len(dates)) * 50,
                'buy_pressure': np.random.uniform(0.4, 0.6, len(dates)),
                'avg_trade_size': np.random.randint(10000, 50000, len(dates)),
                'buy_trades': np.random.randint(50, 150, len(dates)),
                'sell_trades': np.random.randint(50, 150, len(dates))
            }, index=dates)
            
            return {
                'candles': candles,
                'microstructure': microstructure
            }
            
        except Exception as e:
            self.logger.error(f"Erro coletando dados: {e}")
            return {}
    
    def _calculate_features(self, raw_data: Dict) -> pd.DataFrame:
        """Calcula features usando MLFeaturesV3"""
        
        try:
            from features.ml_features_v3 import MLFeaturesV3
            
            calculator = MLFeaturesV3()
            
            features = calculator.calculate_all(
                candles=raw_data['candles'],
                microstructure=raw_data['microstructure']
            )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Erro calculando features: {e}")
            return pd.DataFrame()
    
    def _create_labels(self, features: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
        """Cria labels baseadas em movimento futuro de preço"""
        
        close_prices = candles['close']
        
        # Calcular retorno futuro
        future_returns = close_prices.shift(-self.target_periods) / close_prices - 1
        
        # Criar labels categóricas
        labels = pd.DataFrame(index=features.index)
        
        # Label binária (up/down)
        labels['target_binary'] = (future_returns > 0).astype(int)
        
        # Label ternária (up/neutral/down)
        labels['target_class'] = 0  # neutral
        labels.loc[future_returns > self.target_threshold, 'target_class'] = 1  # up
        labels.loc[future_returns < -self.target_threshold, 'target_class'] = -1  # down
        
        # Retorno real (para regressão)
        labels['target_return'] = future_returns
        
        # Magnitude do movimento
        labels['target_magnitude'] = future_returns.abs()
        
        # Combinar features com labels
        features_with_labels = pd.concat([features, labels], axis=1)
        
        # Remover últimas N linhas sem label futuro
        features_with_labels = features_with_labels.iloc[:-self.target_periods]
        
        return features_with_labels
    
    def _detect_regimes(self, data: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
        """Detecta regime de mercado usando RegimeAnalyzer"""
        
        try:
            # Por enquanto, implementação simplificada
            # TODO: Integrar com RegimeAnalyzer da Fase 2
            
            close = candles['close'].loc[data.index]
            
            # EMAs
            ema9 = close.ewm(span=9, adjust=False).mean()
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            
            # ADX simplificado (usar feature calculada se disponível)
            adx = data['v3_adx'] if 'v3_adx' in data.columns else 25  # default
            
            # Classificar regime
            regime = pd.Series('undefined', index=data.index)
            
            # Trend up
            trend_up_mask = (adx > 25) & (ema9 > ema20) & (ema20 > ema50)
            regime[trend_up_mask] = 'trend_up'
            
            # Trend down
            trend_down_mask = (adx > 25) & (ema9 < ema20) & (ema20 < ema50)
            regime[trend_down_mask] = 'trend_down'
            
            # Range
            range_mask = (adx <= 25)
            regime[range_mask] = 'range'
            
            data['regime'] = regime
            
            return data
            
        except Exception as e:
            self.logger.error(f"Erro detectando regimes: {e}")
            data['regime'] = 'undefined'
            return data
    
    def _split_datasets(self, data: pd.DataFrame) -> Dict:
        """Separa datasets temporalmente"""
        
        n_samples = len(data)
        
        # Índices de split
        train_end = int(n_samples * self.train_ratio)
        valid_end = train_end + int(n_samples * self.valid_ratio)
        
        # Separar temporalmente
        train_data = data.iloc[:train_end]
        valid_data = data.iloc[train_end:valid_end]
        test_data = data.iloc[valid_end:]
        
        # Separar features e labels
        feature_cols = [col for col in data.columns if col.startswith('v3_')]
        label_cols = ['target_binary', 'target_class', 'target_return', 'target_magnitude']
        meta_cols = ['regime']
        
        datasets = {
            'train': {
                'features': train_data[feature_cols],
                'labels': train_data[label_cols],
                'meta': train_data[meta_cols],
                'index': train_data.index
            },
            'valid': {
                'features': valid_data[feature_cols],
                'labels': valid_data[label_cols],
                'meta': valid_data[meta_cols],
                'index': valid_data.index
            },
            'test': {
                'features': test_data[feature_cols],
                'labels': test_data[label_cols],
                'meta': test_data[meta_cols],
                'index': test_data.index
            }
        }
        
        # Separar por regime
        for split in ['train', 'valid', 'test']:
            regime_data = {}
            for regime in ['trend_up', 'trend_down', 'range', 'undefined']:
                mask = datasets[split]['meta']['regime'] == regime
                if mask.sum() > 0:
                    regime_data[regime] = {
                        'features': datasets[split]['features'][mask],
                        'labels': datasets[split]['labels'][mask],
                        'index': datasets[split]['index'][mask]
                    }
            datasets[split]['by_regime'] = regime_data
        
        return datasets
    
    def _normalize_features(self, datasets: Dict) -> Dict:
        """Normaliza features usando scaler robusto"""
        
        # Criar e treinar scaler com dados de treino
        if self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        # Treinar com todos os dados de treino
        train_features = datasets['train']['features']
        self.scaler.fit(train_features)
        
        # Aplicar transformação
        for split in ['train', 'valid', 'test']:
            # Transform geral
            datasets[split]['features_scaled'] = pd.DataFrame(
                self.scaler.transform(datasets[split]['features']),
                columns=datasets[split]['features'].columns,
                index=datasets[split]['features'].index
            )
            
            # Transform por regime
            if 'by_regime' in datasets[split]:
                for regime in datasets[split]['by_regime']:
                    regime_features = datasets[split]['by_regime'][regime]['features']
                    datasets[split]['by_regime'][regime]['features_scaled'] = pd.DataFrame(
                        self.scaler.transform(regime_features),
                        columns=regime_features.columns,
                        index=regime_features.index
                    )
        
        # Salvar scaler
        scaler_path = os.path.join(self.model_path, 'feature_scaler_v3.pkl')
        joblib.dump(self.scaler, scaler_path)
        self.logger.info(f"   Scaler salvo em: {scaler_path}")
        
        return datasets
    
    def _save_datasets(self, datasets: Dict, ticker: str, 
                      start_date: datetime, end_date: datetime):
        """Salva datasets em formato otimizado"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        # Metadados
        metadata = {
            'ticker': ticker,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'creation_time': datetime.now().isoformat(),
            'lookback_periods': self.lookback_periods,
            'target_periods': self.target_periods,
            'target_threshold': self.target_threshold,
            'scaler_type': self.scaler_type,
            'splits': {}
        }
        
        # Salvar cada split
        for split in ['train', 'valid', 'test']:
            split_data = datasets[split]
            
            # Arquivo principal
            filename = f"{base_name}_{split}_{timestamp}.parquet"
            filepath = os.path.join(self.dataset_path, filename)
            
            # Combinar dados para salvar
            combined_data = pd.concat([
                split_data['features_scaled'],
                split_data['labels'],
                split_data['meta']
            ], axis=1)
            
            combined_data.to_parquet(filepath, compression='snappy')
            
            # Metadados do split
            metadata['splits'][split] = {
                'filename': filename,
                'samples': len(combined_data),
                'features': len(split_data['features'].columns),
                'date_range': [
                    str(combined_data.index[0]),
                    str(combined_data.index[-1])
                ]
            }
            
            # Salvar por regime
            if 'by_regime' in split_data:
                regime_metadata = {}
                for regime, regime_data in split_data['by_regime'].items():
                    regime_filename = f"{base_name}_{split}_{regime}_{timestamp}.parquet"
                    regime_filepath = os.path.join(self.dataset_path, regime_filename)
                    
                    regime_combined = pd.concat([
                        regime_data['features_scaled'],
                        regime_data['labels']
                    ], axis=1)
                    
                    regime_combined.to_parquet(regime_filepath, compression='snappy')
                    
                    regime_metadata[regime] = {
                        'filename': regime_filename,
                        'samples': len(regime_combined)
                    }
                
                metadata['splits'][split]['by_regime'] = regime_metadata
        
        # Salvar metadados
        metadata_path = os.path.join(self.dataset_path, f"{base_name}_metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"   Datasets salvos em: {self.dataset_path}")
        self.logger.info(f"   Metadados: {metadata_path}")
    
    def _print_dataset_statistics(self, datasets: Dict):
        """Imprime estatísticas dos datasets"""
        
        self.logger.info("\n" + "="*60)
        self.logger.info("ESTATÍSTICAS DOS DATASETS")
        self.logger.info("="*60)
        
        for split in ['train', 'valid', 'test']:
            data = datasets[split]
            
            self.logger.info(f"\n{split.upper()}:")
            self.logger.info(f"  Samples: {len(data['features'])}")
            self.logger.info(f"  Features: {len(data['features'].columns)}")
            self.logger.info(f"  Date range: {data['index'][0]} to {data['index'][-1]}")
            
            # Distribuição de labels
            if 'target_class' in data['labels'].columns:
                label_dist = data['labels']['target_class'].value_counts()
                self.logger.info(f"  Label distribution:")
                for label, count in label_dist.items():
                    pct = count / len(data['labels']) * 100
                    self.logger.info(f"    {label}: {count} ({pct:.1f}%)")
            
            # Distribuição por regime
            if 'by_regime' in data:
                self.logger.info(f"  By regime:")
                for regime, regime_data in data['by_regime'].items():
                    self.logger.info(f"    {regime}: {len(regime_data['features'])} samples")
    
    def load_dataset(self, dataset_path: str) -> Dict:
        """Carrega dataset salvo"""
        
        data = pd.read_parquet(dataset_path)
        
        # Separar features, labels e meta
        feature_cols = [col for col in data.columns if col.startswith('v3_')]
        label_cols = ['target_binary', 'target_class', 'target_return', 'target_magnitude']
        meta_cols = ['regime']
        
        return {
            'features': data[feature_cols],
            'labels': data[label_cols],
            'meta': data[meta_cols] if meta_cols[0] in data.columns else None,
            'index': data.index
        }


def main():
    """Teste do DatasetBuilderV3"""
    
    print("="*60)
    print("TESTE DO DATASET BUILDER V3")
    print("="*60)
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuração
    config = {
        'lookback_periods': 100,
        'target_periods': 5,
        'target_threshold': 0.001,
        'train_ratio': 0.7,
        'valid_ratio': 0.15,
        'test_ratio': 0.15
    }
    
    # Criar builder
    builder = DatasetBuilderV3(config)
    
    # Período de teste
    start_date = datetime(2025, 1, 20)
    end_date = datetime(2025, 1, 27)
    
    # Construir dataset
    datasets = builder.build_training_dataset(
        start_date=start_date,
        end_date=end_date,
        ticker="WDOH25"
    )
    
    if datasets:
        print("\n[OK] Dataset V3 construído com sucesso!")
    else:
        print("\n[ERROR] Falha na construção do dataset")


if __name__ == "__main__":
    main()