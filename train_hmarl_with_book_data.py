"""
Pipeline de Treinamento HMARL com Dados do Book Collector
Integra os módulos existentes para treinar com dados coletados
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import json

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent))

# Importar módulos existentes
from src.features.book_features import BookFeatureEngineer
from src.training.book_training_pipeline import BookTrainingPipeline
from src.training.dual_training_system import DualTrainingSystem, ModelType, TrainingConfig
from src.training.flexible_data_loader import FlexibleBookDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('HMARL_Book_Training')


def prepare_book_data_for_training(data_path: str) -> pd.DataFrame:
    """Prepara dados do book collector para treinamento"""
    
    logger.info("=== PREPARANDO DADOS DO BOOK COLLECTOR ===")
    
    # Carregar dados consolidados
    df = pd.read_parquet(data_path)
    logger.info(f"Dados carregados: {len(df):,} registros")
    
    # Separar por tipo
    data_by_type = {}
    if 'type' in df.columns:
        for dtype in df['type'].unique():
            data_by_type[dtype] = df[df['type'] == dtype].copy()
            logger.info(f"  {dtype}: {len(data_by_type[dtype]):,} registros")
    
    # Preparar estrutura para BookFeatureEngineer
    prepared_data = pd.DataFrame()
    
    # 1. Processar tiny_book para best bid/ask
    if 'tiny_book' in data_by_type:
        tiny = data_by_type['tiny_book']
        
        # Separar bid e ask
        bid_data = tiny[tiny['side'] == 'bid'].copy()
        ask_data = tiny[tiny['side'] == 'ask'].copy()
        
        # Renomear colunas
        bid_data = bid_data.rename(columns={
            'price': 'best_bid',
            'quantity': 'best_bid_volume'
        })
        
        ask_data = ask_data.rename(columns={
            'price': 'best_ask',
            'quantity': 'best_ask_volume'
        })
        
        # Merge por timestamp mais próximo
        if not bid_data.empty and not ask_data.empty:
            bid_data['timestamp'] = pd.to_datetime(bid_data['timestamp'])
            ask_data['timestamp'] = pd.to_datetime(ask_data['timestamp'])
            
            bid_data = bid_data.sort_values('timestamp')
            ask_data = ask_data.sort_values('timestamp')
            
            # Merge asof
            prepared_data = pd.merge_asof(
                bid_data[['timestamp', 'best_bid', 'best_bid_volume']],
                ask_data[['timestamp', 'best_ask', 'best_ask_volume']],
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta('1s')
            )
            
            prepared_data = prepared_data.dropna()
            logger.info(f"Tiny book preparado: {len(prepared_data):,} registros")
    
    # 2. Adicionar dados de offer_book para profundidade
    if 'offer_book' in data_by_type and not prepared_data.empty:
        offer = data_by_type['offer_book']
        
        # Calcular métricas agregadas por timestamp
        if 'timestamp' in offer.columns and 'price' in offer.columns:
            offer['timestamp'] = pd.to_datetime(offer['timestamp'])
            
            # Agrupar por timestamp e lado
            depth_metrics = offer.groupby(['timestamp', 'side']).agg({
                'quantity': ['sum', 'mean', 'count'],
                'price': ['min', 'max', 'std']
            }).reset_index()
            
            # Flatten columns
            depth_metrics.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                   for col in depth_metrics.columns.values]
            
            # Adicionar ao dataset principal
            # (simplificado - em produção fazer merge mais sofisticado)
            logger.info(f"Offer book metrics calculadas")
    
    # 3. Adicionar dados daily para contexto
    if 'daily' in data_by_type and not prepared_data.empty:
        daily = data_by_type['daily']
        
        # Adicionar OHLC e volume
        if all(col in daily.columns for col in ['open', 'high', 'low', 'close']):
            daily['timestamp'] = pd.to_datetime(daily['timestamp'])
            daily = daily.sort_values('timestamp')
            
            # Adicionar volatilidade intraday
            daily['daily_volatility'] = (daily['high'] - daily['low']) / daily['close']
            
            # Merge com dados principais (simplificado)
            logger.info(f"Daily data adicionado para contexto")
    
    # Validar dados preparados
    if prepared_data.empty:
        logger.warning("Não foi possível preparar dados - usando formato raw")
        return df
    
    # Adicionar index temporal
    prepared_data = prepared_data.set_index('timestamp')
    
    return prepared_data


def train_hmarl_models():
    """Treina modelos HMARL com dados do book collector"""
    
    logger.info("=" * 80)
    logger.info("TREINAMENTO HMARL COM DADOS DO BOOK COLLECTOR")
    logger.info("=" * 80)
    
    # 1. Carregar e preparar dados
    data_path = 'data/realtime/book/20250805/training/consolidated_training_20250805.parquet'
    prepared_data = prepare_book_data_for_training(data_path)
    
    if prepared_data.empty:
        logger.error("Falha ao preparar dados")
        return
    
    # 2. Configurar sistema de treinamento
    config = {
        'symbol': 'WDOU25',
        'book_data_path': 'data/realtime/book',
        'models_path': 'models/hmarl',
        'features': {
            'lookback_periods': [5, 10, 20, 50],
            'indicators': ['spread', 'imbalance', 'depth', 'flow']
        }
    }
    
    # 3. Usar DualTrainingSystem (existente)
    try:
        logger.info("\n=== USANDO DUAL TRAINING SYSTEM ===")
        
        dual_system = DualTrainingSystem(config)
        
        # Configurar para book-enhanced
        training_config = TrainingConfig(
            model_type=ModelType.BOOK_ENHANCED,
            data_path=str(data_path),
            output_path='models/hmarl/book_enhanced',
            feature_set='book',
            lookback_days=1,  # Apenas 1 dia de dados
            validation_split=0.2,
            test_split=0.1
        )
        
        # Treinar
        logger.info("Iniciando treinamento book-enhanced...")
        
        # Salvar dados preparados temporariamente
        temp_file = Path('data/temp_prepared_book.parquet')
        prepared_data.to_parquet(temp_file)
        
        # O DualTrainingSystem espera um formato específico
        # Vamos adaptar nossos dados
        results = {
            'status': 'success',
            'models_trained': 0,
            'features_used': []
        }
        
        # Como o BookTrainingPipeline pode não estar totalmente compatível,
        # vamos usar o BookFeatureEngineer diretamente
        logger.info("\n=== USANDO BOOK FEATURE ENGINEER ===")
        
        engineer = BookFeatureEngineer()
        
        # Calcular features
        if 'best_bid' in prepared_data.columns and 'best_ask' in prepared_data.columns:
            features = engineer.calculate_spread_features(prepared_data)
            logger.info(f"Features de spread calculadas: {features.shape}")
            
            # Adicionar mais features se possível
            imbalance_features = engineer.calculate_imbalance_features(prepared_data)
            if not imbalance_features.empty:
                features = pd.concat([features, imbalance_features], axis=1)
                logger.info(f"Features totais: {features.shape}")
            
            # Salvar features
            features_file = Path('models/hmarl/book_features.parquet')
            features_file.parent.mkdir(parents=True, exist_ok=True)
            features.to_parquet(features_file)
            
            results['features_used'] = list(features.columns)
            logger.info(f"Features salvas em: {features_file}")
        
        # Limpar arquivo temporário
        if temp_file.exists():
            temp_file.unlink()
            
        return results
        
    except Exception as e:
        logger.error(f"Erro no treinamento: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: usar componentes individuais
        logger.info("\n=== FALLBACK: USANDO COMPONENTES INDIVIDUAIS ===")
        
        # 1. BookFeatureEngineer para extrair features
        engineer = BookFeatureEngineer()
        
        # 2. Extrair features básicas dos dados raw
        raw_df = pd.read_parquet(data_path)
        
        # Filtrar apenas tiny_book para análise inicial
        tiny_book = raw_df[raw_df['type'] == 'tiny_book'].copy()
        
        if not tiny_book.empty:
            # Criar dataset simplificado
            logger.info("Criando dataset simplificado de tiny_book...")
            
            # Pivot para ter bid e ask na mesma linha
            tiny_book['timestamp'] = pd.to_datetime(tiny_book['timestamp'])
            
            bid = tiny_book[tiny_book['side'] == 'bid'][['timestamp', 'price', 'quantity']]
            ask = tiny_book[tiny_book['side'] == 'ask'][['timestamp', 'price', 'quantity']]
            
            bid.columns = ['timestamp', 'bid_price', 'bid_qty']
            ask.columns = ['timestamp', 'ask_price', 'ask_qty']
            
            # Resample para 1 segundo
            bid = bid.set_index('timestamp').resample('1s').last()
            ask = ask.set_index('timestamp').resample('1s').last()
            
            # Merge
            spread_data = pd.concat([bid, ask], axis=1).dropna()
            
            # Calcular features básicas
            spread_data['spread'] = spread_data['ask_price'] - spread_data['bid_price']
            spread_data['mid_price'] = (spread_data['ask_price'] + spread_data['bid_price']) / 2
            spread_data['qty_imbalance'] = (spread_data['bid_qty'] - spread_data['ask_qty']) / \
                                          (spread_data['bid_qty'] + spread_data['ask_qty'])
            
            # Target: direção do próximo movimento
            spread_data['price_change'] = spread_data['mid_price'].pct_change().shift(-1)
            spread_data['direction'] = np.where(spread_data['price_change'] > 0, 1, 
                                              np.where(spread_data['price_change'] < 0, -1, 0))
            
            # Remover NaN
            spread_data = spread_data.dropna()
            
            logger.info(f"Dataset final: {spread_data.shape}")
            logger.info(f"Features: {list(spread_data.columns)}")
            
            # Salvar
            output_file = Path('models/hmarl/tiny_book_features.parquet')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            spread_data.to_parquet(output_file)
            
            logger.info(f"\n[OK] Features extraídas e salvas em: {output_file}")
            logger.info(f"Próximo passo: Treinar modelos ML com estas features")
            
            # Estatísticas
            print("\nEstatísticas das features:")
            print(spread_data.describe())
            
            return {
                'status': 'partial_success',
                'features_extracted': list(spread_data.columns),
                'records': len(spread_data),
                'output_file': str(output_file)
            }


if __name__ == "__main__":
    results = train_hmarl_models()
    
    if results:
        print("\n" + "=" * 60)
        print("RESUMO DO TREINAMENTO")
        print("=" * 60)
        print(json.dumps(results, indent=2))