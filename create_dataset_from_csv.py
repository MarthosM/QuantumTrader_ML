"""
Script para criar dataset ML a partir de arquivo CSV com dados de trades
Formato esperado: ticker,date,time,trade_number,price,qty,vol,buy_agent,sell_agent,trade_type,aft
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import sys
from typing import Dict, Optional, Tuple
import argparse

# Adicionar src ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ml.dataset_builder_v3 import DatasetBuilderV3

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class CSVDatasetCreator:
    """Cria dataset ML a partir de arquivo CSV de trades"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def load_trades_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Carrega trades do arquivo CSV"""
        self.logger.info(f"Carregando dados de: {csv_path}")
        
        # Carregar CSV
        trades_df = pd.read_csv(csv_path)
        
        # Verificar colunas esperadas
        expected_cols = ['ticker', 'date', 'time', 'trade_number', 'price', 
                        'qty', 'vol', 'buy_agent', 'sell_agent', 'trade_type']
        
        missing_cols = set(expected_cols) - set(trades_df.columns)
        if missing_cols:
            raise ValueError(f"Colunas faltando no CSV: {missing_cols}")
        
        # Criar datetime index
        trades_df['datetime'] = pd.to_datetime(
            trades_df['date'].astype(str) + ' ' + trades_df['time'].astype(str).str.zfill(6),
            format='%Y%m%d %H%M%S'
        )
        trades_df.set_index('datetime', inplace=True)
        trades_df.sort_index(inplace=True)
        
        # Determinar side (buy/sell) baseado no trade_type ou agentes
        # Assumindo que trade_type indica quem foi o agressor
        trades_df['side'] = trades_df.apply(self._determine_side, axis=1)
        
        self.logger.info(f"Trades carregados: {len(trades_df):,}")
        self.logger.info(f"Período: {trades_df.index[0]} a {trades_df.index[-1]}")
        
        return trades_df
    
    def _determine_side(self, row) -> str:
        """Determina se trade foi compra ou venda baseado no trade_type"""
        trade_type = str(row['trade_type']).strip()
        
        # Mapeamento dos tipos de trade para side
        # Baseado na agressão - quem cruzou o spread
        buy_types = [
            'Compra Agressão',      # Comprador agressor
            'Cross Trade',          # Considerar 50/50 ou usar outra lógica
            'Options Exercise',     # Geralmente compra do ativo
            'Expit',               # Exercício antecipado
        ]
        
        sell_types = [
            'Venda Agressão',       # Vendedor agressor
        ]
        
        # Tipos neutros ou que precisam análise adicional
        neutral_types = [
            'Leilão',              # Pode ser ambos
            'Surveillance',        # Sob monitoramento
            'Over the Counter',    # OTC
            'Derivative Term',     # Derivativos
            'Index',              # Índices
            'BTC',                # Aluguel
            'On Behalf',          # Por conta de terceiros
            'Desconhecido',       # Tipo 32
        ]
        
        # Determinar side
        if trade_type in buy_types:
            return 'BUY'
        elif trade_type in sell_types:
            return 'SELL'
        else:
            # Para tipos neutros, podemos usar outras heurísticas:
            # 1. Analisar se o preço está mais próximo do bid ou ask
            # 2. Ver a direção do preço em relação ao trade anterior
            # 3. Por enquanto, vamos distribuir igualmente baseado no trade_number
            # Isso mantém o balanço mas pode ser melhorado com dados de bid/ask
            return 'BUY' if row['trade_number'] % 2 == 0 else 'SELL'
    
    def aggregate_trades_to_candles(self, trades_df: pd.DataFrame, 
                                   timeframe: str = '1min') -> pd.DataFrame:
        """Agrega trades em candles OHLCV"""
        self.logger.info(f"Agregando trades em candles de {timeframe}")
        
        # Agrupar por período
        grouped = trades_df.groupby(pd.Grouper(freq=timeframe))
        
        # Criar candles
        candles = pd.DataFrame()
        candles['open'] = grouped['price'].first()
        candles['high'] = grouped['price'].max()
        candles['low'] = grouped['price'].min()
        candles['close'] = grouped['price'].last()
        candles['volume'] = grouped['qty'].sum()
        candles['trades'] = grouped['price'].count()
        
        # Remover períodos sem trades
        candles = candles.dropna()
        
        self.logger.info(f"Candles criados: {len(candles)}")
        
        return candles
    
    def calculate_microstructure(self, trades_df: pd.DataFrame, 
                               timeframe: str = '1min') -> pd.DataFrame:
        """Calcula métricas de microestrutura"""
        self.logger.info("Calculando microestrutura de mercado")
        
        # Agrupar por período
        grouped = trades_df.groupby(pd.Grouper(freq=timeframe))
        
        # Calcular métricas
        micro = pd.DataFrame()
        
        # Volume por side
        for side in ['BUY', 'SELL']:
            side_trades = trades_df[trades_df['side'] == side]
            side_grouped = side_trades.groupby(pd.Grouper(freq=timeframe))
            
            micro[f'{side.lower()}_volume'] = side_grouped['qty'].sum()
            micro[f'{side.lower()}_trades'] = side_grouped['qty'].count()
            micro[f'{side.lower()}_avg_size'] = side_grouped['qty'].mean()
        
        # Preencher NaN com 0 (períodos sem trades daquele tipo)
        micro = micro.fillna(0)
        
        # Calcular métricas derivadas
        total_volume = micro['buy_volume'] + micro['sell_volume']
        total_trades = micro['buy_trades'] + micro['sell_trades']
        
        # Evitar divisão por zero
        total_volume = total_volume.replace(0, 1)
        total_trades = total_trades.replace(0, 1)
        
        # Métricas de imbalance
        micro['volume_imbalance'] = (micro['buy_volume'] - micro['sell_volume']) / total_volume
        micro['trade_imbalance'] = (micro['buy_trades'] - micro['sell_trades']) / total_trades
        micro['buy_pressure'] = micro['buy_volume'] / total_volume
        micro['sell_pressure'] = micro['sell_volume'] / total_volume
        
        # Order flow
        micro['order_flow'] = micro['buy_volume'] - micro['sell_volume']
        micro['order_flow_ratio'] = micro['buy_volume'] / (micro['sell_volume'] + 1)
        
        self.logger.info(f"Microestrutura calculada: {len(micro)} períodos")
        
        return micro
    
    def prepare_data_for_dataset_builder(self, trades_df: pd.DataFrame) -> Dict:
        """Prepara dados no formato esperado pelo DatasetBuilderV3"""
        
        # 1. Criar candles
        candles = self.aggregate_trades_to_candles(trades_df)
        
        # 2. Criar microestrutura
        microstructure = self.calculate_microstructure(trades_df)
        
        # 3. Garantir alinhamento temporal
        common_index = candles.index.intersection(microstructure.index)
        candles = candles.loc[common_index]
        microstructure = microstructure.loc[common_index]
        
        # 4. Retornar no formato esperado
        return {
            'candles': candles,
            'microstructure': microstructure,
            'trades': trades_df
        }
    
    def create_ml_dataset(self, csv_path: str, output_prefix: str = None) -> Dict:
        """Processo completo de criação do dataset ML"""
        
        # 1. Carregar trades
        trades_df = self.load_trades_from_csv(csv_path)
        
        # 2. Preparar dados
        data = self.prepare_data_for_dataset_builder(trades_df)
        
        # 3. Usar DatasetBuilderV3 para criar dataset ML
        builder_config = {
            'lookback_periods': self.config.get('lookback_periods', 100),
            'target_periods': self.config.get('target_periods', 5),
            'target_threshold': self.config.get('target_threshold', 0.001),
            'train_ratio': self.config.get('train_ratio', 0.7),
            'valid_ratio': self.config.get('valid_ratio', 0.15),
            'test_ratio': self.config.get('test_ratio', 0.15)
        }
        
        builder = DatasetBuilderV3(builder_config)
        
        # Usar dados preparados (evitar coleta via ProfitDLL)
        builder._data_cache = data
        
        # Sobrescrever método de coleta
        def use_cached_data(*args, **kwargs):
            return builder._data_cache
        
        original_collect = builder._collect_real_data
        builder._collect_real_data = use_cached_data
        
        try:
            # Determinar ticker e período
            ticker = trades_df['ticker'].iloc[0] if 'ticker' in trades_df.columns else 'WDOFUT'
            start_date = trades_df.index[0]
            end_date = trades_df.index[-1]
            
            if output_prefix:
                ticker = output_prefix
            
            # Construir dataset
            datasets = builder.build_training_dataset(
                start_date=start_date,
                end_date=end_date,
                ticker=ticker
            )
            
            return datasets
            
        finally:
            builder._collect_real_data = original_collect
    
    def analyze_data_quality(self, csv_path: str):
        """Analisa a qualidade dos dados no CSV"""
        trades_df = self.load_trades_from_csv(csv_path)
        
        print("\n" + "="*60)
        print("ANÁLISE DE QUALIDADE DOS DADOS")
        print("="*60)
        
        # Informações gerais
        print(f"\nInformações Gerais:")
        print(f"  Total de trades: {len(trades_df):,}")
        print(f"  Período: {trades_df.index[0]} a {trades_df.index[-1]}")
        print(f"  Duração: {(trades_df.index[-1] - trades_df.index[0]).days} dias")
        
        # Distribuição por ticker
        if 'ticker' in trades_df.columns:
            print(f"\nTickers encontrados:")
            for ticker, count in trades_df['ticker'].value_counts().items():
                print(f"  {ticker}: {count:,} trades")
        
        # Análise temporal
        trades_per_day = trades_df.groupby(trades_df.index.date).size()
        print(f"\nDistribuição temporal:")
        print(f"  Dias com dados: {len(trades_per_day)}")
        print(f"  Trades/dia (média): {trades_per_day.mean():,.0f}")
        print(f"  Trades/dia (min): {trades_per_day.min():,}")
        print(f"  Trades/dia (max): {trades_per_day.max():,}")
        
        # Análise de preços
        print(f"\nAnálise de preços:")
        print(f"  Preço mínimo: {trades_df['price'].min():,.2f}")
        print(f"  Preço máximo: {trades_df['price'].max():,.2f}")
        print(f"  Preço médio: {trades_df['price'].mean():,.2f}")
        print(f"  Desvio padrão: {trades_df['price'].std():,.2f}")
        
        # Análise de volume
        print(f"\nAnálise de volume:")
        print(f"  Volume total: {trades_df['qty'].sum():,}")
        print(f"  Volume médio/trade: {trades_df['qty'].mean():,.2f}")
        
        # Análise de tipos de trade
        if 'trade_type' in trades_df.columns:
            print(f"\nTipos de trade encontrados:")
            trade_types = trades_df['trade_type'].value_counts()
            for trade_type, count in trade_types.items():
                pct = count / len(trades_df) * 100
                print(f"  {trade_type}: {count:,} ({pct:.1f}%)")
            
            # Análise de agressão (importante para microestrutura)
            buy_aggression = trades_df[trades_df['trade_type'] == 'Compra Agressão']
            sell_aggression = trades_df[trades_df['trade_type'] == 'Venda Agressão']
            
            if len(buy_aggression) > 0 or len(sell_aggression) > 0:
                total_aggression = len(buy_aggression) + len(sell_aggression)
                buy_pct = len(buy_aggression) / total_aggression * 100 if total_aggression > 0 else 0
                
                print(f"\nAnálise de agressão:")
                print(f"  Compra agressão: {len(buy_aggression):,} ({buy_pct:.1f}%)")
                print(f"  Venda agressão: {len(sell_aggression):,} ({100-buy_pct:.1f}%)")
                print(f"  Balanço: {'Mais compra' if buy_pct > 50 else 'Mais venda'} agressiva")
        
        # Gaps temporais
        time_diff = trades_df.index.to_series().diff()
        large_gaps = time_diff[time_diff > pd.Timedelta(hours=1)]
        
        if len(large_gaps) > 0:
            print(f"\n⚠️ Gaps temporais encontrados (> 1 hora): {len(large_gaps)}")
            print("  Maiores gaps:")
            for idx, gap in large_gaps.nlargest(5).items():
                print(f"    {idx}: {gap}")


def main():
    parser = argparse.ArgumentParser(description='Criar dataset ML a partir de CSV de trades')
    
    parser.add_argument('csv_file', type=str,
                       help='Caminho para o arquivo CSV com dados de trades')
    
    parser.add_argument('--output-prefix', type=str,
                       help='Prefixo para os arquivos de saída')
    
    parser.add_argument('--analyze-only', action='store_true',
                       help='Apenas analisar os dados, sem criar dataset')
    
    parser.add_argument('--lookback', type=int, default=100,
                       help='Períodos de lookback para features (padrão: 100)')
    
    parser.add_argument('--target-periods', type=int, default=5,
                       help='Períodos futuros para labels (padrão: 5)')
    
    parser.add_argument('--target-threshold', type=float, default=0.001,
                       help='Threshold para classificação (padrão: 0.001)')
    
    args = parser.parse_args()
    
    # Verificar se arquivo existe
    if not os.path.exists(args.csv_file):
        print(f"Erro: Arquivo não encontrado: {args.csv_file}")
        return 1
    
    # Configuração
    config = {
        'lookback_periods': args.lookback,
        'target_periods': args.target_periods,
        'target_threshold': args.target_threshold
    }
    
    creator = CSVDatasetCreator(config)
    
    if args.analyze_only:
        # Apenas analisar dados
        creator.analyze_data_quality(args.csv_file)
    else:
        # Criar dataset
        print("="*60)
        print("CRIANDO DATASET ML A PARTIR DE CSV")
        print("="*60)
        
        try:
            datasets = creator.create_ml_dataset(
                args.csv_file,
                output_prefix=args.output_prefix
            )
            
            if datasets:
                print("\n✅ Dataset criado com sucesso!")
                print("\nArquivos salvos em: datasets/")
                print("\nPróximos passos:")
                print("1. Verificar dataset: python analyze_dataset_compatibility.py")
                print("2. Treinar modelos: python src/ml/training_orchestrator_v3.py")
                
        except Exception as e:
            print(f"\n❌ Erro ao criar dataset: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())