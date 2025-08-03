import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from src.data_structure import TradingDataStructure


class DataPipeline:
    """Pipeline unificado para processar dados históricos e em tempo real"""
    
    def __init__(self, data_structure: TradingDataStructure):
        self.data = data_structure
        self.logger = logging.getLogger(__name__)
        self.candle_interval = timedelta(seconds=60)  # 1 minuto
        
    def process_historical_trades(self, trades: List[Dict]) -> Dict[str, pd.DataFrame]:
        """Processa lista de trades históricos e gera DataFrames"""
        try:
            if not trades:
                self.logger.warning("Lista de trades vazia")
                return {'candles': pd.DataFrame(), 'microstructure': pd.DataFrame()}
                
            # Converter trades para DataFrame
            df_trades = pd.DataFrame(trades)
            df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
            df_trades = df_trades.sort_values('timestamp')
            
            # Agrupar por intervalo de candle
            df_trades['candle_time'] = df_trades['timestamp'].dt.floor('1min')
            
            candles_data = []
            micro_data = []
            
            # Processar cada grupo
            for time, group in df_trades.groupby('candle_time'):
                # Gerar candle
                candle = {
                    'timestamp': time,
                    'open': group['price'].iloc[0],
                    'high': group['price'].max(),
                    'low': group['price'].min(),
                    'close': group['price'].iloc[-1],
                    'volume': group['volume'].sum()
                }
                candles_data.append(candle)
                
                # Calcular microestrutura
                buy_trades = group[group['trade_type'] == 2]
                sell_trades = group[group['trade_type'] == 3]
                
                micro = {
                    'timestamp': time,
                    'buy_volume': buy_trades['volume'].sum() if len(buy_trades) > 0 else 0,
                    'sell_volume': sell_trades['volume'].sum() if len(sell_trades) > 0 else 0,
                    'buy_trades': len(buy_trades),
                    'sell_trades': len(sell_trades),
                    'imbalance': (buy_trades['volume'].sum() if len(buy_trades) > 0 else 0) - 
                                (sell_trades['volume'].sum() if len(sell_trades) > 0 else 0),
                    'avg_trade_size': group['volume'].mean(),
                    'trade_count': len(group)
                }
                micro_data.append(micro)
            
            # Criar DataFrames
            self.data.candles = pd.DataFrame(candles_data).set_index('timestamp')
            self.data.microstructure = pd.DataFrame(micro_data).set_index('timestamp')
            
            # Validar e limpar dados
            self._validate_data()
            
            self.logger.info(f"Processados {len(candles_data)} candles e {len(trades)} trades")
            
            return {
                'candles': self.data.candles,
                'microstructure': self.data.microstructure
            }
            
        except Exception as e:
            self.logger.error(f"Erro processando trades históricos: {e}")
            return {'candles': pd.DataFrame(), 'microstructure': pd.DataFrame()}
    
    def get_aligned_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Retorna dados alinhados e prontos para ML"""
        try:
            # Verificar se há dados
            if self.data.candles.empty:
                self.logger.warning("Sem dados de candles disponíveis")
                return pd.DataFrame()
            
            # Começar com candles
            df_aligned = self.data.candles.copy()
            
            # Adicionar microestrutura se disponível
            if not self.data.microstructure.empty:
                # Alinhar índices
                common_index = df_aligned.index.intersection(self.data.microstructure.index)
                if len(common_index) > 0:
                    df_aligned = df_aligned.loc[common_index]
                    micro_cols = ['buy_volume', 'sell_volume', 'imbalance', 
                                  'buy_trades', 'sell_trades', 'avg_trade_size']
                    for col in micro_cols:
                        if col in self.data.microstructure.columns:
                            df_aligned[col] = self.data.microstructure.loc[common_index, col]
            
            # Adicionar indicadores se disponíveis
            if not self.data.indicators.empty:
                # Alinhar com indicadores
                common_index = df_aligned.index.intersection(self.data.indicators.index)
                if len(common_index) > 0:
                    df_aligned = df_aligned.loc[common_index]
                    for col in self.data.indicators.columns:
                        df_aligned[col] = self.data.indicators.loc[common_index, col]
            
            # Limitar quantidade se especificado
            if limit and len(df_aligned) > limit:
                df_aligned = df_aligned.iloc[-limit:]
            
            # Remover NaN
            df_aligned = df_aligned.dropna()
            
            self.logger.info(f"Dados alinhados: {len(df_aligned)} registros, {len(df_aligned.columns)} colunas")
            
            return df_aligned
            
        except Exception as e:
            self.logger.error(f"Erro alinhando dados: {e}")
            return pd.DataFrame()
    
    def _validate_data(self):
        """Valida e limpa os dados"""
        try:
            # Validar candles
            if not self.data.candles.empty:
                # Remover valores negativos ou zero
                self.data.candles = self.data.candles[
                    (self.data.candles['close'] > 0) & 
                    (self.data.candles['volume'] >= 0)
                ]
                
                # Verificar consistência OHLC
                mask = (
                    (self.data.candles['high'] >= self.data.candles['low']) &
                    (self.data.candles['high'] >= self.data.candles['open']) &
                    (self.data.candles['high'] >= self.data.candles['close']) &
                    (self.data.candles['low'] <= self.data.candles['open']) &
                    (self.data.candles['low'] <= self.data.candles['close'])
                )
                self.data.candles = self.data.candles[mask]
            
            # Validar microestrutura
            if not self.data.microstructure.empty:
                # Remover valores negativos
                numeric_cols = ['buy_volume', 'sell_volume', 'buy_trades', 
                               'sell_trades', 'avg_trade_size']
                for col in numeric_cols:
                    if col in self.data.microstructure.columns:
                        self.data.microstructure[col] = self.data.microstructure[col].clip(lower=0)
                        
        except Exception as e:
            self.logger.error(f"Erro validando dados: {e}")
    
    def update_indicators(self, indicators: pd.DataFrame):
        """Atualiza DataFrame de indicadores"""
        try:
            if not indicators.empty:
                self.data.indicators = indicators
                self.logger.info(f"Indicadores atualizados: {len(indicators)} registros")
        except Exception as e:
            self.logger.error(f"Erro atualizando indicadores: {e}")
    
    def clear_old_data(self, keep_days: int = 7):
        """Remove dados antigos para economizar memória"""
        try:
            cutoff_time = datetime.now() - timedelta(days=keep_days)
            
            # Limpar candles antigos
            if not self.data.candles.empty:
                self.data.candles = self.data.candles[self.data.candles.index > cutoff_time]
            
            # Limpar microestrutura antiga
            if not self.data.microstructure.empty:
                self.data.microstructure = self.data.microstructure[
                    self.data.microstructure.index > cutoff_time
                ]
            
            # Limpar indicadores antigos
            if not self.data.indicators.empty:
                self.data.indicators = self.data.indicators[
                    self.data.indicators.index > cutoff_time
                ]
                
            self.logger.info(f"Dados anteriores a {cutoff_time} removidos")
            
        except Exception as e:
            self.logger.error(f"Erro limpando dados antigos: {e}")