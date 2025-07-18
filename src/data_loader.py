import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import os
import json
from pathlib import Path


class DataLoader:
    """Carrega dados de diferentes fontes (histórico, cache, etc)"""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        self.cache_dir = self.data_dir / "cache"
        
        # Criar diretórios se não existirem
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_historical_trades(self, 
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             symbol: str = "BTCUSDT") -> List[Dict]:
        """Carrega trades históricos de arquivo ou API"""
        try:
            # Tentar carregar do cache primeiro
            cache_file = self.cache_dir / f"{symbol}_trades_{start_date}_{end_date}.parquet"
            
            if cache_file.exists():
                self.logger.info(f"Carregando trades do cache: {cache_file}")
                df = pd.read_parquet(cache_file)
                return df.to_dict('records')
            
            # Se não houver cache, tentar carregar de arquivo CSV
            csv_file = self.data_dir / f"{symbol}_trades.csv"
            
            if csv_file.exists():
                self.logger.info(f"Carregando trades de CSV: {csv_file}")
                df = pd.read_csv(csv_file, parse_dates=['timestamp'])
                
                # Filtrar por data se especificado
                if start_date:
                    df = df[df['timestamp'] >= start_date]
                if end_date:
                    df = df[df['timestamp'] <= end_date]
                
                # Salvar no cache para próxima vez
                df.to_parquet(cache_file, index=False)
                
                return df.to_dict('records')
            
            # Se não houver dados, retornar lista vazia
            self.logger.warning(f"Nenhum dado histórico encontrado para {symbol}")
            return []
            
        except Exception as e:
            self.logger.error(f"Erro carregando dados históricos: {e}")
            return []
    
    def load_candles(self, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    interval: str = "1m",
                    symbol: str = "BTCUSDT") -> pd.DataFrame:
        """Carrega dados de candles históricos"""
        try:
            # Tentar carregar do cache
            cache_file = self.cache_dir / f"{symbol}_candles_{interval}_{start_date}_{end_date}.parquet"
            
            if cache_file.exists():
                self.logger.info(f"Carregando candles do cache: {cache_file}")
                return pd.read_parquet(cache_file)
            
            # Tentar carregar de CSV
            csv_file = self.data_dir / f"{symbol}_candles_{interval}.csv"
            
            if csv_file.exists():
                self.logger.info(f"Carregando candles de CSV: {csv_file}")
                df = pd.read_csv(csv_file, parse_dates=['timestamp'], index_col='timestamp')
                
                # Filtrar por data
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                
                # Validar dados OHLCV
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if all(col in df.columns for col in required_cols):
                    # Salvar no cache
                    df.to_parquet(cache_file)
                    return df
                else:
                    self.logger.error(f"Colunas faltando no arquivo de candles: {required_cols}")
                    return pd.DataFrame()
            
            self.logger.warning(f"Nenhum dado de candles encontrado para {symbol}")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Erro carregando candles: {e}")
            return pd.DataFrame()
    
    def load_orderbook_snapshots(self,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None,
                                symbol: str = "BTCUSDT") -> pd.DataFrame:
        """Carrega snapshots do livro de ordens"""
        try:
            file_path = self.data_dir / f"{symbol}_orderbook.parquet"
            
            if file_path.exists():
                self.logger.info(f"Carregando orderbook: {file_path}")
                df = pd.read_parquet(file_path)
                
                # Filtrar por data
                if start_date and 'timestamp' in df.columns:
                    df = df[df['timestamp'] >= start_date]
                if end_date and 'timestamp' in df.columns:
                    df = df[df['timestamp'] <= end_date]
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Erro carregando orderbook: {e}")
            return pd.DataFrame()
    
    def save_processed_data(self, 
                          data: Union[pd.DataFrame, List[Dict]], 
                          filename: str,
                          format: str = "parquet"):
        """Salva dados processados para reuso futuro"""
        try:
            file_path = self.cache_dir / f"{filename}.{format}"
            
            if format == "parquet":
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = data
                df.to_parquet(file_path)
                
            elif format == "csv":
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = data
                df.to_csv(file_path)
                
            elif format == "json":
                with open(file_path, 'w') as f:
                    if isinstance(data, pd.DataFrame):
                        json.dump(data.to_dict('records'), f)
                    else:
                        json.dump(data, f)
            
            self.logger.info(f"Dados salvos em: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Erro salvando dados: {e}")
    
    def clear_cache(self, older_than_days: int = 7):
        """Limpa arquivos de cache antigos"""
        try:
            cutoff_time = datetime.now() - timedelta(days=older_than_days)
            
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file():
                    # Verificar data de modificação
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mod_time < cutoff_time:
                        file_path.unlink()
                        self.logger.info(f"Cache removido: {file_path}")
                        
        except Exception as e:
            self.logger.error(f"Erro limpando cache: {e}")
    
    def get_available_data(self) -> Dict[str, List[str]]:
        """Lista dados disponíveis"""
        try:
            available = {
                'trades': [],
                'candles': [],
                'orderbook': [],
                'cache': []
            }
            
            # Listar arquivos de dados
            for file_path in self.data_dir.glob("*"):
                if file_path.is_file():
                    filename = file_path.name
                    if 'trades' in filename:
                        available['trades'].append(filename)
                    elif 'candles' in filename:
                        available['candles'].append(filename)
                    elif 'orderbook' in filename:
                        available['orderbook'].append(filename)
            
            # Listar cache
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file():
                    available['cache'].append(file_path.name)
            
            return available
            
        except Exception as e:
            self.logger.error(f"Erro listando dados disponíveis: {e}")
            return {'trades': [], 'candles': [], 'orderbook': [], 'cache': []}
    
    def generate_sample_data(self, n_candles: int = 1000) -> Dict[str, pd.DataFrame]:
        """Gera dados de exemplo para testes"""
        try:
            # Gerar timestamps
            end_time = datetime.now()
            timestamps = pd.date_range(
                end=end_time, 
                periods=n_candles, 
                freq='1min'
            )
            
            # Gerar preços com movimento browniano
            price = 50000.0
            prices: List[float] = [price]
            for _ in range(n_candles - 1):
                change = np.random.normal(0, 0.001)  # 0.1% volatilidade
                price = price * (1 + change)
                prices.append(price)
            
            # Gerar candles
            candles = []
            for i, (ts, close_price) in enumerate(zip(timestamps, prices)):
                # Gerar OHLC baseado no close
                high = close_price * (1 + abs(np.random.normal(0, 0.0005)))
                low = close_price * (1 - abs(np.random.normal(0, 0.0005)))
                open_price = prices[i-1] if i > 0 else close_price
                volume = np.random.uniform(10, 100)
                
                candles.append({
                    'timestamp': ts,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume
                })
            
            df_candles = pd.DataFrame(candles).set_index('timestamp')
            
            # Gerar microestrutura
            micro_data = []
            for ts in timestamps:
                buy_volume = np.random.uniform(5, 50)
                sell_volume = np.random.uniform(5, 50)
                
                micro_data.append({
                    'timestamp': ts,
                    'buy_volume': buy_volume,
                    'sell_volume': sell_volume,
                    'buy_trades': int(buy_volume / 5),
                    'sell_trades': int(sell_volume / 5),
                    'imbalance': buy_volume - sell_volume,
                    'avg_trade_size': (buy_volume + sell_volume) / (int(buy_volume/5) + int(sell_volume/5))
                })
            
            df_micro = pd.DataFrame(micro_data).set_index('timestamp')
            
            return {
                'candles': df_candles,
                'microstructure': df_micro
            }
            
        except Exception as e:
            self.logger.error(f"Erro gerando dados de exemplo: {e}")
            return {'candles': pd.DataFrame(), 'microstructure': pd.DataFrame()}