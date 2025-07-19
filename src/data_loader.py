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
        
        # Adicionar estado para candle em formação
        self.current_candle = None
        self.current_candle_time = None
        self.candles_buffer = pd.DataFrame()  # Buffer de candles formados
        
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

    def load_real_time_data(self, connection_manager, symbol: str = "WDO") -> pd.DataFrame:
        """
        Carrega dados em tempo real do broker via ConnectionManager
        
        Args:
            connection_manager: Instância conectada do ConnectionManager
            symbol: Ticker do ativo (padrão: WDO)
            
        Returns:
            pd.DataFrame: Candles de 1 minuto atualizados
        """
        if not connection_manager.connected:
            raise ConnectionError("ConnectionManager não está conectado ao broker")
        
        # Verificar ambiente
        if os.getenv('TRADING_ENV') == 'PRODUCTION':
            if not connection_manager.market_connected:
                raise ConnectionError("Sem conexão com market data em PRODUÇÃO")
        
        # Subscrever ao ticker se necessário
        if not hasattr(self, '_subscribed_tickers'):
            self._subscribed_tickers = set()
            
        if symbol not in self._subscribed_tickers:
            success = connection_manager.subscribe_ticker(symbol, "F")  # F = BMF
            if success:
                self._subscribed_tickers.add(symbol)
            else:
                raise ConnectionError(f"Falha ao subscrever {symbol}")
        
        # Retornar DataFrame vazio que será preenchido via callbacks
        # O ConnectionManager preencherá via callbacks
        return pd.DataFrame()

    def process_trade_to_candle(self, trades_buffer: List[Dict], 
                            interval: str = '1min') -> pd.DataFrame:
        """
        Converte buffer de trades em candles OHLCV
        
        Args:
            trades_buffer: Lista de trades do ConnectionManager
            interval: Intervalo do candle (1min, 5min, etc)
            
        Returns:
            pd.DataFrame: Candles agregados
        """
        if not trades_buffer:
            return pd.DataFrame()
        
        # Converter para DataFrame
        df_trades = pd.DataFrame(trades_buffer)
        df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
        df_trades.set_index('timestamp', inplace=True)
        
        # Agregar em candles
        candles = df_trades.groupby(pd.Grouper(freq=interval)).agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum',
            'quantity': 'sum'
        })
        
        # Renomear colunas
        candles.columns = ['open', 'high', 'low', 'close', 'volume', 'trades']
        
        # Remover candles vazios
        candles = candles[candles['volume'] > 0]
        
        # Validar dados
        if self._validate_candles(candles):
            return candles
        else:
            self.logger.error("Candles falharam validação")
            return pd.DataFrame()

    def _validate_candles(self, candles: pd.DataFrame) -> bool:
        """Valida integridade dos candles"""
        if candles.empty:
            return False
            
        # Verificar colunas necessárias
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in candles.columns for col in required_cols):
            return False
        
        # Verificar valores válidos
        if (candles[['open', 'high', 'low', 'close']] <= 0).any().any():
            self.logger.error("Preços inválidos (<=0) detectados")
            return False
            
        # Verificar consistência OHLC
        invalid_candles = (
            (candles['high'] < candles['low']) |
            (candles['high'] < candles['open']) |
            (candles['high'] < candles['close']) |
            (candles['low'] > candles['open']) |
            (candles['low'] > candles['close'])
        )
        
        if invalid_candles.any():
            self.logger.error("Candles com OHLC inconsistente detectados")
            return False
            
        return True

    def aggregate_candles(self, candles_1min: pd.DataFrame, 
                     target_interval: str) -> pd.DataFrame:
        """
        Agrega candles de 1 minuto para intervalos maiores
        
        Args:
            candles_1min: DataFrame com candles de 1 minuto
            target_interval: Intervalo alvo ('5min', '15min', '1H', etc)
            
        Returns:
            pd.DataFrame: Candles agregados
        """
        if candles_1min.empty:
            return pd.DataFrame()
        
        # Mapear intervalos
        interval_map = {
            '5min': '5T',
            '15min': '15T',
            '30min': '30T',
            '1H': '1H',
            '4H': '4H',
            '1D': '1D'
        }
        
        if target_interval not in interval_map:
            raise ValueError(f"Intervalo não suportado: {target_interval}")
        
        freq = interval_map[target_interval]
        
        # Agregar
        agg_candles = candles_1min.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remover períodos sem dados
        agg_candles = agg_candles[agg_candles['volume'] > 0]
        
        # Validar
        if self._validate_candles(agg_candles):
            return agg_candles
        else:
            return pd.DataFrame()

    def get_multi_timeframe_data(self, candles_1min: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Cria múltiplos timeframes a partir de candles de 1 minuto
        
        Returns:
            Dict com DataFrames para cada timeframe
        """
        timeframes = {}
        
        # Sempre incluir 1 minuto
        timeframes['1min'] = candles_1min
        
        # Criar outros timeframes comuns para day trade
        for tf in ['5min', '15min', '30min', '1H']:
            try:
                timeframes[tf] = self.aggregate_candles(candles_1min, tf)
            except Exception as e:
                self.logger.error(f"Erro agregando {tf}: {e}")
                timeframes[tf] = pd.DataFrame()
        
        return timeframes
    
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
    
    def create_or_update_candle(self, trade: Dict) -> Optional[pd.DataFrame]:
        """
        Cria ou atualiza candle em tempo real com base em um trade
        
        Args:
            trade: Dict com dados do trade {timestamp, price, volume, quantity}
            
        Returns:
            pd.DataFrame: Candle completo se um novo período começou, None caso contrário
        """
        # Validar trade
        if not self._validate_trade(trade):
            return None
        
        # Determinar tempo do candle (truncar para minuto)
        trade_time = pd.to_datetime(trade['timestamp'])
        candle_time = trade_time.floor('1min')
        
        # Se é o primeiro trade ou novo minuto
        if self.current_candle_time is None or candle_time > self.current_candle_time:
            # Finalizar candle anterior se existir
            completed_candle = None
            if self.current_candle is not None:
                completed_candle = self._finalize_current_candle()
            
            # Iniciar novo candle
            self.current_candle_time = candle_time
            self.current_candle = {
                'timestamp': candle_time,
                'open': trade['price'],
                'high': trade['price'],
                'low': trade['price'],
                'close': trade['price'],
                'volume': trade['volume'],
                'trades': 1,
                'buy_volume': trade['volume'] if trade.get('trade_type') == 2 else 0,
                'sell_volume': trade['volume'] if trade.get('trade_type') == 3 else 0
            }
            
            return completed_candle
        
        # Atualizar candle existente
        else:
            if self.current_candle is not None:
                self.current_candle['high'] = max(self.current_candle['high'], trade['price'])
                self.current_candle['low'] = min(self.current_candle['low'], trade['price'])
                self.current_candle['close'] = trade['price']
                self.current_candle['volume'] += trade['volume']
                self.current_candle['trades'] += 1
                
                # Atualizar volume de compra/venda se disponível
                if trade.get('trade_type') == 2:  # Buy
                    self.current_candle['buy_volume'] += trade['volume']
                elif trade.get('trade_type') == 3:  # Sell
                    self.current_candle['sell_volume'] += trade['volume']
            
            return None

    def _validate_trade(self, trade: Dict) -> bool:
        """Valida dados do trade para criação de candle"""
        required_fields = ['timestamp', 'price', 'volume']
        
        # Verificar campos obrigatórios
        for field in required_fields:
            if field not in trade:
                self.logger.error(f"Campo obrigatório ausente no trade: {field}")
                return False
        
        # Validar valores
        if trade['price'] <= 0:
            self.logger.error(f"Preço inválido: {trade['price']}")
            return False
            
        if trade['volume'] < 0:
            self.logger.error(f"Volume inválido: {trade['volume']}")
            return False
        
        # Para WDO, validar range de preço típico
        if os.getenv('TICKER', 'WDO') == 'WDO':
            if trade['price'] < 3000 or trade['price'] > 10000:
                self.logger.warning(f"Preço fora do range típico WDO: {trade['price']}")
                # Não bloquear, apenas avisar
        
        return True

    def _finalize_current_candle(self) -> Optional[pd.DataFrame]:
        """Finaliza o candle atual e adiciona ao buffer"""
        if self.current_candle is None:
            return None
        
        # Criar DataFrame do candle
        candle_df = pd.DataFrame([self.current_candle])
        candle_df.set_index('timestamp', inplace=True)
        
        # Adicionar ao buffer de candles
        if self.candles_buffer.empty:
            self.candles_buffer = candle_df
        else:
            self.candles_buffer = pd.concat([self.candles_buffer, candle_df])
        
        # Limitar tamanho do buffer (manter últimas 1440 candles = 24h)
        if len(self.candles_buffer) > 1440:
            self.candles_buffer = self.candles_buffer.iloc[-1440:]
        
        self.logger.debug(f"Candle finalizado: {self.current_candle_time} - "
                        f"OHLC: [{self.current_candle['open']:.2f}, "
                        f"{self.current_candle['high']:.2f}, "
                        f"{self.current_candle['low']:.2f}, "
                        f"{self.current_candle['close']:.2f}] "
                        f"Vol: {self.current_candle['volume']}")
        
        return candle_df

    def get_current_candles(self, include_partial: bool = False) -> pd.DataFrame:
        """
        Retorna candles formados e opcionalmente o candle parcial
        
        Args:
            include_partial: Se True, inclui o candle em formação
            
        Returns:
            pd.DataFrame: Candles completos (e parcial se solicitado)
        """
        result = self.candles_buffer.copy()
        
        if include_partial and self.current_candle is not None:
            # Adicionar candle parcial
            partial_df = pd.DataFrame([self.current_candle])
            partial_df.set_index('timestamp', inplace=True)
            
            if not result.empty:
                result = pd.concat([result, partial_df])
            else:
                result = partial_df
        
        return result

    def force_close_current_candle(self) -> Optional[pd.DataFrame]:
        """
        Força o fechamento do candle atual (útil no final do pregão)
        
        Returns:
            pd.DataFrame: Candle finalizado ou None
        """
        if self.current_candle is not None:
            completed_candle = self._finalize_current_candle()
            self.current_candle = None
            self.current_candle_time = None
            return completed_candle
        return None