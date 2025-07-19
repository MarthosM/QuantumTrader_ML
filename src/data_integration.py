"""
Integração entre ConnectionManager e DataLoader
Gerencia fluxo de dados real-time para DataFrames
"""

import logging
from typing import Dict, List
from datetime import datetime, timedelta
import pandas as pd
from collections import deque
from threading import Lock

class DataIntegration:
    """Integra dados real-time do broker com estrutura de DataFrames"""
    
    def __init__(self, connection_manager, data_loader):
        self.connection_manager = connection_manager
        self.data_loader = data_loader
        self.logger = logging.getLogger('DataIntegration')
        
        # Buffer para trades (thread-safe)
        self.trades_buffer = deque(maxlen=10000)  # Últimos 10k trades
        self.buffer_lock = Lock()
        
        # DataFrames de candles
        self.candles_1min = pd.DataFrame()
        self.last_candle_time = None
        
        # Registrar callback para trades
        self.connection_manager.register_trade_callback(self._on_trade)
        
    def _on_trade(self, trade_data: Dict):
        """Callback para processar trades em tempo real"""
        try:
            # Validar dados do trade
            if not self._validate_trade(trade_data):
                return
            
            # Criar/atualizar candle
            completed_candle = self.data_loader.create_or_update_candle(trade_data)
            
            # Se um candle foi completado
            if completed_candle is not None:
                # Notificar sistema que novo candle está disponível
                self._on_candle_completed(completed_candle)
                
                # Adicionar ao buffer histórico (opcional)
                with self.buffer_lock:
                    self.trades_buffer.append(trade_data)
                    
        except Exception as e:
            self.logger.error(f"Erro processando trade: {e}")

    def _on_candle_completed(self, candle: pd.DataFrame):
        """Callback quando um candle é completado"""
        try:
            # Log
            self.logger.info(f"Novo candle formado: {candle.index[0]}")
            
            # Aqui pode-se:
            # 1. Calcular indicadores técnicos
            # 2. Atualizar features ML
            # 3. Verificar sinais de trading
            # 4. Salvar em banco de dados
            
            # Por enquanto, apenas registrar
            pass
            
        except Exception as e:
            self.logger.error(f"Erro processando candle completo: {e}")
            
    def _validate_trade(self, trade_data: Dict) -> bool:
        """Valida dados do trade"""
        required_fields = ['timestamp', 'price', 'volume', 'quantity']
        
        # Verificar campos obrigatórios
        for field in required_fields:
            if field not in trade_data:
                self.logger.error(f"Campo obrigatório ausente: {field}")
                return False
        
        # Verificar valores válidos
        if trade_data['price'] <= 0:
            self.logger.error(f"Preço inválido: {trade_data['price']}")
            return False
            
        if trade_data['volume'] < 0:
            self.logger.error(f"Volume inválido: {trade_data['volume']}")
            return False
            
        # Verificar timestamp recente (máximo 1 minuto de atraso)
        now = datetime.now()
        if isinstance(trade_data['timestamp'], str):
            trade_time = pd.to_datetime(trade_data['timestamp'])
        else:
            trade_time = trade_data['timestamp']
            
        if (now - trade_time).total_seconds() > 60:
            self.logger.warning("Trade com timestamp muito antigo ignorado")
            return False
            
        return True
    
    def _form_candle(self, start_time: datetime, end_time: datetime):
        """Forma candle de 1 minuto com trades do buffer"""
        with self.buffer_lock:
            # Filtrar trades do período
            period_trades = [
                t for t in self.trades_buffer
                if start_time <= t['timestamp'] < end_time
            ]
        
        if not period_trades:
            return
        
        # Criar candle
        prices = [t['price'] for t in period_trades]
        volumes = [t['volume'] for t in period_trades]
        
        candle = pd.DataFrame([{
            'timestamp': start_time,
            'open': prices[0],
            'high': max(prices),
            'low': min(prices),
            'close': prices[-1],
            'volume': sum(volumes),
            'trades': len(period_trades)
        }])
        
        candle.set_index('timestamp', inplace=True)
        
        # Adicionar ao DataFrame principal
        if self.candles_1min.empty:
            self.candles_1min = candle
        else:
            self.candles_1min = pd.concat([self.candles_1min, candle])
            
        # Limitar tamanho (manter últimas 1440 candles = 24h)
        if len(self.candles_1min) > 1440:
            self.candles_1min = self.candles_1min.iloc[-1440:]
        
        self.logger.debug(f"Candle formado: {start_time} - OHLC: {candle.iloc[0].to_dict()}")
    
    def get_candles(self, interval: str = '1min') -> pd.DataFrame:
        """
        Retorna candles no intervalo solicitado
        
        Args:
            interval: '1min', '5min', '15min', etc
            
        Returns:
            pd.DataFrame: Candles no intervalo solicitado
        """
        if interval == '1min':
            return self.candles_1min.copy()
        else:
            # Agregar para intervalo maior
            return self.data_loader.aggregate_candles(self.candles_1min, interval)
    
    def get_latest_candles(self, n: int = 100, interval: str = '1min') -> pd.DataFrame:
        """Retorna últimos N candles"""
        candles = self.get_candles(interval)
        return candles.tail(n)
    
    