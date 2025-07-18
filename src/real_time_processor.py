import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
from data_structure import TradingDataStructure


class RealTimeProcessor:
    """Processa dados em tempo real"""
    
    def __init__(self, data_structure: TradingDataStructure, 
                 candle_interval: int = 60):
        self.data = data_structure
        self.candle_interval = timedelta(seconds=candle_interval)
        self.logger = logging.getLogger(__name__)
        
        # Estado do candle atual
        self.current_candle = {
            'trades': [],
            'start_time': None,
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'volume': 0
        }
        
        # Estado da microestrutura atual
        self.current_micro = {
            'buy_volume': 0,
            'sell_volume': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'trades': []
        }
        
        # Buffer para evitar reprocessamento
        self.last_candle_time = None
        self.max_candles = 10000  # Limite de candles em memória
        
    def process_trade(self, trade: Dict) -> bool:
        """Processa um trade em tempo real"""
        try:
            # Validar trade
            if not self._validate_trade(trade):
                return False
            
            timestamp = pd.to_datetime(trade['timestamp'])
            candle_time = timestamp.floor('1min')
            
            # Verificar se é novo candle
            if self.last_candle_time is None or candle_time > self.last_candle_time:
                # Finalizar candle anterior se existir
                if self.last_candle_time is not None:
                    self._finalize_current_candle()
                
                # Iniciar novo candle
                self._start_new_candle(candle_time, trade)
                self.last_candle_time = candle_time
            else:
                # Atualizar candle atual
                self._update_current_candle(trade)
            
            # Atualizar microestrutura
            self._update_microstructure(trade)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro processando trade: {e}")
            return False
    
    def _validate_trade(self, trade: Dict) -> bool:
        """Valida dados do trade"""
        required_fields = ['timestamp', 'price', 'volume', 'trade_type']
        
        # Verificar campos obrigatórios
        for field in required_fields:
            if field not in trade:
                self.logger.warning(f"Campo obrigatório ausente: {field}")
                return False
        
        # Validar valores
        if trade['price'] <= 0:
            self.logger.warning(f"Preço inválido: {trade['price']}")
            return False
            
        if trade['volume'] < 0:
            self.logger.warning(f"Volume inválido: {trade['volume']}")
            return False
            
        if trade['trade_type'] not in [2, 3]:  # 2=buy, 3=sell
            self.logger.warning(f"Tipo de trade inválido: {trade['trade_type']}")
            return False
            
        return True
    
    def _start_new_candle(self, candle_time: pd.Timestamp, trade: Dict):
        """Inicia um novo candle"""
        self.current_candle = {
            'trades': [trade],
            'start_time': candle_time,
            'open': trade['price'],
            'high': trade['price'],
            'low': trade['price'],
            'close': trade['price'],
            'volume': trade['volume']
        }
        
        # Resetar microestrutura
        self.current_micro = {
            'buy_volume': trade['volume'] if trade['trade_type'] == 2 else 0,
            'sell_volume': trade['volume'] if trade['trade_type'] == 3 else 0,
            'buy_trades': 1 if trade['trade_type'] == 2 else 0,
            'sell_trades': 1 if trade['trade_type'] == 3 else 0,
            'trades': [trade]
        }
    
    def _update_current_candle(self, trade: Dict):
        """Atualiza o candle atual com novo trade"""
        self.current_candle['trades'].append(trade)
        self.current_candle['high'] = max(self.current_candle['high'], trade['price'])
        self.current_candle['low'] = min(self.current_candle['low'], trade['price'])
        self.current_candle['close'] = trade['price']
        self.current_candle['volume'] += trade['volume']
    
    def _update_microstructure(self, trade: Dict):
        """Atualiza dados de microestrutura"""
        self.current_micro['trades'].append(trade)
        
        if trade['trade_type'] == 2:  # Buy
            self.current_micro['buy_volume'] += trade['volume']
            self.current_micro['buy_trades'] += 1
        else:  # Sell
            self.current_micro['sell_volume'] += trade['volume']
            self.current_micro['sell_trades'] += 1
    
    def _finalize_current_candle(self):
        """Finaliza o candle atual e adiciona aos dados"""
        try:
            if self.current_candle['start_time'] is None:
                return
            
            # Criar novo candle
            new_candle = pd.DataFrame([{
                'open': self.current_candle['open'],
                'high': self.current_candle['high'],
                'low': self.current_candle['low'],
                'close': self.current_candle['close'],
                'volume': self.current_candle['volume']
            }], index=[self.current_candle['start_time']])
            
            # Adicionar ao DataFrame existente
            if self.data.candles.empty:
                self.data.candles = new_candle
            else:
                self.data.candles = pd.concat([self.data.candles, new_candle])
                
                # Limitar tamanho
                if len(self.data.candles) > self.max_candles:
                    self.data.candles = self.data.candles.iloc[-self.max_candles:]
            
            # Criar dados de microestrutura
            imbalance = self.current_micro['buy_volume'] - self.current_micro['sell_volume']
            total_volume = self.current_micro['buy_volume'] + self.current_micro['sell_volume']
            avg_trade_size = total_volume / len(self.current_micro['trades']) if self.current_micro['trades'] else 0
            
            new_micro = pd.DataFrame([{
                'buy_volume': self.current_micro['buy_volume'],
                'sell_volume': self.current_micro['sell_volume'],
                'buy_trades': self.current_micro['buy_trades'],
                'sell_trades': self.current_micro['sell_trades'],
                'imbalance': imbalance,
                'avg_trade_size': avg_trade_size,
                'trade_count': len(self.current_micro['trades'])
            }], index=[self.current_candle['start_time']])
            
            # Adicionar ao DataFrame existente
            if self.data.microstructure.empty:
                self.data.microstructure = new_micro
            else:
                self.data.microstructure = pd.concat([self.data.microstructure, new_micro])
                
                # Limitar tamanho
                if len(self.data.microstructure) > self.max_candles:
                    self.data.microstructure = self.data.microstructure.iloc[-self.max_candles:]
            
            self.logger.debug(f"Candle finalizado: {self.current_candle['start_time']} - "
                            f"OHLC: {self.current_candle['open']:.2f}/{self.current_candle['high']:.2f}/"
                            f"{self.current_candle['low']:.2f}/{self.current_candle['close']:.2f} "
                            f"Vol: {self.current_candle['volume']}")
            
        except Exception as e:
            self.logger.error(f"Erro finalizando candle: {e}")
    
    def force_close_candle(self):
        """Força o fechamento do candle atual (útil para timeframes personalizados)"""
        if self.current_candle['start_time'] is not None:
            self._finalize_current_candle()
            self.current_candle = {
                'trades': [],
                'start_time': None,
                'open': None,
                'high': None,
                'low': None,
                'close': None,
                'volume': 0
            }
            self.current_micro = {
                'buy_volume': 0,
                'sell_volume': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'trades': []
            }
    
    def get_current_state(self) -> Dict:
        """Retorna estado atual do processamento"""
        return {
            'last_candle_time': self.last_candle_time,
            'current_candle': {
                'start_time': self.current_candle['start_time'],
                'trades_count': len(self.current_candle['trades']),
                'volume': self.current_candle['volume']
            },
            'candles_count': len(self.data.candles),
            'micro_count': len(self.data.microstructure)
        }
    
    def get_latest_data(self, n_candles: int = 100) -> Dict[str, pd.DataFrame]:
        """Retorna os últimos n candles e dados de microestrutura"""
        return {
            'candles': self.data.candles.tail(n_candles) if not self.data.candles.empty else pd.DataFrame(),
            'microstructure': self.data.microstructure.tail(n_candles) if not self.data.microstructure.empty else pd.DataFrame()
        }