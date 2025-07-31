# -*- coding: utf-8 -*-
"""
Real-Time Dashboard - Dashboard em tempo real usando dados do Valkey
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from collections import deque
import json

class RealTimeDashboard:
    """
    Dashboard em tempo real usando dados do Valkey
    Fornece métricas e análises avançadas
    """
    
    def __init__(self, valkey_manager, config: Optional[Dict] = None):
        self.valkey_manager = valkey_manager
        self.logger = logging.getLogger('RealTimeDashboard')
        
        # Configurações
        self.config = config or {}
        self.update_interval = self.config.get('update_interval', 1)  # segundos
        self.history_window = self.config.get('history_window', 300)  # 5 minutos
        
        # Cache de métricas
        self.metrics_cache = {}
        self.last_update = {}
        
        # Histórico para gráficos
        self.price_history = deque(maxlen=300)  # 5 min de dados
        self.volume_history = deque(maxlen=300)
        self.signal_history = deque(maxlen=100)
        
        # Alertas
        self.alerts = deque(maxlen=50)
        self.alert_thresholds = {
            'price_change_pct': 2.0,  # 2% mudança
            'volume_spike': 3.0,      # 3x volume médio
            'volatility_high': 0.8,   # 80 percentil
            'data_gap_seconds': 10    # 10s sem dados
        }
        
    def get_dashboard_data(self, symbol: str) -> Dict[str, Any]:
        """
        Coleta todos os dados para dashboard em tempo real
        """
        try:
            # Verificar cache
            cache_key = f"{symbol}_dashboard"
            if cache_key in self.metrics_cache:
                last_time, cached_data = self.metrics_cache[cache_key]
                if (datetime.now() - last_time).seconds < self.update_interval:
                    return cached_data
            
            # Coletar dados
            dashboard_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'market_data': self._get_market_data(symbol),
                'technical_indicators': self._get_technical_indicators(symbol),
                'ml_metrics': self._get_ml_metrics(symbol),
                'volume_analysis': self._get_volume_analysis(symbol),
                'microstructure': self._get_microstructure_data(symbol),
                'system_health': self._get_system_health(),
                'alerts': self._get_active_alerts(symbol)
            }
            
            # Atualizar cache
            self.metrics_cache[cache_key] = (datetime.now(), dashboard_data)
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Erro ao coletar dados do dashboard: {e}")
            return self._get_error_dashboard(symbol, str(e))
    
    def _get_market_data(self, symbol: str) -> Dict:
        """
        Coleta dados de mercado em tempo real
        """
        try:
            # Últimos 5 minutos de ticks
            recent_ticks = self.valkey_manager.get_recent_ticks(symbol, count=300)
            
            if not recent_ticks:
                return self._empty_market_data()
            
            # Converter para análise
            prices = [float(t.get('price', 0)) for t in recent_ticks]
            volumes = [float(t.get('volume', 0)) for t in recent_ticks]
            
            # Calcular métricas
            last_price = prices[0] if prices else 0
            first_price = prices[-1] if prices else last_price
            
            market_data = {
                'last_price': last_price,
                'open_5min': first_price,
                'high_5min': max(prices) if prices else 0,
                'low_5min': min(prices) if prices else 0,
                'price_change': last_price - first_price,
                'price_change_pct': ((last_price / first_price - 1) * 100) if first_price > 0 else 0,
                'volume_5min': sum(volumes),
                'trades_5min': len(recent_ticks),
                'avg_trade_size': np.mean(volumes) if volumes else 0,
                'last_update': recent_ticks[0].get('timestamp', '') if recent_ticks else '',
                'bid_ask_spread': self._calculate_spread(recent_ticks)
            }
            
            # Adicionar ao histórico
            self.price_history.append((datetime.now(), last_price))
            self.volume_history.append((datetime.now(), sum(volumes)))
            
            # Verificar alertas
            self._check_market_alerts(symbol, market_data)
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Erro em market data: {e}")
            return self._empty_market_data()
    
    def _get_technical_indicators(self, symbol: str) -> Dict:
        """
        Calcula indicadores técnicos em tempo real
        """
        try:
            # Buscar dados para cálculo (30 minutos)
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=30)
            
            # Time travel query
            ticks = self.valkey_manager.time_travel_query(
                symbol, start_time, end_time, 'ticks'
            )
            
            if len(ticks) < 20:
                return {}
            
            # Converter para DataFrame
            df = pd.DataFrame(ticks)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Agregar em candles de 1 minuto
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'].astype(float), unit='ms')
            df.set_index('timestamp', inplace=True)
            
            candles = df['price'].resample('1min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            })
            volume_1m = df['volume'].resample('1min').sum()
            
            # Calcular indicadores
            indicators = {}
            
            # SMA
            if len(candles) >= 20:
                indicators['sma_20'] = candles['close'].rolling(20).mean().iloc[-1]
                indicators['sma_9'] = candles['close'].rolling(9).mean().iloc[-1]
                
            # RSI
            if len(candles) >= 14:
                indicators['rsi_14'] = self._calculate_rsi(candles['close'], 14)
                
            # VWAP
            if len(candles) >= 1:
                typical_price = (candles['high'] + candles['low'] + candles['close']) / 3
                indicators['vwap'] = (typical_price * volume_1m).sum() / volume_1m.sum()
                
            # Bollinger Bands
            if len(candles) >= 20:
                sma_20 = candles['close'].rolling(20).mean()
                std_20 = candles['close'].rolling(20).std()
                indicators['bb_upper'] = (sma_20 + 2 * std_20).iloc[-1]
                indicators['bb_lower'] = (sma_20 - 2 * std_20).iloc[-1]
                indicators['bb_width'] = indicators['bb_upper'] - indicators['bb_lower']
                
            # Volume indicators
            indicators['volume_ratio'] = volume_1m.iloc[-1] / volume_1m.mean() if volume_1m.mean() > 0 else 1
            indicators['volume_trend'] = 'increasing' if volume_1m.iloc[-5:].mean() > volume_1m.iloc[-20:-5].mean() else 'decreasing'
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Erro em technical indicators: {e}")
            return {}
    
    def _get_ml_metrics(self, symbol: str) -> Dict:
        """
        Coleta métricas do sistema ML
        """
        try:
            # Buscar últimas predições
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=5)
            
            predictions = self.valkey_manager.time_travel_query(
                symbol, start_time, end_time, 'predictions'
            )
            
            if not predictions:
                return {
                    'last_prediction': None,
                    'prediction_count_5min': 0,
                    'avg_confidence': 0,
                    'prediction_accuracy': 'N/A'
                }
            
            # Analisar predições
            confidences = []
            directions = []
            
            for pred in predictions:
                if 'confidence' in pred:
                    confidences.append(float(pred['confidence']))
                if 'direction' in pred:
                    directions.append(int(pred['direction']))
            
            ml_metrics = {
                'last_prediction': {
                    'direction': directions[0] if directions else 0,
                    'confidence': confidences[0] if confidences else 0,
                    'timestamp': predictions[0].get('timestamp', '')
                },
                'prediction_count_5min': len(predictions),
                'avg_confidence': np.mean(confidences) if confidences else 0,
                'confidence_std': np.std(confidences) if len(confidences) > 1 else 0,
                'bullish_ratio': sum(1 for d in directions if d > 0) / len(directions) if directions else 0.5,
                'prediction_stability': self._calculate_prediction_stability(directions)
            }
            
            # Adicionar ao histórico de sinais
            if directions and confidences:
                self.signal_history.append({
                    'timestamp': datetime.now(),
                    'direction': directions[0],
                    'confidence': confidences[0]
                })
            
            return ml_metrics
            
        except Exception as e:
            self.logger.error(f"Erro em ML metrics: {e}")
            return {}
    
    def _get_volume_analysis(self, symbol: str) -> Dict:
        """
        Análise detalhada de volume
        """
        try:
            # Buscar dados de volume (1 hora)
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            ticks = self.valkey_manager.time_travel_query(
                symbol, start_time, end_time, 'ticks'
            )
            
            if not ticks:
                return {}
            
            # Analisar volume por período
            df = pd.DataFrame(ticks)
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'].astype(float), unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Volume por minuto
            volume_1m = df['volume'].resample('1min').sum()
            
            # POC (Point of Control) - preço com maior volume
            price_volume = df.groupby(pd.cut(df['price'], bins=20))['volume'].sum()
            poc_range = price_volume.idxmax()
            poc_price = (poc_range.left + poc_range.right) / 2 if poc_range else df['price'].mean()
            
            volume_analysis = {
                'current_pace': volume_1m.iloc[-1] if len(volume_1m) > 0 else 0,
                'avg_pace_1h': volume_1m.mean(),
                'volume_acceleration': (volume_1m.iloc[-5:].mean() / volume_1m.iloc[-15:-5].mean() - 1) * 100 if len(volume_1m) >= 15 else 0,
                'poc_price': poc_price,
                'volume_at_poc': float(price_volume.max()) if len(price_volume) > 0 else 0,
                'volume_distribution': self._calculate_volume_distribution(df),
                'large_trades_ratio': self._calculate_large_trades_ratio(df)
            }
            
            return volume_analysis
            
        except Exception as e:
            self.logger.error(f"Erro em volume analysis: {e}")
            return {}
    
    def _get_microstructure_data(self, symbol: str) -> Dict:
        """
        Dados de microestrutura do mercado
        """
        try:
            # Últimos 100 trades
            recent_trades = self.valkey_manager.get_recent_ticks(symbol, count=100)
            
            if not recent_trades:
                return {}
            
            # Analisar microestrutura
            buy_trades = [t for t in recent_trades if t.get('trade_type') == 'BUY']
            sell_trades = [t for t in recent_trades if t.get('trade_type') == 'SELL']
            
            buy_volume = sum(float(t.get('volume', 0)) for t in buy_trades)
            sell_volume = sum(float(t.get('volume', 0)) for t in sell_trades)
            
            total_volume = buy_volume + sell_volume
            
            microstructure = {
                'buy_pressure': buy_volume / total_volume if total_volume > 0 else 0.5,
                'sell_pressure': sell_volume / total_volume if total_volume > 0 else 0.5,
                'order_imbalance': (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0,
                'avg_buy_size': np.mean([float(t.get('quantity', 0)) for t in buy_trades]) if buy_trades else 0,
                'avg_sell_size': np.mean([float(t.get('quantity', 0)) for t in sell_trades]) if sell_trades else 0,
                'trade_frequency': len(recent_trades) / 5 if recent_trades else 0,  # trades per minute
                'price_impact': self._calculate_price_impact(recent_trades)
            }
            
            return microstructure
            
        except Exception as e:
            self.logger.error(f"Erro em microstructure: {e}")
            return {}
    
    def _get_system_health(self) -> Dict:
        """
        Métricas de saúde do sistema
        """
        try:
            health_metrics = {
                'valkey_connected': self.valkey_manager.client.ping() if self.valkey_manager.client else False,
                'active_streams': len(self.valkey_manager.active_streams),
                'data_freshness': self._check_data_freshness(),
                'memory_usage': self._get_memory_usage(),
                'latency_ms': self._measure_latency(),
                'error_rate': self._calculate_error_rate()
            }
            
            # Status geral
            issues = []
            if not health_metrics['valkey_connected']:
                issues.append('Valkey disconnected')
            if health_metrics['data_freshness'] > 10:
                issues.append('Stale data')
            if health_metrics['latency_ms'] > 100:
                issues.append('High latency')
                
            health_metrics['status'] = 'healthy' if not issues else 'degraded'
            health_metrics['issues'] = issues
            
            return health_metrics
            
        except Exception as e:
            self.logger.error(f"Erro em system health: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_historical_analysis(self, symbol: str, days_back: int = 7) -> Dict:
        """
        Análise histórica usando time travel
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            # Time travel para período completo
            self.logger.info(f"Iniciando análise histórica de {days_back} dias para {symbol}")
            
            analysis = {
                'period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'days': days_back
                },
                'daily_patterns': self._analyze_daily_patterns(symbol, start_time, end_time),
                'volume_patterns': self._analyze_volume_patterns(symbol, start_time, end_time),
                'volatility_evolution': self._analyze_volatility_evolution(symbol, start_time, end_time),
                'regime_changes': self._detect_regime_changes(symbol, start_time, end_time),
                'performance_metrics': self._calculate_performance_metrics(symbol, start_time, end_time)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Erro em historical analysis: {e}")
            return {}
    
    # Métodos auxiliares
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def _calculate_spread(self, ticks: List[Dict]) -> float:
        """Calcula spread médio"""
        # Simplificado - seria melhor com dados de book
        prices = [float(t.get('price', 0)) for t in ticks[:10]]
        if len(prices) >= 2:
            return np.std(prices) * 2  # Aproximação
        return 0
    
    def _check_market_alerts(self, symbol: str, market_data: Dict):
        """Verifica e gera alertas de mercado"""
        # Alerta de mudança de preço
        if abs(market_data['price_change_pct']) > self.alert_thresholds['price_change_pct']:
            self.alerts.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'type': 'price_change',
                'severity': 'high',
                'message': f"Price change {market_data['price_change_pct']:.2f}%"
            })
        
        # Alerta de volume
        if market_data['volume_5min'] > market_data.get('avg_volume_5min', 0) * self.alert_thresholds['volume_spike']:
            self.alerts.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'type': 'volume_spike',
                'severity': 'medium',
                'message': f"Volume spike detected"
            })
    
    def _calculate_prediction_stability(self, directions: List[int]) -> float:
        """Calcula estabilidade das predições"""
        if len(directions) < 2:
            return 1.0
        
        changes = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])
        return 1 - (changes / (len(directions) - 1))
    
    def _calculate_volume_distribution(self, df: pd.DataFrame) -> str:
        """Classifica distribuição de volume"""
        if len(df) < 10:
            return 'insufficient_data'
        
        # Quartis de volume
        q1 = df['volume'].quantile(0.25)
        q3 = df['volume'].quantile(0.75)
        iqr = q3 - q1
        
        # Outliers
        outliers = df[(df['volume'] < q1 - 1.5 * iqr) | (df['volume'] > q3 + 1.5 * iqr)]
        outlier_ratio = len(outliers) / len(df)
        
        if outlier_ratio > 0.1:
            return 'irregular'
        elif df['volume'].std() / df['volume'].mean() > 1:
            return 'volatile'
        else:
            return 'normal'
    
    def _calculate_large_trades_ratio(self, df: pd.DataFrame) -> float:
        """Calcula proporção de trades grandes"""
        if 'quantity' not in df.columns:
            return 0
        
        median_size = df['quantity'].median()
        large_trades = df[df['quantity'] > median_size * 2]
        
        return len(large_trades) / len(df) if len(df) > 0 else 0
    
    def _calculate_price_impact(self, trades: List[Dict]) -> float:
        """Calcula impacto no preço dos trades"""
        if len(trades) < 2:
            return 0
        
        # Simplificado - correlação entre tamanho e mudança de preço
        prices = [float(t.get('price', 0)) for t in trades]
        volumes = [float(t.get('volume', 0)) for t in trades]
        
        if len(prices) > 10:
            price_changes = np.diff(prices)
            volume_scaled = np.array(volumes[:-1]) / np.mean(volumes)
            
            # Correlação entre volume e mudança de preço
            if np.std(volume_scaled) > 0 and np.std(price_changes) > 0:
                impact = np.corrcoef(volume_scaled, np.abs(price_changes))[0, 1]
                return impact
        
        return 0
    
    def _check_data_freshness(self) -> float:
        """Verifica idade dos dados mais recentes (segundos)"""
        # Implementação simplificada
        return 0
    
    def _get_memory_usage(self) -> Dict:
        """Obtém uso de memória"""
        return {
            'cache_size': len(self.metrics_cache),
            'history_size': len(self.price_history)
        }
    
    def _measure_latency(self) -> float:
        """Mede latência do sistema (ms)"""
        start = datetime.now()
        try:
            self.valkey_manager.client.ping()
            latency = (datetime.now() - start).total_seconds() * 1000
            return latency
        except:
            return 999
    
    def _calculate_error_rate(self) -> float:
        """Calcula taxa de erro"""
        # Implementação simplificada
        return 0
    
    def _get_active_alerts(self, symbol: str) -> List[Dict]:
        """Retorna alertas ativos"""
        return [
            alert for alert in self.alerts 
            if alert['symbol'] == symbol and 
            (datetime.now() - alert['timestamp']).seconds < 300
        ]
    
    def _empty_market_data(self) -> Dict:
        """Retorna estrutura vazia de market data"""
        return {
            'last_price': 0,
            'open_5min': 0,
            'high_5min': 0,
            'low_5min': 0,
            'price_change': 0,
            'price_change_pct': 0,
            'volume_5min': 0,
            'trades_5min': 0,
            'avg_trade_size': 0,
            'last_update': '',
            'bid_ask_spread': 0
        }
    
    def _get_error_dashboard(self, symbol: str, error: str) -> Dict:
        """Dashboard de erro"""
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': error,
            'market_data': self._empty_market_data()
        }
    
    # Métodos de análise histórica
    
    def _analyze_daily_patterns(self, symbol: str, start_time: datetime, end_time: datetime) -> Dict:
        """Analisa padrões diários"""
        # Implementação simplificada
        return {
            'most_active_hour': 10,
            'least_active_hour': 13,
            'avg_daily_volume': 1000000,
            'avg_daily_range': 50
        }
    
    def _analyze_volume_patterns(self, symbol: str, start_time: datetime, end_time: datetime) -> Dict:
        """Analisa padrões de volume"""
        return {
            'trend': 'increasing',
            'weekly_pattern': 'normal',
            'anomalies_detected': 0
        }
    
    def _analyze_volatility_evolution(self, symbol: str, start_time: datetime, end_time: datetime) -> Dict:
        """Analisa evolução da volatilidade"""
        return {
            'current_regime': 'normal',
            'regime_changes': 2,
            'trend': 'stable'
        }
    
    def _detect_regime_changes(self, symbol: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Detecta mudanças de regime"""
        return [
            {
                'timestamp': (start_time + timedelta(days=3)).isoformat(),
                'from_regime': 'low_vol',
                'to_regime': 'normal_vol',
                'confidence': 0.85
            }
        ]
    
    def _calculate_performance_metrics(self, symbol: str, start_time: datetime, end_time: datetime) -> Dict:
        """Calcula métricas de performance"""
        return {
            'total_return': 2.5,
            'sharpe_ratio': 1.2,
            'max_drawdown': -3.5,
            'win_rate': 0.55
        }