# src/backtesting/ml_backtester.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json

# Imports para feature generation
from data_structure import TradingDataStructure
from feature_engine import FeatureEngine

class BacktestMode(Enum):
    """Modos de backtesting"""
    SIMPLE = "simple"          # Sem custos/slippage
    REALISTIC = "realistic"    # Com custos realistas
    CONSERVATIVE = "conservative"  # Custos conservadores
    STRESS = "stress"         # Condições adversas

@dataclass
class BacktestConfig:
    """Configuração do backtest"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    commission_per_contract: float = 0.50
    slippage_ticks: int = 1
    tick_value: float = 10.0  # Valor do tick em reais
    margin_per_contract: float = 150.0
    max_positions: int = 3
    mode: BacktestMode = BacktestMode.REALISTIC
    use_intraday_data: bool = True
    include_market_impact: bool = True

@dataclass
class Trade:
    """Estrutura de um trade"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    commission: float
    slippage: float
    pnl: float = 0.0
    return_pct: float = 0.0
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion
    duration: Optional[timedelta] = None
    exit_reason: Optional[str] = None

class AdvancedMLBacktester:
    """Sistema avançado de backtesting para estratégias ML"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Estado do backtest
        self.current_capital = config.initial_capital
        self.positions = {}  # symbol -> position
        self.trades = []     # histórico de trades
        self.equity_curve = []
        self.drawdown_series = []
        
        # Componentes
        self.market_simulator = None
        self.cost_model = None
        self.ml_models = None
        self.feature_engine = None
        
        # Métricas
        self.metrics = {}
        self.detailed_stats = {}
        
    def _ensure_datetime(self, timestamp) -> datetime:
        """
        Garante que o timestamp seja um datetime Python válido
        
        Args:
            timestamp: Timestamp de qualquer tipo (datetime, pd.Timestamp, string, etc.)
            
        Returns:
            datetime: Timestamp convertido para datetime Python
        """
        # Se já é datetime Python nativo, retornar diretamente
        if isinstance(timestamp, datetime):
            return timestamp
        
        try:
            # Se é pandas Timestamp, converter para datetime Python
            if hasattr(timestamp, 'to_pydatetime'):
                return timestamp.to_pydatetime()
            
            # Se tem método .to_datetime() (alguns tipos customizados)
            if hasattr(timestamp, 'to_datetime'):
                result = timestamp.to_datetime()
                if isinstance(result, datetime):
                    return result
            
            # Se é string ou outro tipo, tentar converter via pandas
            try:
                # Usar pandas Timestamp para conversão robusta
                pd_timestamp = pd.Timestamp(timestamp)
                return pd_timestamp.to_pydatetime()
            except (ValueError, TypeError):
                # Fallback: tentar como string
                pd_timestamp = pd.Timestamp(str(timestamp))
                return pd_timestamp.to_pydatetime()
                
        except Exception as e:
            # Último recurso: usar timestamp atual
            self.logger.warning(f"Erro convertendo timestamp {timestamp}: {e}. Usando timestamp atual.")
            return datetime.now()
    
    def initialize(self, ml_models, feature_engine):
        """Inicializa componentes do backtester"""
        self.ml_models = ml_models
        self.feature_engine = feature_engine
        
        # Criar simuladores
        from market_simulator import MarketSimulator
        from cost_model import CostModel
        
        self.market_simulator = MarketSimulator(self.config)
        self.cost_model = CostModel(self.config)
        
        self.logger.info(f"Backtester inicializado - Modo: {self.config.mode.value}")
        
    def run_backtest(self, historical_data: pd.DataFrame, 
                    ml_strategy_params: Optional[Dict] = None) -> Dict:
        """
        Executa backtest completo
        
        Args:
            historical_data: DataFrame com dados históricos
            ml_strategy_params: Parâmetros da estratégia ML
            
        Returns:
            Resultados completos do backtest
        """
        self.logger.info(f"Iniciando backtest de {self.config.start_date} até {self.config.end_date}")
        
        # Resetar estado
        self._reset_state()
        
        # Filtrar dados pelo período
        mask = (historical_data.index >= self.config.start_date) & \
               (historical_data.index <= self.config.end_date)
        backtest_data = historical_data[mask].copy()
        
        if len(backtest_data) == 0:
            raise ValueError("Sem dados para o período especificado")
        
        # Processar cada barra
        process_count = 0
        for timestamp, market_data in backtest_data.iterrows():
            try:
                process_count += 1
                # Converter timestamp para datetime Python
                current_timestamp = self._ensure_datetime(timestamp)
                
                # Atualizar simulador de mercado
                self.market_simulator.update(current_timestamp, market_data)
                
                # Atualizar MAE/MFE das posições abertas
                self._update_open_positions(market_data)
                
                # Verificar stops
                self._check_stops(market_data)
                
                # Gerar features
                features = self._generate_features(backtest_data.loc[:timestamp])
                
                if features is not None:
                    # Obter predição ML
                    ml_signal = self._get_ml_prediction(features, market_data)
                    
                    # Debug da predição
                    if process_count % 100 == 0:  # Log a cada 100 registros
                        self.logger.info(f"Timestamp {timestamp}: Features shape={features.shape if hasattr(features, 'shape') else 'N/A'}, ML Signal={ml_signal}")
                    
                    # Processar sinal
                    if ml_signal['action'] != 'none':
                        self.logger.info(f"SINAL GERADO: {ml_signal} em {timestamp}")
                        self._process_signal(ml_signal, market_data, current_timestamp)
                else:
                    if process_count % 500 == 0:  # Log menos frequente para features None
                        self.logger.info(f"Features None em {timestamp}")
                
                # Atualizar equity
                self._update_equity(market_data, current_timestamp)
                
            except Exception as e:
                self.logger.error(f"Erro processando {timestamp}: {e}")
        
        # Fechar posições abertas
        self._close_all_positions(backtest_data.iloc[-1], "end_of_backtest")
        
        # Calcular métricas finais
        results = self._calculate_final_metrics()
        
        # Análise adicional
        results['trade_analysis'] = self._analyze_trades()
        results['drawdown_analysis'] = self._analyze_drawdowns()
        results['regime_analysis'] = self._analyze_by_regime(backtest_data)
        
        self.logger.info(f"Backtest concluído - {len(self.trades)} trades executados")
        
        return results
    
    def _reset_state(self):
        """Reseta estado do backtester"""
        self.current_capital = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.drawdown_series = []
        self.metrics = {}
        
    def _generate_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Gera features seguindo o fluxo do mapa de dados:
        1. Identifica features necessárias dos modelos ML
        2. Cria DataFrame de candles
        3. Calcula features necessárias usando FeatureEngine
        4. Retorna features formatadas para predição
        """
        try:
            if len(data) < 100:  # Mínimo de dados para features
                return None
            
            # Etapa 1: Identificar features necessárias dos modelos
            required_features = self._get_required_features_from_models()
            if not required_features:
                self.logger.error("Não foi possível identificar features dos modelos")
                return None
            
            # Etapa 2: Preparar DataFrame de candles
            candles_df = self._prepare_candles_dataframe(data)
            if candles_df is None or len(candles_df) < 50:
                self.logger.warning("DataFrame de candles insuficiente para features")
                return None
            
            # Etapa 3: Calcular features usando engine de features
            features_df = self._calculate_features_from_candles(candles_df, required_features)
            
            if features_df is not None and len(features_df) > 0:
                # Retornar apenas a última linha (features atuais)
                return features_df.iloc[-1:]
            else:
                self.logger.warning("Não foi possível calcular features")
                return None
                
        except Exception as e:
            self.logger.error(f"Erro gerando features: {e}")
            return None
    
    def _get_required_features_from_models(self) -> List[str]:
        """Extrai features necessárias dos modelos ML carregados"""
        required_features = []
        
        try:
            if not self.ml_models or len(self.ml_models) == 0:
                # Features conhecidas dos modelos treinados
                return [
                    'high_low_range_20', 'ema_200', 'high_low_range_10', 'bb_upper_50', 'bb_lower_50',
                    'volatility_50', 'ema_50', 'high_low_range_5', 'parkinson_vol_10', 'parkinson_vol_20',
                    'bb_lower_20', 'bb_middle_50', 'bb_upper_20', 'volatility_20', 'vwap',
                    'ema_20', 'volatility_20_lag_1', 'bb_middle_20', 'volatility_20_lag_10', 'volatility_20_lag_5',
                    'gk_vol_20', 'gk_vol_10', 'atr', 'range_percent', 'ema_9',
                    'volatility_10', 'atr_20', 'adx', 'bb_width_50', 'bb_width_20'
                ]
            
            # Extrair features dos modelos carregados
            all_features = set()
            for name, model in self.ml_models.items():
                try:
                    if hasattr(model, 'feature_names_in_'):
                        model_features = list(model.feature_names_in_)
                        all_features.update(model_features)
                        self.logger.debug(f"Modelo {name}: {len(model_features)} features")
                except Exception as model_error:
                    self.logger.warning(f"Erro extraindo features do modelo {name}: {model_error}")
            
            required_features = list(all_features)
            self.logger.info(f"Features necessárias identificadas: {len(required_features)}")
            
            return required_features
            
        except Exception as e:
            self.logger.error(f"Erro identificando features dos modelos: {e}")
            return []
    
    def _prepare_candles_dataframe(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepara DataFrame de candles com colunas OHLCV padronizadas"""
        try:
            candles = data.copy()
            
            # Garantir colunas básicas
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in candles.columns:
                    self.logger.error(f"Coluna {col} não encontrada nos dados")
                    return None
            
            # Converter para numérico
            for col in required_cols:
                candles[col] = pd.to_numeric(candles[col], errors='coerce')
            
            # Remover NaN
            candles = candles.dropna(subset=required_cols)
            
            # Ordenar por timestamp
            candles = candles.sort_index()
            
            self.logger.debug(f"DataFrame candles preparado: {len(candles)} registros")
            return candles
            
        except Exception as e:
            self.logger.error(f"Erro preparando DataFrame de candles: {e}")
            return None
    
    def _calculate_features_from_candles(self, candles_df: pd.DataFrame, 
                                       required_features: List[str]) -> Optional[pd.DataFrame]:
        """Calcula features específicas necessárias a partir dos candles"""
        try:
            # Por enquanto, usar sempre cálculo manual para garantir funcionamento
            return self._calculate_manual_features(candles_df, required_features)
                
        except Exception as e:
            self.logger.error(f"Erro calculando features: {e}")
            return None
    
    def _use_feature_engine(self, candles_df: pd.DataFrame, 
                           required_features: List[str]) -> Optional[pd.DataFrame]:
        """Usa FeatureEngine para calcular features"""
        try:
            # Preparar estrutura de dados para FeatureEngine
            data_structure = self._prepare_data_structure_for_engine(candles_df)
            
            # Tentar usar extract_all_features se existe
            if hasattr(self.feature_engine, 'extract_all_features'):
                features_result = self.feature_engine.extract_all_features(
                    candles_df=data_structure.candles,
                    microstructure_df=data_structure.microstructure
                )
            elif hasattr(self.feature_engine, 'create_features_separated'):
                # Método alternativo
                features_result = self.feature_engine.create_features_separated(
                    data_structure.candles,
                    data_structure.microstructure,
                    data_structure.indicators
                )
            else:
                self.logger.warning("FeatureEngine não tem métodos conhecidos")
                return None
            
            if features_result is not None and len(features_result) > 0:
                # Selecionar apenas features necessárias
                available_features = [f for f in required_features if f in features_result.columns]
                if available_features:
                    return features_result[available_features]
                else:
                    self.logger.warning("Nenhuma feature necessária encontrada no resultado do FeatureEngine")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro usando FeatureEngine: {e}")
            return None
    
    def _prepare_data_structure_for_engine(self, candles_df: pd.DataFrame):
        """Prepara estrutura de dados para FeatureEngine"""
        try:
            from data_structure import TradingDataStructure
            
            data_structure = TradingDataStructure()
            data_structure.initialize_structure()
            data_structure.candles = candles_df.copy()
            
            # Adicionar microstructure básica se não existe
            if 'buy_volume' not in candles_df.columns:
                micro_data = pd.DataFrame(index=candles_df.index)
                micro_data['buy_volume'] = candles_df['volume'] * 0.5  # Assumir distribuição igual
                micro_data['sell_volume'] = candles_df['volume'] * 0.5
                micro_data['trades'] = 100  # Valor padrão
                data_structure.microstructure = micro_data
            
            return data_structure
            
        except Exception as e:
            self.logger.error(f"Erro preparando estrutura para FeatureEngine: {e}")
            
            # Fallback: objeto mock
            class MockDataStructure:
                def __init__(self, candles):
                    self.candles = candles
                    self.microstructure = pd.DataFrame()
                    self.indicators = pd.DataFrame()
            
            return MockDataStructure(candles_df)
    
    def _calculate_manual_features(self, candles_df: pd.DataFrame, 
                                 required_features: List[str]) -> Optional[pd.DataFrame]:
        """Calcula features manualmente quando FeatureEngine não disponível"""
        try:
            df = candles_df.copy()
            self.logger.info(f"Calculando {len(required_features)} features manualmente")
            
            # Converter para numérico
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 1. EMAs (Exponential Moving Averages)
            if 'ema_9' in required_features:
                df['ema_9'] = df['close'].ewm(span=9).mean()
            if 'ema_20' in required_features:
                df['ema_20'] = df['close'].ewm(span=20).mean()
            if 'ema_50' in required_features:
                df['ema_50'] = df['close'].ewm(span=50).mean()
            if 'ema_200' in required_features:
                df['ema_200'] = df['close'].ewm(span=200).mean()
            
            # 2. ATR (Average True Range)
            if any('atr' in f for f in required_features):
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift(1))
                low_close = abs(df['low'] - df['close'].shift(1))
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                
                if 'atr' in required_features:
                    df['atr'] = true_range.rolling(14).mean()
                if 'atr_20' in required_features:
                    df['atr_20'] = true_range.rolling(20).mean()
            
            # 3. ADX (Average Directional Index)
            if 'adx' in required_features:
                df['adx'] = self._calculate_adx(df)
            
            # 4. Bollinger Bands
            if any('bb_' in f for f in required_features):
                self._calculate_bollinger_bands(df, required_features)
            
            # 5. VWAP
            if 'vwap' in required_features:
                df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            # 6. Volatilidade
            if any('volatility' in f for f in required_features):
                self._calculate_volatilities(df, required_features)
            
            # 7. High-Low Range
            if any('high_low_range' in f for f in required_features):
                df['high_low_range_5'] = (df['high'].rolling(5).max() - df['low'].rolling(5).min()) / df['close']
                df['high_low_range_10'] = (df['high'].rolling(10).max() - df['low'].rolling(10).min()) / df['close']
                df['high_low_range_20'] = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']
            
            # 8. Parkinson Volatility
            if any('parkinson_vol' in f for f in required_features):
                log_ratio = pd.Series(np.log(df['high']/df['low']), index=df.index)
                df['parkinson_vol_10'] = np.sqrt((1/(4*np.log(2))) * log_ratio.rolling(10).var())
                df['parkinson_vol_20'] = np.sqrt((1/(4*np.log(2))) * log_ratio.rolling(20).var())
            
            # 9. Garman-Klass Volatility
            if any('gk_vol' in f for f in required_features):
                df['gk_vol_10'] = self._calculate_gk_volatility(df, 10)
                df['gk_vol_20'] = self._calculate_gk_volatility(df, 20)
            
            # 10. Range Percent
            if 'range_percent' in required_features:
                df['range_percent'] = (df['high'] - df['low']) / df['close']
            
            # Limpar dados
            df = df.fillna(0)
            df = df.replace([np.inf, -np.inf], 0)
            
            # Selecionar apenas features necessárias
            available_features = [f for f in required_features if f in df.columns]
            
            if available_features:
                features_df = df[available_features].copy()
                self.logger.info(f"Features calculadas com sucesso: {len(available_features)}/{len(required_features)}")
                return features_df
            else:
                self.logger.warning("Nenhuma feature necessária foi calculada")
                return None
                
        except Exception as e:
            self.logger.error(f"Erro calculando features manuais: {e}")
            return None
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula ADX (Average Directional Index)"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            dm_up = high - high.shift(1)
            dm_down = low.shift(1) - low
            
            dm_up = dm_up.where((dm_up > dm_down) & (dm_up > 0), 0)
            dm_down = dm_down.where((dm_down > dm_up) & (dm_down > 0), 0)
            
            # Smoothed averages
            atr = true_range.rolling(period).mean()
            di_up = 100 * (dm_up.rolling(period).mean() / atr)
            di_down = 100 * (dm_down.rolling(period).mean() / atr)
            
            # ADX calculation
            dx = 100 * abs(di_up - di_down) / (di_up + di_down)
            adx = dx.rolling(period).mean()
            
            return adx.fillna(0)
            
        except Exception:
            return pd.Series(0, index=df.index)
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, required_features: List[str]):
        """Calcula Bollinger Bands para períodos necessários"""
        try:
            for period in [20, 50]:
                if any(f'bb_' in f and f'{period}' in f for f in required_features):
                    sma = df['close'].rolling(period).mean()
                    std = df['close'].rolling(period).std()
                    
                    df[f'bb_upper_{period}'] = sma + (2 * std)
                    df[f'bb_middle_{period}'] = sma
                    df[f'bb_lower_{period}'] = sma - (2 * std)
                    df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / df[f'bb_middle_{period}']
                    
        except Exception as e:
            self.logger.warning(f"Erro calculando Bollinger Bands: {e}")
    
    def _calculate_volatilities(self, df: pd.DataFrame, required_features: List[str]):
        """Calcula diferentes tipos de volatilidade"""
        try:
            returns = df['close'].pct_change()
            
            # Volatilidade básica para diferentes períodos
            for period in [10, 20, 50]:
                if f'volatility_{period}' in required_features:
                    df[f'volatility_{period}'] = returns.rolling(period).std()
                
                # Volatilidade com lag
                if f'volatility_{period}_lag_1' in required_features:
                    df[f'volatility_{period}_lag_1'] = df[f'volatility_{period}'].shift(1)
                if f'volatility_{period}_lag_5' in required_features:
                    df[f'volatility_{period}_lag_5'] = df[f'volatility_{period}'].shift(5)
                if f'volatility_{period}_lag_10' in required_features:
                    df[f'volatility_{period}_lag_10'] = df[f'volatility_{period}'].shift(10)
                    
        except Exception as e:
            self.logger.warning(f"Erro calculando volatilidades: {e}")
    
    def _calculate_gk_volatility(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calcula Garman-Klass Volatility"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            open_price = df['open']
            
            # Garman-Klass formula
            gk = 0.5 * np.log(high/low)**2 - (2*np.log(2)-1) * np.log(close/open_price)**2
            
            return np.sqrt(gk.rolling(period).mean()).fillna(0)
            
        except Exception:
            return pd.Series(0, index=df.index)
    
    def _calculate_simple_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calcula features simples para backtest sem validações rigorosas"""
        try:
            df = data.copy()
            
            # Converter colunas para numérico
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Features técnicas básicas
            # EMAs
            df['ema_9'] = df['close'].ewm(span=9).mean()
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            
            # Price momentum
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_10'] = df['close'].pct_change(10)
            df['momentum_20'] = df['close'].pct_change(20)
            
            # Volatility
            df['volatility_10'] = df['close'].rolling(10).std()
            df['volatility_20'] = df['close'].rolling(20).std()
            
            # Range features
            df['range_hl'] = (df['high'] - df['low']) / df['close']
            df['range_oc'] = (df['open'] - df['close']).abs() / df['close']
            
            # Return features  
            df['return_1'] = df['close'].pct_change(1)
            df['return_5'] = df['close'].pct_change(5)
            df['return_10'] = df['close'].pct_change(10)
            
            # Volume features (se disponível)
            if 'volume' in df.columns:
                df['volume_ma'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma'].where(df['volume_ma'] > 0, 1)
            else:
                df['volume_ma'] = 1000
                df['volume_ratio'] = 1.0
            
            # Microstructure features (se disponível)
            if 'buy_volume' in df.columns and 'sell_volume' in df.columns:
                total_vol = df['buy_volume'] + df['sell_volume']
                df['buy_pressure'] = df['buy_volume'] / total_vol.where(total_vol > 0, 1)
                df['flow_imbalance'] = (df['buy_volume'] - df['sell_volume']) / df.get('volume', total_vol)
            else:
                df['buy_pressure'] = 0.5  # Neutro se não há dados
                df['flow_imbalance'] = 0.0
            
            # Trend indicators  
            df['trend_ema'] = np.where(df['ema_9'] > df['ema_20'], 1, 
                                     np.where(df['ema_9'] < df['ema_20'], -1, 0))
            
            # Price relative to EMAs
            df['price_vs_ema9'] = df['close'] / df['ema_9'].where(df['ema_9'] > 0, df['close']) - 1
            df['price_vs_ema20'] = df['close'] / df['ema_20'].where(df['ema_20'] > 0, df['close']) - 1
            df['price_vs_ema50'] = df['close'] / df['ema_50'].where(df['ema_50'] > 0, df['close']) - 1
            
            # Simple RSI
            delta = df['close'].astype(float).diff()
            gains = delta.where(delta > 0, 0.0).rolling(14).mean()
            losses = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gains / losses.where(losses > 0, 1.0)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Limpar dados
            df = df.fillna(0)
            df = df.replace([np.inf, -np.inf], 0)
            
            # Selecionar apenas features (excluir OHLCV originais)
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume', 'quantidade', 'trades']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            feature_df = df[feature_cols].copy()
            
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Erro calculando features simples: {e}")
            return None
    
    def _prepare_data_structure(self, data: pd.DataFrame):
        """Prepara estrutura de dados para o feature_engine"""
        try:
            # Importar TradingDataStructure
            from data_structure import TradingDataStructure
            
            # Criar estrutura de dados
            data_structure = TradingDataStructure()
            data_structure.initialize_structure()
            
            # Preencher com dados de candles
            data_structure.candles = data.copy()
            
            # Se temos colunas de microestrutura, preencher também
            if 'buy_volume' in data.columns and 'sell_volume' in data.columns:
                micro_data = data[['buy_volume', 'sell_volume']].copy()
                if 'trades' in data.columns:
                    micro_data['trades'] = data['trades']
                data_structure.microstructure = micro_data
            
            return data_structure
            
        except Exception as e:
            self.logger.error(f"Erro preparando estrutura de dados: {e}")
            # Fallback: retornar o DataFrame original
            # Criar um objeto mock com atributo candles
            class MockDataStructure:
                def __init__(self, candles_df):
                    self.candles = candles_df
                    self.microstructure = pd.DataFrame()
                    self.indicators = pd.DataFrame()
            
            return MockDataStructure(data)
    
    def _get_ml_prediction(self, features: pd.DataFrame, 
                          market_data: pd.Series) -> Dict:
        """Obtém predição dos modelos ML"""
        try:
            # Verificar se temos modelos
            if not self.ml_models or len(self.ml_models) == 0:
                # DEBUG: Log quando não há modelos
                self.logger.warning(f"Nenhum modelo ML carregado. ml_models={self.ml_models}")
                # Retornar sinal neutro se não há modelos
                return {
                    'action': 'none', 
                    'confidence': 0,
                    'prediction': np.array([0.33, 0.34, 0.33]),  # Distribuição uniforme
                    'symbol': market_data.get('contract', 'WDOH25'),
                    'price': market_data['close']
                }
            
            # Preparar dados para modelos
            model_features = self._prepare_model_features(features)
            
            # Verificar se temos features válidas
            if model_features.empty:
                return {
                    'action': 'none',
                    'confidence': 0,
                    'prediction': np.array([0.33, 0.34, 0.33]),
                    'symbol': market_data.get('contract', 'WDOH25'),
                    'price': market_data['close']
                }
            
            # Obter predições do ensemble
            predictions = {}
            for name, model in self.ml_models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(model_features)[0]
                    else:
                        pred = model.predict(model_features)[0]
                    predictions[name] = pred
                except Exception as model_error:
                    self.logger.warning(f"Erro predição modelo {name}: {model_error}")
                    continue
            
            # Se nenhum modelo conseguiu fazer predição
            if not predictions:
                return {
                    'action': 'none',
                    'confidence': 0,
                    'prediction': np.array([0.33, 0.34, 0.33]),
                    'symbol': market_data.get('contract', 'WDOH25'),
                    'price': market_data['close']
                }
            
            # Combinar predições (voting ou média ponderada)
            final_prediction = self._combine_predictions(predictions)
            
            # Converter para sinal de trading
            signal = self._prediction_to_signal(final_prediction, market_data)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Erro na predição ML: {e}")
            return {
                'action': 'none', 
                'confidence': 0,
                'prediction': np.array([0.33, 0.34, 0.33]),
                'symbol': market_data.get('contract', 'WDOH25'),
                'price': market_data['close']
            }
    
    def _prediction_to_signal(self, prediction: np.ndarray, 
                            market_data: pd.Series) -> Dict:
        """Converte predição ML em sinal de trading"""
        # Assumindo 3 classes: 0=sell, 1=hold, 2=buy
        pred_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Threshold de confiança
        if confidence < 0.6:
            return {
                'action': 'none', 
                'confidence': confidence,
                'prediction': prediction,
                'symbol': market_data.get('contract', 'WDOH25'),
                'price': market_data.get('close', 0)
            }
        
        # Mapear para ação
        if pred_class == 2:
            action = 'buy'
        elif pred_class == 0:
            action = 'sell'
        else:
            action = 'none'
        
        return {
            'action': action,
            'confidence': confidence,
            'prediction': prediction,
            'symbol': market_data.get('contract', 'WDOH25'),
            'price': market_data.get('close', 0)
        }
    
    def _process_signal(self, signal: Dict, market_data: pd.Series, 
                       timestamp: datetime):
        """Processa sinal de trading"""
        symbol = signal['symbol']
        
        # Verificar se já tem posição
        if symbol in self.positions:
            current_pos = self.positions[symbol]
            
            # Verificar se deve reverter posição
            if (current_pos['side'] == 'long' and signal['action'] == 'sell') or \
               (current_pos['side'] == 'short' and signal['action'] == 'buy'):
                # Fechar posição atual
                self._close_position(symbol, market_data, timestamp, "signal_reversal")
                # Abrir nova posição
                self._open_position(signal, market_data, timestamp)
        else:
            # Verificar limites de posição
            if len(self.positions) < self.config.max_positions:
                self._open_position(signal, market_data, timestamp)
    
    def _open_position(self, signal: Dict, market_data: pd.Series, 
                      timestamp: datetime):
        """Abre nova posição"""
        symbol = signal['symbol']
        side = 'long' if signal['action'] == 'buy' else 'short'
        
        # Calcular custos
        entry_price = self._calculate_entry_price(
            market_data['close'], side, market_data
        )
        
        # Calcular tamanho da posição
        position_size = self._calculate_position_size(signal, market_data)
        
        # Comissão
        commission = self.cost_model.calculate_commission(position_size)
        
        # Criar trade
        trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=None,
            quantity=position_size,
            commission=commission,
            slippage=abs(entry_price - market_data['close']) * position_size
        )
        
        # Adicionar posição
        self.positions[symbol] = {
            'trade': trade,
            'current_price': entry_price,
            'mae': 0,
            'mfe': 0,
            'stop_loss': self._calculate_stop_loss(entry_price, side),
            'take_profit': self._calculate_take_profit(entry_price, side)
        }
        
        # Deduzir capital
        margin_required = position_size * self.config.margin_per_contract
        self.current_capital -= (commission + margin_required)
        
        self.logger.debug(f"Posição aberta: {symbol} {side} @ {entry_price}")
    
    def _close_position(self, symbol: str, market_data: pd.Series,
                       timestamp: datetime, reason: str):
        """Fecha posição existente"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        trade = position['trade']
        
        # Calcular preço de saída
        exit_price = self._calculate_exit_price(
            market_data['close'], trade.side, market_data
        )
        
        # Comissão de saída
        exit_commission = self.cost_model.calculate_commission(trade.quantity)
        
        # Atualizar trade
        trade.exit_time = timestamp
        trade.exit_price = exit_price
        trade.commission += exit_commission
        trade.exit_reason = reason
        trade.duration = timestamp - trade.entry_time
        
        # Calcular PnL
        if trade.side == 'long':
            gross_pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            gross_pnl = (trade.entry_price - exit_price) * trade.quantity
        
        trade.pnl = gross_pnl - trade.commission - trade.slippage
        trade.return_pct = trade.pnl / (trade.entry_price * trade.quantity)
        
        # MAE/MFE finais
        trade.mae = position['mae']
        trade.mfe = position['mfe']
        
        # Adicionar aos trades completados
        self.trades.append(trade)
        
        # Liberar margem
        margin_released = trade.quantity * self.config.margin_per_contract
        self.current_capital += (margin_released + trade.pnl)
        
        # Remover posição
        del self.positions[symbol]
        
        self.logger.debug(
            f"Posição fechada: {symbol} @ {exit_price} - PnL: {trade.pnl:.2f}"
        )
    
    def _update_open_positions(self, market_data: pd.Series):
        """Atualiza MAE/MFE das posições abertas"""
        for symbol, position in self.positions.items():
            trade = position['trade']
            current_price = market_data.get('close', position['current_price'])
            
            # Calcular excursão atual
            if trade.side == 'long':
                excursion = current_price - trade.entry_price
            else:
                excursion = trade.entry_price - current_price
            
            # Atualizar MAE/MFE
            if excursion < 0:
                position['mae'] = min(position['mae'], excursion)
            else:
                position['mfe'] = max(position['mfe'], excursion)
            
            position['current_price'] = current_price
    
    def _check_stops(self, market_data: pd.Series):
        """Verifica stops das posições"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            current_price = market_data.get('close', position['current_price'])
            
            # Verificar stop loss
            if position['stop_loss']:
                if (position['trade'].side == 'long' and 
                    current_price <= position['stop_loss']) or \
                   (position['trade'].side == 'short' and 
                    current_price >= position['stop_loss']):
                    positions_to_close.append((symbol, 'stop_loss'))
            
            # Verificar take profit
            if position['take_profit']:
                if (position['trade'].side == 'long' and 
                    current_price >= position['take_profit']) or \
                   (position['trade'].side == 'short' and 
                    current_price <= position['take_profit']):
                    positions_to_close.append((symbol, 'take_profit'))
        
        # Fechar posições atingidas
        for symbol, reason in positions_to_close:
            # Garantir que temos um timestamp válido
            timestamp = self._ensure_datetime(market_data.name)
            self._close_position(symbol, market_data, timestamp, reason)
    
    def _update_equity(self, market_data: pd.Series, timestamp: datetime):
        """Atualiza curva de equity"""
        # Calcular valor das posições abertas
        open_pnl = 0
        for symbol, position in self.positions.items():
            trade = position['trade']
            current_price = market_data.get('close', position['current_price'])
            
            if trade.side == 'long':
                open_pnl += (current_price - trade.entry_price) * trade.quantity
            else:
                open_pnl += (trade.entry_price - current_price) * trade.quantity
        
        # Equity total
        total_equity = self.current_capital + open_pnl
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'cash': self.current_capital,
            'open_pnl': open_pnl
        })
        
        # Calcular drawdown
        if self.equity_curve:
            peak = max(e['equity'] for e in self.equity_curve)
            drawdown = (total_equity - peak) / peak
            self.drawdown_series.append({
                'timestamp': timestamp,
                'drawdown': drawdown
            })
    
    def _close_all_positions(self, market_data: pd.Series, reason: str):
        """Fecha todas as posições abertas"""
        symbols_to_close = list(self.positions.keys())
        # Garantir que temos um timestamp válido
        timestamp = self._ensure_datetime(market_data.name)
        for symbol in symbols_to_close:
            self._close_position(symbol, market_data, timestamp, reason)
    
    def _calculate_entry_price(self, base_price: float, side: str,
                             market_data: pd.Series) -> float:
        """Calcula preço de entrada com slippage"""
        slippage = self.cost_model.calculate_slippage(
            base_price, side, market_data
        )
        
        if side == 'long':
            return base_price + slippage
        else:
            return base_price - slippage
    
    def _calculate_exit_price(self, base_price: float, side: str,
                            market_data: pd.Series) -> float:
        """Calcula preço de saída com slippage"""
        # Inverter side para saída
        exit_side = 'short' if side == 'long' else 'long'
        return self._calculate_entry_price(base_price, exit_side, market_data)
    
    def _calculate_position_size(self, signal: Dict, 
                               market_data: pd.Series) -> int:
        """Calcula tamanho da posição"""
        # Implementação simples - pode ser melhorada com Kelly Criterion
        base_size = 1
        
        # Ajustar por confiança do sinal
        if signal['confidence'] > 0.8:
            base_size = 2
        elif signal['confidence'] > 0.9:
            base_size = 3
        
        # Limitar pelo capital disponível
        margin_required = base_size * self.config.margin_per_contract
        if margin_required > self.current_capital * 0.5:
            base_size = int((self.current_capital * 0.5) / 
                           self.config.margin_per_contract)
        
        return max(1, base_size)
    
    def _calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calcula stop loss"""
        # Stop de 2% por padrão
        stop_distance = entry_price * 0.02
        
        if side == 'long':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def _calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calcula take profit"""
        # Target de 3% por padrão
        target_distance = entry_price * 0.03
        
        if side == 'long':
            return entry_price + target_distance
        else:
            return entry_price - target_distance
    
    def _prepare_model_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Prepara features para os modelos"""
        # Verificar se os modelos foram inicializados
        if self.ml_models is None or len(self.ml_models) == 0:
            self.logger.warning("Modelos ML não inicializados, retornando todas as features")
            return features
        
        # Obter primeiro modelo válido
        first_model = None
        for model in self.ml_models.values():
            if hasattr(model, 'feature_names_in_'):
                first_model = model
                break
        
        if first_model is None:
            self.logger.warning("Nenhum modelo com feature_names_in_ encontrado, retornando todas as features")
            return features
        
        # Garantir que temos todas as features necessárias
        required_features = first_model.feature_names_in_
        
        # Selecionar apenas features necessárias
        available_features = [f for f in required_features if f in features.columns]
        
        if len(available_features) == 0:
            self.logger.warning("Nenhuma feature necessária encontrada, retornando todas as features")
            return features
        
        return features[available_features]
    
    def _combine_predictions(self, predictions: Dict) -> np.ndarray:
        """Combina predições de múltiplos modelos"""
        # Média simples por enquanto
        pred_arrays = []
        for pred in predictions.values():
            if isinstance(pred, np.ndarray) and len(pred.shape) > 0:
                pred_arrays.append(pred)
            else:
                # Converter predição única em array de probabilidades
                pred_array = np.zeros(3)
                pred_array[int(pred)] = 1.0
                pred_arrays.append(pred_array)
        
        return np.mean(pred_arrays, axis=0)
    
    def _calculate_final_metrics(self) -> Dict:
        """Calcula métricas finais do backtest"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'final_equity': self.config.initial_capital
            }
        
        # Métricas básicas
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        metrics = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades),
            'total_pnl': sum(t.pnl for t in self.trades),
            'final_equity': self.equity_curve[-1]['equity'] if self.equity_curve else self.config.initial_capital,
            'total_return': ((self.equity_curve[-1]['equity'] / self.config.initial_capital) - 1) 
                          if self.equity_curve else 0
        }
        
        # Métricas avançadas
        if winning_trades:
            metrics['avg_win'] = np.mean([t.pnl for t in winning_trades])
            metrics['max_win'] = max(t.pnl for t in winning_trades)
        else:
            metrics['avg_win'] = 0
            metrics['max_win'] = 0
        
        if losing_trades:
            metrics['avg_loss'] = np.mean([t.pnl for t in losing_trades])
            metrics['max_loss'] = min(t.pnl for t in losing_trades)
        else:
            metrics['avg_loss'] = 0
            metrics['max_loss'] = 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expectancy
        metrics['expectancy'] = metrics['total_pnl'] / len(self.trades)
        
        # Métricas de risco
        returns = pd.Series([t.return_pct for t in self.trades])
        if len(returns) > 1:
            metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
            metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
        
        # Drawdown
        if self.drawdown_series:
            metrics['max_drawdown'] = min(d['drawdown'] for d in self.drawdown_series)
            metrics['avg_drawdown'] = np.mean([d['drawdown'] for d in self.drawdown_series if d['drawdown'] < 0])
        
        return metrics
    
    def _analyze_trades(self) -> Dict:
        """Análise detalhada dos trades"""
        if not self.trades:
            return {}
        
        analysis = {
            'by_side': self._analyze_by_side(),
            'by_duration': self._analyze_by_duration(),
            'by_time_of_day': self._analyze_by_time_of_day(),
            'mae_mfe_analysis': self._analyze_mae_mfe(),
            'consecutive_analysis': self._analyze_consecutive_trades()
        }
        
        return analysis
    
    def _analyze_by_side(self) -> Dict:
        """Analisa trades por lado (long/short)"""
        long_trades = [t for t in self.trades if t.side == 'long']
        short_trades = [t for t in self.trades if t.side == 'short']
        
        return {
            'long': {
                'count': len(long_trades),
                'win_rate': len([t for t in long_trades if t.pnl > 0]) / len(long_trades) if long_trades else 0,
                'avg_pnl': np.mean([t.pnl for t in long_trades]) if long_trades else 0
            },
            'short': {
                'count': len(short_trades),
                'win_rate': len([t for t in short_trades if t.pnl > 0]) / len(short_trades) if short_trades else 0,
                'avg_pnl': np.mean([t.pnl for t in short_trades]) if short_trades else 0
            }
        }
    
    def _analyze_by_duration(self) -> Dict:
        """Analisa trades por duração"""
        durations = []
        for trade in self.trades:
            if trade.duration:
                durations.append({
                    'duration_minutes': trade.duration.total_seconds() / 60,
                    'pnl': trade.pnl,
                    'return_pct': trade.return_pct
                })
        
        if not durations:
            return {}
        
        df = pd.DataFrame(durations)
        
        # Agrupar por faixas de duração
        bins = [0, 5, 15, 30, 60, 240, np.inf]
        labels = ['0-5min', '5-15min', '15-30min', '30-60min', '1-4h', '>4h']
        df['duration_range'] = pd.cut(df['duration_minutes'], bins=bins, labels=labels)
        
        analysis = {}
        for range_label in labels:
            range_trades = df[df['duration_range'] == range_label]
            if len(range_trades) > 0:
                analysis[range_label] = {
                    'count': len(range_trades),
                    'avg_pnl': range_trades['pnl'].mean(),
                    'win_rate': (range_trades['pnl'] > 0).mean()
                }
        
        return analysis
    
    def _analyze_by_time_of_day(self) -> Dict:
        """Analisa trades por hora do dia"""
        hourly_stats = {}
        
        for trade in self.trades:
            hour = trade.entry_time.hour
            if hour not in hourly_stats:
                hourly_stats[hour] = {'trades': [], 'pnl': []}
            
            hourly_stats[hour]['trades'].append(trade)
            hourly_stats[hour]['pnl'].append(trade.pnl)
        
        analysis = {}
        for hour, stats in hourly_stats.items():
            analysis[f"{hour:02d}:00"] = {
                'count': len(stats['trades']),
                'total_pnl': sum(stats['pnl']),
                'avg_pnl': np.mean(stats['pnl']),
                'win_rate': len([p for p in stats['pnl'] if p > 0]) / len(stats['pnl'])
            }
        
        return analysis
    
    def _analyze_mae_mfe(self) -> Dict:
        """Analisa Maximum Adverse/Favorable Excursion"""
        mae_values = [abs(t.mae) for t in self.trades if t.mae < 0]
        mfe_values = [t.mfe for t in self.trades if t.mfe > 0]
        
        return {
            'mae': {
                'mean': np.mean(mae_values) if mae_values else 0,
                'max': max(mae_values) if mae_values else 0,
                'percentiles': {
                    '25%': np.percentile(mae_values, 25) if mae_values else 0,
                    '50%': np.percentile(mae_values, 50) if mae_values else 0,
                    '75%': np.percentile(mae_values, 75) if mae_values else 0
                }
            },
            'mfe': {
                'mean': np.mean(mfe_values) if mfe_values else 0,
                'max': max(mfe_values) if mfe_values else 0,
                'percentiles': {
                    '25%': np.percentile(mfe_values, 25) if mfe_values else 0,
                    '50%': np.percentile(mfe_values, 50) if mfe_values else 0,
                    '75%': np.percentile(mfe_values, 75) if mfe_values else 0
                }
            }
        }
    
    def _analyze_consecutive_trades(self) -> Dict:
        """Analisa sequências de trades vencedores/perdedores"""
        if not self.trades:
            return {}
        
        # Sequências
        current_wins = 0
        current_losses = 0
        max_wins = 0
        max_losses = 0
        
        win_streaks = []
        loss_streaks = []
        
        for trade in self.trades:
            if trade.pnl > 0:
                current_wins += 1
                if current_losses > 0:
                    loss_streaks.append(current_losses)
                    current_losses = 0
            else:
                current_losses += 1
                if current_wins > 0:
                    win_streaks.append(current_wins)
                    current_wins = 0
            
            max_wins = max(max_wins, current_wins)
            max_losses = max(max_losses, current_losses)
        
        # Adicionar últimas sequências
        if current_wins > 0:
            win_streaks.append(current_wins)
        if current_losses > 0:
            loss_streaks.append(current_losses)
        
        return {
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses,
            'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
            'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0
        }
    
    def _analyze_drawdowns(self) -> Dict:
        """Análise detalhada de drawdowns"""
        if not self.drawdown_series:
            return {}
        
        # Identificar períodos de drawdown
        drawdown_periods = []
        in_drawdown = False
        current_period = None
        
        for i, point in enumerate(self.drawdown_series):
            if point['drawdown'] < 0 and not in_drawdown:
                # Início de drawdown
                in_drawdown = True
                current_period = {
                    'start': point['timestamp'],
                    'peak_drawdown': point['drawdown'],
                    'values': [point['drawdown']]
                }
            elif point['drawdown'] < 0 and in_drawdown and current_period is not None:
                # Continuação de drawdown
                current_period['values'].append(point['drawdown'])
                if point['drawdown'] < current_period['peak_drawdown']:
                    current_period['peak_drawdown'] = point['drawdown']
            elif point['drawdown'] >= 0 and in_drawdown and current_period is not None:
                # Fim de drawdown
                in_drawdown = False
                current_period['end'] = point['timestamp']
                current_period['duration'] = current_period['end'] - current_period['start']
                current_period['recovery_time'] = len(current_period['values'])
                drawdown_periods.append(current_period)
                current_period = None
        
        # Se terminou em drawdown
        if in_drawdown and current_period:
            current_period['end'] = self.drawdown_series[-1]['timestamp']
            current_period['duration'] = current_period['end'] - current_period['start']
            current_period['recovery_time'] = None  # Não recuperou
            drawdown_periods.append(current_period)
        
        if not drawdown_periods:
            return {'no_drawdowns': True}
        
        # Estatísticas
        return {
            'total_drawdown_periods': len(drawdown_periods),
            'max_drawdown': min(d['peak_drawdown'] for d in drawdown_periods),
            'avg_drawdown': np.mean([d['peak_drawdown'] for d in drawdown_periods]),
            'max_duration': max(d['duration'].total_seconds() / 3600 for d in drawdown_periods),  # em horas
            'avg_recovery_time': np.mean([d['recovery_time'] for d in drawdown_periods if d['recovery_time']]),
            'current_drawdown': self.drawdown_series[-1]['drawdown'] if self.drawdown_series else 0
        }
    
    def _analyze_by_regime(self, data: pd.DataFrame) -> Dict:
        """Analisa performance por regime de mercado"""
        # TODO: Implementar análise por regime
        # Requer integração com detector de regime
        return {}
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, 
                              risk_free_rate: float = 0.02) -> float:
        """Calcula Sharpe Ratio"""
        if len(returns) < 2:
            return 0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        if returns.std() == 0:
            return 0
        
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series,
                               target_return: float = 0) -> float:
        """Calcula Sortino Ratio"""
        if len(returns) < 2:
            return 0
        
        excess_returns = returns - target_return
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = downside_returns.std()
        if downside_std == 0:
            return np.inf
        
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    def export_results(self, filepath: str):
        """Exporta resultados do backtest"""
        results = {
            'config': {
                'start_date': self.config.start_date.isoformat(),
                'end_date': self.config.end_date.isoformat(),
                'initial_capital': self.config.initial_capital,
                'mode': self.config.mode.value
            },
            'metrics': self._calculate_final_metrics(),
            'trades': [self._trade_to_dict(t) for t in self.trades],
            'equity_curve': self.equity_curve,
            'drawdown_series': self.drawdown_series
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Converte Trade para dicionário"""
        return {
            'entry_time': trade.entry_time.isoformat(),
            'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
            'symbol': trade.symbol,
            'side': trade.side,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'quantity': trade.quantity,
            'pnl': trade.pnl,
            'return_pct': trade.return_pct,
            'mae': trade.mae,
            'mfe': trade.mfe,
            'duration_minutes': trade.duration.total_seconds() / 60 if trade.duration else None,
            'exit_reason': trade.exit_reason
        }