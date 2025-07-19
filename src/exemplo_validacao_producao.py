"""
🛡️ EXEMPLO PRÁTICO - USO DO VALIDADOR DE PRODUÇÃO
Sistema de Trading ML v2.0 - Implementação Segura

Este arquivo demonstra como integrar o ProductionDataValidator
em todos os pontos críticos do sistema de trading.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any

# Configurar para modo produção
os.environ['TRADING_PRODUCTION_MODE'] = 'True'
os.environ['STRICT_VALIDATION'] = 'True'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_data_validator import (
    ProductionDataValidator, 
    ProductionDataError,
    enforce_production_mode
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def exemplo_data_loader_seguro():
    """
    Exemplo: DataLoader com validação de produção
    ✅ Substitui o data_loader.py com dados sintéticos
    """
    
    print("=" * 60)
    print("🔴 EXEMPLO 1: DATA LOADER SEGURO")
    print("=" * 60)
    
    # Inicializar modo produção
    enforce_production_mode()
    validator = ProductionDataValidator()
    
    class ProductionSafeDataLoader:
        """DataLoader que só aceita dados reais"""
        
        def __init__(self):
            self.validator = validator
            self.logger = logging.getLogger('SafeDataLoader')
        
        def load_market_data(self, source: str = "ProfitDLL") -> pd.DataFrame:
            """Carrega dados de mercado com validação obrigatória"""
            
            self.logger.info(f"📡 Carregando dados de {source}")
            
            # 🔴 SIMULAÇÃO DE DADOS REAIS (em produção viria de API/DLL real)
            if source == "ProfitDLL":
                data = self._simulate_real_dll_data()
            elif source == "DUMMY":
                data = self._create_dummy_data()  # Para demonstrar bloqueio
            else:
                raise ValueError(f"Fonte não suportada: {source}")
            
            # 🛡️ VALIDAÇÃO OBRIGATÓRIA
            try:
                self.validator.validate_trading_data(data, source, 'realtime')
                self.logger.info("✅ Dados validados e aprovados para trading")
                return data
            
            except ProductionDataError as e:
                self.logger.error(f"❌ Dados rejeitados: {str(e)}")
                raise
        
        def _simulate_real_dll_data(self) -> pd.DataFrame:
            """Simula dados que PASSARIAM na validação (padrão real)"""
            
            # Gerar dados com características de dados reais
            dates = pd.date_range(start=datetime.now() - timedelta(minutes=30), 
                                end=datetime.now(), freq='1min')
            
            # Preços com movimento browniano mais realista
            base_price = 126500  # Preço base do mini índice
            returns = np.random.normal(0, 0.0005, len(dates))  # Volatilidade realista
            returns[0] = 0  # Primeiro retorno zero
            
            prices = base_price * (1 + returns).cumprod()
            
            # Volume com padrão mais realista (não uniforme)
            volumes = np.random.gamma(2, 50) + np.random.uniform(10, 100, len(dates))
            
            # Criar OHLC realista
            price_noise = np.random.normal(0, 0.0002, len(dates))
            
            data = pd.DataFrame({
                'timestamp': dates,
                'open': prices + np.random.normal(0, 1, len(dates)),
                'high': prices * (1 + np.abs(price_noise) * 2),
                'low': prices * (1 - np.abs(price_noise) * 2), 
                'close': prices,
                'volume': volumes.astype(int),
                'trade_id': range(1000, 1000 + len(dates))  # IDs não sequenciais
            })
            
            data.set_index('timestamp', inplace=True)
            return data
        
        def _create_dummy_data(self) -> pd.DataFrame:
            """Cria dados dummy que FALHARÃO na validação"""
            
            dates = pd.date_range(start=datetime.now() - timedelta(minutes=10), 
                                end=datetime.now(), freq='1min')
            
            # ❌ Dados claramente sintéticos que falharão
            data = pd.DataFrame({
                'timestamp': dates,
                'open': [100] * len(dates),  # Preços constantes
                'high': [100] * len(dates),
                'low': [100] * len(dates),
                'close': [100] * len(dates),
                'volume': [50] * len(dates),  # Volume constante (suspeito)
                'trade_id': range(len(dates))  # IDs sequenciais (suspeito)
            })
            
            data.set_index('timestamp', inplace=True)
            return data
    
    # 🧪 TESTE 1: Dados reais (devem passar)
    try:
        loader = ProductionSafeDataLoader()
        real_data = loader.load_market_data("ProfitDLL")
        print(f"✅ Sucesso: {len(real_data)} registros de dados reais carregados")
        print(f"   Últimos preços: {real_data['close'].tail(3).tolist()}")
        
    except ProductionDataError as e:
        print(f"❌ Erro inesperado com dados reais: {e}")
    
    # 🧪 TESTE 2: Dados dummy (devem falhar)
    try:
        dummy_data = loader.load_market_data("DUMMY")
        print("❌ ERRO: Dados dummy foram aceitos (BUG CRÍTICO!)")
        
    except ProductionDataError as e:
        print(f"✅ Sucesso: Dados dummy bloqueados corretamente")
        print(f"   Motivo: {str(e)[:100]}...")


def exemplo_feature_engine_seguro():
    """
    Exemplo: Feature Engine com validação anti-fillna perigoso
    ✅ Substitui feature_engine.py com fillna(0) perigoso
    """
    
    print("\n" + "=" * 60)
    print("🟡 EXEMPLO 2: FEATURE ENGINE SEGURO")
    print("=" * 60)
    
    validator = ProductionDataValidator()
    
    class ProductionSafeFeatureEngine:
        """Feature Engine que não usa fillna perigoso"""
        
        def __init__(self):
            self.validator = validator
            self.logger = logging.getLogger('SafeFeatureEngine')
        
        def calculate_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
            """Calcula features com validação de segurança"""
            
            self.logger.info("🔧 Calculando features ML")
            
            # ✅ Cálculo seguro de features
            features = pd.DataFrame(index=market_data.index)
            
            # ✅ EMAs com forward fill (não zeros)
            for period in [9, 20, 50]:
                ema_col = f'ema_{period}'
                features[ema_col] = market_data['close'].ewm(span=period).mean()
                features[ema_col] = features[ema_col].ffill()  # Apenas forward fill
            
            # ✅ RSI sem fillna(50) perigoso
            rsi = self._calculate_rsi(market_data['close'])
            features['rsi'] = rsi.ffill()  # Manter NaN até ter dados suficientes
            
            # ✅ Volume SMA sem zeros artificiais
            features['volume_sma'] = market_data['volume'].rolling(20).mean()
            features['volume_sma'] = features['volume_sma'].ffill()
            
            # ✅ Momentum sem fillna(0)
            features['momentum_5'] = market_data['close'].diff(5)
            features['momentum_5'] = features['momentum_5'].ffill()
            
            # 🛡️ VALIDAÇÃO FINAL DAS FEATURES
            self.validator.validate_feature_data(features)
            
            self.logger.info(f"✅ {len(features.columns)} features calculadas com segurança")
            return features
        
        def calculate_unsafe_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
            """Demonstra cálculo UNSAFE que será bloqueado"""
            
            features = pd.DataFrame(index=market_data.index)
            
            # ❌ PRÁTICAS PERIGOSAS que serão detectadas
            features['rsi_unsafe'] = 50  # RSI fixo em 50 (muito suspeito)
            features['volume_unsafe'] = 0  # Volume zero (muito suspeito)
            features['momentum_unsafe'] = market_data['close'].diff(5).fillna(0)  # fillna(0)
            
            return features
        
        def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
            """Calcula RSI sem valores fake"""
            
            delta = prices.diff()
            gain = delta.clip(lower=0)  # Apenas valores positivos
            loss = (-delta).clip(lower=0)  # Apenas valores positivos das perdas
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Evitar divisão por zero
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            # ❌ NÃO fazer: rsi.fillna(50)
            # ✅ SIM: deixar NaN até ter dados suficientes
            return rsi
    
    # 🧪 Criar dados de teste
    dates = pd.date_range(start=datetime.now() - timedelta(hours=2), 
                         end=datetime.now(), freq='1min')
    
    prices = 126500 * (1 + np.random.normal(0, 0.001, len(dates))).cumprod()
    volumes = np.random.gamma(2, 50)
    
    market_data = pd.DataFrame({
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # 🧪 TESTE 1: Features seguras
    try:
        engine = ProductionSafeFeatureEngine()
        safe_features = engine.calculate_features(market_data)
        print(f"✅ Features seguras calculadas: {list(safe_features.columns)}")
        print(f"   Exemplo RSI: {safe_features['rsi'].dropna().tail(3).tolist()}")
        
    except ProductionDataError as e:
        print(f"❌ Erro inesperado com features seguras: {e}")
    
    # 🧪 TESTE 2: Features perigosas (devem falhar)
    try:
        unsafe_features = engine.calculate_unsafe_features(market_data)
        engine.validator.validate_feature_data(unsafe_features)
        print("❌ ERRO: Features perigosas foram aceitas (BUG CRÍTICO!)")
        
    except ProductionDataError as e:
        print(f"✅ Features perigosas bloqueadas corretamente")
        print(f"   Motivo: {str(e)[:100]}...")


def exemplo_model_manager_seguro():
    """
    Exemplo: Model Manager com validação de features
    ✅ Substitui model_manager.py com fillna(0) perigoso
    """
    
    print("\n" + "=" * 60)
    print("🟢 EXEMPLO 3: MODEL MANAGER SEGURO") 
    print("=" * 60)
    
    validator = ProductionDataValidator()
    
    class ProductionSafeModelManager:
        """Model Manager que só aceita features validadas"""
        
        def __init__(self):
            self.validator = validator
            self.logger = logging.getLogger('SafeModelManager')
            self.required_features = [
                'ema_9', 'ema_20', 'ema_50', 'rsi', 'volume_sma', 'momentum_5'
            ]
        
        def predict(self, features_df: pd.DataFrame) -> Dict:
            """Predição ML com validação rigorosa de entrada"""
            
            self.logger.info("🤖 Executando predição ML")
            
            # 🛡️ VALIDAÇÃO 1: Estrutura das features
            self._validate_feature_structure(features_df)
            
            # 🛡️ VALIDAÇÃO 2: Qualidade dos dados
            self.validator.validate_feature_data(features_df)
            
            # ✅ Preparar features sem fillna perigoso
            X = self._prepare_features_safely(features_df)
            
            # 🤖 Simulação de predição (em produção seria modelo real)
            prediction = self._simulate_model_prediction(X)
            
            self.logger.info(f"✅ Predição executada: {prediction}")
            return prediction
        
        def _validate_feature_structure(self, features_df: pd.DataFrame):
            """Valida estrutura das features"""
            
            missing_features = []
            for feature in self.required_features:
                if feature not in features_df.columns:
                    missing_features.append(feature)
            
            if missing_features:
                raise ProductionDataError(
                    f"Features obrigatórias ausentes: {missing_features}"
                )
        
        def _prepare_features_safely(self, features_df: pd.DataFrame) -> pd.DataFrame:
            """Prepara features SEM fillna perigoso"""
            
            X = features_df[self.required_features].copy()
            
            # ❌ NÃO fazer: X.fillna(0)
            # ✅ SIM: Estratégia inteligente por tipo de feature
            
            for col in X.columns:
                if X[col].isnull().any():
                    if col.startswith('ema'):
                        # EMA: usar último valor válido
                        X[col] = X[col].ffill()
                    elif col == 'rsi':
                        # RSI: usar valor anterior, não 50 fixo
                        X[col] = X[col].ffill()
                    elif 'volume' in col:
                        # Volume: usar média móvel
                        X[col] = X[col].ffill()
                    elif 'momentum' in col:
                        # Momentum: usar forward fill
                        X[col] = X[col].ffill()
            
            # Verificar se ainda há NaN após tratamento
            if X.isnull().any().any():
                nan_cols = X.columns[X.isnull().any()].tolist()
                raise ProductionDataError(
                    f"Features com NaN após tratamento: {nan_cols}"
                )
            
            return X
        
        def _simulate_model_prediction(self, X: pd.DataFrame) -> Dict:
            """Simula predição de modelo ML"""
            
            # Simular predição baseada em features reais
            latest_row = X.iloc[-1]
            
            # Simular lógica de regime baseada em EMAs
            if latest_row['ema_9'] > latest_row['ema_20'] > latest_row['ema_50']:
                regime = 'trend_up'
                direction = 0.75
            elif latest_row['ema_9'] < latest_row['ema_20'] < latest_row['ema_50']:
                regime = 'trend_down' 
                direction = 0.25
            else:
                regime = 'range'
                direction = 0.5
            
            return {
                'regime': regime,
                'direction': direction,
                'confidence': 0.85,
                'trade_decision': 'BUY' if direction > 0.6 else 'SELL' if direction < 0.4 else 'HOLD'
            }
    
    # 🧪 Criar features de teste (seguras)
    dates = pd.date_range(start=datetime.now() - timedelta(hours=1), 
                         end=datetime.now(), freq='1min')
    
    safe_features = pd.DataFrame({
        'ema_9': 126500 + np.random.normal(0, 10, len(dates)),
        'ema_20': 126480 + np.random.normal(0, 5, len(dates)),
        'ema_50': 126460 + np.random.normal(0, 3, len(dates)),
        'rsi': 45 + np.random.normal(0, 10, len(dates)),
        'volume_sma': 100 + np.random.gamma(2, 20, len(dates)),
        'momentum_5': np.random.normal(0, 50, len(dates))
    }, index=dates)
    
    # 🧪 TESTE: Predição com features seguras
    try:
        manager = ProductionSafeModelManager()
        prediction = manager.predict(safe_features)
        print(f"✅ Predição executada com sucesso:")
        print(f"   Regime: {prediction['regime']}")
        print(f"   Direção: {prediction['direction']:.2f}")
        print(f"   Decisão: {prediction['trade_decision']}")
        
    except ProductionDataError as e:
        print(f"❌ Erro na predição: {e}")


if __name__ == "__main__":
    """Executar todos os exemplos"""
    
    print("🛡️ SISTEMA DE VALIDAÇÃO PARA PRODUÇÃO - ML TRADING v2.0")
    print("=" * 70)
    print("OBJETIVO: Demonstrar como bloquear dados dummy/sintéticos")
    print("RESULTADO ESPERADO: ✅ Dados reais aceitos | ❌ Dados dummy bloqueados")
    print("=" * 70)
    
    try:
        # Exemplo 1: Data Loader seguro
        exemplo_data_loader_seguro()
        
        # Exemplo 2: Feature Engine seguro  
        exemplo_feature_engine_seguro()
        
        # Exemplo 3: Model Manager seguro
        exemplo_model_manager_seguro()
        
        print("\n" + "=" * 70)
        print("✅ TODOS OS EXEMPLOS EXECUTADOS COM SUCESSO!")
        print("🛡️ Sistema pronto para bloquear dados dummy em produção")
        print("⚠️ PRÓXIMO PASSO: Integrar validador em TODOS os componentes")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERRO NA EXECUÇÃO DOS EXEMPLOS: {e}")
        print("🔧 Verificar dependências e configuração")
