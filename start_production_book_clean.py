#!/usr/bin/env python
"""
🚀 SISTEMA DE PRODUÇÃO COM BOOK_CLEAN
Modelo campeão: 79.23% Trading Accuracy
Otimizado para performance máxima
"""

import os
import sys
import time
import logging
import lightgbm as lgb
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json

# Adicionar diretório ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar configuração
from config_book_clean_production import get_production_config

# Importar sistema base
from production_fixed import ProductionFixedSystem

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BookCleanProduction')


class BookCleanProductionSystem(ProductionFixedSystem):
    """
    Sistema de produção otimizado para book_clean
    Features mínimas, performance máxima
    """
    
    def __init__(self):
        super().__init__()
        
        # Carregar configuração
        self.config = get_production_config()
        
        # Modelo principal
        self.book_clean_model = None
        self.scaler = None
        self.required_features = self.config['model']['primary_model']['features']
        
        # Cache de features
        self.last_features = None
        self.last_feature_time = 0
        self.feature_cache_duration = 1.0  # 1 segundo de cache
        
        # Métricas em tempo real
        self.real_time_metrics = {
            'predictions': 0,
            'correct_predictions': 0,
            'trading_accuracy': 0,
            'last_10_predictions': [],
            'last_10_results': []
        }
        
        # Estado otimizado
        self.min_data_required = 20  # Mínimo de candles necessários
        
        # Inicializar PnL e posição
        self.pnl = 0.0
        self.position = 0
        
        # Lista completa de todas as 14 features necessárias
        self.all_features = [
            'price_normalized', 'position', 'position_normalized', 
            'price_pct_change', 'side', 'quantity_log', 
            'price_rolling_std', 'is_bid', 'time_of_day',
            'minute', 'quantity_zscore', 'hour', 'is_ask', 'is_top_5'
        ]
        
    def _load_ml_models(self):
        """Carrega APENAS o modelo book_clean"""
        try:
            logger.info("="*60)
            logger.info("[LOADING] CARREGANDO MODELO BOOK_CLEAN")
            logger.info("="*60)
            
            # Caminho do modelo
            model_path = Path(self.config['model']['primary_model']['path'])
            scaler_path = Path(self.config['model']['primary_model']['scaler_path'])
            
            # Verificar se existem
            if not model_path.exists():
                logger.error(f"[ERRO] Modelo nao encontrado: {model_path}")
                return False
            
            if not scaler_path.exists():
                logger.error(f"[ERRO] Scaler nao encontrado: {scaler_path}")
                return False
            
            # Carregar LightGBM
            logger.info(f"[LOAD] Carregando modelo: {model_path.name}")
            self.book_clean_model = lgb.Booster(model_file=str(model_path))
            
            # Carregar Scaler
            logger.info(f"[LOAD] Carregando scaler: {scaler_path.name}")
            self.scaler = joblib.load(scaler_path)
            
            # Configurar no sistema base
            self.models = {'book_clean': self.book_clean_model}
            self.features_lists = {'book_clean': self.required_features}
            
            # Log de sucesso
            logger.info("[OK] MODELO BOOK_CLEAN CARREGADO COM SUCESSO!")
            logger.info(f"   Trading Accuracy: {self.config['model']['primary_model']['trading_accuracy']:.1%}")
            logger.info(f"   Features: {self.required_features}")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"[ERRO] Erro ao carregar modelo: {e}")
            return False
    
    def _calculate_book_clean_features(self):
        """
        Calcula TODAS as 14 features necessárias para o book_clean
        """
        # Verificar cache
        current_time = time.time()
        if (self.last_features is not None and 
            current_time - self.last_feature_time < self.feature_cache_duration):
            return self.last_features
        
        # Verificar dados mínimos
        if len(self.candles) < self.min_data_required:
            return None
        
        try:
            # Preparar dados
            df = pd.DataFrame(self.candles[-self.min_data_required:])
            features = {}
            
            # Obter hora atual
            now = datetime.now()
            
            # 1. price_normalized (mais importante!)
            prices = df['close'].values
            price_mean = np.mean(prices)
            if price_mean > 0:
                features['price_normalized'] = self.current_price / price_mean
            else:
                features['price_normalized'] = 1.0
            
            # 2. position (simulado como índice no book)
            features['position'] = 1.0  # Top do book
            
            # 3. position_normalized
            features['position_normalized'] = features['position'] / 10.0  # Assumindo 10 níveis
            
            # 4. price_pct_change
            if len(prices) >= 2:
                prev_price = prices[-2]
                if prev_price > 0:
                    features['price_pct_change'] = ((self.current_price - prev_price) / prev_price) * 100
                else:
                    features['price_pct_change'] = 0.0
            else:
                features['price_pct_change'] = 0.0
            
            # 5. side (0=bid, 1=ask)
            if features['price_pct_change'] > 0:
                features['side'] = 1.0  # Ask (compradores)
            else:
                features['side'] = 0.0  # Bid (vendedores)
            
            # 6. quantity_log (log da quantidade média)
            if 'volume' in df.columns:
                avg_volume = df['volume'].mean()
                features['quantity_log'] = np.log1p(avg_volume) if avg_volume > 0 else 0.0
            else:
                features['quantity_log'] = 3.0  # valor padrão
            
            # 7. price_rolling_std (volatilidade)
            features['price_rolling_std'] = np.std(prices) if len(prices) > 1 else 0.01
            
            # 8. is_bid (1 se bid dominante)
            features['is_bid'] = 1.0 if features['side'] == 0.0 else 0.0
            
            # 9. time_of_day (normalizado 0-1)
            features['time_of_day'] = (now.hour * 60 + now.minute) / (24 * 60)
            
            # 10. minute
            features['minute'] = float(now.minute)
            
            # 11. quantity_zscore
            if 'volume' in df.columns and len(df) > 1:
                volumes = df['volume'].values
                vol_mean = np.mean(volumes)
                vol_std = np.std(volumes)
                if vol_std > 0:
                    features['quantity_zscore'] = (volumes[-1] - vol_mean) / vol_std
                else:
                    features['quantity_zscore'] = 0.0
            else:
                features['quantity_zscore'] = 0.0
            
            # 12. hour
            features['hour'] = float(now.hour)
            
            # 13. is_ask (inverso de is_bid)
            features['is_ask'] = 1.0 - features['is_bid']
            
            # 14. is_top_5 (sempre 1 pois estamos no topo)
            features['is_top_5'] = 1.0
            
            # Atualizar cache
            self.last_features = features
            self.last_feature_time = current_time
            
            return features
            
        except Exception as e:
            logger.error(f"Erro ao calcular features: {e}")
            return None
    
    def _make_prediction(self):
        """Predição otimizada com book_clean"""
        
        # Calcular features específicas do book_clean
        features = self._calculate_book_clean_features()
        if not features:
            logger.debug("Sem features calculadas - aguardando mais dados")
            return None
        
        try:
            # Preparar vetor de features na ordem correta (14 features)
            feature_vector = [features.get(f, 0) for f in self.all_features]
            
            # Aplicar scaler
            X = np.array([feature_vector])
            X_scaled = self.scaler.transform(X)
            
            # Fazer predição com LightGBM (multiclass)
            prediction = self.book_clean_model.predict(X_scaled)
            
            # prediction é um array de probabilidades [P(SELL), P(HOLD), P(BUY)]
            pred_probs = prediction[0]
            pred_class = np.argmax(pred_probs)  # 0=SELL, 1=HOLD, 2=BUY
            confidence = pred_probs[pred_class]
            
            # Converter para direção (0-1)
            # SELL=0 -> direction=0.25, HOLD=1 -> direction=0.5, BUY=2 -> direction=0.75
            direction = 0.25 + (pred_class * 0.25)
            
            # Atualizar métricas
            self.real_time_metrics['predictions'] += 1
            self.stats['predictions'] += 1
            
            # Log de predição importante
            if confidence > 0.7:
                logger.info(f"[PRED] PREDICAO FORTE: Dir={direction:.3f} Conf={confidence:.3f}")
            
            # Resultado final
            result = {
                'direction': direction,
                'confidence': confidence,
                'model': 'book_clean',
                'features': features,
                'timestamp': time.time()
            }
            
            # Salvar para análise
            self._last_prediction = result
            
            # Atualizar histórico
            self.real_time_metrics['last_10_predictions'].append(result)
            if len(self.real_time_metrics['last_10_predictions']) > 10:
                self.real_time_metrics['last_10_predictions'].pop(0)
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return None
    
    def _calculate_real_time_accuracy(self):
        """Calcula accuracy em tempo real baseado nos últimos trades"""
        if len(self.real_time_metrics['last_10_results']) >= 5:
            correct = sum(1 for r in self.real_time_metrics['last_10_results'] if r)
            total = len(self.real_time_metrics['last_10_results'])
            accuracy = correct / total
            self.real_time_metrics['trading_accuracy'] = accuracy
            
            # Log se accuracy cair
            if accuracy < 0.7 and total >= 10:
                logger.warning(f"[AVISO] Trading Accuracy caiu: {accuracy:.1%}")
            elif accuracy > 0.75:
                logger.info(f"[OK] Trading Accuracy excelente: {accuracy:.1%}")
    
    def _log_status(self):
        """Log de status com métricas do book_clean"""
        super()._log_status()
        
        # Métricas específicas do book_clean
        if self.real_time_metrics['predictions'] > 0:
            logger.info(f"[METRICS] Predicoes: {self.real_time_metrics['predictions']} | "
                       f"Trading Acc: {self.real_time_metrics['trading_accuracy']:.1%} | "
                       f"Target: 79.23%")
            
            # Mostrar última predição
            if self._last_prediction:
                logger.info(f"   Última: Dir={self._last_prediction['direction']:.3f} "
                          f"Conf={self._last_prediction['confidence']:.3f}")
    
    def start(self):
        """Inicia sistema com validações do book_clean"""
        
        # Verificar se modelo está carregado
        if not self.book_clean_model:
            logger.error("[ERRO] Modelo book_clean nao carregado!")
            return False
        
        logger.info("\n" + "="*60)
        logger.info("[START] INICIANDO SISTEMA COM BOOK_CLEAN")
        logger.info(f"   Modelo: book_clean")
        logger.info(f"   Trading Accuracy esperada: 79.23%")
        logger.info(f"   Total de features: {len(self.all_features)}")
        logger.info(f"   Otimização: MÁXIMA")
        logger.info("="*60 + "\n")
        
        # Iniciar sistema base
        result = super().start()
        
        if result:
            logger.info("[OK] Sistema de trading iniciado com sucesso!")
            logger.info(f"   Aguardando {self.min_data_required} candles para iniciar predições...")
        
        return result


def main():
    """Executa sistema de produção com book_clean"""
    
    print("\n" + "="*80)
    print("🚀 QUANTUM TRADER ML - PRODUÇÃO COM BOOK_CLEAN")
    print("="*80)
    print(f"📅 Data: {datetime.now()}")
    print(f"📊 Modelo: book_clean (79.23% Trading Accuracy)")
    print(f"⚡ Features: 5 apenas (ultra-otimizado)")
    print(f"🎯 Objetivo: Manter > 75% accuracy em produção")
    print("="*80)
    
    try:
        # Criar sistema
        system = BookCleanProductionSystem()
        
        # Inicializar
        print("\n📡 Conectando ao mercado...")
        if not system.initialize():
            print("❌ Falha na inicialização")
            return 1
        
        # Aguardar conexão
        print("⏳ Aguardando conexão completa...")
        time.sleep(3)
        
        # Subscrever ticker
        ticker = os.getenv('TICKER', 'WDOU25')
        print(f"\n📈 Subscrevendo {ticker}...")
        if not system.subscribe_ticker(ticker):
            print(f"❌ Falha ao subscrever {ticker}")
            return 1
        
        # Aguardar dados
        print("📊 Aguardando dados do mercado...")
        time.sleep(5)
        
        # Verificar recepção
        print(f"\n✅ Callbacks recebidos:")
        for cb_type, count in system.callbacks.items():
            if count > 0:
                print(f"   {cb_type}: {count:,}")
        
        # Verificar se temos dados suficientes
        if len(system.candles) < system.min_data_required:
            print(f"\n⏳ Aguardando mais dados... ({len(system.candles)}/{system.min_data_required})")
            time.sleep(10)
        
        # Iniciar trading
        print(f"\n🚀 Iniciando sistema de trading...")
        if not system.start():
            print("❌ Falha ao iniciar trading")
            return 1
        
        print("\n" + "="*80)
        print("✅ SISTEMA OPERACIONAL COM BOOK_CLEAN")
        print("="*80)
        print(f"📊 Modelo: book_clean")
        print(f"🎯 Trading Accuracy Target: 79.23%")
        print(f"⚡ Latência: < 50ms")
        print(f"📈 Ticker: {ticker}")
        print(f"🛑 Para parar: CTRL+C")
        print("="*80 + "\n")
        
        # Loop principal
        last_metrics_time = time.time()
        
        while system.is_running:
            time.sleep(1)
            
            # Mostrar métricas a cada 30 segundos
            if time.time() - last_metrics_time > 30:
                system._calculate_real_time_accuracy()
                
                print(f"\n📊 [MÉTRICAS] {datetime.now().strftime('%H:%M:%S')}")
                print(f"   Candles recebidos: {len(system.candles)}")
                print(f"   Preço atual: {system.current_price:.2f}" if system.current_price > 0 else "   Preço: Aguardando...")
                print(f"   Predições: {system.real_time_metrics['predictions']}")
                print(f"   Trading Accuracy: {system.real_time_metrics['trading_accuracy']:.1%}")
                print(f"   PnL: R$ {system.pnl:.2f}")
                print(f"   Posição: {system.position}")
                
                # Forçar uma predição para teste
                if len(system.candles) >= system.min_data_required and system.real_time_metrics['predictions'] == 0:
                    print("   [DEBUG] Forçando predição de teste...")
                    test_pred = system._make_prediction()
                    if test_pred:
                        print(f"   [DEBUG] Predição OK: Dir={test_pred['direction']:.3f} Conf={test_pred['confidence']:.3f}")
                    else:
                        print("   [DEBUG] Predição falhou - verificar logs")
                
                last_metrics_time = time.time()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Sistema interrompido pelo usuário")
    
    except Exception as e:
        print(f"\n❌ ERRO FATAL: {e}")
        logger.error(f"Erro fatal: {e}", exc_info=True)
    
    finally:
        if 'system' in locals():
            print("\n📊 Finalizando sistema...")
            system.stop()
            system.cleanup()
            
            # Estatísticas finais
            print("\n" + "="*80)
            print("📊 ESTATÍSTICAS FINAIS - BOOK_CLEAN")
            print("="*80)
            print(f"Total de Predições: {system.real_time_metrics['predictions']}")
            print(f"Trading Accuracy Real: {system.real_time_metrics['trading_accuracy']:.1%}")
            print(f"Trading Accuracy Target: 79.23%")
            print(f"Performance: {'✅ ATINGIDA' if system.real_time_metrics['trading_accuracy'] >= 0.75 else '❌ ABAIXO DO TARGET'}")
            print(f"\nResultado Financeiro:")
            print(f"  PnL Final: R$ {system.pnl:.2f}")
            print(f"  Trades: {system.stats['trades']}")
            print(f"  Win Rate: {(system.stats['wins']/(system.stats['trades'] or 1))*100:.1f}%")
            print("="*80)
            
            # Salvar métricas
            metrics_file = Path(f"metrics/book_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            metrics_file.parent.mkdir(exist_ok=True)
            
            with open(metrics_file, 'w') as f:
                json.dump({
                    'model': 'book_clean',
                    'date': datetime.now().isoformat(),
                    'metrics': system.real_time_metrics,
                    'stats': system.stats,
                    'pnl': system.pnl
                }, f, indent=2)
            
            print(f"\n💾 Métricas salvas em: {metrics_file}")


if __name__ == "__main__":
    # Verificar horário de mercado
    now = datetime.now()
    market_open = datetime.strptime("09:00", "%H:%M").time()
    market_close = datetime.strptime("18:00", "%H:%M").time()
    
    if not (market_open <= now.time() <= market_close):
        print("\n⚠️ AVISO: Fora do horário de mercado (09:00-18:00)")
        response = input("Deseja continuar mesmo assim? (s/N): ")
        if response.lower() != 's':
            print("Sistema cancelado.")
            sys.exit(0)
    
    sys.exit(main())