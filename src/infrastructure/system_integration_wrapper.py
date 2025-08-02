"""
System Integration Wrapper - Conecta infraestrutura HMARL ao sistema atual
Mantém compatibilidade total com o sistema existente
"""

import logging
import threading
import time
from typing import Dict, Optional, Any
from datetime import datetime
import orjson

from src.connection_manager_v4 import ConnectionManagerV4
from src.data_structure import TradingDataStructure
from src.real_time_processor import RealTimeProcessor
from src.data_pipeline import DataPipeline
from .zmq_valkey_flow_setup import TradingInfrastructureWithFlow


class HMARLSystemWrapper:
    """
    Wrapper que integra a infraestrutura HMARL ao sistema atual
    sem quebrar nenhuma funcionalidade existente
    """
    
    def __init__(self, trading_system, hmarl_config: Dict):
        """
        Args:
            trading_system: Instância do TradingSystem atual
            hmarl_config: Configuração da infraestrutura HMARL
        """
        self.trading_system = trading_system
        self.config = hmarl_config
        self.logger = logging.getLogger(__name__)
        
        # Infraestrutura HMARL
        self.hmarl_infrastructure = None
        
        # Controle de threads
        self.running = False
        self.bridge_thread = None
        
        # Cache de dados para evitar duplicação
        self.last_processed_trade = None
        self.flow_analysis_enabled = True
        
        # Estatísticas
        self.stats = {
            'trades_processed': 0,
            'flow_events': 0,
            'errors': 0
        }
        
    def initialize(self) -> bool:
        """Inicializa o wrapper mantendo sistema atual intacto"""
        try:
            # 1. Verificar se sistema atual está funcionando
            if not hasattr(self.trading_system, 'connection'):
                self.logger.error("Sistema de trading não inicializado")
                return False
            
            # 2. Inicializar infraestrutura HMARL
            self.hmarl_infrastructure = TradingInfrastructureWithFlow(self.config)
            if not self.hmarl_infrastructure.initialize():
                self.logger.error("Falha ao inicializar infraestrutura HMARL")
                return False
            
            # 3. Interceptar callbacks sem modificar originais
            self._setup_callback_interceptors()
            
            # 4. Iniciar infraestrutura
            self.hmarl_infrastructure.start()
            
            self.logger.info("✅ HMARL System Wrapper inicializado com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro inicializando wrapper: {e}")
            return False
    
    def _setup_callback_interceptors(self):
        """
        Intercepta callbacks do sistema atual para adicionar análise de fluxo
        sem modificar o comportamento original
        """
        
        # Guardar referências dos callbacks originais
        connection = self.trading_system.connection
        
        if hasattr(connection, 'callback'):
            # Salvar callbacks originais
            self._original_trade_callback = connection.callback.on_trade_callback
            self._original_book_callback = connection.callback.on_order_book_callback
            
            # Criar callbacks aprimorados
            def enhanced_trade_callback(asset_id, timestamp, trade_data):
                """Callback aprimorado com análise de fluxo"""
                
                # 1. Executar callback original PRIMEIRO (manter comportamento)
                self._original_trade_callback(asset_id, timestamp, trade_data)
                
                # 2. Adicionar análise de fluxo em paralelo
                if self.flow_analysis_enabled:
                    try:
                        # Preparar dados para análise
                        flow_trade_data = {
                            'symbol': asset_id.ticker if hasattr(asset_id, 'ticker') else str(asset_id),
                            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                            'price': float(trade_data.get('price', 0)),
                            'volume': int(trade_data.get('volume', 0)),
                            'trade_type': int(trade_data.get('trade_type', 0))
                        }
                        
                        # Publicar com análise de fluxo
                        self.hmarl_infrastructure.publish_tick_with_flow(flow_trade_data)
                        
                        self.stats['trades_processed'] += 1
                        
                    except Exception as e:
                        self.stats['errors'] += 1
                        self.logger.error(f"Erro em análise de fluxo: {e}")
            
            def enhanced_book_callback(asset_id, book_data):
                """Callback aprimorado para book"""
                
                # 1. Executar callback original
                self._original_book_callback(asset_id, book_data)
                
                # 2. Atualizar monitor de liquidez
                if self.flow_analysis_enabled and hasattr(self.hmarl_infrastructure, 'liquidity_monitor'):
                    try:
                        self.hmarl_infrastructure.liquidity_monitor.update_book(book_data)
                    except Exception as e:
                        self.logger.error(f"Erro atualizando liquidez: {e}")
            
            # Substituir callbacks
            connection.callback.on_trade_callback = enhanced_trade_callback
            connection.callback.on_order_book_callback = enhanced_book_callback
            
            self.logger.info("📡 Callbacks interceptados com sucesso")
    
    def start_bridge(self):
        """Inicia ponte entre sistema atual e HMARL"""
        if self.running:
            self.logger.warning("Bridge já está rodando")
            return
        
        self.running = True
        self.bridge_thread = threading.Thread(target=self._bridge_worker, daemon=True)
        self.bridge_thread.start()
        
        self.logger.info("🌉 Bridge Sistema-HMARL iniciada")
    
    def _bridge_worker(self):
        """Worker thread para processar dados adicionais"""
        
        while self.running:
            try:
                # Aqui podemos adicionar processamento adicional se necessário
                # Por exemplo, sincronizar features calculadas pelo sistema atual
                
                # Por enquanto, apenas monitorar saúde do sistema
                if hasattr(self.trading_system, 'data') and hasattr(self.trading_system.data, 'check_data_quality'):
                    quality_check = self.trading_system.data.check_data_quality()
                    
                    # Se qualidade estiver baixa, podemos ajustar parâmetros
                    if quality_check.get('has_recent_data', False) is False:
                        self.logger.warning("Sistema sem dados recentes")
                
                # Sleep para não sobrecarregar
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Erro no bridge worker: {e}")
                time.sleep(5)
    
    def get_flow_enhanced_features(self, symbol: str, lookback_minutes: int = 5) -> Dict[str, Any]:
        """
        Retorna features aprimoradas com análise de fluxo
        para serem usadas pelo sistema ML atual
        """
        
        try:
            # 1. Buscar histórico de fluxo
            flow_history = self.hmarl_infrastructure.get_flow_history(symbol, lookback_minutes)
            
            if not flow_history:
                return {}
            
            # 2. Calcular features agregadas
            features = {
                'flow_ofi_1m': 0.0,
                'flow_ofi_5m': 0.0,
                'flow_volume_imbalance': 0.0,
                'flow_aggression_ratio': 0.0,
                'flow_large_trade_ratio': 0.0,
                'tape_speed': 0.0,
                'liquidity_score': 0.0
            }
            
            # Processar dados mais recentes
            if flow_history:
                latest = flow_history[-1]
                if 'analysis' in latest and isinstance(latest['analysis'], dict):
                    analysis = latest['analysis']
                    
                    # OFI values
                    if 'ofi' in analysis:
                        features['flow_ofi_1m'] = analysis['ofi'].get('1', 0.0)
                        features['flow_ofi_5m'] = analysis['ofi'].get('5', 0.0)
                    
                    # Outras métricas
                    features['flow_volume_imbalance'] = analysis.get('volume_imbalance', 0.0)
                    features['flow_aggression_ratio'] = analysis.get('aggression_ratio', 0.5)
                    features['flow_large_trade_ratio'] = analysis.get('large_trade_ratio', 0.0)
                
                # Velocidade do tape
                features['tape_speed'] = float(latest.get('speed', 0.0))
            
            # 3. Adicionar métricas de liquidez se disponível
            if hasattr(self.hmarl_infrastructure, 'liquidity_monitor'):
                if self.hmarl_infrastructure.liquidity_monitor.liquidity_history:
                    latest_liquidity = self.hmarl_infrastructure.liquidity_monitor.liquidity_history[-1]
                    features['liquidity_score'] = latest_liquidity['metrics'].get('liquidity_score', 0.0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Erro obtendo features de fluxo: {e}")
            return {}
    
    def inject_flow_features(self, existing_features: Dict) -> Dict:
        """
        Injeta features de fluxo nas features existentes do sistema
        Método não-invasivo que apenas adiciona novas features
        """
        
        try:
            # Obter símbolo atual do sistema
            symbol = self.config.get('symbol', 'WDOH25')
            
            # Buscar features de fluxo
            flow_features = self.get_flow_enhanced_features(symbol)
            
            # Criar cópia para não modificar original
            enhanced_features = existing_features.copy()
            
            # Adicionar features de fluxo
            for key, value in flow_features.items():
                if key not in enhanced_features:  # Não sobrescrever existentes
                    enhanced_features[key] = value
            
            return enhanced_features
            
        except Exception as e:
            self.logger.error(f"Erro injetando features de fluxo: {e}")
            return existing_features  # Retornar original em caso de erro
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do wrapper"""
        
        infra_metrics = {}
        if self.hmarl_infrastructure:
            infra_metrics = self.hmarl_infrastructure.get_performance_metrics()
        
        return {
            'wrapper_stats': self.stats,
            'infrastructure_metrics': infra_metrics,
            'flow_analysis_enabled': self.flow_analysis_enabled,
            'bridge_running': self.running
        }
    
    def enable_flow_analysis(self, enabled: bool = True):
        """Habilita/desabilita análise de fluxo"""
        self.flow_analysis_enabled = enabled
        self.logger.info(f"Análise de fluxo: {'HABILITADA' if enabled else 'DESABILITADA'}")
    
    def stop(self):
        """Para o wrapper de forma limpa"""
        
        self.running = False
        
        # Aguardar thread terminar
        if self.bridge_thread and self.bridge_thread.is_alive():
            self.bridge_thread.join(timeout=5)
        
        # Restaurar callbacks originais
        if hasattr(self, '_original_trade_callback'):
            self.trading_system.connection.callback.on_trade_callback = self._original_trade_callback
        if hasattr(self, '_original_book_callback'):
            self.trading_system.connection.callback.on_order_book_callback = self._original_book_callback
        
        # Parar infraestrutura
        if self.hmarl_infrastructure:
            self.hmarl_infrastructure.stop()
        
        self.logger.info("🛑 HMARL System Wrapper parado")


class FlowEnhancedFeatureEngine:
    """
    Feature Engine aprimorado que adiciona features de fluxo
    ao FeatureEngine existente
    """
    
    def __init__(self, original_feature_engine, hmarl_wrapper):
        self.original_engine = original_feature_engine
        self.hmarl_wrapper = hmarl_wrapper
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: TradingDataStructure) -> Dict[str, Any]:
        """
        Calcula features originais + features de fluxo
        """
        
        # 1. Calcular features originais
        original_features = self.original_engine.calculate(data)
        
        # 2. Se HMARL habilitado, adicionar features de fluxo
        if self.hmarl_wrapper and self.hmarl_wrapper.flow_analysis_enabled:
            try:
                # Obter última linha de features ML (mais recente)
                if 'ml_features' in original_features and not original_features['ml_features'].empty:
                    latest_features = original_features['ml_features'].iloc[-1].to_dict()
                    
                    # Injetar features de fluxo
                    enhanced_features = self.hmarl_wrapper.inject_flow_features(latest_features)
                    
                    # Atualizar DataFrame
                    for key, value in enhanced_features.items():
                        if key not in original_features['ml_features'].columns:
                            original_features['ml_features'][key] = value
                
            except Exception as e:
                self.logger.error(f"Erro adicionando features de fluxo: {e}")
        
        return original_features


def integrate_hmarl_with_system(trading_system, hmarl_config: Optional[Dict] = None) -> HMARLSystemWrapper:
    """
    Função helper para integrar HMARL com sistema existente
    
    Args:
        trading_system: Instância do TradingSystem atual
        hmarl_config: Configuração HMARL (usa padrão se None)
    
    Returns:
        HMARLSystemWrapper configurado e inicializado
    """
    
    if hmarl_config is None:
        # Configuração padrão
        hmarl_config = {
            'symbol': 'WDOH25',  # Ajustar conforme necessário
            'zmq': {
                'tick_port': 5555,
                'book_port': 5556,
                'flow_port': 5557,
                'footprint_port': 5558,
                'liquidity_port': 5559,
                'tape_port': 5560
            },
            'valkey': {
                'host': 'localhost',
                'port': 6379,
                'stream_maxlen': 100000
            },
            'flow': {
                'ofi_windows': [1, 5, 15, 30, 60],
                'min_confidence': 0.3
            }
        }
    
    # Criar e inicializar wrapper
    wrapper = HMARLSystemWrapper(trading_system, hmarl_config)
    
    if wrapper.initialize():
        wrapper.start_bridge()
        
        # Opcionalmente, substituir feature engine
        if hasattr(trading_system, 'feature_engine'):
            enhanced_engine = FlowEnhancedFeatureEngine(
                trading_system.feature_engine,
                wrapper
            )
            # Aqui você pode decidir se quer substituir ou não
            # trading_system.feature_engine = enhanced_engine
        
        return wrapper
    else:
        raise RuntimeError("Falha ao integrar HMARL com sistema")


# Exemplo de uso:
"""
# No seu main.py ou trading_system.py, após inicializar o sistema:

from src.infrastructure.system_integration_wrapper import integrate_hmarl_with_system

# Sistema existente
trading_system = TradingSystem(config)
trading_system.initialize()

# Adicionar HMARL
hmarl_wrapper = integrate_hmarl_with_system(trading_system)

# Agora o sistema funciona normalmente COM análise de fluxo adicional
trading_system.start('WDOH25')

# Para desabilitar temporariamente análise de fluxo:
# hmarl_wrapper.enable_flow_analysis(False)

# Para obter estatísticas:
# stats = hmarl_wrapper.get_stats()
"""