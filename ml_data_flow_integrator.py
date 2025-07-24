#!/usr/bin/env python3
"""
Integrador de Fluxo de Dados e GUI
Sistema de Trading v2.0

Este módulo integra:
1. DataFlowMonitor - Monitora fluxo de dados
2. GUI Extensions - Exibe predições no monitor
3. Trading System - Sistema principal

Garante que a cada novo candle:
- Features são calculadas
- Predição é executada  
- Resultado é exibido no GUI
"""

import os
import sys
import logging
import threading
import time
import pandas as pd
from typing import Optional, Dict, Any

# Adicionar src ao path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Imports dos módulos criados
from data_flow_monitor import DataFlowMonitor
from gui_prediction_extension import extend_gui_with_prediction_display


class MLDataFlowIntegrator:
    """
    Integrador completo do fluxo ML
    Conecta dados → features → predições → GUI
    """
    
    def __init__(self, trading_system=None):
        self.logger = logging.getLogger('MLIntegrator')
        self.trading_system = trading_system
        
        # Componentes principais
        self.data_flow_monitor = None
        self.gui_extended = False
        
        # Estado de integração
        self.integration_active = False
        self.integration_thread = None
        
        # Configurações
        self.update_interval = 2.0  # segundos
        self.max_retry_attempts = 3
        
    def initialize(self) -> bool:
        """Inicializa todos os componentes"""
        try:
            self.logger.info("🔧 Inicializando integrador ML...")
            
            # 1. Verificar sistema de trading
            if not self.trading_system:
                self.logger.error("Sistema de trading não fornecido")
                return False
                
            # 2. Inicializar monitor de fluxo de dados
            self.data_flow_monitor = DataFlowMonitor(self.trading_system)
            
            # 3. Estender GUI se disponível
            if hasattr(self.trading_system, 'monitor') and self.trading_system.monitor:
                success = extend_gui_with_prediction_display(self.trading_system.monitor)
                if success:
                    self.gui_extended = True
                    self.logger.info("✅ GUI estendido com painéis de predição")
                else:
                    self.logger.warning("⚠️ Falha ao estender GUI")
            else:
                self.logger.info("ℹ️ GUI não disponível, continuando sem extensão visual")
                
            # 4. Configurar hooks no sistema
            self._setup_system_hooks()
            
            self.logger.info("✅ Integrador ML inicializado com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro inicializando integrador: {e}")
            return False
            
    def start_integration(self) -> bool:
        """Inicia a integração ativa"""
        if self.integration_active:
            self.logger.warning("Integração já está ativa")
            return True
            
        try:
            # Inicializar se necessário
            if not self.data_flow_monitor:
                if not self.initialize():
                    return False
                    
            # Iniciar monitor de fluxo
            self.data_flow_monitor.start_monitoring()
            
            # Iniciar thread de integração
            self.integration_active = True
            self.integration_thread = threading.Thread(
                target=self._integration_loop,
                daemon=True,
                name="MLIntegration"
            )
            self.integration_thread.start()
            
            self.logger.info("🚀 Integração ML iniciada")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro iniciando integração: {e}")
            return False
            
    def stop_integration(self):
        """Para a integração"""
        try:
            self.integration_active = False
            
            # Parar monitor de fluxo
            if self.data_flow_monitor:
                self.data_flow_monitor.stop_monitoring()
                
            # Aguardar thread de integração
            if self.integration_thread and self.integration_thread.is_alive():
                self.integration_thread.join(timeout=5)
                
            self.logger.info("⏹️ Integração ML parada")
            
        except Exception as e:
            self.logger.error(f"Erro parando integração: {e}")
            
    def _integration_loop(self):
        """Loop principal de integração"""
        self.logger.info("Loop de integração iniciado")
        
        while self.integration_active:
            try:
                # Atualizar GUI com dados do monitor
                self._update_gui_with_flow_data()
                
                # Aguardar próxima atualização
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Erro no loop de integração: {e}")
                time.sleep(5)  # Aguardar mais tempo em caso de erro
                
    def _update_gui_with_flow_data(self):
        """Atualiza GUI com dados do monitor de fluxo"""
        if not self.gui_extended or not self.data_flow_monitor:
            return
            
        try:
            # Obter resumo do fluxo
            flow_summary = self.data_flow_monitor.get_flow_summary()
            
            # Atualizar status do fluxo no GUI
            if hasattr(self.trading_system.monitor, 'update_flow_status'):
                self.trading_system.monitor.update_flow_status(flow_summary)
                
            # Se há predição atual, atualizar GUI
            if self.data_flow_monitor.current_prediction:
                prediction_data = self._format_prediction_for_gui(
                    self.data_flow_monitor.current_prediction,
                    self.data_flow_monitor.current_features
                )
                
                if hasattr(self.trading_system.monitor, 'update_prediction_data'):
                    self.trading_system.monitor.update_prediction_data(prediction_data)
                    
        except Exception as e:
            self.logger.error(f"Erro atualizando GUI: {e}")
            
    def _format_prediction_for_gui(self, prediction, features_df) -> Dict:
        """Formata dados de predição para o GUI"""
        try:
            # Extrair dados da predição
            prediction_data = {
                'timestamp': prediction.timestamp.strftime('%H:%M:%S'),
                'direction': prediction.direction,
                'magnitude': prediction.magnitude,
                'confidence': prediction.confidence,
                'regime': prediction.regime,
                'model': prediction.model_used,
                'processing_time': f"{prediction.processing_time:.3f}s"
            }
            
            # Extrair dados das features
            features_data = {
                'count': prediction.features_count,
                'sample': prediction.raw_features or {},
            }
            
            # Adicionar últimos valores se features_df disponível
            if features_df is not None and not features_df.empty:
                try:
                    last_values = {}
                    last_row = features_df.iloc[-1]
                    
                    # Pegar algumas features importantes
                    important_features = ['close', 'open', 'high', 'low', 'volume', 
                                        'ema_9', 'ema_20', 'rsi', 'atr']
                    
                    for feature in important_features:
                        if feature in features_df.columns:
                            val = last_row[feature]
                            if not pd.isna(val):
                                last_values[feature] = float(val)
                                
                    features_data['last_values'] = last_values
                    
                except Exception as e:
                    self.logger.debug(f"Erro extraindo últimos valores: {e}")
                    
            return {
                'prediction': prediction_data,
                'features': features_data
            }
            
        except Exception as e:
            self.logger.error(f"Erro formatando predição: {e}")
            return {}
            
    def _setup_system_hooks(self):
        """Configura hooks no sistema para integração"""
        try:
            # Hook para novos candles (se disponível)
            if hasattr(self.trading_system, 'data_structure'):
                # Verificar se podemos adicionar callback
                data_structure = self.trading_system.data_structure
                if hasattr(data_structure, 'add_candle_callback'):
                    data_structure.add_candle_callback(self._on_new_candle)
                    self.logger.info("✅ Hook de novos candles configurado")
                    
            # Hook para predições (se disponível)
            if hasattr(self.trading_system, 'ml_coordinator'):
                ml_coordinator = self.trading_system.ml_coordinator
                if hasattr(ml_coordinator, 'add_prediction_callback'):
                    ml_coordinator.add_prediction_callback(self._on_new_prediction)
                    self.logger.info("✅ Hook de predições configurado")
                    
        except Exception as e:
            self.logger.warning(f"Erro configurando hooks: {e}")
            
    def _on_new_candle(self, candle_data):
        """Callback chamado quando novo candle é recebido"""
        try:
            self.logger.debug(f"Novo candle recebido: {candle_data}")
            
            # O DataFlowMonitor já detectará automaticamente
            # Apenas logar aqui para debug
            
        except Exception as e:
            self.logger.error(f"Erro processando novo candle: {e}")
            
    def _on_new_prediction(self, prediction_result):
        """Callback chamado quando nova predição é feita"""
        try:
            self.logger.debug(f"Nova predição recebida: {prediction_result}")
            
            # Atualizar GUI imediatamente se disponível
            if self.gui_extended and hasattr(self.trading_system.monitor, 'update_prediction_data'):
                prediction_data = self._format_prediction_for_gui(
                    prediction_result, 
                    self.data_flow_monitor.current_features if self.data_flow_monitor else None
                )
                self.trading_system.monitor.update_prediction_data(prediction_data)
                
        except Exception as e:
            self.logger.error(f"Erro processando nova predição: {e}")
            
    def get_integration_status(self) -> Dict[str, Any]:
        """Retorna status da integração"""
        try:
            status: Dict[str, Any] = {
                'initialized': self.data_flow_monitor is not None,
                'gui_extended': self.gui_extended,
                'integration_active': self.integration_active,
                'has_trading_system': self.trading_system is not None
            }
            
            # Adicionar dados do monitor se disponível
            if self.data_flow_monitor:
                flow_summary = self.data_flow_monitor.get_flow_summary()
                status['flow_summary'] = flow_summary
                status['current_prediction'] = self.data_flow_monitor.current_prediction is not None
                status['current_features'] = self.data_flow_monitor.current_features is not None
                
            return status
            
        except Exception as e:
            self.logger.error(f"Erro obtendo status: {e}")
            return {'error': str(e)}
            
    def print_integration_status(self):
        """Imprime status da integração"""
        status = self.get_integration_status()
        
        print(f"\n{'='*60}")
        print(f"ML DATA FLOW INTEGRATOR - STATUS")
        print(f"{'='*60}")
        print(f"⏰ Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔧 Inicializado: {'✅' if status.get('initialized') else '❌'}")
        print(f"🖥️ GUI Estendido: {'✅' if status.get('gui_extended') else '❌'}")
        print(f"🔄 Integração Ativa: {'✅' if status.get('integration_active') else '❌'}")
        print(f"🏭 Sistema Trading: {'✅' if status.get('has_trading_system') else '❌'}")
        
        if 'flow_summary' in status:
            flow = status['flow_summary']
            print(f"\n📊 Resumo do Fluxo:")
            print(f"  • Total Fluxos: {flow.get('total_flows_processed', 0)}")
            print(f"  • Total Predições: {flow.get('total_predictions', 0)}")
            print(f"  • Total Erros: {flow.get('total_errors', 0)}")
            print(f"  • Predição Atual: {'✅' if status.get('current_prediction') else '❌'}")
            print(f"  • Features Atuais: {'✅' if status.get('current_features') else '❌'}")
            
        print(f"{'='*60}\n")


def integrate_ml_data_flow_with_system(trading_system) -> Optional[MLDataFlowIntegrator]:
    """
    Função utilitária para integrar fluxo ML com sistema de trading
    
    Args:
        trading_system: Instância do sistema de trading
        
    Returns:
        MLDataFlowIntegrator configurado ou None se falhar
    """
    logger = logging.getLogger('MLIntegration')
    
    try:
        logger.info("🔧 Configurando integração ML com sistema...")
        
        # Criar integrador
        integrator = MLDataFlowIntegrator(trading_system)
        
        # Inicializar
        if not integrator.initialize():
            logger.error("Falha na inicialização do integrador")
            return None
            
        # Iniciar integração
        if not integrator.start_integration():
            logger.error("Falha ao iniciar integração")
            return None
            
        logger.info("✅ Integração ML configurada com sucesso")
        
        # Adicionar referência no sistema
        trading_system.ml_integrator = integrator
        
        return integrator
        
    except Exception as e:
        logger.error(f"Erro configurando integração ML: {e}")
        return None


def main():
    """Teste do integrador"""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('IntegratorTest')
    logger.info("Testando ML Data Flow Integrator...")
    
    # Criar integrador sem sistema (para teste)
    integrator = MLDataFlowIntegrator()
    
    # Testar métodos básicos
    integrator.print_integration_status()
    
    logger.info("✓ Teste do integrador concluído")


if __name__ == "__main__":
    main()
