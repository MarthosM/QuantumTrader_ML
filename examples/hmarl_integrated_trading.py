"""
Exemplo de Trading Integrado com HMARL
Demonstra como usar o sistema ML com análise de fluxo HMARL
"""

import logging
import time
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Adicionar src ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading_system import TradingSystem
from src.infrastructure.hmarl_ml_integration import integrate_hmarl_with_ml_system
from src.training.dual_training_system import DualTrainingSystem
from src.agents.flow_aware_base_agent import FlowAwareBaseAgent
from src.agents.order_flow_specialist import OrderFlowSpecialist
from src.agents.liquidity_specialist import LiquiditySpecialist


class HMARLIntegratedTrading:
    """
    Sistema de trading completo integrando:
    1. Sistema ML tradicional (tick-only)
    2. Análise de book em tempo real
    3. Agentes HMARL para análise de fluxo
    4. Coordenação flow-aware
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('HMARLIntegratedTrading')
        
        # Componentes principais
        self.trading_system = None
        self.hmarl_bridge = None
        self.dual_trainer = None
        self.flow_agents = []
        
        # Estado
        self.is_running = False
        self.current_symbol = None
        
    def initialize(self):
        """Inicializa todos os componentes"""
        self.logger.info("=== Inicializando Sistema Integrado HMARL ===")
        
        # 1. Criar sistema de trading base
        self.logger.info("1. Criando sistema de trading...")
        self.trading_system = TradingSystem(self.config['trading'])
        self.trading_system.initialize()
        
        # 2. Integrar HMARL
        self.logger.info("2. Integrando HMARL...")
        self.hmarl_bridge = integrate_hmarl_with_ml_system(
            self.trading_system,
            self.config['hmarl']
        )
        
        # 3. Criar sistema de treinamento dual
        self.logger.info("3. Configurando treinamento dual...")
        self.dual_trainer = DualTrainingSystem(self.config['training'])
        
        # 4. Inicializar agentes de fluxo
        self.logger.info("4. Inicializando agentes HMARL...")
        self._initialize_flow_agents()
        
        self.logger.info("✅ Sistema integrado inicializado com sucesso!")
        
    def _initialize_flow_agents(self):
        """Inicializa agentes especializados em análise de fluxo"""
        agent_configs = [
            {
                'type': 'order_flow_specialist',
                'class': OrderFlowSpecialist,
                'config': {
                    'min_confidence': 0.6,
                    'ofi_threshold': 0.3,
                    'use_registry': True
                }
            },
            {
                'type': 'liquidity_specialist',
                'class': LiquiditySpecialist,
                'config': {
                    'min_confidence': 0.5,
                    'depth_analysis_levels': 5,
                    'use_registry': True
                }
            }
        ]
        
        for agent_info in agent_configs:
            try:
                agent = agent_info['class'](agent_info['config'])
                self.flow_agents.append(agent)
                
                # Iniciar agent em thread separada
                import threading
                agent_thread = threading.Thread(
                    target=agent.run_enhanced_agent_loop,
                    daemon=True
                )
                agent_thread.start()
                
                self.logger.info(f"✅ Agente {agent_info['type']} iniciado")
                
            except Exception as e:
                self.logger.error(f"❌ Erro ao criar agente {agent_info['type']}: {e}")
                
    def train_models(self, symbol: str):
        """Treina modelos dual (tick-only e book-enhanced)"""
        self.logger.info(f"\n=== Treinamento de Modelos para {symbol} ===")
        
        # 1. Treinar modelos tick-only (1 ano de dados)
        self.logger.info("Treinando modelos tick-only...")
        tick_results = self.dual_trainer.train_tick_only_models(
            symbols=[symbol],
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now()
        )
        
        # 2. Treinar modelos book-enhanced (30 dias com book)
        self.logger.info("Treinando modelos book-enhanced...")
        book_results = self.dual_trainer.train_book_enhanced_models(
            symbols=[symbol],
            lookback_days=30
        )
        
        # 3. Criar estratégia híbrida
        self.logger.info("Criando estratégia híbrida...")
        hybrid_strategy = self.dual_trainer.create_hybrid_strategy(symbol)
        
        # Salvar resultados
        self.dual_trainer.save_training_report({
            'symbol': symbol,
            'tick_results': tick_results,
            'book_results': book_results,
            'hybrid_strategy': hybrid_strategy,
            'timestamp': datetime.now()
        })
        
        return hybrid_strategy
        
    def start_trading(self, symbol: str):
        """Inicia trading com sistema integrado"""
        self.current_symbol = symbol
        self.is_running = True
        
        self.logger.info(f"\n=== Iniciando Trading Integrado para {symbol} ===")
        self.logger.info("Componentes ativos:")
        self.logger.info("- Sistema ML tradicional")
        self.logger.info("- Análise de book em tempo real")
        self.logger.info(f"- {len(self.flow_agents)} agentes HMARL")
        self.logger.info("- Coordenação flow-aware")
        
        # Iniciar sistema de trading
        self.trading_system.start(symbol)
        
        # Loop principal
        self._trading_loop()
        
    def _trading_loop(self):
        """Loop principal de trading integrado"""
        last_status = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Status periódico
                if current_time - last_status > 30:  # A cada 30 segundos
                    self._print_system_status()
                    last_status = current_time
                    
                # Obter estado aprimorado do mercado
                market_state = self.hmarl_bridge.get_enhanced_market_state(self.current_symbol)
                
                # Verificar consenso de fluxo
                flow_consensus = self.hmarl_bridge.get_flow_consensus(self.current_symbol)
                
                if flow_consensus and flow_consensus['strength'] > 0.7:
                    self.logger.info(f"🎯 Consenso de fluxo forte: {flow_consensus['direction']} "
                                   f"(força: {flow_consensus['strength']:.2f})")
                    
                time.sleep(0.1)  # Pequena pausa
                
            except KeyboardInterrupt:
                self.logger.info("Interrupção do usuário detectada")
                break
            except Exception as e:
                self.logger.error(f"Erro no loop de trading: {e}")
                
    def _print_system_status(self):
        """Imprime status do sistema"""
        self.logger.info("\n=== Status do Sistema ===")
        
        # Status do trading system
        if hasattr(self.trading_system, 'get_status'):
            ml_status = self.trading_system.get_status()
            self.logger.info(f"ML System: {ml_status}")
            
        # Status dos agentes
        active_agents = sum(1 for agent in self.flow_agents if agent.is_active)
        self.logger.info(f"Agentes HMARL ativos: {active_agents}/{len(self.flow_agents)}")
        
        # Cache de flow features
        if self.hmarl_bridge:
            cache_size = len(self.hmarl_bridge.flow_features_cache)
            self.logger.info(f"Flow features em cache: {cache_size}")
            
        # Última análise de fluxo
        if self.current_symbol in self.hmarl_bridge.last_flow_analysis:
            last_time = self.hmarl_bridge.last_flow_analysis[self.current_symbol]
            elapsed = time.time() - last_time
            self.logger.info(f"Última análise de fluxo: {elapsed:.1f}s atrás")
            
    def stop_trading(self):
        """Para o sistema de trading"""
        self.logger.info("\n=== Parando Sistema Integrado ===")
        
        self.is_running = False
        
        # Parar trading system
        if self.trading_system:
            self.trading_system.stop()
            
        # Desligar agentes
        for agent in self.flow_agents:
            agent.shutdown()
            
        # Desligar bridge HMARL
        if self.hmarl_bridge:
            self.hmarl_bridge.shutdown()
            
        self.logger.info("✅ Sistema integrado parado com sucesso")
        
    def run_backtest(self, symbol: str, start_date: datetime, end_date: datetime):
        """Executa backtest com dados históricos"""
        self.logger.info(f"\n=== Backtest Integrado {symbol} ===")
        self.logger.info(f"Período: {start_date.date()} até {end_date.date()}")
        
        # TODO: Implementar backtest que usa tanto tick quanto book data
        # com simulação de agentes HMARL
        
        results = {
            'symbol': symbol,
            'period': f"{start_date} to {end_date}",
            'status': 'not_implemented'
        }
        
        return results


def main():
    """Exemplo principal de uso do sistema integrado"""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuração completa
    config = {
        'trading': {
            'api_version': '4.0.0.30',
            'data_path': 'data/',
            'models_path': 'models/',
            'enable_risk_management': True,
            'max_position_size': 10,
            'max_daily_loss': -5000.0
        },
        'hmarl': {
            'symbol': 'WDOU25',
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
                'stream_maxlen': 100000,
                'ttl_days': 30
            },
            'flow': {
                'ofi_windows': [1, 5, 15, 30, 60],
                'trade_size_thresholds': {
                    'small': 5,
                    'medium': 20,
                    'large': 50,
                    'whale': 100
                }
            }
        },
        'training': {
            'tick_data_path': 'data/historical',
            'book_data_path': 'data/realtime/book',
            'models_path': 'models'
        }
    }
    
    # Criar sistema integrado
    system = HMARLIntegratedTrading(config)
    
    try:
        # Inicializar
        system.initialize()
        
        # Símbolo para operar
        symbol = 'WDOU25'
        
        # Opção 1: Treinar modelos
        print("\n1. Deseja treinar novos modelos? (s/n): ", end='')
        if input().lower() == 's':
            hybrid_strategy = system.train_models(symbol)
            print(f"\nEstratégia híbrida criada:")
            print(f"- Componentes: {hybrid_strategy['components']}")
            print(f"- HMARL: {hybrid_strategy['hmarl_integration']['enabled']}")
            
        # Opção 2: Iniciar trading
        print("\n2. Deseja iniciar trading ao vivo? (s/n): ", end='')
        if input().lower() == 's':
            print(f"\nIniciando trading para {symbol}...")
            print("Pressione Ctrl+C para parar\n")
            system.start_trading(symbol)
            
    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário")
    except Exception as e:
        logging.error(f"Erro: {e}", exc_info=True)
    finally:
        system.stop_trading()
        print("\nSistema finalizado")


if __name__ == "__main__":
    main()