"""
Script para executar o sistema HMARL completo
"""

import sys
import os
import time
import logging
import threading
import signal
from datetime import datetime

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.order_flow_specialist import OrderFlowSpecialistAgent
from agents.footprint_pattern_agent import FootprintPatternAgent
from agents.liquidity_agent import LiquidityAgent
from agents.tape_reading_agent import TapeReadingAgent
from coordination.agent_registry import AgentRegistry
from coordination.flow_aware_coordinator import FlowAwareCoordinator
from systems.flow_aware_feedback_system import FlowAwareFeedbackSystem


class HMARLSystem:
    """Sistema HMARL completo"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Componentes do sistema
        self.registry = None
        self.agents = []
        self.coordinator = None
        self.feedback_system = None
        
        # Control
        self.is_running = False
        
    def initialize(self):
        """Inicializa todos os componentes"""
        self.logger.info("Inicializando sistema HMARL...")
        
        # 1. Iniciar Registry
        self.logger.info("Iniciando Registry...")
        self.registry = AgentRegistry()
        time.sleep(1)  # Dar tempo para inicializar
        
        # 2. Criar agentes
        self.logger.info("Criando agentes especializados...")
        
        agent_configs = {
            'min_confidence': 0.4,
            'use_registry': True
        }
        
        # Order Flow Specialist
        self.agents.append(OrderFlowSpecialistAgent({
            **agent_configs,
            'ofi_threshold': 0.3,
            'delta_threshold': 100
        }))
        self.logger.info("  ✓ OrderFlowSpecialistAgent criado")
        
        # Footprint Pattern Agent
        self.agents.append(FootprintPatternAgent({
            **agent_configs,
            'min_pattern_confidence': 0.5,
            'prediction_weight': 0.7
        }))
        self.logger.info("  ✓ FootprintPatternAgent criado")
        
        # Liquidity Agent
        self.agents.append(LiquidityAgent({
            **agent_configs,
            'min_liquidity_score': 0.3,
            'imbalance_threshold': 0.3
        }))
        self.logger.info("  ✓ LiquidityAgent criado")
        
        # Tape Reading Agent
        self.agents.append(TapeReadingAgent({
            **agent_configs,
            'min_pattern_confidence': 0.5,
            'speed_weight': 0.3
        }))
        self.logger.info("  ✓ TapeReadingAgent criado")
        
        # 3. Criar Coordenador
        self.logger.info("Criando coordenador...")
        self.coordinator = FlowAwareCoordinator()
        
        # 4. Criar Sistema de Feedback
        self.logger.info("Criando sistema de feedback...")
        self.feedback_system = FlowAwareFeedbackSystem()
        
        self.logger.info("✅ Sistema HMARL inicializado com sucesso!")
        
    def start(self):
        """Inicia o sistema"""
        self.logger.info("Iniciando sistema HMARL...")
        self.is_running = True
        
        # Iniciar threads dos agentes
        agent_threads = []
        for agent in self.agents:
            thread = threading.Thread(
                target=agent.run_enhanced_agent_loop,
                name=f"Agent-{agent.agent_id}",
                daemon=True
            )
            thread.start()
            agent_threads.append(thread)
            self.logger.info(f"  ✓ Thread do agente {agent.agent_id} iniciada")
            
        # Iniciar thread do coordenador
        coordinator_thread = threading.Thread(
            target=self.coordinator.run_coordination_loop,
            name="Coordinator",
            daemon=True
        )
        coordinator_thread.start()
        self.logger.info("  ✓ Thread do coordenador iniciada")
        
        # Thread de monitoramento
        monitor_thread = threading.Thread(
            target=self._monitor_system,
            name="Monitor",
            daemon=True
        )
        monitor_thread.start()
        
        self.logger.info("✅ Sistema HMARL em execução!")
        
        # Manter sistema rodando
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Interrupção detectada...")
            self.stop()
            
    def _monitor_system(self):
        """Monitora o estado do sistema"""
        last_stats_time = time.time()
        stats_interval = 30  # segundos
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Imprimir estatísticas periodicamente
                if current_time - last_stats_time >= stats_interval:
                    self._print_system_stats()
                    last_stats_time = current_time
                    
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Erro no monitoramento: {e}")
                
    def _print_system_stats(self):
        """Imprime estatísticas do sistema"""
        print("\n" + "="*60)
        print(f"ESTATÍSTICAS DO SISTEMA - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Estatísticas do Registry
        if self.registry:
            stats = self.registry.get_registry_stats()
            print(f"\nRegistry:")
            print(f"  Agentes totais: {stats['total_agents']}")
            print(f"  Agentes ativos: {stats['active_agents']}")
            print(f"  Distribuição por tipo: {stats['type_distribution']}")
            
        # Estatísticas do Coordenador
        if self.coordinator:
            stats = self.coordinator.get_coordination_stats()
            print(f"\nCoordenador:")
            print(f"  Sinais recebidos: {stats['total_signals_received']}")
            print(f"  Janelas ativas: {stats['active_windows']}")
            
        # Estado dos agentes
        print(f"\nAgentes:")
        for agent in self.agents:
            state = agent.get_state_summary()
            print(f"  {agent.agent_id}:")
            print(f"    Ativo: {state['is_active']}")
            print(f"    Memória: {state['memory_size']} experiências")
            
        print("="*60 + "\n")
        
    def stop(self):
        """Para o sistema"""
        self.logger.info("Parando sistema HMARL...")
        self.is_running = False
        
        # Parar agentes
        for agent in self.agents:
            try:
                agent.shutdown()
                self.logger.info(f"  ✓ Agente {agent.agent_id} parado")
            except Exception as e:
                self.logger.error(f"  ✗ Erro parando agente {agent.agent_id}: {e}")
                
        # Parar coordenador
        if self.coordinator:
            self.coordinator.shutdown()
            self.logger.info("  ✓ Coordenador parado")
            
        # Parar registry
        if self.registry:
            self.registry.shutdown()
            self.logger.info("  ✓ Registry parado")
            
        self.logger.info("✅ Sistema HMARL parado com sucesso!")


def main():
    """Função principal"""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/hmarl_system.log')
        ]
    )
    
    print("\n" + "="*60)
    print("SISTEMA HMARL - HIERARCHICAL MULTI-AGENT RL")
    print("="*60)
    print("Fase 1: Fundação e Features de Fluxo")
    print("Versão: 1.0.0")
    print("="*60 + "\n")
    
    # Criar e inicializar sistema
    system = HMARLSystem()
    
    # Registrar handler para shutdown limpo
    def signal_handler(sig, frame):
        print("\n\nRecebido sinal de interrupção...")
        system.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Inicializar
        system.initialize()
        
        # Aguardar um pouco para estabilizar
        print("\nAguardando sistema estabilizar...")
        time.sleep(3)
        
        # Iniciar
        system.start()
        
    except Exception as e:
        logging.error(f"Erro fatal no sistema: {e}", exc_info=True)
        system.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()