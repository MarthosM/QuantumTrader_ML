"""
Exemplo de Integração HMARL com Sistema Atual
Demonstra como usar a infraestrutura ZeroMQ + Valkey sem quebrar o sistema existente
"""

import sys
import os
import time
import logging
from datetime import datetime

# Adicionar src ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading_system import TradingSystem
from src.infrastructure.system_integration_wrapper import (
    integrate_hmarl_with_system,
    HMARLSystemWrapper
)
from src.infrastructure.zmq_consumers import (
    FlowConsumer,
    MultiStreamConsumer,
    FlowAlertSystem
)


def setup_logging():
    """Configura logging para o exemplo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('hmarl_integration.log')
        ]
    )


def example_1_basic_integration():
    """
    Exemplo 1: Integração básica com sistema atual
    Adiciona análise de fluxo sem modificar nada
    """
    print("\n=== EXEMPLO 1: Integração Básica ===\n")
    
    # 1. Configurar sistema existente
    config = {
        'dll_path': './ProfitDLL64.dll',
        'username': 'user',
        'password': 'pass',
        'models_dir': './models/',
        'historical_days': 1,
        'ml_interval': 60,
        'use_gui': False
    }
    
    print("1. Inicializando sistema de trading atual...")
    trading_system = TradingSystem(config)
    
    # Simular inicialização bem-sucedida
    # trading_system.initialize()
    
    # 2. Adicionar HMARL
    print("\n2. Adicionando infraestrutura HMARL...")
    
    hmarl_config = {
        'symbol': 'WDOH25',
        'zmq': {
            'tick_port': 5555,
            'book_port': 5556,
            'flow_port': 5557,
            'footprint_port': 5558
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
    
    try:
        # Integrar HMARL
        hmarl_wrapper = integrate_hmarl_with_system(trading_system, hmarl_config)
        print("✅ HMARL integrado com sucesso!")
        
        # 3. Sistema agora funciona com análise de fluxo adicional
        print("\n3. Sistema rodando com análise de fluxo...")
        
        # Simular operação
        # trading_system.start('WDOH25')
        
        # 4. Verificar estatísticas
        print("\n4. Estatísticas da integração:")
        stats = hmarl_wrapper.get_stats()
        print(f"   - Análise de fluxo: {'HABILITADA' if stats['flow_analysis_enabled'] else 'DESABILITADA'}")
        print(f"   - Bridge rodando: {'SIM' if stats['bridge_running'] else 'NÃO'}")
        
        return hmarl_wrapper
        
    except Exception as e:
        print(f"❌ Erro na integração: {e}")
        return None


def example_2_flow_enhanced_features():
    """
    Exemplo 2: Usando features aprimoradas com fluxo
    Mostra como obter features adicionais para ML
    """
    print("\n=== EXEMPLO 2: Features Aprimoradas com Fluxo ===\n")
    
    # Assumindo que temos o wrapper do exemplo 1
    hmarl_wrapper = HMARLSystemWrapper(None, {'symbol': 'WDOH25'})
    
    # Simular features existentes do sistema
    existing_features = {
        'ema_9': 100.5,
        'ema_20': 100.3,
        'rsi_14': 55.2,
        'volume_ratio': 1.2,
        'price_momentum': 0.002
    }
    
    print("1. Features originais do sistema:")
    for key, value in existing_features.items():
        print(f"   - {key}: {value}")
    
    # Obter features de fluxo
    print("\n2. Obtendo features de fluxo...")
    flow_features = {
        'flow_ofi_1m': 0.15,
        'flow_ofi_5m': 0.22,
        'flow_volume_imbalance': 0.18,
        'flow_aggression_ratio': 0.65,
        'flow_large_trade_ratio': 0.12,
        'tape_speed': 3.5,
        'liquidity_score': 85.3
    }
    
    print("   Features de fluxo adicionadas:")
    for key, value in flow_features.items():
        print(f"   - {key}: {value}")
    
    # Combinar features
    enhanced_features = existing_features.copy()
    enhanced_features.update(flow_features)
    
    print(f"\n3. Total de features disponíveis: {len(enhanced_features)}")
    print("   ✅ Features prontas para modelo ML aprimorado!")
    
    return enhanced_features


def example_3_real_time_monitoring():
    """
    Exemplo 3: Monitoramento em tempo real com consumers
    Demonstra uso de consumers para análise paralela
    """
    print("\n=== EXEMPLO 3: Monitoramento em Tempo Real ===\n")
    
    symbol = 'WDOH25'
    
    # 1. Criar multi-stream consumer
    print("1. Criando consumer multi-stream...")
    multi_consumer = MultiStreamConsumer(symbol)
    multi_consumer.start()
    print("✅ Consumer iniciado")
    
    # 2. Criar sistema de alertas
    print("\n2. Criando sistema de alertas...")
    alert_system = FlowAlertSystem(symbol)
    alert_system.start()
    print("✅ Sistema de alertas ativo")
    
    # 3. Simular monitoramento
    print("\n3. Monitorando mercado (simulação)...")
    
    for i in range(5):
        time.sleep(1)
        
        # Obter visão unificada
        unified = multi_consumer.get_unified_view()
        
        print(f"\n   Tick {i+1}:")
        print(f"   - Timestamp: {unified['timestamp']}")
        print(f"   - OFI (5m): {unified.get('ofi_5m', 0):.3f}")
        
        # Verificar liquidez
        if 'liquidity_profile' in unified:
            profile = unified['liquidity_profile']
            print(f"   - Score de liquidez: {profile.get('avg_liquidity_score', 0):.1f}")
        
        # Verificar padrões
        if 'recent_patterns' in unified:
            patterns = unified['recent_patterns']
            if patterns:
                print(f"   - Padrões detectados: {len(patterns)}")
    
    # 4. Estatísticas finais
    print("\n4. Estatísticas de monitoramento:")
    stats = multi_consumer.get_stats()
    
    for stream, stream_stats in stats.items():
        print(f"\n   Stream '{stream}':")
        print(f"   - Mensagens recebidas: {stream_stats['messages_received']}")
        print(f"   - Erros: {stream_stats['errors']}")
    
    # Cleanup
    multi_consumer.stop()
    alert_system.stop()
    
    print("\n✅ Monitoramento finalizado")


def example_4_advanced_flow_analysis():
    """
    Exemplo 4: Análise avançada de fluxo
    Demonstra análise detalhada de order flow
    """
    print("\n=== EXEMPLO 4: Análise Avançada de Fluxo ===\n")
    
    # Criar consumer de fluxo
    flow_consumer = FlowConsumer('WDOH25')
    flow_consumer.start()
    
    print("1. Analisando fluxo de ordens...")
    
    # Callback personalizado para análise
    def analyze_flow(flow_data):
        if 'analysis' in flow_data:
            analysis = flow_data['analysis']
            
            # OFI Analysis
            if 'ofi' in analysis:
                ofi_1m = analysis['ofi'].get(1, 0)
                ofi_5m = analysis['ofi'].get(5, 0)
                
                # Detectar divergências
                if abs(ofi_1m - ofi_5m) > 0.3:
                    print(f"\n   ⚠️ DIVERGÊNCIA DETECTADA:")
                    print(f"   - OFI 1m: {ofi_1m:.3f}")
                    print(f"   - OFI 5m: {ofi_5m:.3f}")
                    print(f"   - Possível reversão em andamento")
            
            # Volume Imbalance
            if 'volume_imbalance' in analysis:
                imbalance = analysis['volume_imbalance']
                if abs(imbalance) > 0.7:
                    direction = "COMPRA" if imbalance > 0 else "VENDA"
                    print(f"\n   🔥 PRESSÃO FORTE DE {direction}")
                    print(f"   - Desequilíbrio: {imbalance:.3f}")
            
            # Large Trade Activity
            if 'large_trade_ratio' in analysis:
                ratio = analysis['large_trade_ratio']
                if ratio > 0.25:
                    print(f"\n   🐋 ATIVIDADE DE GRANDES PLAYERS")
                    print(f"   - {ratio*100:.1f}% dos trades são grandes")
    
    # Definir callback
    flow_consumer.callback = analyze_flow
    
    # Simular análise por 10 segundos
    print("\n2. Aguardando eventos de fluxo...")
    time.sleep(10)
    
    # Estatísticas finais
    print("\n3. Resumo da análise:")
    stats = flow_consumer.get_stats()
    print(f"   - Total de eventos analisados: {stats['messages_processed']}")
    
    # Agregações de fluxo
    print("\n4. Agregações de fluxo:")
    for window in ['1m', '5m', '15m']:
        ofi = flow_consumer.get_ofi(window)
        print(f"   - OFI {window}: {ofi:.3f}")
    
    flow_consumer.stop()
    print("\n✅ Análise de fluxo concluída")


def example_5_complete_workflow():
    """
    Exemplo 5: Workflow completo com HMARL
    Demonstra integração completa do início ao fim
    """
    print("\n=== EXEMPLO 5: Workflow Completo HMARL ===\n")
    
    # 1. Setup
    print("1. Configurando ambiente...")
    setup_logging()
    logger = logging.getLogger('workflow')
    
    # 2. Simular sistema de trading
    class MockTradingSystem:
        def __init__(self):
            self.data = None
            self.ml_coordinator = None
            self.feature_engine = None
            self.logger = logger
            
    trading_system = MockTradingSystem()
    
    # 3. Configurar HMARL
    print("\n2. Configurando HMARL...")
    hmarl_config = {
        'symbol': 'WDOH25',
        'zmq': {
            'tick_port': 5555,
            'book_port': 5556,
            'flow_port': 5557
        },
        'valkey': {
            'host': 'localhost',
            'port': 6379
        }
    }
    
    # 4. Criar wrapper
    print("\n3. Criando wrapper de integração...")
    wrapper = HMARLSystemWrapper(trading_system, hmarl_config)
    
    # 5. Workflow de trading
    print("\n4. Executando workflow de trading:")
    
    # Passo 1: Receber dados de mercado
    print("\n   Passo 1: Recebendo dados de mercado...")
    market_data = {
        'symbol': 'WDOH25',
        'price': 5000.0,
        'volume': 100,
        'timestamp': datetime.now()
    }
    print(f"   ✅ Dados recebidos: {market_data['symbol']} @ {market_data['price']}")
    
    # Passo 2: Análise de fluxo
    print("\n   Passo 2: Analisando fluxo...")
    flow_features = wrapper.get_flow_enhanced_features('WDOH25')
    if flow_features:
        print(f"   ✅ Features de fluxo calculadas: {len(flow_features)} features")
    
    # Passo 3: Decisão de trading
    print("\n   Passo 3: Tomando decisão...")
    
    # Simular análise
    ofi = flow_features.get('flow_ofi_5m', 0)
    aggression = flow_features.get('flow_aggression_ratio', 0.5)
    
    if ofi > 0.3 and aggression > 0.6:
        decision = "COMPRA"
        confidence = 0.75
    elif ofi < -0.3 and aggression < 0.4:
        decision = "VENDA"
        confidence = 0.75
    else:
        decision = "NEUTRO"
        confidence = 0.5
    
    print(f"   ✅ Decisão: {decision} (Confiança: {confidence:.2%})")
    
    # Passo 4: Monitoramento
    print("\n   Passo 4: Monitorando execução...")
    print("   ✅ Sistema em monitoramento contínuo")
    
    # 6. Estatísticas finais
    print("\n5. Estatísticas do workflow:")
    stats = wrapper.get_stats()
    print(f"   - Trades processados: {stats['wrapper_stats']['trades_processed']}")
    print(f"   - Eventos de fluxo: {stats['wrapper_stats']['flow_events']}")
    print(f"   - Erros: {stats['wrapper_stats']['errors']}")
    
    print("\n✅ Workflow completo executado com sucesso!")


def main():
    """Função principal que executa todos os exemplos"""
    
    print("\n" + "="*60)
    print("   EXEMPLOS DE INTEGRAÇÃO HMARL")
    print("   Fase 1, Semana 1 - ZeroMQ + Valkey + Flow")
    print("="*60)
    
    # Verificar dependências
    print("\nVerificando dependências...")
    try:
        import zmq
        print("✅ ZeroMQ instalado")
    except ImportError:
        print("❌ ZeroMQ não instalado. Execute: pip install pyzmq")
        return
    
    try:
        import valkey
        print("✅ Valkey instalado")
    except ImportError:
        print("❌ Valkey não instalado. Execute: pip install valkey")
        return
    
    # Menu de exemplos
    examples = {
        '1': ('Integração Básica', example_1_basic_integration),
        '2': ('Features Aprimoradas', example_2_flow_enhanced_features),
        '3': ('Monitoramento Real-Time', example_3_real_time_monitoring),
        '4': ('Análise Avançada de Fluxo', example_4_advanced_flow_analysis),
        '5': ('Workflow Completo', example_5_complete_workflow),
        '0': ('Executar Todos', None)
    }
    
    print("\nExemplos disponíveis:")
    for key, (name, _) in examples.items():
        if key != '0':
            print(f"  {key}. {name}")
    print(f"  0. Executar todos os exemplos")
    print(f"  Q. Sair")
    
    while True:
        choice = input("\nEscolha um exemplo (0-5 ou Q): ").strip().upper()
        
        if choice == 'Q':
            print("\nSaindo...")
            break
        
        if choice == '0':
            # Executar todos
            for key, (name, func) in examples.items():
                if key != '0' and func:
                    try:
                        func()
                    except Exception as e:
                        print(f"\n❌ Erro no exemplo {name}: {e}")
                    time.sleep(2)
        
        elif choice in examples and examples[choice][1]:
            try:
                examples[choice][1]()
            except Exception as e:
                print(f"\n❌ Erro: {e}")
                import traceback
                traceback.print_exc()
        
        else:
            print("Opção inválida!")


if __name__ == "__main__":
    main()