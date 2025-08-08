"""
Script de Inicialização para Produção (Conta Simulador)
Inicia o sistema completo de trading com configurações conservadoras
"""

import os
import sys
import logging
from datetime import datetime
import json
from pathlib import Path
import time
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging detalhado
def setup_logging():
    """Configura sistema de logging para produção"""
    log_dir = Path("logs/production")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Nome do arquivo com timestamp
    log_file = log_dir / f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configurar formatação
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para arquivo
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configurar root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

# Configurações conservadoras para produção inicial
PRODUCTION_CONFIG = {
    # Conexão ProfitDLL
    'connection': {
        'host': '127.0.0.1',
        'port': 8001,
        'username': os.getenv('PROFIT_USERNAME', ''),
        'password': os.getenv('PROFIT_PASSWORD', ''),
        'account_type': 'SIMULADOR',  # Conta simulador
        'symbols': ['WDOU25'],  # Começar com 1 símbolo apenas
    },
    
    # Limites de risco MUITO conservadores
    'risk': {
        'initial_capital': 50000.0,  # Capital inicial
        'max_position_size': 1,  # APENAS 1 contrato por vez
        'max_open_positions': 1,  # APENAS 1 posição por vez
        'max_daily_loss': 500.0,  # Máximo R$ 500 de perda por dia
        'max_drawdown': 0.02,  # 2% máximo drawdown
        'stop_loss_pct': 0.005,  # Stop loss 0.5% (muito apertado)
        'take_profit_pct': 0.01,  # Take profit 1%
        'trailing_stop_pct': 0.003,  # Trailing stop 0.3%
        'max_exposure': 5000.0,  # Exposição máxima R$ 5000
        'commission_per_contract': 5.0,  # Comissão por contrato
    },
    
    # Configurações da estratégia
    'strategy': {
        'confidence_threshold': 0.75,  # Só opera com alta confiança
        'regime_threshold': 0.7,  # Threshold alto para regime
        'tick_weight': 0.4,
        'book_weight': 0.6,
        'min_data_points': 100,  # Mínimo de dados antes de operar
    },
    
    # Configurações de ML
    'ml': {
        'models_path': 'models',
        'retrain_interval_minutes': 60,  # Retreino a cada hora
        'min_samples_retrain': 1000,
        'performance_threshold': 0.6,  # Mínimo 60% de acurácia
    },
    
    # Monitoramento
    'monitoring': {
        'dashboard_port': 5000,
        'update_interval_seconds': 1,
        'alert_email': os.getenv('ALERT_EMAIL', ''),
        'enable_telegram': False,  # Pode ativar depois
    },
    
    # Sistema
    'system': {
        'buffer_size': 10000,
        'max_gap_ms': 500,
        'order_timeout_seconds': 30,
        'enable_online_learning': True,
        'enable_adaptive_monitoring': True,
        'save_state_interval_minutes': 5,
    }
}

def validate_environment():
    """Valida ambiente antes de iniciar"""
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("VALIDANDO AMBIENTE DE PRODUÇÃO")
    logger.info("="*80)
    
    # 1. Verificar credenciais
    if not os.getenv('PROFIT_USERNAME') or not os.getenv('PROFIT_PASSWORD'):
        logger.warning("[WARN] Credenciais ProfitDLL não encontradas nas variáveis de ambiente")
        logger.warning("   Configure PROFIT_USERNAME e PROFIT_PASSWORD")
        return False
    
    # 2. Verificar modelos
    models_path = Path(PRODUCTION_CONFIG['ml']['models_path'])
    if not models_path.exists():
        logger.error("[ERRO] Diretório de modelos não encontrado")
        return False
    
    # Verificar modelos específicos
    tick_model = models_path / 'csv_5m' / 'lightgbm_tick.txt'
    book_model = models_path / 'book_only' / 'lightgbm_book_only_optimized.txt'
    
    if not tick_model.exists():
        logger.error(f"[ERRO] Modelo tick não encontrado: {tick_model}")
        return False
        
    if not book_model.exists():
        logger.error(f"[ERRO] Modelo book não encontrado: {book_model}")
        return False
    
    logger.info("[OK] Modelos encontrados")
    
    # 3. Criar diretórios necessários
    dirs_to_create = ['logs', 'data/checkpoints', 'reports', 'alerts']
    for dir_name in dirs_to_create:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    logger.info("[OK] Diretórios criados")
    
    # 4. Verificar portas
    import socket
    dashboard_port = PRODUCTION_CONFIG['monitoring']['dashboard_port']
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', dashboard_port))
    sock.close()
    
    if result == 0:
        logger.warning(f"[WARN] Porta {dashboard_port} já está em uso (dashboard)")
    
    return True

def create_safety_checks():
    """Cria verificações de segurança adicionais"""
    logger = logging.getLogger(__name__)
    
    # Arquivo de kill switch
    kill_switch_file = Path("STOP_TRADING.txt")
    
    logger.info("="*80)
    logger.info("VERIFICAÇÕES DE SEGURANÇA")
    logger.info("="*80)
    logger.info(f"Kill switch: {kill_switch_file.absolute()}")
    logger.info("Para parar o sistema, crie o arquivo STOP_TRADING.txt")
    logger.info("="*80)
    
    return kill_switch_file

def start_trading_system():
    """Inicia o sistema de trading completo"""
    logger = logging.getLogger(__name__)
    
    try:
        # Importar componentes
        from src.trading_system import TradingSystem
        from src.dashboard.dashboard_integration import DashboardIntegration
        
        logger.info("="*80)
        logger.info("INICIANDO SISTEMA DE TRADING - CONTA SIMULADOR")
        logger.info("="*80)
        logger.info(f"Timestamp: {datetime.now()}")
        logger.info(f"Símbolo: {PRODUCTION_CONFIG['connection']['symbols'][0]}")
        logger.info(f"Max posições: {PRODUCTION_CONFIG['risk']['max_position_size']}")
        logger.info(f"Stop loss: {PRODUCTION_CONFIG['risk']['stop_loss_pct']*100:.1f}%")
        logger.info(f"Take profit: {PRODUCTION_CONFIG['risk']['take_profit_pct']*100:.1f}%")
        logger.info("="*80)
        
        # Criar sistema principal
        trading_system = TradingSystem(PRODUCTION_CONFIG)
        
        # Iniciar dashboard
        dashboard = DashboardIntegration(
            trading_system,
            port=PRODUCTION_CONFIG['monitoring']['dashboard_port']
        )
        
        # Iniciar sistema
        logger.info("Iniciando componentes...")
        trading_system.start()
        dashboard.start()
        
        logger.info("[OK] Sistema iniciado com sucesso!")
        logger.info(f"Dashboard disponível em: http://localhost:{PRODUCTION_CONFIG['monitoring']['dashboard_port']}")
        
        # Loop principal com verificações
        kill_switch = create_safety_checks()
        last_health_check = time.time()
        
        while True:
            # Verificar kill switch
            if kill_switch.exists():
                logger.warning("[STOP] KILL SWITCH ATIVADO - Parando sistema...")
                break
            
            # Health check periódico
            if time.time() - last_health_check > 30:  # A cada 30 segundos
                health = trading_system.get_health_status()
                
                if health['status'] == 'healthy':
                    logger.info(f"[OK] Sistema saudável - P&L: R$ {health['pnl']:.2f}")
                else:
                    logger.warning(f"[WARN] Sistema com problemas: {health['issues']}")
                
                last_health_check = time.time()
            
            # Verificar limites diários
            daily_stats = trading_system.get_daily_statistics()
            if daily_stats['daily_loss'] >= PRODUCTION_CONFIG['risk']['max_daily_loss']:
                logger.error("[ERRO] LIMITE DIÁRIO DE PERDA ATINGIDO - Parando sistema")
                break
            
            time.sleep(1)
        
    except KeyboardInterrupt:
        logger.info("\n[INFO] Interrompido pelo usuário")
    except Exception as e:
        logger.error(f"[ERRO] Erro crítico: {e}", exc_info=True)
    finally:
        logger.info("Finalizando sistema...")
        
        # Salvar estado final
        try:
            final_state = {
                'timestamp': datetime.now().isoformat(),
                'positions': trading_system.get_open_positions(),
                'daily_pnl': trading_system.get_daily_statistics()['pnl'],
                'total_trades': trading_system.get_daily_statistics()['total_trades']
            }
            
            state_file = Path(f"logs/production/final_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(state_file, 'w') as f:
                json.dump(final_state, f, indent=2)
            
            logger.info(f"Estado final salvo em: {state_file}")
        except:
            pass
        
        # Parar componentes
        try:
            trading_system.stop()
            dashboard.stop()
        except:
            pass
        
        logger.info("[OK] Sistema finalizado")

def main():
    """Função principal"""
    # Configurar logging
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    
    print(f"\nLogs salvos em: {log_file}")
    print("\n" + "="*80)
    print("QUANTUM TRADER ML - SISTEMA DE PRODUÇÃO")
    print("="*80)
    print("MODO: CONTA SIMULADOR")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Validar ambiente
    if not validate_environment():
        logger.error("[ERRO] Falha na validação do ambiente")
        return 1
    
    # Confirmar início
    print("\n[ATENCAO] O sistema vai operar em conta SIMULADOR")
    print("   - Máximo 1 contrato por vez")
    print("   - Stop loss: 0.5%")
    print("   - Limite diário: R$ 500")
    print("\nPressione ENTER para continuar ou CTRL+C para cancelar...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n[CANCELADO] Cancelado pelo usuário")
        return 0
    
    # Iniciar sistema
    start_trading_system()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())