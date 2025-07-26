#!/usr/bin/env python3
"""
Script para executar backtest do sistema ML Trading v2.0
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime, timedelta
import logging
from src.backtest_runner import BacktestRunner
from src.ml_backtester import BacktestMode
from src.trading_system import TradingSystemV2
from dotenv import load_dotenv
import argparse

# Carregar variáveis de ambiente
load_dotenv()

def setup_logging():
    """Configura logging para o backtest"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('backtest.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Executa backtest do sistema"""
    parser = argparse.ArgumentParser(description='Backtest do Sistema ML Trading v2.0')
    parser.add_argument('--start', type=str, help='Data inicial (YYYY-MM-DD)', 
                        default=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
    parser.add_argument('--end', type=str, help='Data final (YYYY-MM-DD)', 
                        default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--capital', type=float, help='Capital inicial', default=100000.0)
    parser.add_argument('--mode', type=str, choices=['simple', 'realistic', 'conservative', 'stress'],
                        help='Modo do backtest', default='realistic')
    parser.add_argument('--stress-test', action='store_true', help='Executar stress tests')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("🚀 BACKTEST - ML TRADING SYSTEM v2.0")
    print("=" * 60)
    
    try:
        # Converter datas
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
        
        print(f"📅 Período: {start_date.date()} até {end_date.date()}")
        print(f"💰 Capital inicial: R$ {args.capital:,.2f}")
        print(f"⚙️  Modo: {args.mode.upper()}")
        print(f"🔧 Stress test: {'Sim' if args.stress_test else 'Não'}")
        print("-" * 60)
        
        # Criar sistema de trading
        print("Inicializando sistema de trading...")
        
        # Configuração para backtest (sem conexão real)
        config = {
            'ticker': 'WDO',
            'environment': 'backtest',
            'models_dir': os.getenv('MODELS_DIR'),
            'initial_balance': args.capital,
            'use_gui': False,  # Sem GUI no backtest
            'backtest_mode': True  # Flag especial para backtest
        }
        
        # Criar sistema de trading
        trading_system = TradingSystemV2(config)
        
        # Inicializar sem conexão real
        print("Carregando modelos ML...")
        if hasattr(trading_system, 'model_manager') and trading_system.model_manager:
            models_loaded = len(trading_system.model_manager.models) if hasattr(trading_system.model_manager, 'models') else 0
            print(f"✅ {models_loaded} modelos carregados")
        else:
            print("⚠️  Sem modelos ML - backtest rodará sem predições")
        
        # Criar runner de backtest
        runner = BacktestRunner(trading_system)
        
        # Configuração do backtest
        backtest_config = {
            'initial_capital': args.capital,
            'commission_per_contract': 0.50,  # Custo por contrato WDO
            'slippage_ticks': 1,  # 1 tick de slippage
            'mode': getattr(BacktestMode, args.mode.upper()),
            'run_stress_tests': args.stress_test
        }
        
        print("\n🔄 Executando backtest...")
        print("Por favor aguarde, isso pode levar alguns minutos...")
        
        # Executar backtest
        results = runner.run_ml_backtest(
            start_date=start_date,
            end_date=end_date,
            config=backtest_config
        )
        
        # Exibir resultados
        print("\n" + "=" * 60)
        print("📊 RESULTADOS DO BACKTEST")
        print("=" * 60)
        
        if 'metrics' in results:
            metrics = results['metrics']
            
            # Métricas principais
            print("\n💎 MÉTRICAS PRINCIPAIS:")
            print(f"Capital Final: R$ {metrics.get('final_equity', args.capital):,.2f}")
            print(f"Lucro/Prejuízo: R$ {metrics.get('total_pnl', 0):,.2f}")
            print(f"Retorno Total: {metrics.get('total_return', 0)*100:.2f}%")
            
            # Métricas de risco
            print("\n📈 MÉTRICAS DE RISCO:")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            
            # Estatísticas de trades
            print("\n🎯 ESTATÍSTICAS DE TRADES:")
            print(f"Total de Trades: {metrics.get('total_trades', 0)}")
            print(f"Trades Vencedoras: {metrics.get('winning_trades', 0)}")
            print(f"Trades Perdedoras: {metrics.get('losing_trades', 0)}")
            print(f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            
            # Valores médios
            if metrics.get('total_trades', 0) > 0:
                print("\n💰 VALORES MÉDIOS:")
                print(f"Ganho Médio: R$ {metrics.get('avg_win', 0):,.2f}")
                print(f"Perda Média: R$ {metrics.get('avg_loss', 0):,.2f}")
                print(f"Expectativa: R$ {metrics.get('expectancy', 0):,.2f}")
        
        # Análise adicional
        if 'trade_analysis' in results:
            analysis = results['trade_analysis']
            
            if 'by_side' in analysis:
                print("\n📊 ANÁLISE POR DIREÇÃO:")
                for side, stats in analysis['by_side'].items():
                    print(f"{side.upper()}: {stats['count']} trades, "
                          f"Win Rate: {stats['win_rate']*100:.1f}%, "
                          f"PnL Médio: R$ {stats['avg_pnl']:,.2f}")
        
        # Stress tests
        if args.stress_test and 'stress_tests' in results:
            print("\n🔥 RESULTADOS DOS STRESS TESTS:")
            for test_name, test_results in results['stress_tests'].items():
                print(f"\n{test_name}:")
                print(f"  Retorno: {test_results.get('total_return', 0)*100:.2f}%")
                print(f"  Max DD: {test_results.get('max_drawdown', 0)*100:.2f}%")
                print(f"  Sharpe: {test_results.get('sharpe_ratio', 0):.2f}")
        
        print("\n" + "=" * 60)
        print("✅ Backtest concluído com sucesso!")
        print(f"📄 Relatório detalhado salvo em: backtest_report_*.html")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Erro durante backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())