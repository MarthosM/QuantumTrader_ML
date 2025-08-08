"""
Executa backtest da HybridStrategy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

from src.backtesting.hybrid_backtest import HybridBacktest

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Executa backtest da estratégia híbrida"""
    
    print("="*80)
    print("BACKTEST DA HYBRIDSTRATEGY")
    print("="*80)
    
    # Configuração
    config = {
        'models_path': 'models',
        'initial_capital': 100000,
        'commission': 5.0,  # R$ 5 por contrato
        'slippage': 0.0001,  # 0.01%
        'regime_threshold': 0.6,
        'tick_weight': 0.4,
        'book_weight': 0.6,
        'max_position': 2,
        'stop_loss': 0.02,    # 2%
        'take_profit': 0.03   # 3%
    }
    
    # Criar backtester
    backtester = HybridBacktest(config)
    
    # Período do backtest
    # Usar últimos 30 dias de dados
    end_date = datetime(2024, 7, 30)  # Ajustar para dados disponíveis
    start_date = end_date - timedelta(days=30)
    
    print(f"\nPeríodo do backtest: {start_date.date()} até {end_date.date()}")
    
    try:
        # 1. Carregar dados
        print("\n1. Carregando dados...")
        backtester.load_data(
            start_date=start_date,
            end_date=end_date,
            tick_file=None,  # Usar arquivo padrão
            book_dir='data/realtime/book'  # Tentar carregar book data se disponível
        )
        
        # 2. Executar backtest
        print("\n2. Executando backtest...")
        results = backtester.run_backtest(
            lookback_candles=100,
            candle_timeframe='5min'
        )
        
        # 3. Salvar resultados
        print("\n3. Salvando resultados...")
        backtester.save_results()
        
        # 4. Salvar métricas em JSON
        output_dir = Path('backtest_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = output_dir / f"metrics_{timestamp}.json"
        
        # Remover campos não serializáveis
        metrics_to_save = {
            'config': config,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'performance': {
                'total_trades': results['total_trades'],
                'win_rate': results['win_rate'],
                'profit_factor': results['profit_factor'],
                'sharpe_ratio': results['sharpe_ratio'],
                'max_drawdown': results['max_drawdown'],
                'total_return': results['total_return'],
                'total_pnl': results['total_pnl'],
                'avg_pnl': results['avg_pnl'],
                'avg_win': results['avg_win'],
                'avg_loss': results['avg_loss'],
                'final_capital': results['final_capital']
            }
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        print(f"\n[OK] Métricas salvas em: {metrics_file}")
        
        # 5. Análise de regime
        print("\n" + "="*80)
        print("ANÁLISE DE REGIME")
        print("="*80)
        
        if backtester.signals:
            import pandas as pd
            signals_df = pd.DataFrame(backtester.signals)
            
            # Contar regimes
            regime_counts = signals_df['regime'].value_counts()
            print("\nDistribuição de regimes:")
            for regime, count in regime_counts.items():
                pct = count / len(signals_df) * 100
                print(f"  {regime}: {count} ({pct:.1f}%)")
            
            # Sinais por regime
            print("\nSinais por regime:")
            for regime in signals_df['regime'].unique():
                regime_signals = signals_df[signals_df['regime'] == regime]
                signal_counts = regime_signals['signal'].value_counts()
                print(f"\n  {regime}:")
                for signal, count in signal_counts.items():
                    signal_name = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}[signal]
                    pct = count / len(regime_signals) * 100
                    print(f"    {signal_name}: {count} ({pct:.1f}%)")
        
        print("\n[OK] Backtest concluído com sucesso!")
        
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()