"""
Exemplo: Treinar Modelos Dual (Tick-Only + Book-Enhanced)
Demonstra o uso completo do sistema de treinamento dual integrado com HMARL
"""

import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import json

# Adicionar src ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.tick_training_pipeline import TickTrainingPipeline
from src.training.book_training_pipeline import BookTrainingPipeline
from src.training.dual_training_system import DualTrainingSystem
from src.infrastructure.hmarl_ml_integration import integrate_hmarl_with_ml_system


class DualModelTrainer:
    """
    Coordena o treinamento de modelos dual (tick + book)
    e integra com sistema HMARL
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('DualModelTrainer')
        
        # Pipelines
        self.tick_pipeline = TickTrainingPipeline(config['tick'])
        self.book_pipeline = BookTrainingPipeline(config['book'])
        self.dual_system = DualTrainingSystem(config['dual'])
        
        # Resultados
        self.results = {
            'tick_models': {},
            'book_models': {},
            'hybrid_strategy': {}
        }
        
    def train_all_models(self, symbol: str):
        """Treina todos os modelos para um símbolo"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"TREINAMENTO DUAL COMPLETO: {symbol}")
        self.logger.info(f"{'='*60}\n")
        
        # 1. Treinar modelos tick-only (1 ano)
        self.logger.info("FASE 1: Modelos Tick-Only (1 ano de dados)")
        self.logger.info("-" * 40)
        
        try:
            tick_result = self.tick_pipeline.train_complete_pipeline(
                symbol=symbol,
                lookback_days=365
            )
            
            if tick_result['status'] == 'completed':
                self.results['tick_models'] = tick_result
                self.logger.info("✅ Modelos tick-only treinados com sucesso")
                self._print_tick_summary(tick_result)
            else:
                self.logger.error("❌ Falha no treinamento tick-only")
                
        except Exception as e:
            self.logger.error(f"Erro no pipeline tick-only: {e}")
            
        # 2. Treinar modelos book-enhanced (30 dias)
        self.logger.info("\nFASE 2: Modelos Book-Enhanced (30 dias de book)")
        self.logger.info("-" * 40)
        
        try:
            book_result = self.book_pipeline.train_complete_pipeline(
                symbol=symbol,
                lookback_days=30,
                targets=['spread_next', 'imbalance_direction', 'price_move_5s']
            )
            
            if book_result['status'] == 'completed':
                self.results['book_models'] = book_result
                self.logger.info("✅ Modelos book-enhanced treinados com sucesso")
                self._print_book_summary(book_result)
            else:
                self.logger.error("❌ Falha no treinamento book-enhanced")
                
        except Exception as e:
            self.logger.error(f"Erro no pipeline book-enhanced: {e}")
            
        # 3. Criar estratégia híbrida
        self.logger.info("\nFASE 3: Estratégia Híbrida + HMARL")
        self.logger.info("-" * 40)
        
        try:
            hybrid_strategy = self.dual_system.create_hybrid_strategy(symbol)
            self.results['hybrid_strategy'] = hybrid_strategy
            
            self.logger.info("✅ Estratégia híbrida criada")
            self._print_hybrid_summary(hybrid_strategy)
            
        except Exception as e:
            self.logger.error(f"Erro ao criar estratégia híbrida: {e}")
            
        # 4. Salvar relatório completo
        self._save_complete_report(symbol)
        
        return self.results
        
    def _print_tick_summary(self, result):
        """Imprime resumo dos modelos tick-only"""
        print("\nModelos Tick-Only por Regime:")
        
        for regime, perf in result['performance']['by_regime'].items():
            print(f"  {regime.upper()}:")
            print(f"    - Accuracy: {perf['accuracy']:.4f}")
            print(f"    - F1 Score: {perf['f1_score']:.4f}")
            print(f"    - Amostras: {perf['n_samples']:,}")
            
    def _print_book_summary(self, result):
        """Imprime resumo dos modelos book-enhanced"""
        print("\nModelos Book-Enhanced por Target:")
        
        for target, perf in result['performance']['by_target'].items():
            print(f"  {target}:")
            if perf['type'] == 'regression':
                print(f"    - R²: {perf['performance']['r2']:.4f}")
                print(f"    - MAE: {perf['performance']['mae']:.6f}")
            else:
                print(f"    - Accuracy: {perf['performance']['accuracy']:.4f}")
                print(f"    - F1 Score: {perf['performance']['f1_score']:.4f}")
                
    def _print_hybrid_summary(self, strategy):
        """Imprime resumo da estratégia híbrida"""
        print("\nComponentes da Estratégia Híbrida:")
        
        for component, source in strategy['components'].items():
            print(f"  {component}: {source}")
            
        if strategy['hmarl_integration']['enabled']:
            print("\n✅ HMARL Integration: ATIVO")
            print(f"  Agentes: {len(strategy.get('hmarl_agents', []))}")
            print(f"  Flow Features: {len(strategy['hmarl_integration']['flow_features'])}")
        else:
            print("\n⚠️  HMARL Integration: INATIVO")
            
    def _save_complete_report(self, symbol):
        """Salva relatório completo do treinamento"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = Path(self.config['report_path']) / symbol / f'dual_training_{timestamp}'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Relatório JSON
        report = {
            'symbol': symbol,
            'timestamp': timestamp,
            'tick_models': {
                'status': self.results['tick_models'].get('status', 'failed'),
                'performance': self.results['tick_models'].get('performance', {}),
                'save_paths': self.results['tick_models'].get('save_paths', {})
            },
            'book_models': {
                'status': self.results['book_models'].get('status', 'failed'),
                'performance': self.results['book_models'].get('performance', {}),
                'save_paths': self.results['book_models'].get('save_paths', {})
            },
            'hybrid_strategy': self.results['hybrid_strategy']
        }
        
        with open(report_dir / 'training_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        # README markdown
        readme_content = f"""# Relatório de Treinamento Dual - {symbol}

## Resumo Executivo

Data: {timestamp}

### Modelos Tick-Only (1 ano)
- Status: {self.results['tick_models'].get('status', 'failed')}
- Regimes treinados: {len(self.results['tick_models'].get('models', {}))}
- Performance média: {self._get_tick_performance()}

### Modelos Book-Enhanced (30 dias)
- Status: {self.results['book_models'].get('status', 'failed')}
- Targets treinados: {len(self.results['book_models'].get('models', {}))}
- Performance média: {self._get_book_performance()}

### Estratégia Híbrida
- HMARL: {'ATIVO' if self.results['hybrid_strategy'].get('hmarl_integration', {}).get('enabled') else 'INATIVO'}
- Componentes: {len(self.results['hybrid_strategy'].get('components', {}))}

## Como Usar

### 1. Carregar Modelos Tick-Only
```python
from src.model_manager import ModelManager

# Para cada regime
for regime in ['trend_up', 'trend_down', 'range']:
    model_path = f"models/tick_only/{symbol}/{regime}/model.pkl"
    # Carregar e usar modelo
```

### 2. Carregar Modelos Book-Enhanced
```python
# Para microestrutura
model_path = f"models/book_enhanced/{symbol}/spread_next/model.pkl"
# Carregar e usar para timing
```

### 3. Usar Estratégia Híbrida
```python
from examples.hmarl_integrated_trading import HMARLIntegratedTrading

system = HMARLIntegratedTrading(config)
system.start_trading('{symbol}')
```

## Próximos Passos

1. Validar modelos em ambiente de simulação
2. Ajustar thresholds baseado em backtest
3. Implementar monitoramento em produção
4. Coletar mais dados de book para retreino

---
Gerado automaticamente pelo sistema QuantumTrader ML v2.0
"""
        
        with open(report_dir / 'README.md', 'w') as f:
            f.write(readme_content)
            
        self.logger.info(f"\n📊 Relatório completo salvo em: {report_dir}")
        
    def _get_tick_performance(self):
        """Calcula performance média dos modelos tick"""
        try:
            perf = self.results['tick_models']['performance']['overall']
            return f"Acc={perf['weighted_accuracy']:.3f}, F1={perf['weighted_f1_score']:.3f}"
        except:
            return "N/A"
            
    def _get_book_performance(self):
        """Calcula performance média dos modelos book"""
        try:
            perf = self.results['book_models']['performance']['overall']
            if 'avg_regression_r2' in perf:
                return f"R²={perf['avg_regression_r2']:.3f}"
            elif 'avg_classification_accuracy' in perf:
                return f"Acc={perf['avg_classification_accuracy']:.3f}"
            return "N/A"
        except:
            return "N/A"


def main():
    """Função principal"""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuração completa
    config = {
        'tick': {
            'tick_data_path': 'data/historical',
            'models_path': 'models/tick_only',
            'model_save_path': 'models/tick_only'
        },
        'book': {
            'book_data_path': 'data/realtime/book',
            'models_path': 'models/book_enhanced'
        },
        'dual': {
            'tick_data_path': 'data/historical',
            'book_data_path': 'data/realtime/book',
            'models_path': 'models'
        },
        'report_path': 'reports/training'
    }
    
    # Criar trainer
    trainer = DualModelTrainer(config)
    
    # Símbolo para treinar
    symbol = 'WDOU25'
    
    print(f"\n🚀 Iniciando treinamento dual para {symbol}")
    print("Este processo pode demorar alguns minutos...\n")
    
    # Treinar todos os modelos
    results = trainer.train_all_models(symbol)
    
    print("\n✅ Treinamento completo!")
    print("\nResumo Final:")
    print(f"- Modelos Tick-Only: {results['tick_models'].get('status', 'failed')}")
    print(f"- Modelos Book-Enhanced: {results['book_models'].get('status', 'failed')}")
    print(f"- Estratégia Híbrida: {'Criada' if results['hybrid_strategy'] else 'Falhou'}")
    
    # Sugestão de próximos passos
    print("\n📌 Próximos Passos:")
    print("1. Revisar os relatórios gerados em reports/training/")
    print("2. Executar backtest com os modelos treinados")
    print("3. Testar em ambiente de simulação")
    print("4. Ajustar parâmetros baseado nos resultados")


if __name__ == "__main__":
    main()