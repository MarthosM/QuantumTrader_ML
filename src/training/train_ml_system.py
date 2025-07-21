# train_ml_system.py
"""
Script para treinar o sistema ML de trading
Execute de dentro do diretório src/training/
"""

import sys
import os

# Método mais robusto para adicionar o path correto
current_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = current_dir
src_dir = os.path.dirname(training_dir)
project_root = os.path.dirname(src_dir)

# Adicionar diretórios necessários ao path
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, training_dir)

from datetime import datetime, timedelta
import logging

# Tentar importar de diferentes formas
try:
    from training_orchestrator import TrainingOrchestrator
except ImportError:
    try:
        from src.training.training_orchestrator import TrainingOrchestrator
    except ImportError:
        print("❌ Erro: Não foi possível importar TrainingOrchestrator")
        print(f"📁 Diretório atual: {current_dir}")
        print(f"🔍 Verificar se o arquivo training_orchestrator.py existe")
        sys.exit(1)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Treina o sistema completo de ML"""
    
    # Configuração
    config = {
        'data_path': 'data/historical/',        # Onde estão os CSVs históricos
        'model_save_path': 'models/',           # Onde salvar modelos treinados
        'models_dir': 'saved_models/',          # Diretório de modelos do sistema
        'results_path': 'models/'               # Onde salvar resultados
    }
    
    print("\n" + "="*60)
    print("SISTEMA DE TREINAMENTO ML - TRADING v3.0")
    print("="*60)
    
    # Criar orquestrador
    orchestrator = TrainingOrchestrator(config)
    
    # Definir período de treinamento
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 meses de dados
    
    print(f"\nPeríodo de treinamento:")
    print(f"Início: {start_date.strftime('%Y-%m-%d')}")
    print(f"Fim: {end_date.strftime('%Y-%m-%d')}")
    
    # Símbolos para treinar
    symbols = ['WDOH25']  # Mini-dólar (ajuste conforme necessário)
    print(f"Símbolos: {symbols}")
    
    # Métricas alvo
    target_metrics = {
        'accuracy': 0.60,      # 60% de acurácia
        'f1_score': 0.58,      # F1 score de 0.58
        'avg_confidence': 0.65 # Confiança média de 65%
    }
    
    print("\nMétricas alvo:")
    for metric, value in target_metrics.items():
        print(f"  {metric}: {value}")
    
    # Confirmar execução
    response = input("\nDeseja iniciar o treinamento? (s/n): ")
    if response.lower() != 's':
        print("Treinamento cancelado.")
        return
    
    # Executar treinamento
    try:
        print("\n🚀 Iniciando treinamento do sistema ML...")
        
        results = orchestrator.train_complete_system(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            target_metrics=target_metrics,
            validation_method='walk_forward'  # ou 'purged_cv'
        )
        
        # Gerar relatório
        report_path = orchestrator.generate_training_report()
        print(f"\n📊 Relatório de treinamento salvo em: {report_path}")
        
        # Mostrar resumo dos resultados
        print("\n" + "="*60)
        print("RESUMO DOS RESULTADOS")
        print("="*60)
        
        metrics = results['aggregated_metrics']
        print(f"\nMétricas alcançadas:")
        print(f"  Accuracy: {metrics['accuracy_mean']:.4f} (±{metrics['accuracy_std']:.4f})")
        print(f"  F1 Score: {metrics['f1_score_mean']:.4f} (±{metrics['f1_score_std']:.4f})")
        print(f"  Confiança: {metrics['avg_confidence_mean']:.4f}")
        
        # Verificar se atingiu métricas alvo
        if results.get('target_metrics_achieved'):
            print("\nMétricas alvo:")
            for metric, info in results['target_metrics_achieved'].items():
                status = "✅" if info['success'] else "❌"
                print(f"  {metric}: {info['achieved']:.4f} / {info['target']:.4f} {status}")
        
        # Mostrar onde os modelos foram salvos
        print(f"\n💾 Modelos salvos em:")
        print(f"  {results['save_paths']['ensemble_path']}")
        
        print("\n✅ Treinamento concluído com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro durante treinamento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()