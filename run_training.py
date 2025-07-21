# run_training.py
"""
Script principal para executar o treinamento do sistema ML
Execute este script do diretório raiz do projeto
"""

import sys
import os
from datetime import datetime, timedelta
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Treina o sistema completo de ML"""
    
    # FORÇAR MODO PRODUÇÃO MAS RELAXAR VALIDAÇÃO PARA TREINAMENTO
    os.environ['TRADING_PRODUCTION_MODE'] = 'true'
    os.environ['STRICT_VALIDATION'] = 'false'  # Relaxar validação para permitir dados de exemplo
    os.environ['TRAINING_MODE'] = 'true'  # Sinalizar que estamos em modo treinamento
    os.environ['ALLOW_SYNTHETIC_DATA'] = 'true'  # Permitir dados sintéticos para treinamento
    print("🛡️ MODO PRODUÇÃO ATIVADO - VALIDAÇÃO RELAXADA PARA TREINAMENTO")
    
    # Verificar se estamos no diretório correto
    if not os.path.exists('src/training/training_orchestrator.py'):
        print("❌ Erro: Execute este script do diretório raiz do projeto ML_Tradingv2.0")
        print("📁 Diretório atual:", os.getcwd())
        print("💡 Comando correto: python run_training.py")
        return
    
    # Adicionar src ao path para importações
    sys.path.insert(0, 'src')
    
    try:
        from training.training_orchestrator import TrainingOrchestrator
    except ImportError as e:
        print(f"❌ Erro ao importar TrainingOrchestrator: {e}")
        print("📝 Verifique se todos os arquivos da ETAPA 6 foram criados corretamente")
        return
    
    # Configuração
    config = {
        'data_path': 'src/training/data/historical/',  # Caminho correto para os dados CSV
        'model_save_path': 'src/training/models/',   # Onde salvar modelos treinados  
        'models_dir': 'src/models/',            # Diretório de modelos existentes
        'results_path': 'src/training/models/'     # Onde salvar resultados
    }
    
    print("\n" + "="*60)
    print("🧠 SISTEMA DE TREINAMENTO ML - TRADING v3.0")
    print("="*60)
    
    # Verificar se diretórios existem
    print("\n📁 Verificando estrutura de diretórios...")
    directories_to_check = [
        config['data_path'],
        config['model_save_path'], 
        config['models_dir'],
        config['results_path']
    ]
    
    for directory in directories_to_check:
        if not os.path.exists(directory):
            print(f"📂 Criando diretório: {directory}")
            os.makedirs(directory, exist_ok=True)
        else:
            print(f"✅ Diretório existe: {directory}")
    
    # Criar orquestrador
    try:
        print("\n🛠️ Inicializando orquestrador de treinamento...")
        orchestrator = TrainingOrchestrator(config)
        print("✅ Orquestrador criado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao criar orquestrador: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Definir período de treinamento - usar TODO o período dos dados CSV
    # Os dados CSV vão de 2025-02-03 09:00 até 2025-06-20 17:54
    start_date = datetime(2025, 2, 3, 9, 0)  # Início exato dos dados no CSV
    end_date = datetime(2025, 6, 20, 18, 0)  # Final exato dos dados no CSV
    
    print(f"\n📅 Período de treinamento:")
    print(f"   Início: {start_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"   Fim: {end_date.strftime('%Y-%m-%d %H:%M')}")
    
    # Símbolos para treinar
    symbols = ['WDOH25']  # Mini-dólar (ajuste conforme necessário)
    print(f"📈 Símbolos: {symbols}")
    
    # Métricas alvo
    target_metrics = {
        'win_rate': 0.60,          # 60% win rate
        'sharpe_ratio': 1.5,       # Sharpe ratio 1.5
        'max_drawdown': 0.05,      # Max 5% drawdown
        'avg_confidence': 0.65     # Confiança média de 65%
    }
    
    print("\n🎯 Métricas alvo:")
    for metric, value in target_metrics.items():
        print(f"   {metric}: {value}")
    
    # Verificar se há dados disponíveis
    data_files = []
    if os.path.exists(config['data_path']):
        data_files = [f for f in os.listdir(config['data_path']) if f.endswith('.csv')]
    
    if not data_files:
        print(f"\n⚠️  AVISO: Nenhum arquivo CSV encontrado em {config['data_path']}")
        print("📝 Para executar o treinamento, você precisa de dados históricos em formato CSV")
        print("📄 Formato esperado: timestamp,open,high,low,close,volume,trades,buy_volume,sell_volume,vwap,symbol")
        print("\n💡 Deseja continuar com dados de exemplo? (s/n): ", end="")
        
        response = input()
        if response.lower() != 's':
            print("❌ Treinamento cancelado - adicione dados CSV e tente novamente")
            return
            
        print("🔄 Modo EXEMPLO ativado - será usado para demonstração")
    else:
        print(f"\n📊 Arquivos de dados encontrados: {len(data_files)}")
        for file in data_files[:3]:  # Mostrar apenas primeiros 3
            print(f"   📄 {file}")
        if len(data_files) > 3:
            print(f"   ... e mais {len(data_files) - 3} arquivo(s)")
    
    # Confirmar execução
    print(f"\n❓ Deseja iniciar o treinamento? (s/n): ", end="")
    response = input()
    if response.lower() != 's':
        print("❌ Treinamento cancelado pelo usuário")
        return
    
    # Executar treinamento
    try:
        print("\n🚀 Iniciando treinamento do sistema ML...")
        print("⏱️  Este processo pode levar alguns minutos...")
        
        results = orchestrator.train_complete_system(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            target_metrics=target_metrics,
            validation_method='walk_forward'  # ou 'purged_cv'
        )
        
        # Mostrar resumo dos resultados
        print("\n" + "="*60)
        print("📈 RESUMO DOS RESULTADOS")
        print("="*60)
        
        if 'aggregated_metrics' in results:
            metrics = results['aggregated_metrics']
            print(f"\n🎯 Métricas alcançadas:")
            
            if 'accuracy_mean' in metrics:
                print(f"   Accuracy: {metrics['accuracy_mean']:.4f} (±{metrics.get('accuracy_std', 0):.4f})")
            if 'f1_score_mean' in metrics:
                print(f"   F1 Score: {metrics['f1_score_mean']:.4f} (±{metrics.get('f1_score_std', 0):.4f})")
            if 'win_rate_mean' in metrics:
                print(f"   Win Rate: {metrics['win_rate_mean']:.4f} (±{metrics.get('win_rate_std', 0):.4f})")
            if 'sharpe_ratio_mean' in metrics:
                print(f"   Sharpe Ratio: {metrics['sharpe_ratio_mean']:.4f}")
            if 'avg_confidence_mean' in metrics:
                print(f"   Confiança: {metrics['avg_confidence_mean']:.4f}")
        
        # Verificar se atingiu métricas alvo
        if results.get('target_metrics_achieved'):
            print(f"\n🏆 Status das métricas alvo:")
            for metric, info in results['target_metrics_achieved'].items():
                status = "✅" if info['success'] else "❌"
                print(f"   {metric}: {info['achieved']:.4f} / {info['target']:.4f} {status}")
        
        # Mostrar onde os modelos foram salvos
        if 'save_paths' in results:
            print(f"\n💾 Modelos treinados salvos em:")
            for path_name, path_value in results['save_paths'].items():
                print(f"   📁 {path_name}: {path_value}")
        
        print(f"\n✅ Treinamento concluído com sucesso!")
        print(f"🎉 Sistema ML Trading v3.0 está pronto para uso!")
        
    except Exception as e:
        print(f"\n❌ Erro durante treinamento: {e}")
        print(f"🔍 Detalhes do erro:")
        import traceback
        traceback.print_exc()
        
        print(f"\n💡 Possíveis soluções:")
        print(f"   1. Verifique se todos os arquivos da ETAPA 6 estão presentes")
        print(f"   2. Confirme que há dados CSV no diretório {config['data_path']}")
        print(f"   3. Execute: pip install -r requirements.txt")
        print(f"   4. Verifique os logs acima para mais detalhes")

if __name__ == "__main__":
    main()
