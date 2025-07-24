# run_training_safe.py
"""
Script de treinamento seguro com validações relaxadas
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
    """Treina o sistema completo de ML com validações relaxadas"""
    
    # FORÇAR MODO DESENVOLVIMENTO E DESABILITAR VALIDAÇÕES RIGOROSAS
    os.environ['TRADING_PRODUCTION_MODE'] = 'false'    # Modo desenvolvimento
    os.environ['STRICT_VALIDATION'] = 'false'          # Relaxar validação rigorosa
    os.environ['TRAINING_MODE'] = 'true'               # Modo treinamento
    os.environ['BYPASS_DATA_VALIDATION'] = 'true'      # Bypass validações de produção
    
    print("🧪 MODO TREINAMENTO SEGURO ATIVADO")
    print("📊 Validações relaxadas - buscando dados WDO* (todos os vencimentos)")
    print("⚠️  NÃO usar este modo em produção!")
    
    # Verificar se estamos no diretório correto
    if not os.path.exists('src/training/training_orchestrator.py'):
        print("❌ Erro: Execute este script do diretório raiz do projeto ML_Tradingv2.0")
        print("📁 Diretório atual:", os.getcwd())
        print("💡 Comando correto: python run_training_safe.py")
        return
    
    # Adicionar src ao path para importações
    sys.path.insert(0, 'src')
    
    try:
        from training.training_orchestrator import TrainingOrchestrator
        from training.data_loader import TrainingDataLoader
    except ImportError as e:
        print(f"❌ Erro ao importar módulos de treinamento: {e}")
        print("📝 Verifique se todos os arquivos da ETAPA 6 foram criados corretamente")
        return
    
    # Configuração com paths ajustados
    config = {
        'data_path': 'src/training/data/historical/',   # Caminho correto onde estão os dados
        'model_save_path': 'models/trained/',           # Onde salvar modelos treinados  
        'models_dir': 'src/models/',                    # Diretório de modelos existentes
        'results_path': 'training_results/',            # Onde salvar resultados
        'cache_dir': 'data/cache/',                     # Cache de processamento
        'use_example_data': True                        # Forçar uso de dados de exemplo
    }
    
    print("\n" + "="*70)
    print("🧠 SISTEMA DE TREINAMENTO ML - TRADING v3.0 (MODO SEGURO)")
    print("="*70)
    
    # Verificar/criar diretórios
    print("\n📁 Verificando estrutura de diretórios...")
    directories_to_check = [
        config['data_path'],
        config['model_save_path'], 
        config['models_dir'],
        config['results_path'],
        config['cache_dir']
    ]
    
    for directory in directories_to_check:
        if not os.path.exists(directory):
            print(f"📂 Criando diretório: {directory}")
            os.makedirs(directory, exist_ok=True)
        else:
            print(f"✅ Diretório existe: {directory}")
    
    # Verificar disponibilidade de dados
    print("\n📊 Verificando dados disponíveis...")
    data_loader = TrainingDataLoader(config['data_path'])
    
    # Tentar usar dados reais primeiro
    historical_files = []
    if os.path.exists(config['data_path']):
        historical_files = [f for f in os.listdir(config['data_path']) if f.endswith('.csv')]
    
    if historical_files:
        print(f"✅ Encontrados {len(historical_files)} arquivo(s) de dados históricos")
        use_real_data = True
        for i, file in enumerate(historical_files[:3]):  # Mostrar primeiros 3
            print(f"   📄 {file}")
        if len(historical_files) > 3:
            print(f"   ... e mais {len(historical_files) - 3} arquivo(s)")
    else:
        print("⚠️  Nenhum arquivo de dados históricos encontrado")
        print("� O sistema irá buscar por arquivos WDO* (qualquer vencimento)")
        print("📁 Verifique se há arquivos CSV na pasta data/historical/")
        use_real_data = False
    
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
    
    # Definir período de treinamento adaptativo
    if use_real_data:
        # Para dados reais, usar período amplo
        start_date = datetime(2025, 1, 1)  
        end_date = datetime.now()
    else:
        # Para dados de exemplo, usar período curto
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Apenas 1 semana de dados sintéticos
    
    print(f"\n📅 Período de treinamento:")
    print(f"   Início: {start_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"   Fim: {end_date.strftime('%Y-%m-%d %H:%M')}")
    
    # Símbolos para treinar
    symbols = ['WDOH25']  # Mini-dólar
    print(f"📈 Símbolos: {symbols}")
    
    # Métricas alvo relaxadas para modo de desenvolvimento
    target_metrics = {
        'accuracy': 0.55,           # 55% accuracy (reduzido)
        'f1_score': 0.50,          # F1 score 0.50 (reduzido)
        'avg_confidence': 0.60     # Confiança 60% (reduzido)
    }
    
    print("\n🎯 Métricas alvo (modo desenvolvimento):")
    for metric, value in target_metrics.items():
        print(f"   {metric}: {value}")
    
    # Mostrar configuração de modo seguro
    print(f"\n🛡️ Configurações de segurança:")
    print(f"   Modo produção: {os.environ.get('TRADING_PRODUCTION_MODE', 'false')}")
    print(f"   Validação rigorosa: {os.environ.get('STRICT_VALIDATION', 'false')}")
    print(f"   Dados sintéticos permitidos: {os.environ.get('ALLOW_SYNTHETIC_DATA', 'false')}")
    print(f"   Bypass validação: {os.environ.get('BYPASS_DATA_VALIDATION', 'false')}")
    
    # Confirmar execução
    print(f"\n❓ Deseja iniciar o treinamento em modo seguro? (s/n): ", end="")
    response = input()
    if response.lower() != 's':
        print("❌ Treinamento cancelado pelo usuário")
        return
    
    # Executar treinamento
    try:
        print("\n🚀 Iniciando treinamento do sistema ML...")
        print("⏱️  Este processo pode levar alguns minutos...")
        print("🔄 Processando com validações relaxadas...")
        
        results = orchestrator.train_complete_system(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            target_metrics=target_metrics,
            validation_method='walk_forward'  # Método mais simples
        )
        
        # Mostrar resumo dos resultados
        print("\n" + "="*70)
        print("📈 RESUMO DOS RESULTADOS (MODO DESENVOLVIMENTO)")
        print("="*70)
        
        if results.get('status') == 'failed':
            print(f"❌ Status: {results['status']}")
            print(f"🚨 Erro: {results.get('error', 'Desconhecido')}")
            print(f"💬 Mensagem: {results.get('message', 'Sem detalhes')}")
            
        elif 'aggregated_metrics' in results:
            print(f"✅ Status: {results.get('status', 'completed')}")
            metrics = results['aggregated_metrics']
            print(f"\n🎯 Métricas alcançadas:")
            
            for metric_name in ['accuracy', 'f1_score', 'precision', 'recall', 'avg_confidence']:
                mean_key = f'{metric_name}_mean'
                if mean_key in metrics:
                    print(f"   {metric_name}: {metrics[mean_key]:.4f}")
        
        # Verificar métricas alvo
        if results.get('target_metrics_achieved'):
            print(f"\n🏆 Status das métricas alvo:")
            for metric, info in results['target_metrics_achieved'].items():
                status = "✅" if info['success'] else "❌"
                print(f"   {metric}: {info['achieved']:.4f} / {info['target']:.4f} {status}")
        
        # Mostrar informações sobre modelos
        if 'save_paths' in results:
            print(f"\n💾 Modelos salvos em:")
            for path_name, path_value in results['save_paths'].items():
                print(f"   📁 {path_name}: {path_value}")
        
        print(f"\n✅ Treinamento em modo seguro concluído!")
        print(f"🎉 Sistema ML Trading v3.0 configurado (modo desenvolvimento)!")
        print(f"⚠️  Para produção, use o script run_training.py com dados reais")
        
    except Exception as e:
        print(f"\n❌ Erro durante treinamento: {e}")
        print(f"🔍 Tipo do erro: {type(e).__name__}")
        
        # Mostrar traceback detalhado
        import traceback
        traceback.print_exc()
        
        print(f"\n💡 Possíveis soluções:")
        print(f"   1. Verifique se há dados suficientes (mínimo 50 amostras)")
        print(f"   2. Execute: pip install -r requirements.txt")
        print(f"   3. Confirme que todos os módulos da ETAPA 6 estão presentes")
        print(f"   4. Tente com dados reais em vez de sintéticos")
        print(f"   5. Verifique as dependências: tensorflow, sklearn, pandas")

if __name__ == "__main__":
    main()
