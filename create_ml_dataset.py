"""
Script para criar dataset de treinamento ML com dados reais do ProfitDLL
"""

from src.ml.dataset_builder_v3 import DatasetBuilderV3
from datetime import datetime, timedelta
import logging
import os

def main():
    """Cria dataset de treinamento com dados reais"""
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*60)
    print("CRIAÇÃO DE DATASET ML COM DADOS REAIS")
    print("="*60)
    
    # Configurações do dataset
    config = {
        'lookback_periods': 100,      # Períodos de lookback para features
        'target_periods': 5,          # Períodos futuros para criar labels
        'target_threshold': 0.001,    # Threshold para classificação (0.1%)
        'train_ratio': 0.7,          # 70% para treino
        'valid_ratio': 0.15,         # 15% para validação
        'test_ratio': 0.15,          # 15% para teste
        'data_path': 'data/',
        'dataset_path': 'datasets/',
        'model_path': 'models/'
    }
    
    # Criar builder
    builder = DatasetBuilderV3(config)
    
    # Definir período de coleta
    # Por padrão, últimos 30 dias
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Ticker do WDO (ajustar conforme o mês)
    # Formato: WDOXYY onde X é a letra do mês e YY o ano
    # Q = Agosto/2025
    ticker = 'WDOQ25'  # Agosto/2025
    
    print(f"\nPeríodo: {start_date.date()} até {end_date.date()}")
    print(f"Ticker: {ticker}")
    print(f"Timeframe: 1min")
    print("\nIniciando coleta de dados reais...\n")
    
    try:
        # Construir dataset
        datasets = builder.build_training_dataset(
            start_date=start_date,
            end_date=end_date,
            ticker=ticker,
            timeframe='1min'
        )
        
        if datasets:
            print("\n" + "="*60)
            print("DATASET CRIADO COM SUCESSO!")
            print("="*60)
            
            # Mostrar informações dos datasets
            for split_name, split_data in datasets.items():
                if isinstance(split_data, dict) and 'X' in split_data:
                    print(f"\n{split_name.upper()}:")
                    print(f"  Samples: {len(split_data['X'])}")
                    print(f"  Features: {split_data['X'].shape[1] if len(split_data['X']) > 0 else 0}")
                    
                    if 'y' in split_data and len(split_data['y']) > 0:
                        # Distribuição de classes
                        import pandas as pd
                        y_series = pd.Series(split_data['y'])
                        class_dist = y_series.value_counts(normalize=True)
                        print(f"  Distribuição de classes:")
                        for class_val, pct in class_dist.items():
                            class_name = {0: 'DOWN', 1: 'NEUTRAL', 2: 'UP'}.get(class_val, str(class_val))
                            print(f"    {class_name}: {pct:.1%}")
            
            print(f"\nDatasets salvos em: {builder.dataset_path}")
            print("\nPróximos passos:")
            print("1. Verificar os datasets gerados em 'datasets/'")
            print("2. Executar treinamento com TrainingOrchestrator")
            print("3. Validar modelos treinados")
            
        else:
            print("\n[ERRO] Falha na criação do dataset")
            print("Verifique:")
            print("- ProfitDLL está rodando")
            print("- Você está logado no ProfitChart")
            print("- O ticker está correto")
            
    except Exception as e:
        print(f"\n[ERRO] Erro durante criação do dataset: {e}")
        print("\nDicas de resolução:")
        print("1. Certifique-se que o ProfitChart está aberto e logado")
        print("2. Verifique se o ticker está correto (ex: WDOH25)")
        print("3. Confirme que a DLL está no caminho correto")
        print("4. Tente com um período menor primeiro (ex: 7 dias)")


if __name__ == "__main__":
    main()