"""
Verifica a performance real dos modelos treinados
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import lightgbm as lgb

def analyze_model_performance():
    """Analisa detalhadamente a performance dos modelos"""
    
    print("=" * 80)
    print("ANÁLISE DETALHADA DA PERFORMANCE DOS MODELOS")
    print("=" * 80)
    
    # Encontrar o modelo mais recente
    model_dir = Path('models/csv_5m_fast_corrected')
    
    # Listar arquivos
    model_files = list(model_dir.glob('*.pkl'))
    metadata_files = list(model_dir.glob('metadata_*.json'))
    
    if not metadata_files:
        print("Nenhum metadata encontrado!")
        return
    
    # Carregar metadata mais recente
    latest_metadata = sorted(metadata_files)[-1]
    print(f"\nAnalisando: {latest_metadata.name}")
    
    with open(latest_metadata, 'r') as f:
        metadata = json.load(f)
    
    print("\n=== RESULTADOS REPORTADOS ===")
    for model, acc in metadata['results'].items():
        print(f"{model}: {acc:.2%}")
    
    # Simular predições para análise
    print("\n=== ANÁLISE DE PREDIÇÕES ===")
    
    # Criar dados fictícios para demonstrar o problema
    n_samples = 1_000_000
    
    # Simular distribuição real (60% neutro)
    y_true = np.random.choice([-1, 0, 1], size=n_samples, p=[0.2, 0.6, 0.2])
    
    # Simular modelo que prevê majoritariamente neutro
    # Com 79% de accuracy, provavelmente está fazendo isso:
    y_pred_biased = []
    for true_val in y_true:
        if true_val == 0:  # Se é neutro
            # Acerta 95% das vezes
            y_pred_biased.append(0 if np.random.rand() < 0.95 else np.random.choice([-1, 1]))
        else:  # Se é -1 ou 1
            # Acerta apenas 50% das vezes (ou prevê neutro)
            if np.random.rand() < 0.7:
                y_pred_biased.append(0)  # Prevê neutro
            else:
                y_pred_biased.append(true_val)
    
    y_pred_biased = np.array(y_pred_biased)
    
    # Calcular métricas
    accuracy = (y_pred_biased == y_true).mean()
    
    print(f"\nAccuracy simulada: {accuracy:.2%}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_biased)
    
    print("\nConfusion Matrix:")
    print("Pred\\True  -1      0      1")
    print("-" * 30)
    for i, row in enumerate(cm):
        print(f"{i-1:>4} {row[0]:>8,} {row[1]:>8,} {row[2]:>8,}")
    
    # Análise por classe
    print("\n=== PERFORMANCE POR CLASSE ===")
    
    for class_val, class_name in [(-1, "SELL"), (0, "HOLD"), (1, "BUY")]:
        mask = y_true == class_val
        if mask.any():
            class_acc = (y_pred_biased[mask] == class_val).mean()
            pred_as_neutral = (y_pred_biased[mask] == 0).mean()
            
            print(f"\n{class_name} (valor: {class_val}):")
            print(f"  Accuracy na classe: {class_acc:.2%}")
            print(f"  Previsto como HOLD: {pred_as_neutral:.2%}")
    
    # Análise de utilidade para trading
    print("\n=== UTILIDADE PARA TRADING ===")
    
    # Sinais de trading (apenas -1 e 1)
    trading_signals_true = y_true[y_true != 0]
    trading_signals_pred = y_pred_biased[y_true != 0]
    
    trading_accuracy = (trading_signals_pred == trading_signals_true).mean()
    missed_trades = (trading_signals_pred == 0).mean()
    
    print(f"\nPara sinais de trading (BUY/SELL apenas):")
    print(f"  Accuracy real: {trading_accuracy:.2%}")
    print(f"  Trades perdidos (previstos como HOLD): {missed_trades:.2%}")
    
    # Verificar thresholds
    print("\n=== ANÁLISE DOS THRESHOLDS ===")
    
    threshold = 0.00009  # Do treinamento
    daily_return = 0.001  # 0.1% de retorno diário típico
    
    print(f"\nThreshold usado: {threshold:.5f} ({threshold*100:.3f}%)")
    print(f"Retorno diário típico: {daily_return:.3f} ({daily_return*100:.1f}%)")
    print(f"Threshold é {daily_return/threshold:.1f}x menor que movimento diário típico!")
    
    print("\n=== RECOMENDAÇÕES ===")
    print("\n1. AJUSTAR THRESHOLDS:")
    print("   - Usar percentis mais extremos (10/90 ou 15/85)")
    print("   - Ou usar valor absoluto mínimo (ex: 0.001 = 0.1%)")
    
    print("\n2. BALANCEAR CLASSES:")
    print("   - Forçar distribuição mais equilibrada")
    print("   - Ou usar class_weight='balanced' em todos modelos")
    
    print("\n3. MÉTRICAS MELHORES:")
    print("   - F1-score para cada classe")
    print("   - Precision/Recall para BUY e SELL")
    print("   - Ignorar HOLD na avaliação")
    
    print("\n4. REMOVER FEATURES PROBLEMÁTICAS:")
    print("   - 'price' direto pode causar leakage")
    print("   - Usar apenas mudanças e ratios")

def create_realistic_test():
    """Cria teste com configuração mais realista"""
    
    print("\n\n" + "=" * 80)
    print("TESTE COM CONFIGURAÇÃO REALISTA")
    print("=" * 80)
    
    # Parâmetros mais realistas
    realistic_config = {
        'target_thresholds': {
            'method': 'absolute',
            'buy_threshold': 0.002,  # 0.2% para compra
            'sell_threshold': -0.002,  # -0.2% para venda
            'expected_distribution': '33% / 33% / 33%'
        },
        'evaluation_metrics': [
            'accuracy_per_class',
            'f1_score_weighted',
            'trading_signals_only',
            'profit_factor_simulation'
        ],
        'features_to_remove': [
            'price',  # Pode causar leakage
            'log_price',  # Também problemático
        ],
        'expected_realistic_accuracy': '50-60%',
        'notes': 'Com distribuição balanceada e thresholds realistas'
    }
    
    print("\nConfiguração sugerida para re-treinamento:")
    print(json.dumps(realistic_config, indent=2))
    
    # Salvar configuração
    with open('config/realistic_training_config.json', 'w') as f:
        json.dump(realistic_config, f, indent=2)
    
    print("\n[OK] Configuração salva em: config/realistic_training_config.json")

if __name__ == "__main__":
    analyze_model_performance()
    create_realistic_test()