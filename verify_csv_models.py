"""
Verifica modelos CSV treinados e prepara integração
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime

def check_csv_models():
    """Verifica todos os modelos CSV treinados"""
    
    print("=" * 80)
    print("VERIFICAÇÃO DOS MODELOS CSV TREINADOS")
    print("=" * 80)
    
    # Diretórios de modelos
    model_dirs = [
        Path('models/csv_agent_models'),
        Path('models/csv_tick_models'),
        Path('models/csv_fast_models')
    ]
    
    all_models = []
    
    for model_dir in model_dirs:
        if model_dir.exists():
            print(f"\n=== Diretório: {model_dir} ===")
            
            # Listar arquivos .pkl
            pkl_files = list(model_dir.glob('*.pkl'))
            json_files = list(model_dir.glob('*.json'))
            
            print(f"Modelos encontrados: {len(pkl_files)}")
            print(f"Metadados encontrados: {len(json_files)}")
            
            for pkl_file in pkl_files:
                try:
                    # Carregar modelo
                    model = joblib.load(pkl_file)
                    model_info = {
                        'file': pkl_file.name,
                        'path': str(pkl_file),
                        'type': type(model).__name__,
                        'created': datetime.fromtimestamp(pkl_file.stat().st_mtime)
                    }
                    
                    # Tentar encontrar metadados correspondentes
                    metadata_pattern = pkl_file.stem.replace('csv_', '*')
                    matching_metadata = list(model_dir.glob(f"*{metadata_pattern}*.json"))
                    
                    if matching_metadata:
                        with open(matching_metadata[0], 'r') as f:
                            metadata = json.load(f)
                            model_info['metadata'] = metadata
                    
                    all_models.append(model_info)
                    
                    print(f"\n  Modelo: {pkl_file.name}")
                    print(f"  Tipo: {model_info['type']}")
                    print(f"  Criado: {model_info['created']}")
                    
                    if 'metadata' in model_info:
                        if 'results' in metadata:
                            for model_name, results in metadata['results'].items():
                                if 'accuracy' in results:
                                    print(f"  Accuracy ({model_name}): {results['accuracy']:.2%}")
                    
                except Exception as e:
                    print(f"  Erro ao carregar {pkl_file.name}: {e}")
    
    print(f"\n\nTotal de modelos encontrados: {len(all_models)}")
    
    # Selecionar melhor modelo
    if all_models:
        print("\n=== SELEÇÃO DO MELHOR MODELO ===")
        
        # Ordenar por data de criação (mais recente primeiro)
        all_models.sort(key=lambda x: x['created'], reverse=True)
        
        best_model = all_models[0]
        print(f"\nModelo mais recente: {best_model['file']}")
        print(f"Criado em: {best_model['created']}")
        
        if 'metadata' in best_model:
            print("\nFeatures utilizadas:")
            if 'features' in best_model['metadata']:
                features = best_model['metadata']['features']
                print(f"Total: {len(features)} features")
                print("Principais:", features[:10])
            
            if 'feature_importance' in best_model['metadata']:
                print("\nTop 5 features mais importantes:")
                for feat in best_model['metadata']['feature_importance'][:5]:
                    print(f"  - {feat['feature']}: {feat['importance']:.4f}")
    
    return all_models

def compare_with_book_features():
    """Compara features do CSV com Book Collector"""
    
    print("\n\n" + "=" * 80)
    print("COMPARAÇÃO: FEATURES CSV vs BOOK COLLECTOR")
    print("=" * 80)
    
    # Features do CSV (baseado nos modelos treinados)
    csv_features = [
        # Preço
        'price', 'price_log', 'returns_1', 'returns_5', 'returns_20',
        'volatility_20', 'volatility_50',
        
        # Volume
        'qty', 'qty_log', 'volume_ratio',
        
        # Trade Flow
        'is_buyer_aggressor', 'is_seller_aggressor', 'aggressor_imbalance',
        
        # Agents
        'top_buyer_0-4', 'top_seller_0-4', 'top_buyers_active', 'top_sellers_active',
        
        # Temporal
        'hour', 'minute', 'trade_intensity', 'is_opening', 'is_closing',
        
        # Trade size
        'is_large_trade'
    ]
    
    # Features potenciais do Book Collector
    book_features = [
        # Spread
        'bid_ask_spread', 'spread_ratio', 'spread_volatility',
        
        # Book Imbalance
        'book_imbalance', 'bid_volume', 'ask_volume', 'volume_imbalance',
        
        # Depth
        'book_depth', 'bid_depth', 'ask_depth', 'depth_ratio',
        
        # Microstructure
        'mid_price', 'microprice', 'weighted_mid_price',
        
        # Book dynamics
        'book_updates_per_second', 'order_flow_imbalance',
        
        # Liquidity
        'liquidity_score', 'available_liquidity'
    ]
    
    print("\n=== Features Exclusivas do CSV ===")
    print("(Comportamento de Agentes)")
    agent_features = ['top_buyer_*', 'top_seller_*', 'top_buyers_active', 'top_sellers_active']
    for feat in agent_features:
        print(f"  - {feat}")
    
    print("\n=== Features Exclusivas do Book Collector ===")
    print("(Microestrutura)")
    for feat in book_features[:8]:
        print(f"  - {feat}")
    
    print("\n=== Estratégia de Integração ===")
    print("\n1. MODELO HÍBRIDO:")
    print("   - Base: Features de agent behavior (CSV)")
    print("   - Timing: Features de microestrutura (Book)")
    print("   - Ensemble: Weighted average das predições")
    
    print("\n2. PIPELINE SUGERIDO:")
    print("   a) Treinar modelo base com CSV (contexto macro)")
    print("   b) Treinar modelo micro com Book Collector")
    print("   c) Criar meta-modelo que combina ambos")
    print("   d) Backtest com ambas fontes sincronizadas")

def prepare_integration_config():
    """Prepara configuração para integração"""
    
    print("\n\n" + "=" * 80)
    print("CONFIGURAÇÃO PARA INTEGRAÇÃO")
    print("=" * 80)
    
    config = {
        'csv_models': {
            'path': 'models/csv_fast_models',
            'features_required': [
                'price', 'returns_1', 'returns_5', 'returns_20',
                'volatility_20', 'aggressor_imbalance', 'trade_intensity'
            ],
            'update_frequency': '1min',
            'weight': 0.6
        },
        'book_models': {
            'path': 'models/book_microstructure',
            'features_required': [
                'bid_ask_spread', 'book_imbalance', 'mid_price',
                'liquidity_score'
            ],
            'update_frequency': '1s',
            'weight': 0.4
        },
        'ensemble': {
            'method': 'weighted_average',
            'confidence_threshold': 0.6,
            'position_sizing': 'kelly_criterion',
            'max_position': 5
        },
        'data_sync': {
            'csv_lag': '0s',  # CSV é histórico
            'book_lag': '100ms',  # Book é real-time
            'sync_method': 'forward_fill'
        }
    }
    
    # Salvar configuração
    config_file = Path('config/csv_book_integration.json')
    config_file.parent.mkdir(exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguração salva em: {config_file}")
    
    print("\n=== Próximos Passos para Integração ===")
    print("1. Coletar mais dados do Book Collector (mínimo 1h)")
    print("2. Treinar modelo de microestrutura com Book data")
    print("3. Sincronizar timestamps entre CSV e Book")
    print("4. Implementar ensemble_predictor.py")
    print("5. Backtest com dados combinados")
    
    return config

def main():
    """Executa verificação completa"""
    
    # 1. Verificar modelos CSV
    models = check_csv_models()
    
    # 2. Comparar features
    compare_with_book_features()
    
    # 3. Preparar integração
    config = prepare_integration_config()
    
    print("\n\n" + "=" * 80)
    print("RESUMO EXECUTIVO")
    print("=" * 80)
    
    print(f"\n✓ Modelos CSV treinados: {len(models)}")
    print("✓ Features de agent behavior implementadas")
    print("✓ Configuração de integração preparada")
    print("\n✗ Pendente: Modelos do Book Collector")
    print("✗ Pendente: Pipeline de ensemble")
    print("✗ Pendente: Backtest integrado")
    
    print("\n[STATUS] Sistema pronto para próxima fase!")

if __name__ == "__main__":
    main()