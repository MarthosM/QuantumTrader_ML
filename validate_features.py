"""
Script de Validação de Features
Verifica compatibilidade entre modelos e features disponíveis
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

class FeatureValidator:
    """Validador de features para modelos ML"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.expected_features = 65
        self.feature_categories = {
            "volatility": 10,
            "returns": 10,
            "order_flow": 8,
            "volume": 8,
            "technical": 8,
            "microstructure": 15,
            "temporal": 6
        }
        
        # Lista completa das 65 features esperadas
        self.all_features = [
            # Volatilidade (10)
            "volatility_10", "volatility_20", "volatility_50", "volatility_100",
            "volatility_ratio_10", "volatility_ratio_20", "volatility_ratio_50", "volatility_ratio_100",
            "volatility_gk", "bb_position",
            
            # Retornos (10)
            "returns_1", "returns_2", "returns_5", "returns_10", "returns_20",
            "returns_50", "returns_100", "log_returns_1", "log_returns_5", "log_returns_20",
            
            # Order Flow (8)
            "order_flow_imbalance_10", "order_flow_imbalance_20",
            "order_flow_imbalance_50", "order_flow_imbalance_100",
            "cumulative_signed_volume", "signed_volume",
            "volume_weighted_return", "agent_turnover",
            
            # Volume (8)
            "volume_ratio_20", "volume_ratio_50", "volume_ratio_100",
            "volume_zscore_20", "volume_zscore_50", "volume_zscore_100",
            "trade_intensity", "trade_intensity_ratio",
            
            # Indicadores Técnicos (8)
            "ma_5_20_ratio", "ma_20_50_ratio",
            "momentum_5_20", "momentum_20_50",
            "sharpe_5", "sharpe_20",
            "time_normalized",
            # bb_position já listado em volatilidade
            
            # Microestrutura (15)
            "top_buyer_0_active", "top_buyer_1_active", "top_buyer_2_active",
            "top_buyer_3_active", "top_buyer_4_active",
            "top_seller_0_active", "top_seller_1_active", "top_seller_2_active",
            "top_seller_3_active", "top_seller_4_active",
            "top_buyers_count", "top_sellers_count",
            "buyer_changed", "seller_changed",
            "is_buyer_aggressor", "is_seller_aggressor",
            
            # Temporais (6)
            "minute", "hour", "day_of_week",
            "is_opening_30min", "is_closing_30min", "is_lunch_hour"
        ]
        
        # Remover duplicata (bb_position aparece 2x)
        self.all_features = list(set(self.all_features))
        
    def validate_models(self) -> Dict:
        """Valida todos os modelos no diretório"""
        print("=" * 60)
        print("VALIDAÇÃO DE MODELOS E FEATURES")
        print("=" * 60)
        
        results = {
            "models_found": [],
            "models_validated": [],
            "models_incompatible": [],
            "summary": {}
        }
        
        # Listar todos os modelos .pkl
        model_files = list(self.models_dir.glob("*.pkl"))
        
        # Filtrar apenas modelos (não scalers)
        model_files = [f for f in model_files if 'scaler' not in f.stem.lower()]
        
        print(f"\nModelos encontrados: {len(model_files)}")
        print("-" * 40)
        
        for model_file in model_files:
            model_name = model_file.stem
            results["models_found"].append(model_name)
            
            print(f"\n[{model_name}]")
            
            try:
                # Carregar modelo
                model = joblib.load(model_file)
                
                # Verificar número de features esperadas
                if hasattr(model, 'n_features_in_'):
                    n_features = model.n_features_in_
                elif hasattr(model, 'n_features_'):
                    n_features = model.n_features_
                else:
                    # Tentar inferir testando
                    n_features = self._infer_features(model)
                
                print(f"  Features esperadas: {n_features}")
                
                # Verificar arquivo JSON de features
                json_file = model_file.with_suffix('.json')
                if json_file.exists():
                    with open(json_file) as f:
                        data = json.load(f)
                        feature_list = data.get('features', [])
                        print(f"  Features no JSON: {len(feature_list)}")
                else:
                    feature_list = []
                    print(f"  [AVISO] Arquivo JSON não encontrado")
                
                # Validar compatibilidade
                if n_features == self.expected_features:
                    print(f"  [OK] Modelo compatível com 65 features")
                    results["models_validated"].append(model_name)
                elif n_features == 11:
                    print(f"  [INFO] Modelo usa features básicas (11)")
                    results["models_incompatible"].append({
                        "name": model_name,
                        "features": n_features,
                        "reason": "Modelo básico com 11 features"
                    })
                else:
                    print(f"  [ERRO] Incompatível - esperado: 65, encontrado: {n_features}")
                    results["models_incompatible"].append({
                        "name": model_name,
                        "features": n_features,
                        "reason": f"Número incorreto de features: {n_features}"
                    })
                
                # Testar predição com dados simulados
                self._test_prediction(model, n_features, model_name)
                
            except Exception as e:
                print(f"  [ERRO] Falha ao validar: {e}")
                results["models_incompatible"].append({
                    "name": model_name,
                    "features": "unknown",
                    "reason": str(e)
                })
        
        # Resumo
        print("\n" + "=" * 60)
        print("RESUMO DA VALIDAÇÃO")
        print("=" * 60)
        
        results["summary"] = {
            "total_models": len(model_files),
            "compatible": len(results["models_validated"]),
            "incompatible": len(results["models_incompatible"]),
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"Total de modelos: {results['summary']['total_models']}")
        print(f"Compatíveis (65 features): {results['summary']['compatible']}")
        print(f"Incompatíveis: {results['summary']['incompatible']}")
        
        if results["models_validated"]:
            print(f"\nModelos prontos para uso:")
            for model in results["models_validated"]:
                print(f"  - {model}")
        
        if results["models_incompatible"]:
            print(f"\nModelos que precisam retreino:")
            for model_info in results["models_incompatible"]:
                print(f"  - {model_info['name']}: {model_info['reason']}")
        
        # Salvar relatório
        report_file = Path("feature_validation_report.json")
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Relatório salvo: {report_file}")
        
        return results
    
    def _infer_features(self, model) -> int:
        """Tenta inferir número de features testando o modelo"""
        for n in [11, 65, 30, 45]:  # Testar tamanhos comuns
            try:
                X = np.zeros((1, n))
                model.predict(X)
                return n
            except:
                continue
        return -1
    
    def _test_prediction(self, model, n_features: int, model_name: str):
        """Testa predição com dados simulados"""
        try:
            # Criar dados de teste
            X_test = np.random.randn(1, n_features)
            
            # Fazer predição
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_test)[0]
                print(f"  Teste predição: probabilidades = {pred}")
            else:
                pred = model.predict(X_test)[0]
                print(f"  Teste predição: valor = {pred}")
                
        except Exception as e:
            print(f"  [ERRO] Falha no teste de predição: {e}")
    
    def validate_feature_calculation(self):
        """Valida se conseguimos calcular todas as 65 features"""
        print("\n" + "=" * 60)
        print("VALIDAÇÃO DO CÁLCULO DE FEATURES")
        print("=" * 60)
        
        missing_features = []
        available_features = []
        
        # Simular dados
        candles = pd.DataFrame({
            'close': np.random.randn(200) * 10 + 5450,
            'open': np.random.randn(200) * 10 + 5450,
            'high': np.random.randn(200) * 10 + 5455,
            'low': np.random.randn(200) * 10 + 5445,
            'volume': np.random.randint(100000, 1000000, 200)
        })
        
        # Verificar cada categoria
        for category, count in self.feature_categories.items():
            print(f"\n[{category.upper()}] ({count} features)")
            
            if category == "volatility":
                # Podemos calcular com candles
                available_features.extend([
                    "volatility_10", "volatility_20", "volatility_50", "volatility_100"
                ])
                print("  [OK] Volatilidades básicas disponíveis")
                
            elif category == "returns":
                # Podemos calcular com candles
                available_features.extend([
                    "returns_1", "returns_2", "returns_5", "returns_10", "returns_20"
                ])
                print("  [OK] Retornos disponíveis")
                
            elif category == "order_flow":
                # Precisa de book data
                missing_features.extend([
                    "order_flow_imbalance_10", "order_flow_imbalance_20"
                ])
                print("  [AVISO] Requer dados de book (não disponível em produção atual)")
                
            elif category == "microstructure":
                # Precisa de book e trade data
                missing_features.extend([
                    "top_buyer_0_active", "is_buyer_aggressor"
                ])
                print("  [AVISO] Requer dados de microestrutura (não disponível)")
                
            elif category == "temporal":
                # Podemos calcular sempre
                available_features.extend([
                    "minute", "hour", "day_of_week"
                ])
                print("  [OK] Features temporais disponíveis")
        
        print("\n" + "-" * 40)
        print(f"Features disponíveis: {len(available_features)}/65")
        print(f"Features faltando: {65 - len(available_features)}")
        
        if len(available_features) < 65:
            print("\n[IMPORTANTE] Sistema atual NÃO consegue calcular todas as features!")
            print("Necessário implementar:")
            print("  1. Buffer para dados de book")
            print("  2. Buffer para trades")
            print("  3. Cálculo de order flow")
            print("  4. Análise de microestrutura")
        
        return available_features, missing_features

def main():
    """Executa validação completa"""
    validator = FeatureValidator()
    
    # 1. Validar modelos
    model_results = validator.validate_models()
    
    # 2. Validar capacidade de cálculo
    available, missing = validator.validate_feature_calculation()
    
    # 3. Recomendações
    print("\n" + "=" * 60)
    print("RECOMENDAÇÕES")
    print("=" * 60)
    
    if model_results["summary"]["compatible"] == 0:
        print("\n[CRÍTICO] Nenhum modelo compatível com 65 features!")
        print("\nOpções:")
        print("1. Retreinar modelos com features disponíveis")
        print("2. Implementar sistema completo de 65 features")
        print("3. Usar modelo temporário simplificado")
    else:
        print("\n[OK] Modelos compatíveis encontrados!")
        print("Próximo passo: Implementar cálculo das 65 features")
    
    print("\n" + "=" * 60)
    print("VALIDAÇÃO COMPLETA")
    print("=" * 60)

if __name__ == "__main__":
    main()