"""
Debug System - Verificar Execução de Predições ML
Identifica por que o sistema não está executando predições
"""

import os
import sys
import time
import logging
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def debug_trading_system():
    """Debug do sistema de trading para verificar predições"""
    
    print("=" * 80)
    print("DEBUG: SISTEMA DE PREDICOES ML")
    print("=" * 80)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    try:
        # Configurar logging para capturar detalhes
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        print("1. IMPORTANDO COMPONENTES...")
        
        from trading_system import TradingSystem
        from data_structure import TradingDataStructure
        from feature_engine import FeatureEngine
        from ml_coordinator import MLCoordinator
        from model_manager import ModelManager
        from data_loader import DataLoader
        
        print("   [OK] Imports realizados")
        
        print("\n2. CRIANDO CONFIGURACAO MINIMA...")
        
        # Configuração mínima para teste
        config = {
            'dll_path': r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll",
            'key': '29936354842',
            'username': '29936354842',
            'password': 'Abc123456',
            'account_id': '70562000',
            'broker_id': '33005',
            'ticker': 'WDOQ25',
            'historical_days': 1,
            'ml_interval': 10,  # 10 segundos para teste
            'use_gui': False,
            'models_dir': r'C:\Users\marth\OneDrive\Programacao\Python\Projetos\ML_Tradingv2.0\models',
            'strategy': {
                'direction_threshold': 0.45,
                'magnitude_threshold': 0.00015,
                'confidence_threshold': 0.15
            },
            'risk': {
                'max_daily_loss': 0.05,
                'max_positions': 1,
                'risk_per_trade': 0.02
            }
        }
        
        print("   [OK] Configuracao criada")
        
        print("\n3. INICIALIZANDO SISTEMA...")
        
        # Criar sistema de trading
        system = TradingSystem(config)
        
        # Verificar inicialização
        print("   [INFO] Inicializando sistema...")
        if not system.initialize():
            print("   [ERROR] Falha na inicializacao")
            return False
        
        print("   [OK] Sistema inicializado")
        
        print("\n4. VERIFICANDO COMPONENTES CRITICOS...")
        
        # Verificar componentes essenciais
        components_status = {
            'connection': system.connection is not None,
            'data_structure': system.data_structure is not None,
            'feature_engine': system.feature_engine is not None,
            'ml_coordinator': system.ml_coordinator is not None,
            'model_manager': system.model_manager is not None,
            'data_loader': system.data_loader is not None
        }
        
        for component, status in components_status.items():
            status_text = "OK" if status else "FALHA"
            print(f"   {component}: {status_text}")
        
        if not all(components_status.values()):
            print("   [ERROR] Componentes criticos faltando!")
            return False
        
        print("\n5. CARREGANDO DADOS DE TESTE...")
        
        # Carregar alguns dados para teste
        data_loader = DataLoader()
        sample_data = data_loader.create_sample_data(100)
        
        # Temporariamente desabilitar validação de produção
        os.environ['TRADING_ENV'] = 'development'
        
        system.data_structure.update_candles(sample_data)
        
        candles_count = len(system.data_structure.candles)
        print(f"   [OK] {candles_count} candles carregados")
        
        print("\n6. TESTANDO CALCULO DE FEATURES...")
        
        # Verificar feature engine
        print("   [INFO] Testando FeatureEngine...")
        
        try:
            features_result = system.feature_engine.calculate(
                data=system.data_structure,
                force_recalculate=True,
                use_advanced=True
            )
            
            if features_result and 'success' in features_result and features_result['success']:
                features_df = features_result.get('features')
                if features_df is not None:
                    print(f"   [OK] Features calculadas: {features_df.shape}")
                    print(f"   [INFO] Colunas: {len(features_df.columns)}")
                    
                    # Verificar NaN
                    nan_count = features_df.isnull().sum().sum()
                    total_values = features_df.size
                    fill_rate = ((total_values - nan_count) / total_values) * 100
                    print(f"   [INFO] Taxa de preenchimento: {fill_rate:.1f}%")
                else:
                    print("   [WARN] Features result sem DataFrame")
            else:
                error_msg = features_result.get('error', 'Erro desconhecido') if features_result else 'Resultado None'
                print(f"   [ERROR] Falha no calculo: {error_msg}")
                
        except Exception as e:
            print(f"   [ERROR] Excecao no calculo de features: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n7. TESTANDO ML COORDINATOR...")
        
        # Verificar ML Coordinator
        if system.ml_coordinator:
            print("   [INFO] Testando MLCoordinator...")
            
            try:
                # Tentar predição
                prediction_result = system.ml_coordinator.process_prediction_request(system.data_structure)
                
                if prediction_result:
                    print("   [OK] Predicao realizada!")
                    print(f"   [INFO] Resultado: {prediction_result}")
                    
                    # Verificar campos essenciais
                    essential_fields = ['action', 'confidence', 'direction']
                    for field in essential_fields:
                        if field in prediction_result:
                            print(f"      {field}: {prediction_result[field]}")
                        else:
                            print(f"      {field}: AUSENTE")
                else:
                    print("   [WARN] Predicao retornou None ou vazio")
                    
            except Exception as e:
                print(f"   [ERROR] Excecao na predicao: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("   [ERROR] MLCoordinator nao disponivel")
        
        print("\n8. VERIFICANDO MODELOS ML...")
        
        # Verificar modelos carregados
        if system.model_manager:
            print("   [INFO] Verificando ModelManager...")
            
            if hasattr(system.model_manager, 'models'):
                models_count = len(system.model_manager.models)
                print(f"   [INFO] Modelos carregados: {models_count}")
                
                if models_count > 0:
                    for model_name in system.model_manager.models.keys():
                        print(f"      - {model_name}")
                else:
                    print("   [WARN] Nenhum modelo carregado")
                    
                    # Tentar carregar modelos
                    print("   [INFO] Tentando carregar modelos...")
                    try:
                        models_dir = config.get('models_dir')
                        if os.path.exists(models_dir):
                            print(f"   [INFO] Diretorio de modelos existe: {models_dir}")
                            files = os.listdir(models_dir)
                            print(f"   [INFO] Arquivos encontrados: {files}")
                        else:
                            print(f"   [WARN] Diretorio de modelos nao existe: {models_dir}")
                    except Exception as e:
                        print(f"   [ERROR] Erro verificando modelos: {e}")
            else:
                print("   [ERROR] ModelManager sem atributo 'models'")
        else:
            print("   [ERROR] ModelManager nao disponivel")
        
        print("\n9. SIMULANDO LOOP PRINCIPAL...")
        
        # Simular algumas iterações do loop principal
        print("   [INFO] Simulando execucao do loop principal...")
        
        # Configurar sistema como "rodando"
        system.is_running = True
        system.last_ml_time = None  # Forçar primeira predição
        
        for i in range(5):
            print(f"\n   --- Iteracao {i+1} ---")
            
            # Verificar se deve executar ML
            should_run_ml = system._should_run_ml()
            print(f"   Should run ML: {should_run_ml}")
            
            if should_run_ml:
                print("   [INFO] Executando predicao ML...")
                
                # Simular processamento ML
                try:
                    system._process_ml_prediction()
                    print("   [OK] Predicao processada")
                    
                    # Verificar resultado
                    if hasattr(system, 'last_prediction') and system.last_prediction:
                        pred = system.last_prediction
                        print(f"   [RESULT] Action: {pred.get('action', 'N/A')}")
                        print(f"   [RESULT] Confidence: {pred.get('confidence', 0):.3f}")
                        print(f"   [RESULT] Direction: {pred.get('direction', 'N/A')}")
                    else:
                        print("   [WARN] Nenhuma predicao armazenada")
                        
                except Exception as e:
                    print(f"   [ERROR] Erro processando predicao: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Aguardar próxima iteração
            time.sleep(2)
        
        print("\n10. DIAGNOSTICO FINAL...")
        
        # Diagnóstico final
        final_diagnosis = []
        
        if not all(components_status.values()):
            final_diagnosis.append("Alguns componentes criticos falharam na inicializacao")
        
        if system.model_manager and len(system.model_manager.models) == 0:
            final_diagnosis.append("Nenhum modelo ML foi carregado")
        
        if not hasattr(system, 'last_prediction') or not system.last_prediction:
            final_diagnosis.append("Sistema nao esta gerando predicoes")
        
        if not final_diagnosis:
            final_diagnosis.append("Sistema parece estar funcionando corretamente")
        
        print("   DIAGNOSTICO:")
        for diagnosis in final_diagnosis:
            print(f"   - {diagnosis}")
        
        print("\n" + "=" * 80)
        print("DEBUG CONCLUIDO")
        print("=" * 80)
        
        return len(final_diagnosis) == 1 and "funcionando corretamente" in final_diagnosis[0]
        
    except Exception as e:
        print(f"\n[ERROR] Erro critico no debug: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Iniciando debug do sistema de predicoes...")
    success = debug_trading_system()
    
    print("\n" + "="*80)
    if success:
        print("DEBUG: SISTEMA FUNCIONANDO")
        print("O sistema esta executando predicoes corretamente.")
    else:
        print("DEBUG: PROBLEMAS IDENTIFICADOS")
        print("O sistema tem problemas que impedem as predicoes.")
    print("="*80)