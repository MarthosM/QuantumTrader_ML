"""
Teste específico do fluxo de carregamento de dados históricos
"""

import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configurar logging para debug
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_historical_data_flow():
    print("=" * 80)
    print("TESTE: FLUXO DE CARREGAMENTO DE DADOS HISTÓRICOS")
    print("=" * 80)
    
    # Configurar ambiente
    os.environ['TRADING_ENV'] = 'development'
    
    try:
        # Inicializar sistema
        from src.main import load_config
        from src.trading_system import TradingSystem
        
        config = load_config()
        trading_system = TradingSystem(config)
        
        # Garantir que o sistema está inicializado
        if not trading_system.initialized:
            print("   Inicializando sistema...")
            success = trading_system.initialize()
            if not success:
                print("   ERRO: Falha na inicialização do sistema")
                return False
        
        print(f"1. Sistema inicializado")
        print(f"   Ticker: {trading_system.ticker}")
        print(f"   Connection: {trading_system.connection is not None}")
        print(f"   Connected: {trading_system.connection.connected if trading_system.connection else False}")
        
        # Testar carregamento de dados históricos
        print("\n2. Testando carregamento de dados históricos...")
        
        ticker = "WDOQ25"  # Corresponde ao arquivo de teste
        days_back = 1
        
        success = trading_system._load_historical_data_safe(ticker, days_back)
        print(f"   Resultado: {'SUCESSO' if success else 'FALHA'}")
        
        if success:
            candles_count = len(trading_system.data_structure.candles) if trading_system.data_structure else 0
            print(f"   Candles carregados: {candles_count}")
            
            if candles_count > 0:
                first_candle = trading_system.data_structure.candles.iloc[0]
                last_candle = trading_system.data_structure.candles.iloc[-1]
                print(f"   Primeiro candle: {first_candle.name}")
                print(f"   Último candle: {last_candle.name}")
                
                # Testar cálculo de features
                print("\n3. Testando cálculo de features...")
                try:
                    trading_system._calculate_initial_features()
                    features_count = len(trading_system.data_structure.features.columns) if hasattr(trading_system.data_structure, 'features') and trading_system.data_structure.features is not None else 0
                    print(f"   Features calculadas: {features_count}")
                    
                    if features_count > 0:
                        print("   ✓ Features calculadas com sucesso!")
                        return True
                    else:
                        print("   ✗ Features não foram calculadas")
                        return False
                        
                except Exception as e:
                    print(f"   ✗ Erro calculando features: {e}")
                    return False
            else:
                print("   ✗ Nenhum candle foi carregado")
                return False
        else:
            print("   FALHA no carregamento de dados historicos")
            
            # Debug: verificar se arquivo de teste existe
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            test_file = os.path.join(base_dir, "tests", "data", f"{ticker}_test_data.csv")
            print(f"   Arquivo de teste: {test_file}")
            print(f"   Existe: {os.path.exists(test_file)}")
            
            if os.path.exists(test_file):
                # Tentar carregar manualmente
                print("   Tentando carregar manualmente...")
                import pandas as pd
                try:
                    test_df = pd.read_csv(test_file, parse_dates=['timestamp'], index_col='timestamp')
                    print(f"   Dados no arquivo: {len(test_df)} linhas")
                    print(f"   Período: {test_df.index[0]} até {test_df.index[-1]}")
                except Exception as e:
                    print(f"   Erro lendo arquivo: {e}")
            
            return False
            
    except Exception as e:
        print(f"ERRO GERAL: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_historical_data_flow()
    print("\n" + "=" * 80)
    if success:
        print("SUCESSO: Dados históricos carregados e features calculadas!")
    else:  
        print("FALHA: Problema no carregamento ou cálculo de features")
    print("=" * 80)