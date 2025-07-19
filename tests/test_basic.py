#!/usr/bin/env python3
"""
Teste Básico do Sistema - Diagnóstico
"""

import os
import sys

print("=" * 60)
print("🚀 TESTE BÁSICO DO SISTEMA ML TRADING v2.0")
print("=" * 60)

# Verificar diretório atual
print(f"📁 Diretório atual: {os.getcwd()}")
print(f"🐍 Python: {sys.version}")
print(f"📂 Arquivos no diretório:")
for f in os.listdir('.'):
    if f.endswith('.py'):
        print(f"  - {f}")

print("\n📦 Testando imports básicos...")

# Testar import do src
src_path = os.path.join(os.getcwd(), 'src')
print(f"📂 Caminho src: {src_path}")
print(f"🔍 Src existe: {os.path.exists(src_path)}")

if os.path.exists(src_path):
    print("📄 Arquivos em src:")
    for f in os.listdir(src_path):
        if f.endswith('.py'):
            print(f"  - {f}")
    
    # Adicionar ao path
    sys.path.insert(0, src_path)
    print("✅ Src adicionado ao path")
    
    # Testar imports um por um
    test_imports = [
        'connection_manager',
        'model_manager', 
        'data_structure',
        'data_loader',
        'feature_engine',
        'technical_indicators',
        'ml_features'
    ]
    
    success_count = 0
    
    for module_name in test_imports:
        try:
            module = __import__(module_name)
            print(f"✅ {module_name}: OK")
            success_count += 1
        except Exception as e:
            print(f"❌ {module_name}: {str(e)}")
    
    print(f"\n📊 Resultado: {success_count}/{len(test_imports)} imports OK")
    
    if success_count >= len(test_imports) * 0.8:
        print("✅ Sistema básico funcional!")
        
        # Teste básico de componentes
        print("\n🧪 Testando componentes básicos...")
        
        try:
            from data_structure import TradingDataStructure
            ds = TradingDataStructure()
            ds.initialize_structure()
            print("✅ TradingDataStructure: OK")
        except Exception as e:
            print(f"❌ TradingDataStructure: {e}")
            
        try:
            from data_loader import DataLoader
            dl = DataLoader("test_temp")
            print("✅ DataLoader: OK")
        except Exception as e:
            print(f"❌ DataLoader: {e}")
            
        try:
            from technical_indicators import TechnicalIndicators
            ti = TechnicalIndicators()
            print("✅ TechnicalIndicators: OK")
        except Exception as e:
            print(f"❌ TechnicalIndicators: {e}")
            
        try:
            from feature_engine import ProductionDataValidator
            import logging
            logger = logging.getLogger('test')
            validator = ProductionDataValidator(logger)
            print("✅ ProductionDataValidator: OK")
        except Exception as e:
            print(f"❌ ProductionDataValidator: {e}")
    
    else:
        print("❌ Muitos imports falharam - verificar dependências")
        
else:
    print("❌ Diretório src não encontrado")

print("\n" + "=" * 60)
print("🏁 TESTE BÁSICO FINALIZADO")
print("=" * 60)
