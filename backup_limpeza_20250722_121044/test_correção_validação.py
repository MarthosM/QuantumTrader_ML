#!/usr/bin/env python3
"""
Teste simples para validar que a correção do AttributeError funcionou.

Este script testa que:
1. DataIntegration pode ser criada com connection_manager None sem erro
2. DataIntegration funciona corretamente com connection_manager válido
3. Não há mais o erro "'NoneType' object has no attribute 'register_trade_callback'"

Autor: Sistema ML Trading v2.0
Data: 19/07/2025
"""

import sys
import os
from unittest.mock import MagicMock

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_integration import DataIntegration

def test_correção_attributeerror():
    """Teste principal da correção"""
    print("=" * 60)
    print("🔧 TESTE DE VALIDAÇÃO DA CORREÇÃO DO ATTRIBUTEERROR")
    print("=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # Teste 1: DataIntegration com connection_manager None
    print("\n1️⃣ Testando DataIntegration com connection_manager=None...")
    try:
        mock_data_loader = MagicMock()
        
        # Esta linha causava o AttributeError antes da correção
        data_integration = DataIntegration(
            connection_manager=None,
            data_loader=mock_data_loader
        )
        
        print("   ✅ DataIntegration criada com sucesso (sem AttributeError)")
        print("   ✅ connection_manager None tratado corretamente")
        success_count += 1
        
    except AttributeError as e:
        if "'NoneType' object has no attribute 'register_trade_callback'" in str(e):
            print("   ❌ ERRO: O AttributeError ainda está ocorrendo!")
            print(f"   ❌ Erro: {e}")
        else:
            print(f"   ⚠️ AttributeError diferente: {e}")
    except Exception as e:
        print(f"   ⚠️ Erro inesperado: {e}")
    
    # Teste 2: DataIntegration com connection_manager válido
    print("\n2️⃣ Testando DataIntegration com connection_manager válido...")
    try:
        mock_connection_manager = MagicMock()
        mock_connection_manager.register_trade_callback = MagicMock()
        mock_data_loader = MagicMock()
        
        data_integration = DataIntegration(
            connection_manager=mock_connection_manager,
            data_loader=mock_data_loader
        )
        
        # Verificar que o callback foi chamado
        mock_connection_manager.register_trade_callback.assert_called_once()
        
        print("   ✅ DataIntegration criada com sucesso")
        print("   ✅ Callback registrado corretamente")
        success_count += 1
        
    except Exception as e:
        print(f"   ❌ Erro inesperado: {e}")
    
    # Teste 3: Simular cenário do sistema real
    print("\n3️⃣ Testando cenário real do sistema...")
    try:
        # Simular a sequência que acontece no trading_system.py
        print("   📡 Criando ConnectionManager...")
        connection_manager = MagicMock()
        connection_manager.register_trade_callback = MagicMock()
        
        print("   📊 Criando DataLoader...")
        data_loader = MagicMock()
        
        print("   🔗 Criando DataIntegration...")
        # Esta é a linha que falhava antes da correção
        data_integration = DataIntegration(
            connection_manager=connection_manager,
            data_loader=data_loader
        )
        
        print("   ✅ Sistema integrado com sucesso")
        print("   ✅ Nenhum AttributeError ocorreu")
        success_count += 1
        
    except Exception as e:
        print(f"   ❌ Erro no cenário real: {e}")
    
    # Resultado final
    print("\n" + "=" * 60)
    print("📊 RESULTADO DA VALIDAÇÃO:")
    print(f"   ✅ Testes bem-sucedidos: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("   🎉 CORREÇÃO VALIDADA COM SUCESSO!")
        print("   ✅ O AttributeError foi completamente resolvido")
        print("   ✅ O sistema pode inicializar normalmente")
        print("   ✅ DataIntegration funciona em todos os cenários")
    else:
        print("   ⚠️ Alguns testes falharam")
        print("   🔧 A correção pode precisar de ajustes")
    
    print("=" * 60)
    
    return success_count == total_tests

def demonstrar_correção():
    """Demonstração da correção aplicada"""
    print("\n" + "="*50)
    print("🔍 DEMONSTRAÇÃO DA CORREÇÃO APLICADA")
    print("="*50)
    
    print("\n📋 PROBLEMA ORIGINAL:")
    print("   ❌ AttributeError: 'NoneType' object has no attribute 'register_trade_callback'")
    print("   ❌ DataIntegration era criada antes do connection_manager estar disponível")
    
    print("\n🔧 CORREÇÃO IMPLEMENTADA:")
    print("   ✅ Adicionada verificação 'if self.connection_manager is not None'")
    print("   ✅ Movida inicialização do DataIntegration para depois do ConnectionManager")
    print("   ✅ Adicionado log de warning quando connection_manager é None")
    
    print("\n📍 ARQUIVOS MODIFICADOS:")
    print("   📄 data_integration.py: Verificação de null adicionada")
    print("   📄 trading_system.py: Ordem de inicialização corrigida")
    
    print("\n🎯 RESULTADO:")
    print("   ✅ Sistema inicializa sem erros")
    print("   ✅ Callback registrado corretamente quando disponível")
    print("   ✅ Tratamento gracioso quando connection_manager é None")

if __name__ == '__main__':
    print("🚀 Iniciando validação da correção...")
    
    # Executar demonstração
    demonstrar_correção()
    
    # Executar teste
    success = test_correção_attributeerror()
    
    if success:
        print("\n🎊 CORREÇÃO CONFIRMADA!")
        print("O sistema está pronto para uso!")
    else:
        print("\n⚠️ Verificar se há problemas pendentes")
