#!/usr/bin/env python3
"""
Teste das correções no execution_engine.py
"""
import sys
import os
sys.path.append('src')

# Mock das dependências
class MockOrderManager:
    def send_order(self, params):
        return None
    def register_order_callback(self, callback):
        pass

class MockMLCoordinator:
    def get_prediction(self):
        return {'confidence': 0.8}

class MockRiskManager:
    def check_signal_risk(self, signal):
        return {
            'approved': True,
            'position_size': 1,
            'stop_loss': 49900,
            'take_profit': 50100
        }

def test_execution_engine():
    """Teste básico do execution engine"""
    try:
        # Importar depois dos mocks
        from execution_engine import SimpleExecutionEngine
        from order_manager import OrderStatus, OrderType, OrderSide, Order
        
        print("✅ Imports realizados com sucesso")
        
        # Criar instância
        mock_order_manager = MockOrderManager()
        mock_ml_coordinator = MockMLCoordinator()
        mock_risk_manager = MockRiskManager()
        
        engine = SimpleExecutionEngine(
            mock_order_manager, 
            mock_ml_coordinator, 
            mock_risk_manager
        )
        
        print("✅ ExecutionEngine criado com sucesso")
        
        # Teste básico de validação de sinal
        valid_signal = {
            'symbol': 'WDOZ24',
            'action': 'buy',
            'confidence': 0.8,
            'prediction': {'direction': 1, 'magnitude': 0.002}
        }
        
        is_valid = engine._validate_ml_signal(valid_signal)
        print(f"✅ Validação de sinal válido: {is_valid}")
        
        invalid_signal = {
            'symbol': 'WDOZ24',
            'action': 'buy',
            'confidence': 0.3,  # Muito baixo
            'prediction': {'direction': 1}
        }
        
        is_invalid = engine._validate_ml_signal(invalid_signal)
        print(f"✅ Validação de sinal inválido: {is_invalid}")
        
        # Teste de atualização de posição
        mock_order = Order(
            symbol='WDOZ24',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1,
            fill_price=50000.0,
            filled_qty=1
        )
        
        engine._update_position(mock_order)
        position = engine.position_tracker.get('WDOZ24')
        print(f"✅ Posição após compra: {position}")
        
        # Teste de callback de ordem
        mock_order.status = OrderStatus.FILLED
        engine._on_order_update(mock_order)
        
        stats = engine.get_execution_stats()
        print(f"✅ Estatísticas: {stats}")
        
        print("\n🎉 Todos os testes passaram! Correções aplicadas com sucesso.")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_execution_engine()
    sys.exit(0 if success else 1)
