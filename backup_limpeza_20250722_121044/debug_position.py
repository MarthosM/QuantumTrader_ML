#!/usr/bin/env python3
"""
Debug específico do fechamento de posição
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def debug_position_close():
    """Debug do fechamento de posição"""
    from execution_engine import SimpleExecutionEngine
    from order_manager import Order, OrderSide, OrderType
    from unittest.mock import Mock
    
    # Criar engine
    engine = SimpleExecutionEngine(Mock(), Mock(), Mock())
    
    # Simular posição short 4
    engine.position_tracker['WDOZ24'] = {
        'quantity': -4,
        'avg_price': 50275.0,
        'side': 'short'
    }
    
    print("Posição antes do fechamento:")
    print(f"  {engine.position_tracker['WDOZ24']}")
    
    # Ordem de fechamento: comprar 4 para fechar short 4
    close_order = Order(
        symbol='WDOZ24',
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=4,
        fill_price=50100.0,
        filled_qty=4
    )
    
    print("\\nAntes de _update_position:")
    print(f"  current_qty = {engine.position_tracker['WDOZ24']['quantity']}")
    print(f"  filled_qty = {close_order.filled_qty}")
    print(f"  new_qty seria = {engine.position_tracker['WDOZ24']['quantity'] + close_order.filled_qty}")
    
    engine._update_position(close_order)
    
    print("Posição depois do fechamento:")
    print(f"  {engine.position_tracker['WDOZ24']}")
    
    position = engine.position_tracker['WDOZ24']
    expected = {'quantity': 0, 'avg_price': 0.0, 'side': None}
    print(f"Esperado: {expected}")
    print(f"Obtido:   {position}")
    print(f"Iguais:   {position == expected}")

if __name__ == "__main__":
    debug_position_close()
