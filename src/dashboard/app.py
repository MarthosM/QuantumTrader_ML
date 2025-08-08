"""
Dashboard Real-time para monitoramento do sistema de trading
Usa Flask + SocketIO para atualizações em tempo real
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json
import threading
import time
from datetime import datetime, timedelta
from collections import deque
import logging
import sys
import os

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Imports dos componentes do sistema
from src.portfolio.position_tracker import PositionTracker
from src.risk.risk_manager import RiskManager
from src.execution.order_manager import OrderManager
from src.data.data_synchronizer import DataSynchronizer

# Configuração do Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardServer:
    """
    Servidor do dashboard com integração aos componentes
    """
    
    def __init__(self):
        self.position_tracker = None
        self.risk_manager = None
        self.order_manager = None
        self.data_synchronizer = None
        
        # Dados em memória para o dashboard
        self.price_history = deque(maxlen=500)
        self.equity_history = deque(maxlen=500)
        self.trade_history = deque(maxlen=100)
        self.active_orders = {}
        self.system_status = {
            'is_running': False,
            'connected': False,
            'last_update': None
        }
        
        # Threading
        self.update_thread = None
        self.is_running = False
        
    def initialize(self, position_tracker, risk_manager, order_manager, data_synchronizer):
        """Inicializa com os componentes do sistema"""
        self.position_tracker = position_tracker
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self.data_synchronizer = data_synchronizer
        
        # Registrar callbacks
        self._register_callbacks()
        
        # Iniciar thread de atualização
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info("Dashboard server initialized")
        
    def _register_callbacks(self):
        """Registra callbacks nos componentes"""
        if self.position_tracker:
            self.position_tracker.register_callback('on_position_opened', self._on_position_opened)
            self.position_tracker.register_callback('on_position_closed', self._on_position_closed)
            self.position_tracker.register_callback('on_pnl_update', self._on_pnl_update)
            
        if self.risk_manager:
            self.risk_manager.register_callback('on_risk_alert', self._on_risk_alert)
            self.risk_manager.register_callback('on_stop_loss', self._on_stop_loss)
            self.risk_manager.register_callback('on_circuit_break', self._on_circuit_break)
            
        if self.order_manager:
            self.order_manager.register_callback('on_submitted', self._on_order_submitted)
            self.order_manager.register_callback('on_filled', self._on_order_filled)
            self.order_manager.register_callback('on_cancelled', self._on_order_cancelled)
            
    def _update_loop(self):
        """Loop de atualização periódica"""
        while self.is_running:
            try:
                # Emitir dados atualizados
                self._emit_portfolio_update()
                self._emit_positions_update()
                self._emit_orders_update()
                self._emit_risk_metrics()
                self._emit_system_status()
                
                time.sleep(1)  # Atualizar a cada segundo
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(5)
                
    def _emit_portfolio_update(self):
        """Emite atualização do portfolio"""
        if self.position_tracker:
            metrics = self.position_tracker.get_metrics()
            
            portfolio_data = {
                'total_value': metrics.total_value,
                'cash_balance': metrics.cash_balance,
                'positions_value': metrics.positions_value,
                'total_pnl': metrics.total_pnl,
                'daily_pnl': metrics.daily_pnl,
                'unrealized_pnl': metrics.unrealized_pnl,
                'timestamp': datetime.now().isoformat()
            }
            
            # Adicionar à história
            self.equity_history.append({
                'timestamp': datetime.now().isoformat(),
                'value': metrics.total_value
            })
            
            socketio.emit('portfolio_update', portfolio_data)
            
    def _emit_positions_update(self):
        """Emite atualização das posições"""
        if self.position_tracker:
            positions = self.position_tracker.get_all_positions()
            
            positions_data = []
            for symbol, position in positions.items():
                positions_data.append({
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'entry_time': position.entry_time.isoformat()
                })
                
            socketio.emit('positions_update', positions_data)
            
    def _emit_orders_update(self):
        """Emite atualização das ordens"""
        if self.order_manager:
            orders = self.order_manager.get_open_orders()
            
            orders_data = []
            for order in orders:
                orders_data.append({
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'quantity': order.quantity,
                    'order_type': order.order_type.value,
                    'state': order.state.value,
                    'created_at': order.created_at.isoformat()
                })
                
            socketio.emit('orders_update', orders_data)
            
    def _emit_risk_metrics(self):
        """Emite métricas de risco"""
        if self.risk_manager:
            metrics = self.risk_manager.get_current_metrics()
            
            risk_data = {
                'risk_level': metrics.risk_level.value,
                'current_exposure': metrics.current_exposure,
                'max_exposure': metrics.max_exposure,
                'current_drawdown': metrics.current_drawdown,
                'max_drawdown': metrics.max_drawdown,
                'daily_loss': metrics.daily_loss,
                'open_positions': metrics.open_positions,
                'is_locked': self.risk_manager.is_locked
            }
            
            socketio.emit('risk_update', risk_data)
            
    def _emit_system_status(self):
        """Emite status do sistema"""
        self.system_status['last_update'] = datetime.now().isoformat()
        
        if self.data_synchronizer:
            stats = self.data_synchronizer.get_statistics()
            self.system_status['sync_stats'] = {
                'total_synced': stats.get('total_synced', 0),
                'sync_rate': stats.get('sync_rate', 0),
                'last_sync': stats.get('last_sync_time', '').isoformat() if stats.get('last_sync_time') else None
            }
            
        socketio.emit('system_status', self.system_status)
        
    # Callbacks dos componentes
    def _on_position_opened(self, symbol, position):
        """Callback quando posição é aberta"""
        event_data = {
            'type': 'position_opened',
            'symbol': symbol,
            'quantity': position.quantity,
            'price': position.entry_price,
            'timestamp': datetime.now().isoformat()
        }
        socketio.emit('trade_event', event_data)
        
    def _on_position_closed(self, symbol, pnl):
        """Callback quando posição é fechada"""
        event_data = {
            'type': 'position_closed',
            'symbol': symbol,
            'pnl': pnl,
            'timestamp': datetime.now().isoformat()
        }
        socketio.emit('trade_event', event_data)
        self.trade_history.append(event_data)
        
    def _on_pnl_update(self, metrics):
        """Callback de atualização de P&L"""
        # Já emitido no loop regular
        pass
        
    def _on_risk_alert(self, alert_type, details):
        """Callback de alerta de risco"""
        alert_data = {
            'type': 'risk_alert',
            'alert_type': alert_type,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        socketio.emit('alert', alert_data)
        
    def _on_stop_loss(self, symbol, position):
        """Callback de stop loss"""
        alert_data = {
            'type': 'stop_loss',
            'symbol': symbol,
            'loss': position.get('pnl', 0),
            'timestamp': datetime.now().isoformat()
        }
        socketio.emit('alert', alert_data)
        
    def _on_circuit_break(self, reason):
        """Callback de circuit breaker"""
        alert_data = {
            'type': 'circuit_breaker',
            'reason': reason,
            'severity': 'critical',
            'timestamp': datetime.now().isoformat()
        }
        socketio.emit('alert', alert_data)
        
    def _on_order_submitted(self, order):
        """Callback de ordem submetida"""
        event_data = {
            'type': 'order_submitted',
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'timestamp': datetime.now().isoformat()
        }
        socketio.emit('order_event', event_data)
        
    def _on_order_filled(self, order):
        """Callback de ordem executada"""
        event_data = {
            'type': 'order_filled',
            'order_id': order.order_id,
            'symbol': order.symbol,
            'filled_price': order.filled_price,
            'timestamp': datetime.now().isoformat()
        }
        socketio.emit('order_event', event_data)
        
    def _on_order_cancelled(self, order):
        """Callback de ordem cancelada"""
        event_data = {
            'type': 'order_cancelled',
            'order_id': order.order_id,
            'reason': order.error_message,
            'timestamp': datetime.now().isoformat()
        }
        socketio.emit('order_event', event_data)

# Instância global do servidor
dashboard_server = DashboardServer()

# Rotas do Flask
@app.route('/')
def index():
    """Página principal do dashboard"""
    return render_template('dashboard.html')

@app.route('/api/portfolio')
def get_portfolio():
    """API REST para obter dados do portfolio"""
    if dashboard_server.position_tracker:
        metrics = dashboard_server.position_tracker.get_metrics()
        return jsonify({
            'total_value': metrics.total_value,
            'cash_balance': metrics.cash_balance,
            'total_pnl': metrics.total_pnl,
            'daily_pnl': metrics.daily_pnl,
            'win_rate': metrics.win_rate,
            'sharpe_ratio': metrics.sharpe_ratio
        })
    return jsonify({'error': 'System not initialized'}), 503

@app.route('/api/positions')
def get_positions():
    """API REST para obter posições"""
    if dashboard_server.position_tracker:
        positions = dashboard_server.position_tracker.get_all_positions()
        positions_list = []
        for symbol, pos in positions.items():
            positions_list.append({
                'symbol': symbol,
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl
            })
        return jsonify(positions_list)
    return jsonify([])

@app.route('/api/trades')
def get_trades():
    """API REST para obter histórico de trades"""
    if dashboard_server.position_tracker:
        trades = dashboard_server.position_tracker.get_trade_history(days=7)
        trades_list = []
        for trade in trades:
            trades_list.append({
                'symbol': trade.symbol,
                'entry_time': trade.entry_time.isoformat(),
                'exit_time': trade.exit_time.isoformat(),
                'side': trade.side,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct
            })
        return jsonify(trades_list)
    return jsonify([])

@app.route('/api/risk_metrics')
def get_risk_metrics():
    """API REST para obter métricas de risco"""
    if dashboard_server.risk_manager:
        metrics = dashboard_server.risk_manager.get_current_metrics()
        stats = dashboard_server.risk_manager.get_statistics()
        return jsonify({
            'risk_level': metrics.risk_level.value,
            'current_exposure': metrics.current_exposure,
            'current_drawdown': metrics.current_drawdown,
            'daily_loss': metrics.daily_loss,
            'is_locked': dashboard_server.risk_manager.is_locked,
            'statistics': stats
        })
    return jsonify({'error': 'System not initialized'}), 503

@app.route('/api/orders')
def get_orders():
    """API REST para obter ordens"""
    if dashboard_server.order_manager:
        orders = dashboard_server.order_manager.get_open_orders()
        stats = dashboard_server.order_manager.get_statistics()
        return jsonify({
            'open_orders': [order.to_dict() for order in orders],
            'statistics': stats
        })
    return jsonify({'open_orders': [], 'statistics': {}})

@app.route('/api/equity_curve')
def get_equity_curve():
    """API REST para obter curva de equity"""
    return jsonify(list(dashboard_server.equity_history))

@app.route('/api/system/status')
def get_system_status():
    """API REST para status do sistema"""
    return jsonify(dashboard_server.system_status)

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Cliente conectou"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'data': 'Connected to dashboard'})

@socketio.on('disconnect')
def handle_disconnect():
    """Cliente desconectou"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('request_update')
def handle_request_update(data):
    """Cliente solicitou atualização"""
    update_type = data.get('type', 'all')
    
    if update_type == 'portfolio' or update_type == 'all':
        dashboard_server._emit_portfolio_update()
    if update_type == 'positions' or update_type == 'all':
        dashboard_server._emit_positions_update()
    if update_type == 'orders' or update_type == 'all':
        dashboard_server._emit_orders_update()
    if update_type == 'risk' or update_type == 'all':
        dashboard_server._emit_risk_metrics()

def start_dashboard(host='0.0.0.0', port=5000, debug=False):
    """Inicia o servidor do dashboard"""
    logger.info(f"Starting dashboard server on {host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug)

if __name__ == '__main__':
    # Para testes locais
    start_dashboard(debug=True)