"""
Monitor Web - Dashboard em tempo real para o QuantumTrader ML
Interface web moderna com gráficos interativos
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import json
import time
from datetime import datetime
from pathlib import Path
import threading
from collections import deque
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = 'quantum-trader-ml-2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Dados globais
monitor_data = {
    'status': 'Desconectado',
    'price': 0,
    'position': 0,
    'daily_pnl': 0,
    'total_pnl': 0,
    'trades': 0,
    'win_rate': 0,
    'predictions': deque(maxlen=100),
    'trades_history': deque(maxlen=50),
    'logs': deque(maxlen=200)
}

# Thread de monitoramento
monitoring = False
monitor_thread = None

@app.route('/')
def index():
    return render_template('monitor.html')

@app.route('/api/status')
def get_status():
    return jsonify({
        'status': monitor_data['status'],
        'price': monitor_data['price'],
        'position': monitor_data['position'],
        'daily_pnl': monitor_data['daily_pnl'],
        'total_pnl': monitor_data['total_pnl'],
        'trades': monitor_data['trades'],
        'win_rate': monitor_data['win_rate']
    })

@app.route('/api/predictions')
def get_predictions():
    return jsonify(list(monitor_data['predictions']))

@app.route('/api/trades')
def get_trades():
    return jsonify(list(monitor_data['trades_history']))

@socketio.on('connect')
def handle_connect():
    emit('connected', {'data': 'Connected to QuantumTrader ML Monitor'})
    # Enviar dados iniciais
    emit('update_status', get_status().json)

@socketio.on('start_monitoring')
def handle_start_monitoring():
    global monitoring, monitor_thread
    if not monitoring:
        monitoring = True
        monitor_thread = threading.Thread(target=monitor_logs, daemon=True)
        monitor_thread.start()
        emit('monitoring_started', {'status': 'Monitoring started'})

def monitor_logs():
    """Monitora os logs em tempo real"""
    global monitoring
    
    log_dir = Path('logs/production')
    last_position = 0
    
    while monitoring:
        try:
            # Encontrar arquivo de log mais recente
            log_files = list(log_dir.glob('final_*.log'))
            if not log_files:
                time.sleep(1)
                continue
                
            latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                # Ir para última posição lida
                f.seek(last_position)
                
                new_lines = f.readlines()
                if new_lines:
                    for line in new_lines:
                        process_log_line(line)
                    last_position = f.tell()
                    
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Erro ao monitorar logs: {e}")
            time.sleep(1)

def process_log_line(line):
    """Processa linha de log e extrai informações"""
    global monitor_data
    
    try:
        # Adicionar ao log
        monitor_data['logs'].append({
            'time': datetime.now().isoformat(),
            'text': line.strip()
        })
        
        # Extrair informações
        if '[STATUS]' in line:
            # Atualizar métricas
            if 'Price:' in line:
                price = float(re.search(r'Price:\s*([\d.]+)', line).group(1))
                monitor_data['price'] = price
                
            if 'Pos:' in line:
                pos = int(re.search(r'Pos:\s*(-?\d+)', line).group(1))
                monitor_data['position'] = pos
                
            if 'Trades:' in line:
                trades = int(re.search(r'Trades:\s*(\d+)', line).group(1))
                monitor_data['trades'] = trades
                
            if 'Win Rate:' in line:
                win_rate = float(re.search(r'Win Rate:\s*([\d.]+)%', line).group(1))
                monitor_data['win_rate'] = win_rate
                
            if 'P&L:' in line:
                pnl = float(re.search(r'P&L:\s*R\$\s*([-\d.]+)', line).group(1))
                monitor_data['total_pnl'] = pnl
                
            # Emitir atualização
            socketio.emit('update_status', {
                'price': monitor_data['price'],
                'position': monitor_data['position'],
                'trades': monitor_data['trades'],
                'win_rate': monitor_data['win_rate'],
                'total_pnl': monitor_data['total_pnl']
            })
            
        elif '[ML]' in line and 'Dir:' in line:
            # Extrair predição
            match = re.search(r'Dir:\s*([\d.]+).*Conf:\s*([\d.]+).*Models:\s*(\d+)', line)
            if match:
                prediction = {
                    'time': datetime.now().isoformat(),
                    'direction': float(match.group(1)),
                    'confidence': float(match.group(2)),
                    'models': int(match.group(3))
                }
                monitor_data['predictions'].append(prediction)
                
                # Emitir predição
                socketio.emit('new_prediction', prediction)
                
        elif '[ORDER]' in line:
            # Extrair ordem
            side = 'BUY' if 'COMPRA' in line else 'SELL'
            match = re.search(r'@\s*([\d.]+)', line)
            if match:
                price = float(match.group(1))
                trade = {
                    'time': datetime.now().isoformat(),
                    'side': side,
                    'price': price,
                    'pnl': 0
                }
                monitor_data['trades_history'].append(trade)
                
                # Emitir trade
                socketio.emit('new_trade', trade)
                
        elif '[P&L]' in line and 'Diário:' in line:
            # Extrair P&L diário
            match = re.search(r'Diário:\s*R\$\s*([-\d.]+)', line)
            if match:
                daily_pnl = float(match.group(1))
                monitor_data['daily_pnl'] = daily_pnl
                socketio.emit('update_daily_pnl', {'daily_pnl': daily_pnl})
                
        elif 'SISTEMA OPERACIONAL' in line:
            monitor_data['status'] = 'Operacional'
            socketio.emit('update_status', {'status': 'Operacional'})
            
        # Emitir log
        socketio.emit('new_log', {
            'time': datetime.now().isoformat(),
            'text': line.strip()
        })
        
    except Exception as e:
        print(f"Erro ao processar linha: {e}")

if __name__ == '__main__':
    # Criar diretório de templates se não existir
    Path('templates').mkdir(exist_ok=True)
    
    # Criar template HTML
    html_template = '''<!DOCTYPE html>
<html>
<head>
    <title>QuantumTrader ML Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: Arial, sans-serif; 
            background: #1a1a1a; 
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { 
            text-align: center; 
            margin-bottom: 30px;
            padding: 20px;
            background: #2d2d2d;
            border-radius: 10px;
        }
        .metrics { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #444;
        }
        .metric-title {
            font-size: 14px;
            color: #888;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
        }
        .positive { color: #4CAF50; }
        .negative { color: #f44336; }
        .neutral { color: #FFC107; }
        .charts {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        .chart-container {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #444;
        }
        .logs {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #444;
            max-height: 400px;
            overflow-y: auto;
        }
        .log-entry {
            padding: 5px;
            border-bottom: 1px solid #444;
            font-family: monospace;
            font-size: 12px;
        }
        .status-online { color: #4CAF50; }
        .status-offline { color: #f44336; }
        #connectionStatus {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 10px 20px;
            background: #2d2d2d;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div id="connectionStatus" class="status-offline">Desconectado</div>
    
    <div class="container">
        <div class="header">
            <h1>QuantumTrader ML - Monitor</h1>
            <p>Dashboard de Trading em Tempo Real</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-title">Status</div>
                <div class="metric-value" id="status">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Preço Atual</div>
                <div class="metric-value" id="price">R$ --</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Posição</div>
                <div class="metric-value" id="position">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">P&L Diário</div>
                <div class="metric-value" id="dailyPnl">R$ 0.00</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Trades</div>
                <div class="metric-value" id="trades">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Win Rate</div>
                <div class="metric-value" id="winRate">0%</div>
            </div>
        </div>
        
        <div class="charts">
            <div class="chart-container">
                <h3>Predições ML</h3>
                <canvas id="predictionsChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>P&L Acumulado</h3>
                <canvas id="pnlChart"></canvas>
            </div>
        </div>
        
        <div class="logs">
            <h3>Logs do Sistema</h3>
            <div id="logsContainer"></div>
        </div>
    </div>
    
    <script>
        const socket = io();
        let predictionsData = [];
        let pnlData = [];
        
        // Charts
        const predChart = new Chart(document.getElementById('predictionsChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Direção',
                    data: [],
                    borderColor: '#4CAF50',
                    tension: 0.1
                }, {
                    label: 'Confiança',
                    data: [],
                    borderColor: '#FFC107',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
        
        const pnlChart = new Chart(document.getElementById('pnlChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'P&L',
                    data: [],
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
        
        // Socket events
        socket.on('connect', () => {
            document.getElementById('connectionStatus').textContent = 'Conectado';
            document.getElementById('connectionStatus').className = 'status-online';
            socket.emit('start_monitoring');
        });
        
        socket.on('disconnect', () => {
            document.getElementById('connectionStatus').textContent = 'Desconectado';
            document.getElementById('connectionStatus').className = 'status-offline';
        });
        
        socket.on('update_status', (data) => {
            if (data.status) document.getElementById('status').textContent = data.status;
            if (data.price) document.getElementById('price').textContent = `R$ ${data.price.toFixed(2)}`;
            if (data.position !== undefined) {
                document.getElementById('position').textContent = data.position;
                document.getElementById('position').className = data.position === 0 ? 'neutral' : 'positive';
            }
            if (data.trades !== undefined) document.getElementById('trades').textContent = data.trades;
            if (data.win_rate !== undefined) document.getElementById('winRate').textContent = `${data.win_rate.toFixed(1)}%`;
            if (data.total_pnl !== undefined) {
                pnlData.push(data.total_pnl);
                updatePnlChart();
            }
        });
        
        socket.on('update_daily_pnl', (data) => {
            const elem = document.getElementById('dailyPnl');
            elem.textContent = `R$ ${data.daily_pnl.toFixed(2)}`;
            elem.className = data.daily_pnl >= 0 ? 'positive' : 'negative';
        });
        
        socket.on('new_prediction', (data) => {
            predictionsData.push(data);
            if (predictionsData.length > 50) predictionsData.shift();
            updatePredictionsChart();
        });
        
        socket.on('new_log', (data) => {
            const logsContainer = document.getElementById('logsContainer');
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            logEntry.textContent = `[${new Date(data.time).toLocaleTimeString()}] ${data.text}`;
            logsContainer.insertBefore(logEntry, logsContainer.firstChild);
            
            // Limitar logs
            while (logsContainer.children.length > 100) {
                logsContainer.removeChild(logsContainer.lastChild);
            }
        });
        
        function updatePredictionsChart() {
            const labels = predictionsData.map((_, i) => i);
            const directions = predictionsData.map(p => p.direction);
            const confidences = predictionsData.map(p => p.confidence);
            
            predChart.data.labels = labels;
            predChart.data.datasets[0].data = directions;
            predChart.data.datasets[1].data = confidences;
            predChart.update();
        }
        
        function updatePnlChart() {
            const labels = pnlData.map((_, i) => i);
            pnlChart.data.labels = labels;
            pnlChart.data.datasets[0].data = pnlData;
            pnlChart.update();
        }
    </script>
</body>
</html>'''
    
    # Salvar template
    with open('templates/monitor.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("\n" + "="*60)
    print("QUANTUM TRADER ML - MONITOR WEB")
    print("="*60)
    print("Acesse: http://localhost:5000")
    print("="*60 + "\n")
    
    socketio.run(app, debug=False, port=5000)