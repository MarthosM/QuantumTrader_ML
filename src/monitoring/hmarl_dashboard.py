"""
Dashboard de Monitoramento HMARL
Dashboard web simples usando Flask para monitorar o sistema HMARL
"""

from flask import Flask, render_template, jsonify
import time
import threading
from datetime import datetime
from collections import deque
import sys
import os

# Adicionar diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coordination.agent_registry import AgentRegistryClient


class DashboardData:
    """Armazena dados para o dashboard"""
    
    def __init__(self):
        self.agents_history = deque(maxlen=100)
        self.signals_history = deque(maxlen=100)
        self.decisions_history = deque(maxlen=50)
        self.performance_history = deque(maxlen=100)
        
        # Métricas em tempo real
        self.current_agents = {}
        self.total_signals = 0
        self.total_decisions = 0
        self.success_rate = 0.0
        
        # Cliente do registry
        self.registry_client = None
        try:
            self.registry_client = AgentRegistryClient()
        except:
            pass
            
    def update_agents_data(self):
        """Atualiza dados dos agentes"""
        if not self.registry_client:
            return
            
        try:
            agents = self.registry_client.get_all_agents()
            
            self.current_agents = {
                agent['agent_id']: {
                    'type': agent['agent_type'],
                    'status': agent['status'],
                    'last_heartbeat': agent['last_heartbeat'],
                    'metrics': agent['performance_metrics']
                }
                for agent in agents
            }
            
            # Adicionar ao histórico
            self.agents_history.append({
                'timestamp': time.time(),
                'active_count': sum(1 for a in agents if a['status'] == 'active'),
                'total_count': len(agents)
            })
            
        except Exception as e:
            print(f"Erro atualizando dados dos agentes: {e}")
            
    def add_signal(self, signal_data):
        """Adiciona sinal ao histórico"""
        self.signals_history.append({
            'timestamp': time.time(),
            'agent_id': signal_data.get('agent_id'),
            'action': signal_data.get('action'),
            'confidence': signal_data.get('confidence')
        })
        self.total_signals += 1
        
    def add_decision(self, decision_data):
        """Adiciona decisão ao histórico"""
        self.decisions_history.append({
            'timestamp': time.time(),
            'action': decision_data.get('action'),
            'confidence': decision_data.get('confidence'),
            'consensus': decision_data.get('flow_consensus', {})
        })
        self.total_decisions += 1
        
    def get_dashboard_data(self):
        """Retorna dados formatados para o dashboard"""
        # Atualizar dados dos agentes
        self.update_agents_data()
        
        # Calcular estatísticas
        active_agents = sum(1 for a in self.current_agents.values() if a['status'] == 'active')
        
        # Taxa de sinais (últimos 60 segundos)
        current_time = time.time()
        recent_signals = sum(
            1 for s in self.signals_history 
            if current_time - s['timestamp'] < 60
        )
        signal_rate = recent_signals / 60.0  # sinais por segundo
        
        return {
            'timestamp': datetime.now().isoformat(),
            'agents': {
                'total': len(self.current_agents),
                'active': active_agents,
                'details': self.current_agents
            },
            'signals': {
                'total': self.total_signals,
                'rate': signal_rate,
                'recent': list(self.signals_history)[-10:]  # últimos 10
            },
            'decisions': {
                'total': self.total_decisions,
                'recent': list(self.decisions_history)[-10:]  # últimas 10
            },
            'performance': {
                'success_rate': self.success_rate,
                'history': list(self.performance_history)
            }
        }


# Criar aplicação Flask
app = Flask(__name__)
dashboard_data = DashboardData()


@app.route('/')
def index():
    """Página principal do dashboard"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>HMARL Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .stat-card h3 {
                margin: 0 0 10px 0;
                color: #666;
                font-size: 14px;
                text-transform: uppercase;
            }
            .stat-value {
                font-size: 32px;
                font-weight: bold;
                color: #2196F3;
            }
            .stat-label {
                color: #999;
                font-size: 12px;
            }
            .section {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .section h2 {
                margin-top: 0;
                color: #333;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                text-align: left;
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f8f8f8;
                font-weight: bold;
            }
            .status-active {
                color: #4CAF50;
                font-weight: bold;
            }
            .status-inactive {
                color: #f44336;
                font-weight: bold;
            }
            .action-buy {
                color: #4CAF50;
            }
            .action-sell {
                color: #f44336;
            }
            .action-hold {
                color: #FF9800;
            }
            .timestamp {
                color: #666;
                font-size: 12px;
            }
            .refresh-info {
                text-align: center;
                color: #666;
                font-size: 12px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 HMARL Dashboard</h1>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Agentes Ativos</h3>
                    <div class="stat-value" id="active-agents">-</div>
                    <div class="stat-label">de <span id="total-agents">-</span> agentes totais</div>
                </div>
                
                <div class="stat-card">
                    <h3>Sinais Gerados</h3>
                    <div class="stat-value" id="total-signals">-</div>
                    <div class="stat-label"><span id="signal-rate">-</span> sinais/seg</div>
                </div>
                
                <div class="stat-card">
                    <h3>Decisões Tomadas</h3>
                    <div class="stat-value" id="total-decisions">-</div>
                    <div class="stat-label">coordenadas pelo sistema</div>
                </div>
                
                <div class="stat-card">
                    <h3>Taxa de Sucesso</h3>
                    <div class="stat-value" id="success-rate">-</div>
                    <div class="stat-label">performance geral</div>
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Agentes</h2>
                <table id="agents-table">
                    <thead>
                        <tr>
                            <th>ID do Agente</th>
                            <th>Tipo</th>
                            <th>Status</th>
                            <th>Sinais</th>
                            <th>Confiança Média</th>
                            <th>Último Heartbeat</th>
                        </tr>
                    </thead>
                    <tbody id="agents-tbody">
                        <!-- Preenchido via JavaScript -->
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>📈 Sinais Recentes</h2>
                <table id="signals-table">
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Agente</th>
                            <th>Ação</th>
                            <th>Confiança</th>
                        </tr>
                    </thead>
                    <tbody id="signals-tbody">
                        <!-- Preenchido via JavaScript -->
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>🎯 Decisões Recentes</h2>
                <table id="decisions-table">
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Ação</th>
                            <th>Confiança</th>
                            <th>Consenso</th>
                        </tr>
                    </thead>
                    <tbody id="decisions-tbody">
                        <!-- Preenchido via JavaScript -->
                    </tbody>
                </table>
            </div>
            
            <div class="refresh-info">
                Atualização automática a cada 2 segundos | Última atualização: <span id="last-update">-</span>
            </div>
        </div>
        
        <script>
            function updateDashboard() {
                fetch('/api/data')
                    .then(response => response.json())
                    .then(data => {
                        // Atualizar estatísticas
                        document.getElementById('active-agents').textContent = data.agents.active;
                        document.getElementById('total-agents').textContent = data.agents.total;
                        document.getElementById('total-signals').textContent = data.signals.total;
                        document.getElementById('signal-rate').textContent = data.signals.rate.toFixed(2);
                        document.getElementById('total-decisions').textContent = data.decisions.total;
                        document.getElementById('success-rate').textContent = (data.performance.success_rate * 100).toFixed(1) + '%';
                        
                        // Atualizar tabela de agentes
                        const agentsTbody = document.getElementById('agents-tbody');
                        agentsTbody.innerHTML = '';
                        
                        for (const [agentId, agent] of Object.entries(data.agents.details)) {
                            const row = agentsTbody.insertRow();
                            row.innerHTML = `
                                <td>${agentId}</td>
                                <td>${agent.type}</td>
                                <td class="status-${agent.status}">${agent.status.toUpperCase()}</td>
                                <td>${agent.metrics.signals_generated || 0}</td>
                                <td>${(agent.metrics.avg_confidence || 0).toFixed(2)}</td>
                                <td class="timestamp">${new Date(agent.last_heartbeat * 1000).toLocaleTimeString()}</td>
                            `;
                        }
                        
                        // Atualizar tabela de sinais
                        const signalsTbody = document.getElementById('signals-tbody');
                        signalsTbody.innerHTML = '';
                        
                        data.signals.recent.reverse().forEach(signal => {
                            const row = signalsTbody.insertRow();
                            row.innerHTML = `
                                <td class="timestamp">${new Date(signal.timestamp * 1000).toLocaleTimeString()}</td>
                                <td>${signal.agent_id || '-'}</td>
                                <td class="action-${signal.action}">${signal.action ? signal.action.toUpperCase() : '-'}</td>
                                <td>${(signal.confidence || 0).toFixed(2)}</td>
                            `;
                        });
                        
                        // Atualizar tabela de decisões
                        const decisionsTbody = document.getElementById('decisions-tbody');
                        decisionsTbody.innerHTML = '';
                        
                        data.decisions.recent.reverse().forEach(decision => {
                            const row = decisionsTbody.insertRow();
                            const consensus = decision.consensus;
                            const consensusText = consensus ? `${consensus.direction} (${(consensus.strength || 0).toFixed(2)})` : '-';
                            
                            row.innerHTML = `
                                <td class="timestamp">${new Date(decision.timestamp * 1000).toLocaleTimeString()}</td>
                                <td class="action-${decision.action}">${decision.action ? decision.action.toUpperCase() : '-'}</td>
                                <td>${(decision.confidence || 0).toFixed(2)}</td>
                                <td>${consensusText}</td>
                            `;
                        });
                        
                        // Atualizar timestamp
                        document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                    })
                    .catch(error => {
                        console.error('Erro atualizando dashboard:', error);
                    });
            }
            
            // Atualizar a cada 2 segundos
            setInterval(updateDashboard, 2000);
            
            // Atualização inicial
            updateDashboard();
        </script>
    </body>
    </html>
    '''


@app.route('/api/data')
def api_data():
    """API endpoint para dados do dashboard"""
    return jsonify(dashboard_data.get_dashboard_data())


def run_dashboard(host='127.0.0.1', port=5000):
    """Executa o dashboard"""
    print(f"\n{'='*60}")
    print(f"HMARL Dashboard iniciado em http://{host}:{port}")
    print(f"{'='*60}\n")
    
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run_dashboard()