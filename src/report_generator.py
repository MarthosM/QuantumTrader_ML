# src/backtesting/report_generator.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import json
from datetime import datetime

class BacktestReportGenerator:
    """Gerador de relatórios para backtesting"""
    
    def __init__(self):
        self.report_data = {}
        
    def generate_comprehensive_report(self, backtest_results: Dict,
                                    output_path: Optional[str] = None) -> str:
        """Gera relatório HTML completo"""
        
        # Preparar dados
        self.report_data = backtest_results
        
        # Gerar componentes do relatório
        summary_html = self._generate_summary_section()
        metrics_html = self._generate_metrics_section()
        trades_html = self._generate_trades_analysis()
        charts_html = self._generate_charts()
        risk_html = self._generate_risk_analysis()
        
        # Montar relatório completo
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Relatório de Backtest - ML Trading System</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .metric-box {{
            display: inline-block;
            padding: 20px;
            margin: 10px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #2196F3;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        .positive {{
            color: #4CAF50;
        }}
        .negative {{
            color: #f44336;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .chart {{
            margin: 20px 0;
            text-align: center;
        }}
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Relatório de Backtest - Sistema ML Trading</h1>
        <p>Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        {summary_html}
        {metrics_html}
        {charts_html}
        {trades_html}
        {risk_html}
        
        <div class="warning">
            <strong>Aviso:</strong> Resultados passados não garantem performance futura. 
            Este backtest é uma simulação e pode não refletir totalmente as condições reais de mercado.
        </div>
    </div>
</body>
</html>
"""
        
        # Salvar arquivo
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        return html_content
    
    def _generate_summary_section(self) -> str:
        """Gera seção de resumo"""
        config = self.report_data.get('config', {})
        metrics = self.report_data.get('metrics', {})
        
        return f"""
        <h2>Resumo Executivo</h2>
        <div class="summary">
            <p><strong>Período:</strong> {config.get('start_date', 'N/A')} até {config.get('end_date', 'N/A')}</p>
            <p><strong>Capital Inicial:</strong> R$ {config.get('initial_capital', 0):,.2f}</p>
            <p><strong>Capital Final:</strong> R$ {metrics.get('final_equity', 0):,.2f}</p>
            <p><strong>Retorno Total:</strong> <span class="{'positive' if metrics.get('total_return', 0) > 0 else 'negative'}">{metrics.get('total_return', 0)*100:.2f}%</span></p>
            <p><strong>Modo de Backtest:</strong> {config.get('mode', 'N/A')}</p>
        </div>
        """
    
    def _generate_metrics_section(self) -> str:
        """Gera seção de métricas principais"""
        metrics = self.report_data.get('metrics', {})
        
        metric_boxes = []
        
        # Win Rate
        win_rate = metrics.get('win_rate', 0)
        metric_boxes.append(f"""
            <div class="metric-box">
                <div class="metric-value">{win_rate*100:.1f}%</div>
                <div class="metric-label">Taxa de Acerto</div>
            </div>
        """)
        
        # Sharpe Ratio
        sharpe = metrics.get('sharpe_ratio', 0)
        metric_boxes.append(f"""
            <div class="metric-box">
                <div class="metric-value">{sharpe:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
        """)
        
        # Max Drawdown
        max_dd = metrics.get('max_drawdown', 0)
        metric_boxes.append(f"""
            <div class="metric-box">
                <div class="metric-value negative">{max_dd*100:.1f}%</div>
                <div class="metric-label">Drawdown Máximo</div>
            </div>
        """)
        
        # Profit Factor
        profit_factor = metrics.get('profit_factor', 0)
        metric_boxes.append(f"""
            <div class="metric-box">
                <div class="metric-value">{profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
        """)
        
        # Total de Trades
        total_trades = metrics.get('total_trades', 0)
        metric_boxes.append(f"""
            <div class="metric-box">
                <div class="metric-value">{total_trades}</div>
                <div class="metric-label">Total de Trades</div>
            </div>
        """)
        
        # Expectancy
        expectancy = metrics.get('expectancy', 0)
        metric_boxes.append(f"""
            <div class="metric-box">
                <div class="metric-value {'positive' if expectancy > 0 else 'negative'}">R$ {expectancy:.2f}</div>
                <div class="metric-label">Expectativa/Trade</div>
            </div>
        """)
        
        return f"""
        <h2>Métricas de Performance</h2>
        <div class="metrics-grid">
            {''.join(metric_boxes)}
        </div>
        """
    
    def _generate_trades_analysis(self) -> str:
        """Gera análise detalhada dos trades"""
        trade_analysis = self.report_data.get('trade_analysis', {})
        
        # Análise por lado
        by_side = trade_analysis.get('by_side', {})
        
        # Análise por duração
        by_duration = trade_analysis.get('by_duration', {})
        
        # Criar tabela de análise por lado
        side_table = """
        <h3>Análise por Direção</h3>
        <table>
            <tr>
                <th>Direção</th>
                <th>Quantidade</th>
                <th>Taxa de Acerto</th>
                <th>PnL Médio</th>
            </tr>
        """
        
        for side in ['long', 'short']:
            if side in by_side:
                data = by_side[side]
                side_table += f"""
                <tr>
                    <td>{side.capitalize()}</td>
                    <td>{data.get('count', 0)}</td>
                    <td>{data.get('win_rate', 0)*100:.1f}%</td>
                    <td class="{'positive' if data.get('avg_pnl', 0) > 0 else 'negative'}">
                        R$ {data.get('avg_pnl', 0):.2f}
                    </td>
                </tr>
                """
        
        side_table += "</table>"
        
        # Criar tabela de análise por duração
        duration_table = """
        <h3>Análise por Duração</h3>
        <table>
            <tr>
                <th>Duração</th>
                <th>Quantidade</th>
                <th>PnL Médio</th>
                <th>Taxa de Acerto</th>
            </tr>
        """
        
        for duration_range, data in by_duration.items():
            duration_table += f"""
            <tr>
                <td>{duration_range}</td>
                <td>{data.get('count', 0)}</td>
                <td class="{'positive' if data.get('avg_pnl', 0) > 0 else 'negative'}">
                    R$ {data.get('avg_pnl', 0):.2f}
                </td>
                <td>{data.get('win_rate', 0)*100:.1f}%</td>
            </tr>
            """
        
        duration_table += "</table>"
        
        return f"""
        <h2>Análise de Trades</h2>
        {side_table}
        {duration_table}
        """
    
    def _generate_charts(self) -> str:
        """Gera gráficos (placeholder - em produção, gerar imagens reais)"""
        return """
        <h2>Gráficos de Performance</h2>
        <div class="chart">
            <p><em>Curva de Equity</em></p>
            <p>[Gráfico seria inserido aqui]</p>
        </div>
        <div class="chart">
            <p><em>Drawdown ao Longo do Tempo</em></p>
            <p>[Gráfico seria inserido aqui]</p>
        </div>
        """
    
    def _generate_risk_analysis(self) -> str:
        """Gera análise de risco"""
        metrics = self.report_data.get('metrics', {})
        drawdown_analysis = self.report_data.get('drawdown_analysis', {})
        
        return f"""
        <h2>Análise de Risco</h2>
        <div class="risk-metrics">
            <h3>Métricas de Risco</h3>
            <ul>
                <li><strong>Drawdown Máximo:</strong> {metrics.get('max_drawdown', 0)*100:.2f}%</li>
                <li><strong>Sharpe Ratio:</strong> {metrics.get('sharpe_ratio', 0):.2f}</li>
                <li><strong>Sortino Ratio:</strong> {metrics.get('sortino_ratio', 0):.2f}</li>
                <li><strong>Períodos em Drawdown:</strong> {drawdown_analysis.get('total_drawdown_periods', 0)}</li>
                <li><strong>Tempo Médio de Recuperação:</strong> {drawdown_analysis.get('avg_recovery_time', 0):.1f} períodos</li>
            </ul>
            
            <h3>Análise de Sequências</h3>
            <ul>
                <li><strong>Máx. Vitórias Consecutivas:</strong> {self.report_data.get('trade_analysis', {}).get('consecutive_analysis', {}).get('max_consecutive_wins', 0)}</li>
                <li><strong>Máx. Perdas Consecutivas:</strong> {self.report_data.get('trade_analysis', {}).get('consecutive_analysis', {}).get('max_consecutive_losses', 0)}</li>
            </ul>
        </div>
        """