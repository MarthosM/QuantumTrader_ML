"""
Dashboard em tempo real para monitorar sistema HMARL com dados reais
Visualiza fluxo de ordens, decisões de agentes e métricas de performance
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
import numpy as np
from typing import Dict, List, Optional

# Importar componentes necessários
from src.infrastructure.valkey_connection import ValkeyConnection


class HMARLRealtimeDashboard:
    """Dashboard para monitoramento em tempo real do sistema HMARL"""
    
    def __init__(self):
        self.valkey = ValkeyConnection()
        
    def get_recent_trades(self, ticker: str, minutes: int = 5) -> pd.DataFrame:
        """Obtém trades recentes do Valkey"""
        try:
            # Calcular timestamp inicial
            end_time = int(time.time() * 1000)
            start_time = end_time - (minutes * 60 * 1000)
            
            # Buscar dados do stream
            trades = self.valkey.xrange(
                f'market_data:{ticker}',
                f'{start_time}-0',
                f'{end_time}-0'
            )
            
            if not trades:
                return pd.DataFrame()
            
            # Converter para DataFrame
            data = []
            for trade_id, fields in trades:
                data.append({
                    'timestamp': pd.to_datetime(fields.get(b'timestamp', b'').decode()),
                    'price': float(fields.get(b'price', 0)),
                    'volume': float(fields.get(b'volume', 0)),
                    'quantity': int(fields.get(b'quantity', 0))
                })
            
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            st.error(f"Erro obtendo trades: {e}")
            return pd.DataFrame()
    
    def get_agent_decisions(self, limit: int = 10) -> List[Dict]:
        """Obtém decisões recentes dos agentes"""
        try:
            decisions = self.valkey.xrevrange(
                'agent_decisions:all',
                '+',
                '-',
                count=limit
            )
            
            result = []
            for decision_id, fields in decisions:
                decision_data = json.loads(fields.get(b'decision', b'{}').decode())
                decision_data['timestamp'] = fields.get(b'timestamp', b'').decode()
                result.append(decision_data)
            
            return result
            
        except Exception as e:
            st.error(f"Erro obtendo decisões: {e}")
            return []
    
    def get_agent_performance(self) -> Dict:
        """Obtém métricas de performance dos agentes"""
        try:
            agents = ['order_flow_specialist', 'footprint_pattern', 'tape_reading', 'liquidity']
            performance = {}
            
            for agent in agents:
                # Buscar última performance registrada
                data = self.valkey.xrevrange(
                    f'agent_performance:{agent}',
                    '+',
                    '-',
                    count=1
                )
                
                if data:
                    _, fields = data[0]
                    performance[agent] = {
                        'accuracy': float(fields.get(b'accuracy', 0)),
                        'signals': int(fields.get(b'signal_count', 0)),
                        'win_rate': float(fields.get(b'win_rate', 0))
                    }
                else:
                    performance[agent] = {
                        'accuracy': 0.0,
                        'signals': 0,
                        'win_rate': 0.0
                    }
            
            return performance
            
        except Exception as e:
            st.error(f"Erro obtendo performance: {e}")
            return {}
    
    def get_flow_metrics(self, ticker: str) -> Dict:
        """Obtém métricas de fluxo de ordens"""
        try:
            # Buscar últimas features de fluxo
            features = self.valkey.xrevrange(
                f'features:{ticker}',
                '+',
                '-',
                count=1
            )
            
            if features:
                _, fields = features[0]
                feature_data = json.loads(fields.get(b'features', b'{}').decode())
                return feature_data
            
            return {}
            
        except Exception as e:
            st.error(f"Erro obtendo métricas de fluxo: {e}")
            return {}
    
    def plot_price_and_volume(self, df: pd.DataFrame) -> go.Figure:
        """Cria gráfico de preço e volume"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=('Preço', 'Volume')
        )
        
        # Gráfico de preço
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['price'],
                mode='lines',
                name='Preço',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Gráfico de volume
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color='gray'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            xaxis_rangeslider_visible=False
        )
        
        fig.update_xaxes(title_text="Tempo", row=2, col=1)
        fig.update_yaxes(title_text="Preço", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def plot_order_flow_imbalance(self, ticker: str, minutes: int = 30) -> go.Figure:
        """Cria gráfico de Order Flow Imbalance"""
        # Simulação - em produção viria do FlowFeatureSystem
        time_points = pd.date_range(end=datetime.now(), periods=minutes, freq='1min')
        ofi_values = np.random.normal(0, 0.3, minutes)
        ofi_cumulative = np.cumsum(ofi_values)
        
        fig = go.Figure()
        
        # OFI instantâneo
        fig.add_trace(go.Bar(
            x=time_points,
            y=ofi_values,
            name='OFI',
            marker_color=['red' if x < 0 else 'green' for x in ofi_values]
        ))
        
        # OFI acumulado
        fig.add_trace(go.Scatter(
            x=time_points,
            y=ofi_cumulative,
            name='OFI Acumulado',
            line=dict(color='purple', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Order Flow Imbalance',
            yaxis=dict(title='OFI'),
            yaxis2=dict(title='OFI Acumulado', overlaying='y', side='right'),
            height=400
        )
        
        return fig
    
    def plot_agent_signals(self, decisions: List[Dict]) -> go.Figure:
        """Visualiza sinais dos agentes"""
        if not decisions:
            return go.Figure()
        
        # Processar decisões
        df_decisions = pd.DataFrame(decisions)
        
        # Contar sinais por agente
        agent_counts = {}
        for decision in decisions:
            agent = decision.get('selected_agent', 'unknown')
            action = decision.get('action', 'hold')
            
            if agent not in agent_counts:
                agent_counts[agent] = {'buy': 0, 'sell': 0, 'hold': 0}
            
            agent_counts[agent][action] += 1
        
        # Criar gráfico
        fig = go.Figure()
        
        agents = list(agent_counts.keys())
        buy_counts = [agent_counts[a]['buy'] for a in agents]
        sell_counts = [agent_counts[a]['sell'] for a in agents]
        hold_counts = [agent_counts[a]['hold'] for a in agents]
        
        fig.add_trace(go.Bar(name='Buy', x=agents, y=buy_counts, marker_color='green'))
        fig.add_trace(go.Bar(name='Sell', x=agents, y=sell_counts, marker_color='red'))
        fig.add_trace(go.Bar(name='Hold', x=agents, y=hold_counts, marker_color='gray'))
        
        fig.update_layout(
            title='Sinais dos Agentes',
            barmode='stack',
            height=400
        )
        
        return fig


def main():
    """Função principal do dashboard"""
    st.set_page_config(
        page_title="HMARL Real-Time Dashboard",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 HMARL Real-Time Dashboard")
    st.markdown("Monitoramento em tempo real do sistema multi-agente com análise de fluxo")
    
    # Inicializar dashboard
    dashboard = HMARLRealtimeDashboard()
    
    # Sidebar com controles
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        ticker = st.text_input("Ticker", value="WDOQ25")
        refresh_rate = st.slider("Taxa de atualização (segundos)", 1, 30, 5)
        lookback_minutes = st.slider("Período de análise (minutos)", 1, 60, 5)
        
        st.markdown("---")
        
        # Status da conexão
        st.header("📡 Status")
        try:
            if dashboard.valkey.ping():
                st.success("✅ Valkey conectado")
            else:
                st.error("❌ Valkey desconectado")
        except:
            st.error("❌ Erro na conexão")
    
    # Container para auto-refresh
    placeholder = st.empty()
    
    # Loop de atualização
    while True:
        with placeholder.container():
            # Layout em colunas
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.header("📈 Dados de Mercado")
                
                # Obter dados recentes
                df_trades = dashboard.get_recent_trades(ticker, lookback_minutes)
                
                if not df_trades.empty:
                    # Gráfico de preço e volume
                    fig_price = dashboard.plot_price_and_volume(df_trades)
                    st.plotly_chart(fig_price, use_container_width=True)
                    
                    # Estatísticas
                    col1_1, col1_2, col1_3, col1_4 = st.columns(4)
                    with col1_1:
                        st.metric("Último Preço", f"{df_trades['price'].iloc[-1]:.2f}")
                    with col1_2:
                        st.metric("Volume Total", f"{df_trades['volume'].sum():.0f}")
                    with col1_3:
                        st.metric("Trades", len(df_trades))
                    with col1_4:
                        price_change = df_trades['price'].iloc[-1] - df_trades['price'].iloc[0]
                        st.metric("Variação", f"{price_change:.2f}")
                else:
                    st.warning("Aguardando dados de mercado...")
            
            with col2:
                st.header("🔄 Análise de Fluxo")
                
                # Order Flow Imbalance
                fig_ofi = dashboard.plot_order_flow_imbalance(ticker, lookback_minutes)
                st.plotly_chart(fig_ofi, use_container_width=True)
                
                # Métricas de fluxo
                flow_metrics = dashboard.get_flow_metrics(ticker)
                if flow_metrics:
                    st.metric("RSI", f"{flow_metrics.get('rsi', 0):.2f}")
                    st.metric("Volume Ratio", f"{flow_metrics.get('volume_ma', 0):.2f}")
            
            with col3:
                st.header("🤖 Decisões dos Agentes")
                
                # Obter decisões recentes
                decisions = dashboard.get_agent_decisions(20)
                
                if decisions:
                    # Gráfico de sinais
                    fig_signals = dashboard.plot_agent_signals(decisions)
                    st.plotly_chart(fig_signals, use_container_width=True)
                    
                    # Última decisão
                    st.subheader("Última Decisão")
                    last_decision = decisions[0]
                    st.json({
                        'Agente': last_decision.get('selected_agent', 'N/A'),
                        'Ação': last_decision.get('action', 'N/A'),
                        'Confiança': f"{last_decision.get('confidence', 0):.2%}",
                        'Timestamp': last_decision.get('timestamp', 'N/A')
                    })
                else:
                    st.info("Aguardando decisões dos agentes...")
            
            # Performance dos Agentes
            st.header("📊 Performance dos Agentes")
            
            agent_performance = dashboard.get_agent_performance()
            if agent_performance:
                perf_df = pd.DataFrame(agent_performance).T
                
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    # Gráfico de accuracy
                    fig_acc = px.bar(
                        x=perf_df.index,
                        y=perf_df['accuracy'],
                        title="Accuracy dos Agentes",
                        labels={'x': 'Agente', 'y': 'Accuracy'}
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)
                
                with col_p2:
                    # Tabela de métricas
                    st.dataframe(perf_df, use_container_width=True)
            
            # Timestamp da última atualização
            st.caption(f"Última atualização: {datetime.now().strftime('%H:%M:%S')}")
        
        # Aguardar próxima atualização
        time.sleep(refresh_rate)


if __name__ == "__main__":
    main()