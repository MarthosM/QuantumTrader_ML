class PerformanceAnalyzer:
    """Análise contínua de performance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trade_history = []
        self.performance_metrics = {}
        self.attribution_analyzer = PerformanceAttribution()
        
    def analyze_trade(self, trade: Dict[str, Any]):
        """Analisa um trade individual"""
        self.trade_history.append(trade)
        
        # Atualizar métricas em tempo real
        self._update_metrics(trade)
        
        # Análise de atribuição
        attribution = self.attribution_analyzer.analyze(trade)
        trade['attribution'] = attribution
        
        return {
            'trade_analysis': self._analyze_single_trade(trade),
            'impact_on_metrics': self._calculate_metric_impact(trade),
            'attribution': attribution
        }
        
    def _update_metrics(self, trade: Dict[str, Any]):
        """Atualiza métricas de performance"""
        # Calcular P&L
        if trade['status'] == 'closed':
            pnl = trade['exit_price'] - trade['entry_price']
            pnl_percent = pnl / trade['entry_price']
            
            # Atualizar estatísticas
            if 'daily_pnl' not in self.performance_metrics:
                self.performance_metrics['daily_pnl'] = []
                
            self.performance_metrics['daily_pnl'].append({
                'timestamp': trade['exit_time'],
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'symbol': trade['symbol']
            })
            
        # Recalcular métricas agregadas
        self._recalculate_aggregate_metrics()
        
    def _recalculate_aggregate_metrics(self):
        """Recalcula métricas agregadas"""
        if not self.trade_history:
            return
            
        closed_trades = [t for t in self.trade_history if t['status'] == 'closed']
        
        if closed_trades:
            # Win rate
            wins = [t for t in closed_trades if t['pnl'] > 0]
            self.performance_metrics['win_rate'] = len(wins) / len(closed_trades)
            
            # Profit factor
            gross_profit = sum(t['pnl'] for t in wins)
            gross_loss = abs(sum(t['pnl'] for t in closed_trades if t['pnl'] < 0))
            self.performance_metrics['profit_factor'] = (
                gross_profit / gross_loss if gross_loss > 0 else float('inf')
            )
            
            # Sharpe ratio (simplificado)
            returns = [t['pnl_percent'] for t in closed_trades]
            if len(returns) > 1:
                self.performance_metrics['sharpe_ratio'] = (
                    np.mean(returns) / np.std(returns) * np.sqrt(252)
                )
                
            # Maximum drawdown
            self.performance_metrics['max_drawdown'] = self._calculate_max_drawdown()
            
    def get_performance_report(self, period: str = 'today') -> Dict[str, Any]:
        """Gera relatório de performance"""
        # Filtrar trades por período
        filtered_trades = self._filter_trades_by_period(period)
        
        if not filtered_trades:
            return {'message': 'Sem trades no período'}
            
        report = {
            'period': period,
            'summary': {
                'total_trades': len(filtered_trades),
                'win_rate': self.performance_metrics.get('win_rate', 0),
                'profit_factor': self.performance_metrics.get('profit_factor', 0),
                'sharpe_ratio': self.performance_metrics.get('sharpe_ratio', 0),
                'max_drawdown': self.performance_metrics.get('max_drawdown', 0)
            },
            'detailed_metrics': self._calculate_detailed_metrics(filtered_trades),
            'attribution': self.attribution_analyzer.get_summary(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
        
    def _generate_recommendations(self) -> List[str]:
        """Gera recomendações baseadas na performance"""
        recommendations = []
        
        # Verificar win rate
        if self.performance_metrics.get('win_rate', 0) < 0.45:
            recommendations.append(
                "Win rate abaixo do esperado. Considere revisar os critérios de entrada."
            )
            
        # Verificar drawdown
        if self.performance_metrics.get('max_drawdown', 0) > 0.05:
            recommendations.append(
                "Drawdown elevado. Considere reduzir o tamanho das posições."
            )
            
        # Verificar profit factor
        if self.performance_metrics.get('profit_factor', 0) < 1.5:
            recommendations.append(
                "Profit factor baixo. Analise a relação risco/retorno dos trades."
            )
            
        return recommendations