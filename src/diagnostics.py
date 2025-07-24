class DiagnosticSuite:
    """Conjunto de ferramentas de diagnóstico"""
    
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.diagnostics = {
            'system': SystemDiagnostics(),
            'data': DataDiagnostics(),
            'model': ModelDiagnostics(),
            'execution': ExecutionDiagnostics()
        }
        
    def run_full_diagnostics(self) -> Dict[str, Any]:
        """Executa diagnóstico completo do sistema"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'running',
            'diagnostics': {}
        }
        
        for name, diagnostic in self.diagnostics.items():
            try:
                result = diagnostic.run(self.trading_system)
                results['diagnostics'][name] = result
            except Exception as e:
                results['diagnostics'][name] = {
                    'status': 'error',
                    'error': str(e)
                }
                
        # Determinar status geral
        all_ok = all(
            d.get('status') == 'ok' 
            for d in results['diagnostics'].values()
        )
        results['status'] = 'healthy' if all_ok else 'issues_detected'
        
        return results
        
    def quick_health_check(self) -> Dict[str, bool]:
        """Verificação rápida de saúde"""
        checks = {
            'connection': self._check_connection(),
            'data_flow': self._check_data_flow(),
            'models_loaded': self._check_models(),
            'risk_limits': self._check_risk_limits(),
            'execution_ready': self._check_execution()
        }
        
        checks['overall'] = all(checks.values())
        return checks
        
    def get_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identifica gargalos no sistema"""
        bottlenecks = []
        
        # Verificar latência de componentes
        latencies = self._measure_component_latencies()
        for component, latency in latencies.items():
            if latency > 50:  # ms
                bottlenecks.append({
                    'component': component,
                    'type': 'latency',
                    'value': latency,
                    'severity': 'high' if latency > 100 else 'medium'
                })
                
        # Verificar uso de recursos
        resources = self._check_resource_usage()
        if resources['memory_percent'] > 80:
            bottlenecks.append({
                'component': 'system',
                'type': 'memory',
                'value': resources['memory_percent'],
                'severity': 'high'
            })
            
        # Verificar filas
        queue_sizes = self._check_queue_sizes()
        for queue_name, size in queue_sizes.items():
            if size > 100:
                bottlenecks.append({
                    'component': queue_name,
                    'type': 'queue_backlog',
                    'value': size,
                    'severity': 'medium'
                })
                
        return bottlenecks