# src/integration/training_integration.py
"""
Integração do sistema de treinamento com o sistema de trading existente
"""

from typing import Dict, List, Optional
import logging
from pathlib import Path

class TrainingIntegration:
    """Integra sistema de treinamento com sistema de trading"""
    
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.logger = logging.getLogger(__name__)
        
    def update_models_from_training(self, training_results: Dict) -> bool:
        """
        Atualiza modelos do sistema de trading com resultados do treinamento
        
        Args:
            training_results: Resultados do TrainingOrchestrator
            
        Returns:
            bool: True se atualização bem sucedida
        """
        try:
            # Extrair caminho do ensemble
            ensemble_path = training_results['save_paths']['ensemble_path']
            
            # Atualizar ModelManager
            model_manager = self.trading_system.model_manager
            
            # Fazer backup dos modelos atuais
            model_manager.backup_current_models()
            
            # Carregar novos modelos
            success = model_manager.load_models_from_path(ensemble_path)
            
            if success:
                # Atualizar lista de features
                feature_names = training_results['selected_features']
                model_manager.update_required_features(feature_names)
                
                # Reinicializar FeatureEngine com novas features
                self.trading_system.feature_engine.sync_with_model(feature_names)
                
                self.logger.info("Modelos atualizados com sucesso do treinamento")
                return True
            else:
                # Restaurar backup se falhar
                model_manager.restore_from_backup()
                self.logger.error("Falha ao carregar novos modelos")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro atualizando modelos: {e}")
            return False
    
    def schedule_periodic_retraining(self, interval_days: int = 30):
        """Agenda retreinamento periódico do sistema"""
        # Implementar agendamento de retreinamento
        pass
    
    def validate_production_readiness(self, training_results: Dict) -> Dict:
        """Valida se modelos treinados estão prontos para produção"""
        validation = {
            'ready': True,
            'checks': {}
        }
        
        # Verificar métricas mínimas
        metrics = training_results['aggregated_metrics']
        
        # Check 1: Accuracy
        min_accuracy = 0.55
        actual_accuracy = metrics.get('accuracy_mean', 0)
        validation['checks']['accuracy'] = {
            'passed': actual_accuracy >= min_accuracy,
            'value': actual_accuracy,
            'threshold': min_accuracy
        }
        
        # Check 2: Número de trades
        min_trades = 100
        n_trades = sum(r['validation_metrics'].get('n_trades', 0) 
                      for r in training_results['validation_results'])
        validation['checks']['n_trades'] = {
            'passed': n_trades >= min_trades,
            'value': n_trades,
            'threshold': min_trades
        }
        
        # Check 3: Estabilidade entre folds
        accuracy_std = metrics.get('accuracy_std', 1.0)
        max_std = 0.05
        validation['checks']['stability'] = {
            'passed': accuracy_std <= max_std,
            'value': accuracy_std,
            'threshold': max_std
        }
        
        # Resultado final
        validation['ready'] = all(
            check['passed'] for check in validation['checks'].values()
        )
        
        return validation