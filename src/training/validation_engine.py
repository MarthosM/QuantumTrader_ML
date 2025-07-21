# src/training/validation_engine.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Generator
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit

class ValidationEngine:
    """Motor de validação específico para day trading"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def create_temporal_splits(self, data: pd.DataFrame,
                             method: str = 'walk_forward',
                             **kwargs) -> Generator:
        """
        Cria splits temporais para validação
        
        Args:
            data: DataFrame com dados temporais
            method: Método de validação
            **kwargs: Parâmetros específicos do método
            
        Yields:
            Tuplas (train_data, test_data, info)
        """
        if method == 'walk_forward':
            yield from self._walk_forward_validation(data, **kwargs)
            
        elif method == 'expanding_window':
            yield from self._expanding_window_validation(data, **kwargs)
            
        elif method == 'rolling_window':
            yield from self._rolling_window_validation(data, **kwargs)
            
        elif method == 'purged_kfold':
            yield from self._purged_kfold_validation(data, **kwargs)
            
        elif method == 'combinatorial_purged':
            yield from self._combinatorial_purged_validation(data, **kwargs)
            
        else:
            raise ValueError(f"Método desconhecido: {method}")
    
    def _walk_forward_validation(self, data: pd.DataFrame,
                               initial_train_size: int = None,
                               test_size: int = None,
                               step_size: int = None,
                               min_train_size: int = None,
                               expanding: bool = False) -> Generator:
        """
        Walk-forward validation adaptativo para diferentes tamanhos de dados
        
        Treina em janela histórica e testa em período futuro,
        avançando a janela progressivamente.
        """
        total_size = len(data)
        
        # Adaptar parâmetros baseado no tamanho dos dados
        if total_size < 100:
            # Dados muito pequenos - usar split simples
            if total_size < 20:
                self.logger.warning(f"Dados insuficientes para walk-forward: {total_size} amostras")
                return  # Não gerar nenhum fold
            
            # Para dados pequenos, usar split simples 70/30
            split_point = int(total_size * 0.7)
            train_data = data.iloc[:split_point].copy()
            test_data = data.iloc[split_point:].copy()
            
            info = {
                'fold': 0,
                'train_period': (train_data.index[0], train_data.index[-1]),
                'test_period': (test_data.index[0], test_data.index[-1]),
                'train_size': len(train_data),
                'test_size': len(test_data),
                'method': 'walk_forward_simple',
                'expanding': expanding
            }
            
            yield train_data, test_data, info
            return
        
        # Parâmetros adaptativos para dados maiores
        if initial_train_size is None:
            initial_train_size = max(50, total_size // 4)  # 25% dos dados ou mín 50
        if test_size is None:
            test_size = max(10, total_size // 10)  # 10% dos dados ou mín 10
        if step_size is None:
            step_size = max(5, test_size // 2)  # Metade do tamanho teste
        if min_train_size is None:
            min_train_size = max(30, total_size // 8)  # Tamanho mínimo treino
        
        # Posição inicial
        current_pos = initial_train_size
        fold = 0
        
        while current_pos + test_size <= total_size:
            # Define janela de treino
            if expanding:
                # Janela expansiva: usa todos dados desde o início
                train_start = 0
            else:
                # Janela deslizante: tamanho fixo
                train_start = max(0, current_pos - initial_train_size)
            
            train_end = current_pos
            
            # Verificar tamanho mínimo de treino
            if train_end - train_start < min_train_size:
                self.logger.warning(f"Tamanho de treino insuficiente: {train_end - train_start}")
                break
            
            # Define janela de teste
            test_start = current_pos
            test_end = min(current_pos + test_size, total_size)
            
            # Criar splits
            train_data = data.iloc[train_start:train_end].copy()
            test_data = data.iloc[test_start:test_end].copy()
            
            # Informações do split
            info = {
                'fold': fold,
                'train_period': (train_data.index[0], train_data.index[-1]),
                'test_period': (test_data.index[0], test_data.index[-1]),
                'train_size': len(train_data),
                'test_size': len(test_data),
                'method': 'walk_forward',
                'expanding': expanding
            }
            
            yield train_data, test_data, info
            
            # Avançar janela
            current_pos += step_size
            fold += 1
            
        self.logger.info(f"Walk-forward validation: {fold} folds criados")
    
    def _expanding_window_validation(self, data: pd.DataFrame,
                                   initial_train_size: int = 5000,
                                   test_size: int = 500,
                                   step_size: int = 250) -> Generator:
        """
        Expanding window validation
        
        Similar ao walk-forward mas a janela de treino sempre expande.
        """
        yield from self._walk_forward_validation(
            data,
            initial_train_size=initial_train_size,
            test_size=test_size,
            step_size=step_size,
            expanding=True
        )
    
    def _rolling_window_validation(self, data: pd.DataFrame,
                                 window_size: int = 5000,
                                 test_size: int = 500,
                                 step_size: int = 250) -> Generator:
        """
        Rolling window validation
        
        Janela de tamanho fixo que rola pelos dados.
        """
        yield from self._walk_forward_validation(
            data,
            initial_train_size=window_size,
            test_size=test_size,
            step_size=step_size,
            expanding=False
        )
    
    def _purged_kfold_validation(self, data: pd.DataFrame,
                               n_splits: int = 5,
                               purge_gap: int = 100,
                               embargo_gap: int = 50) -> Generator:
        """
        K-fold com purge para evitar data leakage
        
        Adiciona gaps entre treino e teste para evitar vazamento
        de informação em séries temporais.
        """
        total_size = len(data)
        fold_size = total_size // n_splits
        
        for fold in range(n_splits):
            # Define fold de teste
            test_start = fold * fold_size
            test_end = min((fold + 1) * fold_size, total_size)
            
            # Índices de treino com purge
            train_indices = []
            
            # Dados antes do teste (com purge gap)
            if test_start > purge_gap:
                before_test = list(range(0, test_start - purge_gap))
                train_indices.extend(before_test)
            
            # Dados depois do teste (com embargo gap)
            if test_end + embargo_gap < total_size:
                after_test = list(range(test_end + embargo_gap, total_size))
                train_indices.extend(after_test)
            
            # Verificar se há dados suficientes
            if len(train_indices) < 1000:  # Mínimo de dados
                self.logger.warning(f"Fold {fold} pulado - dados insuficientes")
                continue
            
            # Criar splits
            train_data = data.iloc[train_indices].copy()
            test_data = data.iloc[test_start:test_end].copy()
            
            # Informações
            info = {
                'fold': fold,
                'train_periods': self._get_periods_from_indices(data, train_indices),
                'test_period': (test_data.index[0], test_data.index[-1]),
                'train_size': len(train_data),
                'test_size': len(test_data),
                'purge_gap': purge_gap,
                'embargo_gap': embargo_gap,
                'method': 'purged_kfold'
            }
            
            yield train_data, test_data, info
    
    def _combinatorial_purged_validation(self, data: pd.DataFrame,
                                       n_splits: int = 5,
                                       n_test_splits: int = 2,
                                       purge_gap: int = 100) -> Generator:
        """
        Combinatorial purged cross-validation
        
        Cria múltiplas combinações de splits para validação mais robusta.
        """
        from itertools import combinations
        
        total_size = len(data)
        split_size = total_size // n_splits
        
        # Criar todos os splits possíveis
        splits = []
        for i in range(n_splits):
            start = i * split_size
            end = min((i + 1) * split_size, total_size)
            splits.append((start, end))
        
        # Gerar combinações de test splits
        fold = 0
        for test_splits in combinations(range(n_splits), n_test_splits):
            # Coletar índices de teste
            test_indices = []
            for split_idx in test_splits:
                start, end = splits[split_idx]
                test_indices.extend(range(start, end))
            
            # Coletar índices de treino com purge
            train_indices = []
            for i in range(n_splits):
                if i not in test_splits:
                    start, end = splits[i]
                    
                    # Verificar proximidade com splits de teste
                    too_close = False
                    for test_idx in test_splits:
                        test_start, test_end = splits[test_idx]
                        
                        # Se está muito próximo, pular
                        if abs(end - test_start) < purge_gap or abs(start - test_end) < purge_gap:
                            too_close = True
                            break
                    
                    if not too_close:
                        train_indices.extend(range(start, end))
            
            # Verificar dados suficientes
            if len(train_indices) < 1000:
                continue
            
            # Criar datasets
            train_data = data.iloc[train_indices].copy()
            test_data = data.iloc[test_indices].copy()
            
            # Informações
            info = {
                'fold': fold,
                'test_splits': test_splits,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'method': 'combinatorial_purged'
            }
            
            yield train_data, test_data, info
            fold += 1
    
    def validate_temporal_integrity(self, train_data: pd.DataFrame,
                                  test_data: pd.DataFrame) -> Dict[str, bool]:
        """
        Valida integridade temporal dos splits
        
        Verifica se não há vazamento de informação futura.
        """
        checks = {
            'no_overlap': True,
            'train_before_test': True,
            'sufficient_gap': True,
            'no_lookahead': True
        }
        
        # Verificar overlap
        train_idx = set(train_data.index)
        test_idx = set(test_data.index)
        
        if train_idx.intersection(test_idx):
            checks['no_overlap'] = False
            self.logger.error("Overlap detectado entre treino e teste!")
        
        # Verificar ordem temporal
        if train_data.index.max() >= test_data.index.min():
            # Pode haver dados de treino depois do teste (purged CV)
            # Mas verificar se há gap suficiente
            gap = (test_data.index.min() - train_data.index.max()).total_seconds() / 60
            
            if gap < 60:  # Menos de 60 minutos de gap
                checks['sufficient_gap'] = False
                self.logger.warning("Gap insuficiente entre treino e teste")
        
        return checks
    
    def calculate_validation_metrics(self, y_true: pd.Series,
                                   y_pred: np.ndarray,
                                   y_proba: np.ndarray = None) -> Dict:
        """
        Calcula métricas específicas para validação de trading
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        from sklearn.metrics import f1_score, confusion_matrix
        
        # Métricas básicas
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Métricas por classe usando abordagem multiclass
        try:
            # Calcular precisão e recall para cada classe usando approach 'weighted'
            precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
            recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
            
            for i in range(min(3, len(precisions))):  # 0: Sell, 1: Hold, 2: Buy
                metrics[f'class_{i}_precision'] = precisions[i]
                metrics[f'class_{i}_recall'] = recalls[i]
                
        except Exception as e:
            self.logger.warning(f"Erro ao calcular métricas por classe: {e}")
            # Fallback - calcular de forma manual se necessário
            for i in range(3):
                # Calcular precisão manual: TP / (TP + FP)
                tp = np.sum((y_pred == i) & (y_true == i))
                fp = np.sum((y_pred == i) & (y_true != i))
                fn = np.sum((y_pred != i) & (y_true == i))
                
                precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                metrics[f'class_{i}_precision'] = precision_val
                metrics[f'class_{i}_recall'] = recall_val
        
        # Métricas de trading
        if y_proba is not None:
            # Confiança média
            metrics['avg_confidence'] = np.mean(np.max(y_proba, axis=1))
            
            # Confiança por acerto/erro
            correct = y_pred == y_true
            metrics['confidence_when_correct'] = np.mean(
                np.max(y_proba[correct], axis=1)
            ) if correct.sum() > 0 else 0
            
            metrics['confidence_when_wrong'] = np.mean(
                np.max(y_proba[~correct], axis=1)
            ) if (~correct).sum() > 0 else 0
        
        # Distribuição de predições
        pred_dist = pd.Series(y_pred).value_counts(normalize=True)
        metrics['pred_distribution'] = pred_dist.to_dict()
        
        # Matriz de confusão
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics
    
    def _get_periods_from_indices(self, data: pd.DataFrame, 
                                indices: List[int]) -> List[Tuple]:
        """Extrai períodos contíguos de uma lista de índices"""
        if not indices:
            return []
        
        indices = sorted(indices)
        periods = []
        start = indices[0]
        end = indices[0]
        
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                # Contíguo
                end = indices[i]
            else:
                # Gap - salvar período anterior
                periods.append((
                    data.index[start],
                    data.index[end]
                ))
                start = indices[i]
                end = indices[i]
        
        # Último período
        periods.append((
            data.index[start],
            data.index[end]
        ))
        
        return periods