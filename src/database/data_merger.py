"""
Data Merger - Combinador Inteligente de Fontes de Dados
======================================================

Este módulo combina dados de múltiplas fontes (ProfitDLL, CSV, APIs)
garantindo consistência e eliminando conflitos.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
from dataclasses import dataclass
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DataSource:
    """Representa uma fonte de dados"""
    name: str
    data: pd.DataFrame
    quality_score: float
    priority: int  # 1 = maior prioridade
    metadata: Dict[str, Any]


@dataclass
class MergeConflict:
    """Representa um conflito durante merge"""
    timestamp: datetime
    field: str
    source1: str
    value1: Any
    source2: str
    value2: Any
    resolution: str
    resolved_value: Any


class DataMerger:
    """Combina dados de múltiplas fontes com resolução inteligente de conflitos"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa merger
        
        Args:
            config: Configurações do merger
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configurações de merge
        self.merge_config = {
            'conflict_resolution': 'quality',  # 'quality', 'priority', 'average', 'newest'
            'tolerance': {
                'price': 0.001,  # 0.1%
                'volume': 0.01,  # 1%
                'time': 1000  # 1 segundo em ms
            },
            'chunk_size': 10000,
            'parallel_workers': 4
        }
        
        # Atualizar com config
        if 'merge_config' in self.config:
            self.merge_config.update(self.config['merge_config'])
        
        # Cache de hashes para detectar duplicatas
        self._hash_cache = {}
        
        # Registro de conflitos
        self.conflicts = []
        
        # Estatísticas
        self.stats = {
            'total_records': 0,
            'merged_records': 0,
            'conflicts_resolved': 0,
            'duplicates_removed': 0,
            'sources_used': []
        }
    
    def merge_sources(self,
                     sources: List[DataSource],
                     start_date: datetime,
                     end_date: datetime,
                     data_type: str = 'trades') -> pd.DataFrame:
        """
        Combina múltiplas fontes de dados
        
        Args:
            sources: Lista de fontes de dados
            start_date: Data inicial
            end_date: Data final
            data_type: Tipo de dados sendo combinado
            
        Returns:
            DataFrame combinado e validado
        """
        self.logger.info(f"Iniciando merge de {len(sources)} fontes para {data_type}")
        
        # Resetar estatísticas
        self.stats = {
            'total_records': sum(len(s.data) for s in sources),
            'merged_records': 0,
            'conflicts_resolved': 0,
            'duplicates_removed': 0,
            'sources_used': [s.name for s in sources]
        }
        
        # Filtrar período
        filtered_sources = []
        for source in sources:
            filtered_data = self._filter_date_range(source.data, start_date, end_date)
            if not filtered_data.empty:
                filtered_sources.append(DataSource(
                    name=source.name,
                    data=filtered_data,
                    quality_score=source.quality_score,
                    priority=source.priority,
                    metadata=source.metadata
                ))
        
        if not filtered_sources:
            self.logger.warning("Nenhum dado no período especificado")
            return pd.DataFrame()
        
        # Ordenar por prioridade
        filtered_sources.sort(key=lambda x: (x.priority, -x.quality_score))
        
        # Estratégia de merge baseada no tipo
        if data_type == 'trades':
            merged = self._merge_trades(filtered_sources)
        elif data_type == 'candles':
            merged = self._merge_candles(filtered_sources)
        elif data_type == 'book':
            merged = self._merge_book(filtered_sources)
        else:
            merged = self._merge_generic(filtered_sources)
        
        # Pós-processamento
        merged = self._post_process(merged, data_type)
        
        self.stats['merged_records'] = len(merged)
        
        # Log estatísticas
        self._log_merge_stats()
        
        return merged
    
    def _merge_trades(self, sources: List[DataSource]) -> pd.DataFrame:
        """Merge específico para trades"""
        if len(sources) == 1:
            return sources[0].data
        
        # Usar estratégia de chunks para eficiência
        chunk_size = self.merge_config['chunk_size']
        
        # Primeira fonte como base
        base_df = sources[0].data.copy()
        base_df['_source'] = sources[0].name
        base_df['_quality'] = sources[0].quality_score
        
        # Adicionar hash para detecção de duplicatas
        base_df['_hash'] = base_df.apply(self._compute_trade_hash, axis=1)
        
        # Merge incremental com outras fontes
        for source in sources[1:]:
            self.logger.info(f"Merging {source.name} ({len(source.data)} trades)")
            
            # Processar em chunks
            for start_idx in range(0, len(source.data), chunk_size):
                end_idx = min(start_idx + chunk_size, len(source.data))
                chunk = source.data.iloc[start_idx:end_idx].copy()
                
                chunk['_source'] = source.name
                chunk['_quality'] = source.quality_score
                chunk['_hash'] = chunk.apply(self._compute_trade_hash, axis=1)
                
                # Identificar novos trades vs conflitos
                base_df = self._merge_trade_chunk(base_df, chunk, source.name)
        
        # Remover colunas auxiliares
        columns_to_drop = ['_source', '_quality', '_hash']
        base_df = base_df.drop(columns=[col for col in columns_to_drop if col in base_df.columns])
        
        return base_df
    
    def _merge_candles(self, sources: List[DataSource]) -> pd.DataFrame:
        """Merge específico para candles"""
        if len(sources) == 1:
            return sources[0].data
        
        # Agrupar por timestamp
        all_candles = pd.DataFrame()
        
        for source in sources:
            df = source.data.copy()
            df['_source'] = source.name
            df['_quality'] = source.quality_score
            df['_priority'] = source.priority
            all_candles = pd.concat([all_candles, df])
        
        # Resolver conflitos por timestamp
        merged = all_candles.groupby('datetime').apply(
            lambda x: self._resolve_candle_conflict(x) if len(x) > 1 else x.iloc[0]
        ).reset_index(drop=True)
        
        # Remover colunas auxiliares
        columns_to_drop = ['_source', '_quality', '_priority']
        merged = merged.drop(columns=[col for col in columns_to_drop if col in merged.columns])
        
        return merged
    
    def _merge_book(self, sources: List[DataSource]) -> pd.DataFrame:
        """Merge específico para order book"""
        # Similar ao merge de trades mas com lógica específica para bid/ask
        return self._merge_generic(sources)
    
    def _merge_generic(self, sources: List[DataSource]) -> pd.DataFrame:
        """Merge genérico para outros tipos"""
        # Concatenar e remover duplicatas
        all_data = pd.DataFrame()
        
        for source in sources:
            df = source.data.copy()
            df['_source'] = source.name
            all_data = pd.concat([all_data, df])
        
        # Remover duplicatas mantendo fonte de maior qualidade
        if 'datetime' in all_data.columns:
            all_data = all_data.sort_values(['datetime', '_source'])
            all_data = all_data.drop_duplicates(subset=['datetime'], keep='first')
        
        # Remover coluna auxiliar
        if '_source' in all_data.columns:
            all_data = all_data.drop(columns=['_source'])
        
        return all_data
    
    def _merge_trade_chunk(self, base_df: pd.DataFrame, chunk: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Merge um chunk de trades com resolução de conflitos"""
        # Identificar trades únicos vs duplicados
        chunk_hashes = set(chunk['_hash'])
        base_hashes = set(base_df['_hash'])
        
        # Trades completamente novos
        new_hashes = chunk_hashes - base_hashes
        new_trades = chunk[chunk['_hash'].isin(new_hashes)]
        
        # Potenciais conflitos (mesmo timestamp, dados diferentes)
        conflict_mask = chunk['datetime'].isin(base_df['datetime'])
        potential_conflicts = chunk[conflict_mask & ~chunk['_hash'].isin(base_hashes)]
        
        # Adicionar novos trades
        if not new_trades.empty:
            base_df = pd.concat([base_df, new_trades], ignore_index=True)
            self.stats['duplicates_removed'] += len(chunk) - len(new_trades) - len(potential_conflicts)
        
        # Resolver conflitos
        if not potential_conflicts.empty:
            for _, trade in potential_conflicts.iterrows():
                base_df = self._resolve_trade_conflict(base_df, trade, source_name)
        
        return base_df
    
    def _resolve_trade_conflict(self, base_df: pd.DataFrame, new_trade: pd.Series, source_name: str) -> pd.DataFrame:
        """Resolve conflito entre trades"""
        # Encontrar trade conflitante
        mask = base_df['datetime'] == new_trade['datetime']
        existing_trades = base_df[mask]
        
        if existing_trades.empty:
            # Sem conflito real, adicionar
            base_df = pd.concat([base_df, new_trade.to_frame().T], ignore_index=True)
            return base_df
        
        existing_trade = existing_trades.iloc[0]
        
        # Verificar se é realmente um conflito
        price_diff = abs(existing_trade['price'] - new_trade['price']) / existing_trade['price']
        volume_diff = abs(existing_trade['volume'] - new_trade['volume']) / existing_trade['volume']
        
        if price_diff <= self.merge_config['tolerance']['price'] and \
           volume_diff <= self.merge_config['tolerance']['volume']:
            # Diferença dentro da tolerância, manter existente
            return base_df
        
        # Resolver conflito baseado na estratégia
        resolution_method = self.merge_config['conflict_resolution']
        
        if resolution_method == 'quality':
            # Manter fonte com maior qualidade
            if new_trade['_quality'] > existing_trade['_quality']:
                base_df.loc[mask] = new_trade
                resolved_value = new_trade['price']
            else:
                resolved_value = existing_trade['price']
        
        elif resolution_method == 'average':
            # Média dos valores
            base_df.loc[mask, 'price'] = (existing_trade['price'] + new_trade['price']) / 2
            base_df.loc[mask, 'volume'] = (existing_trade['volume'] + new_trade['volume']) / 2
            resolved_value = base_df.loc[mask, 'price'].iloc[0]
        
        elif resolution_method == 'priority':
            # Já está ordenado por prioridade
            resolved_value = existing_trade['price']
        
        else:  # newest
            base_df.loc[mask] = new_trade
            resolved_value = new_trade['price']
        
        # Registrar conflito
        self.conflicts.append(MergeConflict(
            timestamp=new_trade['datetime'],
            field='price',
            source1=existing_trade['_source'],
            value1=existing_trade['price'],
            source2=source_name,
            value2=new_trade['price'],
            resolution=resolution_method,
            resolved_value=resolved_value
        ))
        
        self.stats['conflicts_resolved'] += 1
        
        return base_df
    
    def _resolve_candle_conflict(self, group: pd.DataFrame) -> pd.Series:
        """Resolve conflito entre candles do mesmo timestamp"""
        if len(group) == 1:
            return group.iloc[0]
        
        # Estratégia baseada em qualidade e prioridade
        if self.merge_config['conflict_resolution'] == 'quality':
            # Ponderação por qualidade
            weights = group['_quality'].values
            weights = weights / weights.sum()
            
            result = pd.Series()
            result['datetime'] = group['datetime'].iloc[0]
            
            # OHLC ponderado
            for col in ['open', 'high', 'low', 'close']:
                if col in group.columns:
                    result[col] = np.average(group[col], weights=weights)
            
            # Volume é soma
            if 'volume' in group.columns:
                result['volume'] = group['volume'].sum()
            
            # Registrar conflito
            self.stats['conflicts_resolved'] += len(group) - 1
            
            return result
        
        else:
            # Usar fonte de maior prioridade/qualidade
            best_idx = group['_priority'].argmin()
            if group['_priority'].iloc[best_idx] == group['_priority'].min():
                # Se há empate em prioridade, usar qualidade
                priority_mask = group['_priority'] == group['_priority'].min()
                best_idx = group[priority_mask]['_quality'].argmax()
            
            self.stats['conflicts_resolved'] += len(group) - 1
            return group.iloc[best_idx]
    
    def _compute_trade_hash(self, trade: pd.Series) -> str:
        """Computa hash único para um trade"""
        # Usar timestamp, price, volume para identificar uniqueness
        key = f"{trade['datetime']}_{trade['price']:.2f}_{trade['volume']}"
        return hashlib.md5(key.encode()).hexdigest()[:8]
    
    def _filter_date_range(self, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Filtra DataFrame por período"""
        if df.empty or 'datetime' not in df.columns:
            return df
        
        mask = (df['datetime'] >= start_date) & (df['datetime'] <= end_date)
        return df[mask].copy()
    
    def _post_process(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Pós-processamento do DataFrame merged"""
        if df.empty:
            return df
        
        # Ordenar por tempo
        if 'datetime' in df.columns:
            df = df.sort_values('datetime')
        
        # Resetar índice
        df = df.reset_index(drop=True)
        
        # Validações específicas por tipo
        if data_type == 'candles':
            # Garantir consistência OHLC
            df = self._ensure_ohlc_consistency(df)
        
        elif data_type == 'trades':
            # Remover trades com volume zero
            if 'volume' in df.columns:
                df = df[df['volume'] > 0]
        
        return df
    
    def _ensure_ohlc_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Garante que high >= low, etc"""
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # High deve ser o máximo
            df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
            
            # Low deve ser o mínimo
            df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        return df
    
    def _log_merge_stats(self):
        """Log estatísticas do merge"""
        self.logger.info("="*50)
        self.logger.info("ESTATÍSTICAS DO MERGE")
        self.logger.info("="*50)
        self.logger.info(f"Total de registros originais: {self.stats['total_records']:,}")
        self.logger.info(f"Registros após merge: {self.stats['merged_records']:,}")
        self.logger.info(f"Redução: {(1 - self.stats['merged_records']/self.stats['total_records']):.1%}")
        self.logger.info(f"Conflitos resolvidos: {self.stats['conflicts_resolved']:,}")
        self.logger.info(f"Duplicatas removidas: {self.stats['duplicates_removed']:,}")
        self.logger.info(f"Fontes utilizadas: {', '.join(self.stats['sources_used'])}")
        self.logger.info("="*50)
    
    def get_conflict_report(self) -> pd.DataFrame:
        """Retorna relatório de conflitos resolvidos"""
        if not self.conflicts:
            return pd.DataFrame()
        
        conflicts_data = []
        for conflict in self.conflicts:
            conflicts_data.append({
                'timestamp': conflict.timestamp,
                'field': conflict.field,
                'source1': conflict.source1,
                'value1': conflict.value1,
                'source2': conflict.source2,
                'value2': conflict.value2,
                'diff_pct': abs(conflict.value1 - conflict.value2) / conflict.value1 * 100,
                'resolution': conflict.resolution,
                'resolved_value': conflict.resolved_value
            })
        
        return pd.DataFrame(conflicts_data)
    
    def parallel_merge(self,
                      sources: List[DataSource],
                      start_date: datetime,
                      end_date: datetime,
                      data_type: str = 'trades') -> pd.DataFrame:
        """
        Merge paralelo para grandes volumes de dados
        
        Divide o período em chunks e processa em paralelo
        """
        # Dividir período em chunks
        days = (end_date - start_date).days
        chunk_days = max(1, days // self.merge_config['parallel_workers'])
        
        date_chunks = []
        current = start_date
        while current < end_date:
            chunk_end = min(current + timedelta(days=chunk_days), end_date)
            date_chunks.append((current, chunk_end))
            current = chunk_end + timedelta(days=1)
        
        # Processar em paralelo
        merged_chunks = []
        
        with ThreadPoolExecutor(max_workers=self.merge_config['parallel_workers']) as executor:
            futures = []
            
            for chunk_start, chunk_end in date_chunks:
                future = executor.submit(
                    self.merge_sources,
                    sources,
                    chunk_start,
                    chunk_end,
                    data_type
                )
                futures.append(future)
            
            # Coletar resultados
            for future in as_completed(futures):
                try:
                    chunk_result = future.result()
                    if not chunk_result.empty:
                        merged_chunks.append(chunk_result)
                except Exception as e:
                    self.logger.error(f"Erro no merge paralelo: {e}")
        
        # Combinar chunks
        if merged_chunks:
            final_result = pd.concat(merged_chunks, ignore_index=True)
            final_result = final_result.sort_values('datetime')
            return final_result.reset_index(drop=True)
        
        return pd.DataFrame()


if __name__ == "__main__":
    # Teste do merger
    merger = DataMerger()
    
    # Criar fontes de exemplo
    base_time = pd.date_range('2024-01-01 09:00', periods=100, freq='1min')
    
    # Fonte 1: Alta qualidade
    source1_data = pd.DataFrame({
        'datetime': base_time,
        'price': np.random.normal(5900, 10, 100),
        'volume': np.random.randint(100, 500, 100)
    })
    
    # Fonte 2: Média qualidade com alguns conflitos
    source2_data = pd.DataFrame({
        'datetime': base_time[::2],  # Metade dos pontos
        'price': np.random.normal(5905, 15, 50),  # Preços ligeiramente diferentes
        'volume': np.random.randint(150, 600, 50)
    })
    
    # Criar objetos DataSource
    sources = [
        DataSource(
            name="ProfitDLL",
            data=source1_data,
            quality_score=0.95,
            priority=1,
            metadata={'source': 'realtime'}
        ),
        DataSource(
            name="CSV_Historical",
            data=source2_data,
            quality_score=0.80,
            priority=2,
            metadata={'source': 'historical'}
        )
    ]
    
    # Executar merge
    merged = merger.merge_sources(
        sources=sources,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),
        data_type='trades'
    )
    
    print(f"\nDados merged: {len(merged)} registros")
    
    # Ver conflitos
    conflicts = merger.get_conflict_report()
    if not conflicts.empty:
        print(f"\nConflitos resolvidos: {len(conflicts)}")
        print(conflicts.head())