"""
Integração entre ConnectionManager e DataLoader
Gerencia fluxo de dados real-time para DataFrames
"""

import logging
from typing import Dict, List
from datetime import datetime, timedelta
import pandas as pd
from collections import deque
from threading import Lock

class DataIntegration:
    """Integra dados real-time do broker com estrutura de DataFrames"""
    
    def __init__(self, connection_manager, data_loader):
        self.connection_manager = connection_manager
        self.data_loader = data_loader
        self.logger = logging.getLogger('DataIntegration')
        
        # Buffer para trades (thread-safe)
        self.trades_buffer = deque(maxlen=10000)  # Últimos 10k trades
        self.buffer_lock = Lock()
        
        # DataFrames de candles
        self.candles_1min = pd.DataFrame()
        self.last_candle_time = None
        
        # Contador para dados históricos
        self._historical_data_count = 0
        
        # Flag para marcar conclusão do carregamento histórico
        self._historical_loading_complete = False
        
        # Controle para print do DataFrame
        self._last_dataframe_print = None
        self._candle_count = 0
        
        # Registrar callback para trades (só se connection_manager não for None)
        if self.connection_manager is not None:
            self.connection_manager.register_trade_callback(self._on_trade)
            self.logger.info("Trade callback registrado com ConnectionManager")
        else:
            self.logger.warning("ConnectionManager está None - callback de trades não registrado")
        
    def _on_trade(self, trade_data: Dict):
        """Callback para processar trades em tempo real"""
        try:
            # Verificar se é evento especial de conclusão de dados históricos
            if trade_data.get('event_type') == 'historical_data_complete':
                self.logger.info("🎉 Recebido sinal de conclusão de dados históricos")
                
                # Fazer log do DataFrame criado
                self.log_dataframe_summary()
                
                # Marcar que carregamento histórico foi concluído
                self._historical_loading_complete = True
                
                return
            
            # Validar dados do trade
            if not self._validate_trade(trade_data):
                return
            
            # Criar/atualizar candle
            completed_candle = self.data_loader.create_or_update_candle(trade_data)
            
            # Se um candle foi completado
            if completed_candle is not None:
                # Notificar sistema que novo candle está disponível
                self._on_candle_completed(completed_candle)
                
                # Adicionar ao buffer histórico (opcional)
                with self.buffer_lock:
                    self.trades_buffer.append(trade_data)
                    
        except Exception as e:
            self.logger.error(f"Erro processando trade: {e}")

    def _on_candle_completed(self, candle: pd.DataFrame):
        """Callback quando um candle é completado"""
        try:
            # Incrementar contador de candles
            self._candle_count += 1
            
            # Log básico
            self.logger.info(f"Novo candle formado: {candle.index[0]}")
            
            # PRINT DO DATAFRAME a cada 10 candles ou na primeira vez
            if self._candle_count == 1 or self._candle_count % 10 == 0:
                self._print_dataframe_summary()
            
            # Também fazer print se passou muito tempo desde o último
            from datetime import datetime, timedelta
            now = datetime.now()
            if (self._last_dataframe_print is None or 
                (now - self._last_dataframe_print) > timedelta(minutes=5)):
                self._print_dataframe_summary()
            
        except Exception as e:
            self.logger.error(f"Erro processando candle completo: {e}")
    
    def _print_dataframe_summary(self):
        """Imprime resumo do DataFrame atual"""
        try:
            from datetime import datetime
            
            # MÉTODO 1: Tentar dados do data_loader
            df = None
            source = ""
            
            if hasattr(self.data_loader, 'candles_df') and not self.data_loader.candles_df.empty:
                df = self.data_loader.candles_df
                source = "data_loader.candles_df"
            elif hasattr(self.data_loader, 'candles_buffer') and not self.data_loader.candles_buffer.empty:
                df = self.data_loader.candles_buffer
                source = "data_loader.candles_buffer"
            elif hasattr(self, 'candles_1min') and not self.candles_1min.empty:
                df = self.candles_1min
                source = "data_integration.candles_1min"
            
            if df is not None and not df.empty:
                print("\n" + "="*80)
                print("📊 RESUMO DO DATAFRAME DE CANDLES ATUALIZADO")
                print("="*80)
                print(f"🕐 Timestamp: {datetime.now().strftime('%H:%M:%S')}")
                print(f"� Fonte: {source}")
                print(f"�📈 Total de candles: {len(df)}")
                
                if len(df) > 0:
                    print(f"📅 Período: {df.index.min()} até {df.index.max()}")
                    print(f"⏱️  Duração: {df.index.max() - df.index.min()}")
                    
                    # Verificar quais colunas existem
                    available_cols = df.columns.tolist()
                    print(f"� Colunas disponíveis: {available_cols}")
                    
                    if 'close' in df.columns:
                        print(f"�💰 Último preço: R$ {df['close'].iloc[-1]:,.2f}")
                        print(f"� Preço máximo: R$ {df['high'].max():,.2f}" if 'high' in df.columns else "")
                        print(f"� Preço mínimo: R$ {df['low'].min():,.2f}" if 'low' in df.columns else "")
                        print(f"� Preço médio: R$ {df['close'].mean():,.2f}")
                    
                    if 'volume' in df.columns:
                        print(f"� Volume total: {df['volume'].sum():,.0f}")
                    
                    if 'trades' in df.columns:
                        print(f"� Trades processados: {df['trades'].sum():,.0f}")
                    
                    # Últimos 5 candles (ou todos se menos de 5)
                    num_to_show = min(5, len(df))
                    print(f"\n🔍 ÚLTIMOS {num_to_show} CANDLES:")
                    
                    # Mostrar apenas colunas principais se existirem
                    display_cols = []
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in df.columns:
                            display_cols.append(col)
                    
                    if display_cols:
                        print(df[display_cols].tail(num_to_show).to_string())
                    else:
                        print(df.tail(num_to_show).to_string())
                
                print("="*80)
                
                # Atualizar timestamp do último print
                self._last_dataframe_print = datetime.now()
                
            else:
                print("\n" + "="*80)
                print("⚠️ DATAFRAME DE CANDLES VAZIO")
                print("="*80)
                print(f"🕐 Timestamp: {datetime.now().strftime('%H:%M:%S')}")
                print(f"📊 Total de candles: 0")
                print(f"🔍 Verificado em:")
                print(f"   - data_loader.candles_df: {'Existe' if hasattr(self.data_loader, 'candles_df') else 'Não existe'}")
                print(f"   - data_loader.candles_buffer: {'Existe' if hasattr(self.data_loader, 'candles_buffer') else 'Não existe'}")
                print(f"   - data_integration.candles_1min: {'Existe' if hasattr(self, 'candles_1min') else 'Não existe'}")
                
                # Tentar mostrar informações do data_loader
                if hasattr(self.data_loader, '__dict__'):
                    print(f"🔧 Atributos do DataLoader: {list(self.data_loader.__dict__.keys())}")
                
                print("💡 Aguardando dados históricos ou tempo real...")
                print("="*80)
                
        except Exception as e:
            self.logger.error(f"Erro imprimindo DataFrame: {e}")
            print(f"\n❌ ERRO ao imprimir DataFrame: {e}")
    
    def print_current_dataframe(self):
        """Método público para imprimir DataFrame quando solicitado"""
        self._print_dataframe_summary()
    
    def get_dataframe_stats(self) -> dict:
        """Retorna estatísticas do DataFrame atual"""
        try:
            if hasattr(self.data_loader, 'candles_df') and not self.data_loader.candles_df.empty:
                df = self.data_loader.candles_df
                return {
                    'total_candles': len(df),
                    'start_time': df.index.min(),
                    'end_time': df.index.max(),
                    'last_price': float(df['close'].iloc[-1]),
                    'total_volume': float(df['volume'].sum()),
                    'max_price': float(df['high'].max()),
                    'min_price': float(df['low'].min()),
                    'avg_price': float(df['close'].mean())
                }
            else:
                return {'total_candles': 0, 'status': 'DataFrame vazio'}
        except Exception as e:
            self.logger.error(f"Erro obtendo estatísticas: {e}")
            return {'error': str(e)}
    
    def force_create_test_dataframe(self):
        """Força criação de DataFrame de teste para demonstração"""
        try:
            from datetime import datetime, timedelta
            import pandas as pd
            import numpy as np
            
            print("🧪 Criando DataFrame de teste...")
            
            # Criar dados de teste
            base_time = datetime.now() - timedelta(minutes=30)
            
            # Gerar dados de candles realistas
            dates = []
            data = []
            
            for i in range(10):  # 10 candles de teste
                candle_time = base_time + timedelta(minutes=i)
                dates.append(candle_time)
                
                # Preços realistas para WDO
                base_price = 5100 + (i * 0.5)  
                high = base_price + np.random.uniform(0.5, 2.0)
                low = base_price - np.random.uniform(0.5, 2.0)
                close = base_price + np.random.uniform(-1.0, 1.0)
                volume = np.random.randint(50, 200)
                trades = np.random.randint(5, 20)
                
                data.append({
                    'open': base_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                    'trades': trades
                })
            
            # Criar DataFrame
            test_df = pd.DataFrame(data, index=pd.to_datetime(dates))
            
            # Armazenar no data_loader
            if not hasattr(self.data_loader, 'candles_df'):
                self.data_loader.candles_df = pd.DataFrame()
            
            self.data_loader.candles_df = test_df
            
            # Também armazenar localmente
            self.candles_1min = test_df
            
            print("✅ DataFrame de teste criado com sucesso!")
            print(f"📊 {len(test_df)} candles criados")
            
            # Fazer print imediato
            self._print_dataframe_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro criando DataFrame de teste: {e}")
            return False
            
    def _validate_trade(self, trade_data: Dict) -> bool:
        """Valida dados do trade"""
        required_fields = ['timestamp', 'price', 'volume', 'quantity']
        
        # Verificar campos obrigatórios
        for field in required_fields:
            if field not in trade_data:
                self.logger.error(f"Campo obrigatório ausente: {field}")
                return False
        
        # Verificar valores válidos
        if trade_data['price'] <= 0:
            self.logger.error(f"Preço inválido: {trade_data['price']}")
            return False
            
        if trade_data['volume'] < 0:
            self.logger.error(f"Volume inválido: {trade_data['volume']}")
            return False
            
        # Verificar timestamp - LÓGICA MELHORADA
        # CORREÇÃO: Permitir dados históricos E evitar loops
        now = datetime.now()
        if isinstance(trade_data['timestamp'], str):
            trade_time = pd.to_datetime(trade_data['timestamp'])
        else:
            trade_time = trade_data['timestamp']
            
        # Se é dado histórico, aplicar lógica específica
        is_historical = trade_data.get('is_historical', False)
        
        if is_historical:
            # Incrementar contador para dados históricos
            self._historical_data_count += 1
            
            # Para dados históricos: aceitar qualquer data até 7 dias atrás
            days_old = (now - trade_time).days
            if days_old > 7:
                # Dados muito antigos (mais de 7 dias) podem ser inválidos
                if self._historical_data_count % 10000 == 0:  # Log ocasional
                    self.logger.warning(f"Dados históricos muito antigos ({days_old} dias) - verificar se são válidos")
            
            # Log informativo ocasional
            if self._historical_data_count % 1000 == 0:
                self.logger.info(f"Processando dados históricos de {days_old} dias atrás ({self._historical_data_count} processados)")
                
        else:
            # Para dados em tempo real: máximo 5 minutos de atraso (não 1 minuto)
            # CORREÇÃO: Aumentar tolerância para 5 minutos para evitar rejeições desnecessárias
            seconds_old = (now - trade_time).total_seconds()
            if seconds_old > 300:  # 5 minutos
                self.logger.warning(f"Trade tempo real muito antigo ignorado ({seconds_old:.0f}s)")
                return False
            elif seconds_old > 60:  # Entre 1-5 minutos: warning mas aceita
                if self._historical_data_count % 100 == 0:  # Log ocasional
                    self.logger.info(f"Trade tempo real com {seconds_old:.0f}s de atraso aceito")
            
        return True
    
    def _form_candle(self, start_time: datetime, end_time: datetime):
        """Forma candle de 1 minuto com trades do buffer"""
        with self.buffer_lock:
            # Filtrar trades do período
            period_trades = [
                t for t in self.trades_buffer
                if start_time <= t['timestamp'] < end_time
            ]
        
        if not period_trades:
            return
        
        # Criar candle
        prices = [t['price'] for t in period_trades]
        volumes = [t['volume'] for t in period_trades]
        
        candle = pd.DataFrame([{
            'timestamp': start_time,
            'open': prices[0],
            'high': max(prices),
            'low': min(prices),
            'close': prices[-1],
            'volume': sum(volumes),
            'trades': len(period_trades)
        }])
        
        candle.set_index('timestamp', inplace=True)
        
        # Adicionar ao DataFrame principal
        if self.candles_1min.empty:
            self.candles_1min = candle
        else:
            self.candles_1min = pd.concat([self.candles_1min, candle])
            
        # Limitar tamanho (manter últimas 1440 candles = 24h)
        if len(self.candles_1min) > 1440:
            self.candles_1min = self.candles_1min.iloc[-1440:]
        
        self.logger.debug(f"Candle formado: {start_time} - OHLC: {candle.iloc[0].to_dict()}")
        
        # Log informativo sobre novo candle
        self.logger.info(f"Novo candle formado: {start_time}")
    
    def log_dataframe_summary(self):
        """
        Log informações do DataFrame de candles criado
        """
        try:
            if self.candles_1min.empty:
                self.logger.warning("❌ DataFrame de candles está vazio - nenhum dado foi processado")
                return
            
            # Informações básicas
            total_candles = len(self.candles_1min)
            
            # Período coberto
            start_time = self.candles_1min.index.min()
            end_time = self.candles_1min.index.max()
            duration = end_time - start_time
            
            # Estatísticas dos dados
            total_volume = self.candles_1min['volume'].sum()
            total_trades = self.candles_1min['trades'].sum()
            avg_price = self.candles_1min['close'].mean()
            
            self.logger.info("=" * 50)
            self.logger.info("📊 RESUMO DO DATAFRAME DE CANDLES CRIADO")
            self.logger.info("=" * 50)
            self.logger.info(f"✅ Total de candles: {total_candles}")
            self.logger.info(f"📅 Período: {start_time} até {end_time}")
            self.logger.info(f"⏱️  Duração: {duration}")
            self.logger.info(f"📈 Volume total: {total_volume:,.0f}")
            self.logger.info(f"🔄 Total de trades: {total_trades:,.0f}")
            self.logger.info(f"💰 Preço médio: R$ {avg_price:,.2f}")
            self.logger.info(f"🎯 Dados históricos processados: {self._historical_data_count:,.0f}")
            
            # Verificar continuidade dos dados
            expected_candles = int(duration.total_seconds() / 60) + 1
            if total_candles < expected_candles * 0.8:  # Se menos de 80% dos candles esperados
                self.logger.warning(f"⚠️ Possível gap nos dados - Esperado: ~{expected_candles}, Atual: {total_candles}")
            else:
                self.logger.info(f"✅ Cobertura de dados boa: {(total_candles/expected_candles)*100:.1f}%")
                
            # Último candle
            if not self.candles_1min.empty:
                last_candle = self.candles_1min.iloc[-1]
                self.logger.info(f"🕐 Último candle: {self.candles_1min.index[-1]} - Close: R$ {last_candle['close']:,.2f}")
            
            self.logger.info("=" * 50)
            
        except Exception as e:
            self.logger.error(f"Erro no log do DataFrame: {e}")
    
    def get_candles(self, interval: str = '1min') -> pd.DataFrame:
        """
        Retorna candles no intervalo solicitado
        
        Args:
            interval: '1min', '5min', '15min', etc
            
        Returns:
            pd.DataFrame: Candles no intervalo solicitado
        """
        if interval == '1min':
            return self.candles_1min.copy()
        else:
            # Agregar para intervalo maior
            return self.data_loader.aggregate_candles(self.candles_1min, interval)
    
    def get_latest_candles(self, n: int = 100, interval: str = '1min') -> pd.DataFrame:
        """Retorna últimos N candles"""
        candles = self.get_candles(interval)
        return candles.tail(n)
    
    