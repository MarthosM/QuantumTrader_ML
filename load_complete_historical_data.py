"""
Script para carregar dados históricos COMPLETOS do ProfitDLL
Inclui: OHLCV, bid/ask, trades agregados, e book de ofertas
Baseado no Manual ProfitDLL
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import threading
from ctypes import WINFUNCTYPE, WinDLL, c_int, c_wchar_p, c_double, c_uint, c_char, c_longlong, c_void_p, Structure
from typing import List, Dict, Optional
import json

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


class TAssetID(Structure):
    _fields_ = [
        ("pwcTicker", c_wchar_p),
        ("pwcBolsa", c_wchar_p), 
        ("nFeed", c_int)
    ]


class HistoricalDataCollector:
    """
    Coleta dados históricos completos do ProfitDLL
    Inclui todos os dados necessários para cálculo de features ML
    """
    
    def __init__(self, dll_path: str = None):
        self.dll_path = dll_path or r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
        self.dll = None
        self.logger = logging.getLogger(__name__)
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Estruturas de dados coletados
        self.candles_data = []      # Dados OHLCV
        self.trades_data = []       # Negócios realizados
        self.book_data = []         # Book de ofertas
        self.daily_data = []        # Dados diários
        
        # Controle de coleta
        self.collection_complete = threading.Event()
        self.connected = False
        self.market_connected = False
        
        # Estatísticas
        self.stats = {
            'candles': 0,
            'trades': 0,
            'book_updates': 0,
            'daily': 0
        }
        
    def initialize_dll(self):
        """Inicializa DLL e configura todos os callbacks necessários"""
        try:
            self.logger.info(f"Carregando DLL de: {self.dll_path}")
            self.dll = WinDLL(self.dll_path)
            
            # Configurar todos os callbacks
            self._setup_state_callback()
            self._setup_historical_callbacks()
            self._setup_book_callback()
            self._setup_daily_callback()
            
            # Inicializar DLL
            init_result = self.dll.DLLInitialize()
            self.logger.info(f"DLL inicializada: {init_result}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro inicializando DLL: {e}")
            return False
    
    def _setup_state_callback(self):
        """Configura callback de estado de conexão"""
        @WINFUNCTYPE(None, c_int, c_int)
        def state_callback(state_type, state):
            self.logger.debug(f"Estado: tipo={state_type}, valor={state}")
            
            if state_type == 0:  # Login
                if state == 0:
                    self.connected = True
                    self.logger.info("✅ Login realizado com sucesso")
            elif state_type == 2:  # Market Data
                if state == 4:
                    self.market_connected = True
                    self.logger.info("✅ Market Data conectado")
                    
        self.dll.DLLSetStateCallback(state_callback)
    
    def _setup_historical_callbacks(self):
        """Configura callbacks para dados históricos"""
        
        # 1. Callback de Candles (OHLCV + bid/ask)
        @WINFUNCTYPE(None, c_wchar_p, c_wchar_p, c_wchar_p, c_int, c_int, c_int,
                     c_double, c_double, c_double, c_double, c_double, c_double, c_double)
        def historical_candle_callback(date, time, ticker, period, nTrades, quantity,
                                     volume, open_price, high, low, close, bid, ask):
            try:
                # Combinar data e hora
                datetime_str = f"{date} {time}"
                dt = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S")
                
                candle = {
                    'datetime': dt,
                    'ticker': ticker,
                    'period': period,
                    'open': float(open_price),
                    'high': float(high),
                    'low': float(low),
                    'close': float(close),
                    'volume': float(volume),
                    'quantity': int(quantity),
                    'trades': int(nTrades),
                    'bid': float(bid),
                    'ask': float(ask),
                    'spread': float(ask) - float(bid),
                    'mid_price': (float(bid) + float(ask)) / 2
                }
                
                self.candles_data.append(candle)
                self.stats['candles'] += 1
                
                if self.stats['candles'] % 100 == 0:
                    self.logger.info(f"Candles coletados: {self.stats['candles']}")
                    
            except Exception as e:
                self.logger.error(f"Erro processando candle: {e}")
        
        self.dll.DLLSetHistoricalDataCallback(historical_candle_callback)
        
        # 2. Callback de Negócios (Trades)
        @WINFUNCTYPE(None, c_wchar_p, c_wchar_p, c_uint, c_double, c_double, c_int,
                     c_wchar_p, c_wchar_p, c_wchar_p, c_int, c_int)
        def historical_trades_callback(date, time, corretora, qtde, price, volume,
                                     buy_agent, sell_agent, trade_type, trade_id, seq_trade_id):
            try:
                # Parse datetime com milliseconds se disponível
                datetime_str = f"{date} {time}"
                try:
                    dt = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S.%f")
                except:
                    dt = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S")
                
                trade = {
                    'datetime': dt,
                    'price': float(price),
                    'volume': float(volume),
                    'quantity': int(qtde),
                    'buy_agent': buy_agent,
                    'sell_agent': sell_agent,
                    'trade_type': trade_type,  # Aggressor side
                    'broker': corretora,
                    'trade_id': trade_id,
                    'seq_id': seq_trade_id
                }
                
                self.trades_data.append(trade)
                self.stats['trades'] += 1
                
            except Exception as e:
                self.logger.error(f"Erro processando trade: {e}")
        
        self.dll.DLLSetHistoricalTradesCallback(historical_trades_callback)
    
    def _setup_book_callback(self):
        """Configura callback para book de ofertas histórico"""
        
        # Callback simplificado para top of book histórico
        @WINFUNCTYPE(None, c_wchar_p, c_wchar_p, c_int, c_int, c_double, c_int, c_double)
        def book_callback(date, time, position, side, price, quantity, count):
            try:
                datetime_str = f"{date} {time}"
                dt = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S.%f")
                
                book_update = {
                    'datetime': dt,
                    'position': position,
                    'side': 'bid' if side == 0 else 'ask',
                    'price': float(price),
                    'quantity': int(quantity),
                    'count': int(count)
                }
                
                self.book_data.append(book_update)
                self.stats['book_updates'] += 1
                
            except Exception as e:
                self.logger.error(f"Erro processando book: {e}")
        
        # Tentar registrar callback de book se disponível
        try:
            if hasattr(self.dll, 'DLLSetHistoricalBookCallback'):
                self.dll.DLLSetHistoricalBookCallback(book_callback)
                self.logger.info("Callback de book histórico configurado")
        except:
            self.logger.warning("Callback de book histórico não disponível")
    
    def _setup_daily_callback(self):
        """Configura callback para dados diários agregados"""
        
        @WINFUNCTYPE(None, c_wchar_p, c_double, c_double, c_double, c_double,
                     c_double, c_int, c_double, c_double)
        def daily_callback(date, open_price, high, low, close, volume,
                          quantity, financial_volume, trades):
            try:
                dt = datetime.strptime(date, "%d/%m/%Y")
                
                daily = {
                    'date': dt,
                    'open': float(open_price),
                    'high': float(high),
                    'low': float(low),
                    'close': float(close),
                    'volume': float(volume),
                    'quantity': int(quantity),
                    'financial_volume': float(financial_volume),
                    'trades': int(trades)
                }
                
                self.daily_data.append(daily)
                self.stats['daily'] += 1
                
            except Exception as e:
                self.logger.error(f"Erro processando daily: {e}")
        
        # Registrar se disponível
        try:
            if hasattr(self.dll, 'DLLSetDailyDataCallback'):
                self.dll.DLLSetDailyDataCallback(daily_callback)
                self.logger.info("Callback de dados diários configurado")
        except:
            self.logger.warning("Callback de dados diários não disponível")
    
    def connect(self, username: str = None, password: str = None):
        """Conecta ao servidor"""
        try:
            self.logger.info("Conectando ao servidor ProfitChart...")
            
            # Usar credenciais do ambiente ou parâmetros
            user = username or os.getenv("PROFIT_USER", "")
            pwd = password or os.getenv("PROFIT_PASSWORD", "")
            
            # Servidor
            server = os.getenv("PROFIT_SERVER", "profitdemo.profitchart.com.br")
            
            # Conectar
            if user and pwd:
                # Login com credenciais
                self.logger.info(f"Conectando com usuário: {user}")
                connect_result = self.dll.DLLConnect2(server, user, pwd)
            else:
                # Conexão simples
                port = "80"
                connect_result = self.dll.DLLConnect(server, port)
            
            self.logger.info(f"Resultado conexão: {connect_result}")
            
            # Aguardar conexão
            timeout = 30
            start = time.time()
            while not self.market_connected and (time.time() - start) < timeout:
                time.sleep(0.5)
                
            if not self.market_connected:
                self.logger.error("Timeout esperando conexão com market data")
                return False
                
            self.logger.info("✅ Conexão estabelecida com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro conectando: {e}")
            return False
    
    def load_historical_data(self, ticker: str, start_date: datetime, 
                           end_date: datetime, period_minutes: int = 1) -> Dict[str, pd.DataFrame]:
        """
        Carrega dados históricos completos em blocos de 9 dias
        
        Returns:
            Dict com DataFrames:
            - 'candles': OHLCV + bid/ask
            - 'trades': Negócios agregados
            - 'microstructure': Métricas calculadas
        """
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Carregando dados históricos de {ticker}")
        self.logger.info(f"Período: {start_date.date()} até {end_date.date()}")
        self.logger.info(f"Timeframe: {period_minutes} minuto(s)")
        self.logger.info(f"{'='*60}\n")
        
        # Limpar dados anteriores
        self.candles_data = []
        self.trades_data = []
        self.book_data = []
        self.stats = {'candles': 0, 'trades': 0, 'book_updates': 0, 'daily': 0}
        
        # Calcular número de blocos necessários
        total_days = (end_date - start_date).days
        num_blocks = (total_days // 9) + (1 if total_days % 9 > 0 else 0)
        
        self.logger.info(f"Total de dias: {total_days}")
        self.logger.info(f"Número de blocos (9 dias cada): {num_blocks}")
        
        current_start = start_date
        block_num = 0
        
        # Carregar em blocos de 9 dias
        while current_start < end_date:
            block_num += 1
            current_end = min(current_start + timedelta(days=8, hours=23, minutes=59), end_date)
            
            self.logger.info(f"\n[Bloco {block_num}/{num_blocks}] {current_start.date()} até {current_end.date()}")
            
            # Solicitar dados do bloco
            self._request_block_data(ticker, current_start, current_end, period_minutes)
            
            # Aguardar processamento
            wait_time = 10  # Aguardar mais tempo para garantir todos os dados
            self.logger.info(f"Aguardando {wait_time} segundos para processar dados...")
            time.sleep(wait_time)
            
            # Log progresso
            self.logger.info(f"Dados coletados até agora:")
            self.logger.info(f"  - Candles: {self.stats['candles']}")
            self.logger.info(f"  - Trades: {self.stats['trades']}")
            self.logger.info(f"  - Book updates: {self.stats['book_updates']}")
            
            # Próximo bloco
            current_start = current_end + timedelta(minutes=1)
            
            # Pausa entre blocos para não sobrecarregar
            if current_start < end_date:
                time.sleep(2)
        
        # Processar e retornar dados
        return self._process_collected_data()
    
    def _request_block_data(self, ticker: str, start_date: datetime, 
                          end_date: datetime, period_minutes: int):
        """Solicita dados de um bloco específico"""
        try:
            # Criar estrutura AssetID
            asset = TAssetID()
            asset.pwcTicker = ticker
            asset.pwcBolsa = "BOVESPA"
            asset.nFeed = 0
            
            # Formatar datas
            start_str = start_date.strftime("%d/%m/%Y")
            end_str = end_date.strftime("%d/%m/%Y")
            
            self.logger.info(f"Solicitando dados: {ticker} de {start_str} até {end_str}")
            
            # 1. Solicitar dados de candles (OHLCV + bid/ask)
            result_candles = self.dll.DLLGetHistoricalData(
                asset,
                c_int(period_minutes),
                start_str,
                end_str
            )
            self.logger.debug(f"Resultado candles: {result_candles}")
            
            # 2. Solicitar dados de negócios se disponível
            try:
                if hasattr(self.dll, 'DLLGetHistoricalTrades'):
                    result_trades = self.dll.DLLGetHistoricalTrades(
                        asset,
                        start_str,
                        end_str
                    )
                    self.logger.debug(f"Resultado trades: {result_trades}")
            except Exception as e:
                self.logger.warning(f"Trades históricos não disponíveis: {e}")
            
            # 3. Solicitar book histórico se disponível
            try:
                if hasattr(self.dll, 'DLLGetHistoricalBook'):
                    result_book = self.dll.DLLGetHistoricalBook(
                        asset,
                        start_str,
                        end_str,
                        c_int(5)  # Top 5 níveis
                    )
                    self.logger.debug(f"Resultado book: {result_book}")
            except Exception as e:
                self.logger.warning(f"Book histórico não disponível: {e}")
                
        except Exception as e:
            self.logger.error(f"Erro solicitando dados: {e}")
    
    def _process_collected_data(self) -> Dict[str, pd.DataFrame]:
        """Processa dados coletados e gera DataFrames finais"""
        
        self.logger.info("\n" + "="*60)
        self.logger.info("PROCESSANDO DADOS COLETADOS")
        self.logger.info("="*60)
        
        result = {}
        
        # 1. Processar Candles
        if self.candles_data:
            self.logger.info(f"\nProcessando {len(self.candles_data)} candles...")
            
            df_candles = pd.DataFrame(self.candles_data)
            df_candles.set_index('datetime', inplace=True)
            df_candles.sort_index(inplace=True)
            
            # Remover duplicatas
            df_candles = df_candles[~df_candles.index.duplicated(keep='first')]
            
            # Adicionar colunas calculadas
            df_candles['spread_pct'] = (df_candles['spread'] / df_candles['close']) * 100
            df_candles['typical_price'] = (df_candles['high'] + df_candles['low'] + df_candles['close']) / 3
            
            result['candles'] = df_candles
            
            self.logger.info(f"Candles processados: {len(df_candles)}")
            self.logger.info(f"Período: {df_candles.index[0]} até {df_candles.index[-1]}")
        
        # 2. Processar Trades e criar microestrutura
        if self.trades_data:
            self.logger.info(f"\nProcessando {len(self.trades_data)} trades...")
            
            df_trades = pd.DataFrame(self.trades_data)
            df_trades.set_index('datetime', inplace=True)
            df_trades.sort_index(inplace=True)
            
            # Agregar trades por minuto para criar microestrutura
            df_micro = self._create_microstructure_from_trades(df_trades, df_candles)
            result['microstructure'] = df_micro
            
            self.logger.info(f"Microestrutura criada: {len(df_micro)} registros")
        
        # 3. Combinar todos os dados
        if 'candles' in result and 'microstructure' in result:
            # Merge candles com microestrutura
            df_complete = pd.merge(
                result['candles'],
                result['microstructure'],
                left_index=True,
                right_index=True,
                how='left'
            )
            
            # Preencher valores faltantes
            df_complete = df_complete.fillna(method='ffill').fillna(0)
            
            result['complete'] = df_complete
            
            self.logger.info(f"\nDataset completo criado:")
            self.logger.info(f"Shape: {df_complete.shape}")
            self.logger.info(f"Colunas: {len(df_complete.columns)}")
        
        return result
    
    def _create_microstructure_from_trades(self, df_trades: pd.DataFrame, 
                                         df_candles: pd.DataFrame) -> pd.DataFrame:
        """Cria métricas de microestrutura agregando trades por período"""
        
        # Agrupar trades por minuto (ou período do candle)
        freq = f"{(df_candles.index[1] - df_candles.index[0]).seconds//60}T"
        
        # Identificar lado do trade (simplificado)
        # Se trade_type indica aggressor, usar isso
        # Senão, inferir pelo movimento do preço
        df_trades['is_buy'] = df_trades['trade_type'].apply(
            lambda x: 1 if x in ['BUY', 'COMPRA', 'C'] else 0
        )
        
        # Se trade_type não é claro, usar tick rule
        if df_trades['is_buy'].sum() == 0:
            df_trades['price_change'] = df_trades['price'].diff()
            df_trades['is_buy'] = (df_trades['price_change'] >= 0).astype(int)
        
        # Agregar por período
        micro_agg = df_trades.resample(freq).agg({
            'price': ['first', 'last', 'mean', 'std'],
            'quantity': 'sum',
            'volume': 'sum',
            'is_buy': ['sum', 'count']
        })
        
        # Flatten column names
        micro_agg.columns = ['_'.join(col).strip() for col in micro_agg.columns.values]
        
        # Criar DataFrame de microestrutura
        df_micro = pd.DataFrame(index=df_candles.index)
        
        # Mapear dados agregados
        for idx in df_micro.index:
            if idx in micro_agg.index:
                row = micro_agg.loc[idx]
                
                total_trades = row.get('is_buy_count', 0)
                buy_trades = row.get('is_buy_sum', 0)
                sell_trades = total_trades - buy_trades
                
                total_volume = row.get('volume_sum', 0)
                total_quantity = row.get('quantity_sum', 0)
                
                # Estimar volume por lado
                if total_trades > 0:
                    buy_ratio = buy_trades / total_trades
                else:
                    buy_ratio = 0.5
                
                df_micro.loc[idx, 'buy_trades'] = buy_trades
                df_micro.loc[idx, 'sell_trades'] = sell_trades
                df_micro.loc[idx, 'total_trades'] = total_trades
                df_micro.loc[idx, 'buy_volume'] = total_volume * buy_ratio
                df_micro.loc[idx, 'sell_volume'] = total_volume * (1 - buy_ratio)
                df_micro.loc[idx, 'avg_trade_size'] = total_volume / (total_quantity + 1e-10)
                df_micro.loc[idx, 'trade_imbalance'] = (buy_trades - sell_trades) / (total_trades + 1e-10)
                df_micro.loc[idx, 'volume_imbalance'] = df_micro.loc[idx, 'buy_volume'] - df_micro.loc[idx, 'sell_volume']
                
                # Volatilidade dos preços dos trades
                price_std = row.get('price_std', 0)
                if pd.notna(price_std) and price_std > 0:
                    df_micro.loc[idx, 'trade_price_volatility'] = price_std
        
        # Preencher zeros onde não houve trades
        df_micro = df_micro.fillna(0)
        
        return df_micro
    
    def save_complete_dataset(self, data: Dict[str, pd.DataFrame], base_filename: str = "wdo_complete"):
        """Salva datasets completos em diferentes formatos"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Salvar dataset completo para ML
        if 'complete' in data:
            filename = f"{base_filename}_ml_ready_{timestamp}.csv"
            data['complete'].to_csv(filename)
            self.logger.info(f"\n✅ Dataset ML salvo: {filename}")
            
            # Estatísticas
            df = data['complete']
            self.logger.info(f"Shape: {df.shape}")
            self.logger.info(f"Período: {df.index[0]} até {df.index[-1]}")
            self.logger.info(f"Colunas disponíveis: {len(df.columns)}")
            
            # Verificar dados essenciais
            essential_cols = ['open', 'high', 'low', 'close', 'volume', 
                            'bid', 'ask', 'buy_volume', 'sell_volume']
            missing = [col for col in essential_cols if col not in df.columns]
            
            if missing:
                self.logger.warning(f"⚠️ Colunas faltando: {missing}")
            else:
                self.logger.info("✅ Todas colunas essenciais presentes!")
        
        # 2. Salvar dados separados para análise
        if 'candles' in data:
            filename = f"{base_filename}_candles_{timestamp}.csv"
            data['candles'].to_csv(filename)
            self.logger.info(f"Candles salvos: {filename}")
        
        if 'microstructure' in data:
            filename = f"{base_filename}_microstructure_{timestamp}.csv"
            data['microstructure'].to_csv(filename)
            self.logger.info(f"Microestrutura salva: {filename}")
        
        # 3. Salvar metadados
        metadata = {
            'collection_date': datetime.now().isoformat(),
            'stats': self.stats,
            'period': {
                'start': str(data['candles'].index[0]) if 'candles' in data else None,
                'end': str(data['candles'].index[-1]) if 'candles' in data else None
            },
            'columns': {
                'candles': list(data['candles'].columns) if 'candles' in data else [],
                'complete': list(data['complete'].columns) if 'complete' in data else []
            }
        }
        
        with open(f"{base_filename}_metadata_{timestamp}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Metadados salvos: {base_filename}_metadata_{timestamp}.json")
    
    def disconnect(self):
        """Desconecta do servidor"""
        try:
            if self.dll:
                self.dll.DLLDisconnect()
                self.dll.DLLFinalize()
                self.logger.info("✅ Desconectado com sucesso")
        except Exception as e:
            self.logger.error(f"Erro desconectando: {e}")


def main():
    """Função principal para coletar dados históricos completos"""
    
    # Configurações
    TICKER = "WDOH25"  # Contrato atual do mini-índice
    START_DATE = datetime(2025, 1, 27, 9, 0)   # Início da coleta
    END_DATE = datetime(2025, 1, 27, 17, 0)    # Fim da coleta (1 dia para teste)
    PERIOD_MINUTES = 1  # Timeframe de 1 minuto
    
    print("="*80)
    print("COLETOR DE DADOS HISTÓRICOS COMPLETOS - PROFITDLL")
    print("="*80)
    print(f"\nConfiguração:")
    print(f"  Ticker: {TICKER}")
    print(f"  Período: {START_DATE} até {END_DATE}")
    print(f"  Timeframe: {PERIOD_MINUTES} minuto(s)")
    print("="*80)
    
    collector = HistoricalDataCollector()
    
    try:
        # Inicializar
        print("\n1. Inicializando DLL...")
        if not collector.initialize_dll():
            print("❌ Erro inicializando DLL")
            return
        
        # Conectar
        print("\n2. Conectando ao servidor...")
        if not collector.connect():
            print("❌ Erro conectando ao servidor")
            return
        
        # Carregar dados
        print("\n3. Carregando dados históricos...")
        print("   (Isso pode levar alguns minutos)")
        
        data = collector.load_historical_data(TICKER, START_DATE, END_DATE, PERIOD_MINUTES)
        
        # Salvar dados
        if data:
            print("\n4. Salvando datasets...")
            collector.save_complete_dataset(data, base_filename=f"{TICKER.lower()}_complete")
            
            print("\n✅ COLETA CONCLUÍDA COM SUCESSO!")
            
            # Mostrar resumo
            if 'complete' in data:
                df = data['complete']
                print(f"\nResumo do dataset completo:")
                print(f"  Total de registros: {len(df)}")
                print(f"  Total de colunas: {len(df.columns)}")
                print(f"  Memória utilizada: {df.memory_usage().sum() / 1024 / 1024:.1f} MB")
                
                print(f"\nPrimeiras colunas disponíveis:")
                for i, col in enumerate(df.columns[:20], 1):
                    print(f"  {i:2d}. {col}")
                
                if len(df.columns) > 20:
                    print(f"  ... e mais {len(df.columns) - 20} colunas")
        else:
            print("❌ Nenhum dado foi coletado")
            
    except Exception as e:
        print(f"\n❌ Erro durante coleta: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n5. Desconectando...")
        collector.disconnect()
    
    print("\n" + "="*80)
    print("Processo finalizado!")
    print("="*80)


if __name__ == "__main__":
    # Verificar se tem credenciais no ambiente
    if not os.getenv("PROFIT_USER"):
        print("\n⚠️ ATENÇÃO: Variáveis de ambiente não configuradas!")
        print("Configure as seguintes variáveis:")
        print("  PROFIT_USER=seu_usuario")
        print("  PROFIT_PASSWORD=sua_senha")
        print("  PROFIT_SERVER=servidor_profit")
        print("\nOu edite o script com suas credenciais.")
        
        response = input("\nDeseja continuar mesmo assim? (s/n): ")
        if response.lower() != 's':
            exit()
    
    main()