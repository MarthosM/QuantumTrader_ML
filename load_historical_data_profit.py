"""
Script para carregar dados históricos do ProfitDLL
Carrega dados de 9 em 9 dias e concatena no formato adequado para features ML
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from ctypes import WINFUNCTYPE, WinDLL, c_int, c_wchar_p, c_double, c_uint, c_char, c_longlong, c_void_p, Structure
import threading
from typing import List, Dict

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


class TAssetID(Structure):
    _fields_ = [
        ("pwcTicker", c_wchar_p),
        ("pwcBolsa", c_wchar_p), 
        ("nFeed", c_int)
    ]


class HistoricalDataLoader:
    """Carrega dados históricos do ProfitDLL em blocos"""
    
    def __init__(self, dll_path: str = None):
        self.dll_path = dll_path or r"C:\Users\marth\Downloads\ProfitDLL\DLLs\Win64\ProfitDLL.dll"
        self.dll = None
        self.logger = logging.getLogger(__name__)
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Dados coletados
        self.historical_data = []
        self.candles_data = []
        self.current_batch = []
        self.data_ready = threading.Event()
        
        # Controle de estado
        self.connected = False
        self.loading_complete = False
        
    def initialize_dll(self):
        """Inicializa DLL e configura callbacks"""
        try:
            self.logger.info(f"Carregando DLL de: {self.dll_path}")
            self.dll = WinDLL(self.dll_path)
            
            # Configurar callbacks essenciais
            self._setup_callbacks()
            
            # Inicializar DLL
            init_result = self.dll.DLLInitialize()
            self.logger.info(f"DLL inicializada: {init_result}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro inicializando DLL: {e}")
            return False
    
    def _setup_callbacks(self):
        """Configura callbacks necessários"""
        
        # Callback de estado de conexão
        @WINFUNCTYPE(None, c_int, c_int)
        def state_callback(state_type, state):
            if state_type == 2 and state == 4:  # Market conectado
                self.connected = True
                self.logger.info("✅ Mercado conectado")
                
        self.dll.DLLSetStateCallback(state_callback)
        
        # Callback de dados históricos (candles)
        @WINFUNCTYPE(None, c_wchar_p, c_wchar_p, c_wchar_p, c_int, c_int, c_int, 
                     c_int, c_double, c_double, c_double, c_double, c_double, c_double)
        def historical_candle_callback(date, time, ticker, period, nTrades, quantity,
                                     volume, open_price, high, low, close, bid, ask):
            try:
                # Combinar data e hora
                datetime_str = f"{date} {time}"
                dt = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S")
                
                candle = {
                    'datetime': dt,
                    'ticker': ticker,
                    'open': float(open_price),
                    'high': float(high),
                    'low': float(low),
                    'close': float(close),
                    'volume': float(volume),
                    'quantity': int(quantity),
                    'trades': int(nTrades),
                    'bid': float(bid),
                    'ask': float(ask)
                }
                
                self.current_batch.append(candle)
                
                if len(self.current_batch) % 1000 == 0:
                    self.logger.info(f"Carregados {len(self.current_batch)} candles...")
                    
            except Exception as e:
                self.logger.error(f"Erro processando candle: {e}")
        
        self.dll.DLLSetHistoricalDataCallback(historical_candle_callback)
        
        # Callback de negócios (para dados de microestrutura)
        @WINFUNCTYPE(None, c_wchar_p, c_wchar_p, c_uint, c_double, c_double, c_int, 
                     c_wchar_p, c_wchar_p, c_wchar_p, c_int, c_int)
        def historical_trades_callback(date, time, corretora, qtde, price, volume,
                                     buy_agent, sell_agent, trade_type, trade_id, seq_trade_id):
            try:
                datetime_str = f"{date} {time}"
                dt = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S.%f")
                
                trade = {
                    'datetime': dt,
                    'price': float(price),
                    'volume': float(volume),
                    'quantity': int(qtde),
                    'buy_agent': buy_agent,
                    'sell_agent': sell_agent,
                    'trade_type': trade_type,
                    'broker': corretora
                }
                
                self.historical_data.append(trade)
                
            except Exception as e:
                self.logger.error(f"Erro processando trade: {e}")
        
        self.dll.DLLSetHistoricalTradesCallback(historical_trades_callback)
    
    def connect(self):
        """Conecta ao servidor"""
        try:
            self.logger.info("Conectando ao servidor...")
            
            # Conectar
            server = "producao.nelogica.com.br"
            port = "8184"
            connect_result = self.dll.DLLConnect(server, port)
            self.logger.info(f"Resultado conexão: {connect_result}")
            
            # Aguardar conexão
            timeout = 30
            start = time.time()
            while not self.connected and (time.time() - start) < timeout:
                time.sleep(0.5)
                
            if not self.connected:
                self.logger.error("Timeout esperando conexão")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Erro conectando: {e}")
            return False
    
    def load_historical_data(self, ticker: str, start_date: datetime, end_date: datetime,
                           period_minutes: int = 1) -> pd.DataFrame:
        """
        Carrega dados históricos em blocos de 9 dias
        
        Args:
            ticker: Símbolo do ativo (ex: "WDOH25")
            start_date: Data inicial
            end_date: Data final
            period_minutes: Período em minutos (1 para M1)
            
        Returns:
            DataFrame com todos os dados concatenados
        """
        
        self.logger.info(f"Carregando dados históricos de {ticker}")
        self.logger.info(f"Período: {start_date} até {end_date}")
        
        all_data = []
        current_start = start_date
        
        # Carregar em blocos de 9 dias
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=9), end_date)
            
            self.logger.info(f"\nCarregando bloco: {current_start} até {current_end}")
            
            # Limpar batch atual
            self.current_batch = []
            
            # Solicitar dados históricos
            self._request_historical_data(ticker, current_start, current_end, period_minutes)
            
            # Aguardar dados
            time.sleep(5)  # Tempo para processar
            
            # Adicionar batch aos dados totais
            if self.current_batch:
                all_data.extend(self.current_batch)
                self.logger.info(f"Bloco carregado: {len(self.current_batch)} candles")
            else:
                self.logger.warning(f"Nenhum dado recebido para o bloco")
            
            # Próximo bloco
            current_start = current_end + timedelta(minutes=1)
            
            # Pequena pausa entre requisições
            time.sleep(1)
        
        # Converter para DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            # Remover duplicatas
            df = df[~df.index.duplicated(keep='first')]
            
            self.logger.info(f"\nTotal carregado: {len(df)} candles")
            self.logger.info(f"Período final: {df.index[0]} até {df.index[-1]}")
            
            return df
        else:
            self.logger.error("Nenhum dado foi carregado")
            return pd.DataFrame()
    
    def _request_historical_data(self, ticker: str, start_date: datetime, 
                               end_date: datetime, period_minutes: int):
        """Solicita dados históricos do servidor"""
        try:
            # Criar estrutura AssetID
            asset = TAssetID()
            asset.pwcTicker = ticker
            asset.pwcBolsa = "BOVESPA"
            asset.nFeed = 0
            
            # Formatar datas
            start_str = start_date.strftime("%d/%m/%Y")
            end_str = end_date.strftime("%d/%m/%Y")
            
            # Período (em minutos)
            period = c_int(period_minutes)
            
            self.logger.info(f"Solicitando: {ticker} de {start_str} até {end_str}")
            
            # Solicitar dados
            result = self.dll.DLLGetHistoricalData(
                asset,
                period,
                start_str,
                end_str
            )
            
            self.logger.info(f"Resultado da solicitação: {result}")
            
        except Exception as e:
            self.logger.error(f"Erro solicitando dados históricos: {e}")
    
    def process_and_save_data(self, df: pd.DataFrame, output_file: str):
        """
        Processa dados e salva no formato adequado para ML
        
        O formato inclui todas as colunas necessárias para cálculo de features:
        - OHLCV básico
        - Volume de compra/venda  
        - Microestrutura (bid/ask spread, trades)
        """
        
        if df.empty:
            self.logger.error("DataFrame vazio, nada para processar")
            return
        
        self.logger.info("Processando dados para formato ML...")
        
        # Calcular volume de compra/venda estimado
        # Se close > open, assumir mais volume de compra
        df['price_change'] = df['close'] - df['open']
        df['buy_ratio'] = df['price_change'].apply(lambda x: 0.6 if x > 0 else 0.4)
        
        # Estimar volumes
        df['buy_volume'] = df['volume'] * df['buy_ratio']
        df['sell_volume'] = df['volume'] * (1 - df['buy_ratio'])
        
        # Adicionar spread
        df['spread'] = df['ask'] - df['bid']
        df['spread_pct'] = (df['spread'] / df['close']) * 100
        
        # Trade intensity
        df['trade_intensity'] = df['trades'] / df['quantity']
        
        # Retornos
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # High-Low range
        df['hl_range'] = df['high'] - df['low']
        df['hl_pct'] = (df['hl_range'] / df['close']) * 100
        
        # VWAP aproximado
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Adicionar contrato
        if 'ticker' in df.columns:
            df['contract'] = df['ticker']
        else:
            # Inferir contrato baseado na data
            month = df.index[0].month
            year = df.index[0].year
            month_codes = {1:'F', 2:'G', 3:'H', 4:'J', 5:'K', 6:'M',
                          7:'N', 8:'Q', 9:'U', 10:'V', 11:'X', 12:'Z'}
            df['contract'] = f"WDO{month_codes[month]}{str(year)[-2:]}"
        
        # Reorganizar colunas no formato esperado
        columns_order = [
            'open', 'high', 'low', 'close', 'volume', 'quantity',
            'buy_volume', 'sell_volume', 'trades', 'bid', 'ask',
            'spread', 'spread_pct', 'vwap', 'returns', 'log_returns',
            'hl_range', 'hl_pct', 'trade_intensity', 'contract'
        ]
        
        # Manter apenas colunas que existem
        final_columns = [col for col in columns_order if col in df.columns]
        df_final = df[final_columns].copy()
        
        # Preencher NaN
        df_final = df_final.fillna(method='ffill').fillna(0)
        
        # Salvar
        df_final.to_csv(output_file)
        self.logger.info(f"Dados salvos em: {output_file}")
        self.logger.info(f"Shape: {df_final.shape}")
        self.logger.info(f"Colunas: {list(df_final.columns)}")
        
        # Estatísticas
        self.logger.info("\nEstatísticas dos dados:")
        self.logger.info(f"Período: {df_final.index[0]} até {df_final.index[-1]}")
        self.logger.info(f"Total candles: {len(df_final)}")
        self.logger.info(f"Volume médio: {df_final['volume'].mean():,.0f}")
        self.logger.info(f"Trades médios: {df_final['trades'].mean():.0f}")
        self.logger.info(f"Spread médio: {df_final['spread_pct'].mean():.3f}%")
    
    def disconnect(self):
        """Desconecta do servidor"""
        try:
            if self.dll:
                self.dll.DLLDisconnect()
                self.dll.DLLFinalize()
                self.logger.info("Desconectado")
        except Exception as e:
            self.logger.error(f"Erro desconectando: {e}")


def main():
    """Função principal para carregar dados históricos"""
    
    # Configurações
    TICKER = "WDOH25"  # Ajustar conforme necessário
    START_DATE = datetime(2025, 2, 1, 9, 0)  # Início do mês
    END_DATE = datetime(2025, 2, 7, 17, 0)   # Uma semana de dados
    OUTPUT_FILE = "wdo_historical_ml_ready.csv"
    
    print("="*80)
    print("CARREGADOR DE DADOS HISTÓRICOS - PROFITDLL")
    print("="*80)
    
    loader = HistoricalDataLoader()
    
    try:
        # Inicializar
        if not loader.initialize_dll():
            print("Erro inicializando DLL")
            return
        
        # Conectar
        if not loader.connect():
            print("Erro conectando ao servidor")
            return
        
        # Carregar dados
        print(f"\nCarregando dados de {TICKER}...")
        df = loader.load_historical_data(TICKER, START_DATE, END_DATE)
        
        if not df.empty:
            # Processar e salvar
            loader.process_and_save_data(df, OUTPUT_FILE)
            
            print(f"\n✅ Dados carregados e salvos com sucesso!")
            print(f"Arquivo: {OUTPUT_FILE}")
        else:
            print("❌ Nenhum dado foi carregado")
            
    except Exception as e:
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        loader.disconnect()
    
    print("\nProcesso finalizado")


if __name__ == "__main__":
    main()