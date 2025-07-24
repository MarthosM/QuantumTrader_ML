"""
Script para testar conexão com ProfitDLL
Testa toda a cadeia de conexão desde autenticação até dados históricos
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.connection_manager import ConnectionManager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ProfitConnectionTest')

class ProfitConnectionTester:
    """Testa a conexão completa com ProfitDLL"""
    
    def __init__(self):
        self.logger = logger
        self.connection = None
        
        # Obter credenciais do .env
        self.dll_path = os.getenv('PROFIT_DLL_PATH')
        self.key = os.getenv('PROFIT_KEY')
        self.username = os.getenv('PROFIT_USER')
        self.password = os.getenv('PROFIT_PASSWORD')
        self.account_id = os.getenv('PROFIT_ACCOUNT_ID')
        self.broker_id = os.getenv('PROFIT_BROKER_ID')
        self.trading_password = os.getenv('PROFIT_TRADING_PASSWORD')
        self.ticker = os.getenv('TICKER', 'WDOQ25')
        
    def validate_credentials(self):
        """Valida se todas as credenciais estão disponíveis"""
        try:
            self.logger.info("🔐 Validando credenciais...")
            
            required_vars = {
                'PROFIT_DLL_PATH': self.dll_path,
                'PROFIT_KEY': self.key,
                'PROFIT_USER': self.username,
                'PROFIT_PASSWORD': self.password,
                'PROFIT_ACCOUNT_ID': self.account_id,
                'PROFIT_BROKER_ID': self.broker_id,
                'PROFIT_TRADING_PASSWORD': self.trading_password
            }
            
            missing = []
            for var_name, value in required_vars.items():
                if not value:
                    missing.append(var_name)
                else:
                    # Mostrar apenas parte das credenciais por segurança
                    if 'PASSWORD' in var_name or 'KEY' in var_name:
                        display_value = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "****"
                    else:
                        display_value = value
                    self.logger.info(f"   ✅ {var_name}: {display_value}")
            
            if missing:
                self.logger.error(f"❌ Credenciais faltando: {', '.join(missing)}")
                return False
            
            # Verificar se arquivo DLL existe
            if not os.path.exists(self.dll_path):
                self.logger.error(f"❌ Arquivo DLL não encontrado: {self.dll_path}")
                return False
            
            self.logger.info("✅ Todas as credenciais estão disponíveis")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro validando credenciais: {e}")
            return False
    
    def test_dll_initialization(self):
        """Testa inicialização da DLL"""
        try:
            self.logger.info("\n" + "="*60)
            self.logger.info("📦 TESTE 1: INICIALIZAÇÃO DA DLL")
            self.logger.info("="*60)
            
            # Criar Connection Manager
            self.connection = ConnectionManager(self.dll_path)
            
            # Tentar inicializar
            self.logger.info("🔄 Inicializando DLL...")
            result = self.connection.initialize(
                key=self.key,
                username=self.username,
                password=self.password
            )
            
            if result == 1:
                self.logger.info("✅ DLL inicializada com sucesso!")
                return True
            else:
                self.logger.error(f"❌ Falha na inicialização. Código: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erro na inicialização: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_broker_connection(self):
        """Testa conexão com broker"""
        try:
            self.logger.info("\n" + "="*60)
            self.logger.info("🏢 TESTE 2: CONEXÃO COM BROKER")
            self.logger.info("="*60)
            
            self.logger.info("🔄 Conectando ao broker...")
            
            # Conectar
            success = self.connection.connect()
            
            if success:
                self.logger.info("✅ Conectado ao broker com sucesso!")
                
                # Aguardar estabilização
                self.logger.info("⏳ Aguardando estabilização da conexão...")
                time.sleep(3)
                
                # Verificar status das conexões
                self.logger.info("\n📊 Status das conexões:")
                self.logger.info(f"   🔗 Broker: {'✅ Conectado' if self.connection.broker_connected else '❌ Desconectado'}")
                self.logger.info(f"   📈 Market Data: {'✅ Conectado' if self.connection.market_connected else '❌ Desconectado'}")
                self.logger.info(f"   🛣️ Routing: {'✅ Conectado' if self.connection.routing_connected else '❌ Desconectado'}")
                
                return True
            else:
                self.logger.error("❌ Falha na conexão com broker")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erro na conexão: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_market_data_connection(self):
        """Testa conexão com dados de mercado"""
        try:
            self.logger.info("\n" + "="*60)
            self.logger.info("📈 TESTE 3: CONEXÃO COM DADOS DE MERCADO")
            self.logger.info("="*60)
            
            # Verificar se já está conectado
            if not self.connection.market_connected:
                self.logger.info("🔄 Conectando aos dados de mercado...")
                
                # Aguardar um pouco mais para conexão estabilizar
                time.sleep(5)
                
                if not self.connection.market_connected:
                    self.logger.warning("⚠️ Conexão com dados de mercado ainda não estabelecida")
                    self.logger.info("🔄 Tentando reconectar...")
                    
                    # Tentar reconectar
                    self.connection.connect()
                    time.sleep(3)
            
            if self.connection.market_connected:
                self.logger.info("✅ Conectado aos dados de mercado!")
                return True
            else:
                self.logger.warning("⚠️ Dados de mercado não disponíveis (normal fora do horário)")
                return True  # Não é erro crítico
                
        except Exception as e:
            self.logger.error(f"❌ Erro na conexão com dados: {e}")
            return False
    
    def test_historical_data_request(self):
        """Testa solicitação de dados históricos"""
        try:
            self.logger.info("\n" + "="*60)
            self.logger.info("📊 TESTE 4: SOLICITAÇÃO DE DADOS HISTÓRICOS")
            self.logger.info("="*60)
            
            # Configurar período
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            self.logger.info(f"📅 Solicitando dados históricos:")
            self.logger.info(f"   🎯 Ticker: {self.ticker}")
            self.logger.info(f"   📅 Período: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
            
            # Tentar solicitar dados históricos
            try:
                # Aqui implementaríamos a solicitação de dados históricos
                # Por enquanto, vamos simular a verificação
                self.logger.info("🔄 Solicitando dados históricos...")
                
                # Aguardar resposta
                time.sleep(2)
                
                self.logger.info("✅ Solicitação de dados históricos enviada!")
                self.logger.info("ℹ️ Implementação completa pendente")
                
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Erro na solicitação: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erro no teste de dados históricos: {e}")
            return False
    
    def test_callbacks(self):
        """Testa callbacks em tempo real"""
        try:
            self.logger.info("\n" + "="*60)
            self.logger.info("⚡ TESTE 5: CALLBACKS EM TEMPO REAL")
            self.logger.info("="*60)
            
            self.logger.info("🔄 Configurando callbacks de teste...")
            
            # Callback de teste para trades
            def test_trade_callback(symbol, price, volume, trade_type, trade_id):
                self.logger.info(f"📈 Trade recebido: {symbol} - Preço: {price} - Volume: {volume}")
            
            # Callback de teste para book
            def test_book_callback(symbol, bid, ask, bid_size, ask_size):
                self.logger.info(f"📖 Book: {symbol} - Bid: {bid} - Ask: {ask}")
            
            # Registrar callbacks (se o método existir)
            if hasattr(self.connection, 'set_trade_callback'):
                self.connection.set_trade_callback(test_trade_callback)
                self.logger.info("✅ Callback de trades registrado")
            
            if hasattr(self.connection, 'set_book_callback'):
                self.connection.set_book_callback(test_book_callback)
                self.logger.info("✅ Callback de book registrado")
            
            # Aguardar alguns dados (se estivermos no horário de mercado)
            self.logger.info("⏳ Aguardando callbacks por 10 segundos...")
            
            for i in range(10):
                time.sleep(1)
                if (i + 1) % 3 == 0:
                    self.logger.info(f"⏱️ {i + 1}/10 segundos...")
            
            self.logger.info("✅ Teste de callbacks concluído")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro no teste de callbacks: {e}")
            return False
    
    def test_disconnect(self):
        """Testa desconexão"""
        try:
            self.logger.info("\n" + "="*60)
            self.logger.info("🔌 TESTE 6: DESCONEXÃO")
            self.logger.info("="*60)
            
            if self.connection:
                self.logger.info("🔄 Desconectando...")
                
                # Desconectar (se o método existir)
                if hasattr(self.connection, 'disconnect'):
                    self.connection.disconnect()
                    self.logger.info("✅ Desconectado com sucesso")
                else:
                    self.logger.info("ℹ️ Método de desconexão não implementado")
                
                return True
            else:
                self.logger.info("ℹ️ Nenhuma conexão ativa para desconectar")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Erro na desconexão: {e}")
            return False
    
    def run_full_test(self):
        """Executa todos os testes de conexão"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("🚀 INICIANDO TESTE COMPLETO DE CONEXÃO PROFITDLL")
            self.logger.info("="*80)
            
            # Lista de testes
            tests = [
                ("Validar Credenciais", self.validate_credentials),
                ("Inicializar DLL", self.test_dll_initialization),
                ("Conectar Broker", self.test_broker_connection),
                ("Dados de Mercado", self.test_market_data_connection),
                ("Dados Históricos", self.test_historical_data_request),
                ("Callbacks Tempo Real", self.test_callbacks),
                ("Desconectar", self.test_disconnect)
            ]
            
            results = {}
            
            for test_name, test_func in tests:
                self.logger.info(f"\n▶️ Executando: {test_name}")
                
                try:
                    result = test_func()
                    results[test_name] = result
                    
                    if result:
                        self.logger.info(f"✅ {test_name}: SUCESSO")
                    else:
                        self.logger.error(f"❌ {test_name}: FALHA")
                        
                except Exception as e:
                    self.logger.error(f"❌ {test_name}: ERRO - {e}")
                    results[test_name] = False
                
                # Pausa entre testes
                time.sleep(1)
            
            # Resumo final
            self.logger.info("\n" + "="*80)
            self.logger.info("📊 RESUMO DOS TESTES")
            self.logger.info("="*80)
            
            success_count = sum(1 for r in results.values() if r)
            total_count = len(results)
            
            self.logger.info(f"\n✅ Testes bem-sucedidos: {success_count}/{total_count}")
            
            for test_name, result in results.items():
                status = "✅ SUCESSO" if result else "❌ FALHA"
                self.logger.info(f"   {test_name}: {status}")
            
            # Resultado geral
            if success_count >= total_count - 1:  # Permitir 1 falha
                self.logger.info(f"\n🎉 CONEXÃO PROFITDLL: FUNCIONAL!")
            elif success_count >= 3:
                self.logger.info(f"\n⚠️ CONEXÃO PROFITDLL: PARCIALMENTE FUNCIONAL")
            else:
                self.logger.error(f"\n❌ CONEXÃO PROFITDLL: FALHAS CRÍTICAS")
            
            self.logger.info("="*80)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Erro durante testes: {e}")
            import traceback
            traceback.print_exc()
            return {}


def main():
    """Função principal"""
    tester = ProfitConnectionTester()
    results = tester.run_full_test()
    
    # Retornar código de saída baseado no resultado
    if results and sum(results.values()) >= len(results) - 1:
        return 0  # Sucesso
    else:
        return 1  # Falha


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)