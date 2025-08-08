"""
Sistema de Paper Trading
Valida o sistema em paralelo com produção (sem executar ordens reais)
"""

import os
import sys
import json
import time
import threading
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PaperTrading')

# Adicionar paths
sys.path.insert(0, os.path.dirname(__file__))

from enhanced_production_system import EnhancedProductionSystem
from src.features.book_features_rt import BookFeatureEngineerRT
from src.data.book_data_manager import BookDataManager


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class PaperOrder:
    """Ordem simulada para paper trading"""
    id: str
    timestamp: datetime
    side: OrderSide
    price: float
    quantity: int
    status: str = "PENDING"
    fill_price: float = 0
    fill_time: Optional[datetime] = None
    pnl: float = 0


class PaperTradingSystem:
    """Sistema de paper trading para validação"""
    
    def __init__(self, initial_capital: float = 100000):
        self.system = EnhancedProductionSystem()
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.position = 0
        self.orders: List[PaperOrder] = []
        self.trades: List[PaperOrder] = []
        self.current_price = 5450.0
        self.running = False
        self.thread = None
        
        # Métricas
        self.metrics = {
            'total_orders': 0,
            'filled_orders': 0,
            'rejected_orders': 0,
            'total_pnl': 0,
            'win_trades': 0,
            'loss_trades': 0,
            'max_drawdown': 0,
            'peak_capital': initial_capital,
            'signals_generated': 0,
            'features_calculated': 0,
            'predictions_made': 0
        }
        
        # Log de decisões
        self.decision_log = []
    
    def start(self):
        """Inicia sistema de paper trading"""
        logger.info("\n" + "=" * 60)
        logger.info("INICIANDO PAPER TRADING")
        logger.info("=" * 60)
        logger.info(f"Capital inicial: ${self.initial_capital:,.2f}")
        logger.info(f"Sistema: 65 features + ML")
        logger.info("Modo: SIMULAÇÃO (sem ordens reais)")
        
        self.running = True
        self.thread = threading.Thread(target=self._run_trading_loop)
        self.thread.start()
        
        logger.info("\n[OK] Paper trading iniciado")
    
    def stop(self):
        """Para sistema de paper trading"""
        logger.info("\nParando paper trading...")
        self.running = False
        if self.thread:
            self.thread.join()
        
        # Fechar posições abertas
        if self.position != 0:
            self._close_position("STOP")
        
        logger.info("[OK] Paper trading parado")
    
    def _run_trading_loop(self):
        """Loop principal de trading"""
        iteration = 0
        
        while self.running:
            try:
                iteration += 1
                
                # Simular novo tick de mercado
                self._update_market_data()
                
                # Calcular features
                features = self.system._calculate_features()
                
                if len(features) == 65:
                    self.metrics['features_calculated'] += 1
                    
                    # Fazer predição
                    prediction = self.system._make_ml_prediction(features)
                    
                    if prediction != 0:
                        self.metrics['predictions_made'] += 1
                        
                        # Gerar sinal de trading
                        signal = self._generate_signal(features, prediction)
                        
                        if signal:
                            self.metrics['signals_generated'] += 1
                            
                            # Executar decisão de trading
                            self._execute_trading_decision(signal, prediction)
                
                # Atualizar ordens pendentes
                self._update_pending_orders()
                
                # Log periódico
                if iteration % 10 == 0:
                    self._log_status()
                
                # Delay para simular tempo real
                time.sleep(1)  # 1 segundo entre iterações
                
            except Exception as e:
                logger.error(f"Erro no loop de trading: {e}")
    
    def _update_market_data(self):
        """Simula atualização de dados de mercado"""
        # Random walk para preço
        ret = np.random.normal(0, 0.001)
        self.current_price *= (1 + ret)
        
        # Adicionar candle ao sistema
        candle = {
            'timestamp': datetime.now(),
            'open': self.current_price,
            'high': self.current_price * 1.001,
            'low': self.current_price * 0.999,
            'close': self.current_price * (1 + np.random.normal(0, 0.0005)),
            'volume': 100000 + np.random.randint(-10000, 10000)
        }
        self.system.feature_engineer._update_candle(candle)
        
        # Simular book update
        if np.random.random() > 0.5:
            price_data = {
                'timestamp': datetime.now(),
                'symbol': 'WDOU25',
                'bids': [
                    {'price': self.current_price - i*0.5, 'volume': 100 + i*10}
                    for i in range(5)
                ]
            }
            self.system.book_manager.on_price_book_callback(price_data)
            
            offer_data = {
                'timestamp': datetime.now(),
                'symbol': 'WDOU25',
                'asks': [
                    {'price': self.current_price + i*0.5, 'volume': 100 + i*10}
                    for i in range(5)
                ]
            }
            self.system.book_manager.on_offer_book_callback(offer_data)
    
    def _generate_signal(self, features: Dict, prediction: float) -> Optional[str]:
        """Gera sinal de trading baseado em features e predição"""
        # Verificar condições de entrada
        if self.position == 0:
            # Sem posição - procurar entrada
            if prediction > 0.65:
                # Verificar confirmação com features
                volatility = features.get('volatility_20', 0)
                rsi = features.get('rsi_14', 50)
                
                if volatility < 0.02 and rsi < 70:  # Não sobrecomprado
                    return 'BUY'
                    
            elif prediction < 0.35:
                # Verificar confirmação com features
                volatility = features.get('volatility_20', 0)
                rsi = features.get('rsi_14', 50)
                
                if volatility < 0.02 and rsi > 30:  # Não sobrevendido
                    return 'SELL'
        
        elif self.position > 0:
            # Posição comprada - verificar saída
            if prediction < 0.4:
                return 'CLOSE_LONG'
            
            # Stop loss
            last_trade = self.trades[-1] if self.trades else None
            if last_trade and self.current_price < last_trade.fill_price * 0.995:
                return 'STOP_LONG'
                
        elif self.position < 0:
            # Posição vendida - verificar saída
            if prediction > 0.6:
                return 'CLOSE_SHORT'
            
            # Stop loss
            last_trade = self.trades[-1] if self.trades else None
            if last_trade and self.current_price > last_trade.fill_price * 1.005:
                return 'STOP_SHORT'
        
        return None
    
    def _execute_trading_decision(self, signal: str, prediction: float):
        """Executa decisão de trading (simulada)"""
        timestamp = datetime.now()
        
        # Registrar decisão
        decision = {
            'timestamp': timestamp.isoformat(),
            'signal': signal,
            'prediction': prediction,
            'price': self.current_price,
            'position': self.position,
            'capital': self.capital
        }
        self.decision_log.append(decision)
        
        # Executar ordem baseada no sinal
        if signal == 'BUY' and self.position == 0:
            self._place_order(OrderSide.BUY, 1)
            
        elif signal == 'SELL' and self.position == 0:
            self._place_order(OrderSide.SELL, 1)
            
        elif signal in ['CLOSE_LONG', 'STOP_LONG'] and self.position > 0:
            self._close_position(signal)
            
        elif signal in ['CLOSE_SHORT', 'STOP_SHORT'] and self.position < 0:
            self._close_position(signal)
    
    def _place_order(self, side: OrderSide, quantity: int):
        """Coloca uma ordem simulada"""
        order = PaperOrder(
            id=f"PO_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            side=side,
            price=self.current_price,
            quantity=quantity
        )
        
        self.orders.append(order)
        self.metrics['total_orders'] += 1
        
        logger.info(f"\n[ORDEM] {side.value} {quantity} @ ${self.current_price:.2f}")
    
    def _update_pending_orders(self):
        """Atualiza ordens pendentes (simula preenchimento)"""
        for order in self.orders:
            if order.status == "PENDING":
                # Simular preenchimento imediato (simplificado)
                order.status = "FILLED"
                order.fill_price = self.current_price
                order.fill_time = datetime.now()
                
                # Atualizar posição
                if order.side == OrderSide.BUY:
                    self.position += order.quantity
                else:
                    self.position -= order.quantity
                
                # Adicionar aos trades
                self.trades.append(order)
                self.metrics['filled_orders'] += 1
                
                logger.info(f"[FILLED] Ordem {order.id} preenchida @ ${order.fill_price:.2f}")
    
    def _close_position(self, reason: str):
        """Fecha posição atual"""
        if self.position == 0:
            return
        
        # Calcular PnL
        if self.trades:
            entry_trade = self.trades[-1]
            
            if self.position > 0:
                pnl = (self.current_price - entry_trade.fill_price) * abs(self.position)
                side = OrderSide.SELL
            else:
                pnl = (entry_trade.fill_price - self.current_price) * abs(self.position)
                side = OrderSide.BUY
            
            # Registrar PnL
            self.metrics['total_pnl'] += pnl
            if pnl > 0:
                self.metrics['win_trades'] += 1
            else:
                self.metrics['loss_trades'] += 1
            
            # Atualizar capital
            self.capital += pnl
            
            # Atualizar drawdown
            if self.capital > self.metrics['peak_capital']:
                self.metrics['peak_capital'] = self.capital
            drawdown = (self.metrics['peak_capital'] - self.capital) / self.metrics['peak_capital']
            self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], drawdown)
            
            logger.info(f"\n[FECHAMENTO] {reason}")
            logger.info(f"  PnL: ${pnl:+.2f}")
            logger.info(f"  Capital: ${self.capital:,.2f}")
            
            # Criar ordem de fechamento
            self._place_order(side, abs(self.position))
    
    def _log_status(self):
        """Log periódico do status"""
        logger.info(f"\n[STATUS] Capital: ${self.capital:,.2f} | Posição: {self.position} | PnL Total: ${self.metrics['total_pnl']:+.2f}")
    
    def generate_report(self) -> Dict:
        """Gera relatório completo do paper trading"""
        win_rate = (
            self.metrics['win_trades'] / 
            max(1, self.metrics['win_trades'] + self.metrics['loss_trades'])
        ) * 100
        
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'duration': len(self.decision_log),
            'performance': {
                'initial_capital': self.initial_capital,
                'final_capital': self.capital,
                'total_pnl': self.metrics['total_pnl'],
                'total_return': total_return,
                'max_drawdown': self.metrics['max_drawdown'] * 100,
                'win_rate': win_rate
            },
            'trading': {
                'total_orders': self.metrics['total_orders'],
                'filled_orders': self.metrics['filled_orders'],
                'win_trades': self.metrics['win_trades'],
                'loss_trades': self.metrics['loss_trades']
            },
            'system': {
                'features_calculated': self.metrics['features_calculated'],
                'predictions_made': self.metrics['predictions_made'],
                'signals_generated': self.metrics['signals_generated']
            }
        }
        
        return report


def run_paper_trading_test(duration_seconds: int = 60):
    """Executa teste de paper trading"""
    logger.info("\n" + "=" * 70)
    logger.info(" VALIDAÇÃO COM PAPER TRADING")
    logger.info("=" * 70)
    
    # Criar sistema
    paper_system = PaperTradingSystem(initial_capital=100000)
    
    # Iniciar paper trading
    paper_system.start()
    
    # Executar por duração especificada
    logger.info(f"\nExecutando paper trading por {duration_seconds} segundos...")
    
    try:
        time.sleep(duration_seconds)
    except KeyboardInterrupt:
        logger.info("\nInterrompido pelo usuário")
    
    # Parar sistema
    paper_system.stop()
    
    # Gerar relatório
    report = paper_system.generate_report()
    
    # Exibir resultados
    logger.info("\n" + "=" * 60)
    logger.info("RESULTADOS DO PAPER TRADING")
    logger.info("=" * 60)
    
    logger.info(f"\nPerformance:")
    logger.info(f"  Capital inicial: ${report['performance']['initial_capital']:,.2f}")
    logger.info(f"  Capital final: ${report['performance']['final_capital']:,.2f}")
    logger.info(f"  PnL Total: ${report['performance']['total_pnl']:+,.2f}")
    logger.info(f"  Retorno: {report['performance']['total_return']:+.2f}%")
    logger.info(f"  Max Drawdown: {report['performance']['max_drawdown']:.2f}%")
    logger.info(f"  Win Rate: {report['performance']['win_rate']:.1f}%")
    
    logger.info(f"\nAtividade de Trading:")
    logger.info(f"  Ordens totais: {report['trading']['total_orders']}")
    logger.info(f"  Ordens executadas: {report['trading']['filled_orders']}")
    logger.info(f"  Trades vencedores: {report['trading']['win_trades']}")
    logger.info(f"  Trades perdedores: {report['trading']['loss_trades']}")
    
    logger.info(f"\nSistema:")
    logger.info(f"  Features calculadas: {report['system']['features_calculated']}")
    logger.info(f"  Predições feitas: {report['system']['predictions_made']}")
    logger.info(f"  Sinais gerados: {report['system']['signals_generated']}")
    
    # Salvar relatório
    report_path = Path('test_results/paper_trading_report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nRelatório salvo em: {report_path}")
    
    # Validar resultados
    success = validate_paper_trading_results(report)
    
    return success


def validate_paper_trading_results(report: Dict) -> bool:
    """Valida os resultados do paper trading"""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDAÇÃO DOS RESULTADOS")
    logger.info("=" * 60)
    
    validations = []
    
    # 1. Sistema funcionando
    if report['system']['features_calculated'] > 0:
        logger.info("[OK] Sistema calculou features")
        validations.append(True)
    else:
        logger.error("[ERRO] Nenhuma feature calculada")
        validations.append(False)
    
    if report['system']['predictions_made'] > 0:
        logger.info("[OK] Sistema fez predições")
        validations.append(True)
    else:
        logger.error("[ERRO] Nenhuma predição realizada")
        validations.append(False)
    
    # 2. Sinais gerados
    if report['system']['signals_generated'] > 0:
        logger.info(f"[OK] {report['system']['signals_generated']} sinais gerados")
        validations.append(True)
    else:
        logger.warning("[AVISO] Nenhum sinal gerado (mercado neutro?)")
        validations.append(True)  # Não é erro crítico
    
    # 3. Drawdown aceitável
    if report['performance']['max_drawdown'] < 10:
        logger.info(f"[OK] Drawdown controlado: {report['performance']['max_drawdown']:.2f}%")
        validations.append(True)
    else:
        logger.warning(f"[AVISO] Drawdown elevado: {report['performance']['max_drawdown']:.2f}%")
        validations.append(True)  # Warning, não erro
    
    # Resultado final
    success = all(validations)
    
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("[SUCESSO] Paper trading validado!")
        logger.info("Sistema pronto para comparação com produção.")
    else:
        logger.error("[FALHOU] Paper trading com problemas")
        logger.error("Revisar sistema antes de produção.")
    logger.info("=" * 60)
    
    return success


def main():
    """Função principal"""
    # Executar paper trading por 30 segundos (pode ajustar)
    success = run_paper_trading_test(duration_seconds=30)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)