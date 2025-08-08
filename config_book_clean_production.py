"""
Configuração Otimizada para Produção com book_clean
Modelo campeão: 79.23% Trading Accuracy
"""

import os
from pathlib import Path
from datetime import datetime

# ==========================================
# CONFIGURAÇÃO DO MODELO PRINCIPAL
# ==========================================

MODEL_CONFIG = {
    'primary_model': {
        'name': 'book_clean',
        'path': 'models/book_clean/lightgbm_book_clean_20250807_095345.txt',
        'scaler_path': 'models/book_clean/scaler_20250807_095345.pkl',
        'type': 'lightgbm',
        'trading_accuracy': 0.7923,
        'features': [
            'price_normalized',      # Feature mais importante
            'position',              # 2ª mais importante
            'position_normalized',   # 3ª mais importante
            'price_pct_change',      # 4ª mais importante
            'side'                   # 5ª mais importante
        ],
        'weight': 0.7  # 70% peso nas decisões
    },
    
    'fallback_model': {
        'name': 'book_moderate',
        'path': 'models/book_moderate/',
        'type': 'ensemble',
        'trading_accuracy': 0.6955,
        'weight': 0.2  # 20% peso
    },
    
    'validation_model': {
        'name': 'csv_5m_fast_corrected',
        'path': 'models/csv_5m_fast_corrected/',
        'type': 'ensemble',
        'accuracy': 0.7964,
        'weight': 0.1  # 10% peso para validação
    }
}

# ==========================================
# CONFIGURAÇÃO DE TRADING
# ==========================================

TRADING_CONFIG = {
    # Thresholds otimizados para book_clean
    'confidence_threshold': 0.65,      # Confiança mínima
    'direction_threshold': 0.60,       # Direção clara (>0.6 buy, <0.4 sell)
    'min_spread': 0.5,                 # Spread mínimo em pontos
    'max_position': 1,                 # Máximo de contratos
    
    # Risk Management
    'stop_loss_points': 10,            # Stop loss em pontos
    'take_profit_points': 20,          # Take profit em pontos (2:1)
    'max_daily_loss': 500,             # Perda máxima diária R$
    'max_daily_trades': 20,            # Máximo de trades por dia
    
    # Timeframes
    'candle_period': '1min',           # Período dos candles
    'prediction_interval': 1000,       # Predição a cada 1 segundo
    'feature_lookback': 20,            # Candles para calcular features
}

# ==========================================
# CONFIGURAÇÃO DE FEATURES
# ==========================================

FEATURE_CONFIG = {
    # Features do book_clean (apenas 5!)
    'book_features': {
        'price_normalized': {
            'calculation': 'price / rolling_mean(price, 20)',
            'importance': 1468112.22
        },
        'position': {
            'calculation': 'position_in_book',
            'importance': 567253.01
        },
        'position_normalized': {
            'calculation': 'position / max_position',
            'importance': 210000.70
        },
        'price_pct_change': {
            'calculation': '(price - prev_price) / prev_price * 100',
            'importance': 158369.44
        },
        'side': {
            'calculation': '0 if bid else 1',
            'importance': 81736.42
        }
    },
    
    # Features adicionais para monitoramento (não usadas na predição)
    'monitoring_features': [
        'quantity_log',
        'price_rolling_std',
        'is_bid',
        'time_of_day',
        'minute'
    ]
}

# ==========================================
# CONFIGURAÇÃO DO SISTEMA
# ==========================================

SYSTEM_CONFIG = {
    # Conexão
    'ticker': os.getenv('TICKER', 'WDOU25'),
    'dll_path': './ProfitDLL64.dll',
    
    # Performance
    'use_multiprocessing': False,  # book_clean é rápido, não precisa
    'cache_predictions': True,      # Cache de 1 segundo
    'batch_size': 1,               # Processar 1 por vez (real-time)
    
    # Logging
    'log_level': 'INFO',
    'log_predictions': True,
    'log_trades': True,
    'save_metrics': True,
    
    # Monitoring
    'enable_dashboard': True,
    'dashboard_port': 8080,
    'metrics_interval': 30,  # Segundos
}

# ==========================================
# HORÁRIOS DE MERCADO
# ==========================================

MARKET_HOURS = {
    'pre_market_start': '08:45',
    'market_open': '09:00',
    'market_close': '17:55',
    'after_market_end': '18:00',
    
    # Horários de maior liquidez (melhores para trading)
    'prime_hours': [
        ('09:15', '11:30'),  # Manhã
        ('14:00', '16:30')   # Tarde
    ]
}

# ==========================================
# CONFIGURAÇÃO HMARL (se disponível)
# ==========================================

HMARL_CONFIG = {
    'enabled': True,
    'weight_in_decisions': 0.15,  # 15% peso adicional se HMARL concordar
    'min_agent_confidence': 0.7,
    'min_agent_agreement': 0.6,   # 60% dos agentes concordando
}

# ==========================================
# MÉTRICAS DE SUCESSO
# ==========================================

SUCCESS_METRICS = {
    'target_trading_accuracy': 0.75,   # Meta: manter > 75%
    'target_win_rate': 0.55,          # Meta: > 55% trades vencedores
    'target_profit_factor': 1.5,      # Meta: ganhos/perdas > 1.5
    'target_sharpe_ratio': 1.0,       # Meta: Sharpe > 1.0
    'max_drawdown': 0.10,             # Limite: drawdown < 10%
}

# ==========================================
# FUNÇÃO DE VALIDAÇÃO
# ==========================================

def validate_configuration():
    """Valida se a configuração está correta"""
    errors = []
    warnings = []
    
    # Verificar se modelo existe
    model_path = Path(MODEL_CONFIG['primary_model']['path'])
    if not model_path.exists():
        errors.append(f"Modelo principal não encontrado: {model_path}")
    
    # Verificar scaler
    scaler_path = Path(MODEL_CONFIG['primary_model']['scaler_path'])
    if not scaler_path.exists():
        errors.append(f"Scaler não encontrado: {scaler_path}")
    
    # Verificar DLL
    dll_path = Path(SYSTEM_CONFIG['dll_path'])
    if not dll_path.exists():
        warnings.append(f"DLL não encontrada em {dll_path}, tentará caminho alternativo")
    
    # Verificar configurações de risco
    if TRADING_CONFIG['stop_loss_points'] > TRADING_CONFIG['take_profit_points']:
        warnings.append("Stop loss maior que take profit - risco/retorno desfavorável")
    
    return errors, warnings

# ==========================================
# CONFIGURAÇÃO RÁPIDA PARA PRODUÇÃO
# ==========================================

def get_production_config():
    """Retorna configuração pronta para produção"""
    
    # Validar configuração
    errors, warnings = validate_configuration()
    
    if errors:
        print("[ERRO] ERROS NA CONFIGURACAO:")
        for error in errors:
            print(f"   - {error}")
        raise ValueError("Configuracao invalida, corrija os erros acima")
    
    if warnings:
        print("[AVISO] AVISOS:")
        for warning in warnings:
            print(f"   - {warning}")
    
    config = {
        'model': MODEL_CONFIG,
        'trading': TRADING_CONFIG,
        'features': FEATURE_CONFIG,
        'system': SYSTEM_CONFIG,
        'market': MARKET_HOURS,
        'hmarl': HMARL_CONFIG,
        'metrics': SUCCESS_METRICS
    }
    
    print("[OK] Configuracao validada para producao!")
    print(f"[INFO] Modelo principal: {MODEL_CONFIG['primary_model']['name']}")
    print(f"[INFO] Trading Accuracy: {MODEL_CONFIG['primary_model']['trading_accuracy']:.1%}")
    print(f"[INFO] Features: {len(MODEL_CONFIG['primary_model']['features'])} apenas")
    
    return config

# ==========================================
# EXEMPLO DE USO
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CONFIGURAÇÃO BOOK_CLEAN PARA PRODUÇÃO")
    print("="*60)
    
    config = get_production_config()
    
    print("\n📋 Resumo da Configuração:")
    print(f"   Modelo: {config['model']['primary_model']['name']}")
    print(f"   Accuracy: {config['model']['primary_model']['trading_accuracy']:.1%}")
    print(f"   Features: {config['model']['primary_model']['features']}")
    print(f"   Threshold: {config['trading']['confidence_threshold']}")
    print(f"   Stop/Take: {config['trading']['stop_loss_points']}/{config['trading']['take_profit_points']}")
    print(f"   Max Loss: R$ {config['trading']['max_daily_loss']}")
    
    print("\n[OK] Pronto para producao!")
    print("   Use: from config_book_clean_production import get_production_config")
    print("="*60)