# -*- coding: utf-8 -*-
"""
Configuração centralizada para ZMQ e Valkey
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

class ZMQValkeyConfig:
    """Configuração centralizada para ZMQ e Valkey"""
    
    # ZeroMQ Configuration
    ZMQ_ENABLED = os.getenv('ZMQ_ENABLED', 'false').lower() == 'true'
    ZMQ_TICK_PORT = int(os.getenv('ZMQ_TICK_PORT', 5555))
    ZMQ_BOOK_PORT = int(os.getenv('ZMQ_BOOK_PORT', 5556))
    ZMQ_HISTORY_PORT = int(os.getenv('ZMQ_HISTORY_PORT', 5557))
    ZMQ_SIGNAL_PORT = int(os.getenv('ZMQ_SIGNAL_PORT', 5558))
    
    # Valkey Configuration
    VALKEY_ENABLED = os.getenv('VALKEY_ENABLED', 'false').lower() == 'true'
    VALKEY_HOST = os.getenv('VALKEY_HOST', 'localhost')
    VALKEY_PORT = int(os.getenv('VALKEY_PORT', 6379))
    VALKEY_PASSWORD = os.getenv('VALKEY_PASSWORD', None)
    VALKEY_DB = int(os.getenv('VALKEY_DB', 0))
    
    # Stream Configuration
    STREAM_MAX_LEN = int(os.getenv('STREAM_MAX_LENGTH', 100000))
    STREAM_RETENTION_DAYS = int(os.getenv('STREAM_RETENTION_DAYS', 30))
    
    # Feature Configuration
    TIME_TRAVEL_ENABLED = os.getenv('TIME_TRAVEL_ENABLED', 'false').lower() == 'true'
    TIME_TRAVEL_LOOKBACK_MINUTES = int(os.getenv('TIME_TRAVEL_LOOKBACK_MINUTES', 120))
    FAST_MODE_LATENCY_THRESHOLD = 0.1  # 100ms
    
    # Enhanced ML
    ENHANCED_ML_ENABLED = os.getenv('ENHANCED_ML_ENABLED', 'false').lower() == 'true'
    FORCE_FAST_MODE = os.getenv('FORCE_FAST_MODE', 'false').lower() == 'true'
    FALLBACK_ON_ERROR = os.getenv('FALLBACK_ON_ERROR', 'true').lower() == 'true'
    
    @classmethod
    def get_zmq_urls(cls):
        """Retorna URLs ZMQ configuradas"""
        return {
            'tick': f"tcp://localhost:{cls.ZMQ_TICK_PORT}",
            'book': f"tcp://localhost:{cls.ZMQ_BOOK_PORT}",
            'history': f"tcp://localhost:{cls.ZMQ_HISTORY_PORT}",
            'signal': f"tcp://localhost:{cls.ZMQ_SIGNAL_PORT}"
        }
    
    @classmethod
    def is_enhanced_enabled(cls):
        """Verifica se alguma funcionalidade enhanced está habilitada"""
        return cls.ZMQ_ENABLED or cls.VALKEY_ENABLED or cls.TIME_TRAVEL_ENABLED
    
    @classmethod
    def get_status(cls):
        """Retorna status da configuração"""
        return {
            'zmq': cls.ZMQ_ENABLED,
            'valkey': cls.VALKEY_ENABLED,
            'time_travel': cls.TIME_TRAVEL_ENABLED,
            'enhanced_ml': cls.ENHANCED_ML_ENABLED,
            'any_enabled': cls.is_enhanced_enabled()
        }