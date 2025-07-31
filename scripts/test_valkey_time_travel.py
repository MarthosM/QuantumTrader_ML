#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Teste de time travel com Valkey"""

import valkey
from datetime import datetime, timedelta
import json

def test_time_travel():
    client = valkey.Valkey(host='localhost', port=6379)
    
    print("Testando Time Travel com Valkey...")
    
    # Criar dados hist�ricos
    stream_key = "stream:ticks:WDOQ25"
    
    # Simular 1 hora de dados
    now = datetime.now()
    
    for minutes_ago in range(60, 0, -1):
        timestamp = now - timedelta(minutes=minutes_ago)
        timestamp_ms = int(timestamp.timestamp() * 1000)
        
        tick_data = {
            "symbol": "WDOQ25",
            "price": str(5000 + minutes_ago),
            "volume": str(100 + minutes_ago % 10),
            "timestamp": timestamp.isoformat()
        }
        
        client.xadd(
            stream_key,
            tick_data,
            id=f"{timestamp_ms}-0"
        )
    
    print(f"[OK] Adicionados 60 minutos de dados hist�ricos")
    
    # Time travel query - �ltimos 10 minutos
    end_time = now
    start_time = now - timedelta(minutes=10)
    
    start_id = f"{int(start_time.timestamp() * 1000)}-0"
    end_id = f"{int(end_time.timestamp() * 1000)}-0"
    
    entries = client.xrange(stream_key, start_id, end_id)
    
    print(f"\n[Info] Time Travel Query (�ltimos 10 minutos):")
    print(f"Encontrados {len(entries)} ticks")
    
    if entries:
        first_tick = {k.decode(): v.decode() for k, v in entries[0][1].items()}
        last_tick = {k.decode(): v.decode() for k, v in entries[-1][1].items()}
        
        print(f"Primeiro tick: {first_tick['timestamp']} - Pre�o: {first_tick['price']}")
        print(f"�ltimo tick: {last_tick['timestamp']} - Pre�o: {last_tick['price']}")

if __name__ == "__main__":
    test_time_travel()
