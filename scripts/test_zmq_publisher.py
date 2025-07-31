#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Teste básico de publicação ZMQ"""

import zmq
import time
import json
from datetime import datetime

def test_zmq_publisher():
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5555")
    
    print("ZMQ Publisher iniciado na porta 5555")
    print("Publicando dados de teste...")
    
    symbols = ["WDOQ25", "WINQ25", "INDQ25"]
    
    try:
        while True:
            for symbol in symbols:
                tick_data = {
                    "symbol": symbol,
                    "price": 5000 + (hash(str(time.time())) % 100),
                    "volume": 100 + (hash(str(time.time())) % 50),
                    "timestamp": datetime.now().isoformat()
                }
                
                topic = f"tick_{symbol}".encode()
                data = json.dumps(tick_data).encode()
                
                publisher.send_multipart([topic, data])
                print(f"Publicado: {symbol} - {tick_data['price']}")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nParando publisher...")
    finally:
        publisher.close()
        context.term()

if __name__ == "__main__":
    test_zmq_publisher()
