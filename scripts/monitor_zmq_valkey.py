#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Monitor simples para ZMQ + Valkey"""

import zmq
import valkey
import json
import time
from datetime import datetime

def monitor_system():
    # Conectar ZMQ
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://localhost:5555")
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.setsockopt(zmq.RCVTIMEO, 1000)
    
    # Conectar Valkey
    valkey_client = valkey.Valkey(host='localhost', port=6379)
    
    print("[Monitor] ZMQ + Valkey")
    print("="*50)
    
    zmq_count = 0
    last_stats_time = time.time()
    
    try:
        while True:
            # Monitor ZMQ
            try:
                topic, data = subscriber.recv_multipart()
                zmq_count += 1
                
                if zmq_count % 10 == 0:
                    tick = json.loads(data)
                    print(f"ZMQ: {tick['symbol']} - ${tick['price']} - {datetime.now():%H:%M:%S}")
                    
            except zmq.Again:
                pass
            
            # Stats a cada 5 segundos
            if time.time() - last_stats_time > 5:
                # Contar streams no Valkey
                streams = valkey_client.keys("stream:*")
                
                total_entries = 0
                for stream in streams:
                    info = valkey_client.xinfo_stream(stream)
                    total_entries += info['length']
                
                print(f"\n[Stats] ZMQ msgs: {zmq_count} | Valkey streams: {len(streams)} | Total entries: {total_entries}")
                print("-"*50)
                
                last_stats_time = time.time()
                
    except KeyboardInterrupt:
        print("\nMonitor parado")
    finally:
        subscriber.close()
        context.term()

if __name__ == "__main__":
    monitor_system()
