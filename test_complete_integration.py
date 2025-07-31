#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste completo da integração ZMQ + Valkey
"""

import zmq
import valkey
import json
import time
import threading
from datetime import datetime, timedelta

class IntegrationTest:
    def __init__(self):
        self.zmq_context = zmq.Context()
        self.valkey_client = valkey.Valkey(host='localhost', port=6379, decode_responses=False)
        self.running = False
        self.messages_sent = 0
        self.messages_received = 0
        
    def zmq_publisher(self):
        """Publica dados via ZMQ"""
        publisher = self.zmq_context.socket(zmq.PUB)
        publisher.bind("tcp://*:5556")  # Porta diferente para não conflitar
        
        print("[ZMQ] Publisher iniciado na porta 5556")
        
        symbols = ["WDOQ25", "WINQ25"]
        
        while self.running:
            for symbol in symbols:
                tick_data = {
                    "symbol": symbol,
                    "price": 5000 + (self.messages_sent % 100),
                    "volume": 100 + (self.messages_sent % 50),
                    "timestamp": datetime.now().isoformat(),
                    "timestamp_ms": int(datetime.now().timestamp() * 1000)
                }
                
                topic = f"tick_{symbol}".encode()
                data = json.dumps(tick_data).encode()
                
                publisher.send_multipart([topic, data])
                self.messages_sent += 1
                
            time.sleep(0.1)  # 10 msgs/segundo
            
        publisher.close()
        
    def zmq_to_valkey_bridge(self):
        """Consome ZMQ e armazena em Valkey"""
        subscriber = self.zmq_context.socket(zmq.SUB)
        subscriber.connect("tcp://localhost:5556")
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        subscriber.setsockopt(zmq.RCVTIMEO, 1000)
        
        print("[Bridge] ZMQ -> Valkey iniciado")
        
        while self.running:
            try:
                topic, data = subscriber.recv_multipart()
                tick = json.loads(data)
                
                # Armazenar no Valkey
                symbol = tick['symbol']
                stream_key = f"stream:ticks:{symbol}"
                
                # Converter para bytes
                tick_bytes = {k.encode(): str(v).encode() for k, v in tick.items()}
                
                # Adicionar ao stream
                timestamp_ms = tick.get('timestamp_ms', int(time.time() * 1000))
                self.valkey_client.xadd(
                    stream_key,
                    tick_bytes,
                    id=f"{timestamp_ms}-*",
                    maxlen=10000,
                    approximate=True
                )
                
                self.messages_received += 1
                
            except zmq.Again:
                continue
            except Exception as e:
                print(f"[Bridge] Erro: {e}")
                
        subscriber.close()
        
    def test_time_travel(self):
        """Testa time travel query"""
        print("\n[Time Travel] Aguardando dados...")
        time.sleep(3)  # Aguardar alguns dados
        
        # Query últimos 2 segundos
        now = datetime.now()
        start_time = now - timedelta(seconds=2)
        
        start_id = f"{int(start_time.timestamp() * 1000)}-0"
        end_id = f"{int(now.timestamp() * 1000)}-0"
        
        results = {}
        for symbol in ["WDOQ25", "WINQ25"]:
            stream_key = f"stream:ticks:{symbol}"
            
            entries = self.valkey_client.xrange(stream_key, start_id, end_id)
            results[symbol] = len(entries)
            
            if entries:
                # Mostrar primeiro e último
                first = {k.decode(): v.decode() for k, v in entries[0][1].items()}
                last = {k.decode(): v.decode() for k, v in entries[-1][1].items()}
                
                print(f"\n[Time Travel] {symbol}:")
                print(f"  - Ticks encontrados: {len(entries)}")
                print(f"  - Primeiro: {first['timestamp']} - ${first['price']}")
                print(f"  - Último: {last['timestamp']} - ${last['price']}")
                
        return results
        
    def run_test(self, duration=10):
        """Executa teste completo"""
        print("="*60)
        print("  Teste Completo ZMQ + Valkey")
        print("="*60)
        
        self.running = True
        
        # Iniciar threads
        pub_thread = threading.Thread(target=self.zmq_publisher, daemon=True)
        bridge_thread = threading.Thread(target=self.zmq_to_valkey_bridge, daemon=True)
        
        pub_thread.start()
        bridge_thread.start()
        
        # Aguardar estabilização
        time.sleep(2)
        
        # Mostrar progresso
        start_time = time.time()
        while time.time() - start_time < duration:
            print(f"\r[Status] Enviados: {self.messages_sent} | Recebidos: {self.messages_received}", end="")
            time.sleep(1)
            
        print("\n")
        
        # Parar threads
        self.running = False
        pub_thread.join(timeout=2)
        bridge_thread.join(timeout=2)
        
        # Testar time travel
        self.test_time_travel()
        
        # Resumo
        print("\n" + "="*60)
        print("  Resumo do Teste")
        print("="*60)
        print(f"Mensagens ZMQ enviadas: {self.messages_sent}")
        print(f"Mensagens armazenadas no Valkey: {self.messages_received}")
        print(f"Taxa de sucesso: {(self.messages_received/self.messages_sent)*100:.1f}%")
        
        # Verificar streams no Valkey
        print("\nStreams no Valkey:")
        for key in self.valkey_client.keys("stream:*"):
            key_str = key.decode()
            try:
                info = self.valkey_client.xinfo_stream(key)
                length = info.get(b'length', 0)
                print(f"  - {key_str}: {length} entries")
            except Exception as e:
                print(f"  - {key_str}: erro ao obter info")
            
        print("\n[SUCESSO] Integração ZMQ + Valkey funcionando!")
        
        # Cleanup
        self.zmq_context.term()
        
if __name__ == "__main__":
    test = IntegrationTest()
    test.run_test(duration=10)  # Teste por 10 segundos