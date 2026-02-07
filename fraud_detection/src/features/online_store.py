import json
from kafka import KafkaConsumer
import redis
import os

consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers='127.0.0.1:29092',
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    group_id='feature-writer',             # set a consumer group
    enable_auto_commit=True,               # auto commit offsets periodically
    auto_offset_reset='earliest',          # if no committed offset, start at beginning
    consumer_timeout_ms=10000              # exit iteration if no messages for 10s (useful for debugging)
)


r = redis.Redis(host='localhost',port=6379)

for msg in consumer:
    payload = msg.value
    user_id = payload.get('user_id')
    key = f"user:{user_id}:features"
    flat = {k:json.dumps(v) for k,v in payload.items() if k != 'user_id'}
    r.hset(key,mapping=flat)
    print(f"{key} added.")



