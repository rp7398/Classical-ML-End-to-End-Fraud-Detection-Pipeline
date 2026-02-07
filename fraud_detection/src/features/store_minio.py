import json,os
from kafka import KafkaConsumer
import boto3
from datetime import datetime

print("Consumer started, waiting for messages...")

consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers='127.0.0.1:29092',
    group_id = 'feature-consumer-v1',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer = lambda v:json.loads(v.decode('utf-8'))
)
s3 = boto3.client(
    's3',
    endpoint_url="http://localhost:9000",
    aws_access_key_id = 'minioadmin',
    aws_secret_access_key= 'minioadmin'
)
bucket = 'feature-bucket2'
for msg in consumer:
    label = msg.value
    date = datetime.utcnow().isoformat()
    key = f"labels/{date}/{label['tx_id']}/.json"
    s3.put_object(Bucket=bucket,Key=key,Body=json.dumps(label))
    print(f"Uploaded to minio {bucket} {key}")
