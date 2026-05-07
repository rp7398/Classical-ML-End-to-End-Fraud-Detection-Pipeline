import json, time
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import boto3
from datetime import datetime

KAFKA_BOOTSTRAP = '127.0.0.1:29092'
MINIO_ENDPOINT = 'http://localhost:9000'
BUCKET = 'feature-bucket2'


def make_consumer():
    while True:
        try:
            c = KafkaConsumer(
                'transactions',
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id='feature-consumer-v1',
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                consumer_timeout_ms=-1,
            )
            print("Feature consumer connected to Kafka.")
            return c
        except NoBrokersAvailable:
            print("Waiting for Kafka...")
            time.sleep(3)


s3 = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
)

print("Feature consumer starting...")
consumer = make_consumer()

for msg in consumer:
    record = msg.value
    date = datetime.utcnow().strftime('%Y-%m-%d')
    key = f"labels/{date}/{record['tx_id']}.json"
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(record))
    print(f"Stored feature → {BUCKET}/{key}")
