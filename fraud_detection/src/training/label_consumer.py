import json, time
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import boto3
from datetime import datetime

KAFKA_BOOTSTRAP = '127.0.0.1:29092'
MINIO_ENDPOINT = 'http://localhost:9000'
BUCKET = 'fraud-bucket2'


def make_consumer():
    while True:
        try:
            c = KafkaConsumer(
                'lebels',
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id='label-consumer-v2',
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                consumer_timeout_ms=-1,
            )
            print("Label consumer connected to Kafka.")
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

print("Label consumer starting...")
consumer = make_consumer()

for msg in consumer:
    label = msg.value
    date = datetime.utcnow().strftime('%Y-%m-%d')
    key = f"labels/{date}/{label['tx_id']}.json"
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(label))
    print(f"Stored label → {BUCKET}/{key}")
