import ujson as json
import time, random, uuid, threading
from kafka import KafkaProducer
from datetime import datetime

KAFKA_BOOTSTRAP = '127.0.0.1:29092'
TX_TOPIC = 'transactions'
LABEL_TOPIC = 'lebels'


def make_producer():
    while True:
        try:
            p = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                request_timeout_ms=10000,
                retries=5,
            )
            print("Producer connected to Kafka.")
            return p
        except Exception as e:
            print(f"Waiting for Kafka... ({e})")
            time.sleep(3)


def gen_tx():
    return {
        'tx_id': str(uuid.uuid4()),
        'user_id': random.randint(1, 10000),
        'amount': round(random.expovariate(1 / 50), 2),
        'device': random.choice(['web', 'ios', 'android']),
        'ip_hash': random.randint(0, 2**32 - 1),
        'ts': datetime.utcnow().isoformat(),
        'metadata': {'country': random.choice(['US', 'IN', 'BR', 'NZ', 'CN'])}
    }, random.random() < 0.01


def send_labels_later(p, tx, is_fraud, delay_seconds):
    def _send():
        time.sleep(delay_seconds)
        label = {
            'tx_id': tx['tx_id'],
            'user_id': tx['user_id'],
            'amount': tx['amount'],
            'device': tx['device'],
            'ip_hash': tx['ip_hash'],
            'metadata': tx['metadata'],
            'confirmed_fraud': bool(is_fraud),
            'label_ts': datetime.utcnow().isoformat()
        }
        p.send(LABEL_TOPIC, label)
        p.flush()
    threading.Thread(target=_send, daemon=True).start()


def main(qps=50, delay_seconds=60 * 30):
    p = make_producer()
    print(f"Sending at {qps} tx/s ...")
    while True:
        tx, is_fraud = gen_tx()
        p.send(TX_TOPIC, tx)
        print(f"Sent tx {tx['tx_id']} fraud={is_fraud}")
        if random.random() < 0.02 or is_fraud:
            send_labels_later(p, tx, is_fraud, delay_seconds)
        time.sleep(1 / qps)


if __name__ == '__main__':
    main()
